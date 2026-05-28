from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback


class OptimizerStateTransfer(Callback):
    """Persist optimizer state across multiple `trainer.fit()` calls."""

    def __init__(self) -> None:
        self._saved_state: dict[str, object] | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._saved_state is None:
            return
        if len(trainer.optimizers) == 0:
            return
        trainer.optimizers[0].load_state_dict(self._saved_state)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if len(trainer.optimizers) == 0:
            return
        self._saved_state = trainer.optimizers[0].state_dict()


class EMACallback(Callback):
    """Exponential moving average of model weights.

    Maintains shadow params updated after each optimizer.step():
        shadow = decay * shadow + (1-decay) * live

    During validation, swaps live <-> shadow so val/loss reflects the EMA model.
    Restored to live weights after validation so training keeps using live.

    Self-play and the live training step keep using the live weights (we need
    exploration signal from the active model). Use the EMA for checkpoint persistence
    and eval gating, where smoothness matters.

    Decay typically 0.999 (slow average) or 0.99 (more responsive). Increase the
    decay as training progresses for stronger smoothing.
    """

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self._shadow: dict[str, torch.Tensor] | None = None
        self._backup: dict[str, torch.Tensor] | None = None

    def _trainable_named_params(
        self,
        pl_module: pl.LightningModule,
    ) -> list[tuple[str, torch.nn.Parameter]]:
        return [
            (name, param)
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        ]

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.decay <= 0.0:
            return
        if self._shadow is None:
            # First fit() call: snapshot live as the initial shadow.
            self._shadow = {
                name: param.detach().clone()
                for name, param in self._trainable_named_params(pl_module)
            }
            return
        # Subsequent fit() calls: shadow already populated from previous run.
        # Move it to the current device in case Lightning re-placed the model.
        for name, param in self._trainable_named_params(pl_module):
            if name in self._shadow:
                self._shadow[name] = self._shadow[name].to(param.device)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if self.decay <= 0.0 or self._shadow is None:
            return
        with torch.no_grad():
            for name, param in self._trainable_named_params(pl_module):
                shadow = self._shadow.get(name)
                if shadow is None or shadow.shape != param.shape:
                    self._shadow[name] = param.detach().clone()
                    continue
                shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def on_validation_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.decay <= 0.0 or self._shadow is None:
            return
        self._backup = {}
        with torch.no_grad():
            for name, param in self._trainable_named_params(pl_module):
                if name in self._shadow:
                    self._backup[name] = param.detach().clone()
                    param.data.copy_(self._shadow[name])

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self._backup is None:
            return
        with torch.no_grad():
            for name, param in self._trainable_named_params(pl_module):
                if name in self._backup:
                    param.data.copy_(self._backup[name])
        self._backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._shadow) if self._shadow is not None else {}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        if not state_dict:
            return
        self._shadow = {k: v.detach().clone() for k, v in state_dict.items()}


from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    import onnxruntime as ort


class OnnxModel(torch.nn.Module):
    """Drop-in replacement for system.model backed by ONNX Runtime (CPU)."""

    _session: ort.InferenceSession
    _input_names: set[str]
    num_input_channels: int
    num_actions: int
    onnx_path: Path

    def __init__(self, onnx_path: str | Path, *, num_input_channels: int, num_actions: int) -> None:
        super().__init__()
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self.num_input_channels = int(num_input_channels)
        self.num_actions = int(num_actions)
        self.onnx_path = Path(onnx_path)

    def forward(self, board: torch.Tensor, action_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        inputs: dict[str, Any] = {"board": board.detach().cpu().numpy().astype(np.float32, copy=False)}
        if "action_mask" in self._input_names and action_mask is not None:
            inputs["action_mask"] = action_mask.detach().cpu().numpy().astype(np.float32, copy=False)
        policy_np, value_np = self._session.run(["policy", "value"], inputs)
        policy = torch.from_numpy(np.asarray(policy_np, dtype=np.float32))
        value = torch.from_numpy(np.asarray(value_np, dtype=np.float32))
        return policy, value

    def to(self, *args: Any, **kwargs: Any) -> OnnxModel:  # noqa: ANN401  # ONNX Runtime CPU only here
        del args, kwargs
        return self


__all__ = ["OnnxModel"]

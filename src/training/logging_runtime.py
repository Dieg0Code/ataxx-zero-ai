from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path

from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger

from training.config_runtime import cfg_bool, cfg_str, log


def _wandb_run_name() -> str:
    explicit = cfg_str("wandb_run_name").strip()
    return explicit or cfg_str("hf_run_id").strip() or "ataxx-zero"


def build_training_logger(log_dir: Path) -> Logger:
    """
    Prefer W&B when explicitly configured, then TensorBoard, then CSV.
    """
    if cfg_bool("wandb_enabled") and os.environ.get("WANDB_API_KEY"):
        try:
            wandb_mod = import_module("pytorch_lightning.loggers")
            return wandb_mod.WandbLogger(
                project=cfg_str("wandb_project").strip() or "ataxx-zero",
                entity=cfg_str("wandb_entity").strip() or None,
                name=_wandb_run_name(),
                save_dir=str(log_dir),
                log_model=False,
            )
        except (ModuleNotFoundError, ImportError) as exc:
            log(f"W&B logger unavailable ({exc}); falling back to TensorBoard/CSV.")
    elif cfg_bool("wandb_enabled"):
        log("W&B enabled but WANDB_API_KEY is missing; falling back to TensorBoard/CSV.")
    try:
        return TensorBoardLogger(save_dir=str(log_dir), name="ataxx_zero")
    except ModuleNotFoundError as exc:
        log(f"TensorBoard logger unavailable ({exc}); falling back to CSVLogger.")
        return CSVLogger(save_dir=str(log_dir), name="ataxx_zero")


__all__ = ["build_training_logger"]

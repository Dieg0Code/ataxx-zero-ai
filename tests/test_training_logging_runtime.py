from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pytorch_lightning.loggers import CSVLogger

from training.config_runtime import CONFIG
from training.logging_runtime import build_training_logger


class TestTrainingLoggingRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._backup = dict(CONFIG)

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._backup)

    def test_build_training_logger_falls_back_to_csv_when_tensorboard_missing(self) -> None:
        CONFIG["wandb_enabled"] = False
        with patch(
            "training.logging_runtime.TensorBoardLogger",
            side_effect=ModuleNotFoundError("tensorboard missing"),
        ):
            logger = build_training_logger(Path(".tmp_test_train_logger"))
        self.assertIsInstance(logger, CSVLogger)

    def test_build_training_logger_uses_wandb_when_configured(self) -> None:
        CONFIG["wandb_enabled"] = True
        CONFIG["wandb_project"] = "ataxx-zero"
        CONFIG["wandb_entity"] = "dieg0code-ai"
        CONFIG["wandb_run_name"] = "policy_spatial_v10"
        calls: list[dict[str, object]] = []

        class FakeWandbLogger:
            def __init__(self, **kwargs: object) -> None:
                calls.append(kwargs)

        with patch.dict("os.environ", {"WANDB_API_KEY": "secret"}, clear=False), patch(
            "training.logging_runtime.import_module",
            return_value=SimpleNamespace(WandbLogger=FakeWandbLogger),
        ):
            logger = build_training_logger(Path(".tmp_test_train_logger"))

        self.assertIsInstance(logger, FakeWandbLogger)
        self.assertEqual(calls[0]["project"], "ataxx-zero")
        self.assertEqual(calls[0]["entity"], "dieg0code-ai")
        self.assertEqual(calls[0]["name"], "policy_spatial_v10")


if __name__ == "__main__":
    unittest.main()

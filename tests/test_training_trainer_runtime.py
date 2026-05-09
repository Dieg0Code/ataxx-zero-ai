from __future__ import annotations

import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from pytorch_lightning.strategies import DDPStrategy

from training.config_runtime import CONFIG
from training.trainer_runtime import (
    extract_training_metrics,
    resolve_trainer_hw,
    resolve_trainer_strategy,
)


class TestTrainingTrainerRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._backup = dict(CONFIG)

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._backup)

    def test_resolve_trainer_strategy_ddp_uses_configured_timeout(self) -> None:
        CONFIG["ddp_timeout_seconds"] = 75
        resolved = resolve_trainer_strategy("ddp")
        self.assertIsInstance(resolved, DDPStrategy)
        strategy = resolved
        self.assertEqual(strategy._timeout, timedelta(seconds=75))
        self.assertEqual(strategy._start_method, "popen")

    def test_resolve_trainer_strategy_ddp_spawn_uses_spawn_start_method(self) -> None:
        CONFIG["ddp_timeout_seconds"] = 90
        resolved = resolve_trainer_strategy("ddp_spawn")
        self.assertIsInstance(resolved, DDPStrategy)
        strategy = resolved
        self.assertEqual(strategy._timeout, timedelta(seconds=90))
        self.assertEqual(strategy._start_method, "spawn")

    def test_resolve_trainer_strategy_passthrough_for_auto(self) -> None:
        resolved = resolve_trainer_strategy("auto")
        self.assertEqual(resolved, "auto")

    def test_resolve_trainer_hw_downgrades_ddp_spawn_when_one_gpu_available(self) -> None:
        CONFIG["trainer_devices"] = 2
        CONFIG["trainer_strategy"] = "ddp_spawn"
        with (
            patch("training.trainer_runtime.torch.cuda.is_available", return_value=True),
            patch("training.trainer_runtime.torch.cuda.get_device_capability", return_value=(8, 0)),
            patch("training.trainer_runtime.torch.cuda.device_count", return_value=1),
        ):
            accelerator, devices, strategy = resolve_trainer_hw()
        self.assertEqual(accelerator, "gpu")
        self.assertEqual(devices, 1)
        self.assertEqual(strategy, "auto")

    def test_resolve_trainer_hw_keeps_ddp_spawn_when_two_gpus_available(self) -> None:
        CONFIG["trainer_devices"] = 2
        CONFIG["trainer_strategy"] = "ddp_spawn"
        with (
            patch("training.trainer_runtime.torch.cuda.is_available", return_value=True),
            patch("training.trainer_runtime.torch.cuda.get_device_capability", return_value=(8, 0)),
            patch("training.trainer_runtime.torch.cuda.device_count", return_value=2),
        ):
            accelerator, devices, strategy = resolve_trainer_hw()
        self.assertEqual(accelerator, "gpu")
        self.assertEqual(devices, 2)
        self.assertEqual(strategy, "ddp_spawn")

    def test_extract_training_metrics_flattens_lightning_logged_metrics(self) -> None:
        trainer = SimpleNamespace(
            logged_metrics={
                "train/loss": 1.23,
                "train/loss_value": 0.45,
                "train/loss_policy": 0.78,
                "train/value_mae": 0.12,
                "train/policy_accuracy": 0.66,
                "train/lr": 0.0003,
                "extra/should_be_ignored": 999.0,
            },
        )

        result = extract_training_metrics(trainer)  # type: ignore[arg-type]

        self.assertEqual(result["train_loss_total"], 1.23)
        self.assertAlmostEqual(result["train_loss_value"], 0.45)
        self.assertAlmostEqual(result["train_loss_policy"], 0.78)
        self.assertAlmostEqual(result["train_value_mae"], 0.12)
        self.assertAlmostEqual(result["train_policy_accuracy"], 0.66)
        self.assertAlmostEqual(result["train_lr"], 0.0003)
        self.assertNotIn("extra/should_be_ignored", result)
        self.assertNotIn("extra_should_be_ignored", result)

    def test_extract_training_metrics_returns_empty_for_warmup_only_iter(self) -> None:
        trainer = SimpleNamespace(logged_metrics={}, callback_metrics={})
        self.assertEqual(extract_training_metrics(trainer), {})  # type: ignore[arg-type]

    def test_extract_training_metrics_skips_missing_keys(self) -> None:
        trainer = SimpleNamespace(logged_metrics={"train/loss": 0.9, "train/lr": 0.001})

        result = extract_training_metrics(trainer)  # type: ignore[arg-type]

        self.assertEqual(set(result.keys()), {"train_loss_total", "train_lr"})

    def test_extract_training_metrics_falls_back_to_callback_metrics(self) -> None:
        trainer = SimpleNamespace(logged_metrics={}, callback_metrics={"train/loss": 2.0})

        result = extract_training_metrics(trainer)  # type: ignore[arg-type]

        self.assertEqual(result, {"train_loss_total": 2.0})

    def test_resolve_trainer_hw_falls_back_to_cpu_for_unsupported_cuda_capability(self) -> None:
        CONFIG["trainer_devices"] = 1
        CONFIG["trainer_strategy"] = "auto"
        with (
            patch("training.trainer_runtime.torch.cuda.is_available", return_value=True),
            patch("training.trainer_runtime.torch.cuda.get_device_capability", return_value=(6, 0)),
        ):
            accelerator, devices, strategy = resolve_trainer_hw()
        self.assertEqual(accelerator, "cpu")
        self.assertEqual(devices, 1)
        self.assertEqual(strategy, "auto")


if __name__ == "__main__":
    unittest.main()

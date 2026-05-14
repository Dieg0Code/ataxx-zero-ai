from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from training.config_runtime import CONFIG, apply_cli_overrides, parse_args


class TestTrainingConfigRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._backup = dict(CONFIG)

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._backup)

    def test_hf_bootstrap_flags_are_applied_from_cli(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--hf",
                "--hf-run-id",
                "policy_target_v2",
                "--hf-bootstrap-run-id",
                "policy_source_v1",
                "--hf-reset-iteration",
            ],
        ):
            args = parse_args()

        apply_cli_overrides(args)

        self.assertTrue(bool(CONFIG["hf_enabled"]))
        self.assertEqual(str(CONFIG["hf_run_id"]), "policy_target_v2")
        self.assertEqual(str(CONFIG["hf_bootstrap_run_id"]), "policy_source_v1")
        self.assertTrue(bool(CONFIG["hf_reset_iteration"]))

    def test_advanced_heuristic_levels_are_accepted_in_cli(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--eval-heuristic-level",
                "sentinel",
                "--eval-heuristic-levels",
                "hard,apex,sentinel",
                "--warmup-heuristic-level",
                "apex",
                "--warmup-heuristic-levels",
                "hard,apex,sentinel",
            ],
        ):
            args = parse_args()

        apply_cli_overrides(args)

        self.assertEqual(str(CONFIG["eval_heuristic_level"]), "sentinel")
        self.assertEqual(str(CONFIG["eval_heuristic_levels"]), "hard,apex,sentinel")
        self.assertEqual(str(CONFIG["warmup_heuristic_level"]), "apex")
        self.assertEqual(str(CONFIG["warmup_heuristic_levels"]), "hard,apex,sentinel")

    def test_recent_mix_and_split_flags_are_applied(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--train-recent-fraction",
                "0.8",
                "--train-recent-window-fraction",
                "0.35",
                "--no-shuffle-train-val-split",
            ],
        ):
            args = parse_args()

        apply_cli_overrides(args)

        self.assertAlmostEqual(float(CONFIG["train_recent_fraction"]), 0.8)
        self.assertAlmostEqual(float(CONFIG["train_recent_window_fraction"]), 0.35)
        self.assertFalse(bool(CONFIG["shuffle_train_val_split"]))

    def test_eval_regression_gate_flags_are_applied(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--no-restore-best-on-regression",
                "--eval-regression-delta",
                "0.04",
                "--eval-regression-patience",
                "3",
            ],
        ):
            args = parse_args()

        apply_cli_overrides(args)

        self.assertFalse(bool(CONFIG["restore_best_on_regression"]))
        self.assertAlmostEqual(float(CONFIG["eval_regression_delta"]), 0.04)
        self.assertEqual(int(CONFIG["eval_regression_patience"]), 3)

    def test_curated_pretrain_flags_are_applied(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--pretrain-dataset",
                "data/curated/v10_pretrain.npz",
                "--pretrain-epochs",
                "3",
            ],
        ):
            args = parse_args()

        apply_cli_overrides(args)

        self.assertEqual(str(CONFIG["pretrain_dataset_path"]), "data/curated/v10_pretrain.npz")
        self.assertEqual(int(CONFIG["pretrain_epochs"]), 3)

    def test_absolute_eval_gate_flags_are_applied(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--baseline-checkpoint",
                "liga",
                "--baseline-composite",
                "0.81",
                "--baseline-h2h-min-score",
                "0.45",
                "--eval-absolute-patience",
                "2",
                "--eval-absolute-delta",
                "0.03",
                "--eval-absolute-action",
                "abort",
            ],
        ):
            args = parse_args()

        apply_cli_overrides(args)

        self.assertEqual(str(CONFIG["baseline_checkpoint"]), "liga")
        self.assertAlmostEqual(float(CONFIG["baseline_composite"]), 0.81)
        self.assertAlmostEqual(float(CONFIG["baseline_h2h_min_score"]), 0.45)
        self.assertEqual(int(CONFIG["eval_absolute_patience"]), 2)
        self.assertAlmostEqual(float(CONFIG["eval_absolute_delta"]), 0.03)
        self.assertEqual(str(CONFIG["eval_absolute_action"]), "abort")


if __name__ == "__main__":
    unittest.main()

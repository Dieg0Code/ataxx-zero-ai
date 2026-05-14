from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from training.absolute_gate_runtime import evaluate_absolute_gate
from training.config_runtime import CONFIG


class TestTrainingAbsoluteGateRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._backup = dict(CONFIG)
        CONFIG["baseline_checkpoint"] = "liga"
        CONFIG["baseline_composite"] = 0.81
        CONFIG["baseline_h2h_min_score"] = 0.45
        CONFIG["eval_absolute_patience"] = 2
        CONFIG["eval_absolute_delta"] = 0.03
        CONFIG["eval_games"] = 8
        CONFIG["eval_sims"] = 16

    def tearDown(self) -> None:
        CONFIG.clear()
        CONFIG.update(self._backup)

    def test_absolute_gate_combines_composite_and_h2h_failures(self) -> None:
        with patch(
            "training.absolute_gate_runtime.resolve_checkpoint",
            return_value=Path("checkpoints/policy_spatial_v8_iter_180.pt"),
        ), patch(
            "training.absolute_gate_runtime.run_match_results_to_summary",
            return_value={
                "games": 8,
                "checkpoint_a_wins": 2,
                "checkpoint_b_wins": 6,
                "draws": 0,
                "checkpoint_a_score": 0.25,
            },
        ):
            abs_count, h2h_count, stats, failed = evaluate_absolute_gate(
                candidate_checkpoint=Path("candidate.ckpt"),
                current_composite=0.75,
                absolute_fail_count=1,
                h2h_fail_count=1,
                device="cpu",
                c_puct=1.5,
                seed=7,
            )

        self.assertEqual(abs_count, 2)
        self.assertEqual(h2h_count, 2)
        self.assertTrue(failed)
        self.assertEqual(stats["baseline_checkpoint"], "liga")
        self.assertEqual(stats["baseline_h2h_score"], 0.25)


if __name__ == "__main__":
    unittest.main()

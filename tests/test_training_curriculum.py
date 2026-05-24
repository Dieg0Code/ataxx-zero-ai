from __future__ import annotations

import unittest

import numpy as np

from agents.heuristic import HEURISTIC_LEVELS
from training.curriculum import get_curriculum_mix, sample_opponent_from_curriculum


class TestTrainingCurriculum(unittest.TestCase):
    def test_mix_probabilities_sum_to_one_per_group(self) -> None:
        for iteration in (1, 6, 13, 25, 45, 75, 120):
            mix = get_curriculum_mix(iteration)
            self.assertAlmostEqual(mix["self"] + mix["heuristic"] + mix["random"], 1.0, places=6)
            heu_sum = sum(mix[f"heu_{level}"] for level in HEURISTIC_LEVELS)
            self.assertAlmostEqual(heu_sum, 1.0, places=6)

    def test_heuristic_bucket_is_zero_all_phases(self) -> None:
        # v13: heurísticas fuera del self-play. Si esto se rompe es regresión.
        for iteration in (1, 6, 13, 25, 45, 75, 120, 200):
            mix = get_curriculum_mix(iteration)
            self.assertEqual(mix["heuristic"], 0.0)

    def test_curriculum_progressively_reduces_random(self) -> None:
        # Sin heurísticas, la única exploración no-policy es random. Decrece.
        early = get_curriculum_mix(1)["random"]
        late = get_curriculum_mix(120)["random"]
        self.assertGreaterEqual(early, late)

    def test_self_dominates_all_phases(self) -> None:
        for iteration in (1, 20, 70, 200):
            mix = get_curriculum_mix(iteration)
            self.assertGreaterEqual(mix["self"], 0.90)

    def test_sampling_never_returns_heuristic_in_v13(self) -> None:
        rng = np.random.default_rng(seed=7)
        for iteration in (1, 10, 30, 100):
            for _ in range(200):
                opp, lvl = sample_opponent_from_curriculum(rng=rng, iteration=iteration)
                self.assertIn(opp, ("self", "random"))
                self.assertIn(lvl, HEURISTIC_LEVELS)


if __name__ == "__main__":
    unittest.main()

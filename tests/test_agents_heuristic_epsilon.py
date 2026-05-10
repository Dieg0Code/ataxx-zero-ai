from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.heuristic import heuristic_move
from game.board import AtaxxBoard


class TestHeuristicEpsilon(unittest.TestCase):
    """epsilon-greedy in heuristic_move breaks determinism for self-play training."""

    def test_epsilon_zero_is_deterministic_for_greedy_levels(self) -> None:
        # Levels that pick best-score move (hard/apex/sentinel/gambit) should
        # be reproducible when called with the same seed and epsilon=0.
        for level in ("hard", "apex", "sentinel", "gambit"):
            board = AtaxxBoard()
            move_a = heuristic_move(board, np.random.default_rng(7), level, epsilon=0.0)
            move_b = heuristic_move(board, np.random.default_rng(7), level, epsilon=0.0)
            self.assertEqual(move_a, move_b, msg=f"non-deterministic at level={level}")

    def test_epsilon_one_returns_random_move(self) -> None:
        board = AtaxxBoard()
        valid = set(board.get_valid_moves())
        for seed in range(20):
            move = heuristic_move(
                board, np.random.default_rng(seed), "hard", epsilon=1.0,
            )
            self.assertIn(move, valid)

    def test_epsilon_partial_diverges_from_greedy(self) -> None:
        # With epsilon>0, the move distribution should differ from epsilon=0
        # over many trials. Pick a level (sentinel) that is otherwise greedy.
        board = AtaxxBoard()
        greedy_move = heuristic_move(board, np.random.default_rng(0), "sentinel", epsilon=0.0)
        random_count = 0
        n_trials = 200
        for seed in range(n_trials):
            move = heuristic_move(
                board, np.random.default_rng(seed), "sentinel", epsilon=0.5,
            )
            if move != greedy_move:
                random_count += 1
        # epsilon=0.5 with many valid moves should produce non-greedy choices
        # in a non-trivial fraction of trials.
        self.assertGreater(random_count, n_trials * 0.1)

    def test_epsilon_out_of_range_raises(self) -> None:
        board = AtaxxBoard()
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            heuristic_move(board, rng, "hard", epsilon=-0.1)
        with self.assertRaises(ValueError):
            heuristic_move(board, rng, "hard", epsilon=1.5)


if __name__ == "__main__":
    unittest.main()

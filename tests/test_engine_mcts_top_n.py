from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engine.mcts import MCTS, MCTSNode


def _build_root(visit_value_prior: list[tuple[int, int, float, float]]) -> MCTSNode:
    root = MCTSNode(prior=1.0)
    for action_idx, visits, value, prior in visit_value_prior:
        child = MCTSNode(prior=prior)
        child.visit_count = visits
        child.value_sum = value * visits
        root.children[action_idx] = child
    return root


class TestMCTSTopNActions(unittest.TestCase):
    def test_returns_empty_for_no_children(self) -> None:
        root = MCTSNode(prior=1.0)
        self.assertEqual(MCTS.top_n_actions(root, n=3), [])

    def test_returns_empty_for_none_root(self) -> None:
        self.assertEqual(MCTS.top_n_actions(None, n=3), [])

    def test_orders_by_visit_count_descending(self) -> None:
        root = _build_root([
            (10, 50, 0.20, 0.10),
            (20, 100, 0.50, 0.20),
            (30, 25, -0.10, 0.05),
        ])
        result = MCTS.top_n_actions(root, n=3)
        self.assertEqual([item[0] for item in result], [20, 10, 30])
        self.assertEqual([item[1] for item in result], [100, 50, 25])

    def test_truncates_to_n(self) -> None:
        root = _build_root([(idx, idx * 10, 0.0, 0.1) for idx in range(1, 8)])
        result = MCTS.top_n_actions(root, n=3)
        self.assertEqual(len(result), 3)
        # Top three by visit count are idx=7,6,5 with visits 70,60,50.
        self.assertEqual([item[1] for item in result], [70, 60, 50])

    def test_value_returned_is_avg(self) -> None:
        root = _build_root([(42, 4, 0.75, 0.3)])
        result = MCTS.top_n_actions(root, n=1)
        self.assertEqual(len(result), 1)
        action_idx, visits, value, prior = result[0]
        self.assertEqual(action_idx, 42)
        self.assertEqual(visits, 4)
        self.assertAlmostEqual(value, 0.75)
        self.assertAlmostEqual(prior, 0.3)

    def test_n_zero_or_negative_returns_empty(self) -> None:
        root = _build_root([(1, 10, 0.0, 0.1)])
        self.assertEqual(MCTS.top_n_actions(root, n=0), [])
        self.assertEqual(MCTS.top_n_actions(root, n=-1), [])


if __name__ == "__main__":
    unittest.main()

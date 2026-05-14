from __future__ import annotations

import unittest

from game.actions import ACTION_SPACE
from ui.arena.hud import (
    _is_forced_pass,
    _normalize_brain_tab,
    _pct_from_value,
    _select_perspective,
    _value_for_side,
)


class TestUiArenaHudPerspective(unittest.TestCase):
    def test_single_ai_on_blue_inverts_red_value(self) -> None:
        label, side = _select_perspective(
            p1_agent="human",
            p2_agent="model",
            turn_player=1,
            thinker_player=-1,
        )

        self.assertEqual(label, "IA")
        self.assertEqual(side, -1)
        self.assertEqual(_pct_from_value(_value_for_side(-0.74, side)), 87.0)

    def test_single_ai_on_red_keeps_red_value(self) -> None:
        label, side = _select_perspective(
            p1_agent="model",
            p2_agent="human",
            turn_player=-1,
            thinker_player=1,
        )

        self.assertEqual(label, "IA")
        self.assertEqual(side, 1)
        self.assertEqual(_pct_from_value(_value_for_side(-0.74, side)), 13.0)

    def test_two_ai_uses_current_turn_as_perspective(self) -> None:
        label, side = _select_perspective(
            p1_agent="model",
            p2_agent="heuristic",
            turn_player=-1,
            thinker_player=1,
        )

        self.assertEqual(label, "TURNO")
        self.assertEqual(side, -1)

    def test_single_pass_move_is_forced_state_not_decision_confidence(self) -> None:
        self.assertTrue(_is_forced_pass([(ACTION_SPACE.pass_index, 400, 0.0, 1.0)]))

    def test_brain_tab_normalizes_to_three_views(self) -> None:
        self.assertEqual(_normalize_brain_tab(0), 0)
        self.assertEqual(_normalize_brain_tab(3), 0)
        self.assertEqual(_normalize_brain_tab(5), 2)
        self.assertEqual(_normalize_brain_tab("red"), 0)


if __name__ == "__main__":
    unittest.main()

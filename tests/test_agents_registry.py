from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.registry import (
    agent_kind,
    canonicalize_agent_spec,
    heuristic_level_for_agent,
    parse_agent_spec,
)


class TestAgentsRegistry(unittest.TestCase):
    def test_parse_heuristic_spec_uses_default_level_when_missing_variant(self) -> None:
        parsed = parse_agent_spec("heuristic", default_heuristic_level="apex")

        self.assertEqual(parsed.kind, "heuristic")
        self.assertEqual(parsed.variant, "apex")
        self.assertEqual(parsed.canonical, "heuristic:apex")

    def test_parse_heuristic_spec_accepts_explicit_variant(self) -> None:
        parsed = parse_agent_spec("heuristic:sentinel", default_heuristic_level="normal")

        self.assertEqual(parsed.kind, "heuristic")
        self.assertEqual(parsed.variant, "sentinel")

    def test_canonicalize_agent_spec_leaves_model_without_variant(self) -> None:
        self.assertEqual(canonicalize_agent_spec("model"), "model")
        self.assertEqual(agent_kind("model"), "model")
        self.assertIsNone(heuristic_level_for_agent("model"))

    def test_invalid_heuristic_variant_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_agent_spec("heuristic:bogus", default_heuristic_level="normal")

    def test_unknown_agent_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_agent_spec("futurebot", default_heuristic_level="normal")


if __name__ == "__main__":
    unittest.main()

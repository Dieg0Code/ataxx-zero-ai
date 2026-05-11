from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import registry

FIXTURE = {
    "schema_version": 1,
    "default_codename": "liga",
    "models": [
        {
            "codename": "alpha",
            "version": "v1",
            "file": "alpha.pt",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "iter": 10,
            "lore": "primer modelo",
            "eval": {"vs_heuristic": {}, "composite": None, "source": "pending"},
        },
        {
            "codename": "bravo",
            "version": "v2",
            "file": "bravo.pt",
            "trained_at": "2026-02-01T00:00:00+00:00",
            "iter": 50,
            "lore": "segundo",
            "eval": {
                "vs_heuristic": {
                    "hard": {"games": 12, "score": 0.4, "source": "gauntlet"},
                    "apex": {"games": 12, "score": 0.6, "source": "gauntlet"},
                },
                "composite": 0.5,
                "evaluated_at": "2026-02-02T00:00:00+00:00",
                "source": "gauntlet",
            },
        },
        {
            "codename": "liga",
            "version": "v3",
            "file": "liga.pt",
            "trained_at": "2026-03-01T00:00:00+00:00",
            "iter": 100,
            "lore": "más reciente",
            "eval": {
                "vs_heuristic": {
                    "hard": {"games": 12, "score": 0.7, "source": "gauntlet"},
                },
                "composite": 0.7,
                "source": "gauntlet",
            },
        },
    ],
}


class _RegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self._make_tmp())
        self.tmp.write_text(json.dumps(FIXTURE), encoding="utf-8")
        # Patch module-level paths so tests don't touch the real registry.
        self._patches = [
            patch.object(registry, "REGISTRY_PATH", self.tmp),
            patch.object(registry, "CHECKPOINT_DIR", self.tmp.parent),
        ]
        for p in self._patches:
            p.start()
        # Ensure the per-codename .pt fixtures exist so resolve() returns them.
        for entry in FIXTURE["models"]:
            file_name: str = entry["file"]  # type: ignore[assignment]
            (self.tmp.parent / file_name).write_bytes(b"")

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()

    @staticmethod
    def _make_tmp() -> str:
        import tempfile

        d = tempfile.mkdtemp(prefix="registry_test_")
        return str(Path(d) / "registry.json")


class ResolveTests(_RegistryTestCase):
    def test_resolve_by_codename(self) -> None:
        path = registry.resolve("alpha")
        self.assertEqual(path.name, "alpha.pt")

    def test_resolve_by_version(self) -> None:
        self.assertEqual(registry.resolve("v2").name, "bravo.pt")

    def test_resolve_by_filename(self) -> None:
        self.assertEqual(registry.resolve("liga.pt").name, "liga.pt")

    def test_resolve_by_filestem(self) -> None:
        self.assertEqual(registry.resolve("liga").name, "liga.pt")

    def test_resolve_latest_picks_most_recent(self) -> None:
        self.assertEqual(registry.resolve("latest").name, "liga.pt")

    def test_resolve_best_picks_highest_composite(self) -> None:
        self.assertEqual(registry.resolve("best").name, "liga.pt")

    def test_resolve_default(self) -> None:
        self.assertEqual(registry.resolve("default").name, "liga.pt")

    def test_resolve_unknown_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            registry.resolve("does-not-exist-xyz")

    def test_resolve_path_passthrough(self) -> None:
        existing = self.tmp.parent / "alpha.pt"
        out = registry.resolve(str(existing))
        self.assertTrue(out.is_file())


class RankedModelsTests(_RegistryTestCase):
    def test_ranked_by_composite(self) -> None:
        ranked = registry.ranked_models(metric="composite")
        codes = [m["codename"] for m in ranked]
        self.assertEqual(codes[0], "liga")
        self.assertEqual(codes[1], "bravo")
        self.assertEqual(codes[-1], "alpha")  # no score → last

    def test_ranked_by_specific_level(self) -> None:
        ranked = registry.ranked_models(metric="apex")
        # only bravo has apex; the rest go to the tail
        self.assertEqual(ranked[0]["codename"], "bravo")

    def test_ranked_by_rr(self) -> None:
        # Inject round_robin scores
        data = registry.load_registry()
        for m, rr in zip(data["models"], [0.3, 0.8, 0.6], strict=False):
            m["eval"]["round_robin"] = {"score": rr, "wins": 1, "losses": 1, "draws": 0, "games": 4}
        registry.save_registry(data)
        ranked = registry.ranked_models(metric="rr")
        self.assertEqual([m["codename"] for m in ranked[:3]], ["bravo", "liga", "alpha"])

    def test_ranked_by_combined_requires_both(self) -> None:
        data = registry.load_registry()
        # bravo and liga have composite; only liga gets rr
        liga = next(m for m in data["models"] if m["codename"] == "liga")
        liga["eval"]["round_robin"] = {"score": 0.9}
        registry.save_registry(data)
        ranked = registry.ranked_models(metric="combined")
        # liga has both → ranked first; others have no combined → tail
        self.assertEqual(ranked[0]["codename"], "liga")


class UpdateEvalTests(_RegistryTestCase):
    def test_update_merges_levels(self) -> None:
        registry.update_eval(
            "bravo",
            {
                "vs_heuristic": {
                    "sentinel": {"games": 12, "score": 0.55, "source": "gauntlet"},
                },
                "composite": 0.516,
                "source": "gauntlet",
            },
        )
        data = registry.load_registry()
        bravo = next(m for m in data["models"] if m["codename"] == "bravo")
        self.assertIn("hard", bravo["eval"]["vs_heuristic"])
        self.assertIn("sentinel", bravo["eval"]["vs_heuristic"])
        self.assertAlmostEqual(bravo["eval"]["composite"], 0.516, places=3)


if __name__ == "__main__":
    unittest.main()

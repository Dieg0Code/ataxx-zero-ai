from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from ui.arena.model_runtime import (
    build_model_labels_by_player,
    build_model_mcts_by_player,
    resolve_model_checkpoints,
)


class TestUiArenaModelRuntime(unittest.TestCase):
    def test_resolve_model_checkpoints_uses_shared_fallback(self) -> None:
        checkpoints = resolve_model_checkpoints(
            shared_checkpoint="checkpoints/shared.pt",
            p1_agent="model",
            p2_agent="model",
            p1_checkpoint="",
            p2_checkpoint="",
        )

        self.assertEqual(checkpoints[1], "checkpoints/shared.pt")
        self.assertEqual(checkpoints[-1], "checkpoints/shared.pt")

    def test_resolve_model_checkpoints_prefers_side_specific_paths(self) -> None:
        checkpoints = resolve_model_checkpoints(
            shared_checkpoint="checkpoints/shared.pt",
            p1_agent="model",
            p2_agent="model",
            p1_checkpoint="checkpoints/p1.pt",
            p2_checkpoint="checkpoints/p2.pt",
        )

        self.assertEqual(checkpoints[1], "checkpoints/p1.pt")
        self.assertEqual(checkpoints[-1], "checkpoints/p2.pt")

    def test_resolve_model_checkpoints_requires_model_path_for_each_model_side(self) -> None:
        with self.assertRaises(ValueError):
            resolve_model_checkpoints(
                shared_checkpoint="",
                p1_agent="model",
                p2_agent="heuristic",
                p1_checkpoint="",
                p2_checkpoint="",
            )

    def test_build_model_mcts_by_player_reuses_same_checkpoint_runtime(self) -> None:
        loaded: list[str] = []

        def fake_load_system(checkpoint_path: str, device: str) -> SimpleNamespace:
            loaded.append(f"{checkpoint_path}@{device}")
            return SimpleNamespace(model=f"model:{checkpoint_path}")

        def fake_mcts_factory(
            *,
            model: str,
            c_puct: float,
            n_simulations: int,
            device: str,
        ) -> tuple[str, float, int, str]:
            return (model, c_puct, n_simulations, device)

        runtimes = build_model_mcts_by_player(
            checkpoints_by_player={1: "checkpoints/shared.pt", -1: "checkpoints/shared.pt"},
            device="cpu",
            c_puct=1.5,
            n_simulations=64,
            load_system=fake_load_system,
            mcts_factory=fake_mcts_factory,
        )

        self.assertEqual(loaded, ["checkpoints/shared.pt@cpu"])
        self.assertIs(runtimes[1], runtimes[-1])


    def test_build_model_labels_uses_codename_and_iter_from_registry(self) -> None:
        registry: dict[str, dict[str, object]] = {
            "nemesis": {"codename": "nemesis", "iter": 152},
            "legion": {"codename": "legion"},  # sin iter
        }

        def fake_find_model(name: str) -> dict[str, object] | None:
            return registry.get(name)

        with mock.patch("model.registry.find_model", side_effect=fake_find_model):
            labels = build_model_labels_by_player(
                name_args_by_player={1: "nemesis", -1: "legion"},
            )

        self.assertEqual(labels[1], "NEMESIS (iter 152)")
        self.assertEqual(labels[-1], "LEGION")

    def test_build_model_labels_falls_back_to_stem_for_unregistered_path(self) -> None:
        with mock.patch("model.registry.find_model", return_value=None):
            labels = build_model_labels_by_player(
                name_args_by_player={1: "runs_history/_dl/model_iter_134.pt"},
            )

        self.assertEqual(labels[1], "model_iter_134")


if __name__ == "__main__":
    unittest.main()

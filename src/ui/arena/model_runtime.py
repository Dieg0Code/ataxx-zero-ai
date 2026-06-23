from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.types import Agent

PlayerCheckpoints = dict[int, str]
LoadSystem = Callable[[str, str], Any]
MCTSFactory = Callable[..., object]


def resolve_model_checkpoints(
    *,
    shared_checkpoint: str,
    p1_agent: Agent,
    p2_agent: Agent,
    p1_checkpoint: str,
    p2_checkpoint: str,
) -> PlayerCheckpoints:
    checkpoints: PlayerCheckpoints = {}
    for player, agent, side_checkpoint in (
        (1, p1_agent, p1_checkpoint),
        (-1, p2_agent, p2_checkpoint),
    ):
        if agent != "model":
            continue
        checkpoint_path = side_checkpoint.strip() or shared_checkpoint.strip()
        if checkpoint_path == "":
            raise ValueError(
                f"Model agent for {'p1' if player == 1 else 'p2'} requires a checkpoint path.",
            )
        checkpoints[player] = checkpoint_path
    return checkpoints


def build_model_labels_by_player(
    *,
    name_args_by_player: dict[int, str],
) -> dict[int, str]:
    """Map each player to a HUD label 'CODENAME (iter N)' resolved from the registry.

    `name_args_by_player` carries the raw checkpoint argument per player (codename,
    version, or path). Codename and iteration come from `registry.json` (no .pt
    load needed). Checkpoints outside the registry fall back to the filename stem.
    The separator stays ASCII so the pixel HUD font renders it without tofu.
    """
    from model.registry import find_model

    labels: dict[int, str] = {}
    for player, name in name_args_by_player.items():
        entry = find_model(name)
        if entry is not None:
            codename = str(entry.get("codename", name)).upper()
            iteration = entry.get("iter")
            labels[player] = (
                f"{codename} (iter {iteration})" if iteration is not None else codename
            )
        else:
            labels[player] = Path(name).stem or name
    return labels


def build_model_mcts_by_player(
    *,
    checkpoints_by_player: PlayerCheckpoints,
    device: str,
    c_puct: float,
    n_simulations: int,
    load_system: LoadSystem | None = None,
    mcts_factory: MCTSFactory | None = None,
) -> dict[int, object | None]:
    if load_system is None:
        from ui.arena.play_pygame_app import _load_system

        load_system = _load_system
    if mcts_factory is None:
        from engine.mcts import MCTS

        mcts_factory = MCTS

    if load_system is None:
        raise RuntimeError("load_system must be resolved before building model runtimes.")
    if mcts_factory is None:
        raise RuntimeError("mcts_factory must be resolved before building model runtimes.")
    runtime_by_checkpoint: dict[str, object] = {}
    mcts_by_player: dict[int, object | None] = {1: None, -1: None}
    for player, checkpoint_path in checkpoints_by_player.items():
        runtime = runtime_by_checkpoint.get(checkpoint_path)
        if runtime is None:
            system = load_system(checkpoint_path, device)
            model = getattr(system, "model", None)
            if model is None:
                raise ValueError(
                    f"Loaded system for checkpoint '{checkpoint_path}' does not expose a model.",
                )
            runtime = mcts_factory(
                model=model,
                c_puct=c_puct,
                n_simulations=n_simulations,
                device=device,
            )
            runtime_by_checkpoint[checkpoint_path] = runtime
        mcts_by_player[player] = runtime
    return mcts_by_player


__all__ = [
    "build_model_labels_by_player",
    "build_model_mcts_by_player",
    "resolve_model_checkpoints",
]

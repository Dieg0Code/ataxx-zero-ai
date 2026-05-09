from __future__ import annotations

import numpy as np

from agents.heuristic import heuristic_move
from agents.model_agent import model_move
from agents.random_agent import random_move
from agents.registry import parse_agent_spec
from agents.types import Agent
from game.board import AtaxxBoard
from game.types import Move


def pick_ai_move(
    board: AtaxxBoard,
    agent: Agent,
    rng: np.random.Generator,
    heuristic_level: str,
    mcts: object | None,
) -> tuple[Move | None, str]:
    if not board.has_valid_moves():
        return None, f"{agent} passed (no legal moves)"

    agent_spec = parse_agent_spec(agent, default_heuristic_level=heuristic_level)

    if agent_spec.kind == "random":
        return random_move(board=board, rng=rng), "Random AI move played"
    if agent_spec.kind == "heuristic":
        return (
            heuristic_move(board=board, rng=rng, level=str(agent_spec.variant)),
            f"Heuristic AI [{agent_spec.variant}] move played",
        )
    if agent_spec.kind == "model":
        return model_move(board=board, mcts=mcts), "Model AI move played"
    raise ValueError(f"Unsupported agent: {agent_spec.kind}")

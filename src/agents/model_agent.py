from __future__ import annotations

from typing import Any

import numpy as np

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.types import Move


def model_move(board: AtaxxBoard, mcts: object | None) -> Move | None:
    from engine.mcts import MCTS

    if not isinstance(mcts, MCTS):
        raise RuntimeError("Model agent selected but MCTS is not initialized.")
    probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=0.0)
    action_idx = int(np.argmax(probs))
    return ACTION_SPACE.decode(action_idx)


def model_move_with_diagnostics(
    board: AtaxxBoard,
    mcts: object | None,
) -> tuple[Move | None, dict[str, Any]]:
    """Return (move, diagnostics) so the arena HUD can render what MCTS saw.

    `diagnostics` keys:
      - top_moves: list[(action_idx, visits, value, prior)] sorted by visits desc.
      - root_value: float in [-1, 1], MCTS estimate for the player to move.
    """
    from engine.mcts import MCTS

    if not isinstance(mcts, MCTS):
        raise RuntimeError("Model agent selected but MCTS is not initialized.")
    probs, root = mcts.run_with_root(
        board=board,
        root=None,
        add_dirichlet_noise=False,
        temperature=0.0,
    )
    action_idx = int(np.argmax(probs))
    move = ACTION_SPACE.decode(action_idx)
    diagnostics: dict[str, Any] = {
        "top_moves": MCTS.top_n_actions(root, n=3),
        "root_value": float(root.value()) if root is not None else 0.0,
    }
    return move, diagnostics

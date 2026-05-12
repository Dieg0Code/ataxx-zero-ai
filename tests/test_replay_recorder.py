from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.constants import BOARD_SIZE, OBSERVATION_CHANNELS
from ui.arena.replay_recorder import (
    ReplayMetadata,
    ReplayRecorder,
    one_hot_policy,
)


def _metadata(tmp_path: Path) -> ReplayMetadata:
    return ReplayMetadata(
        mode="tournament",
        player_ai="liga",
        starter="human",
        mcts_sims=80,
        tournament_id="2026-05-12",
        player_human="Felipe",
        round_idx=0,
        match_idx=0,
        is_tiebreak=False,
        p1_label="Felipe",
        p2_label="liga",
    )


def _play_n_moves(rec: ReplayRecorder, n: int, rng: np.random.Generator) -> AtaxxBoard:
    """Aplica n movimientos validos al board, grabando cada uno."""
    board = AtaxxBoard()
    for _ in range(n):
        moves = board.get_valid_moves()
        if not moves:
            break
        idx = int(rng.integers(0, len(moves)))
        move = moves[idx]
        action_idx = ACTION_SPACE.encode(move)
        policy = one_hot_policy(action_idx, ACTION_SPACE.num_actions)
        rec.record_move(
            board_before=board,
            policy=policy,
            action_idx=action_idx,
            player=int(board.current_player),
        )
        board.step(move)
    return board


def test_one_hot_policy_shape_and_sum() -> None:
    p = one_hot_policy(action_idx=42, num_actions=ACTION_SPACE.num_actions)
    assert p.shape == (ACTION_SPACE.num_actions,)
    assert p.dtype == np.float32
    assert float(p.sum()) == 1.0
    assert int(np.argmax(p)) == 42


def test_one_hot_policy_invalid_index_returns_zero_vector() -> None:
    p = one_hot_policy(action_idx=-1, num_actions=ACTION_SPACE.num_actions)
    assert float(p.sum()) == 0.0


def test_finalize_writes_npz_and_sidecar_with_expected_shapes(tmp_path: Path) -> None:
    rec = ReplayRecorder(
        save_path=tmp_path / "felipe_r0_p0.npz",
        metadata=_metadata(tmp_path),
    )
    rng = np.random.default_rng(seed=42)
    _play_n_moves(rec, n=6, rng=rng)
    out = rec.finalize(winner=1, forced_draw=False)
    assert out is not None
    assert out.exists()

    data = np.load(out)
    assert set(data.files) == {"observations", "policies", "values"}
    assert data["observations"].shape == (6, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert data["policies"].shape == (6, ACTION_SPACE.num_actions)
    assert data["values"].shape == (6,)
    assert data["observations"].dtype == np.float32
    assert data["policies"].dtype == np.float32
    assert data["values"].dtype == np.float32

    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["mode"] == "tournament"
    assert payload["player_human"] == "Felipe"
    assert payload["player_ai"] == "liga"
    assert payload["n_moves"] == 6
    assert payload["winner"] == 1
    assert payload["shapes"]["values"] == [6]


def test_finalize_values_follow_winner_perspective(tmp_path: Path) -> None:
    rec = ReplayRecorder(
        save_path=tmp_path / "game.npz",
        metadata=_metadata(tmp_path),
    )
    rng = np.random.default_rng(seed=7)
    _play_n_moves(rec, n=4, rng=rng)
    out = rec.finalize(winner=1, forced_draw=False)
    assert out is not None
    data = np.load(out)
    values = data["values"]
    # En self-play los turnos alternan entre P1 y P2. Winner=1 implica que los
    # turnos de P1 reciben target +1 y los de P2 reciben target -1 (sin shaping
    # adicional, los valores deberian estar exactamente en {-1, 0, 1}).
    assert set(np.unique(values).astype(float)).issubset({-1.0, 1.0})


def test_finalize_with_no_moves_returns_none_and_writes_nothing(tmp_path: Path) -> None:
    rec = ReplayRecorder(
        save_path=tmp_path / "empty.npz",
        metadata=_metadata(tmp_path),
    )
    out = rec.finalize(winner=0, forced_draw=False)
    assert out is None
    assert not (tmp_path / "empty.npz").exists()
    assert not (tmp_path / "empty.json").exists()

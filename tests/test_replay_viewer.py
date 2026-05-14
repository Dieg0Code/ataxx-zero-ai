from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.constants import PLAYER_1, PLAYER_2
from ui.arena.replay_recorder import one_hot_policy
from ui.arena.replay_viewer import (
    ReplayFrame,
    action_from_policy,
    active_animation_done,
    cycle_quality_tag,
    discover_replays,
    grid_from_observation,
    load_replay,
    player_for_turn,
    replay_animation_duration_ms,
    replay_render_frame,
    resolve_replay_paths,
    spanish_date_label,
    update_replay_metadata,
)


def test_resolve_replay_paths_accepts_npz_and_json(tmp_path: Path) -> None:
    npz_path = tmp_path / "game.npz"
    json_path = tmp_path / "game.json"
    np.savez_compressed(
        npz_path,
        observations=np.zeros((0, 11, 7, 7), dtype=np.float32),
        policies=np.zeros((0, ACTION_SPACE.num_actions), dtype=np.float32),
        values=np.zeros((0,), dtype=np.float32),
    )
    json_path.write_text("{}", encoding="utf-8")

    assert resolve_replay_paths(npz_path).npz_path == npz_path
    assert resolve_replay_paths(json_path).npz_path == npz_path


def test_grid_from_observation_restores_absolute_player_perspective() -> None:
    board = AtaxxBoard()
    obs_p1 = board.get_observation()
    grid_p1 = grid_from_observation(obs_p1, PLAYER_1)
    assert np.array_equal(grid_p1, board.grid)

    board.current_player = PLAYER_2
    obs_p2 = board.get_observation()
    grid_p2 = grid_from_observation(obs_p2, PLAYER_2)
    assert np.array_equal(grid_p2, board.grid)


def test_action_from_policy_decodes_argmax_and_pass() -> None:
    move = (0, 0, 1, 1)
    policy = one_hot_policy(ACTION_SPACE.encode(move), ACTION_SPACE.num_actions)
    action_idx, decoded, prob = action_from_policy(policy)
    assert action_idx == ACTION_SPACE.encode(move)
    assert decoded == move
    assert prob == 1.0

    pass_policy = one_hot_policy(ACTION_SPACE.pass_index, ACTION_SPACE.num_actions)
    _, decoded_pass, _ = action_from_policy(pass_policy)
    assert decoded_pass is None


def test_load_replay_builds_frames_from_training_npz(tmp_path: Path) -> None:
    board = AtaxxBoard()
    moves = [(0, 0, 1, 1), (6, 0, 5, 1)]
    observations = []
    policies = []
    values = []
    for move in moves:
        observations.append(board.get_observation())
        policies.append(one_hot_policy(ACTION_SPACE.encode(move), ACTION_SPACE.num_actions))
        values.append(1.0 if board.current_player == PLAYER_1 else -1.0)
        board.step(move)
    npz_path = tmp_path / "sample.npz"
    np.savez_compressed(
        npz_path,
        observations=np.stack(observations).astype(np.float32),
        policies=np.stack(policies).astype(np.float32),
        values=np.asarray(values, dtype=np.float32),
    )
    npz_path.with_suffix(".json").write_text(
        json.dumps({"mode": "play", "p1_label": "human", "p2_label": "model", "winner": 1}),
        encoding="utf-8",
    )

    replay = load_replay(npz_path)

    assert replay.metadata["mode"] == "play"
    assert len(replay.frames) == 3
    assert replay.frames[0].player == PLAYER_1
    assert replay.frames[1].player == PLAYER_2
    assert replay.frames[0].move == moves[0]
    assert replay.frames[-1].is_final


def test_player_for_turn_alternates_from_red() -> None:
    assert player_for_turn(0) == PLAYER_1
    assert player_for_turn(1) == PLAYER_2
    assert player_for_turn(2) == PLAYER_1


def test_spanish_date_label_matches_requested_format() -> None:
    from datetime import date

    assert spanish_date_label(date(2026, 5, 13)) == "Miercoles 13 de Mayo, 2026"


def test_discover_replays_orders_by_date_and_uses_nickname(tmp_path: Path) -> None:
    older = tmp_path / "2026-05-12_10-00-00.npz"
    newer = tmp_path / "2026-05-13_09-00-00.npz"
    for path in (older, newer):
        np.savez_compressed(
            path,
            observations=np.zeros((1, 11, 7, 7), dtype=np.float32),
            policies=np.zeros((1, ACTION_SPACE.num_actions), dtype=np.float32),
            values=np.zeros((1,), dtype=np.float32),
        )
    newer.with_suffix(".json").write_text(
        json.dumps({"nickname": "salto doble al centro", "n_moves": 42, "winner": 1}),
        encoding="utf-8",
    )

    items = discover_replays(tmp_path)

    assert [item.npz_path for item in items] == [newer, older]
    assert items[0].title == "salto doble al centro"
    assert items[0].date_label == "Miercoles 13 de Mayo, 2026"


def test_update_replay_metadata_writes_nickname_and_event(tmp_path: Path) -> None:
    npz_path = tmp_path / "game.npz"
    np.savez_compressed(
        npz_path,
        observations=np.zeros((1, 11, 7, 7), dtype=np.float32),
        policies=np.zeros((1, ACTION_SPACE.num_actions), dtype=np.float32),
        values=np.zeros((1,), dtype=np.float32),
    )

    update_replay_metadata(npz_path, nickname="centro suicida", event="AIEP 4tos medios")

    payload = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["nickname"] == "centro suicida"
    assert payload["event"] == "AIEP 4tos medios"


def test_cycle_quality_tag_updates_replay_metadata(tmp_path: Path) -> None:
    npz_path = tmp_path / "game.npz"
    np.savez_compressed(
        npz_path,
        observations=np.zeros((1, 11, 7, 7), dtype=np.float32),
        policies=np.zeros((1, ACTION_SPACE.num_actions), dtype=np.float32),
        values=np.zeros((1,), dtype=np.float32),
    )

    assert cycle_quality_tag(npz_path) == "good"
    payload = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["quality_tag"] == "good"


def test_replay_render_frame_animates_preview_then_applies_move() -> None:
    board = AtaxxBoard()
    move = (0, 0, 1, 1)
    policy = one_hot_policy(ACTION_SPACE.encode(move), ACTION_SPACE.num_actions)
    action_idx, _decoded, prob = action_from_policy(policy)
    frame = load_replay_frame_for_test(board, move, action_idx, prob)

    preview = replay_render_frame(
        frame=frame,
        playing=True,
        anim_start_ms=1000,
        now_ms=1100,
        speed=1.0,
    )
    assert preview.preview_move == move
    assert np.array_equal(preview.board.grid, board.grid)

    applied = replay_render_frame(
        frame=frame,
        playing=True,
        anim_start_ms=1000,
        now_ms=1700,
        speed=1.0,
    )
    assert applied.preview_move is None
    assert int(applied.board.grid[1, 1]) == PLAYER_1
    assert applied.move_cells
    assert active_animation_done(
        frame=frame,
        anim_start_ms=1000,
        now_ms=1000 + replay_animation_duration_ms(frame) + 1,
        speed=1.0,
    )


def load_replay_frame_for_test(
    board: AtaxxBoard,
    move: tuple[int, int, int, int],
    action_idx: int,
    prob: float,
) -> ReplayFrame:
    return ReplayFrame(
        index=0,
        grid=board.grid.copy(),
        player=int(board.current_player),
        move=move,
        action_idx=action_idx,
        policy_prob=prob,
        value=1.0,
    )

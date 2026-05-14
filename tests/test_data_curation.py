from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np

from data.curation import curate_npz_paths, save_curated_dataset
from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE, OBSERVATION_CHANNELS


def _write_npz(
    path: Path,
    *,
    n: int = 2,
    policy_sum_zero: bool = False,
    policy_value: float = 1.0,
    value: float = 1.0,
) -> None:
    observations = np.zeros((n, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    observations[:, 3] = np.linspace(0.0, 0.8, n, dtype=np.float32).reshape(n, 1, 1)
    policies = np.zeros((n, ACTION_SPACE.num_actions), dtype=np.float32)
    if not policy_sum_zero:
        policies[:, 0] = policy_value
    values = np.full((n,), value, dtype=np.float32)
    np.savez_compressed(path, observations=observations, policies=policies, values=values)


def test_curator_filters_invalid_policy_and_excluded_metadata(tmp_path: Path) -> None:
    good = tmp_path / "good.npz"
    bad_policy = tmp_path / "bad_policy.npz"
    excluded = tmp_path / "excluded.npz"
    _write_npz(good, n=2)
    _write_npz(bad_policy, n=2, policy_sum_zero=True)
    _write_npz(excluded, n=2)
    excluded.with_suffix(".json").write_text(
        json.dumps({"quality_tag": "bug"}),
        encoding="utf-8",
    )

    dataset = curate_npz_paths([good, bad_policy, excluded])

    assert dataset.observations.shape[0] == 2
    assert dataset.report["input_examples"] == 6
    assert dataset.report["kept_examples"] == 2
    assert dataset.report["discarded_examples"] == 4


def test_curator_oversamples_human_replays_and_writes_report(tmp_path: Path) -> None:
    human = tmp_path / "play_sessions" / "human.npz"
    human.parent.mkdir()
    _write_npz(human, n=3)
    human.with_suffix(".json").write_text(
        json.dumps({"mode": "play", "player_human": "Diego", "quality_tag": "good"}),
        encoding="utf-8",
    )

    dataset = curate_npz_paths([human], human_oversample=4)
    output = tmp_path / "curated" / "v10_pretrain.npz"
    report_path = save_curated_dataset(dataset, output)

    assert dataset.observations.shape[0] == 12
    assert dataset.report["sources"][0]["human"] is True
    assert dataset.report["sources"][0]["oversample_factor"] == 4
    assert output.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["phase_counts"]["late"] > 0


def test_curator_normalizes_policy_and_keeps_tagged_human_draw(tmp_path: Path) -> None:
    human = tmp_path / "human.npz"
    _write_npz(human, n=1, policy_value=2.0)
    human.with_suffix(".json").write_text(
        json.dumps({"player_human": "Diego", "quality_tag": "late_game", "forced_draw": True}),
        encoding="utf-8",
    )

    dataset = curate_npz_paths([human], human_oversample=1)

    assert dataset.policies.shape[0] == 1
    assert np.isclose(np.sum(dataset.policies[0]), 1.0)


def test_select_buffer_files_filters_hf_run_buffers() -> None:
    namespace = runpy.run_path("scripts/fetch_hf_buffers.py", run_name="fetch_hf_buffers_test")
    select_buffer_files = namespace["select_buffer_files"]

    files = select_buffer_files(
        [
            "runs/policy_spatial_v8/buffer_iter_001.npz",
            "runs/policy_spatial_v8/model_iter_001.pt",
            "runs/policy_spatial_v9/buffer_iter_001.npz",
            "runs/policy_spatial_v8/buffer_iter_010.npz",
        ],
        run_id="policy_spatial_v8",
    )

    assert files == [
        "runs/policy_spatial_v8/buffer_iter_001.npz",
        "runs/policy_spatial_v8/buffer_iter_010.npz",
    ]

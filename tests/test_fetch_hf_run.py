from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_hf_run import (  # type: ignore[import-not-found]
    best_eval_iter,
    copy_flat_checkpoints,
    install_playable_checkpoints,
    load_env_file,
)


def test_best_eval_iter_prefers_actual_eval_score() -> None:
    rows: list[dict[str, object]] = [
        {"iter": 1, "score": None, "best_eval_score": 0.1},
        {"iter": 2, "score": 0.4, "best_eval_score": 0.4},
        {"iter": 3, "score": None, "best_eval_score": 0.9},
        {"iter": 4, "score": 0.7, "best_eval_score": 0.7},
    ]

    assert best_eval_iter(rows) == 4


def test_load_env_file_does_not_override_existing_env(tmp_path: Path) -> None:
    token_key = "HF_" + "TOKEN"
    expected = "from-" + "env"
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                f"{token_key}=from-file",
                "EMPTY_LINE_FOLLOWS",
                "",
                "# comment",
                "QUOTED='ok'",
            ]
        ),
        encoding="utf-8",
    )

    old = os.environ.get(token_key)
    os.environ[token_key] = expected
    try:
        load_env_file(env_path)
        assert os.environ[token_key] == expected
        assert os.environ["QUOTED"] == "ok"
    finally:
        if old is None:
            os.environ.pop(token_key, None)
        else:
            os.environ[token_key] = old
        os.environ.pop("QUOTED", None)


def test_copy_flat_checkpoints_creates_playable_paths(tmp_path: Path) -> None:
    nested = tmp_path / "checkpoints" / "runs" / "policy_spatial_v10"
    nested.mkdir(parents=True)
    source = nested / "model_iter_222.pt"
    source.write_bytes(b"checkpoint")

    copied = copy_flat_checkpoints([source], tmp_path)

    assert copied == [tmp_path / "model_iter_222.pt"]
    assert (tmp_path / "model_iter_222.pt").read_bytes() == b"checkpoint"


def test_install_playable_checkpoints_uses_project_naming(tmp_path: Path) -> None:
    source = tmp_path / "model_iter_222.pt"
    source.write_bytes(b"checkpoint")
    checkpoint_dir = tmp_path / "checkpoints"

    installed = install_playable_checkpoints(
        [source],
        run_id="policy_spatial_v10",
        checkpoint_dir=checkpoint_dir,
    )

    target = checkpoint_dir / "policy_spatial_v10_iter_222.pt"
    assert installed == [target]
    assert target.read_bytes() == b"checkpoint"

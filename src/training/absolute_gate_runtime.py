from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from inference.checkpoint_duel_runtime import run_match_results_to_summary
from model.registry import resolve as resolve_checkpoint
from training.config_runtime import cfg_bool, cfg_float, cfg_int, cfg_str
from training.eval_gating import compute_absolute_score_gate, compute_h2h_gate


def _download_hf_baseline_checkpoint(run_id: str) -> Path:
    run_id = run_id.strip()
    if run_id == "":
        raise ValueError("baseline_hf_run_id is empty")
    repo_id = cfg_str("hf_repo_id").strip()
    if repo_id == "":
        raise ValueError("hf_repo_id is empty")

    hub_mod = __import__("huggingface_hub", fromlist=["HfApi", "hf_hub_download"])
    api = hub_mod.HfApi(token=os.environ.get("HF_TOKEN") or None)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    prefix = f"runs/{run_id}/"
    model_files = [
        name
        for name in files
        if name.startswith(prefix)
        and Path(name).name.startswith("model_iter_")
        and (name.endswith(".pt") or name.endswith(".ckpt"))
    ]
    if not model_files:
        raise FileNotFoundError(f"No HF baseline model files found for run '{run_id}'")
    latest = max(model_files, key=lambda name: int(Path(name).stem.split("_")[2]))
    local_dir = Path(cfg_str("hf_local_dir") or ".hf_baselines") / "baselines" / run_id
    local_path = hub_mod.hf_hub_download(
        repo_id=repo_id,
        filename=latest,
        repo_type="model",
        token=os.environ.get("HF_TOKEN") or None,
        local_dir=str(local_dir),
    )
    return Path(local_path)


def _resolve_baseline_checkpoint(name: str) -> Path:
    try:
        return resolve_checkpoint(name)
    except (FileNotFoundError, ValueError, KeyError):
        if not cfg_bool("hf_enabled"):
            raise
        return _download_hf_baseline_checkpoint(cfg_str("baseline_hf_run_id"))


def evaluate_absolute_gate(
    *,
    candidate_checkpoint: Path,
    current_composite: float,
    absolute_fail_count: int,
    h2h_fail_count: int,
    device: str,
    c_puct: float,
    seed: int,
) -> tuple[int, int, dict[str, float | int | str], bool]:
    patience = cfg_int("eval_absolute_patience")
    baseline_name = cfg_str("baseline_checkpoint").strip()
    if patience <= 0 or baseline_name == "":
        return absolute_fail_count, h2h_fail_count, {}, False

    absolute_fail_count, absolute_failed = compute_absolute_score_gate(
        current_score=current_composite,
        baseline_score=cfg_float("baseline_composite"),
        delta=cfg_float("eval_absolute_delta"),
        current_streak=absolute_fail_count,
        patience=patience,
    )

    baseline_path = _resolve_baseline_checkpoint(baseline_name)
    h2h_summary = run_match_results_to_summary(
        checkpoint_a=candidate_checkpoint,
        checkpoint_b=baseline_path,
        games=cfg_int("eval_games"),
        device=device,
        mcts_sims=cfg_int("eval_sims"),
        c_puct=c_puct,
        seed=seed,
    )
    h2h_score = float(h2h_summary["checkpoint_a_score"])
    h2h_fail_count, h2h_failed = compute_h2h_gate(
        h2h_score=h2h_score,
        min_score=cfg_float("baseline_h2h_min_score"),
        current_streak=h2h_fail_count,
        patience=patience,
    )
    stats: dict[str, float | int | str] = {
        "baseline_checkpoint": baseline_name,
        "baseline_path": str(baseline_path),
        "baseline_composite": cfg_float("baseline_composite"),
        "baseline_h2h_score": h2h_score,
        "baseline_h2h_games": int(h2h_summary["games"]),
        "baseline_h2h_wins": int(h2h_summary["checkpoint_a_wins"]),
        "baseline_h2h_losses": int(h2h_summary["checkpoint_b_wins"]),
        "baseline_h2h_draws": int(h2h_summary["draws"]),
        "absolute_fail_count": absolute_fail_count,
        "h2h_fail_count": h2h_fail_count,
    }
    return absolute_fail_count, h2h_fail_count, stats, bool(absolute_failed or h2h_failed)


def absolute_gate_message(stats: dict[str, Any]) -> str:
    return (
        f"absolute gate baseline={stats.get('baseline_checkpoint')} "
        f"h2h={float(stats.get('baseline_h2h_score', 0.0)):.3f} "
        f"abs_fail={int(stats.get('absolute_fail_count', 0))} "
        f"h2h_fail={int(stats.get('h2h_fail_count', 0))}"
    )


__all__ = [
    "absolute_gate_message",
    "evaluate_absolute_gate",
]

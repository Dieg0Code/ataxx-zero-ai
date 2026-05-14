from __future__ import annotations

from pathlib import Path
from typing import Any

from inference.checkpoint_duel_runtime import run_match_results_to_summary
from model.registry import resolve as resolve_checkpoint
from training.config_runtime import cfg_float, cfg_int, cfg_str
from training.eval_gating import compute_absolute_score_gate, compute_h2h_gate


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

    baseline_path = resolve_checkpoint(baseline_name)
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

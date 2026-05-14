from __future__ import annotations


def compute_regression_gate(
    *,
    current_score: float,
    best_score: float,
    regression_delta: float,
    current_streak: int,
    patience: int,
) -> tuple[int, bool]:
    """
    Update regression streak and indicate whether best-checkpoint restore is recommended.
    """
    if patience <= 0:
        return 0, False

    if current_score + max(0.0, regression_delta) < best_score:
        next_streak = current_streak + 1
    else:
        next_streak = 0
    should_restore = next_streak >= patience
    return next_streak, should_restore


def compute_absolute_score_gate(
    *,
    current_score: float,
    baseline_score: float,
    delta: float,
    current_streak: int,
    patience: int,
) -> tuple[int, bool]:
    if patience <= 0:
        return 0, False
    threshold = baseline_score - max(0.0, delta)
    next_streak = current_streak + 1 if current_score < threshold else 0
    return next_streak, next_streak >= patience


def compute_h2h_gate(
    *,
    h2h_score: float,
    min_score: float,
    current_streak: int,
    patience: int,
) -> tuple[int, bool]:
    if patience <= 0:
        return 0, False
    next_streak = current_streak + 1 if h2h_score < min_score else 0
    return next_streak, next_streak >= patience


__all__ = [
    "compute_absolute_score_gate",
    "compute_h2h_gate",
    "compute_regression_gate",
]

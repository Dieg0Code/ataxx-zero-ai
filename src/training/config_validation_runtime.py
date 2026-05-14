from __future__ import annotations

from collections.abc import Callable

from agents.heuristic import is_supported_heuristic_level


def validate_reward_shaping_config(cfg_float: Callable[[str], float]) -> None:
    if cfg_float("reward_shaping_scale") < 0.0:
        raise ValueError("CONFIG['reward_shaping_scale'] must be >= 0.")
    reward_gamma = cfg_float("reward_shaping_gamma")
    if not 0.0 <= reward_gamma <= 1.0:
        raise ValueError("CONFIG['reward_shaping_gamma'] must be in [0, 1].")
    if cfg_float("reward_shaping_material_weight") < 0.0:
        raise ValueError("CONFIG['reward_shaping_material_weight'] must be >= 0.")
    if cfg_float("reward_shaping_mobility_weight") < 0.0:
        raise ValueError("CONFIG['reward_shaping_mobility_weight'] must be >= 0.")
    if cfg_float("reward_shaping_draw_penalty") < 0.0:
        raise ValueError("CONFIG['reward_shaping_draw_penalty'] must be >= 0.")


def validate_supported_heuristic_csv(*, raw_levels: str, setting_name: str) -> None:
    if raw_levels == "":
        return
    for level in [part.strip() for part in raw_levels.split(",") if part.strip()]:
        if not is_supported_heuristic_level(level):
            raise ValueError(
                f"CONFIG['{setting_name}'] contains unsupported level '{level}'.",
            )


def validate_bootstrap_warmup_config(
    *,
    cfg_bool: Callable[[str], bool],
    cfg_int: Callable[[str], int],
    cfg_str: Callable[[str], str],
) -> None:
    # Blocks the post-bootstrap distribution-reset trap from PM04: inheriting
    # weights with a cleared replay and no warmup feeds the mature model into
    # the iter<=12 curriculum (88% heuristic) and corrupts it.
    bootstrap_run_id = cfg_str("hf_bootstrap_run_id").strip()
    if bootstrap_run_id == "":
        return
    if not cfg_bool("hf_reset_iteration"):
        return
    if cfg_int("warmup_games") > 0 and cfg_int("warmup_epochs") > 0:
        return
    raise ValueError(
        f"Unsafe bootstrap config: hf_bootstrap_run_id='{bootstrap_run_id}' "
        "with hf_reset_iteration=true requires warmup_games > 0 AND "
        "warmup_epochs > 0. See src/model/docs/postmortem/04/README.md.",
    )


def validate_absolute_eval_gate_config(
    *,
    cfg_float: Callable[[str], float],
    cfg_int: Callable[[str], int],
    cfg_str: Callable[[str], str],
) -> None:
    for key in ("baseline_composite", "baseline_h2h_min_score"):
        if not 0.0 <= cfg_float(key) <= 1.0:
            raise ValueError(f"CONFIG['{key}'] must be in [0, 1].")
    if cfg_int("eval_absolute_patience") < 0:
        raise ValueError("CONFIG['eval_absolute_patience'] must be >= 0.")
    if cfg_float("eval_absolute_delta") < 0.0:
        raise ValueError("CONFIG['eval_absolute_delta'] must be >= 0.")
    if cfg_str("eval_absolute_action") not in {"abort", "restore_best", "warn"}:
        raise ValueError("CONFIG['eval_absolute_action'] must be abort, restore_best, or warn.")


__all__ = [
    "validate_absolute_eval_gate_config",
    "validate_bootstrap_warmup_config",
    "validate_reward_shaping_config",
    "validate_supported_heuristic_csv",
]

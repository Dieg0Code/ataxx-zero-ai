from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE, OBSERVATION_CHANNELS

QUALITY_EXCLUDE = {"bad", "bug"}
QUALITY_PRIORITIZE = {"good", "brilliant", "late_game", "demo"}


@dataclass(frozen=True)
class CuratedDataset:
    observations: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    report: dict[str, Any]


def _load_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_human_source(metadata: dict[str, Any], fallback: bool) -> bool:
    player_human = metadata.get("player_human")
    if isinstance(player_human, str) and player_human.strip() != "":
        return True
    labels = [
        metadata.get("starter"),
        metadata.get("p1_label"),
        metadata.get("p2_label"),
        metadata.get("mode"),
    ]
    if any(isinstance(item, str) and "human" in item.lower() for item in labels):
        return True
    return fallback


def _phase_counts(observations: np.ndarray) -> dict[str, int]:
    if observations.size == 0:
        return {"opening": 0, "mid": 0, "late": 0}
    phase = observations[:, 3, 0, 0]
    return {
        "opening": int(np.sum(phase < 0.25)),
        "mid": int(np.sum((phase >= 0.25) & (phase < 0.75))),
        "late": int(np.sum(phase >= 0.75)),
    }


def _validate_arrays(
    observations: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    n = min(len(observations), len(policies), len(values))
    reasons = {
        "shape": 0,
        "finite": 0,
        "policy_negative": 0,
        "policy_sum": 0,
        "value_range": 0,
    }
    if (
        observations.ndim != 4
        or observations.shape[1:] != (OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        or policies.ndim != 2
        or policies.shape[1] != ACTION_SPACE.num_actions
        or values.ndim != 1
    ):
        reasons["shape"] = n
        return np.zeros((n,), dtype=bool), reasons

    valid = np.ones((n,), dtype=bool)
    obs = observations[:n]
    pol = policies[:n]
    val = values[:n]

    finite_mask = np.isfinite(obs).all(axis=(1, 2, 3)) & np.isfinite(pol).all(axis=1)
    finite_mask &= np.isfinite(val)
    reasons["finite"] = int(np.sum(valid & ~finite_mask))
    valid &= finite_mask

    non_negative_mask = np.all(pol >= 0.0, axis=1)
    reasons["policy_negative"] = int(np.sum(valid & ~non_negative_mask))
    valid &= non_negative_mask

    policy_sums = np.sum(pol, axis=1)
    policy_mask = policy_sums > 0.0
    reasons["policy_sum"] = int(np.sum(valid & ~policy_mask))
    valid &= policy_mask

    value_mask = (val >= -1.0) & (val <= 1.0)
    reasons["value_range"] = int(np.sum(valid & ~value_mask))
    valid &= value_mask
    return valid, reasons


def curate_npz_paths(
    paths: list[Path],
    *,
    human_oversample: int = 4,
    human_path_hint: str = "play_sessions",
) -> CuratedDataset:
    observations_out: list[np.ndarray] = []
    policies_out: list[np.ndarray] = []
    values_out: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    total_discarded = 0

    for path in paths:
        metadata = _load_sidecar(path)
        source_report: dict[str, Any] = {
            "path": path.as_posix(),
            "input_examples": 0,
            "kept_examples": 0,
            "discarded_examples": 0,
            "discard_reasons": {},
            "human": False,
            "oversample_factor": 1,
        }
        try:
            data = np.load(path)
            observations = np.asarray(data["observations"], dtype=np.float32)
            policies = np.asarray(data["policies"], dtype=np.float32)
            values = np.asarray(data["values"], dtype=np.float32)
        except (OSError, KeyError, ValueError):
            source_report["discard_reasons"] = {"load_error": 1}
            sources.append(source_report)
            total_discarded += 1
            continue

        n = min(len(observations), len(policies), len(values))
        total_in += n
        source_report["input_examples"] = n
        quality_tag = str(metadata.get("quality_tag", "")).strip().lower()
        is_human = _is_human_source(metadata, human_path_hint in path.as_posix())
        useful_human_draw = is_human and quality_tag in QUALITY_PRIORITIZE
        if quality_tag in QUALITY_EXCLUDE or (
            bool(metadata.get("forced_draw", False)) and not useful_human_draw
        ):
            source_report["discarded_examples"] = n
            source_report["discard_reasons"] = {"metadata_excluded": n}
            source_report["human"] = is_human
            sources.append(source_report)
            total_discarded += n
            continue

        valid, reasons = _validate_arrays(observations, policies, values)
        kept = int(np.sum(valid))
        source_report["discarded_examples"] = int(n - kept)
        source_report["discard_reasons"] = {k: v for k, v in reasons.items() if v > 0}
        if kept > 0:
            factor = max(1, int(human_oversample)) if is_human else 1
            if quality_tag in QUALITY_PRIORITIZE and is_human:
                factor = max(factor, int(human_oversample))
            kept_policies = policies[:n][valid]
            kept_policies = kept_policies / np.sum(kept_policies, axis=1, keepdims=True)
            observations_out.extend([observations[:n][valid]] * factor)
            policies_out.extend([kept_policies] * factor)
            values_out.extend([values[:n][valid]] * factor)
            source_report["human"] = is_human
            source_report["oversample_factor"] = factor
            source_report["kept_examples"] = kept * factor
            total_out += kept * factor
        total_discarded += int(n - kept)
        sources.append(source_report)

    if observations_out:
        observations_final = np.concatenate(observations_out).astype(np.float32, copy=False)
        policies_final = np.concatenate(policies_out).astype(np.float32, copy=False)
        values_final = np.concatenate(values_out).astype(np.float32, copy=False)
    else:
        observations_final = np.zeros(
            (0, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
        policies_final = np.zeros((0, ACTION_SPACE.num_actions), dtype=np.float32)
        values_final = np.zeros((0,), dtype=np.float32)

    report = {
        "input_examples": total_in,
        "kept_examples": len(values_final),
        "discarded_examples": total_discarded,
        "phase_counts": _phase_counts(observations_final),
        "sources": sources,
    }
    if total_out != len(values_final):
        report["kept_examples"] = len(values_final)
    return CuratedDataset(observations_final, policies_final, values_final, cast(dict[str, Any], report))


def save_curated_dataset(dataset: CuratedDataset, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=dataset.observations,
        policies=dataset.policies,
        values=dataset.values,
    )
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(dataset.report, indent=2), encoding="utf-8")
    return report_path


__all__ = [
    "CuratedDataset",
    "curate_npz_paths",
    "save_curated_dataset",
]

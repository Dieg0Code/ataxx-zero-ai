from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_STATS_PATH = Path.home() / ".ataxx_arena_stats.json"


def _key(p1: str, p2: str) -> str:
    return f"{p1}__VS__{p2}"


def load_stats() -> dict[str, Any]:
    if not _STATS_PATH.is_file():
        return {}
    try:
        return json.loads(_STATS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_stats(stats: dict[str, Any]) -> None:
    try:
        _STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    except OSError:
        pass


def record_result(p1_label: str, p2_label: str, result: int) -> dict[str, int]:
    """Persist a result for the (p1, p2) pair. result=+1 P1 win, -1 P2 win, 0 draw."""
    stats = load_stats()
    key = _key(p1_label, p2_label)
    bucket = stats.get(key, {"w": 0, "l": 0, "d": 0})
    if result > 0:
        bucket["w"] += 1
    elif result < 0:
        bucket["l"] += 1
    else:
        bucket["d"] += 1
    stats[key] = bucket
    save_stats(stats)
    return bucket


def get_record(p1_label: str, p2_label: str) -> dict[str, int]:
    stats = load_stats()
    return stats.get(_key(p1_label, p2_label), {"w": 0, "l": 0, "d": 0})

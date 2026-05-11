"""Single source of truth for trained model generations.

Reads `checkpoints/registry.json`. Allows resolving codenames (e.g. "liga",
"centinela", "bogo") or versions ("v8") to concrete checkpoint paths, ranking
generations by eval score, and persisting fresh eval results in-place.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
REGISTRY_PATH = CHECKPOINT_DIR / "registry.json"


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Registry not found at {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(data: dict[str, Any]) -> None:
    REGISTRY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def list_models() -> list[dict[str, Any]]:
    return load_registry().get("models", [])


def find_model(name: str) -> dict[str, Any] | None:
    """Find a model entry by codename, version, or filename. Case-insensitive."""
    if not name:
        return None
    needle = name.strip().lower()
    for entry in list_models():
        if entry.get("codename", "").lower() == needle:
            return entry
        if entry.get("version", "").lower() == needle:
            return entry
        if entry.get("file", "").lower() == needle:
            return entry
        # Allow stem match (without .pt)
        if Path(entry.get("file", "")).stem.lower() == needle:
            return entry
    return None


def resolve(name_or_path: str) -> Path:
    """Resolve a codename, version, filename, special alias, or path to a Path.

    Special aliases:
      - "latest": most recently trained model
      - "best":   highest composite eval score (falls back to latest if no scores)
      - "default": registry's default_codename
    """
    if not name_or_path:
        raise ValueError("Empty checkpoint name/path")

    candidate = name_or_path.strip()
    lowered = candidate.lower()

    if lowered in ("latest", "newest"):
        models = sorted(
            list_models(),
            key=lambda m: m.get("trained_at", ""),
            reverse=True,
        )
        if not models:
            raise ValueError("No models in registry")
        return CHECKPOINT_DIR / models[0]["file"]

    if lowered == "best":
        ranked = ranked_models(metric="composite")
        if ranked:
            return CHECKPOINT_DIR / ranked[0]["file"]
        # Fallback to latest if no eval data exists yet.
        return resolve("latest")

    if lowered == "default":
        try:
            data = load_registry()
            default = data.get("default_codename")
            if default:
                return resolve(default)
        except FileNotFoundError:
            pass

    # Try registry lookup first.
    entry = find_model(candidate)
    if entry is not None:
        return CHECKPOINT_DIR / entry["file"]

    # Fall back to treating it as a filesystem path (absolute or relative).
    direct = Path(candidate)
    if direct.is_file():
        return direct
    repo_relative = REPO_ROOT / candidate
    if repo_relative.is_file():
        return repo_relative
    ckpt_relative = CHECKPOINT_DIR / candidate
    if ckpt_relative.is_file():
        return ckpt_relative

    raise FileNotFoundError(
        f"Could not resolve '{name_or_path}' to a checkpoint. "
        f"Try a codename ({', '.join(m['codename'] for m in list_models())}), "
        f"a version (v1..v8), or a path."
    )


def ranked_models(metric: str = "composite") -> list[dict[str, Any]]:
    """Return models sorted by the given eval metric, highest first.

    Metrics:
      - "composite": average score vs heuristic gauntlet
      - "rr": round-robin score (head-to-head among generations)
      - "combined": average of composite and rr (only if both present)
      - "<level>": specific heuristic level (hard, apex, sentinel, …)

    Models without a score for the requested metric go to the end.
    """
    def score_for(entry: dict[str, Any]) -> float | None:
        eval_data = entry.get("eval", {})
        if metric == "composite":
            return eval_data.get("composite")
        if metric == "rr":
            rr = eval_data.get("round_robin")
            return rr.get("score") if isinstance(rr, dict) else None
        if metric == "combined":
            comp = eval_data.get("composite")
            rr = eval_data.get("round_robin", {}).get("score") if isinstance(eval_data.get("round_robin"), dict) else None
            if comp is None or rr is None:
                return None
            return (float(comp) + float(rr)) / 2.0
        per_level = eval_data.get("vs_heuristic", {}).get(metric)
        if isinstance(per_level, dict):
            return per_level.get("score")
        return None

    with_score: list[tuple[float, dict[str, Any]]] = []
    without_score: list[dict[str, Any]] = []
    for entry in list_models():
        s = score_for(entry)
        if s is None:
            without_score.append(entry)
        else:
            with_score.append((float(s), entry))
    with_score.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in with_score] + without_score


def update_eval(codename: str, eval_data: dict[str, Any]) -> dict[str, Any]:
    """Persist new eval results for a model. Returns the updated entry."""
    data = load_registry()
    target = None
    for entry in data.get("models", []):
        if entry.get("codename") == codename:
            target = entry
            break
    if target is None:
        raise KeyError(f"Codename '{codename}' not in registry")
    existing = target.get("eval", {})
    # Merge per-level results so callers can update one level at a time.
    if "vs_heuristic" in eval_data:
        merged = dict(existing.get("vs_heuristic", {}))
        merged.update(eval_data["vs_heuristic"])
        eval_data["vs_heuristic"] = merged
    if "head_to_head" in eval_data:
        merged_h2h = dict(existing.get("head_to_head", {}))
        merged_h2h.update(eval_data["head_to_head"])
        eval_data["head_to_head"] = merged_h2h
    target["eval"] = {**existing, **eval_data}
    save_registry(data)
    return target


__all__ = [
    "CHECKPOINT_DIR",
    "REGISTRY_PATH",
    "find_model",
    "list_models",
    "load_registry",
    "ranked_models",
    "resolve",
    "save_registry",
    "update_eval",
]

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint against one or more heuristic levels.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Codename (liga, centinela…), version (v8), or path to .pt/.ckpt.",
    )
    parser.add_argument(
        "--levels",
        default="hard,apex,sentinel",
        help="Comma-separated heuristic levels (e.g. easy,normal,hard,apex,gambit,sentinel).",
    )
    parser.add_argument("--games", type=int, default=64, help="Games per level.")
    parser.add_argument("--sims", type=int, default=160, help="MCTS simulations per move.")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args()


def evaluate_checkpoint_vs_levels(
    checkpoint: Path,
    *,
    levels: list[str],
    games: int = 64,
    sims: int = 160,
    c_puct: float = 1.5,
    seed: int = 42,
    device: str = "auto",
    progress: bool = True,
) -> dict[str, Any]:
    """Run a gauntlet of `games` matches per level. Reusable from other scripts."""
    _ensure_src_on_path()
    from agents.heuristic import is_supported_heuristic_level
    from inference.checkpoint_duel_runtime import (
        load_system_from_checkpoint,
        resolve_device,
    )
    from training.eval_runtime import evaluate_model

    for level in levels:
        if not is_supported_heuristic_level(level):
            raise ValueError(f"Unsupported heuristic level: {level}")

    resolved_device = resolve_device(device)
    system = load_system_from_checkpoint(checkpoint, device=resolved_device)

    results: dict[str, dict[str, float | int | str]] = {}
    se_estimate = 0.5 / max(1, games) ** 0.5
    for idx, level in enumerate(levels):
        stats = evaluate_model(
            system=system,
            device=resolved_device,
            games=int(games),
            sims=int(sims),
            c_puct=float(c_puct),
            heuristic_level=level,
            seed=int(seed) + idx * 997,
        )
        results[level] = stats
        if progress:
            wins = int(stats["wins"])
            losses = int(stats["losses"])
            draws = int(stats["draws"])
            score = float(stats["score"])
            ci_low = max(0.0, score - 1.96 * se_estimate)
            ci_high = min(1.0, score + 1.96 * se_estimate)
            print(
                f"vs {level:<10} W={wins} L={losses} D={draws} "
                f"score={score:.3f}  IC95%=[{ci_low:.3f}, {ci_high:.3f}]",
            )

    composite = sum(float(stats["score"]) for stats in results.values()) / max(1, len(results))
    return {
        "composite": composite,
        "levels": results,
        "device": resolved_device,
        "games": games,
        "sims": sims,
    }


def main() -> None:
    args = _parse_args()
    _ensure_src_on_path()
    from model.registry import resolve as resolve_checkpoint

    checkpoint = resolve_checkpoint(args.checkpoint)
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]

    se_estimate = 0.5 / max(1, args.games) ** 0.5
    print(f"Checkpoint : {checkpoint}")
    print(f"Games/level: {args.games} (SE~{se_estimate:.3f})")
    print(f"MCTS sims  : {args.sims}")
    print(f"Levels     : {', '.join(levels)}")
    print("")

    summary = evaluate_checkpoint_vs_levels(
        checkpoint,
        levels=levels,
        games=int(args.games),
        sims=int(args.sims),
        c_puct=float(args.c_puct),
        seed=int(args.seed),
        device=args.device,
        progress=True,
    )

    print("")
    print(f"COMPOSITE  score={summary['composite']:.3f}")

    if args.json:
        print("")
        print(json.dumps(
            {"composite": summary["composite"], "levels": summary["levels"]},
            indent=2,
            default=str,
        ))


if __name__ == "__main__":
    main()

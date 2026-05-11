"""Run a gauntlet eval (vs hard/apex/sentinel by default) for every model in the
registry and persist the results back to checkpoints/registry.json.

Resumable: existing scores are preserved unless --overwrite is passed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa todos los modelos del registry contra heurísticas y guarda resultados."
    )
    parser.add_argument("--levels", default="hard,apex,sentinel")
    parser.add_argument("--games", type=int, default=12, help="Games por nivel (default 12).")
    parser.add_argument("--sims", type=int, default=80, help="MCTS sims (default 80).")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--only",
        default="",
        help="Lista separada por comas de codenames a evaluar (default: todos).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-ejecutar incluso si ya hay resultados (source != 'pending').",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _force_utf8_stdout()
    _ensure_src_on_path()

    from model.registry import list_models, update_eval

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_checkpoint_vs_heuristic import evaluate_checkpoint_vs_levels

    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    models = list_models()
    if only:
        models = [m for m in models if m.get("codename") in only]

    print(f"Modelos a evaluar : {[m['codename'] for m in models]}")
    print(f"Niveles           : {levels}")
    print(f"Games/nivel       : {args.games}")
    print(f"MCTS sims         : {args.sims}")
    print("")

    from model.registry import CHECKPOINT_DIR

    for model in models:
        codename = model["codename"]
        existing_source = model.get("eval", {}).get("source", "pending")
        if existing_source != "pending" and not args.overwrite:
            print(f"[skip] {codename}: ya tiene eval (source={existing_source}). Usá --overwrite para repetir.")
            continue

        ckpt = CHECKPOINT_DIR / model["file"]
        if not ckpt.is_file():
            print(f"[warn] {codename}: archivo no existe en disco ({ckpt}). Saltando.")
            continue

        print(f"\n=== Evaluando {codename} ({model['version']}) ===")
        print(f"  archivo: {ckpt.name}")
        try:
            summary = evaluate_checkpoint_vs_levels(
                ckpt,
                levels=levels,
                games=int(args.games),
                sims=int(args.sims),
                c_puct=float(args.c_puct),
                seed=int(args.seed),
                device=args.device,
                progress=True,
            )
        except Exception as exc:
            print(f"[error] {codename}: {exc}")
            continue

        vs_heuristic = {
            level: {
                "games": int(stats.get("games", args.games)) if isinstance(stats, dict) else args.games,
                "score": float(stats.get("score", 0.0)) if isinstance(stats, dict) else 0.0,
                "wins": int(stats.get("wins", 0)) if isinstance(stats, dict) else 0,
                "losses": int(stats.get("losses", 0)) if isinstance(stats, dict) else 0,
                "draws": int(stats.get("draws", 0)) if isinstance(stats, dict) else 0,
                "source": "gauntlet",
            }
            for level, stats in summary["levels"].items()
        }
        update_eval(
            codename,
            {
                "vs_heuristic": vs_heuristic,
                "composite": float(summary["composite"]),
                "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "source": "gauntlet",
                "config": {"games": args.games, "sims": args.sims, "c_puct": args.c_puct},
            },
        )
        print(f"  COMPOSITE = {summary['composite']:.3f}  (guardado)")

    print("\nListo. Para ver el ranking actualizado:")
    print("  uv run python scripts/list_models.py")


if __name__ == "__main__":
    main()

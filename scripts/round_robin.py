"""Round-robin head-to-head entre todas las generaciones del registry.

Cada par juega `--games` partidas (mitad cada uno como P1). Persiste el score
del par en `head_to_head` de ambos modelos en `checkpoints/registry.json`.

Uso típico:
    uv run python scripts/round_robin.py --games 8 --sims 80
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
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
    parser = argparse.ArgumentParser(description="Round-robin entre generaciones del registry.")
    parser.add_argument("--games", type=int, default=8, help="Partidas por par (mitad cada lado).")
    parser.add_argument("--sims", type=int, default=80, help="MCTS sims por jugada.")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="Seed base. Cada par usa seed+offset.")
    parser.add_argument(
        "--only",
        default="",
        help="Codenames separados por coma — restringir round-robin a un subconjunto.",
    )
    parser.add_argument(
        "--exclude",
        default="aprendiz-tardio",
        help="Codenames a excluir (default: aprendiz-tardio para no duplicar identidad).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _force_utf8_stdout()
    _ensure_src_on_path()

    from inference.checkpoint_duel_runtime import (
        play_checkpoint_match_results,
        resolve_device,
        summarize_match_results,
    )
    from model.registry import CHECKPOINT_DIR, list_models, update_eval

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    models = [m for m in list_models() if m.get("file")]
    if only:
        models = [m for m in models if m["codename"] in only]
    models = [m for m in models if m["codename"] not in exclude]

    print(f"Round-robin entre: {[m['codename'] for m in models]}")
    print(f"Games por par: {args.games}  |  MCTS sims: {args.sims}")
    print("")

    device = resolve_device(args.device)
    pairs = list(itertools.combinations(models, 2))
    aggregate: dict[str, dict[str, float | int]] = {
        m["codename"]: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        for m in models
    }

    for idx, (a, b) in enumerate(pairs, start=1):
        ckpt_a = CHECKPOINT_DIR / a["file"]
        ckpt_b = CHECKPOINT_DIR / b["file"]
        if not ckpt_a.is_file() or not ckpt_b.is_file():
            print(f"[skip] {a['codename']} vs {b['codename']}: archivo faltante.")
            continue

        print(f"[{idx}/{len(pairs)}] {a['codename']:16s} vs {b['codename']:16s}", flush=True)
        try:
            results = play_checkpoint_match_results(
                checkpoint_a=ckpt_a,
                checkpoint_b=ckpt_b,
                games=int(args.games),
                device=device,
                mcts_sims=int(args.sims),
                c_puct=float(args.c_puct),
                seed=int(args.seed) + idx * 1009,
            )
            summary = summarize_match_results(results=results)
        except Exception as exc:
            print(f"  [error] {exc}")
            continue

        a_wins = int(summary["checkpoint_a_wins"])
        b_wins = int(summary["checkpoint_b_wins"])
        draws = int(summary["draws"])
        a_score = float(summary["checkpoint_a_score"])
        b_score = 1.0 - a_score
        print(
            f"  {a['codename']} {a_wins}-{b_wins}-{draws} {b['codename']}  "
            f"({a['codename']} score={a_score:.3f})"
        )

        # Update aggregates
        aggregate[a["codename"]]["games"] = int(aggregate[a["codename"]]["games"]) + a_wins + b_wins + draws
        aggregate[a["codename"]]["wins"] = int(aggregate[a["codename"]]["wins"]) + a_wins
        aggregate[a["codename"]]["losses"] = int(aggregate[a["codename"]]["losses"]) + b_wins
        aggregate[a["codename"]]["draws"] = int(aggregate[a["codename"]]["draws"]) + draws
        aggregate[b["codename"]]["games"] = int(aggregate[b["codename"]]["games"]) + a_wins + b_wins + draws
        aggregate[b["codename"]]["wins"] = int(aggregate[b["codename"]]["wins"]) + b_wins
        aggregate[b["codename"]]["losses"] = int(aggregate[b["codename"]]["losses"]) + a_wins
        aggregate[b["codename"]]["draws"] = int(aggregate[b["codename"]]["draws"]) + draws

        # Persist per-pair head_to_head (symmetric)
        update_eval(a["codename"], {"head_to_head": {b["codename"]: round(a_score, 4)}})
        update_eval(b["codename"], {"head_to_head": {a["codename"]: round(b_score, 4)}})

    # Compute per-model rr_score (avg points / total games)
    print("\nRanking round-robin")
    print("-" * 60)
    print(f"{'Codename':<16} {'W-L-D':<10} {'Score':>6}  {'Games':>5}")
    rows = []
    for codename, agg in aggregate.items():
        games = int(agg["games"])
        wins = int(agg["wins"])
        losses = int(agg["losses"])
        draws = int(agg["draws"])
        score = (wins + 0.5 * draws) / games if games > 0 else None
        rows.append((codename, wins, losses, draws, score, games))
    rows.sort(key=lambda r: (-1.0 if r[4] is None else r[4]), reverse=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for codename, w, l, d, score, games in rows:
        sc = "  ?  " if score is None else f"{score:.3f}"
        print(f"{codename:<16} {w}-{l}-{d:<3}     {sc:>6}  {games:>5}")
        if score is not None:
            update_eval(
                codename,
                {
                    "round_robin": {
                        "score": round(float(score), 4),
                        "wins": w,
                        "losses": l,
                        "draws": d,
                        "games": games,
                        "evaluated_at": now,
                    },
                },
            )

    print("\nLista actualizada en checkpoints/registry.json")
    print("  uv run python scripts/list_models.py --metric rr")
    print("  uv run python scripts/list_models.py --metric combined")


if __name__ == "__main__":
    main()

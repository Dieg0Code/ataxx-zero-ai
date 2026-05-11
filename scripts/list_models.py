"""Print a ranked table of all model generations from the registry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _force_utf8_stdout() -> None:
    # Windows consoles default to cp1252 which mangles accents and arrows.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _short_lore(lore: str, width: int = 70) -> str:
    if not lore:
        return "—"
    if len(lore) <= width:
        return lore
    return lore[: width - 1] + "…"


def _format_score(s: float | None) -> str:
    return "  ?  " if s is None else f"{s:.3f}"


def _format_date(iso: str | None) -> str:
    return "—" if not iso else iso.split("T")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabla rankeada de generaciones de modelos.")
    parser.add_argument(
        "--metric",
        default="combined",
        choices=["combined", "composite", "rr", "hard", "apex", "sentinel", "easy", "normal", "gambit"],
        help="Métrica de ranking (default: combined = composite + round-robin).",
    )
    parser.add_argument("--full", action="store_true", help="Lore completo + hparams.")
    parser.add_argument("--ascii", action="store_true", help="(legacy) usar separadores ASCII; ya es default.")
    parser.add_argument("--json", action="store_true", help="Emite JSON.")
    args = parser.parse_args()

    _force_utf8_stdout()
    _ensure_src_on_path()
    from model.registry import ranked_models

    rows = ranked_models(metric=args.metric)

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    print(f"RANKING (metric={args.metric})")
    print("=" * 92)
    print(f"{'#':>2}  {'Apodo':<16} {'Ver':<5} {'Score':>6}  {'Fecha':<11}  Lore")
    print("-" * 92)
    for i, entry in enumerate(rows, start=1):
        codename = entry.get("codename", "?")
        version = entry.get("version", "?")
        eval_data = entry.get("eval", {})
        if args.metric == "composite":
            score = eval_data.get("composite")
        elif args.metric == "rr":
            rr = eval_data.get("round_robin")
            score = rr.get("score") if isinstance(rr, dict) else None
        elif args.metric == "combined":
            comp = eval_data.get("composite")
            rr_obj = eval_data.get("round_robin")
            rr_score = rr_obj.get("score") if isinstance(rr_obj, dict) else None
            score = (float(comp) + float(rr_score)) / 2.0 if (comp is not None and rr_score is not None) else None
        else:
            per = eval_data.get("vs_heuristic", {}).get(args.metric)
            score = per.get("score") if isinstance(per, dict) else None
        date = _format_date(entry.get("trained_at"))
        lore = entry.get("lore", "")
        print(
            f"{i:>2}  {codename:<16} {version:<5} {_format_score(score):>6}  {date:<11}  {_short_lore(lore)}"
        )

    print("")
    print("Tip: usar el apodo en CLI →")
    print("  uv run python scripts/play_pygame.py --opponent model --ckpt liga --sims 80")
    print("  uv run python scripts/eval_checkpoint_vs_heuristic.py --checkpoint centinela --games 12")
    print("  uv run python scripts/compare_checkpoints.py --checkpoint-a liga --checkpoint-b centinela --games 16")

    if args.full:
        print("")
        print("DETALLE")
        print("=" * 92)
        for entry in rows:
            print(f"\n{entry.get('codename')} ({entry.get('version')})")
            print(f"  archivo  : {entry.get('file')}")
            print(f"  fecha    : {entry.get('trained_at')}")
            print(f"  iter     : {entry.get('iter')}")
            hparams = entry.get("hparams", {})
            if hparams:
                hp = ", ".join(f"{k}={v}" for k, v in hparams.items())
                print(f"  hparams  : {hp}")
            pm = entry.get("postmortem")
            if pm:
                print(f"  postmortem: {pm}")
            print(f"  lore     : {entry.get('lore', '')}")
            ev = entry.get("eval", {})
            vsh = ev.get("vs_heuristic", {})
            if vsh:
                bits = [f"{lvl}={info.get('score', '?'):.3f}" for lvl, info in vsh.items() if isinstance(info, dict)]
                print(f"  eval     : composite={ev.get('composite', '?')}  " + "  ".join(bits))
            h2h = ev.get("head_to_head", {})
            if h2h:
                print(f"  head-to-head: " + ", ".join(f"vs {k}: {v}" for k, v in h2h.items()))


if __name__ == "__main__":
    main()

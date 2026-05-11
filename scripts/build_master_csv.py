"""Combine all per-run history CSVs into one master CSV ready for Power BI.

Reads `runs_history/policy_spatial_v*/*.csv` and produces
`runs_history/all_runs.csv` with codename/version/run_id columns added so the
data can be sliced by generation in Power BI.

Usage:
    uv run python scripts/build_master_csv.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HISTORY_DIR = REPO / "runs_history"
OUT_PATH = HISTORY_DIR / "all_runs.csv"

# Mapping run_id → (version, codename). For runs without a checkpoint in our
# registry (v1, v5 — both failed early), we propose tentative codenames.
RUN_METADATA: dict[str, tuple[str, str]] = {
    "policy_spatial_v1": ("v1s", "embrion"),       # spatial v1, peak 0.0 — first spatial form
    "policy_spatial_v2": ("v2",  "reflejo"),
    "policy_spatial_v3": ("v3",  "chispazo"),
    "policy_spatial_v4": ("v4",  "aprendiz"),
    "policy_spatial_v5": ("v5",  "tropezon"),      # peak 0.028 — failed mid-run
    "policy_spatial_v6": ("v6",  "centinela"),
    "policy_spatial_v7": ("v7",  "amnesia"),
    "policy_spatial_v8": ("v8",  "liga"),
}

INPUT_COLUMNS = [
    "iter",
    "timestamp",
    "replay_size",
    "score",
    "best_eval_score",
    "eval_total_wins",
    "eval_total_losses",
    "eval_total_draws",
    "eval_score_hard",
    "eval_score_apex",
    "eval_score_sentinel",
    "train_loss_total",
    "train_loss_value",
    "train_loss_policy",
    "train_value_mae",
    "train_policy_accuracy",
    "train_lr",
]

OUTPUT_COLUMNS = [
    "run_id",
    "version",
    "codename",
    "iter",
    "is_final_iter",
    "timestamp",
    "date",
    "replay_size",
    "score",
    "best_eval_score",
    "eval_total_wins",
    "eval_total_losses",
    "eval_total_draws",
    "eval_total_games",
    "eval_score_hard",
    "eval_score_apex",
    "eval_score_sentinel",
    "train_loss_total",
    "train_loss_value",
    "train_loss_policy",
    "train_value_mae",
    "train_policy_accuracy",
    "train_lr",
]


def load_run(run_id: str) -> list[dict[str, str]]:
    csv_path = HISTORY_DIR / run_id / f"{run_id}_history.csv"
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def enrich_row(
    row: dict[str, str], *, run_id: str, version: str, codename: str, max_iter: int
) -> dict[str, object]:
    iter_val = to_int(row.get("iter"))
    wins = to_int(row.get("eval_total_wins"))
    losses = to_int(row.get("eval_total_losses"))
    draws = to_int(row.get("eval_total_draws"))
    eval_total = (
        (wins or 0) + (losses or 0) + (draws or 0)
        if any(v is not None for v in (wins, losses, draws))
        else None
    )
    timestamp = row.get("timestamp", "") or ""
    date = timestamp.split("T")[0] if timestamp else ""
    enriched: dict[str, object] = {
        "run_id": run_id,
        "version": version,
        "codename": codename,
        "iter": iter_val,
        "is_final_iter": 1 if iter_val == max_iter else 0,
        "timestamp": timestamp,
        "date": date,
        "eval_total_games": eval_total,
    }
    for col in INPUT_COLUMNS:
        if col in ("iter", "timestamp"):
            continue
        enriched[col] = row.get(col, "")
    return enriched


def main() -> int:
    if not HISTORY_DIR.is_dir():
        print(f"runs_history/ no existe en {REPO}.", file=sys.stderr)
        return 1

    all_rows: list[dict[str, object]] = []
    for run_id, (version, codename) in RUN_METADATA.items():
        rows = load_run(run_id)
        if not rows:
            print(f"[skip] {run_id}: no hay CSV en runs_history/")
            continue
        max_iter = max(
            (to_int(r.get("iter")) or 0 for r in rows),
            default=0,
        )
        for row in rows:
            all_rows.append(
                enrich_row(row, run_id=run_id, version=version, codename=codename, max_iter=max_iter),
            )
        print(f"[ok] {run_id}: {len(rows)} filas, último iter={max_iter}")

    all_rows.sort(key=lambda r: (str(r.get("timestamp") or ""), int(r.get("iter") or 0)))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})

    print(f"\nMaster CSV: {OUT_PATH}")
    print(f"Total filas: {len(all_rows)}")
    print(f"Columnas: {len(OUTPUT_COLUMNS)}")

    # Quick summary per generation
    print("\nResumen por generación:")
    print(f"{'codename':<12} {'version':<6} {'iters':>6} {'peak_eval':>10} {'final_eval':>10}")
    for run_id, (version, codename) in RUN_METADATA.items():
        rows = [r for r in all_rows if r["run_id"] == run_id]
        if not rows:
            continue
        peak = max(
            (float(r["best_eval_score"]) for r in rows if r.get("best_eval_score") not in (None, "")),
            default=None,
        )
        last = rows[-1]
        peak_str = f"{peak:.4f}" if peak is not None else "  -"
        last_eval = last.get("best_eval_score") or "-"
        if isinstance(last_eval, str) and last_eval not in ("", "-"):
            try:
                last_eval = f"{float(last_eval):.4f}"
            except ValueError:
                pass
        print(f"{codename:<12} {version:<6} {len(rows):>6} {peak_str:>10} {last_eval:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

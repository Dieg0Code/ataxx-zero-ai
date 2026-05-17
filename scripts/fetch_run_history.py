"""Fetch all metadata_iter_*.json for a HF training run and emit a CSV.

Usage:
    python scripts/fetch_run_history.py policy_spatial_v7
    python scripts/fetch_run_history.py policy_spatial_v6 --repo-id dieg0code/ataxx-zero

The script downloads the lightweight metadata files from the HF model repo and
flattens them into one CSV row per iteration. Heavy artifacts (.pt, .npz) are
ignored. Token is read from --token, env HF_TOKEN, or anonymous (works for
public repos).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO = "dieg0code/ataxx-zero"

CSV_COLUMNS = [
    "iter",
    "timestamp",
    "replay_size",
    "score",
    "best_eval_score",
    "eval_total_wins",
    "eval_total_losses",
    "eval_total_draws",
    "eval_score_easy",
    "eval_score_normal",
    "eval_score_hard",
    "eval_score_apex",
    "eval_score_gambit",
    "eval_score_sentinel",
    "baseline_checkpoint",
    "baseline_h2h_score",
    "baseline_h2h_wins",
    "baseline_h2h_losses",
    "baseline_h2h_draws",
    "baseline_h2h_games",
    "absolute_fail_count",
    "h2h_fail_count",
    "champion_duel_opponent",
    "champion_duel_score",
    "champion_duel_wins",
    "champion_duel_losses",
    "champion_duel_draws",
    "champion_duel_games",
    "train_loss_total",
    "train_loss_value",
    "train_loss_policy",
    "train_value_mae",
    "train_policy_accuracy",
    "train_lr",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", help="HF run id, e.g. policy_spatial_v7")
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--out-dir", default=None, help="Default: runs_history/<run_id>/")
    p.add_argument("--token", default=None, help="HF token (falls back to env HF_TOKEN)")
    return p.parse_args()


def iter_from_filename(name: str) -> int:
    # metadata_iter_140.json -> 140
    return int(Path(name).stem.split("_")[2])


def download_metadata_files(
    *, api: HfApi, repo_id: str, run_id: str, token: str | None, out_dir: Path
) -> list[Path]:
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
    prefix = f"runs/{run_id}/metadata_iter_"
    targets = sorted([f for f in files if f.startswith(prefix) and f.endswith(".json")])
    if not targets:
        print(f"No metadata_iter_*.json found under runs/{run_id}/ in {repo_id}.")
        return []
    print(f"Found {len(targets)} metadata files. Downloading...")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, repo_path in enumerate(targets, 1):
        local = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="model",
            token=token,
            local_dir=str(out_dir),
        )
        paths.append(Path(local))
        if i % 25 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)}")
    return paths


def flatten_metadata(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data.get("stats", {}) or {}
    row: dict[str, object] = {col: stats.get(col) for col in CSV_COLUMNS}
    row["iter"] = data.get("iteration")
    row["timestamp"] = data.get("timestamp")
    return row


def write_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    rows.sort(key=lambda r: int(r["iter"]) if r["iter"] is not None else -1)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(r["iter"]) if r["iter"] is not None else -1)
    first, last = rows[0], rows[-1]
    best_row = max(
        (r for r in rows if r.get("best_eval_score") is not None),
        key=lambda r: float(r["best_eval_score"]),  # type: ignore[arg-type]
        default=None,
    )
    print()
    print(f"Iters covered: {first['iter']} .. {last['iter']}  ({len(rows)} files)")
    print(f"Final replay_size: {last.get('replay_size')}")
    print(f"Final best_eval_score: {last.get('best_eval_score')}")
    if best_row is not None:
        print(
            "Peak best_eval_score: "
            f"{best_row['best_eval_score']} at iter {best_row['iter']}"
        )
    eval_rows = [r for r in rows if r.get("score") is not None]
    if eval_rows:
        print(f"Iters with eval data: {len(eval_rows)}")


def main() -> int:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or None
    out_dir = Path(args.out_dir or f"runs_history/{args.run_id}")
    api = HfApi()

    paths = download_metadata_files(
        api=api,
        repo_id=args.repo_id,
        run_id=args.run_id,
        token=token,
        out_dir=out_dir,
    )
    if not paths:
        return 1

    rows = [flatten_metadata(p) for p in paths]
    out_csv = out_dir / f"{args.run_id}_history.csv"
    write_csv(rows, out_csv)
    print(f"\nWrote {out_csv}")
    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

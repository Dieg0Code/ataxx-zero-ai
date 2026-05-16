from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from fetch_hf_buffers import select_buffer_files
from fetch_run_history import (
    DEFAULT_REPO,
    download_metadata_files,
    flatten_metadata,
    summarize,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a full HF training run for local inspection."
    )
    parser.add_argument("run_id", help="HF run id, e.g. policy_spatial_v10")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--out-dir", default=None, help="Default: runs_history/<run_id>")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Local playable checkpoint directory.",
    )
    parser.add_argument(
        "--skip-buffers",
        action="store_true",
        help="Only download metadata and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=int,
        default=[],
        help="Extra checkpoint iteration to download. Can be passed more than once.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def best_eval_iter(rows: list[dict[str, object]]) -> int | None:
    scored_rows = [r for r in rows if r.get("score") not in (None, "")]
    if scored_rows:
        best = max(scored_rows, key=lambda r: float(r["score"]))  # type: ignore[arg-type]
        return int(best["iter"]) if best.get("iter") is not None else None
    best_rows = [r for r in rows if r.get("best_eval_score") not in (None, "")]
    if best_rows:
        best = max(
            best_rows,
            key=lambda r: float(r["best_eval_score"]),  # type: ignore[arg-type]
        )
        return int(best["iter"]) if best.get("iter") is not None else None
    return None


def download_buffers(
    *, api: object, repo_id: str, run_id: str, token: str | None, out_dir: Path
) -> int:
    hub_mod = __import__("huggingface_hub", fromlist=["hf_hub_download"])
    hf_hub_download = hub_mod.hf_hub_download
    files = select_buffer_files(
        list(api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)),
        run_id=run_id,
    )
    if not files:
        print(f"No buffers found for run_id={run_id}.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(files)} buffer files...")
    for index, filename in enumerate(files, 1):
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=str(out_dir),
        )
        if index % 25 == 0 or index == len(files):
            print(f"  {index}/{len(files)}")
    return len(files)


def download_checkpoints(
    *,
    api: object,
    repo_id: str,
    run_id: str,
    token: str | None,
    out_dir: Path,
    iterations: set[int],
) -> list[Path]:
    hub_mod = __import__("huggingface_hub", fromlist=["hf_hub_download"])
    hf_hub_download = hub_mod.hf_hub_download
    files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", token=token))
    downloaded: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for iteration in sorted(iterations):
        filename = f"runs/{run_id}/model_iter_{iteration:03d}.pt"
        if filename not in files:
            print(f"Checkpoint not found: {Path(filename).name}")
            continue
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=str(out_dir),
        )
        downloaded.append(Path(local))
        print(f"Downloaded checkpoint: {Path(local).name}")
    return downloaded


def copy_flat_checkpoints(paths: list[Path], out_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for source in paths:
        target = out_dir / source.name
        if source.resolve() == target.resolve():
            copied.append(target)
            continue
        shutil.copy2(source, target)
        copied.append(target)
        print(f"Ready for play: {target}")
    return copied


def install_playable_checkpoints(
    paths: list[Path], *, run_id: str, checkpoint_dir: Path
) -> list[Path]:
    installed: list[Path] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        iter_part = source.stem.removeprefix("model_")
        target = checkpoint_dir / f"{run_id}_{iter_part}.pt"
        shutil.copy2(source, target)
        installed.append(target)
        print(f"Playable checkpoint: {target}")
    return installed


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    token = os.environ.get(str(args.token_env), "").strip() or None
    if token is None:
        print(f"Warning: {args.token_env} is not set; using anonymous HF access.")

    hub_mod = __import__("huggingface_hub", fromlist=["HfApi"])
    api = hub_mod.HfApi(token=token)
    out_dir = Path(args.out_dir or f"runs_history/{args.run_id}")

    metadata_paths = download_metadata_files(
        api=api,
        repo_id=str(args.repo_id),
        run_id=str(args.run_id),
        token=token,
        out_dir=out_dir,
    )
    if not metadata_paths:
        return 1

    rows = [flatten_metadata(path) for path in metadata_paths]
    write_csv(rows, out_dir / f"{args.run_id}_history.csv")
    summarize(rows)

    latest_iter = max(int(row["iter"]) for row in rows if row.get("iter") is not None)
    checkpoint_iters = {latest_iter, *args.checkpoint}
    best_iter = best_eval_iter(rows)
    if best_iter is not None:
        checkpoint_iters.add(best_iter)

    if not args.skip_buffers:
        count = download_buffers(
            api=api,
            repo_id=str(args.repo_id),
            run_id=str(args.run_id),
            token=token,
            out_dir=out_dir / "buffers",
        )
        print(f"Downloaded {count} buffer(s).")

    downloaded = download_checkpoints(
        api=api,
        repo_id=str(args.repo_id),
        run_id=str(args.run_id),
        token=token,
        out_dir=out_dir / "checkpoints",
        iterations=checkpoint_iters,
    )
    copy_flat_checkpoints(downloaded, out_dir)
    install_playable_checkpoints(
        downloaded,
        run_id=str(args.run_id),
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    print(f"Downloaded {len(downloaded)} checkpoint(s).")
    print(f"Run data ready at {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

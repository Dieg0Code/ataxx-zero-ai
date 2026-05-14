from __future__ import annotations

import argparse
import os
from pathlib import Path


def select_buffer_files(files: list[str], *, run_id: str) -> list[str]:
    prefix = f"runs/{run_id.strip()}/"
    selected = [
        name
        for name in files
        if name.startswith(prefix)
        and Path(name).name.startswith("buffer_iter_")
        and name.endswith(".npz")
    ]
    return sorted(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HF buffer_iter_*.npz files.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(str(args.token_env), "").strip() or None
    hub_mod = __import__("huggingface_hub", fromlist=["HfApi", "hf_hub_download"])
    api = hub_mod.HfApi(token=token)
    hf_hub_download = hub_mod.hf_hub_download

    files = select_buffer_files(
        list(api.list_repo_files(repo_id=args.repo_id, repo_type="model")),
        run_id=str(args.run_id),
    )
    if not files:
        raise SystemExit(f"No buffers found for run_id={args.run_id}.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        local = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=str(output_dir),
        )
        print(local)
    print(f"Downloaded {len(files)} buffer(s) to {output_dir}.")


if __name__ == "__main__":
    main()


__all__ = ["select_buffer_files"]

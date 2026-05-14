from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse and watch Ataxx replays saved by the arena.")
    parser.add_argument(
        "--replay",
        default="",
        help="Path to a replay .npz or its .json sidecar.",
    )
    parser.add_argument(
        "--root",
        default="tournament_replays",
        help="Replay library root used when --replay is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_src_on_path()
    from ui.arena.replay_viewer import run_replay_library, run_replay_viewer

    args = _parse_args()
    if args.replay:
        run_replay_viewer(args.replay)
        return
    run_replay_library(args.root)


if __name__ == "__main__":
    main()

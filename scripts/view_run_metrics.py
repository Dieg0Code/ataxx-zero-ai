from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the Ataxx training run dashboard.")
    parser.add_argument("--root", default="runs_history")
    args = parser.parse_args()
    _ensure_src_on_path()
    from ui.arena.run_dashboard import run_metrics_dashboard

    run_metrics_dashboard(args.root)


if __name__ == "__main__":
    main()

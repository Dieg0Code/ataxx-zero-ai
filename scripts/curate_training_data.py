from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.curation import curate_npz_paths, save_curated_dataset  # noqa: E402


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        parsed_float = float(value)
        if not parsed_float.is_integer():
            raise argparse.ArgumentTypeError("must be a positive integer") from None
        parsed = int(parsed_float)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _glob_paths(pattern: str) -> list[Path]:
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        anchor = Path(pattern_path.anchor)
        relative = str(pattern_path.relative_to(anchor))
        return [path for path in anchor.glob(relative) if path.suffix == ".npz"]
    return [path for path in Path.cwd().glob(pattern) if path.suffix == ".npz"]


def _resolve_sources(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            paths.extend(sorted(candidate.rglob("*.npz")))
            continue
        matches = _glob_paths(pattern)
        if matches:
            paths.extend(sorted(path for path in matches if path.suffix == ".npz"))
            continue
        if candidate.suffix == ".npz":
            paths.append(candidate)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate Ataxx training NPZ datasets.")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--human-oversample", type=_positive_int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _resolve_sources(list(args.source))
    if not paths:
        raise SystemExit("No .npz sources found.")
    dataset = curate_npz_paths(
        paths,
        human_oversample=max(1, int(args.human_oversample)),
    )
    if len(dataset.values) == 0:
        raise SystemExit("Curated dataset is empty.")
    report_path = save_curated_dataset(dataset, Path(args.output))
    print(
        f"Curated {len(dataset.values)} examples from {len(paths)} source(s). "
        f"Report: {report_path}",
    )


if __name__ == "__main__":
    main()

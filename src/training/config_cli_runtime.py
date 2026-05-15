from __future__ import annotations

import argparse


def add_curated_pretrain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pretrain-dataset", default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)


def add_absolute_eval_gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--baseline-composite", type=float, default=None)
    parser.add_argument("--baseline-h2h-min-score", type=float, default=None)
    parser.add_argument("--eval-absolute-patience", type=int, default=None)
    parser.add_argument("--eval-absolute-delta", type=float, default=None)
    parser.add_argument("--eval-absolute-action", choices=["abort", "restore_best", "warn"], default=None)
    parser.add_argument("--eval-absolute-min-iteration", type=int, default=None)


def apply_curated_pretrain_overrides(
    *,
    args: argparse.Namespace,
    config: dict[str, int | float | bool | str],
) -> None:
    if args.pretrain_dataset is not None:
        config["pretrain_dataset_path"] = args.pretrain_dataset
    if args.pretrain_epochs is not None:
        config["pretrain_epochs"] = max(0, args.pretrain_epochs)


def apply_absolute_eval_gate_overrides(
    *,
    args: argparse.Namespace,
    config: dict[str, int | float | bool | str],
) -> None:
    updates = {
        "baseline_checkpoint": args.baseline_checkpoint.strip()
        if args.baseline_checkpoint is not None
        else None,
        "baseline_composite": args.baseline_composite,
        "baseline_h2h_min_score": args.baseline_h2h_min_score,
        "eval_absolute_patience": max(0, args.eval_absolute_patience)
        if args.eval_absolute_patience is not None
        else None,
        "eval_absolute_delta": max(0.0, args.eval_absolute_delta)
        if args.eval_absolute_delta is not None
        else None,
        "eval_absolute_action": args.eval_absolute_action,
        "eval_absolute_min_iteration": max(0, args.eval_absolute_min_iteration)
        if args.eval_absolute_min_iteration is not None
        else None,
    }
    for key, value in updates.items():
        if value is not None:
            config[key] = value


__all__ = [
    "add_absolute_eval_gate_args",
    "add_curated_pretrain_args",
    "apply_absolute_eval_gate_overrides",
    "apply_curated_pretrain_overrides",
]

"""Simula la transformación de Power Query sobre runs_history/all_runs.csv para
detectar errores de conversión de tipo ANTES de abrir Power BI Desktop.

Lee el CSV con las mismas columnas que define el M expression de all_runs.tmdl
y prueba que cada columna numérica/fecha convierte limpiamente (vacíos → null).

Uso:
    uv run python scripts/simulate_powerquery.py

Sale con código 1 si encuentra valores que rompen el type conversion.
"""
from __future__ import annotations

import csv
import sys
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "runs_history" / "all_runs.csv"

INT_COLS = [
    "iter", "is_final_iter", "replay_size",
    "eval_total_wins", "eval_total_losses", "eval_total_draws", "eval_total_games",
]
NUM_COLS = [
    "score", "best_eval_score",
    "eval_score_hard", "eval_score_apex", "eval_score_sentinel",
    "train_loss_total", "train_loss_value", "train_loss_policy",
    "train_value_mae", "train_policy_accuracy", "train_lr",
]
DATETIME_COLS = ["timestamp"]
DATE_COLS = ["date"]
TEXT_COLS = ["run_id", "version", "codename"]


def try_int(value: str) -> tuple[bool, str]:
    if value == "":
        return True, "null"
    try:
        int(float(value))
        return True, "int"
    except ValueError:
        return False, f"NO ES INT: '{value}'"


def try_number(value: str) -> tuple[bool, str]:
    if value == "":
        return True, "null"
    try:
        float(value)
        return True, "number"
    except ValueError:
        return False, f"NO ES NUMBER: '{value}'"


def try_datetime(value: str) -> tuple[bool, str]:
    if value == "":
        return True, "null"
    try:
        datetime.fromisoformat(value)
        return True, "datetime"
    except ValueError:
        return False, f"NO ES DATETIME: '{value}'"


def try_date(value: str) -> tuple[bool, str]:
    if value == "":
        return True, "null"
    try:
        date.fromisoformat(value)
        return True, "date"
    except ValueError:
        return False, f"NO ES DATE: '{value}'"


def main() -> int:
    if not CSV_PATH.is_file():
        print(f"[error] no existe {CSV_PATH}", file=sys.stderr)
        return 1

    with CSV_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    print(f"Leídas {len(rows)} filas de {CSV_PATH.name}")
    print(f"Encabezados: {list(rows[0].keys()) if rows else '(vacío)'}\n")

    summary: dict[str, dict[str, int]] = {}
    problems: list[str] = []

    for col_set, parser, label in [
        (INT_COLS, try_int, "int64"),
        (NUM_COLS, try_number, "number"),
        (DATETIME_COLS, try_datetime, "datetime"),
        (DATE_COLS, try_date, "date"),
    ]:
        for col in col_set:
            stats = {"ok": 0, "null": 0, "fail": 0}
            for idx, row in enumerate(rows, start=2):  # +2 for 1-index + header
                value = row.get(col, "")
                ok, kind = parser(value)
                if not ok:
                    stats["fail"] += 1
                    if len(problems) < 30:
                        problems.append(f"  fila {idx}, columna {col} ({label}): {kind}")
                elif kind == "null":
                    stats["null"] += 1
                else:
                    stats["ok"] += 1
            summary[col] = stats

    print(f"{'Columna':<25} {'Tipo':<10} {'OK':>5} {'NULL':>6} {'FALLA':>6}")
    print("-" * 60)
    for col_set, label in [(INT_COLS, "int64"), (NUM_COLS, "number"), (DATETIME_COLS, "datetime"), (DATE_COLS, "date")]:
        for col in col_set:
            s = summary[col]
            warning = "  ← BLOQUEA POWERBI" if s["fail"] > 0 else ""
            print(f"{col:<25} {label:<10} {s['ok']:>5} {s['null']:>6} {s['fail']:>6}{warning}")

    total_fail = sum(s["fail"] for s in summary.values())
    if total_fail == 0:
        print("\nOK — todas las celdas convierten limpiamente. Power BI debería cargar sin error.")
        return 0
    print(f"\n{total_fail} celdas con problema. Detalle:")
    for p in problems:
        print(p)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

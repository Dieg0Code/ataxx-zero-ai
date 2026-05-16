from __future__ import annotations

from pathlib import Path

from ui.arena.run_dashboard import (
    discover_run_histories,
    load_history_rows,
    metric_series,
)


def test_load_history_rows_and_metric_series(tmp_path: Path) -> None:
    csv_path = tmp_path / "policy_spatial_v10_history.csv"
    csv_path.write_text(
        "iter,score,train_loss_total\n"
        "1,,3.0\n"
        "2,0.25,2.5\n"
        "3,0.50,2.0\n",
        encoding="utf-8",
    )

    rows = load_history_rows(csv_path)

    assert metric_series(rows, "score") == [(2, 0.25), (3, 0.5)]
    assert metric_series(rows, "train_loss_total") == [(1, 3.0), (2, 2.5), (3, 2.0)]


def test_discover_run_histories_orders_by_latest_iter(tmp_path: Path) -> None:
    old_dir = tmp_path / "policy_spatial_v8"
    new_dir = tmp_path / "policy_spatial_v10"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "policy_spatial_v8_history.csv").write_text(
        "iter,score\n5,0.8\n",
        encoding="utf-8",
    )
    (new_dir / "policy_spatial_v10_history.csv").write_text(
        "iter,score\n1,\n12,0.2\n",
        encoding="utf-8",
    )

    items = discover_run_histories(tmp_path)

    assert [item.run_id for item in items] == ["policy_spatial_v10", "policy_spatial_v8"]
    assert items[0].latest_iter == 12
    assert items[0].latest_score == 0.2

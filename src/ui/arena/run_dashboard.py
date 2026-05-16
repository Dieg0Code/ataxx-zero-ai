from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pygame

from ui.arena.fonts import load_arena_fonts
from ui.arena.layout import WIN_H, WIN_W
from ui.arena.theme import (
    BG_BOTTOM,
    BG_TOP,
    HUD_EVAL_GRID,
    HUD_EVAL_LINE,
    HUD_HISTORY_TEXT,
    HUD_TITLE,
    HUD_VALUE_AXIS,
    HUD_VALUE_BAR_BG,
    PANEL_BG,
    PANEL_BORDER,
    PIECE_P1,
    PIECE_P2,
    TEXT_DIM,
    TEXT_MAIN,
)

_METRICS = (
    ("score", "Composite", "promedio contra hard/apex/sentinel; mas alto es mejor"),
    ("eval_score_hard", "Hard", "score contra heuristica hard; mas alto es mejor"),
    ("eval_score_apex", "Apex", "score contra heuristica apex; mas alto es mejor"),
    ("eval_score_sentinel", "Sentinel", "score contra sentinel; mas alto es mejor"),
    ("train_loss_total", "Loss", "error de entrenamiento; mas bajo es mejor"),
    ("train_policy_accuracy", "Policy acc", "porcentaje de jugadas imitadas exactamente"),
    ("replay_size", "Replay", "ejemplos acumulados en el buffer"),
)


@dataclass(frozen=True)
class RunHistoryItem:
    run_id: str
    path: Path
    rows: int
    latest_iter: int
    latest_score: float | None


def discover_run_histories(root: str | Path = "runs_history") -> list[RunHistoryItem]:
    root_path = Path(root)
    items: list[RunHistoryItem] = []
    for csv_path in sorted(root_path.glob("*/*_history.csv")):
        rows = load_history_rows(csv_path)
        if not rows:
            continue
        run_id = csv_path.parent.name
        latest = rows[-1]
        items.append(
            RunHistoryItem(
                run_id=run_id,
                path=csv_path,
                rows=len(rows),
                latest_iter=int(float(str(latest.get("iter", 0) or 0))),
                latest_score=_float_or_none(latest.get("score")),
            )
        )
    items.sort(key=lambda item: (item.latest_iter, item.run_id), reverse=True)
    return items


def load_history_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def metric_series(rows: list[dict[str, str]], metric: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for row in rows:
        value = _float_or_none(row.get(metric))
        iteration = _float_or_none(row.get("iter"))
        if value is None or iteration is None:
            continue
        points.append((int(iteration), value))
    return points


def run_metrics_dashboard(root: str | Path = "runs_history") -> None:
    pygame.init()
    try:
        font, small, _big = load_arena_fonts()
        window = pygame.display.set_mode(_scaled_window_size())
        pygame.display.set_caption("Ataxx run metrics")
        scene = pygame.Surface((WIN_W, WIN_H))
        clock = pygame.time.Clock()
        items = discover_run_histories(root)
        selected = 0
        metric_idx = 0
        rows: list[dict[str, str]] = load_history_rows(items[0].path) if items else []
        running = True
        pygame.key.set_repeat(280, 45)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        items = discover_run_histories(root)
                        selected = min(selected, max(0, len(items) - 1))
                        rows = load_history_rows(items[selected].path) if items else []
                    elif event.key == pygame.K_DOWN and items:
                        selected = min(len(items) - 1, selected + 1)
                        rows = load_history_rows(items[selected].path)
                    elif event.key == pygame.K_UP and items:
                        selected = max(0, selected - 1)
                        rows = load_history_rows(items[selected].path)
                    elif event.key in (pygame.K_TAB, pygame.K_RIGHT):
                        metric_idx = (metric_idx + 1) % len(_METRICS)
                    elif event.key == pygame.K_LEFT:
                        metric_idx = (metric_idx - 1) % len(_METRICS)
            _draw_dashboard(
                scene,
                root=Path(root),
                items=items,
                selected=selected,
                rows=rows,
                metric_idx=metric_idx,
                font=font,
                small=small,
            )
            _flip_scene(scene, window)
            clock.tick(60)
    finally:
        pygame.quit()


def _draw_dashboard(
    surf: pygame.Surface,
    *,
    root: Path,
    items: list[RunHistoryItem],
    selected: int,
    rows: list[dict[str, str]],
    metric_idx: int,
    font: pygame.font.Font,
    small: pygame.font.Font,
) -> None:
    _draw_background(surf)
    margin = 28
    surf.blit(font.render("RUN METRICS", True, HUD_TITLE), (margin, 18))
    root_text = small.render(f"{len(items)} runs | {root}", True, TEXT_DIM)
    surf.blit(root_text, (WIN_W - margin - root_text.get_width(), 24))
    pygame.draw.line(surf, PANEL_BORDER, (margin, 58), (WIN_W - margin, 58), width=1)
    list_rect = pygame.Rect(margin, 78, 360, WIN_H - 122)
    chart_rect = pygame.Rect(list_rect.right + 24, 104, WIN_W - list_rect.right - 52, 270)
    summary_rect = pygame.Rect(chart_rect.left, chart_rect.bottom + 14, chart_rect.width, 94)
    lower_rect = pygame.Rect(chart_rect.left, summary_rect.bottom + 14, chart_rect.width, 128)
    _draw_run_list(surf, list_rect, items=items, selected=selected, font=font, small=small)
    _draw_metric_tabs(surf, chart_rect.left, 74, metric_idx=metric_idx, small=small)
    metric, label, description = _METRICS[metric_idx]
    selected_item = items[selected] if items else None
    selected_series = metric_series(rows, metric)
    _draw_metric_summary(
        surf,
        summary_rect,
        item=selected_item,
        label=label,
        description=description,
        series=selected_series,
        font=font,
        small=small,
    )
    _draw_chart(
        surf,
        chart_rect,
        title=label,
        series=selected_series,
        font=font,
        small=small,
        color=HUD_EVAL_LINE,
    )
    _draw_chart(
        surf,
        lower_rect,
        title="Eval score por nivel   azul=hard   rojo=sentinel",
        series=metric_series(rows, "eval_score_hard"),
        font=font,
        small=small,
        color=PIECE_P2,
        second=metric_series(rows, "eval_score_sentinel"),
        second_color=PIECE_P1,
    )
    help_text = "up/down run | tab/right metrica | left metrica anterior | r recargar | q salir"
    surf.blit(small.render(help_text, True, TEXT_DIM), (margin, WIN_H - 34))


def _draw_run_list(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    items: list[RunHistoryItem],
    selected: int,
    font: pygame.font.Font,
    small: pygame.font.Font,
) -> None:
    pygame.draw.rect(surf, PANEL_BG, rect)
    pygame.draw.rect(surf, PANEL_BORDER, rect, width=1)
    if not items:
        surf.blit(small.render("No hay CSVs en runs_history.", True, TEXT_DIM), (rect.left + 14, rect.top + 14))
        return
    row_h = 54
    start = max(0, min(selected - 4, max(0, len(items) - 10)))
    for idx, item in enumerate(items[start : start + 10], start=start):
        y = rect.top + 10 + (idx - start) * row_h
        row = pygame.Rect(rect.left + 8, y, rect.width - 16, row_h - 6)
        pygame.draw.rect(surf, HUD_VALUE_BAR_BG if idx == selected else PANEL_BG, row)
        if idx == selected:
            pygame.draw.rect(surf, HUD_HISTORY_TEXT, row, width=2)
        name = _ellipsize(font, item.run_id, row.width - 24)
        surf.blit(font.render(name, True, TEXT_MAIN if idx == selected else HUD_TITLE), (row.left + 10, row.top + 2))
        score = "--" if item.latest_score is None else f"{item.latest_score:.3f}"
        meta = f"iter {item.latest_iter} | {item.rows} pts | score {score}"
        surf.blit(small.render(meta, True, TEXT_DIM), (row.left + 10, row.top + 28))


def _draw_metric_summary(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    item: RunHistoryItem | None,
    label: str,
    description: str,
    series: list[tuple[int, float]],
    font: pygame.font.Font,
    small: pygame.font.Font,
) -> None:
    pygame.draw.rect(surf, PANEL_BG, rect)
    pygame.draw.rect(surf, PANEL_BORDER, rect, width=1)
    run_id = item.run_id if item is not None else "sin run"
    surf.blit(font.render(run_id, True, HUD_TITLE), (rect.left + 14, rect.top + 8))
    surf.blit(small.render(f"{label}: {description}", True, TEXT_DIM), (rect.left + 14, rect.top + 36))
    if not series:
        surf.blit(small.render("Sin puntos para esta metrica.", True, HUD_HISTORY_TEXT), (rect.left + 14, rect.top + 62))
        return
    latest_it, latest = series[-1]
    best_it, best = _best_point(series, lower_is_better=label == "Loss")
    trend = _trend_text(series, lower_is_better=label == "Loss")
    summary = f"ultimo {latest:.4g} @ iter {latest_it} | mejor {best:.4g} @ iter {best_it} | {trend}"
    summary = _ellipsize(small, summary, rect.width - 28)
    surf.blit(small.render(summary, True, HUD_HISTORY_TEXT), (rect.left + 14, rect.top + 62))


def _draw_metric_tabs(surf: pygame.Surface, x: int, y: int, *, metric_idx: int, small: pygame.font.Font) -> None:
    cursor = x
    for idx, (_metric, label, _description) in enumerate(_METRICS):
        color = HUD_HISTORY_TEXT if idx == metric_idx else TEXT_DIM
        text = small.render(label, True, color)
        surf.blit(text, (cursor, y))
        if idx == metric_idx:
            pygame.draw.line(surf, color, (cursor, y + text.get_height()), (cursor + text.get_width(), y + text.get_height()), 2)
        cursor += text.get_width() + 22


def _draw_chart(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    title: str,
    series: list[tuple[int, float]],
    font: pygame.font.Font,
    small: pygame.font.Font,
    color: tuple[int, int, int],
    second: list[tuple[int, float]] | None = None,
    second_color: tuple[int, int, int] = PIECE_P1,
) -> None:
    pygame.draw.rect(surf, HUD_VALUE_BAR_BG, rect)
    pygame.draw.rect(surf, PANEL_BORDER, rect, width=1)
    surf.blit(font.render(title, True, HUD_TITLE), (rect.left + 14, rect.top + 10))
    axis_w = 62
    plot = pygame.Rect(rect.left + axis_w, rect.top + 58, rect.width - axis_w - 22, rect.height - 88)
    for i in range(5):
        y = plot.top + int(i * plot.height / 4)
        pygame.draw.line(surf, HUD_EVAL_GRID, (plot.left, y), (plot.right, y), width=1)
    all_values = [value for _it, value in series]
    if second:
        all_values.extend(value for _it, value in second)
    if len(all_values) < 2:
        msg = small.render("Sin datos suficientes.", True, TEXT_DIM)
        surf.blit(msg, (plot.centerx - msg.get_width() // 2, plot.centery))
        return
    min_v, max_v = min(all_values), max(all_values)
    if abs(max_v - min_v) < 1e-9:
        max_v = min_v + 1.0
    _draw_series(surf, plot, series=series, min_v=min_v, max_v=max_v, color=color)
    if second:
        _draw_series(surf, plot, series=second, min_v=min_v, max_v=max_v, color=second_color)
    max_label = small.render(f"{max_v:.3g}", True, HUD_VALUE_AXIS)
    min_label = small.render(f"{min_v:.3g}", True, HUD_VALUE_AXIS)
    surf.blit(max_label, (plot.left - max_label.get_width() - 10, plot.top - 8))
    surf.blit(min_label, (plot.left - min_label.get_width() - 10, plot.bottom - 12))
    last_it, last_value = series[-1]
    first_it = series[0][0]
    footer = _ellipsize(small, f"iter {first_it}-{last_it} | ultimo {last_value:.4g}", rect.width - 28)
    surf.blit(small.render(footer, True, TEXT_DIM), (rect.left + 14, rect.bottom - 28))


def _draw_series(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    series: list[tuple[int, float]],
    min_v: float,
    max_v: float,
    color: tuple[int, int, int],
) -> None:
    min_it, max_it = series[0][0], series[-1][0]
    span_it = max(1, max_it - min_it)
    points = [
        (
            int(rect.left + ((it - min_it) / span_it) * rect.width),
            int(rect.bottom - ((value - min_v) / (max_v - min_v)) * rect.height),
        )
        for it, value in series
    ]
    if len(points) >= 2:
        pygame.draw.lines(surf, color, False, points, width=2)
    pygame.draw.circle(surf, color, points[-1], 4)


def _draw_background(surf: pygame.Surface) -> None:
    for y in range(WIN_H):
        t = y / max(1, WIN_H - 1)
        color = tuple(int(BG_TOP[i] * (1.0 - t) + BG_BOTTOM[i] * t) for i in range(3))
        pygame.draw.line(surf, color, (0, y), (WIN_W, y))


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _best_point(series: list[tuple[int, float]], *, lower_is_better: bool) -> tuple[int, float]:
    return min(series, key=lambda point: point[1]) if lower_is_better else max(series, key=lambda point: point[1])


def _trend_text(series: list[tuple[int, float]], *, lower_is_better: bool) -> str:
    if len(series) < 2:
        return "tendencia sin datos"
    previous = series[-2][1]
    current = series[-1][1]
    delta = current - previous
    improved = delta < 0 if lower_is_better else delta > 0
    if abs(delta) < 1e-9:
        return "estable"
    direction = "mejora" if improved else "baja"
    return f"{direction} {delta:+.4g} vs punto anterior"


def _ellipsize(font: pygame.font.Font, text: str, max_width: int) -> str:
    if font.size(text)[0] <= max_width:
        return text
    trimmed = text
    while trimmed and font.size(trimmed + "...")[0] > max_width:
        trimmed = trimmed[:-1]
    return trimmed + "..."


def _scaled_window_size() -> tuple[int, int]:
    info = pygame.display.Info()
    scale = min(max(640, info.current_w - 60) / WIN_W, max(480, info.current_h - 160) / WIN_H, 1.0)
    return int(WIN_W * scale), int(WIN_H * scale)


def _flip_scene(scene: pygame.Surface, window: pygame.Surface) -> None:
    target = pygame.display.get_surface() or window
    tw, th = target.get_size()
    scale = min(tw / WIN_W, th / WIN_H)
    scaled = pygame.transform.smoothscale(scene, (int(WIN_W * scale), int(WIN_H * scale)))
    target.fill((0, 0, 0))
    target.blit(scaled, ((tw - scaled.get_width()) // 2, (th - scaled.get_height()) // 2))
    pygame.display.flip()


__all__ = [
    "RunHistoryItem",
    "discover_run_histories",
    "load_history_rows",
    "metric_series",
    "run_metrics_dashboard",
]

from __future__ import annotations

from typing import Any

import pygame

from game.actions import ACTION_SPACE
from game.types import Move
from ui.arena.theme import (
    HUD_EVAL_GRID,
    HUD_EVAL_LINE,
    HUD_HISTORY_TEXT,
    HUD_TITLE,
    HUD_VALUE_AXIS,
    HUD_VALUE_BAR_BG,
    HUD_VISITS_BAR,
    HUD_VISITS_BAR_DIM,
    PANEL_ACCENT,
    PANEL_BG,
    PANEL_BORDER,
    PIECE_P1,
    PIECE_P2,
    TEXT_DIM,
    TEXT_MAIN,
)

# Notation: cell (row,col) where row 0 is top → "a..g" file, "1..7" rank.
_FILES = "abcdefg"


def _cell_to_notation(row: int, col: int) -> str:
    if not (0 <= row < 7 and 0 <= col < 7):
        return "??"
    return f"{_FILES[col]}{7 - row}"


def move_notation(move: Move | None) -> str:
    if move is None:
        return "pass"
    sr, sc, tr, tc = move
    src = _cell_to_notation(sr, sc)
    dst = _cell_to_notation(tr, tc)
    # Two-square move = jump (>), one-square = clone (+).
    is_jump = max(abs(tr - sr), abs(tc - sc)) > 1
    sep = ">" if is_jump else "+"
    return f"{src}{sep}{dst}"


def _draw_panel_frame(surf: pygame.Surface, rect: pygame.Rect, title: str, font_title: pygame.font.Font) -> pygame.Rect:
    """Draw the section background + border + title; return inner content rect."""
    pygame.draw.rect(surf, PANEL_BG, rect)
    pygame.draw.rect(surf, PANEL_BORDER, rect, width=2)
    title_surface = font_title.render(title, True, HUD_TITLE)
    surf.blit(title_surface, (rect.left + 14, rect.top + 10))
    pygame.draw.line(
        surf,
        PANEL_BORDER,
        (rect.left + 14, rect.top + 10 + title_surface.get_height() + 4),
        (rect.right - 14, rect.top + 10 + title_surface.get_height() + 4),
        width=1,
    )
    inner_top = rect.top + 10 + title_surface.get_height() + 12
    return pygame.Rect(rect.left + 14, inner_top, rect.width - 28, rect.bottom - inner_top - 14)


def draw_top_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    p1_name: str,
    p2_name: str,
    p1_count: int,
    p2_count: int,
    turn_player: int,
    turn_index: int,
    mode: str,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
) -> None:
    inner = _draw_panel_frame(surf, rect, "ATAXX ARENA", font_title)
    line_h = font_body.get_linesize()
    y = inner.top
    rows = [
        (f"Mode    : {mode}", TEXT_MAIN),
        (f"Turn #  : {turn_index}", TEXT_MAIN),
        (
            f"To move : {'P1 (RED)' if turn_player == 1 else 'P2 (BLUE)'}",
            PIECE_P1 if turn_player == 1 else PIECE_P2,
        ),
    ]
    for text, color in rows:
        surf.blit(font_body.render(text, True, color), (inner.left, y))
        y += line_h
    y += 6
    # Score row
    score_text = f"P1 {p1_name}: {p1_count:>2}    P2 {p2_name}: {p2_count:>2}"
    surf.blit(font_body.render(score_text, True, TEXT_DIM), (inner.left, y))


def draw_top_moves_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    top_moves: list[tuple[int, int, float, float]],
    root_value: float,
    thinker_player: int | None,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
) -> None:
    inner = _draw_panel_frame(surf, rect, "MCTS THINKING", font_title)
    line_h = font_body.get_linesize()

    if not top_moves:
        msg = "(no model agent moved yet)"
        surf.blit(font_body.render(msg, True, TEXT_DIM), (inner.left, inner.top))
        return

    total_visits = sum(visits for _, visits, _, _ in top_moves) or 1
    bar_color = PIECE_P1 if thinker_player == 1 else PIECE_P2

    # Top moves rows: notation | visit-bar | percent
    y = inner.top
    label_w = 80
    right_w = 110
    bar_left = inner.left + label_w
    bar_right = inner.right - right_w
    bar_w = bar_right - bar_left
    for action_idx, visits, _value, _prior in top_moves:
        move = ACTION_SPACE.decode(action_idx)
        notation = move_notation(move)
        pct = visits / total_visits

        # Notation
        surf.blit(font_body.render(notation, True, TEXT_MAIN), (inner.left, y))

        # Bar
        bar_rect = pygame.Rect(bar_left, y + 4, bar_w, line_h - 8)
        pygame.draw.rect(surf, HUD_VISITS_BAR_DIM, bar_rect)
        fill = pygame.Rect(bar_rect.left, bar_rect.top, int(bar_w * pct), bar_rect.height)
        pygame.draw.rect(surf, bar_color, fill)
        pygame.draw.rect(surf, HUD_VISITS_BAR, bar_rect, width=1)

        # Percent (value is already shown in the WIN PROB bar below)
        right_text = f"{int(pct * 100):>3}% ({visits})"
        right_surf = font_body.render(right_text, True, TEXT_DIM)
        surf.blit(right_surf, (bar_right + 6, y))

        y += line_h + 2

    y += 8
    # Value bar (-1..+1) representing root value for the player who just thought.
    surf.blit(font_body.render("WIN PROB", True, TEXT_MAIN), (inner.left, y))
    y += line_h
    bar_h = 14
    val_rect = pygame.Rect(inner.left, y, inner.width, bar_h)
    pygame.draw.rect(surf, HUD_VALUE_BAR_BG, val_rect)
    pygame.draw.rect(surf, PANEL_BORDER, val_rect, width=1)
    # Center axis
    cx = val_rect.centerx
    pygame.draw.line(surf, HUD_VALUE_AXIS, (cx, val_rect.top), (cx, val_rect.bottom), width=1)
    # Marker
    clamped = max(-1.0, min(1.0, float(root_value)))
    marker_x = int(cx + clamped * (val_rect.width // 2 - 2))
    marker_color = bar_color
    pygame.draw.rect(
        surf,
        marker_color,
        pygame.Rect(marker_x - 3, val_rect.top - 2, 6, val_rect.height + 4),
    )
    y += bar_h + 4
    surf.blit(
        font_body.render(f"value = {clamped:+.3f}", True, TEXT_DIM),
        (inner.left, y),
    )


def draw_history_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    move_history: list[tuple[int, Move | None]],
    eval_history: list[float],
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
) -> None:
    inner = _draw_panel_frame(surf, rect, "HISTORY / EVAL", font_title)

    # Split inner into left (move list) and right (eval graph) halves.
    half_w = inner.width // 2 - 6
    list_rect = pygame.Rect(inner.left, inner.top, half_w, inner.height)
    graph_rect = pygame.Rect(inner.right - half_w, inner.top, half_w, inner.height)

    line_h = font_body.get_linesize()
    visible = move_history[-8:]
    start_index = len(move_history) - len(visible)
    y = list_rect.top
    for offset, (player, move) in enumerate(visible):
        prefix = "P1" if player == 1 else "P2"
        text = f"{start_index + offset + 1:>2}. {prefix} {move_notation(move)}"
        color = PIECE_P1 if player == 1 else PIECE_P2
        surf.blit(font_body.render(text, True, color), (list_rect.left, y))
        y += line_h
    if not visible:
        surf.blit(font_body.render("(no moves)", True, TEXT_DIM), (list_rect.left, y))

    # Eval graph
    pygame.draw.rect(surf, HUD_VALUE_BAR_BG, graph_rect)
    pygame.draw.rect(surf, PANEL_BORDER, graph_rect, width=1)
    # Horizontal mid line
    mid_y = graph_rect.centery
    pygame.draw.line(surf, HUD_EVAL_GRID, (graph_rect.left + 1, mid_y), (graph_rect.right - 1, mid_y), width=1)

    if len(eval_history) >= 2:
        n = len(eval_history)
        x_step = max(1, (graph_rect.width - 4) / max(1, n - 1))
        points: list[tuple[int, int]] = []
        for i, value in enumerate(eval_history):
            clamped = max(-1.0, min(1.0, float(value)))
            x = int(graph_rect.left + 2 + i * x_step)
            y = int(mid_y - clamped * (graph_rect.height // 2 - 4))
            points.append((x, y))
        if len(points) >= 2:
            pygame.draw.lines(surf, HUD_EVAL_LINE, False, points, width=2)
        # last point dot in accent color
        last_x, last_y = points[-1]
        pygame.draw.circle(surf, PANEL_ACCENT, (last_x, last_y), 3)
    else:
        text = "eval needs >= 2 turns"
        text_surf = font_body.render(text, True, TEXT_DIM)
        surf.blit(
            text_surf,
            (graph_rect.centerx - text_surf.get_width() // 2, mid_y - line_h // 2),
        )

    # Note: silence ruff "unused" for HUD_HISTORY_TEXT until we wire amber later.
    _ = HUD_HISTORY_TEXT


def draw_hud(
    surf: pygame.Surface,
    *,
    arena_state: dict[str, Any],
    p1_name: str,
    p2_name: str,
    p1_count: int,
    p2_count: int,
    turn_player: int,
    turn_index: int,
    mode: str,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
) -> None:
    """Render the three HUD panels on the right side of the window."""
    from ui.arena.layout import hud_bottom_rect, hud_mid_rect, hud_top_rect

    draw_top_panel(
        surf,
        hud_top_rect(),
        p1_name=p1_name,
        p2_name=p2_name,
        p1_count=p1_count,
        p2_count=p2_count,
        turn_player=turn_player,
        turn_index=turn_index,
        mode=mode,
        font_title=font_title,
        font_body=font_body,
    )
    top_moves = arena_state.get("last_top_moves", [])
    if not isinstance(top_moves, list):
        top_moves = []
    root_value = float(arena_state.get("last_root_value", 0.0))
    thinker = arena_state.get("last_thinker")
    thinker_player = int(thinker) if isinstance(thinker, int) else None
    draw_top_moves_panel(
        surf,
        hud_mid_rect(),
        top_moves=top_moves,
        root_value=root_value,
        thinker_player=thinker_player,
        font_title=font_title,
        font_body=font_body,
    )
    move_history = arena_state.get("move_history", [])
    eval_history = arena_state.get("eval_history", [])
    if not isinstance(move_history, list):
        move_history = []
    if not isinstance(eval_history, list):
        eval_history = []
    draw_history_panel(
        surf,
        hud_bottom_rect(),
        move_history=move_history,
        eval_history=eval_history,
        font_title=font_title,
        font_body=font_body,
    )

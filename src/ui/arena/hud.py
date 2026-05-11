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
    speed_mult: float = 1.0,
    paused: bool = False,
    stats_record: dict[str, int] | None = None,
) -> None:
    inner = _draw_panel_frame(surf, rect, "PARTIDA", font_title)
    line_h = font_body.get_linesize()
    y = inner.top
    mode_es = {"play": "humano vs IA", "spectate": "IA vs IA"}.get(mode, mode)
    color_turn = PIECE_P1 if turn_player == 1 else PIECE_P2
    name_turn = "ROJO (P1)" if turn_player == 1 else "AZUL (P2)"
    rows = [
        (f"Modo    : {mode_es}", TEXT_MAIN),
        (f"Turno   : {turn_index}", TEXT_MAIN),
        (f"Mueve   : {name_turn}", color_turn),
    ]
    for text, color in rows:
        surf.blit(font_body.render(text, True, color), (inner.left, y))
        y += line_h
    y += 4
    score_text = f"Piezas  : P1 {p1_count:>2}   -   P2 {p2_count:>2}"
    surf.blit(font_body.render(score_text, True, TEXT_DIM), (inner.left, y))
    y += line_h
    if stats_record is not None:
        rec = stats_record
        wld = f"Récord  : G {rec.get('w', 0)}  P {rec.get('l', 0)}  E {rec.get('d', 0)}"
        surf.blit(font_body.render(wld, True, HUD_HISTORY_TEXT), (inner.left, y))
    speed_str = f"{speed_mult:g}x"
    state_str = f"PAUSA  vel {speed_str}" if paused else f"vel {speed_str}"
    surf.blit(
        font_body.render(state_str, True, PIECE_P1 if paused else TEXT_DIM),
        (rect.right - 14 - font_body.size(state_str)[0], rect.top + 12),
    )


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
    inner = _draw_panel_frame(surf, rect, "ANÁLISIS DE LA IA", font_title)
    line_h = font_body.get_linesize()
    bar_color = PIECE_P1 if thinker_player == 1 else PIECE_P2
    thinker_name = "ROJO" if thinker_player == 1 else ("AZUL" if thinker_player == -1 else "—")

    # Header row explains what the bars mean.
    header = "Jugadas más exploradas"
    surf.blit(font_body.render(header, True, TEXT_DIM), (inner.left, inner.top))
    y = inner.top + line_h + 2

    if not top_moves:
        surf.blit(
            font_body.render("(esperando que piense la IA…)", True, TEXT_DIM),
            (inner.left, y),
        )
        return

    total_visits = sum(visits for _, visits, _, _ in top_moves) or 1
    label_w = 90
    right_w = 130
    bar_left = inner.left + label_w
    bar_right = inner.right - right_w
    bar_w = bar_right - bar_left
    for action_idx, visits, _value, _prior in top_moves:
        move = ACTION_SPACE.decode(action_idx)
        notation = move_notation(move)
        pct = visits / total_visits

        surf.blit(font_body.render(notation, True, TEXT_MAIN), (inner.left, y))

        bar_rect = pygame.Rect(bar_left, y + 4, bar_w, line_h - 8)
        pygame.draw.rect(surf, HUD_VISITS_BAR_DIM, bar_rect)
        fill = pygame.Rect(bar_rect.left, bar_rect.top, int(bar_w * pct), bar_rect.height)
        pygame.draw.rect(surf, bar_color, fill)
        pygame.draw.rect(surf, HUD_VISITS_BAR, bar_rect, width=1)

        right_text = f"{int(pct * 100):>3}% ({visits} sim)"
        right_surf = font_body.render(right_text, True, TEXT_DIM)
        surf.blit(right_surf, (bar_right + 6, y))

        y += line_h + 2

    y += 10
    # Win probability is stored already normalized to P1 (ROJO) perspective.
    # +1.0 means ROJO is winning, -1.0 means AZUL is winning.
    clamped_p1 = max(-1.0, min(1.0, float(root_value)))
    win_pct_p1 = (clamped_p1 + 1.0) * 50.0
    if win_pct_p1 >= 60:
        prob_color = PIECE_P1
    elif win_pct_p1 <= 40:
        prob_color = PIECE_P2
    else:
        prob_color = HUD_HISTORY_TEXT  # zona neutra → ámbar

    surf.blit(
        font_body.render("Probabilidad de ganar (ROJO):", True, TEXT_MAIN),
        (inner.left, y),
    )
    y += line_h
    big = font_title.render(f"{win_pct_p1:5.1f}%", True, prob_color)
    surf.blit(big, (inner.left, y))
    # Side note: current thinker + raw value, kept tiny for science cred.
    note = f"  según {thinker_name}   value = {clamped_p1:+.3f}"
    note_surf = font_body.render(note, True, TEXT_DIM)
    surf.blit(
        note_surf,
        (inner.left + big.get_width() + 4, y + (big.get_height() - note_surf.get_height()) // 2),
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
    inner = _draw_panel_frame(surf, rect, "HISTORIAL / EVALUACIÓN", font_title)

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

    speed_mult_val = float(arena_state.get("speed_mult", 1.0))
    paused_val = bool(arena_state.get("paused", False))
    stats_rec = arena_state.get("stats_record")
    if not isinstance(stats_rec, dict):
        stats_rec = None
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
        speed_mult=speed_mult_val,
        paused=paused_val,
        stats_record=stats_rec,
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

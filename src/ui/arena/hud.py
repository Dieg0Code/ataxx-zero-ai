from __future__ import annotations

import math
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

_FILES = "abcdefg"
_GOOD_COLOR = HUD_VISITS_BAR
_BAD_COLOR = PIECE_P1
_NEUTRAL_COLOR = HUD_HISTORY_TEXT


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
    is_jump = max(abs(tr - sr), abs(tc - sc)) > 1
    sep = ">" if is_jump else "+"
    return f"{src}{sep}{dst}"


def _side_name(player: int | None) -> str:
    if player == 1:
        return "ROJO"
    if player == -1:
        return "AZUL"
    return "--"


def _select_perspective(
    *,
    p1_agent: str,
    p2_agent: str,
    turn_player: int,
    thinker_player: int | None,
) -> tuple[str, int]:
    ai_sides = [
        player
        for player, agent in ((1, p1_agent), (-1, p2_agent))
        if agent != "human"
    ]
    if len(ai_sides) == 1:
        return "IA", ai_sides[0]
    if len(ai_sides) > 1:
        return "TURNO", turn_player
    if thinker_player in (1, -1):
        return "ANALISIS", int(thinker_player)
    return "TURNO", turn_player


def _value_for_side(root_value_p1: float, side: int) -> float:
    clamped_p1 = max(-1.0, min(1.0, float(root_value_p1)))
    return clamped_p1 if side == 1 else -clamped_p1


def _pct_from_value(value: float) -> float:
    return (max(-1.0, min(1.0, float(value))) + 1.0) * 50.0


def _evaluation_color(win_pct: float) -> tuple[int, int, int]:
    if win_pct >= 60.0:
        return _GOOD_COLOR
    if win_pct <= 40.0:
        return _BAD_COLOR
    return _NEUTRAL_COLOR


def _position_label(win_pct: float) -> str:
    if win_pct >= 65.0:
        return "Favorable"
    if win_pct <= 35.0:
        return "En riesgo"
    return "Pareja"


def _is_forced_pass(top_moves: list[tuple[int, int, float, float]]) -> bool:
    return len(top_moves) == 1 and ACTION_SPACE.decode(top_moves[0][0]) is None


def _search_metrics(top_moves: list[tuple[int, int, float, float]]) -> tuple[str, int, str]:
    visits = [max(0, visits) for _, visits, _, _ in top_moves]
    total = sum(visits)
    if total <= 0:
        return "Baja", 0, "Alta"
    shares = [v / total for v in visits if v > 0]
    top = shares[0] if shares else 0.0
    second = shares[1] if len(shares) > 1 else 0.0
    gap_pts = round((top - second) * 100)
    if len(shares) <= 1:
        entropy = 0.0
    else:
        entropy = -sum(p * math.log(p) for p in shares) / math.log(len(shares))
    if top >= 0.45 and gap_pts >= 15:
        confidence = "Alta"
    elif top >= 0.35 or gap_pts >= 8:
        confidence = "Media"
    else:
        confidence = "Baja"
    uncertainty = "Alta" if entropy >= 0.85 else ("Media" if entropy >= 0.55 else "Baja")
    return confidence, gap_pts, uncertainty


def _draw_panel_frame(
    surf: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    font_title: pygame.font.Font,
) -> pygame.Rect:
    pygame.draw.rect(surf, PANEL_BG, rect)
    pygame.draw.rect(surf, PANEL_BORDER, rect, width=2)
    title_surface = font_title.render(title, True, HUD_TITLE)
    surf.blit(title_surface, (rect.left + 14, rect.top + 10))
    line_y = rect.top + 10 + title_surface.get_height() + 4
    pygame.draw.line(
        surf,
        PANEL_BORDER,
        (rect.left + 14, line_y),
        (rect.right - 14, line_y),
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
    tournament_banner: tuple[str, str] | None = None,
) -> None:
    title = tournament_banner[0] if tournament_banner else "PARTIDA"
    inner = _draw_panel_frame(surf, rect, title, font_title)
    line_h = font_body.get_linesize()
    y = inner.top
    if tournament_banner is not None:
        subtitle_surf = font_body.render(tournament_banner[1], True, PANEL_ACCENT)
        surf.blit(subtitle_surf, (inner.left, y))
        y += line_h + 2
    mode_es = {"play": "humano vs IA", "spectate": "IA vs IA"}.get(mode, mode)
    color_turn = PIECE_P1 if turn_player == 1 else PIECE_P2
    name_turn = "ROJO (P1)" if turn_player == 1 else "AZUL (P2)"
    rows = [
        (f"Modo    : {mode_es}", TEXT_MAIN),
        (f"Lados   : R {p1_name}  |  A {p2_name}", TEXT_DIM),
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
        wld = f"Record  : G {rec.get('w', 0)}  P {rec.get('l', 0)}  E {rec.get('d', 0)}"
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
    p1_agent: str,
    p2_agent: str,
    turn_player: int,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
) -> None:
    inner = _draw_panel_frame(surf, rect, "ANALISIS DE LA IA", font_title)
    line_h = font_body.get_linesize()
    bar_color = PIECE_P1 if thinker_player == 1 else PIECE_P2
    perspective_label, perspective_player = _select_perspective(
        p1_agent=p1_agent,
        p2_agent=p2_agent,
        turn_player=turn_player,
        thinker_player=thinker_player,
    )
    clamped_p1 = max(-1.0, min(1.0, float(root_value)))
    win_pct_p1 = (clamped_p1 + 1.0) * 50.0
    win_pct_perspective = _pct_from_value(_value_for_side(clamped_p1, perspective_player))
    prob_color = _evaluation_color(win_pct_perspective)

    y = inner.top
    side_text = f"{perspective_label} juega {_side_name(perspective_player)}"
    surf.blit(font_body.render(side_text, True, TEXT_DIM), (inner.left, y))
    y += line_h
    big = font_title.render(f"{win_pct_perspective:5.1f}%", True, prob_color)
    surf.blit(big, (inner.left, y))
    status = font_body.render(_position_label(win_pct_perspective), True, prob_color)
    surf.blit(status, (inner.left + big.get_width() + 12, y + 8))
    y += big.get_height() + 4

    raw = f"Modelo: ROJO {win_pct_p1:4.1f}%  |  AZUL {100.0 - win_pct_p1:4.1f}%"
    surf.blit(font_body.render(raw, True, TEXT_DIM), (inner.left, y))
    y += line_h + 10

    if not top_moves:
        waiting = font_body.render("Esperando analisis...", True, TEXT_DIM)
        surf.blit(waiting, (inner.left, y))
        return

    total_visits = sum(visits for _, visits, _, _ in top_moves) or 1
    if _is_forced_pass(top_moves):
        forced = font_body.render("Sin jugadas legales", True, HUD_HISTORY_TEXT)
        surf.blit(forced, (inner.left, y))
        y += line_h
        note = "La IA debe pasar; no hay decision que comparar."
        surf.blit(font_body.render(note, True, TEXT_DIM), (inner.left, y))
        y += line_h + 6
        sims = f"Busqueda: {total_visits} sims en estado forzado"
        surf.blit(font_body.render(sims, True, TEXT_DIM), (inner.left, y))
        return

    surf.blit(font_body.render("Candidatas MCTS", True, TEXT_DIM), (inner.left, y))
    y += line_h + 4

    label_w = 88
    right_w = 124
    bar_left = inner.left + label_w
    bar_right = inner.right - right_w
    bar_w = bar_right - bar_left
    for action_idx, visits, _value, _prior in top_moves[:3]:
        move = ACTION_SPACE.decode(action_idx)
        notation = move_notation(move)
        pct = visits / total_visits

        surf.blit(font_body.render(notation, True, TEXT_MAIN), (inner.left, y))
        bar_rect = pygame.Rect(bar_left, y + 5, bar_w, max(5, line_h - 10))
        pygame.draw.rect(surf, HUD_VISITS_BAR_DIM, bar_rect)
        fill = pygame.Rect(bar_rect.left, bar_rect.top, int(bar_w * pct), bar_rect.height)
        pygame.draw.rect(surf, bar_color, fill)
        pygame.draw.rect(surf, HUD_VISITS_BAR, bar_rect, width=1)

        right_text = f"{int(pct * 100):>3}%  {visits} sim"
        right_surf = font_body.render(right_text, True, TEXT_DIM)
        surf.blit(right_surf, (bar_right + 6, y))
        y += line_h + 4

    y += 4
    confidence, _gap_pts, uncertainty = _search_metrics(top_moves)
    summary = f"Decision: {confidence}  |  Incertidumbre: {uncertainty}  |  Sims: {total_visits}"
    surf.blit(font_body.render(summary, True, HUD_HISTORY_TEXT), (inner.left, y))


def draw_history_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    move_history: list[tuple[int, Move | None]],
    eval_history: list[float],
    perspective_player: int,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
) -> None:
    inner = _draw_panel_frame(surf, rect, "HISTORIAL / EVALUACION", font_title)
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

    pygame.draw.rect(surf, HUD_VALUE_BAR_BG, graph_rect)
    pygame.draw.rect(surf, PANEL_BORDER, graph_rect, width=1)
    graph_label = f"ventaja {_side_name(perspective_player)}"
    label_surf = font_body.render(graph_label, True, HUD_VALUE_AXIS)
    surf.blit(label_surf, (graph_rect.left + 6, graph_rect.top + 4))
    mid_y = graph_rect.centery
    pygame.draw.line(
        surf,
        HUD_EVAL_GRID,
        (graph_rect.left + 1, mid_y),
        (graph_rect.right - 1, mid_y),
        width=1,
    )

    if len(eval_history) >= 2:
        n = len(eval_history)
        x_step = max(1, (graph_rect.width - 4) / max(1, n - 1))
        points: list[tuple[int, int]] = []
        for i, value in enumerate(eval_history):
            clamped = _value_for_side(float(value), perspective_player)
            x = int(graph_rect.left + 2 + i * x_step)
            y = int(mid_y - clamped * (graph_rect.height // 2 - 4))
            points.append((x, y))
        if len(points) >= 2:
            pygame.draw.lines(surf, HUD_EVAL_LINE, False, points, width=2)
        last_x, last_y = points[-1]
        pygame.draw.circle(surf, PANEL_ACCENT, (last_x, last_y), 3)
    else:
        text_surf = font_body.render("eval needs >= 2 turns", True, TEXT_DIM)
        surf.blit(
            text_surf,
            (graph_rect.centerx - text_surf.get_width() // 2, mid_y - line_h // 2),
        )


def draw_hud(
    surf: pygame.Surface,
    *,
    arena_state: dict[str, Any],
    p1_name: str,
    p2_name: str,
    p1_agent: str,
    p2_agent: str,
    p1_count: int,
    p2_count: int,
    turn_player: int,
    turn_index: int,
    mode: str,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
) -> None:
    from ui.arena.layout import hud_bottom_rect, hud_mid_rect, hud_top_rect

    speed_mult_val = float(arena_state.get("speed_mult", 1.0))
    paused_val = bool(arena_state.get("paused", False))
    stats_rec = arena_state.get("stats_record")
    if not isinstance(stats_rec, dict):
        stats_rec = None
    banner_raw = arena_state.get("tournament_banner")
    if (
        isinstance(banner_raw, tuple)
        and len(banner_raw) == 2
        and all(isinstance(s, str) for s in banner_raw)
    ):
        banner = banner_raw
    else:
        banner = None

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
        tournament_banner=banner,
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
        p1_agent=p1_agent,
        p2_agent=p2_agent,
        turn_player=turn_player,
        font_title=font_title,
        font_body=font_body,
    )

    move_history = arena_state.get("move_history", [])
    eval_history = arena_state.get("eval_history", [])
    if not isinstance(move_history, list):
        move_history = []
    if not isinstance(eval_history, list):
        eval_history = []
    _, perspective_player = _select_perspective(
        p1_agent=p1_agent,
        p2_agent=p2_agent,
        turn_player=turn_player,
        thinker_player=thinker_player,
    )
    draw_history_panel(
        surf,
        hud_bottom_rect(),
        move_history=move_history,
        eval_history=eval_history,
        perspective_player=perspective_player,
        font_title=font_title,
        font_body=font_body,
    )

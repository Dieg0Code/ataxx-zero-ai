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
_BRAIN_TABS = ("RESUMEN", "RED", "MCTS")


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


def _normalize_brain_tab(value: object) -> int:
    if isinstance(value, int):
        return value % len(_BRAIN_TABS)
    return 0


def _draw_activation_heatmap(
    surf: pygame.Surface,
    rect: pygame.Rect,
    heatmap: object,
) -> None:
    pygame.draw.rect(surf, HUD_VALUE_BAR_BG, rect)
    pygame.draw.rect(surf, PANEL_BORDER, rect, width=1)
    if not hasattr(heatmap, "shape"):
        return
    cell = max(2, min(rect.width, rect.height) // 7)
    offset_x = rect.left + (rect.width - (cell * 7)) // 2
    offset_y = rect.top + (rect.height - (cell * 7)) // 2
    for row in range(7):
        for col in range(7):
            value = float(heatmap[row, col])  # type: ignore[index]
            value = max(0.0, min(1.0, value))
            color = (
                int(20 + (HUD_VISITS_BAR[0] - 20) * value),
                int(28 + (HUD_VISITS_BAR[1] - 28) * value),
                int(46 + (HUD_VISITS_BAR[2] - 46) * value),
            )
            square = pygame.Rect(offset_x + col * cell, offset_y + row * cell, cell - 1, cell - 1)
            pygame.draw.rect(surf, color, square)


def _brain_top_line(brain: dict[str, Any]) -> str:
    policy_top = brain.get("policy_top")
    if not isinstance(policy_top, list) or len(policy_top) == 0:
        return "Intuicion: --"
    first = policy_top[0]
    if not isinstance(first, tuple) or len(first) != 2:
        return "Intuicion: --"
    action_idx, prob = first
    if not isinstance(action_idx, int):
        return "Intuicion: --"
    move = move_notation(ACTION_SPACE.decode(action_idx))
    return f"Red: {move} {float(prob) * 100:4.1f}%"


def _mcts_top_line(top_moves: list[tuple[int, int, float, float]]) -> str:
    if not top_moves:
        return "MCTS: --"
    action_idx, visits, _value, _prior = top_moves[0]
    total = sum(max(0, item[1]) for item in top_moves) or 1
    return f"MCTS: {move_notation(ACTION_SPACE.decode(action_idx))} {visits / total * 100:4.1f}%"


def _draw_compact_mcts_rows(
    surf: pygame.Surface,
    *,
    top_moves: list[tuple[int, int, float, float]],
    total_visits: int,
    rect: pygame.Rect,
    font_body: pygame.font.Font,
    bar_color: tuple[int, int, int],
) -> None:
    row_h = max(16, font_body.get_linesize() - 2)
    label_w = 62
    pct_w = 46
    bar_x = rect.left + label_w
    bar_w = max(40, rect.width - label_w - pct_w)
    for idx, (action_idx, visits, _value, _prior) in enumerate(top_moves[:3]):
        y = rect.top + idx * row_h
        pct = visits / max(1, total_visits)
        move = move_notation(ACTION_SPACE.decode(action_idx))
        surf.blit(font_body.render(move, True, TEXT_MAIN), (rect.left, y))
        bar_rect = pygame.Rect(bar_x, y + 5, bar_w, 7)
        pygame.draw.rect(surf, HUD_VISITS_BAR_DIM, bar_rect)
        fill = pygame.Rect(bar_rect.left, bar_rect.top, int(bar_rect.width * pct), bar_rect.height)
        pygame.draw.rect(surf, bar_color, fill)
        pct_text = f"{pct * 100:>4.0f}%"
        surf.blit(font_body.render(pct_text, True, TEXT_DIM), (bar_rect.right + 6, y))


def _draw_policy_rows(
    surf: pygame.Surface,
    *,
    policy_top: object,
    rect: pygame.Rect,
    font_body: pygame.font.Font,
) -> None:
    if not isinstance(policy_top, list) or len(policy_top) == 0:
        surf.blit(font_body.render("Sin policy disponible", True, TEXT_DIM), rect.topleft)
        return
    row_h = max(16, font_body.get_linesize() - 2)
    label_w = 62
    pct_w = 52
    bar_x = rect.left + label_w
    bar_w = max(40, rect.width - label_w - pct_w)
    for idx, item in enumerate(policy_top[:3]):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        action_idx, prob = item
        if not isinstance(action_idx, int):
            continue
        y = rect.top + idx * row_h
        pct = max(0.0, min(1.0, float(prob)))
        move = move_notation(ACTION_SPACE.decode(action_idx))
        surf.blit(font_body.render(move, True, TEXT_MAIN), (rect.left, y))
        bar_rect = pygame.Rect(bar_x, y + 5, bar_w, 7)
        pygame.draw.rect(surf, HUD_VISITS_BAR_DIM, bar_rect)
        fill = pygame.Rect(bar_rect.left, bar_rect.top, int(bar_rect.width * pct), bar_rect.height)
        pygame.draw.rect(surf, HUD_HISTORY_TEXT, fill)
        pygame.draw.rect(surf, HUD_VISITS_BAR, bar_rect, width=1)
        pct_text = f"{pct * 100:>4.0f}%"
        surf.blit(font_body.render(pct_text, True, TEXT_DIM), (bar_rect.right + 6, y))


def _draw_brain_tabs(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    active_tab: int,
    font_body: pygame.font.Font,
) -> pygame.Rect:
    line_h = font_body.get_linesize()
    x = rect.left
    for idx, label in enumerate(_BRAIN_TABS):
        active = idx == active_tab
        color = HUD_HISTORY_TEXT if active else TEXT_DIM
        text = font_body.render(label, True, color)
        surf.blit(text, (x, rect.top))
        if active:
            underline_y = rect.top + text.get_height() + 1
            pygame.draw.line(surf, color, (x, underline_y), (x + text.get_width(), underline_y), width=2)
        x += text.get_width() + 22
    hint = "TAB"
    hint_surf = font_body.render(hint, True, TEXT_DIM)
    surf.blit(hint_surf, (rect.right - hint_surf.get_width(), rect.top))
    return pygame.Rect(rect.left, rect.top + line_h + 8, rect.width, rect.height - line_h - 8)


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


def _draw_brain_summary_tab(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    top_moves: list[tuple[int, int, float, float]],
    brain: dict[str, Any] | None,
    perspective_label: str,
    perspective_player: int,
    win_pct_p1: float,
    win_pct_perspective: float,
    prob_color: tuple[int, int, int],
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
) -> None:
    line_h = font_body.get_linesize()
    y = rect.top
    surf.blit(font_body.render(f"{perspective_label} {_side_name(perspective_player)}", True, TEXT_DIM), (rect.left, y))
    y += line_h
    big = font_title.render(f"{win_pct_perspective:5.1f}%", True, prob_color)
    surf.blit(big, (rect.left, y))
    status = font_body.render(_position_label(win_pct_perspective), True, prob_color)
    surf.blit(status, (rect.left + big.get_width() + 12, y + 8))
    y += big.get_height() + 4
    raw = f"ROJO {win_pct_p1:4.1f}%   AZUL {100.0 - win_pct_p1:4.1f}%"
    surf.blit(font_body.render(raw, True, TEXT_DIM), (rect.left, y))

    heat_size = 88
    if brain is not None:
        heat_rect = pygame.Rect(rect.right - heat_size, rect.top + 54, heat_size, heat_size)
        _draw_activation_heatmap(surf, heat_rect, brain.get("activation_heatmap"))
        red_line = _brain_top_line(brain)
    else:
        red_line = "Red: --"
    mcts_line = _mcts_top_line(top_moves)
    info_y = rect.bottom - (line_h * 3)
    surf.blit(font_body.render(red_line, True, HUD_HISTORY_TEXT), (rect.left, info_y))
    surf.blit(font_body.render(mcts_line, True, TEXT_MAIN), (rect.left, info_y + line_h))
    if top_moves:
        total_visits = sum(max(0, item[1]) for item in top_moves) or 1
        confidence, _gap_pts, uncertainty = _search_metrics(top_moves)
        summary = f"CONF {confidence}   INC {uncertainty}   {total_visits} sims"
        surf.blit(font_body.render(summary, True, TEXT_DIM), (rect.left, info_y + line_h * 2))


def _draw_brain_red_tab(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    brain: dict[str, Any] | None,
    perspective_player: int,
    font_body: pygame.font.Font,
) -> None:
    line_h = font_body.get_linesize()
    if brain is None:
        surf.blit(font_body.render("Esperando activacion de la red...", True, TEXT_DIM), rect.topleft)
        return
    arch = brain.get("architecture")
    if isinstance(arch, dict):
        arch_text = (
            f"{arch.get('tokens', '?')} tok | "
            f"{arch.get('layers', '?')} capas | "
            f"{arch.get('heads', '?')} heads | "
            f"d{arch.get('d_model', '?')}"
        )
    else:
        arch_text = "Transformer policy/value"
    surf.blit(font_body.render(arch_text, True, TEXT_DIM), rect.topleft)

    heat_size = min(142, rect.height - line_h - 10)
    heat_rect = pygame.Rect(rect.right - heat_size, rect.top + line_h + 6, heat_size, heat_size)
    _draw_activation_heatmap(surf, heat_rect, brain.get("activation_heatmap"))

    brain_value = float(brain.get("value", 0.0))
    brain_pct_p1 = _pct_from_value(brain_value)
    side_pct = brain_pct_p1 if perspective_player == 1 else 100.0 - brain_pct_p1
    y = rect.top + line_h + 8
    surf.blit(font_body.render(f"Eval {_side_name(perspective_player)} {side_pct:4.1f}%", True, TEXT_MAIN), (rect.left, y))
    y += line_h + 8
    surf.blit(font_body.render("Policy directa", True, TEXT_DIM), (rect.left, y))
    rows_rect = pygame.Rect(rect.left, y + line_h + 4, heat_rect.left - rect.left - 14, line_h * 3)
    _draw_policy_rows(
        surf,
        policy_top=brain.get("policy_top"),
        rect=rows_rect,
        font_body=font_body,
    )


def _draw_brain_mcts_tab(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    top_moves: list[tuple[int, int, float, float]],
    bar_color: tuple[int, int, int],
    font_body: pygame.font.Font,
) -> None:
    line_h = font_body.get_linesize()
    if not top_moves:
        surf.blit(font_body.render("MCTS: esperando...", True, TEXT_DIM), rect.topleft)
        return
    total_visits = sum(visits for _, visits, _, _ in top_moves) or 1
    if _is_forced_pass(top_moves):
        surf.blit(font_body.render("Sin jugadas legales", True, HUD_HISTORY_TEXT), rect.topleft)
        surf.blit(font_body.render(f"Pass forzado | {total_visits} sims", True, TEXT_DIM), (rect.left, rect.top + line_h))
        return
    surf.blit(font_body.render("Top 3 por visitas", True, TEXT_DIM), rect.topleft)
    rows_rect = pygame.Rect(rect.left, rect.top + line_h + 8, rect.width - 8, line_h * 3)
    _draw_compact_mcts_rows(
        surf,
        top_moves=top_moves,
        total_visits=total_visits,
        rect=rows_rect,
        font_body=font_body,
        bar_color=bar_color,
    )
    confidence, gap_pts, uncertainty = _search_metrics(top_moves)
    y = rows_rect.bottom + 12
    surf.blit(font_body.render(f"Confianza     {confidence}", True, HUD_HISTORY_TEXT), (rect.left, y))
    y += line_h
    surf.blit(font_body.render(f"Incertidumbre {uncertainty}", True, HUD_HISTORY_TEXT), (rect.left, y))
    y += line_h
    surf.blit(font_body.render(f"Gap           +{gap_pts} pts", True, TEXT_DIM), (rect.left, y))
    y += line_h
    surf.blit(font_body.render(f"Simulaciones  {total_visits}", True, TEXT_DIM), (rect.left, y))


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
        (f"Modo  : {mode_es}", TEXT_MAIN),
        (f"R {p1_name}  |  A {p2_name}", TEXT_DIM),
        (f"Turno : {turn_index}", TEXT_MAIN),
        (f"Mueve : {name_turn}", color_turn),
    ]
    for text, color in rows:
        surf.blit(font_body.render(text, True, color), (inner.left, y))
        y += line_h
    y += 4
    score_text = f"Piezas: R {p1_count:>2}  |  A {p2_count:>2}"
    surf.blit(font_body.render(score_text, True, TEXT_DIM), (inner.left, y))
    y += line_h
    speed_str = f"{speed_mult:g}x"
    state_str = f"PAUSA  vel {speed_str}" if paused else f"vel {speed_str}"
    surf.blit(
        font_body.render(state_str, True, PIECE_P1 if paused else TEXT_DIM),
        (rect.right - 14 - font_body.size(state_str)[0], rect.top + 12),
    )
    _ = stats_record


def draw_top_moves_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    *,
    top_moves: list[tuple[int, int, float, float]],
    root_value: float,
    thinker_player: int | None,
    brain: dict[str, Any] | None,
    brain_tab: int,
    p1_agent: str,
    p2_agent: str,
    turn_player: int,
    font_title: pygame.font.Font,
    font_body: pygame.font.Font,
) -> None:
    inner = _draw_panel_frame(surf, rect, "CEREBRO IA", font_title)
    bar_color = PIECE_P1 if thinker_player == 1 else PIECE_P2
    active_tab = _normalize_brain_tab(brain_tab)
    content = _draw_brain_tabs(
        surf,
        inner,
        active_tab=active_tab,
        font_body=font_body,
    )
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

    if active_tab == 1:
        _draw_brain_red_tab(
            surf,
            content,
            brain=brain,
            perspective_player=perspective_player,
            font_body=font_body,
        )
        return
    if active_tab == 2:
        _draw_brain_mcts_tab(
            surf,
            content,
            top_moves=top_moves,
            bar_color=bar_color,
            font_body=font_body,
        )
        return
    _draw_brain_summary_tab(
        surf,
        content,
        top_moves=top_moves,
        brain=brain,
        perspective_label=perspective_label,
        perspective_player=perspective_player,
        win_pct_p1=win_pct_p1,
        win_pct_perspective=win_pct_perspective,
        prob_color=prob_color,
        font_title=font_title,
        font_body=font_body,
    )


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
    brain_tab = _normalize_brain_tab(arena_state.get("brain_tab", 0))
    draw_top_moves_panel(
        surf,
        hud_mid_rect(),
        top_moves=top_moves,
        root_value=root_value,
        thinker_player=thinker_player,
        brain=arena_state.get("last_brain") if isinstance(arena_state.get("last_brain"), dict) else None,
        brain_tab=brain_tab,
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

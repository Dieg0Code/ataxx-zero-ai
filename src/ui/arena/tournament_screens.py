"""Pantallas Pygame del modo torneo: roster, sorteo, pre/post, tiebreak, podio.

Solo dibujan + manejan eventos. No conocen el estado mas alla de lo que reciben
por parametro. La logica vive en `tournament.py` y el runner.

Sistema de diseno:
- Header con titulo + breadcrumbs de sesion.
- Body con cards y paneles centrales.
- Footer con acciones tipo barra de teclas.
- Acentos de color: rojo P1 si humano arranca, azul P2 si arranca la IA.
- Jerarquia tipografica: hero(80) > title(56) > heading(40) > subhead(30) > body(26) > small(20).
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from ui.arena.layout import BOARD_AREA, WIN_H, WIN_W
from ui.arena.theme import (
    BG_BOTTOM,
    BG_TOP,
    HUD_TITLE,
    HUD_VISITS_BAR,
    PANEL_ACCENT,
    PANEL_BG,
    PANEL_BORDER,
    PIECE_P1,
    PIECE_P2,
    TARGET,
    TEXT_DIM,
    TEXT_MAIN,
)
from ui.arena.tournament import MatchRecord, TournamentState

# Paleta extendida para podio y badges.
GOLD = (255, 215, 80)
SILVER = (200, 210, 220)
BRONZE = (205, 127, 50)

# Acentos suaves para fondos translucidos de cards.
CARD_BG = (16, 22, 44)
CARD_BG_ACTIVE = (22, 30, 60)
CARD_DIM = (14, 18, 36)


@dataclass
class ScreenFonts:
    """Jerarquia tipografica para todas las pantallas del torneo."""

    small: pygame.font.Font
    body: pygame.font.Font
    subhead: pygame.font.Font
    heading: pygame.font.Font
    title: pygame.font.Font
    hero: pygame.font.Font

    @classmethod
    def from_dict(cls, raw: dict[str, pygame.font.Font]) -> ScreenFonts:
        return cls(
            small=raw["small"],
            body=raw["body"],
            subhead=raw["subhead"],
            heading=raw["heading"],
            title=raw["title"],
            hero=raw["hero"],
        )

    # Backwards-compatible accessors para el match overlay (durante la partida)
    # que reutiliza la fuente body como `font` y small como antes.
    @property
    def font(self) -> pygame.font.Font:
        return self.body


# ----- Primitivas de pintado -----


def _vertical_gradient(surface: pygame.Surface) -> None:
    """Fondo degradado vertical con vinetas en las esquinas para look pulido."""
    h = surface.get_height()
    w = surface.get_width()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
        pygame.draw.line(surface, (r, g, b), (0, y), (w, y))


def _draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    fill: tuple[int, int, int] = CARD_BG,
    border: tuple[int, int, int] = PANEL_BORDER,
    border_width: int = 2,
    radius: int = 14,
) -> None:
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, width=border_width, border_radius=radius)


def _draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: tuple[int, int],
    *,
    color: tuple[int, int, int] = TEXT_MAIN,
    anchor: str = "topleft",
) -> pygame.Rect:
    img = font.render(text, True, color)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


def _draw_centered(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    y: int,
    *,
    color: tuple[int, int, int] = TEXT_MAIN,
) -> pygame.Rect:
    return _draw_text(surface, font, text, (WIN_W // 2, y), color=color, anchor="center")


def _draw_kbd_chip(
    surface: pygame.Surface,
    font: pygame.font.Font,
    key: str,
    label: str,
    center: tuple[int, int],
    *,
    accent: tuple[int, int, int] = PANEL_ACCENT,
    dim: tuple[int, int, int] = TEXT_DIM,
) -> pygame.Rect:
    """Dibuja un chip con la tecla destacada + descripcion al lado."""
    key_img = font.render(f"[{key}]", True, accent)
    label_img = font.render(label, True, dim)
    gap = 8
    total_w = key_img.get_width() + gap + label_img.get_width()
    x = center[0] - total_w // 2
    y = center[1] - max(key_img.get_height(), label_img.get_height()) // 2
    surface.blit(key_img, (x, y))
    surface.blit(label_img, (x + key_img.get_width() + gap, y))
    return pygame.Rect(x, y, total_w, max(key_img.get_height(), label_img.get_height()))


def _draw_header(
    surface: pygame.Surface,
    fonts: ScreenFonts,
    *,
    title: str,
    subtitle: str | None = None,
) -> int:
    """Banda superior con titulo y subtitulo. Devuelve la y del borde inferior."""
    band = pygame.Rect(0, 0, WIN_W, 120)
    inner = pygame.Surface(band.size, pygame.SRCALPHA)
    inner.fill((0, 0, 0, 0))
    surface.blit(inner, band)
    _draw_centered(surface, fonts.title, title, 52, color=HUD_TITLE)
    if subtitle:
        _draw_centered(surface, fonts.small, subtitle, 92, color=TEXT_DIM)
    # Linea separadora suave.
    pygame.draw.line(
        surface, PANEL_BORDER,
        (WIN_W // 4, 118), (3 * WIN_W // 4, 118),
        width=1,
    )
    return 120


def _draw_footer(
    surface: pygame.Surface,
    fonts: ScreenFonts,
    chips: list[tuple[str, str]],
    *,
    accent_overrides: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    """Barra inferior con teclas: cada chip es ('A', 'aceptar y empezar')."""
    overrides = accent_overrides or {}
    y_top = WIN_H - 60
    pygame.draw.line(surface, PANEL_BORDER, (40, y_top), (WIN_W - 40, y_top), width=1)
    if not chips:
        return
    slot = WIN_W // len(chips)
    for idx, (key, label) in enumerate(chips):
        center = (slot * idx + slot // 2, y_top + 30)
        accent = overrides.get(key, PANEL_ACCENT)
        _draw_kbd_chip(surface, fonts.small, key, label, center, accent=accent)


def _starter_palette(starter: str) -> tuple[tuple[int, int, int], str]:
    """Devuelve (color, texto-corto) segun quien arranca la partida."""
    if starter == "human":
        return PIECE_P1, "Humano (rojo)"
    return PIECE_P2, "IA (azul)"


# ----- Pantalla: Roster -----


def draw_roster(
    surface: pygame.Surface,
    fonts: ScreenFonts,
    *,
    state: TournamentState,
    present_flags: list[bool],
    selected_index: int,
) -> None:
    _vertical_gradient(surface)
    _draw_header(
        surface, fonts,
        title="TORNEO ATAXXZERO",
        subtitle=(
            f"Sesion del {state.tournament_id}   ·   vs {state.config.model_label}"
            f"   ·   sims = {state.config.mcts_sims}"
        ),
    )

    _draw_centered(
        surface, fonts.subhead,
        f"Formato: {state.config.rounds} rondas · Bo{state.config.bo_size} · "
        "misma config para todos",
        160, color=TEXT_DIM,
    )
    _draw_centered(
        surface, fonts.body,
        "Marca quien esta presente",
        205, color=TEXT_MAIN,
    )

    list_rect = pygame.Rect(
        WIN_W // 2 - 320, 235, 640, 60 + len(state.config.players) * 56,
    )
    _draw_panel(surface, list_rect)

    for idx, player in enumerate(state.config.players):
        row_top = list_rect.y + 24 + idx * 56
        row_rect = pygame.Rect(list_rect.x + 16, row_top, list_rect.width - 32, 48)
        is_selected = idx == selected_index
        is_present = present_flags[idx] if idx < len(present_flags) else False
        if is_selected:
            _draw_panel(
                surface, row_rect,
                fill=CARD_BG_ACTIVE, border=PANEL_ACCENT,
                border_width=1, radius=10,
            )
        # Tecla numero a la izquierda como chip cuadrado.
        key_rect = pygame.Rect(row_rect.x + 16, row_rect.y + 8, 36, 32)
        _draw_panel(
            surface, key_rect,
            fill=PANEL_BG, border=PANEL_ACCENT,
            border_width=1, radius=6,
        )
        _draw_text(
            surface, fonts.body, str(idx + 1),
            (key_rect.centerx, key_rect.centery),
            color=PANEL_ACCENT, anchor="center",
        )
        # Checkbox.
        box_rect = pygame.Rect(row_rect.x + 70, row_rect.y + 10, 28, 28)
        pygame.draw.rect(surface, PANEL_BORDER, box_rect, width=2, border_radius=6)
        if is_present:
            pygame.draw.rect(
                surface, HUD_VISITS_BAR,
                box_rect.inflate(-10, -10), border_radius=4,
            )
        # Nombre.
        name_color = TEXT_MAIN if is_present else TEXT_DIM
        if is_selected and not is_present:
            name_color = TARGET
        _draw_text(
            surface, fonts.subhead, player,
            (row_rect.x + 116, row_rect.centery),
            color=name_color, anchor="midleft",
        )
        # Estado a la derecha.
        status_text = "presente" if is_present else "ausente (derrota)"
        status_color = HUD_VISITS_BAR if is_present else TEXT_DIM
        _draw_text(
            surface, fonts.small, status_text,
            (row_rect.right - 16, row_rect.centery),
            color=status_color, anchor="midright",
        )

    presentes = sum(1 for f in present_flags if f)
    ausentes = len(present_flags) - presentes
    _draw_centered(
        surface, fonts.body,
        f"{presentes} presentes  ·  {ausentes} ausentes",
        list_rect.bottom + 28, color=TEXT_DIM,
    )

    _draw_footer(
        surface, fonts,
        [
            ("1/2/3", "alternar presencia"),
            ("ENTER", "sortear orden"),
            ("F11", "pantalla completa"),
            ("Q", "salir"),
        ],
    )


# ----- Pantalla: Sorteo animado -----


def draw_shuffle(
    surface: pygame.Surface,
    fonts: ScreenFonts,
    *,
    state: TournamentState,
    rolling_names: list[str],
    final_order: list[str] | None,
    progress: float,
) -> None:
    _vertical_gradient(surface)
    _draw_header(
        surface, fonts,
        title="SORTEO",
        subtitle="Definiendo orden de jugadores",
    )

    if final_order is None:
        _draw_centered(
            surface, fonts.body, "Procesando...",
            175, color=TEXT_DIM,
        )
        if not rolling_names:
            return
        roll_idx = int(progress * len(rolling_names)) % max(1, len(rolling_names))
        # Carrusel vertical de 3 nombres rotando.
        for offset in (-1, 0, 1):
            idx = (roll_idx + offset) % len(rolling_names)
            name = rolling_names[idx]
            y = 320 + offset * 80
            if offset == 0:
                font = fonts.heading
                color = TEXT_MAIN
                # Highlight del centro.
                panel = pygame.Rect(WIN_W // 2 - 220, y - 32, 440, 64)
                _draw_panel(
                    surface, panel,
                    fill=CARD_BG_ACTIVE, border=PANEL_ACCENT,
                    border_width=2, radius=10,
                )
            else:
                font = fonts.subhead
                color = TEXT_DIM
            _draw_centered(surface, font, name, y, color=color)
    else:
        _draw_centered(
            surface, fonts.subhead, "Orden definitivo",
            170, color=PANEL_ACCENT,
        )
        # Cards horizontales con el numero de orden grande.
        card_w = 300
        gap = 24
        total_w = len(final_order) * card_w + (len(final_order) - 1) * gap
        start_x = (WIN_W - total_w) // 2 if total_w < WIN_W - 80 else 40
        card_h = 180
        card_y = 240
        # Si los cards no caben horizontal, apilamos vertical (fallback para mas jugadores).
        if total_w > WIN_W - 80:
            for idx, name in enumerate(final_order):
                row = pygame.Rect(
                    WIN_W // 2 - 280, card_y + idx * 60, 560, 50,
                )
                _draw_panel(surface, row)
                _draw_text(
                    surface, fonts.heading, f"{idx + 1}",
                    (row.x + 30, row.centery),
                    color=PANEL_ACCENT, anchor="midleft",
                )
                _draw_text(
                    surface, fonts.subhead, name,
                    (row.x + 100, row.centery),
                    color=TEXT_MAIN, anchor="midleft",
                )
        else:
            for idx, name in enumerate(final_order):
                card_rect = pygame.Rect(
                    start_x + idx * (card_w + gap), card_y, card_w, card_h,
                )
                _draw_panel(
                    surface, card_rect,
                    fill=CARD_BG, border=PANEL_ACCENT,
                    border_width=2, radius=14,
                )
                _draw_centered_in_rect(
                    surface, fonts.title, f"#{idx + 1}",
                    card_rect, dy=-30, color=PANEL_ACCENT,
                )
                _draw_centered_in_rect(
                    surface, fonts.heading, name,
                    card_rect, dy=40, color=TEXT_MAIN,
                )
        _draw_centered(
            surface, fonts.body,
            "Cada jugador completa sus 2 rondas (Bo3) antes de pasar al siguiente.",
            WIN_H - 110, color=TEXT_DIM,
        )

    chips = (
        [("Q", "cancelar")]
        if final_order is None
        else [("SPACE", "empezar primer match"), ("Q", "cancelar")]
    )
    _draw_footer(surface, fonts, chips)


def _draw_centered_in_rect(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    rect: pygame.Rect,
    *,
    dy: int = 0,
    color: tuple[int, int, int] = TEXT_MAIN,
) -> None:
    img = font.render(text, True, color)
    pos = img.get_rect(center=(rect.centerx, rect.centery + dy))
    surface.blit(img, pos)


# ----- Pantalla: Pre-partida -----


def draw_pre_match(
    surface: pygame.Surface,
    fonts: ScreenFonts,
    *,
    state: TournamentState,
    match: MatchRecord,
) -> None:
    _vertical_gradient(surface)
    completed = state.total_completed_regular()
    total = state.total_active()
    _draw_header(
        surface, fonts,
        title="TORNEO",
        subtitle=(
            f"Partida {completed + 1} de {total}   ·   "
            f"Sesion {state.tournament_id}"
        ),
    )

    # Panel principal del matchup con dos cards laterales.
    starter_color, starter_text = _starter_palette(match.starter)
    panel = pygame.Rect(WIN_W // 2 - 360, 155, 720, 220)
    _draw_panel(
        surface, panel,
        fill=CARD_BG, border=starter_color,
        border_width=2, radius=16,
    )

    # Ronda + partida arriba del panel como tag.
    tag_text = (
        f"RONDA {match.round_idx + 1} / {state.config.rounds}     "
        f"PARTIDA {match.match_idx + 1} / {state.config.bo_size}"
    )
    _draw_centered(surface, fonts.small, tag_text, panel.y + 26, color=TEXT_DIM)

    # Card del humano (izquierda).
    human_card = pygame.Rect(panel.x + 24, panel.y + 56, 296, 140)
    _draw_panel(
        surface, human_card,
        fill=CARD_BG_ACTIVE if match.starter == "human" else CARD_BG,
        border=PIECE_P1, border_width=2, radius=12,
    )
    _draw_centered_in_rect(
        surface, fonts.small, "JUGADOR HUMANO",
        human_card, dy=-50, color=TEXT_DIM,
    )
    _draw_centered_in_rect(
        surface, fonts.heading, match.player,
        human_card, dy=-10, color=PIECE_P1,
    )
    human_color = "ROJO" if match.starter == "human" else "AZUL"
    _draw_centered_in_rect(
        surface, fonts.small, f"juega {human_color}",
        human_card, dy=34, color=TEXT_DIM,
    )

    # VS al centro.
    _draw_centered_in_rect(
        surface, fonts.title, "VS",
        panel, dy=8, color=TEXT_DIM,
    )

    # Card de la IA (derecha).
    ai_card = pygame.Rect(panel.right - 24 - 296, panel.y + 56, 296, 140)
    _draw_panel(
        surface, ai_card,
        fill=CARD_BG_ACTIVE if match.starter == "ai" else CARD_BG,
        border=PIECE_P2, border_width=2, radius=12,
    )
    _draw_centered_in_rect(
        surface, fonts.small, "MODELO",
        ai_card, dy=-50, color=TEXT_DIM,
    )
    _draw_centered_in_rect(
        surface, fonts.heading, state.config.model_label,
        ai_card, dy=-10, color=PIECE_P2,
    )
    ai_color = "AZUL" if match.starter == "human" else "ROJO"
    _draw_centered_in_rect(
        surface, fonts.small, f"juega {ai_color}",
        ai_card, dy=34, color=TEXT_DIM,
    )

    # Banda con quien inicia + sims (debajo del panel).
    starter_panel = pygame.Rect(WIN_W // 2 - 260, panel.bottom + 24, 520, 50)
    _draw_panel(
        surface, starter_panel,
        fill=PANEL_BG, border=starter_color, border_width=1, radius=10,
    )
    _draw_text(
        surface, fonts.small, "INICIA:",
        (starter_panel.x + 24, starter_panel.centery),
        color=TEXT_DIM, anchor="midleft",
    )
    _draw_text(
        surface, fonts.subhead, starter_text,
        (starter_panel.x + 120, starter_panel.centery),
        color=starter_color, anchor="midleft",
    )
    _draw_text(
        surface, fonts.small, f"sims = {state.config.mcts_sims}",
        (starter_panel.right - 24, starter_panel.centery),
        color=TEXT_DIM, anchor="midright",
    )

    # Acumulado del jugador.
    standing = state.standing_for(match.player)
    accum_y = starter_panel.bottom + 28
    _draw_text(
        surface, fonts.small, f"{match.player} acumula",
        (WIN_W // 2 - 220, accum_y),
        color=TEXT_DIM, anchor="midleft",
    )
    stats = (
        f"{standing.points:.1f} pt   ·   "
        f"{standing.wins}V {standing.draws}E {standing.losses}D   ·   "
        f"Δ {standing.piece_diff:+d}"
    )
    _draw_text(
        surface, fonts.subhead, stats,
        (WIN_W // 2 - 20, accum_y),
        color=TEXT_MAIN, anchor="midleft",
    )

    _draw_footer(
        surface, fonts,
        [
            ("A", "aceptar y empezar"),
            ("F", "forfeit (se fue)"),
            ("F11", "pantalla completa"),
            ("Q", "terminar torneo"),
        ],
        accent_overrides={"F": TARGET, "Q": PIECE_P1},
    )


# ----- Pantalla: Post-partida -----


def draw_post_match(
    surface: pygame.Surface,
    fonts: ScreenFonts,
    *,
    state: TournamentState,
    match: MatchRecord,
) -> None:
    _vertical_gradient(surface)

    veredict = {"W": "VICTORIA", "L": "DERROTA", "D": "EMPATE"}.get(match.result, "—")
    veredict_color = (
        HUD_VISITS_BAR if match.result == "W"
        else TARGET if match.result == "D"
        else PIECE_P1
    )
    _draw_header(
        surface, fonts,
        title=veredict,
        subtitle=(
            f"{match.player} · Ronda {match.round_idx + 1} P{match.match_idx + 1} · "
            f"Inicio: {_starter_palette(match.starter)[1]}"
        ),
    )
    # Redibujamos el titulo con el color del veredicto encima.
    _draw_centered(surface, fonts.title, veredict, 52, color=veredict_color)

    # Marcador principal.
    score_panel = pygame.Rect(WIN_W // 2 - 360, 150, 720, 130)
    _draw_panel(
        surface, score_panel,
        fill=CARD_BG, border=veredict_color, border_width=2, radius=14,
    )
    # Lado humano.
    _draw_text(
        surface, fonts.small, match.player.upper(),
        (score_panel.x + 60, score_panel.y + 30),
        color=PIECE_P1, anchor="topleft",
    )
    _draw_text(
        surface, fonts.hero, str(match.human_pieces),
        (score_panel.x + 60, score_panel.y + 50),
        color=TEXT_MAIN, anchor="topleft",
    )
    # Separador.
    _draw_text(
        surface, fonts.heading, "—",
        (score_panel.centerx, score_panel.centery),
        color=TEXT_DIM, anchor="center",
    )
    # Lado modelo.
    label_w = fonts.small.size(state.config.model_label.upper())[0]
    _draw_text(
        surface, fonts.small, state.config.model_label.upper(),
        (score_panel.right - 60 - label_w, score_panel.y + 30),
        color=PIECE_P2, anchor="topleft",
    )
    ai_num_w = fonts.hero.size(str(match.ai_pieces))[0]
    _draw_text(
        surface, fonts.hero, str(match.ai_pieces),
        (score_panel.right - 60 - ai_num_w, score_panel.y + 50),
        color=TEXT_MAIN, anchor="topleft",
    )

    # Linea de puntos + delta.
    points = {"W": 1.0, "D": 0.5, "L": 0.0}.get(match.result, 0.0)
    detail = (
        f"+{points:.1f} pt    ·    Δ {match.piece_diff:+d}    ·    "
        f"{match.halfmoves} medias jugadas"
    )
    _draw_centered(surface, fonts.subhead, detail, score_panel.bottom + 30, color=TEXT_DIM)

    # Tabla acumulada.
    table_top = score_panel.bottom + 80
    _draw_text(
        surface, fonts.subhead, "ACUMULADO",
        (WIN_W // 2 - 320, table_top),
        color=PANEL_ACCENT, anchor="topleft",
    )

    table_rect = pygame.Rect(WIN_W // 2 - 340, table_top + 36, 680, 168)
    _draw_panel(surface, table_rect)
    standings = [state.standing_for(p) for p in state.config.players]
    standings = [s for s in standings if s.played > 0]
    standings.sort(key=lambda s: (s.points, s.wins, s.piece_diff), reverse=True)
    headers = ("#", "Jugador", "Puntos", "V/E/D", "Δ piezas")
    cols_x = [
        table_rect.x + 30,
        table_rect.x + 80,
        table_rect.x + 290,
        table_rect.x + 410,
        table_rect.x + 540,
    ]
    head_y = table_rect.y + 12
    for col, header in zip(cols_x, headers, strict=False):
        _draw_text(
            surface, fonts.small, header,
            (col, head_y),
            color=TEXT_DIM, anchor="topleft",
        )
    pygame.draw.line(
        surface, PANEL_BORDER,
        (table_rect.x + 20, head_y + 24),
        (table_rect.right - 20, head_y + 24),
        width=1,
    )
    for row_idx, st in enumerate(standings[:4]):
        row_y = head_y + 38 + row_idx * 28
        is_match_player = st.player == match.player
        row_color = TEXT_MAIN if is_match_player else TEXT_DIM
        if is_match_player:
            highlight = pygame.Rect(
                table_rect.x + 12, row_y - 4,
                table_rect.width - 24, 26,
            )
            pygame.draw.rect(surface, CARD_BG_ACTIVE, highlight, border_radius=6)
        _draw_text(surface, fonts.small, f"{row_idx + 1}", (cols_x[0], row_y), color=row_color)
        _draw_text(surface, fonts.body, st.player, (cols_x[1], row_y - 4), color=row_color)
        _draw_text(
            surface, fonts.body, f"{st.points:.1f}",
            (cols_x[2], row_y - 4), color=row_color,
        )
        _draw_text(
            surface, fonts.body,
            f"{st.wins}V {st.draws}E {st.losses}D",
            (cols_x[3], row_y - 4), color=row_color,
        )
        _draw_text(
            surface, fonts.body, f"{st.piece_diff:+d}",
            (cols_x[4], row_y - 4), color=row_color,
        )

    _draw_footer(
        surface, fonts,
        [("SPACE", "continuar al siguiente match")],
    )


# ----- Pantalla: Pre-tiebreak -----


def draw_tiebreak_intro(
    surface: pygame.Surface,
    fonts: ScreenFonts,
    *,
    state: TournamentState,
    match: MatchRecord,
    other_player: str,
) -> None:
    _vertical_gradient(surface)
    _draw_header(
        surface, fonts,
        title="PARTIDA RELAMPAGO",
        subtitle="Empate persistente despues de los tres criterios",
    )

    starter_color, starter_text = _starter_palette(match.starter)
    panel = pygame.Rect(WIN_W // 2 - 360, 160, 720, 200)
    _draw_panel(
        surface, panel,
        fill=CARD_BG, border=TARGET, border_width=2, radius=16,
    )
    _draw_centered_in_rect(
        surface, fonts.small, "DESEMPATE CONTRA",
        panel, dy=-65, color=TEXT_DIM,
    )
    _draw_centered_in_rect(
        surface, fonts.heading, other_player,
        panel, dy=-25, color=PIECE_P1,
    )
    _draw_centered_in_rect(
        surface, fonts.subhead,
        f"{match.player}  vs  {state.config.model_label}",
        panel, dy=30, color=TEXT_MAIN,
    )
    _draw_centered_in_rect(
        surface, fonts.small, f"sims = {state.config.mcts_sims}",
        panel, dy=70, color=TEXT_DIM,
    )

    starter_panel = pygame.Rect(WIN_W // 2 - 260, panel.bottom + 28, 520, 50)
    _draw_panel(
        surface, starter_panel,
        fill=PANEL_BG, border=starter_color, border_width=1, radius=10,
    )
    _draw_text(
        surface, fonts.small, "INICIA:",
        (starter_panel.x + 24, starter_panel.centery),
        color=TEXT_DIM, anchor="midleft",
    )
    _draw_text(
        surface, fonts.subhead, starter_text,
        (starter_panel.x + 120, starter_panel.centery),
        color=starter_color, anchor="midleft",
    )

    _draw_footer(
        surface, fonts,
        [("A", "aceptar y empezar"), ("Q", "terminar torneo")],
    )


# ----- Pantalla: Podio final -----


def draw_final_podium(
    surface: pygame.Surface,
    fonts: ScreenFonts,
    *,
    state: TournamentState,
) -> None:
    _vertical_gradient(surface)
    _draw_header(
        surface, fonts,
        title="PODIO FINAL",
        subtitle=(
            f"Torneo {state.tournament_id}   ·   vs {state.config.model_label}"
            f"   ·   sims = {state.config.mcts_sims}"
        ),
    )

    ranking = state.ranking()
    medal_colors = (GOLD, SILVER, BRONZE)
    medal_labels = ("1°", "2°", "3°")

    for idx in range(3):
        slot_rect = pygame.Rect(WIN_W // 2 - 360, 155 + idx * 110, 720, 96)
        accent = medal_colors[idx]
        _draw_panel(
            surface, slot_rect,
            fill=CARD_BG, border=accent, border_width=2, radius=14,
        )
        # Medalla badge cuadrado a la izquierda.
        badge = pygame.Rect(slot_rect.x + 16, slot_rect.y + 16, 64, 64)
        _draw_panel(
            surface, badge,
            fill=PANEL_BG, border=accent, border_width=2, radius=10,
        )
        _draw_centered_in_rect(
            surface, fonts.heading, medal_labels[idx],
            badge, color=accent,
        )
        if idx < len(ranking):
            st = ranking[idx]
            _draw_text(
                surface, fonts.heading, st.player,
                (slot_rect.x + 100, slot_rect.y + 16),
                color=TEXT_MAIN, anchor="topleft",
            )
            details = (
                f"{st.points:.1f} pt   ·   "
                f"{st.wins}V {st.draws}E {st.losses}D   ·   "
                f"Δ piezas {st.piece_diff:+d}"
            )
            _draw_text(
                surface, fonts.body, details,
                (slot_rect.x + 100, slot_rect.y + 60),
                color=TEXT_DIM, anchor="topleft",
            )
            bonus_text = f"+{state.bonus_for_rank(idx):.1f}"
            _draw_text(
                surface, fonts.title, bonus_text,
                (slot_rect.right - 28, slot_rect.centery),
                color=accent, anchor="midright",
            )
            _draw_text(
                surface, fonts.small, "BONUS NOTA FINAL",
                (slot_rect.right - 28, slot_rect.centery + 30),
                color=TEXT_DIM, anchor="midright",
            )
        else:
            _draw_text(
                surface, fonts.body, "sin participante",
                (slot_rect.x + 100, slot_rect.centery),
                color=TEXT_DIM, anchor="midleft",
            )

    note_lines: list[str] = []
    if state.tiebreak_matches():
        note_lines.append("Hubo desempate por partida relampago.")
    note_lines.append("Bonus aplicado sobre la evaluacion final del 19/05/2026.")
    for offset, line in enumerate(note_lines):
        _draw_centered(
            surface, fonts.small, line,
            WIN_H - 110 + offset * 22, color=PANEL_ACCENT,
        )

    _draw_footer(
        surface, fonts,
        [("P", "guardar screenshot"), ("Q", "salir")],
    )


# ----- Overlay durante una partida del torneo -----


def draw_match_overlay(
    surface: pygame.Surface,
    fonts: ScreenFonts,
    *,
    state: TournamentState,
    match: MatchRecord,
) -> None:
    """Banner discreto encima del board (no se solapa con el HUD)."""
    starter_color, _ = _starter_palette(match.starter)
    if match.is_tiebreak:
        line1 = "PARTIDA RELAMPAGO"
        line2 = (
            f"{match.player} vs {state.config.model_label}  ·  "
            f"Inicia {_starter_palette(match.starter)[1]}"
        )
    else:
        completed = state.total_completed_regular()
        total = state.total_active()
        line1 = (
            f"TORNEO  ·  Partida {completed + 1} / {total}  ·  "
            f"R{match.round_idx + 1} P{match.match_idx + 1}"
        )
        line2 = (
            f"{match.player}  vs  {state.config.model_label}   "
            f"Inicia {_starter_palette(match.starter)[1]}"
        )
    img1 = fonts.small.render(line1, True, PANEL_ACCENT)
    img2 = fonts.small.render(line2, True, TEXT_DIM)
    pad_x = 16
    pad_y = 6
    w = max(img1.get_width(), img2.get_width()) + pad_x * 2
    h = img1.get_height() + img2.get_height() + pad_y * 2 + 2
    bg_rect = pygame.Rect(0, 0, w, h)
    # Centrado en el AREA DEL BOARD (no en la ventana), para no solapar el HUD.
    bg_rect.midtop = (BOARD_AREA // 2, 6)
    overlay = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
    overlay.fill((*PANEL_BG, 230))
    surface.blit(overlay, bg_rect)
    pygame.draw.rect(surface, starter_color, bg_rect, width=2, border_radius=10)
    surface.blit(img1, (bg_rect.centerx - img1.get_width() // 2, bg_rect.y + pad_y))
    surface.blit(
        img2,
        (
            bg_rect.centerx - img2.get_width() // 2,
            bg_rect.y + pad_y + img1.get_height() + 2,
        ),
    )


__all__ = (
    "PIECE_P1",
    "PIECE_P2",
    "ScreenFonts",
    "draw_final_podium",
    "draw_match_overlay",
    "draw_post_match",
    "draw_pre_match",
    "draw_roster",
    "draw_shuffle",
    "draw_tiebreak_intro",
)

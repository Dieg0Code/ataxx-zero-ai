from __future__ import annotations

from pathlib import Path

import pygame

from ui.arena.theme import USE_PIXEL_FONT

_FONT_PATH = Path(__file__).parent / "assets" / "VT323-Regular.ttf"


def load_font(size: int, bold: bool = False) -> pygame.font.Font:
    if USE_PIXEL_FONT and _FONT_PATH.is_file():
        # VT323 needs ~1.2x the SysFont size for equivalent legibility.
        scaled = round(size * 1.2)
        return pygame.font.Font(str(_FONT_PATH), scaled)
    return pygame.font.SysFont("consolas", size, bold=bold)


def load_arena_fonts() -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
    font = load_font(26)
    small = load_font(20)
    big = load_font(104, bold=True)
    return font, small, big


def load_tournament_fonts() -> dict[str, pygame.font.Font]:
    """Fuentes con jerarquia mas fina para las pantallas del torneo.

    Mantiene `small`/`body` compatibles con la arena, y agrega tamanos
    intermedios para titulos secundarios y headings sin caer en `big=104`.
    """
    return {
        "small": load_font(20),
        "body": load_font(26),
        "subhead": load_font(30, bold=True),
        "heading": load_font(40, bold=True),
        "title": load_font(56, bold=True),
        "hero": load_font(80, bold=True),
    }

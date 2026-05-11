from __future__ import annotations

import math

import pygame

from ui.arena.layout import WIN_H, WIN_W
from ui.arena.theme import (
    CRT_SCANLINE_ALPHA,
    CRT_VIGNETTE_ALPHA,
    USE_CRT_OVERLAY,
    VIGNETTE,
)

_scanline_surface: pygame.Surface | None = None
_vignette_surface: pygame.Surface | None = None


def _build_scanlines() -> pygame.Surface:
    surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    for y in range(0, WIN_H, 2):
        pygame.draw.line(surf, (0, 0, 0, CRT_SCANLINE_ALPHA), (0, y), (WIN_W, y), 1)
    return surf


def _build_vignette() -> pygame.Surface:
    surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    cx, cy = WIN_W / 2.0, WIN_H / 2.0
    max_d = math.hypot(cx, cy)
    inner = max_d * 0.55
    band_h = 4
    for y in range(0, WIN_H, band_h):
        ny = y + band_h / 2.0
        for x in range(0, WIN_W, band_h):
            nx = x + band_h / 2.0
            d = math.hypot(nx - cx, ny - cy)
            t = max(0.0, (d - inner) / (max_d - inner))
            a = int(CRT_VIGNETTE_ALPHA * (t * t))
            if a <= 0:
                continue
            pygame.draw.rect(surf, (*VIGNETTE, a), (x, y, band_h, band_h))
    return surf


def apply_crt(surface: pygame.Surface) -> None:
    if not USE_CRT_OVERLAY:
        return
    global _scanline_surface, _vignette_surface
    if _scanline_surface is None:
        _scanline_surface = _build_scanlines()
    if _vignette_surface is None:
        _vignette_surface = _build_vignette()
    surface.blit(_scanline_surface, (0, 0))
    surface.blit(_vignette_surface, (0, 0))

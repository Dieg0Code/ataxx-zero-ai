from __future__ import annotations

import pygame

from ui.arena.theme import PIECE_P1, PIECE_P2, PIECE_SHADOW, PIXEL_PIECE_BLOCK

_PIECE_CACHE: dict[tuple[int, int, tuple[int, int, int]], pygame.Surface] = {}


def _highlight_for(color: tuple[int, int, int]) -> tuple[int, int, int]:
    # Lift each channel toward white for a pixel-art specular dot.
    return tuple(min(255, c + 90) for c in color)  # type: ignore[return-value]


def render_pixel_piece(
    diameter: int,
    color: tuple[int, int, int],
    block: int = PIXEL_PIECE_BLOCK,
) -> pygame.Surface:
    """Build a chunky pixel-art coin sprite (cached). Returns a surface (D x D) with alpha."""
    key = (diameter, block, color)
    cached = _PIECE_CACHE.get(key)
    if cached is not None:
        return cached

    pad = block * 2  # space for shadow offset
    size = diameter + pad
    surf = pygame.Surface((size, size), pygame.SRCALPHA)

    cx = cy = size // 2
    radius = diameter // 2
    # Quantize all coords to the block grid so squares align cleanly.
    grid_min = -(radius // block) * block - block
    grid_max = (radius // block) * block + block
    highlight = _highlight_for(color)
    shadow_off = max(2, block // 2)

    for dy in range(grid_min, grid_max + 1, block):
        for dx in range(grid_min, grid_max + 1, block):
            d = ((dx + block / 2) ** 2 + (dy + block / 2) ** 2) ** 0.5
            if d > radius:
                continue
            # Shadow square (offset)
            pygame.draw.rect(
                surf,
                PIECE_SHADOW,
                (cx + dx + shadow_off, cy + dy + shadow_off, block, block),
            )

    for dy in range(grid_min, grid_max + 1, block):
        for dx in range(grid_min, grid_max + 1, block):
            d = ((dx + block / 2) ** 2 + (dy + block / 2) ** 2) ** 0.5
            if d > radius:
                continue
            # Specular: brighter near top-left quadrant
            spec = (dx + dy) < -(radius * 0.6)
            fill = highlight if spec else color
            pygame.draw.rect(surf, fill, (cx + dx, cy + dy, block, block))

    _PIECE_CACHE[key] = surf
    return surf


def blit_pixel_piece(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    radius: int,
    player: int,
) -> None:
    color = PIECE_P1 if player > 0 else PIECE_P2
    diameter = max(8, radius * 2)
    sprite = render_pixel_piece(diameter, color)
    rect = sprite.get_rect()
    rect.center = (cx, cy)
    surface.blit(sprite, rect.topleft)

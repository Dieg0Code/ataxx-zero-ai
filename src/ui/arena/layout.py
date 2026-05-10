from __future__ import annotations

import pygame

# Window: 1280x720 (16:9). Board area on the left, HUD panels on the right.
WIN_W = 1280
WIN_H = 720

BS = 7
PAD = 24
# Board fits into 720x720 area: 696 board + 24 pad each side.
BOARD_AREA = 720
CELL = (BOARD_AREA - 2 * PAD) // BS  # 96
BOARD_PX = BS * CELL  # 672

# HUD panel takes the rest of the width. Three vertical sections.
HUD_X = BOARD_AREA  # 720
HUD_W = WIN_W - HUD_X  # 560
HUD_TOP_H = 200
HUD_MID_H = 300
HUD_BOTTOM_H = WIN_H - HUD_TOP_H - HUD_MID_H  # 220

# Backwards compat: SIDE was the old sidebar width. Some callers may import.
SIDE = HUD_W


def hud_top_rect() -> pygame.Rect:
    return pygame.Rect(HUD_X, 0, HUD_W, HUD_TOP_H)


def hud_mid_rect() -> pygame.Rect:
    return pygame.Rect(HUD_X, HUD_TOP_H, HUD_W, HUD_MID_H)


def hud_bottom_rect() -> pygame.Rect:
    return pygame.Rect(HUD_X, HUD_TOP_H + HUD_MID_H, HUD_W, HUD_BOTTOM_H)

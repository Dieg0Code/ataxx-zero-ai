from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.constants import BOARD_SIZE, EMPTY, PLAYER_1, PLAYER_2
from game.types import Move
from ui.arena.fonts import load_arena_fonts
from ui.arena.hud import move_notation
from ui.arena.layout import HUD_X, WIN_H, WIN_W
from ui.arena.match_runtime import apply_move_with_feedback, infection_wave_schedule
from ui.arena.theme import (
    BG_BOTTOM,
    BG_TOP,
    FLASH_INFECT,
    HUD_HISTORY_TEXT,
    HUD_TITLE,
    HUD_VALUE_BAR_BG,
    INFECT_PULSE_MS,
    MOVE_PREVIEW_MS_AI,
    MOVE_PULSE_MS,
    PANEL_BG,
    PANEL_BORDER,
    PIECE_P1,
    PIECE_P2,
    TEXT_DIM,
    TEXT_MAIN,
)


@dataclass(frozen=True)
class ReplayPaths:
    npz_path: Path
    json_path: Path | None


@dataclass(frozen=True)
class ReplayFrame:
    index: int
    grid: np.ndarray
    player: int
    move: Move | None
    action_idx: int
    policy_prob: float
    value: float
    is_final: bool = False


@dataclass(frozen=True)
class LoadedReplay:
    path: Path
    metadata: dict[str, Any]
    frames: list[ReplayFrame]


@dataclass(frozen=True)
class ReplayRenderFrame:
    board: AtaxxBoard
    preview_move: Move | None
    preview_started_at: int | None
    preview_until: int
    now_ms: int
    move_cells: list[tuple[int, int]]
    infect_cells: list[tuple[int, int]]
    infection_hidden: dict[tuple[int, int], tuple[int, int]]
    move_until: int
    infect_until: int


@dataclass(frozen=True)
class ReplayListItem:
    npz_path: Path
    json_path: Path | None
    metadata: dict[str, Any]
    sort_dt: datetime
    date_label: str
    title: str
    subtitle: str


_SPANISH_DAYS = (
    "Lunes",
    "Martes",
    "Miercoles",
    "Jueves",
    "Viernes",
    "Sabado",
    "Domingo",
)
_SPANISH_MONTHS = (
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
)
_LIBRARY_VISIBLE_ROWS = 8
_QUALITY_TAGS = ("", "good", "brilliant", "late_game", "demo", "bad", "bug")


def resolve_replay_paths(path: str | Path) -> ReplayPaths:
    candidate = Path(path)
    if candidate.suffix.lower() == ".json":
        npz_path = candidate.with_suffix(".npz")
        json_path = candidate
    else:
        npz_path = candidate
        json_candidate = candidate.with_suffix(".json")
        json_path = json_candidate if json_candidate.is_file() else None
    if not npz_path.is_file():
        raise FileNotFoundError(f"Replay NPZ not found: {npz_path}")
    if json_path is not None and not json_path.is_file():
        raise FileNotFoundError(f"Replay metadata JSON not found: {json_path}")
    return ReplayPaths(npz_path=npz_path, json_path=json_path)


def spanish_date_label(value: date) -> str:
    day = _SPANISH_DAYS[value.weekday()]
    month = _SPANISH_MONTHS[value.month - 1]
    return f"{day} {value.day} de {month}, {value.year}"


def discover_replays(root: str | Path = "tournament_replays") -> list[ReplayListItem]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    items: list[ReplayListItem] = []
    for npz_path in sorted(root_path.rglob("*.npz")):
        json_path = npz_path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if json_path.is_file():
            try:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        sort_dt = _replay_sort_datetime(npz_path=npz_path, metadata=metadata)
        title = _replay_title(npz_path=npz_path, metadata=metadata)
        subtitle = _replay_subtitle(metadata)
        items.append(
            ReplayListItem(
                npz_path=npz_path,
                json_path=json_path if json_path.is_file() else None,
                metadata=metadata,
                sort_dt=sort_dt,
                date_label=spanish_date_label(sort_dt.date()),
                title=title,
                subtitle=subtitle,
            ),
        )
    items.sort(key=lambda item: (item.sort_dt, item.npz_path.name), reverse=True)
    return items


def update_replay_metadata(path: str | Path, **updates: str) -> None:
    paths = resolve_replay_paths(path)
    json_path = paths.json_path or paths.npz_path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if json_path.is_file():
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    for key, value in updates.items():
        clean = value.strip()
        if clean:
            metadata[key] = clean
        else:
            metadata.pop(key, None)
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def cycle_quality_tag(path: str | Path) -> str:
    paths = resolve_replay_paths(path)
    json_path = paths.json_path or paths.npz_path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if json_path.is_file():
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    current = str(metadata.get("quality_tag", "")).strip()
    try:
        next_idx = (_QUALITY_TAGS.index(current) + 1) % len(_QUALITY_TAGS)
    except ValueError:
        next_idx = 0
    next_tag = _QUALITY_TAGS[next_idx]
    update_replay_metadata(paths.npz_path, quality_tag=next_tag)
    return next_tag


def _replay_sort_datetime(*, npz_path: Path, metadata: dict[str, Any]) -> datetime:
    tournament_id = metadata.get("tournament_id")
    if isinstance(tournament_id, str):
        try:
            return datetime.fromisoformat(tournament_id)
        except ValueError:
            pass
    stem_prefix = npz_path.stem[:19]
    try:
        return datetime.strptime(stem_prefix, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return datetime.fromtimestamp(npz_path.stat().st_mtime)


def _replay_title(*, npz_path: Path, metadata: dict[str, Any]) -> str:
    nickname = metadata.get("nickname")
    if isinstance(nickname, str) and nickname.strip():
        return nickname.strip()
    event = metadata.get("event")
    if isinstance(event, str) and event.strip():
        return event.strip()
    human = metadata.get("player_human")
    ai = metadata.get("player_ai")
    if isinstance(human, str) and isinstance(ai, str):
        return f"{human} vs {ai}"
    return npz_path.stem


def _replay_subtitle(metadata: dict[str, Any]) -> str:
    p1 = str(metadata.get("p1_label", "ROJO"))
    p2 = str(metadata.get("p2_label", "AZUL"))
    moves = metadata.get("n_moves", "?")
    winner = metadata.get("winner")
    if winner == 1:
        winner_text = "gana ROJO"
    elif winner == -1:
        winner_text = "gana AZUL"
    elif winner == 0:
        winner_text = "empate"
    else:
        winner_text = "sin resultado"
    return f"{p1} / {p2} | {moves} jugadas | {winner_text}"


def player_for_turn(turn_index: int) -> int:
    return PLAYER_1 if turn_index % 2 == 0 else PLAYER_2


def grid_from_observation(observation: np.ndarray, player: int) -> np.ndarray:
    if observation.shape[0] < 3 or observation.shape[1:] != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"Expected observation shape (C, {BOARD_SIZE}, {BOARD_SIZE}), got {observation.shape}",
        )
    own = observation[0] > 0.5
    opponent = observation[1] > 0.5
    grid = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=np.int8)
    grid[own] = player
    grid[opponent] = PLAYER_2 if player == PLAYER_1 else PLAYER_1
    return grid


def action_from_policy(policy: np.ndarray) -> tuple[int, Move | None, float]:
    if policy.shape != (ACTION_SPACE.num_actions,):
        raise ValueError(
            f"Expected policy shape ({ACTION_SPACE.num_actions},), got {policy.shape}",
        )
    action_idx = int(np.argmax(policy))
    return action_idx, ACTION_SPACE.decode(action_idx), float(policy[action_idx])


def load_replay(path: str | Path) -> LoadedReplay:
    paths = resolve_replay_paths(path)
    metadata: dict[str, Any] = {}
    if paths.json_path is not None:
        metadata = json.loads(paths.json_path.read_text(encoding="utf-8"))

    data = np.load(paths.npz_path)
    observations = np.asarray(data["observations"], dtype=np.float32)
    policies = np.asarray(data["policies"], dtype=np.float32)
    values = np.asarray(data["values"], dtype=np.float32)
    if observations.ndim != 4:
        raise ValueError(f"observations must be rank-4, got shape {observations.shape}")
    if policies.ndim != 2 or policies.shape[0] != observations.shape[0]:
        raise ValueError(
            f"policies shape {policies.shape} does not match observations {observations.shape}",
        )
    if values.shape != (observations.shape[0],):
        raise ValueError(f"values shape {values.shape} does not match observations")

    frames: list[ReplayFrame] = []
    for idx, observation in enumerate(observations):
        player = player_for_turn(idx)
        grid = grid_from_observation(observation, player)
        action_idx, move, prob = action_from_policy(policies[idx])
        frames.append(
            ReplayFrame(
                index=idx,
                grid=grid,
                player=player,
                move=move,
                action_idx=action_idx,
                policy_prob=prob,
                value=float(values[idx]),
            ),
        )

    final_frame = _build_final_frame(frames)
    if final_frame is not None:
        frames.append(final_frame)

    return LoadedReplay(path=paths.npz_path, metadata=metadata, frames=frames)


def _build_final_frame(frames: list[ReplayFrame]) -> ReplayFrame | None:
    if not frames:
        return None
    last = frames[-1]
    board = AtaxxBoard(grid=last.grid.copy(), player=last.player)
    try:
        board.step(last.move)
    except ValueError:
        return None
    return ReplayFrame(
        index=last.index + 1,
        grid=board.grid.copy(),
        player=int(board.current_player),
        move=None,
        action_idx=ACTION_SPACE.pass_index,
        policy_prob=0.0,
        value=0.0,
        is_final=True,
    )


def run_replay_viewer(path: str | Path) -> None:
    replay = load_replay(path)
    if not replay.frames:
        raise ValueError("Replay has no frames.")

    pygame.init()
    try:
        font, small, big = load_arena_fonts()
        window = pygame.display.set_mode(_scaled_window_size())
        pygame.display.set_caption(f"Ataxx replay - {replay.path.name}")
        screen = pygame.Surface((WIN_W, WIN_H))
        clock = pygame.time.Clock()
        frame_idx = 0
        playing = False
        speed = 1.0
        anim_start_ms: int | None = None
        running = True
        while running:
            now_ms = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        playing = not playing
                        anim_start_ms = now_ms if playing else None
                    elif event.key in (pygame.K_RIGHT, pygame.K_s):
                        frame_idx = min(len(replay.frames) - 1, frame_idx + 1)
                        playing = False
                        anim_start_ms = None
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        frame_idx = max(0, frame_idx - 1)
                        playing = False
                        anim_start_ms = None
                    elif event.key == pygame.K_HOME:
                        frame_idx = 0
                        playing = False
                        anim_start_ms = None
                    elif event.key == pygame.K_END:
                        frame_idx = len(replay.frames) - 1
                        playing = False
                        anim_start_ms = None
                    elif event.key == pygame.K_1:
                        speed = 1.0
                    elif event.key == pygame.K_2:
                        speed = 2.0
                    elif event.key == pygame.K_4:
                        speed = 4.0
                    elif event.key == pygame.K_p:
                        out_dir = Path("arena_screenshots")
                        out_dir.mkdir(exist_ok=True)
                        pygame.image.save(screen, str(out_dir / f"replay_{now_ms}.png"))
            if playing and active_animation_done(
                frame=replay.frames[frame_idx],
                anim_start_ms=anim_start_ms,
                now_ms=now_ms,
                speed=speed,
            ):
                frame_idx = min(len(replay.frames) - 1, frame_idx + 1)
                anim_start_ms = now_ms
                if frame_idx >= len(replay.frames) - 1:
                    playing = False
                    anim_start_ms = None

            _draw_replay_scene(
                screen,
                replay=replay,
                frame_idx=frame_idx,
                playing=playing,
                speed=speed,
                anim_start_ms=anim_start_ms,
                font=font,
                small=small,
                big=big,
            )
            _flip_scene(screen, window)
            clock.tick(60)
    finally:
        pygame.quit()


def run_replay_library(root: str | Path = "tournament_replays") -> None:
    pygame.init()
    try:
        font, small, big = load_arena_fonts()
        window = pygame.display.set_mode(_scaled_window_size())
        pygame.display.set_caption("Ataxx replay library")
        screen = pygame.Surface((WIN_W, WIN_H))
        clock = pygame.time.Clock()
        items = discover_replays(root)
        selected = 0
        offset = 0
        mode = "library"
        active_replay: LoadedReplay | None = None
        frame_idx = 0
        playing = False
        speed = 1.0
        anim_start_ms: int | None = None
        edit_kind: str | None = None
        edit_text = ""
        running = True
        while running:
            now_ms = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif edit_kind is not None:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            edit_kind = None
                            edit_text = ""
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            if items:
                                key = "nickname" if edit_kind == "nickname" else "event"
                                update_replay_metadata(items[selected].npz_path, **{key: edit_text})
                                items = discover_replays(root)
                                selected = min(selected, max(0, len(items) - 1))
                            edit_kind = None
                            edit_text = ""
                        elif event.key == pygame.K_BACKSPACE:
                            edit_text = edit_text[:-1]
                        elif event.unicode and event.unicode.isprintable():
                            edit_text += event.unicode
                    continue
                elif mode == "viewer" and active_replay is not None:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            mode = "library"
                            active_replay = None
                            playing = False
                            anim_start_ms = None
                        elif event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            playing = not playing
                            anim_start_ms = now_ms if playing else None
                        elif event.key in (pygame.K_RIGHT, pygame.K_s):
                            frame_idx = min(len(active_replay.frames) - 1, frame_idx + 1)
                            playing = False
                            anim_start_ms = None
                        elif event.key in (pygame.K_LEFT, pygame.K_a):
                            frame_idx = max(0, frame_idx - 1)
                            playing = False
                            anim_start_ms = None
                        elif event.key == pygame.K_HOME:
                            frame_idx = 0
                            playing = False
                            anim_start_ms = None
                        elif event.key == pygame.K_END:
                            frame_idx = len(active_replay.frames) - 1
                            playing = False
                            anim_start_ms = None
                        elif event.key == pygame.K_1:
                            speed = 1.0
                        elif event.key == pygame.K_2:
                            speed = 2.0
                        elif event.key == pygame.K_4:
                            speed = 4.0
                    continue
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        items = discover_replays(root)
                        selected = min(selected, max(0, len(items) - 1))
                    elif event.key == pygame.K_DOWN:
                        selected = min(len(items) - 1, selected + 1)
                    elif event.key == pygame.K_UP:
                        selected = max(0, selected - 1)
                    elif event.key == pygame.K_PAGEUP:
                        selected = max(0, selected - 8)
                    elif event.key == pygame.K_PAGEDOWN:
                        selected = min(len(items) - 1, selected + 8)
                    elif event.key == pygame.K_HOME:
                        selected = 0
                    elif event.key == pygame.K_END:
                        selected = max(0, len(items) - 1)
                    elif event.key == pygame.K_n and items:
                        edit_kind = "nickname"
                        edit_text = str(items[selected].metadata.get("nickname", ""))
                    elif event.key == pygame.K_e and items:
                        edit_kind = "event"
                        edit_text = str(items[selected].metadata.get("event", ""))
                    elif event.key == pygame.K_t and items:
                        cycle_quality_tag(items[selected].npz_path)
                        items = discover_replays(root)
                        selected = min(selected, max(0, len(items) - 1))
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and items:
                        active_replay = load_replay(items[selected].npz_path)
                        frame_idx = 0
                        playing = False
                        anim_start_ms = None
                        mode = "viewer"

            if mode == "viewer" and active_replay is not None:
                if playing and active_animation_done(
                    frame=active_replay.frames[frame_idx],
                    anim_start_ms=anim_start_ms,
                    now_ms=now_ms,
                    speed=speed,
                ):
                    frame_idx = min(len(active_replay.frames) - 1, frame_idx + 1)
                    anim_start_ms = now_ms
                    if frame_idx >= len(active_replay.frames) - 1:
                        playing = False
                        anim_start_ms = None
                _draw_replay_scene(
                    screen,
                    replay=active_replay,
                    frame_idx=frame_idx,
                    playing=playing,
                    speed=speed,
                    anim_start_ms=anim_start_ms,
                    font=font,
                    small=small,
                    big=big,
                )
            else:
                selected, offset = _clamp_library_view(items, selected, offset)
                _draw_library_scene(
                    screen,
                    root=Path(root),
                    items=items,
                    selected=selected,
                    offset=offset,
                    edit_kind=edit_kind,
                    edit_text=edit_text,
                    font=font,
                    small=small,
                )
            _flip_scene(screen, window)
            clock.tick(60)
    finally:
        pygame.quit()


def _clamp_library_view(
    items: list[ReplayListItem],
    selected: int,
    offset: int,
) -> tuple[int, int]:
    if not items:
        return 0, 0
    selected = max(0, min(selected, len(items) - 1))
    visible_rows = _LIBRARY_VISIBLE_ROWS
    if selected < offset:
        offset = selected
    if selected >= offset + visible_rows:
        offset = selected - visible_rows + 1
    offset = max(0, min(offset, max(0, len(items) - visible_rows)))
    return selected, offset


def _draw_library_scene(
    surf: pygame.Surface,
    *,
    root: Path,
    items: list[ReplayListItem],
    selected: int,
    offset: int,
    edit_kind: str | None,
    edit_text: str,
    font: pygame.font.Font,
    small: pygame.font.Font,
) -> None:
    surf.fill(BG_TOP)
    for y in range(WIN_H):
        t = y / max(1, WIN_H - 1)
        color = (
            int(BG_TOP[0] * (1.0 - t) + BG_BOTTOM[0] * t),
            int(BG_TOP[1] * (1.0 - t) + BG_BOTTOM[1] * t),
            int(BG_TOP[2] * (1.0 - t) + BG_BOTTOM[2] * t),
        )
        pygame.draw.line(surf, color, (0, y), (WIN_W, y))
    margin = 28
    x = margin
    y = 22
    surf.blit(font.render("BIBLIOTECA DE REPLAYS", True, HUD_TITLE), (x, y))
    root_label = _ellipsize_text(small, str(root), 300)
    count = small.render(f"{len(items)} partidas | {root_label}", True, TEXT_DIM)
    surf.blit(count, (WIN_W - margin - count.get_width(), y + 4))
    y += font.get_linesize() + 8
    pygame.draw.line(surf, PANEL_BORDER, (x, y), (WIN_W - margin, y), width=1)
    y += 14
    if not items:
        surf.blit(small.render("No hay replays en esta carpeta.", True, TEXT_DIM), (x, y))
        return

    row_h = 68
    visible = items[offset : offset + _LIBRARY_VISIBLE_ROWS]
    for row_idx, item in enumerate(visible):
        item_idx = offset + row_idx
        row_top = y + row_idx * row_h
        row_rect = pygame.Rect(x, row_top, WIN_W - margin * 2, row_h - 8)
        if item_idx == selected:
            pygame.draw.rect(surf, HUD_VALUE_BAR_BG, row_rect)
            pygame.draw.rect(surf, HUD_HISTORY_TEXT, row_rect, width=2)
        else:
            pygame.draw.rect(surf, PANEL_BORDER, row_rect, width=1)
        left_x = row_rect.left + 12
        right_x = row_rect.left + 360
        right_w = row_rect.right - right_x - 12
        date_surf = small.render(item.date_label, True, HUD_HISTORY_TEXT)
        surf.blit(date_surf, (left_x, row_rect.top + 8))
        title_color = TEXT_MAIN if item_idx == selected else HUD_TITLE
        title = _ellipsize_text(font, item.title, right_w)
        surf.blit(font.render(title, True, title_color), (right_x, row_rect.top + 5))
        event = item.metadata.get("event")
        if isinstance(event, str) and event.strip() and event.strip() != item.title:
            subtitle = f"{event.strip()} | {item.subtitle}"
        else:
            subtitle = item.subtitle
        quality_tag = str(item.metadata.get("quality_tag", "")).strip()
        if quality_tag != "":
            subtitle = f"{quality_tag} | {subtitle}"
        subtitle = _ellipsize_text(small, subtitle, right_w)
        surf.blit(small.render(subtitle, True, TEXT_DIM), (right_x, row_rect.top + 37))
        rel = item.npz_path.as_posix()
        rel = _ellipsize_text(small, rel, 320)
        surf.blit(small.render(rel, True, TEXT_DIM), (left_x, row_rect.top + 36))

    help_text = "Enter ver | n apodo | e evento | t calidad | r recargar | q salir"
    surf.blit(small.render(help_text, True, TEXT_DIM), (x, WIN_H - 34))
    if edit_kind is not None:
        _draw_text_editor(surf, kind=edit_kind, text=edit_text, font=font, small=small)


def _scaled_window_size() -> tuple[int, int]:
    info = pygame.display.Info()
    avail_w = max(640, info.current_w - 60)
    avail_h = max(480, info.current_h - 160)
    display_scale = min(avail_w / WIN_W, avail_h / WIN_H, 1.0)
    return int(WIN_W * display_scale), int(WIN_H * display_scale)


def _flip_scene(scene: pygame.Surface, window: pygame.Surface) -> None:
    target = pygame.display.get_surface() or window
    tw, th = target.get_size()
    sw, sh = scene.get_size()
    scale = min(tw / sw, th / sh)
    out_w = int(sw * scale)
    out_h = int(sh * scale)
    scaled = pygame.transform.smoothscale(scene, (out_w, out_h))
    target.fill((0, 0, 0))
    target.blit(scaled, ((tw - out_w) // 2, (th - out_h) // 2))
    pygame.display.flip()


def _ellipsize_text(font: pygame.font.Font, text: str, max_width: int) -> str:
    if font.size(text)[0] <= max_width:
        return text
    suffix = "..."
    suffix_w = font.size(suffix)[0]
    if suffix_w >= max_width:
        return suffix
    trimmed = text
    while trimmed and font.size(trimmed + suffix)[0] > max_width:
        trimmed = trimmed[:-1]
    return trimmed + suffix


def _draw_text_editor(
    surf: pygame.Surface,
    *,
    kind: str,
    text: str,
    font: pygame.font.Font,
    small: pygame.font.Font,
) -> None:
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surf.blit(overlay, (0, 0))
    box = pygame.Rect(220, 250, 840, 170)
    pygame.draw.rect(surf, PANEL_BG, box)
    pygame.draw.rect(surf, HUD_HISTORY_TEXT, box, width=2)
    label = "APODO" if kind == "nickname" else "ETIQUETA / EVENTO"
    surf.blit(font.render(label, True, HUD_TITLE), (box.left + 24, box.top + 18))
    input_rect = pygame.Rect(box.left + 24, box.top + 72, box.width - 48, 44)
    pygame.draw.rect(surf, HUD_VALUE_BAR_BG, input_rect)
    pygame.draw.rect(surf, PANEL_BORDER, input_rect, width=1)
    shown = text[-70:]
    surf.blit(font.render(shown + "_", True, TEXT_MAIN), (input_rect.left + 10, input_rect.top + 4))
    hint = "Enter guardar | Esc cancelar | dejar vacio borra el campo"
    surf.blit(small.render(hint, True, TEXT_DIM), (box.left + 24, box.bottom - 34))


def _draw_replay_scene(
    surf: pygame.Surface,
    *,
    replay: LoadedReplay,
    frame_idx: int,
    playing: bool,
    speed: float,
    anim_start_ms: int | None,
    font: pygame.font.Font,
    small: pygame.font.Font,
    big: pygame.font.Font,
) -> None:
    frame = replay.frames[frame_idx]
    from ui.arena.render import draw_arena

    render_frame = replay_render_frame(
        frame=frame,
        playing=playing,
        anim_start_ms=anim_start_ms,
        now_ms=pygame.time.get_ticks(),
        speed=speed,
    )
    arena_state = _arena_state_for_replay(replay=replay, frame_idx=frame_idx, speed=speed)
    draw_arena(
        screen=surf,
        font=font,
        small=small,
        big=big,
        board=render_frame.board,
        selected=None,
        legal_targets=[],
        legal_target_kind={},
        hover_cell=None,
        hover_targets=[],
        hover_target_kind={},
        p1_agent=str(replay.metadata.get("p1_label", "ROJO")),
        p2_agent=str(replay.metadata.get("p2_label", "AZUL")),
        turn_agent="replay",
        status="REPLAY: space play/pausa  |  flechas paso  |  Esc biblioteca",
        p1_level="-",
        p2_level="-",
        recent=render_frame.move_cells + render_frame.infect_cells,
        move_cells=render_frame.move_cells,
        infect_cells=render_frame.infect_cells,
        infection_hidden=render_frame.infection_hidden,
        preview_move=render_frame.preview_move,
        preview_started_at=render_frame.preview_started_at,
        preview_until=render_frame.preview_until,
        now_ms=render_frame.now_ms,
        move_until=render_frame.move_until,
        infect_until=render_frame.infect_until,
        shake_offset=(0, 0),
        flash_start=0,
        flash_until=0,
        flash_color=FLASH_INFECT,
        piece_pop={},
        particles=[],
        intro_start=0,
        intro_until=0,
        game_over_started=None,
        final_counts=None,
        arena_state=arena_state,
    )
    _draw_replay_hud(
        surf,
        replay=replay,
        frame=frame,
        frame_idx=frame_idx,
        playing=playing,
        speed=speed,
        font=font,
        small=small,
    )


def _board_from_replay_frame(frame: ReplayFrame) -> AtaxxBoard:
    board = AtaxxBoard(grid=frame.grid.copy(), player=frame.player)
    board.half_moves = int(frame.index)
    return board


def replay_animation_duration_ms(frame: ReplayFrame) -> int:
    if frame.is_final or frame.move is None:
        return max(260, MOVE_PREVIEW_MS_AI)
    return MOVE_PREVIEW_MS_AI + max(MOVE_PULSE_MS, INFECT_PULSE_MS) + 160


def active_animation_done(
    *,
    frame: ReplayFrame,
    anim_start_ms: int | None,
    now_ms: int,
    speed: float,
) -> bool:
    if anim_start_ms is None:
        return False
    elapsed = int((now_ms - anim_start_ms) * max(0.25, speed))
    return elapsed >= replay_animation_duration_ms(frame)


def replay_render_frame(
    *,
    frame: ReplayFrame,
    playing: bool,
    anim_start_ms: int | None,
    now_ms: int,
    speed: float,
) -> ReplayRenderFrame:
    board = _board_from_replay_frame(frame)
    if not playing or anim_start_ms is None or frame.is_final or frame.move is None:
        return ReplayRenderFrame(
            board=board,
            preview_move=None if frame.is_final else frame.move,
            preview_started_at=None,
            preview_until=0,
            now_ms=now_ms,
            move_cells=[],
            infect_cells=[],
            infection_hidden={},
            move_until=0,
            infect_until=0,
        )

    scaled_elapsed = int((now_ms - anim_start_ms) * max(0.25, speed))
    render_now = anim_start_ms + scaled_elapsed
    preview_end = anim_start_ms + MOVE_PREVIEW_MS_AI
    if scaled_elapsed < MOVE_PREVIEW_MS_AI:
        return ReplayRenderFrame(
            board=board,
            preview_move=frame.move,
            preview_started_at=anim_start_ms,
            preview_until=preview_end,
            now_ms=render_now,
            move_cells=[],
            infect_cells=[],
            infection_hidden={},
            move_until=0,
            infect_until=0,
        )

    changed, move_cells, infect_cells, old_vals, destination = apply_move_with_feedback(
        board,
        frame.move,
    )
    reveal_schedule = infection_wave_schedule(
        destination=destination,
        infection_cells=infect_cells,
        start_ms=preview_end + 40,
    )
    infection_hidden = {
        cell: (reveal_schedule.get(cell, preview_end), old_val)
        for cell, old_val in old_vals.items()
        if render_now < reveal_schedule.get(cell, preview_end)
    }
    return ReplayRenderFrame(
        board=board,
        preview_move=None,
        preview_started_at=None,
        preview_until=0,
        now_ms=render_now,
        move_cells=move_cells or changed,
        infect_cells=infect_cells,
        infection_hidden=infection_hidden,
        move_until=preview_end + MOVE_PULSE_MS,
        infect_until=preview_end + INFECT_PULSE_MS,
    )


def _arena_state_for_replay(
    *,
    replay: LoadedReplay,
    frame_idx: int,
    speed: float,
) -> dict[str, object]:
    frame = replay.frames[frame_idx]
    metadata = replay.metadata
    history = [
        (item.player, item.move)
        for item in replay.frames[:frame_idx]
        if not item.is_final
    ]
    eval_history = [
        float(item.value if item.player == PLAYER_1 else -item.value)
        for item in replay.frames[: frame_idx + 1]
        if not item.is_final
    ]
    subtitle = (
        f"{frame_idx + 1}/{len(replay.frames)}  "
        f"{metadata.get('p1_label', 'ROJO')} vs {metadata.get('p2_label', 'AZUL')}"
    )
    return {
        "last_top_moves": [],
        "last_root_value": eval_history[-1] if eval_history else 0.0,
        "last_thinker": int(frame.player),
        "eval_history": eval_history,
        "move_history": history,
        "speed_mult": float(speed),
        "paused": True,
        "brain_tab": 0,
        "tournament_banner": ("REPLAY", subtitle),
    }


def _draw_replay_hud(
    surf: pygame.Surface,
    *,
    replay: LoadedReplay,
    frame: ReplayFrame,
    frame_idx: int,
    playing: bool,
    speed: float,
    font: pygame.font.Font,
    small: pygame.font.Font,
) -> None:
    rect = pygame.Rect(HUD_X, 0, WIN_W - HUD_X, WIN_H)
    pygame.draw.rect(surf, PANEL_BG, rect)
    pygame.draw.line(surf, PANEL_BORDER, (rect.left, 0), (rect.left, WIN_H), width=2)
    x = rect.left + 16
    y = 16
    line_h = small.get_linesize()
    surf.blit(font.render("REPLAY", True, HUD_TITLE), (x, y))
    state = "PLAY" if playing else "PAUSA"
    state_surf = small.render(f"{state}  {speed:g}x", True, HUD_HISTORY_TEXT if playing else TEXT_DIM)
    surf.blit(state_surf, (rect.right - 16 - state_surf.get_width(), y + 4))
    y += font.get_linesize() + 10
    pygame.draw.line(surf, PANEL_BORDER, (x, y), (rect.right - 16, y), width=1)
    y += 14

    meta = replay.metadata
    rows = [
        ("Archivo", replay.path.name),
        ("Modo", str(meta.get("mode", "?"))),
        ("P1", str(meta.get("p1_label", "ROJO"))),
        ("P2", str(meta.get("p2_label", "AZUL"))),
        ("Turno", f"{min(frame_idx + 1, len(replay.frames))}/{len(replay.frames)}"),
        ("Mueve", "ROJO" if frame.player == PLAYER_1 else "AZUL"),
        ("Jugada", "final" if frame.is_final else move_notation(frame.move)),
        ("Policy", f"{frame.policy_prob * 100:4.1f}%"),
        ("Target", f"{frame.value:+.2f}"),
    ]
    for label, value in rows:
        color = PIECE_P1 if value == "ROJO" else (PIECE_P2 if value == "AZUL" else TEXT_MAIN)
        surf.blit(small.render(f"{label:<7}: ", True, TEXT_DIM), (x, y))
        surf.blit(small.render(value, True, color), (x + 96, y))
        y += line_h
    y += 8
    if "winner" in meta:
        winner = int(meta.get("winner", 0))
        winner_text = "ROJO" if winner == PLAYER_1 else ("AZUL" if winner == PLAYER_2 else "EMPATE")
        surf.blit(small.render(f"Ganador : {winner_text}", True, HUD_HISTORY_TEXT), (x, y))
        y += line_h
    progress_w = rect.width - 32
    progress = frame_idx / max(1, len(replay.frames) - 1)
    bar_rect = pygame.Rect(x, y + 10, progress_w, 12)
    pygame.draw.rect(surf, HUD_VALUE_BAR_BG, bar_rect)
    fill = pygame.Rect(bar_rect.left, bar_rect.top, int(bar_rect.width * progress), bar_rect.height)
    pygame.draw.rect(surf, HUD_HISTORY_TEXT, fill)
    pygame.draw.rect(surf, PANEL_BORDER, bar_rect, width=1)
    y += 42
    controls = [
        "space play/pausa",
        "left/right o a/s: paso",
        "home/end: inicio/final",
        "1/2/4 velocidad",
        "p screenshot   q salir",
    ]
    for text in controls:
        surf.blit(small.render(text, True, TEXT_DIM), (x, y))
        y += line_h


__all__ = [
    "LoadedReplay",
    "ReplayFrame",
    "ReplayListItem",
    "ReplayPaths",
    "ReplayRenderFrame",
    "action_from_policy",
    "active_animation_done",
    "cycle_quality_tag",
    "discover_replays",
    "grid_from_observation",
    "load_replay",
    "player_for_turn",
    "replay_animation_duration_ms",
    "replay_render_frame",
    "resolve_replay_paths",
    "run_replay_library",
    "run_replay_viewer",
    "spanish_date_label",
    "update_replay_metadata",
]

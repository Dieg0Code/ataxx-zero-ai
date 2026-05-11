from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pygame
import pygame.sndarray
import torch

# Ensure `src/` is importable when running as:
# `uv run python scripts/play_pygame.py ...`
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.heuristic import HEURISTIC_LEVELS  # noqa: E402
from agents.registry import (  # noqa: E402
    agent_kind,
    canonicalize_agent_spec,
    heuristic_level_for_agent,
)
from ui.arena.layout import BOARD_PX, CELL, PAD, WIN_H, WIN_W  # noqa: E402
from ui.arena.model_runtime import (  # noqa: E402
    build_model_mcts_by_player,
    resolve_model_checkpoints,
)
from ui.arena.theme import (  # noqa: E402
    AI_DELAY_JITTER,
    AI_DELAY_MAX,
    AI_DELAY_MIN,
    FLASH_INFECT,
    FLASH_INFECT_MS,
    FLASH_MOVE,
    FLASH_MOVE_MS,
    HIGHLIGHT_MS,
    INFECT_PULSE_MS,
    INFECTION_STEP_MS,
    INTRO_STEP_MS,
    INTRO_STEPS,
    MOVE_PREVIEW_MS_AI,
    MOVE_PREVIEW_MS_HUMAN,
    MOVE_PULSE_MS,
    PARTICLE_LIFE_MS,
    PARTICLE_MAX_COUNT,
    PIECE_POP_MS,
    SHAKE_INFECT_MS,
    SHAKE_INFECT_PX,
    SHAKE_MOVE_MS,
    SHAKE_MOVE_PX,
)

if TYPE_CHECKING:
    from agents.types import Agent
    from game.board import AtaxxBoard
    from game.types import Move
    from model.system import AtaxxZero
    from ui.arena.effects import Particle

PLAYER_1 = 1
PLAYER_2 = -1

def _ensure_src_on_path() -> None:
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ataxx arena (play/spectate).")
    parser.add_argument("--mode", default="play", choices=["play", "spectate"])
    parser.add_argument("--checkpoint", "--ckpt", default="")
    parser.add_argument("--p1-checkpoint", "--ckpt1", default="")
    parser.add_argument("--p2-checkpoint", "--ckpt2", default="")
    parser.add_argument("--opponent", "--opp", default="heuristic")
    parser.add_argument("--human-player", "--human-side", default="p1", choices=["p1", "p2"])
    parser.add_argument("--p1-agent", "--agent1", default="")
    parser.add_argument("--p2-agent", "--agent2", default="")
    parser.add_argument("--heuristic-level", "--level", default="normal", choices=list(HEURISTIC_LEVELS))
    parser.add_argument("--p1-level", "--level1", default="", choices=["", *list(HEURISTIC_LEVELS)])
    parser.add_argument("--p2-level", "--level2", default="", choices=["", *list(HEURISTIC_LEVELS)])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mcts-sims", "--sims", type=int, default=160)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="RNG seed. Use -1 for non-deterministic runs.",
    )
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return device


def _resolve_agents(args: argparse.Namespace) -> tuple[Agent, Agent]:
    if args.p1_agent or args.p2_agent:
        p1 = args.p1_agent or "heuristic"
        p2 = args.p2_agent or "heuristic"
    elif args.mode == "spectate":
        p1, p2 = "heuristic", "heuristic"
    elif args.human_player == "p1":
        p1, p2 = "human", args.opponent
    else:
        p1, p2 = args.opponent, "human"

    p1 = canonicalize_agent_spec(
        p1,
        default_heuristic_level=args.p1_level or args.heuristic_level,
    )
    p2 = canonicalize_agent_spec(
        p2,
        default_heuristic_level=args.p2_level or args.heuristic_level,
    )

    if args.mode == "spectate" and (
        agent_kind(p1, default_heuristic_level=args.heuristic_level) == "human"
        or agent_kind(p2, default_heuristic_level=args.heuristic_level) == "human"
    ):
        raise ValueError("spectate mode requires AI vs AI")
    if args.mode == "play" and (
        agent_kind(p1, default_heuristic_level=args.heuristic_level) != "human"
        and agent_kind(p2, default_heuristic_level=args.heuristic_level) != "human"
    ):
        raise ValueError("play mode requires at least one human")
    return p1, p2


def _resolve_heuristic_levels(
    args: argparse.Namespace,
    p1_agent: Agent,
    p2_agent: Agent,
) -> tuple[str, str]:
    p1_level = heuristic_level_for_agent(
        p1_agent,
        default_heuristic_level=args.p1_level or args.heuristic_level,
    )
    p2_level = heuristic_level_for_agent(
        p2_agent,
        default_heuristic_level=args.p2_level or args.heuristic_level,
    )
    return p1_level or "-", p2_level or "-"


def _load_system(checkpoint_path: str, device: str) -> AtaxxZero:
    from model.checkpoint_compat import (
        adapt_state_dict_observation_channels,
        drop_legacy_policy_head,
        extract_checkpoint_state_dict,
        extract_model_kwargs,
        has_legacy_flat_policy_head,
    )
    from model.system import AtaxxZero

    system = AtaxxZero()
    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format")
        state_dict_obj = extract_checkpoint_state_dict(checkpoint)
        system = AtaxxZero(**extract_model_kwargs(checkpoint))
        adapted = adapt_state_dict_observation_channels(
            state_dict_obj,
            target_channels=int(system.model.num_input_channels),
        )
        if has_legacy_flat_policy_head(adapted):
            print(
                "[checkpoint_compat] legacy flat policy_head detected; "
                "loading encoder+value only, policy stays randomly initialized."
            )
            adapted = drop_legacy_policy_head(adapted)
            system.load_state_dict(adapted, strict=False)
        else:
            try:
                system.load_state_dict(adapted)
            except RuntimeError as exc:
                raise ValueError(
                    "Checkpoint incompatible con architecture policy_head espacial; "
                    "reentrena o usa carga parcial manual (strict=False)."
                ) from exc
    system.eval()
    system.to(device)
    return system


def _cell_from_pos(mx: int, my: int) -> tuple[int, int] | None:
    bx = mx - PAD
    by = my - PAD
    if bx < 0 or by < 0 or bx >= BOARD_PX or by >= BOARD_PX:
        return None
    return int(by // CELL), int(bx // CELL)


def _moves_from_origin(board: AtaxxBoard, origin: tuple[int, int]) -> list[Move]:
    r0, c0 = origin
    return [m for m in board.get_valid_moves() if m[0] == r0 and m[1] == c0]


def _targets_and_kinds(moves: list[Move]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], str]]:
    targets: list[tuple[int, int]] = []
    kinds: dict[tuple[int, int], str] = {}
    for r1, c1, r2, c2 in moves:
        target = (r2, c2)
        targets.append(target)
        dist = max(abs(r1 - r2), abs(c1 - c2))
        kinds[target] = "clone" if dist == 1 else "jump"
    return targets, kinds


def _counts(board: AtaxxBoard) -> tuple[int, int]:
    return int(np.sum(board.grid == PLAYER_1)), int(np.sum(board.grid == PLAYER_2))


def _result_text(board: AtaxxBoard) -> str:
    result = board.get_result()
    if result == 1:
        return "Gana: P1 (ROJO)"
    if result == -1:
        return "Gana: P2 (AZUL)"
    return "Empate"

def _ai_delay_ms(board: AtaxxBoard, agent: Agent, sims: int, rng: np.random.Generator) -> int:
    valid_count = len(board.get_valid_moves())
    complexity = min(260, valid_count * 9)
    model_bonus = 120 if agent == "model" else 0
    sims_bonus = min(180, sims // 2) if agent == "model" else 0
    jitter = int(rng.integers(0, AI_DELAY_JITTER + 1))
    delay = AI_DELAY_MIN + complexity + model_bonus + sims_bonus + jitter
    return int(max(AI_DELAY_MIN, min(AI_DELAY_MAX, delay)))


def _changed_cells(before: np.ndarray, after: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(before != after)
    return [(int(r), int(c)) for r, c in coords]


def _apply_move_with_feedback(
    board: AtaxxBoard,
    move: Move | None,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    dict[tuple[int, int], int],
    tuple[int, int] | None,
]:
    before = board.grid.copy()
    player = board.current_player
    board.step(move)
    changed = _changed_cells(before, board.grid)

    move_cells: list[tuple[int, int]] = []
    infect_cells: list[tuple[int, int]] = []
    old_infection_values: dict[tuple[int, int], int] = {}
    destination: tuple[int, int] | None = None
    if move is not None:
        r1, c1, r2, c2 = move
        destination = (r2, c2)
        move_cells.append((r2, c2))
        if max(abs(r1 - r2), abs(c1 - c2)) == 2:
            move_cells.append((r1, c1))
        move_set = set(move_cells)
        for rr, cc in changed:
            if (rr, cc) in move_set:
                continue
            if int(before[rr, cc]) == -player and int(board.grid[rr, cc]) == player:
                infect_cells.append((rr, cc))
                old_infection_values[(rr, cc)] = int(before[rr, cc])
    return changed, move_cells, infect_cells, old_infection_values, destination


def _pick_ai_move(
    board: AtaxxBoard,
    agent: Agent,
    rng: np.random.Generator,
    heuristic_level: str,
    mcts: object | None,
) -> tuple[Move | None, str, dict[str, object] | None]:
    """Pick a move plus optional MCTS diagnostics for the HUD.

    Returns (move, status_text, diagnostics). `diagnostics` is a dict with
    `top_moves` and `root_value` for model agents, otherwise None.
    """
    from agents.registry import agent_kind as _agent_kind
    from agents.selector import pick_ai_move

    if _agent_kind(agent) == "model":
        from agents.model_agent import model_move_with_diagnostics

        move, diagnostics = model_move_with_diagnostics(board=board, mcts=mcts)
        return move, "Model AI move played", diagnostics
    move, text = pick_ai_move(
        board=board,
        agent=agent,
        rng=rng,
        heuristic_level=heuristic_level,
        mcts=mcts,
    )
    return move, text, None


def _infection_wave_schedule(
    destination: tuple[int, int] | None,
    infection_cells: list[tuple[int, int]],
    start_ms: int,
) -> dict[tuple[int, int], int]:
    if destination is None or len(infection_cells) == 0:
        return {}
    dr, dc = destination

    def sort_key(cell: tuple[int, int]) -> tuple[float, int]:
        rr, cc = cell
        angle = np.arctan2(rr - dr, cc - dc)
        radius = abs(rr - dr) + abs(cc - dc)
        return (float(angle), radius)

    ordered = sorted(infection_cells, key=sort_key)
    return {
        cell: start_ms + (idx * INFECTION_STEP_MS)
        for idx, cell in enumerate(ordered)
    }


def _make_tone(freq_hz: float, duration_ms: int, volume: float = 0.12) -> pygame.mixer.Sound | None:
    from ui.arena.effects import make_tone

    return make_tone(freq_hz=freq_hz, duration_ms=duration_ms, volume=volume)


def _build_sfx() -> dict[str, pygame.mixer.Sound | None]:
    from ui.arena.effects import build_sfx

    return build_sfx()


def _play_sfx(sfx: dict[str, pygame.mixer.Sound | None], key: str) -> None:
    from ui.arena.effects import play_sfx

    play_sfx(sfx=sfx, key=key)


def _spawn_particles(
    rng: np.random.Generator,
    particles: list[Particle],
    cell: tuple[int, int],
    color: tuple[int, int, int],
    now_ms: int,
    intensity: int,
) -> None:
    from ui.arena.effects import spawn_particles

    spawn_particles(
        rng=rng,
        particles=particles,
        cell=cell,
        color=color,
        now_ms=now_ms,
        intensity=intensity,
        pad=PAD,
        cell_px=CELL,
        particle_max_count=PARTICLE_MAX_COUNT,
        particle_life_ms=PARTICLE_LIFE_MS,
    )


def _wrap_text_line(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    from ui.arena.effects import wrap_text_line

    return wrap_text_line(font=font, text=text, max_width=max_width)


def _draw(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small: pygame.font.Font,
    big: pygame.font.Font,
    board: AtaxxBoard,
    selected: tuple[int, int] | None,
    legal_targets: list[tuple[int, int]],
    legal_target_kind: dict[tuple[int, int], str],
    hover_cell: tuple[int, int] | None,
    hover_targets: list[tuple[int, int]],
    hover_target_kind: dict[tuple[int, int], str],
    p1_agent: Agent,
    p2_agent: Agent,
    turn_agent: Agent,
    status: str,
    p1_level: str,
    p2_level: str,
    recent: list[tuple[int, int]],
    move_cells: list[tuple[int, int]],
    infect_cells: list[tuple[int, int]],
    infection_hidden: dict[tuple[int, int], tuple[int, int]],
    preview_move: Move | None,
    preview_started_at: int | None,
    preview_until: int,
    now_ms: int,
    move_until: int,
    infect_until: int,
    shake_offset: tuple[int, int],
    flash_start: int,
    flash_until: int,
    flash_color: tuple[int, int, int],
    piece_pop: dict[tuple[int, int], tuple[int, int]],
    particles: list[Particle],
    intro_start: int,
    intro_until: int,
    game_over_started: int | None,
    final_counts: tuple[int, int] | None,
    arena_state: dict[str, object] | None = None,
) -> None:
    from ui.arena.render import draw_arena

    draw_arena(
        screen=screen,
        font=font,
        small=small,
        big=big,
        board=board,
        selected=selected,
        legal_targets=legal_targets,
        legal_target_kind=legal_target_kind,
        hover_cell=hover_cell,
        hover_targets=hover_targets,
        hover_target_kind=hover_target_kind,
        p1_agent=p1_agent,
        p2_agent=p2_agent,
        turn_agent=turn_agent,
        status=status,
        p1_level=p1_level,
        p2_level=p2_level,
        recent=recent,
        move_cells=move_cells,
        infect_cells=infect_cells,
        infection_hidden=infection_hidden,
        preview_move=preview_move,
        preview_started_at=preview_started_at,
        preview_until=preview_until,
        now_ms=now_ms,
        move_until=move_until,
        infect_until=infect_until,
        shake_offset=shake_offset,
        flash_start=flash_start,
        flash_until=flash_until,
        flash_color=flash_color,
        piece_pop=piece_pop,
        particles=particles,
        intro_start=intro_start,
        intro_until=intro_until,
        game_over_started=game_over_started,
        final_counts=final_counts,
        arena_state=arena_state,
    )


def main() -> None:
    _ensure_src_on_path()
    from game.board import AtaxxBoard

    args = _parse_args()
    device = _resolve_device(args.device)
    p1_agent, p2_agent = _resolve_agents(args)
    p1_level, p2_level = _resolve_heuristic_levels(args, p1_agent, p2_agent)
    rng = np.random.default_rng(seed=None if args.seed < 0 else args.seed)

    model_mcts_by_player = {PLAYER_1: None, PLAYER_2: None}
    if agent_kind(p1_agent, default_heuristic_level=args.heuristic_level) == "model" or agent_kind(
        p2_agent,
        default_heuristic_level=args.heuristic_level,
    ) == "model":
        from model.registry import resolve as resolve_codename

        def _resolve_or_blank(name: str) -> str:
            if not name:
                return ""
            try:
                return str(resolve_codename(name))
            except (FileNotFoundError, ValueError):
                return name

        checkpoints_by_player = resolve_model_checkpoints(
            shared_checkpoint=_resolve_or_blank(args.checkpoint),
            p1_agent=p1_agent,
            p2_agent=p2_agent,
            p1_checkpoint=_resolve_or_blank(args.p1_checkpoint),
            p2_checkpoint=_resolve_or_blank(args.p2_checkpoint),
        )
        model_mcts_by_player = build_model_mcts_by_player(
            checkpoints_by_player=checkpoints_by_player,
            device=device,
            c_puct=args.c_puct,
            n_simulations=args.mcts_sims,
            load_system=_load_system,
        )

    pygame.init()
    info = pygame.display.Info()
    # Leave room for title bar + top/bottom taskbars on small displays.
    avail_w = max(640, info.current_w - 60)
    avail_h = max(480, info.current_h - 160)
    display_scale = min(avail_w / WIN_W, avail_h / WIN_H, 1.0)
    win_size = (int(WIN_W * display_scale), int(WIN_H * display_scale))
    window = pygame.display.set_mode(win_size)
    pygame.display.set_caption("Ataxx Arena")
    screen = pygame.Surface((WIN_W, WIN_H)) if display_scale < 1.0 else window

    def _virtual_mouse_pos() -> tuple[int, int]:
        mx, my = pygame.mouse.get_pos()
        if display_scale >= 1.0:
            return mx, my
        return int(mx / display_scale), int(my / display_scale)
    from ui.arena.fonts import load_arena_fonts
    font, small, big = load_arena_fonts()
    clock = pygame.time.Clock()

    board = AtaxxBoard()
    selected: tuple[int, int] | None = None
    legal_targets: list[tuple[int, int]] = []
    legal_target_kind: dict[tuple[int, int], str] = {}
    hover_targets: list[tuple[int, int]] = []
    hover_target_kind: dict[tuple[int, int], str] = {}
    hover_cell: tuple[int, int] | None = None
    status = "Ready"
    recent: list[tuple[int, int]] = []
    recent_until = 0
    move_cells: list[tuple[int, int]] = []
    move_until = 0
    infect_cells: list[tuple[int, int]] = []
    infect_until = 0
    infection_hidden: dict[tuple[int, int], tuple[int, int]] = {}
    preview_move: Move | None = None
    preview_started_at: int | None = None
    preview_until = 0
    pending_move: Move | None = None
    pending_apply_at: int | None = None
    ai_ready_at: dict[int, int | None] = {PLAYER_1: None, PLAYER_2: None}
    shake_start = 0
    shake_until = 0
    shake_max_px = 0.0
    flash_start = 0
    flash_until = 0
    flash_color = FLASH_MOVE
    piece_pop: dict[tuple[int, int], tuple[int, int]] = {}
    particles: list[Particle] = []
    sfx = _build_sfx()
    intro_start = pygame.time.get_ticks()
    intro_until = intro_start + (INTRO_STEP_MS * len(INTRO_STEPS))
    game_over_started: int | None = None
    final_counts: tuple[int, int] | None = None

    # Diagnostics consumed by the HUD (Fase 0 plumbing). `last_top_moves` is
    # populated only after a model agent moves; human/heuristic/random leave
    # the previous values intact so the panel doesn't blink to empty.
    arena_state: dict[str, object] = {
        "last_top_moves": [],
        "last_root_value": 0.0,
        "last_thinker": None,
        "eval_history": [],
        "move_history": [],
        "speed_mult": 1.0,
        "paused": False,
        "stats_record": {"w": 0, "l": 0, "d": 0},
    }
    speed_mult = 1.0
    paused = False
    step_once = False
    stats_recorded = False
    p1_label = f"{p1_agent}({p1_level})" if p1_level != "-" else str(p1_agent)
    p2_label = f"{p2_agent}({p2_level})" if p2_level != "-" else str(p2_agent)
    from ui.arena import stats as arena_stats

    arena_state["stats_record"] = arena_stats.get_record(p1_label, p2_label)
    screenshots_dir = Path("arena_screenshots")

    running = True
    while running:
        now_ms = pygame.time.get_ticks()
        if recent and now_ms >= recent_until:
            recent = []
        if particles:
            particles = [p for p in particles if now_ms < float(p["end"])]
        if piece_pop:
            piece_pop = {cell: span for cell, span in piece_pop.items() if now_ms < span[1]}
        if infection_hidden:
            revealed_cells: list[tuple[int, int]] = [
                cell for cell, data in infection_hidden.items() if now_ms >= data[0]
            ]
            infection_hidden = {
                cell: data
                for cell, data in infection_hidden.items()
                if now_ms < data[0]
            }
            for cell in revealed_cells:
                piece_pop[cell] = (now_ms, now_ms + PIECE_POP_MS)
                _spawn_particles(rng, particles, cell, FLASH_INFECT, now_ms, intensity=8)

        turn_agent = p1_agent if board.current_player == PLAYER_1 else p2_agent
        intro_active = now_ms < intro_until
        if board.is_game_over() and game_over_started is None:
            game_over_started = now_ms
            final_counts = _counts(board)
            if not stats_recorded:
                arena_state["stats_record"] = arena_stats.record_result(
                    p1_label, p2_label, int(board.get_result()),
                )
                stats_recorded = True
        human_turn = turn_agent == "human" and not board.is_game_over() and not intro_active
        ai_turn = turn_agent != "human" and not board.is_game_over() and not intro_active
        hover_cell = None
        hover_targets = []
        hover_target_kind = {}
        if human_turn and pending_apply_at is None:
            hover_cell = _cell_from_pos(*_virtual_mouse_pos())
            if selected is None and hover_cell is not None:
                hr, hc = hover_cell
                if int(board.grid[hr, hc]) == board.current_player:
                    hmoves = _moves_from_origin(board, hover_cell)
                    hover_targets, hover_target_kind = _targets_and_kinds(hmoves)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    arena_state["paused"] = paused
                elif event.key == pygame.K_s and paused:
                    step_once = True
                elif event.key == pygame.K_1:
                    speed_mult = 1.0
                    arena_state["speed_mult"] = speed_mult
                elif event.key == pygame.K_2:
                    speed_mult = 2.0
                    arena_state["speed_mult"] = speed_mult
                elif event.key == pygame.K_4:
                    speed_mult = 4.0
                    arena_state["speed_mult"] = speed_mult
                elif event.key == pygame.K_p:
                    screenshots_dir.mkdir(exist_ok=True)
                    fname = pygame.time.get_ticks()
                    out_path = screenshots_dir / f"arena_{fname}.png"
                    pygame.image.save(window if screen is not window else screen, str(out_path))
                    status = f"Captura guardada: {out_path}"
                elif event.key == pygame.K_r:
                    board = AtaxxBoard()
                    selected = None
                    legal_targets = []
                    legal_target_kind = {}
                    hover_targets = []
                    hover_target_kind = {}
                    hover_cell = None
                    status = "Partida reiniciada"
                    ai_ready_at = {PLAYER_1: None, PLAYER_2: None}
                    recent = []
                    move_cells = []
                    infect_cells = []
                    infection_hidden = {}
                    preview_move = None
                    preview_started_at = None
                    preview_until = 0
                    pending_move = None
                    pending_apply_at = None
                    shake_start = 0
                    shake_until = 0
                    shake_max_px = 0.0
                    flash_start = 0
                    flash_until = 0
                    flash_color = FLASH_MOVE
                    piece_pop = {}
                    particles = []
                    intro_start = now_ms
                    intro_until = intro_start + (INTRO_STEP_MS * len(INTRO_STEPS))
                    game_over_started = None
                    final_counts = None
                    stats_recorded = False
                    arena_state["last_top_moves"] = []
                    arena_state["eval_history"] = []
                    arena_state["move_history"] = []
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and human_turn
                and pending_apply_at is None
            ):
                cell = _cell_from_pos(*_virtual_mouse_pos())
                if cell is None:
                    continue
                row, col = cell
                if selected is None:
                    if int(board.grid[row, col]) == board.current_player:
                        selected = (row, col)
                        moves = _moves_from_origin(board, selected)
                        legal_targets, legal_target_kind = _targets_and_kinds(moves)
                else:
                    if (row, col) == selected:
                        selected = None
                        legal_targets = []
                        legal_target_kind = {}
                        continue
                    if int(board.grid[row, col]) == board.current_player:
                        selected = (row, col)
                        moves = _moves_from_origin(board, selected)
                        legal_targets, legal_target_kind = _targets_and_kinds(moves)
                        continue
                    candidate: Move | None = None
                    for move in _moves_from_origin(board, selected):
                        if move[2] == row and move[3] == col:
                            candidate = move
                            break
                    if candidate is not None:
                        selected = None
                        legal_targets = []
                        legal_target_kind = {}
                        pending_move = candidate
                        preview_started_at = now_ms
                        pending_apply_at = now_ms + MOVE_PREVIEW_MS_HUMAN
                        preview_move = candidate
                        preview_until = int(pending_apply_at)
                        status = "Preparando jugada humana..."
                        ai_ready_at = {PLAYER_1: None, PLAYER_2: None}

        if human_turn and not board.has_valid_moves() and pending_apply_at is None:
            pending_move = None
            preview_started_at = now_ms
            pending_apply_at = now_ms + MOVE_PREVIEW_MS_HUMAN
            preview_move = None
            preview_until = int(pending_apply_at)
            status = "Humano pasa turno..."
            ai_ready_at = {PLAYER_1: None, PLAYER_2: None}

        ai_gate_open = ai_turn and pending_apply_at is None and (not paused or step_once)
        if ai_gate_open:
            player = board.current_player
            ready_at = ai_ready_at[player]
            if ready_at is None:
                base_delay = _ai_delay_ms(board, turn_agent, args.mcts_sims, rng)
                ai_ready_at[player] = now_ms + max(40, int(base_delay / max(0.5, speed_mult)))
                status = f"{turn_agent} pensando..."
            elif now_ms >= ready_at:
                move, move_text, diagnostics = _pick_ai_move(
                    board=board,
                    agent=turn_agent,
                    rng=rng,
                    heuristic_level=p1_level if player == PLAYER_1 else p2_level,
                    mcts=model_mcts_by_player[player],
                )
                if diagnostics is not None:
                    arena_state["last_top_moves"] = list(diagnostics.get("top_moves", []))  # type: ignore[arg-type]
                    raw_value = float(cast(float, diagnostics.get("root_value", 0.0) or 0.0))
                    # Normalize to P1 (ROJO) perspective so the HUD reads
                    # intuitively: +1 = P1 winning, -1 = P2 winning.
                    arena_state["last_root_value"] = raw_value if player == PLAYER_1 else -raw_value
                    arena_state["last_thinker"] = player
                pending_move = move
                preview_started_at = now_ms
                preview_dur = max(40, int(MOVE_PREVIEW_MS_AI / max(0.5, speed_mult)))
                pending_apply_at = now_ms + preview_dur
                preview_move = move
                preview_until = int(pending_apply_at)
                status = move_text.replace("played", "queued")
                ai_ready_at = {PLAYER_1: None, PLAYER_2: None}
                step_once = False
            else:
                status = f"{turn_agent} pensando..."
        elif paused and ai_turn and pending_apply_at is None:
            status = "PAUSA  (espacio=reanudar, s=paso, 1/2/4=velocidad)"

        if pending_apply_at is not None and now_ms >= pending_apply_at:
            applied_move = pending_move
            applied_player = int(board.current_player)
            changed, move_cells, infect_cells, old_vals, destination = _apply_move_with_feedback(
                board,
                pending_move,
            )
            # Append turn entry to history. If no model thought happened this
            # turn, repeat the previous eval so the chart x-axis stays in sync.
            cast(list, arena_state["move_history"]).append((applied_player, applied_move))
            cast(list, arena_state["eval_history"]).append(
                float(cast(float, arena_state.get("last_root_value", 0.0) or 0.0)),
            )
            recent = changed
            recent_until = now_ms + HIGHLIGHT_MS
            move_until = now_ms + MOVE_PULSE_MS
            infect_until = now_ms + INFECT_PULSE_MS

            reveal_schedule = _infection_wave_schedule(
                destination=destination,
                infection_cells=infect_cells,
                start_ms=now_ms + 40,
            )
            infection_hidden = {
                cell: (reveal_schedule.get(cell, now_ms), old_val)
                for cell, old_val in old_vals.items()
            }

            pending_move = None
            pending_apply_at = None
            preview_move = None
            preview_started_at = None
            preview_until = 0
            status = "Jugada resuelta"

            has_infection = len(infect_cells) > 0
            shake_start = now_ms
            shake_until = now_ms + (SHAKE_INFECT_MS if has_infection else SHAKE_MOVE_MS)
            shake_max_px = SHAKE_INFECT_PX if has_infection else SHAKE_MOVE_PX
            flash_start = now_ms
            flash_until = now_ms + (FLASH_INFECT_MS if has_infection else FLASH_MOVE_MS)
            flash_color = FLASH_INFECT if has_infection else FLASH_MOVE
            _play_sfx(sfx, "infect" if has_infection else "move")

            for rr, cc in move_cells:
                if int(board.grid[rr, cc]) != 0:
                    piece_pop[(rr, cc)] = (now_ms, now_ms + PIECE_POP_MS)
                    _spawn_particles(rng, particles, (rr, cc), FLASH_MOVE, now_ms, intensity=10)

        shake_offset = (0, 0)
        if shake_until > now_ms:
            shake_progress = (now_ms - shake_start) / max(1, shake_until - shake_start)
            amp = shake_max_px * (1.0 - shake_progress)
            sx = int(np.sin(now_ms * 0.040) * amp)
            sy = int(np.cos(now_ms * 0.033) * (amp * 0.7))
            shake_offset = (sx, sy)

        _draw(
            screen=screen,
            font=font,
            small=small,
            big=big,
            board=board,
            selected=selected,
            legal_targets=legal_targets,
            legal_target_kind=legal_target_kind,
            hover_cell=hover_cell,
            hover_targets=hover_targets,
            hover_target_kind=hover_target_kind,
            p1_agent=p1_agent,
            p2_agent=p2_agent,
            turn_agent=turn_agent,
            status=status,
            p1_level=p1_level,
            p2_level=p2_level,
            recent=recent,
            move_cells=move_cells,
            infect_cells=infect_cells,
            infection_hidden=infection_hidden,
            preview_move=preview_move,
            preview_started_at=preview_started_at,
            preview_until=preview_until,
            now_ms=now_ms,
            move_until=move_until,
            infect_until=infect_until,
            shake_offset=shake_offset,
            flash_start=flash_start,
            flash_until=flash_until,
            flash_color=flash_color,
            piece_pop=piece_pop,
            particles=particles,
            intro_start=intro_start,
            intro_until=intro_until,
            game_over_started=game_over_started,
            final_counts=final_counts,
            arena_state=arena_state,
        )
        if screen is not window:
            pygame.transform.smoothscale(screen, win_size, window)
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()



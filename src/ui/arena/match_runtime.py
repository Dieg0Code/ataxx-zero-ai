"""Game loop reusable de una partida en la arena Pygame.

Extraido tal cual de `play_pygame_app.main()` para que tanto el modo play normal
como el modo torneo usen exactamente la misma maquinaria: misma intro 3-2-1-FIGHT,
mismas particulas, mismo sonido, mismo HUD tactico, mismas teclas.

La unica diferencia con la version inline anterior es que entra por parametros
explicitos (no `args` global) y expone dos hooks opcionales:
- `recorder`: si esta presente, se graba cada movimiento como training data.
- `overlay_fn`: si esta presente, se llama despues de `draw_arena` para dibujar
  capas externas (banner del torneo, marcador en vivo, lo que sea).

El caller controla el ciclo de vida de la pygame display: este modulo no llama
`pygame.init()`, `pygame.display.set_mode()` ni `pygame.quit()`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pygame

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.types import Move
from ui.arena.layout import BOARD_PX, CELL, PAD
from ui.arena.theme import (
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
    from ui.arena.effects import Particle
    from ui.arena.replay_recorder import ReplayRecorder

PLAYER_1 = 1
PLAYER_2 = -1


@dataclass
class MatchOutcome:
    """Resultado de una partida — lo usa el caller para reportar al usuario / state."""

    reason: str                   # "quit" | "completed" | "restart"
    board: AtaxxBoard
    final_counts: tuple[int, int] | None
    result: int | None            # 1, -1, 0, o None si reason="quit" antes del game over


# ----- Helpers compartidos (importados desde play_pygame_app pero ahora viven aca) -----


def cell_from_pos(mx: int, my: int) -> tuple[int, int] | None:
    bx = mx - PAD
    by = my - PAD
    if bx < 0 or by < 0 or bx >= BOARD_PX or by >= BOARD_PX:
        return None
    return int(by // CELL), int(bx // CELL)


def moves_from_origin(board: AtaxxBoard, origin: tuple[int, int]) -> list[Move]:
    r0, c0 = origin
    return [m for m in board.get_valid_moves() if m[0] == r0 and m[1] == c0]


def targets_and_kinds(
    moves: list[Move],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], str]]:
    targets: list[tuple[int, int]] = []
    kinds: dict[tuple[int, int], str] = {}
    for r1, c1, r2, c2 in moves:
        target = (r2, c2)
        targets.append(target)
        dist = max(abs(r1 - r2), abs(c1 - c2))
        kinds[target] = "clone" if dist == 1 else "jump"
    return targets, kinds


def counts(board: AtaxxBoard) -> tuple[int, int]:
    return int(np.sum(board.grid == PLAYER_1)), int(np.sum(board.grid == PLAYER_2))


def ai_delay_ms(
    board: AtaxxBoard,
    agent: Agent,
    sims: int,
    rng: np.random.Generator,
) -> int:
    valid_count = len(board.get_valid_moves())
    complexity = min(260, valid_count * 9)
    model_bonus = 120 if agent == "model" else 0
    sims_bonus = min(180, sims // 2) if agent == "model" else 0
    jitter = int(rng.integers(0, AI_DELAY_JITTER + 1))
    delay = AI_DELAY_MIN + complexity + model_bonus + sims_bonus + jitter
    return int(max(AI_DELAY_MIN, min(AI_DELAY_MAX, delay)))


def changed_cells(before: np.ndarray, after: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(before != after)
    return [(int(r), int(c)) for r, c in coords]


def apply_move_with_feedback(
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
    changed = changed_cells(before, board.grid)

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


def pick_ai_move(
    board: AtaxxBoard,
    agent: Agent,
    rng: np.random.Generator,
    heuristic_level: str,
    mcts: object | None,
) -> tuple[Move | None, str, dict[str, object] | None]:
    from agents.registry import agent_kind as _agent_kind
    from agents.selector import pick_ai_move as _pick

    if _agent_kind(agent) == "model":
        from agents.model_agent import model_move_with_diagnostics

        move, diagnostics = model_move_with_diagnostics(board=board, mcts=mcts)
        return move, "Model AI move played", diagnostics
    move, text = _pick(
        board=board,
        agent=agent,
        rng=rng,
        heuristic_level=heuristic_level,
        mcts=mcts,
    )
    return move, text, None


def infection_wave_schedule(
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


def spawn_particles_at(
    rng: np.random.Generator,
    particles: list[Particle],
    cell: tuple[int, int],
    color: tuple[int, int, int],
    now_ms: int,
    intensity: int,
) -> None:
    from ui.arena.effects import spawn_particles as _spawn

    _spawn(
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


# ----- Game loop principal -----


def play_single_match(
    *,
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    font: pygame.font.Font,
    small: pygame.font.Font,
    big: pygame.font.Font,
    board: AtaxxBoard,
    p1_agent: Agent,
    p2_agent: Agent,
    p1_label: str,
    p2_label: str,
    p1_level: str,
    p2_level: str,
    mcts_by_player: dict[int, object | None],
    fps: int,
    mcts_sims: int,
    rng: np.random.Generator,
    sfx: dict[str, pygame.mixer.Sound | None] | None = None,
    overlay_fn: Callable[[pygame.Surface], None] | None = None,
    top_banner: tuple[str, str] | None = None,
    recorder: ReplayRecorder | None = None,
    return_after_game_over: bool = False,
    on_fullscreen_event: Callable[[pygame.event.Event], bool] | None = None,
    record_arena_stats: bool = True,
    screenshots_dir: Path | None = None,
    flip: Callable[[], None] | None = None,
) -> MatchOutcome:
    """Corre una partida con toda la maquinaria de la arena.

    Mantiene exactamente el comportamiento del loop original de
    `play_pygame_app.main()` (intro, particles, shake, flash, sfx, pausa,
    speed mult, screenshots, etc.). Diferencias controladas por parametros:

    - `recorder`: si se pasa, cada movimiento (humano o IA) se graba para
      generar training data al final.
    - `overlay_fn`: se ejecuta despues de `draw_arena` cada frame.
    - `return_after_game_over=True`: vuelve al caller al detectar game over,
      sin esperar SPACE/R. Lo usa el torneo.
    - `on_fullscreen_event`: callback que recibe cada KEYDOWN y devuelve True
      si lo consumio (p.ej. F11). Default None = ignora F11.
    """
    from ui.arena import stats as arena_stats
    from ui.arena.render import draw_arena

    if screenshots_dir is None:
        screenshots_dir = Path("arena_screenshots")
    if sfx is None:
        sfx = {}

    clock = pygame.time.Clock()

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
    intro_start = pygame.time.get_ticks()
    intro_until = intro_start + (INTRO_STEP_MS * len(INTRO_STEPS))
    game_over_started: int | None = None
    final_counts: tuple[int, int] | None = None

    arena_state: dict[str, object] = {
        "last_top_moves": [],
        "last_root_value": 0.0,
        "last_thinker": None,
        "eval_history": [],
        "move_history": [],
        "speed_mult": 1.0,
        "paused": False,
        "brain_tab": 0,
        "stats_record": (
            arena_stats.get_record(p1_label, p2_label) if record_arena_stats else {"w": 0, "l": 0, "d": 0}
        ),
    }
    if top_banner is not None:
        arena_state["tournament_banner"] = top_banner
    speed_mult = 1.0
    paused = False
    step_once = False
    stats_recorded = False
    # Si tenemos recorder, capturamos tambien el policy del modelo via diagnostics
    # cuando el siguiente movimiento sea de la IA.
    pending_recorder_policy: np.ndarray | None = None
    pending_recorder_action_idx: int | None = None
    pending_recorder_player: int | None = None

    def _do_flip() -> None:
        if flip is not None:
            flip()
            return
        if screen is not window:
            scaled = pygame.transform.smoothscale(screen, window.get_size())
            window.blit(scaled, (0, 0))
        pygame.display.flip()

    running = True
    exit_reason = "quit"
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
                spawn_particles_at(rng, particles, cell, FLASH_INFECT, now_ms, intensity=8)

        turn_agent = p1_agent if board.current_player == PLAYER_1 else p2_agent
        intro_active = now_ms < intro_until
        if board.is_game_over() and game_over_started is None:
            game_over_started = now_ms
            final_counts = counts(board)
            if record_arena_stats and not stats_recorded:
                arena_state["stats_record"] = arena_stats.record_result(
                    p1_label, p2_label, int(board.get_result()),
                )
                stats_recorded = True
            if return_after_game_over:
                exit_reason = "completed"
                # Permitir un frame final para dibujar el game over.
                running = False

        human_turn = turn_agent == "human" and not board.is_game_over() and not intro_active
        ai_turn = turn_agent != "human" and not board.is_game_over() and not intro_active

        hover_cell = None
        hover_targets = []
        hover_target_kind = {}
        if human_turn and pending_apply_at is None:
            hover_cell = _virtual_mouse_to_cell(display_scale, screen)
            if selected is None and hover_cell is not None:
                hr, hc = hover_cell
                if int(board.grid[hr, hc]) == board.current_player:
                    hmoves = moves_from_origin(board, hover_cell)
                    hover_targets, hover_target_kind = targets_and_kinds(hmoves)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                exit_reason = "quit"
            elif on_fullscreen_event is not None and on_fullscreen_event(event):
                continue
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                    exit_reason = "quit"
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    arena_state["paused"] = paused
                elif event.key == pygame.K_s and paused:
                    step_once = True
                elif event.key == pygame.K_TAB:
                    current_tab = int(cast(int, arena_state.get("brain_tab", 0)))
                    arena_state["brain_tab"] = (current_tab + 1) % 3
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
                    pygame.image.save(
                        window if screen is not window else screen, str(out_path),
                    )
                    status = f"Captura guardada: {out_path}"
                elif (
                    event.key == pygame.K_r
                    and not return_after_game_over
                ):
                    running = False
                    exit_reason = "restart"
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and human_turn
                and pending_apply_at is None
            ):
                cell = _virtual_mouse_to_cell(display_scale, screen)
                if cell is None:
                    continue
                row, col = cell
                if selected is None:
                    if int(board.grid[row, col]) == board.current_player:
                        selected = (row, col)
                        m = moves_from_origin(board, selected)
                        legal_targets, legal_target_kind = targets_and_kinds(m)
                else:
                    if (row, col) == selected:
                        selected = None
                        legal_targets = []
                        legal_target_kind = {}
                        continue
                    if int(board.grid[row, col]) == board.current_player:
                        selected = (row, col)
                        m = moves_from_origin(board, selected)
                        legal_targets, legal_target_kind = targets_and_kinds(m)
                        continue
                    candidate: Move | None = None
                    for move in moves_from_origin(board, selected):
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
                        # Capturar policy one-hot del movimiento humano para el recorder.
                        if recorder is not None:
                            from ui.arena.replay_recorder import one_hot_policy
                            a_idx = ACTION_SPACE.encode(candidate)
                            pending_recorder_policy = one_hot_policy(
                                a_idx, ACTION_SPACE.num_actions,
                            )
                            pending_recorder_action_idx = a_idx
                            pending_recorder_player = int(board.current_player)

        if human_turn and not board.has_valid_moves() and pending_apply_at is None:
            pending_move = None
            preview_started_at = now_ms
            pending_apply_at = now_ms + MOVE_PREVIEW_MS_HUMAN
            preview_move = None
            preview_until = int(pending_apply_at)
            status = "Humano pasa turno..."
            ai_ready_at = {PLAYER_1: None, PLAYER_2: None}
            if recorder is not None:
                from ui.arena.replay_recorder import one_hot_policy
                pending_recorder_policy = one_hot_policy(
                    ACTION_SPACE.pass_index, ACTION_SPACE.num_actions,
                )
                pending_recorder_action_idx = ACTION_SPACE.pass_index
                pending_recorder_player = int(board.current_player)

        ai_gate_open = ai_turn and pending_apply_at is None and (not paused or step_once)
        if ai_gate_open:
            player = board.current_player
            ready_at = ai_ready_at[player]
            if ready_at is None:
                base_delay = ai_delay_ms(board, turn_agent, mcts_sims, rng)
                ai_ready_at[player] = now_ms + max(40, int(base_delay / max(0.5, speed_mult)))
                status = f"{turn_agent} pensando..."
            elif now_ms >= ready_at:
                move, move_text, diagnostics = pick_ai_move(
                    board=board,
                    agent=turn_agent,
                    rng=rng,
                    heuristic_level=p1_level if player == PLAYER_1 else p2_level,
                    mcts=mcts_by_player[player],
                )
                if diagnostics is not None:
                    arena_state["last_top_moves"] = list(
                        cast(list, diagnostics.get("top_moves", [])),
                    )
                    raw_value = float(
                        cast(float, diagnostics.get("root_value", 0.0) or 0.0),
                    )
                    arena_state["last_root_value"] = (
                        raw_value if player == PLAYER_1 else -raw_value
                    )
                    arena_state["last_thinker"] = player
                    arena_state["last_brain"] = diagnostics.get("brain")
                pending_move = move
                preview_started_at = now_ms
                preview_dur = max(40, int(MOVE_PREVIEW_MS_AI / max(0.5, speed_mult)))
                pending_apply_at = now_ms + preview_dur
                preview_move = move
                preview_until = int(pending_apply_at)
                status = move_text.replace("played", "queued")
                ai_ready_at = {PLAYER_1: None, PLAYER_2: None}
                step_once = False
                # Capturar policy + action_idx para el recorder.
                if recorder is not None:
                    if diagnostics is not None and "policy" in diagnostics:
                        pending_recorder_policy = np.asarray(
                            cast(np.ndarray, diagnostics["policy"]),
                            dtype=np.float32,
                        )
                        pending_recorder_action_idx = int(
                            cast(int, diagnostics.get("action_idx", ACTION_SPACE.encode(move))),
                        )
                    else:
                        # Heuristic / random: one-hot del move elegido.
                        from ui.arena.replay_recorder import one_hot_policy
                        a_idx = ACTION_SPACE.encode(move)
                        pending_recorder_policy = one_hot_policy(
                            a_idx, ACTION_SPACE.num_actions,
                        )
                        pending_recorder_action_idx = a_idx
                    pending_recorder_player = int(player)
            else:
                status = f"{turn_agent} pensando..."
        elif paused and ai_turn and pending_apply_at is None:
            status = "PAUSA  (espacio=reanudar, s=paso, 1/2/4=velocidad)"

        if pending_apply_at is not None and now_ms >= pending_apply_at:
            applied_move = pending_move
            applied_player = int(board.current_player)
            # Grabar antes del step (la observation se captura sobre el board pre-move).
            if (
                recorder is not None
                and pending_recorder_policy is not None
                and pending_recorder_action_idx is not None
                and pending_recorder_player is not None
            ):
                recorder.record_move(
                    board_before=board,
                    policy=pending_recorder_policy,
                    action_idx=pending_recorder_action_idx,
                    player=pending_recorder_player,
                )
            pending_recorder_policy = None
            pending_recorder_action_idx = None
            pending_recorder_player = None

            changed, move_cells, infect_cells, old_vals, destination = apply_move_with_feedback(
                board, pending_move,
            )
            cast(list, arena_state["move_history"]).append((applied_player, applied_move))
            cast(list, arena_state["eval_history"]).append(
                float(cast(float, arena_state.get("last_root_value", 0.0) or 0.0)),
            )
            recent = changed
            recent_until = now_ms + HIGHLIGHT_MS
            move_until = now_ms + MOVE_PULSE_MS
            infect_until = now_ms + INFECT_PULSE_MS

            reveal_schedule = infection_wave_schedule(
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
            from ui.arena.effects import play_sfx as _play_sfx
            _play_sfx(sfx=sfx, key="infect" if has_infection else "move")

            for rr, cc in move_cells:
                if int(board.grid[rr, cc]) != 0:
                    piece_pop[(rr, cc)] = (now_ms, now_ms + PIECE_POP_MS)
                    spawn_particles_at(
                        rng, particles, (rr, cc), FLASH_MOVE, now_ms, intensity=10,
                    )

        shake_offset = (0, 0)
        if shake_until > now_ms:
            shake_progress = (now_ms - shake_start) / max(1, shake_until - shake_start)
            amp = shake_max_px * (1.0 - shake_progress)
            sx = int(np.sin(now_ms * 0.040) * amp)
            sy = int(np.cos(now_ms * 0.033) * (amp * 0.7))
            shake_offset = (sx, sy)

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
        if overlay_fn is not None:
            overlay_fn(screen)
        _do_flip()
        clock.tick(fps)

    return MatchOutcome(
        reason=exit_reason,
        board=board,
        final_counts=final_counts,
        result=int(board.get_result()) if board.is_game_over() else None,
    )


def _virtual_mouse_to_cell(
    display_scale: float,
    screen: pygame.Surface,
) -> tuple[int, int] | None:
    mx, my = pygame.mouse.get_pos()
    target = pygame.display.get_surface()
    if target is None:
        return cell_from_pos(mx, my)
    tw, th = target.get_size()
    sw, sh = screen.get_size()
    scale = min(tw / sw, th / sh)
    if scale <= 0:
        return cell_from_pos(mx, my)
    offset_x = (tw - int(sw * scale)) // 2
    offset_y = (th - int(sh * scale)) // 2
    vx = int((mx - offset_x) / scale)
    vy = int((my - offset_y) / scale)
    return cell_from_pos(vx, vy)


__all__ = (
    "MatchOutcome",
    "play_single_match",
)

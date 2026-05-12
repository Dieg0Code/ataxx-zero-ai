"""Runner del modo torneo en la arena.

Orquesta roster -> sorteo -> partidas (con starter parametrizable) -> tiebreak
-> podio, reutilizando la maquinaria de la arena para cada partida.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pygame
import torch

# Asegura que `src/` este en sys.path cuando se ejecuta como modulo.
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.types import Agent  # noqa: E402
from game.board import AtaxxBoard  # noqa: E402
from ui.arena.layout import WIN_H, WIN_W  # noqa: E402
from ui.arena.match_runtime import play_single_match  # noqa: E402
from ui.arena.model_runtime import (  # noqa: E402
    build_model_mcts_by_player,
    resolve_model_checkpoints,
)
from ui.arena.replay_recorder import ReplayMetadata, ReplayRecorder  # noqa: E402
from ui.arena.tournament import (  # noqa: E402
    MatchRecord,
    TournamentConfig,
    TournamentState,
)
from ui.arena.tournament_screens import (  # noqa: E402
    ScreenFonts,
    draw_final_podium,
    draw_post_match,
    draw_pre_match,
    draw_roster,
    draw_shuffle,
    draw_tiebreak_intro,
)

PLAYER_1 = 1
PLAYER_2 = -1

SHUFFLE_DURATION_MS = 2000
POST_MATCH_MIN_DISPLAY_MS = 600


# Estado global del display (mutado por _toggle_fullscreen y leido por _flip).
_DISPLAY_STATE: dict[str, object] = {
    "fullscreen": False,
    "windowed_size": (WIN_W, WIN_H),
}


def _toggle_fullscreen() -> None:
    """Alterna entre ventana y pantalla completa preservando aspect ratio."""
    fullscreen = bool(_DISPLAY_STATE.get("fullscreen", False))
    if fullscreen:
        size = _DISPLAY_STATE.get("windowed_size", (WIN_W, WIN_H))
        if isinstance(size, tuple):
            pygame.display.set_mode(size)
        _DISPLAY_STATE["fullscreen"] = False
    else:
        # Guarda el tamano de ventana actual para volver despues.
        surf = pygame.display.get_surface()
        if surf is not None:
            _DISPLAY_STATE["windowed_size"] = surf.get_size()
        pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        _DISPLAY_STATE["fullscreen"] = True


def _maybe_handle_fullscreen(event: pygame.event.Event) -> bool:
    """F11 toggles fullscreen. Devuelve True si consumio el evento."""
    if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
        _toggle_fullscreen()
        return True
    return False


# ----- Utilidades de event loop -----


def _flip(scene: pygame.Surface, _window: pygame.Surface, _display_scale: float) -> None:
    """Vuelca el surface logico al window real. Re-escala al tamano actual."""
    target = pygame.display.get_surface()
    if target is None:
        return
    if scene is target:
        pygame.display.flip()
        return
    # Calcula tamano que preserve aspect ratio dentro del target actual.
    tw, th = target.get_size()
    sw, sh = scene.get_size()
    scale = min(tw / sw, th / sh)
    out_w = int(sw * scale)
    out_h = int(sh * scale)
    scaled = pygame.transform.smoothscale(scene, (out_w, out_h))
    target.fill((0, 0, 0))
    target.blit(scaled, ((tw - out_w) // 2, (th - out_h) // 2))
    pygame.display.flip()


def _await_keys(
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    draw_fn: Callable[[pygame.Surface], None],
    allowed_keys: set[int],
) -> int | None:
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if _maybe_handle_fullscreen(event):
                continue
            if event.type == pygame.KEYDOWN and event.key in allowed_keys:
                return int(event.key)
        draw_fn(screen)
        _flip(screen, window, display_scale)
        clock.tick(60)


# ----- Pantalla: Roster -----


def _show_roster(
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    fonts: ScreenFonts,
    state: TournamentState,
) -> list[bool] | None:
    n = len(state.config.players)
    present_flags = [True] * n
    selected_index = 0
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if _maybe_handle_fullscreen(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return None
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and any(
                    present_flags,
                ):
                    return present_flags
                if event.key == pygame.K_UP:
                    selected_index = (selected_index - 1) % n
                if event.key == pygame.K_DOWN:
                    selected_index = (selected_index + 1) % n
                if pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if 0 <= idx < n:
                        present_flags[idx] = not present_flags[idx]
                        selected_index = idx
                if event.key == pygame.K_SPACE:
                    present_flags[selected_index] = not present_flags[selected_index]
        draw_roster(
            screen, fonts,
            state=state,
            present_flags=present_flags,
            selected_index=selected_index,
        )
        _flip(screen, window, display_scale)
        clock.tick(60)


# ----- Pantalla: Sorteo animado -----


def _show_shuffle(
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    fonts: ScreenFonts,
    state: TournamentState,
    rng: random.Random,
) -> bool:
    rolling = list(state.present)
    if not rolling:
        return False
    clock = pygame.time.Clock()
    start_ms = pygame.time.get_ticks()
    final_order: list[str] | None = None
    while True:
        elapsed = pygame.time.get_ticks() - start_ms
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if _maybe_handle_fullscreen(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                if final_order is not None and event.key == pygame.K_SPACE:
                    state.order = final_order
                    return True
        if final_order is None and elapsed >= SHUFFLE_DURATION_MS:
            final_order = state.draw_order(rng)
        progress = (elapsed % 600) / 600.0
        rolling_local = list(rolling)
        rng.shuffle(rolling_local)
        draw_shuffle(
            screen, fonts,
            state=state,
            rolling_names=rolling_local,
            final_order=final_order,
            progress=progress,
        )
        _flip(screen, window, display_scale)
        clock.tick(30)


# ----- Pantallas pre/post/tiebreak -----


def _show_pre_match(
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    fonts: ScreenFonts,
    state: TournamentState,
    match: MatchRecord,
) -> str:
    allowed = {pygame.K_a, pygame.K_f, pygame.K_q}
    key = _await_keys(
        screen, window, display_scale,
        lambda surf: draw_pre_match(surf, fonts, state=state, match=match),
        allowed,
    )
    if key is None or key == pygame.K_q:
        return "quit"
    if key == pygame.K_f:
        return "forfeit"
    return "accept"


def _show_tiebreak_intro(
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    fonts: ScreenFonts,
    state: TournamentState,
    match: MatchRecord,
    other_player: str,
) -> str:
    allowed = {pygame.K_a, pygame.K_q}
    key = _await_keys(
        screen, window, display_scale,
        lambda surf: draw_tiebreak_intro(
            surf, fonts, state=state, match=match, other_player=other_player,
        ),
        allowed,
    )
    if key is None or key == pygame.K_q:
        return "quit"
    return "accept"


def _show_post_match(
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    fonts: ScreenFonts,
    state: TournamentState,
    match: MatchRecord,
) -> None:
    start = pygame.time.get_ticks()
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if _maybe_handle_fullscreen(event):
                continue
            if (
                event.type == pygame.KEYDOWN
                and event.key in (pygame.K_SPACE, pygame.K_RETURN)
                and pygame.time.get_ticks() - start >= POST_MATCH_MIN_DISPLAY_MS
            ):
                return
        draw_post_match(screen, fonts, state=state, match=match)
        _flip(screen, window, display_scale)
        clock.tick(60)


def _show_podium(
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    fonts: ScreenFonts,
    state: TournamentState,
) -> None:
    screenshots_dir = Path("arena_screenshots")
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if _maybe_handle_fullscreen(event):
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return
                if event.key == pygame.K_p:
                    screenshots_dir.mkdir(exist_ok=True)
                    fname = pygame.time.get_ticks()
                    out_path = screenshots_dir / f"tournament_final_{fname}.png"
                    pygame.image.save(
                        window if screen is not window else screen,
                        str(out_path),
                    )
        draw_final_podium(screen, fonts, state=state)
        _flip(screen, window, display_scale)
        clock.tick(60)




def run_tournament(args: argparse.Namespace) -> None:
    """Orquesta el flujo completo del torneo."""
    players_raw = args.players if args.players else "Felipe,Julio,Vicente"
    players = tuple(p.strip() for p in players_raw.split(",") if p.strip())
    if not players:
        raise SystemExit("Hay que configurar al menos un jugador via --players")

    save_path = Path(args.tournament_save).expanduser()
    config = TournamentConfig(
        players=players,
        model_label=args.tournament_model,
        mcts_sims=int(args.tournament_sims),
        save_path=save_path,
    )

    state: TournamentState | None = None
    if args.tournament_resume:
        state = TournamentState.load(save_path)
        if state is None:
            print(f"[tournament] No habia estado en {save_path}; empiezo de cero.")
    if state is None:
        state = TournamentState.new(config)

    from model.registry import resolve as resolve_codename
    try:
        ckpt_path = str(resolve_codename(config.model_label))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"No se pudo resolver el modelo '{config.model_label}': {exc}",
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        device = "cpu"

    pygame.init()
    info = pygame.display.Info()
    avail_w = max(640, info.current_w - 60)
    avail_h = max(480, info.current_h - 160)
    display_scale = min(avail_w / WIN_W, avail_h / WIN_H, 1.0)
    win_size = (int(WIN_W * display_scale), int(WIN_H * display_scale))
    _DISPLAY_STATE["windowed_size"] = win_size
    if getattr(args, "fullscreen", False):
        pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        _DISPLAY_STATE["fullscreen"] = True
    else:
        pygame.display.set_mode(win_size)
        _DISPLAY_STATE["fullscreen"] = False
    pygame.display.set_caption("Ataxx Tournament")
    window = pygame.display.get_surface()
    # Siempre dibujamos en el scene logico de WIN_W x WIN_H y _flip escala.
    screen = pygame.Surface((WIN_W, WIN_H))

    from ui.arena.effects import build_sfx
    from ui.arena.fonts import load_arena_fonts, load_tournament_fonts
    arena_font, arena_small, arena_big = load_arena_fonts()
    fonts = ScreenFonts.from_dict(load_tournament_fonts())
    sfx = build_sfx()

    rng_np = np.random.default_rng(seed=None if args.seed < 0 else args.seed)
    rng = random.Random(None if args.seed < 0 else args.seed)  # noqa: S311

    if not state.order:
        flags = _show_roster(screen, window, display_scale, fonts, state)
        if flags is None:
            pygame.quit()
            return
        present = [state.config.players[i] for i, f in enumerate(flags) if f]
        state.set_present(present)
        state.persist()

        if not _show_shuffle(screen, window, display_scale, fonts, state, rng):
            state.finish()
            state.persist()
            pygame.quit()
            return
        state.persist()

    from ui.arena.play_pygame_app import _load_system
    checkpoints_by_player = resolve_model_checkpoints(
        shared_checkpoint="",
        p1_agent="human",
        p2_agent="model",
        p1_checkpoint="",
        p2_checkpoint=ckpt_path,
    )
    mcts_by_player = build_model_mcts_by_player(
        checkpoints_by_player=checkpoints_by_player,
        device=device,
        c_puct=1.5,
        n_simulations=config.mcts_sims,
        load_system=_load_system,
    )
    # El MCTS esta vinculado al modelo, no al lado. Reusamos cualquiera de los dos.
    mcts = mcts_by_player.get(PLAYER_2) or mcts_by_player.get(PLAYER_1)

    # Fase 1: partidas regulares (2 rondas x Bo3 por jugador).
    arena_fonts = (arena_font, arena_small, arena_big)
    if not _run_regular_phase(
        screen=screen, window=window, display_scale=display_scale,
        fonts=fonts, arena_fonts=arena_fonts, sfx=sfx,
        state=state, mcts=mcts, rng_np=rng_np,
    ):
        pygame.quit()
        return

    # Fase 2: tiebreaks si hace falta.
    _run_tiebreak_phase(
        screen=screen, window=window, display_scale=display_scale,
        fonts=fonts, arena_fonts=arena_fonts, sfx=sfx,
        state=state, mcts=mcts, rng=rng, rng_np=rng_np,
    )

    state.finished = True
    state.persist()
    _show_podium(screen, window, display_scale, fonts, state)
    pygame.quit()


def _counts_for_human(
    board: AtaxxBoard, human_side: int,
) -> tuple[int, int, int]:
    if human_side == PLAYER_1:
        return int(board.p1_count), int(board.p2_count), int(board.half_moves)
    return int(board.p2_count), int(board.p1_count), int(board.half_moves)


def _tournament_replay_path(state: TournamentState, match: MatchRecord) -> Path:
    """Path para el NPZ del replay del torneo."""
    import time

    base = Path("tournament_replays") / state.tournament_id
    timestamp = int(time.time())
    if match.is_tiebreak:
        stem = f"tiebreak_{match.player}_iter{match.match_idx}_{timestamp}"
    else:
        stem = (
            f"{match.player}_r{match.round_idx}_p{match.match_idx}_"
            f"{match.starter}_{timestamp}"
        )
    return base / f"{stem}.npz"


def _tournament_banner(state: TournamentState, match: MatchRecord) -> tuple[str, str]:
    """Genera las dos lineas del banner que reemplaza el header del HUD top panel."""
    starter_text = "Humano (rojo)" if match.starter == "human" else f"IA ({state.config.model_label})"
    if match.is_tiebreak:
        title = f"RELAMPAGO  ·  iter {match.match_idx + 1}"
        subtitle = (
            f"{match.player} vs {state.config.model_label}  ·  Inicia {starter_text}"
        )
        return title, subtitle
    completed = state.total_completed_regular()
    total = state.total_active()
    title = (
        f"TORNEO  ·  Partida {completed + 1} / {total}  ·  "
        f"R{match.round_idx + 1} P{match.match_idx + 1}"
    )
    subtitle = (
        f"{match.player}  vs  {state.config.model_label}   ·   Inicia {starter_text}"
    )
    return title, subtitle


def _build_tournament_recorder(
    state: TournamentState, match: MatchRecord,
) -> ReplayRecorder:
    metadata = ReplayMetadata(
        mode="tournament",
        player_ai=state.config.model_label,
        starter=match.starter,
        mcts_sims=state.config.mcts_sims,
        tournament_id=state.tournament_id,
        player_human=match.player,
        round_idx=None if match.is_tiebreak else match.round_idx,
        match_idx=match.match_idx,
        is_tiebreak=match.is_tiebreak,
        p1_label=(
            match.player if match.starter == "human" else state.config.model_label
        ),
        p2_label=(
            state.config.model_label if match.starter == "human" else match.player
        ),
    )
    return ReplayRecorder(
        save_path=_tournament_replay_path(state, match), metadata=metadata,
    )


def _play_match_via_arena(
    *,
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    fonts: ScreenFonts,
    arena_fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font],
    sfx: dict[str, pygame.mixer.Sound | None],
    state: TournamentState,
    match: MatchRecord,
    mcts: object | None,
    rng_np: np.random.Generator,
) -> tuple[int, int, int, str]:
    """Corre una partida del torneo via play_single_match. Devuelve (h, a, hm, reason)."""
    arena_font, arena_small, arena_big = arena_fonts
    human_side = PLAYER_1 if match.starter == "human" else PLAYER_2
    p1_agent: Agent = "human" if human_side == PLAYER_1 else "model"
    p2_agent: Agent = "model" if human_side == PLAYER_1 else "human"
    p1_label = (
        match.player if human_side == PLAYER_1 else state.config.model_label
    )
    p2_label = (
        state.config.model_label if human_side == PLAYER_1 else match.player
    )
    mcts_by_player: dict[int, object | None] = {
        PLAYER_1: mcts if p1_agent == "model" else None,
        PLAYER_2: mcts if p2_agent == "model" else None,
    }
    recorder = _build_tournament_recorder(state, match)
    banner = _tournament_banner(state, match)

    outcome = play_single_match(
        screen=screen, window=window, display_scale=display_scale,
        font=arena_font, small=arena_small, big=arena_big,
        board=AtaxxBoard(),
        p1_agent=p1_agent, p2_agent=p2_agent,
        p1_label=p1_label, p2_label=p2_label,
        p1_level="-", p2_level="-",
        mcts_by_player=mcts_by_player,
        fps=60, mcts_sims=state.config.mcts_sims, rng=rng_np, sfx=sfx,
        top_banner=banner,
        recorder=recorder,
        return_after_game_over=True,
        on_fullscreen_event=_maybe_handle_fullscreen,
        record_arena_stats=False,
        flip=lambda: _flip(screen, window, display_scale),
    )
    if outcome.result is not None:
        recorder.finalize(winner=int(outcome.result), forced_draw=False)
    h, a, hm = _counts_for_human(outcome.board, human_side)
    return h, a, hm, outcome.reason


def _run_regular_phase(
    *,
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    fonts: ScreenFonts,
    arena_fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font],
    sfx: dict[str, pygame.mixer.Sound | None],
    state: TournamentState,
    mcts: object | None,
    rng_np: np.random.Generator,
) -> bool:
    """Devuelve False si el usuario abandono el torneo cerrando la ventana."""
    while True:
        match = state.next_pending_match()
        if match is None:
            return True
        decision = _show_pre_match(
            screen, window, display_scale, fonts, state, match,
        )
        if decision == "quit":
            state.finish()
            state.persist()
            return True
        if decision == "forfeit":
            state.forfeit_remaining_for(match.player)
            state.persist()
            continue
        h, a, hm, reason = _play_match_via_arena(
            screen=screen, window=window, display_scale=display_scale,
            fonts=fonts, arena_fonts=arena_fonts, sfx=sfx,
            state=state, match=match, mcts=mcts, rng_np=rng_np,
        )
        state.record_result(
            match, human_pieces=h, ai_pieces=a, halfmoves=hm,
        )
        state.persist()
        if reason == "quit":
            state.finish()
            state.persist()
            return False
        _show_post_match(screen, window, display_scale, fonts, state, match)


def _run_tiebreak_phase(
    *,
    screen: pygame.Surface,
    window: pygame.Surface,
    display_scale: float,
    fonts: ScreenFonts,
    arena_fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font],
    sfx: dict[str, pygame.mixer.Sound | None],
    state: TournamentState,
    mcts: object | None,
    rng: random.Random,
    rng_np: np.random.Generator,
) -> None:
    while True:
        pair = state.needs_tiebreak()
        if pair is None:
            return
        a, b = pair
        rec_a, rec_b = state.add_tiebreak_pair(a, b, rng=rng)
        state.persist()
        for current, other in ((rec_a, b), (rec_b, a)):
            decision = _show_tiebreak_intro(
                screen, window, display_scale, fonts, state, current, other,
            )
            if decision == "quit":
                state.finish()
                state.persist()
                return
            h, ai_, hm, reason = _play_match_via_arena(
                screen=screen, window=window, display_scale=display_scale,
                fonts=fonts, arena_fonts=arena_fonts, sfx=sfx,
                state=state, match=current, mcts=mcts, rng_np=rng_np,
            )
            state.record_result(
                current, human_pieces=h, ai_pieces=ai_, halfmoves=hm,
            )
            state.persist()
            if reason == "quit":
                state.finish()
                state.persist()
                return
            _show_post_match(screen, window, display_scale, fonts, state, current)


__all__ = ("run_tournament",)

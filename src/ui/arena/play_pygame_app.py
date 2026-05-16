from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
from ui.arena.layout import WIN_H, WIN_W  # noqa: E402
from ui.arena.model_runtime import (  # noqa: E402
    build_model_mcts_by_player,
    resolve_model_checkpoints,
)

if TYPE_CHECKING:
    from agents.types import Agent
    from model.system import AtaxxZero
    from ui.arena.replay_recorder import ReplayRecorder

PLAYER_1 = 1
PLAYER_2 = -1

def _ensure_src_on_path() -> None:
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ataxx arena (play/spectate/tournament).")
    parser.add_argument(
        "--mode", default="play", choices=["play", "spectate", "tournament"],
    )
    parser.add_argument(
        "--players", default="Felipe,Julio,Vicente",
        help="Lista separada por comas con el roster del torneo.",
    )
    parser.add_argument("--tournament-model", default="liga")
    parser.add_argument("--tournament-sims", type=int, default=400)
    parser.add_argument(
        "--tournament-save", default=str(Path.home() / ".ataxx_tournament.json"),
    )
    parser.add_argument("--tournament-resume", action="store_true")
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="Arrancar la arena en pantalla completa (toggle con F11).",
    )
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



def _maybe_handle_fullscreen_play(event: pygame.event.Event) -> bool:
    """F11 toggle fullscreen para el modo play."""
    if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
        surf = pygame.display.get_surface()
        if surf is None:
            return True
        is_full = bool(surf.get_flags() & pygame.FULLSCREEN)
        if is_full:
            pygame.display.set_mode((WIN_W, WIN_H))
        else:
            pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        return True
    return False


def _should_record_replay(args: argparse.Namespace, p1_agent: Agent, p2_agent: Agent) -> bool:
    """Graba si hay al menos un agente IA (no humano-humano hotseat)."""
    if getattr(args, "no_record_replay", False):
        return False
    kinds = {
        agent_kind(p1_agent, default_heuristic_level=args.heuristic_level),
        agent_kind(p2_agent, default_heuristic_level=args.heuristic_level),
    }
    return "human" in kinds and kinds != {"human"}


def _build_play_recorder(
    args: argparse.Namespace,
    p1_agent: Agent,
    p2_agent: Agent,
    p1_label: str,
    p2_label: str,
) -> ReplayRecorder | None:
    import time

    from ui.arena.replay_recorder import ReplayMetadata, ReplayRecorder

    if not _should_record_replay(args, p1_agent, p2_agent):
        return None
    p1_kind = agent_kind(p1_agent, default_heuristic_level=args.heuristic_level)
    if p1_kind == "human":
        starter = "human"
        human_label = p1_label
        ai_label = p2_label
    else:
        starter = "ai"
        human_label = p2_label
        ai_label = p1_label
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path("tournament_replays") / "play_sessions" / f"{stamp}.npz"
    metadata = ReplayMetadata(
        mode="play",
        player_ai=str(ai_label),
        starter=starter,
        mcts_sims=int(args.mcts_sims),
        player_human=str(human_label),
        p1_label=str(p1_label),
        p2_label=str(p2_label),
    )
    return ReplayRecorder(save_path=save_path, metadata=metadata)


def main() -> None:
    _ensure_src_on_path()
    from game.board import AtaxxBoard

    args = _parse_args()
    if args.mode == "tournament":
        from ui.arena.tournament_runner import run_tournament
        run_tournament(args)
        return
    device = _resolve_device(args.device)
    p1_agent, p2_agent = _resolve_agents(args)
    p1_level, p2_level = _resolve_heuristic_levels(args, p1_agent, p2_agent)
    rng = np.random.default_rng(seed=None if args.seed < 0 else args.seed)

    model_mcts_by_player: dict[int, object | None] = {PLAYER_1: None, PLAYER_2: None}
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
    avail_w = max(640, info.current_w - 60)
    avail_h = max(480, info.current_h - 160)
    display_scale = min(avail_w / WIN_W, avail_h / WIN_H, 1.0)
    win_size = (int(WIN_W * display_scale), int(WIN_H * display_scale))
    if getattr(args, "fullscreen", False):
        pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        pygame.display.set_mode(win_size)
    pygame.display.set_caption("Ataxx Arena")
    window = pygame.display.get_surface()
    # Siempre dibujamos en un scene logico 1280x720 y escalamos al display real.
    screen = pygame.Surface((WIN_W, WIN_H))

    from ui.arena.effects import build_sfx
    from ui.arena.fonts import load_arena_fonts
    from ui.arena.match_runtime import play_single_match
    font, small, big = load_arena_fonts()
    sfx = build_sfx()

    p1_label = f"{p1_agent}({p1_level})" if p1_level != "-" else str(p1_agent)
    p2_label = f"{p2_agent}({p2_level})" if p2_level != "-" else str(p2_agent)

    def _flip_play() -> None:
        target = pygame.display.get_surface()
        if target is None:
            return
        if screen is target:
            pygame.display.flip()
            return
        tw, th = target.get_size()
        sw, sh = screen.get_size()
        scale = min(tw / sw, th / sh)
        out_w = int(sw * scale)
        out_h = int(sh * scale)
        scaled = pygame.transform.smoothscale(screen, (out_w, out_h))
        target.fill((0, 0, 0))
        target.blit(scaled, ((tw - out_w) // 2, (th - out_h) // 2))
        pygame.display.flip()

    while True:
        recorder = _build_play_recorder(args, p1_agent, p2_agent, p1_label, p2_label)
        outcome = play_single_match(
            screen=screen, window=window, display_scale=display_scale,
            font=font, small=small, big=big,
            board=AtaxxBoard(),
            p1_agent=p1_agent, p2_agent=p2_agent,
            p1_label=p1_label, p2_label=p2_label,
            p1_level=p1_level, p2_level=p2_level,
            mcts_by_player=model_mcts_by_player,
            fps=args.fps, mcts_sims=args.mcts_sims, rng=rng, sfx=sfx,
            recorder=recorder,
            return_after_game_over=False,
            on_fullscreen_event=_maybe_handle_fullscreen_play,
            flip=_flip_play,
        )
        if recorder is not None and outcome.result is not None:
            final_p1, final_p2 = (
                outcome.final_counts if outcome.final_counts is not None else (None, None)
            )
            recorder.finalize(
                winner=int(outcome.result),
                forced_draw=False,
                final_p1_count=final_p1,
                final_p2_count=final_p2,
            )
        if outcome.reason == "quit":
            break
        # "restart" -> recrear board y volver a llamar (loop continue).

    pygame.quit()



if __name__ == "__main__":
    main()



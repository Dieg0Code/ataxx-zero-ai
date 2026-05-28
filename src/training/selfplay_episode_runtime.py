from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from agents.heuristic import heuristic_move
from game.actions import ACTION_SPACE
from training.config_runtime import cfg_bool, cfg_float, cfg_int, log
from training.reward_runtime import (
    HistoryEntry,
    compute_state_potential,
    compute_transition_shaping_reward,
)

if TYPE_CHECKING:
    from engine.mcts import MCTS, MCTSNode
    from game.board import AtaxxBoard
    from training.league_runtime import CheckpointModelSpec

_WORKER_MCTS: object | None = None
_WORKER_OPPONENT_MCTS_POOL: dict[str, object] | None = None


def _one_hot_policy(action_idx: int) -> np.ndarray:
    policy = np.zeros(ACTION_SPACE.num_actions, dtype=np.float32)
    policy[action_idx] = 1.0
    return policy


def _append_history_entry(
    *,
    game_history: list[HistoryEntry],
    board: AtaxxBoard,
    player_at_turn: int,
    policy: np.ndarray,
    action_idx: int,
    mcts_q: float = 0.0,
) -> None:
    shaping_enabled = cfg_bool("reward_shaping_enabled")
    observation = board.get_observation()
    shaping_reward = 0.0
    before_potential = 0.0
    if shaping_enabled:
        before_potential = compute_state_potential(board, player_at_turn)
    board.step(ACTION_SPACE.decode(action_idx))
    if shaping_enabled:
        after_potential = compute_state_potential(board, player_at_turn)
        shaping_reward = compute_transition_shaping_reward(
            before_potential=before_potential,
            after_potential=after_potential,
        )
    # v15+ history entries are 5-tuples (obs, policy, player, shaping, mcts_q).
    # mcts_q es root.value() desde la perspectiva del que mueve, capturado tras MCTS.
    game_history.append((observation, policy, player_at_turn, shaping_reward, float(mcts_q)))


def _resolve_selfplay_worker_device(worker_devices: tuple[str, ...]) -> str:
    if len(worker_devices) == 0:
        return "cpu"
    if len(worker_devices) == 1:
        return worker_devices[0]

    identity = getattr(mp.current_process(), "_identity", ())
    if len(identity) > 0 and isinstance(identity[0], int):
        slot = max(0, int(identity[0]) - 1)
        return worker_devices[slot % len(worker_devices)]

    name = mp.current_process().name
    suffix = name.rsplit("-", maxsplit=1)[-1]
    if suffix.isdigit():
        slot = max(0, int(suffix) - 1)
        return worker_devices[slot % len(worker_devices)]
    return worker_devices[0]


def compute_action_probs(
    board: AtaxxBoard,
    mcts: MCTS,
    root: MCTSNode | None,
    add_noise: bool,
    temperature: float,
) -> tuple[np.ndarray, MCTSNode | None]:
    probs, updated_root = mcts.run_with_root(
        board=board,
        root=root,
        add_dirichlet_noise=add_noise,
        temperature=temperature,
    )
    total_prob = float(np.sum(probs))
    if total_prob > 0.0:
        return probs, updated_root

    valid_moves = board.get_valid_moves()
    fallback = ACTION_SPACE.mask_from_moves(
        valid_moves,
        include_pass=(len(valid_moves) == 0),
    )
    return fallback / float(np.sum(fallback)), updated_root


def select_action_idx(
    probs: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if temperature > 0.0:
        return int(rng.choice(len(probs), p=probs))
    return int(np.argmax(probs))


def random_move(
    board: AtaxxBoard,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    moves = board.get_valid_moves()
    if len(moves) == 0:
        return None
    return moves[int(rng.integers(0, len(moves)))]


def play_episode(
    mcts: MCTS,
    add_noise: bool,
    temp_threshold: int,
    rng: np.random.Generator,
    opponent_type: str,
    opponent_heuristic_level: str,
    model_player: int,
    opponent_checkpoint_mcts: MCTS | None = None,
) -> tuple[list[HistoryEntry], int, int, bool, int, int]:
    from game.board import AtaxxBoard

    board = AtaxxBoard()

    # v15.1: random opening — con prob p jugamos 1..N random plies antes del
    # MCTS real. No entran al history (policy target seria one-hot random,
    # ruido puro). Solo siembra diversidad de aperturas.
    random_opening_prob = max(0.0, min(1.0, cfg_float("random_opening_prob")))
    random_opening_max_plies = max(0, cfg_int("random_opening_max_plies"))
    if random_opening_prob > 0.0 and random_opening_max_plies > 0:
        if float(rng.random()) < random_opening_prob:
            n_random = int(rng.integers(1, random_opening_max_plies + 1))
            for _ in range(n_random):
                if board.is_game_over():
                    break
                move = random_move(board, rng)
                if move is None:
                    break
                board.step(move)

    model_root = None
    opponent_root = None
    game_history: list[HistoryEntry] = []
    turn_idx = 0

    # v15.1: tope duro de plies por episodio. 0 = sin tope (legacy <= v15).
    max_half_moves_cap = max(0, cfg_int("max_half_moves_per_episode"))

    # v15.2: PCR (playout cap randomization). Si pcr_enabled, cada turno
    # decide entre full-sim (entra al history) o fast-sim (solo avanza).
    pcr_enabled = cfg_bool("pcr_enabled")
    pcr_fast_sims = max(1, cfg_int("pcr_fast_sims"))
    pcr_full_sim_prob = max(0.0, min(1.0, cfg_float("pcr_full_sim_prob")))
    full_sims_original = mcts.n_simulations

    # v15.2: resignacion. Contadores por lado de plies consecutivos con
    # mcts_q < resign_value_threshold (medido solo en full-sim turns).
    resign_enabled = cfg_bool("resign_enabled")
    resign_threshold = cfg_float("resign_value_threshold")
    resign_consecutive = max(1, cfg_int("resign_consecutive_plies"))
    resign_min_plies = max(0, cfg_int("resign_min_plies"))
    resign_audit_prob = max(0.0, min(1.0, cfg_float("resign_audit_prob")))
    resign_counters: dict[int, int] = {1: 0, -1: 0}
    resigned_side: int | None = None

    def _check_resign(side: int, mcts_q: float) -> None:
        nonlocal resigned_side
        if not resign_enabled or turn_idx < resign_min_plies:
            return
        if mcts_q < resign_threshold:
            resign_counters[side] += 1
            if resign_counters[side] >= resign_consecutive:
                # Audit: con prob p saltamos la resignacion y dejamos que el
                # game corra. Permite verificar offline cuantas "resigns" eran
                # falsos positivos (el lado supuesto perdedor termina ganando).
                if resign_audit_prob > 0.0 and float(rng.random()) < resign_audit_prob:
                    resign_counters[side] = 0
                    return
                resigned_side = side
        else:
            resign_counters[side] = 0

    try:
        while not board.is_game_over():
            if max_half_moves_cap > 0 and turn_idx >= max_half_moves_cap:
                break
            turn_idx += 1
            is_model_turn = board.current_player == model_player

            # PCR aplica solo a turnos del modelo (model + self). En checkpoint
            # opponent dejamos full sims (oponente frozen juega siempre fuerte).
            is_full_sim_turn = (not pcr_enabled) or float(rng.random()) < pcr_full_sim_prob

            if is_model_turn:
                temperature = 1.0 if turn_idx <= temp_threshold else 0.0
                mcts.n_simulations = full_sims_original if is_full_sim_turn else pcr_fast_sims
                probs, model_root = compute_action_probs(
                    board=board,
                    mcts=mcts,
                    root=model_root,
                    add_noise=add_noise and is_full_sim_turn,
                    temperature=temperature,
                )
                player_at_turn = int(board.current_player)
                action_idx = select_action_idx(probs=probs, temperature=temperature, rng=rng)
                policy = probs.astype(np.float32, copy=False)
                mcts_q = float(model_root.value()) if model_root is not None else 0.0
                if is_full_sim_turn:
                    _append_history_entry(
                        game_history=game_history,
                        board=board,
                        player_at_turn=player_at_turn,
                        policy=policy,
                        action_idx=action_idx,
                        mcts_q=mcts_q,
                    )
                    _check_resign(player_at_turn, mcts_q)
                else:
                    board.step(ACTION_SPACE.decode(action_idx))
                model_root = mcts.advance_root(model_root, action_idx)
                if opponent_type == "checkpoint" and opponent_checkpoint_mcts is not None:
                    opponent_root = opponent_checkpoint_mcts.advance_root(opponent_root, action_idx)
                if resigned_side is not None:
                    break
                continue

            if opponent_type == "self":
                temperature = 1.0 if turn_idx <= temp_threshold else 0.0
                mcts.n_simulations = full_sims_original if is_full_sim_turn else pcr_fast_sims
                probs, model_root = compute_action_probs(
                    board=board,
                    mcts=mcts,
                    root=model_root,
                    add_noise=add_noise and is_full_sim_turn,
                    temperature=temperature,
                )
                player_at_turn = int(board.current_player)
                action_idx = select_action_idx(probs=probs, temperature=temperature, rng=rng)
                policy = probs.astype(np.float32, copy=False)
                mcts_q = float(model_root.value()) if model_root is not None else 0.0
                if is_full_sim_turn:
                    _append_history_entry(
                        game_history=game_history,
                        board=board,
                        player_at_turn=player_at_turn,
                        policy=policy,
                        action_idx=action_idx,
                        mcts_q=mcts_q,
                    )
                    _check_resign(player_at_turn, mcts_q)
                else:
                    board.step(ACTION_SPACE.decode(action_idx))
                model_root = mcts.advance_root(model_root, action_idx)
                if resigned_side is not None:
                    break
                continue

            if opponent_type == "checkpoint":
                if opponent_checkpoint_mcts is None:
                    raise RuntimeError("Checkpoint opponent selected without checkpoint MCTS.")
                temperature = 1.0 if turn_idx <= temp_threshold else 0.0
                probs, opponent_root = compute_action_probs(
                    board=board,
                    mcts=opponent_checkpoint_mcts,
                    root=opponent_root,
                    add_noise=False,
                    temperature=temperature,
                )
                player_at_turn = int(board.current_player)
                action_idx = select_action_idx(probs=probs, temperature=temperature, rng=rng)
                mcts_q = float(opponent_root.value()) if opponent_root is not None else 0.0
                _append_history_entry(
                    game_history=game_history,
                    board=board,
                    player_at_turn=player_at_turn,
                    policy=probs.astype(np.float32, copy=False),
                    action_idx=action_idx,
                    mcts_q=mcts_q,
                )
                _check_resign(player_at_turn, mcts_q)
                model_root = mcts.advance_root(model_root, action_idx)
                opponent_root = opponent_checkpoint_mcts.advance_root(opponent_root, action_idx)
                if resigned_side is not None:
                    break
                continue

            if opponent_type == "heuristic":
                move = heuristic_move(
                    board,
                    rng,
                    opponent_heuristic_level,
                    epsilon=cfg_float("opponent_heuristic_epsilon"),
                )
                action_idx = ACTION_SPACE.encode(move)
                _append_history_entry(
                    game_history=game_history,
                    board=board,
                    player_at_turn=int(board.current_player),
                    policy=_one_hot_policy(action_idx),
                    action_idx=action_idx,
                )
                model_root = mcts.advance_root(model_root, action_idx)
                continue

            move = random_move(board, rng)
            board.step(move)
            model_root = mcts.advance_root(model_root, ACTION_SPACE.encode(move))
    finally:
        mcts.n_simulations = full_sims_original

    p1_count = int(board.p1_count)
    p2_count = int(board.p2_count)

    # v15.2: si un lado se rindio, ese lado pierde (winner = -resigned_side).
    if resigned_side is not None:
        return (game_history, -resigned_side, turn_idx, False, p1_count, p2_count)

    # v15.1: si cortamos por max_half_moves_cap sin terminal, tratamos como
    # forced draw (recibe draw_penalty via outcome_value_target).
    if not board.is_game_over():
        return (game_history, 0, turn_idx, True, p1_count, p2_count)

    return (
        game_history,
        board.get_result(),
        turn_idx,
        board.is_forced_draw(),
        p1_count,
        p2_count,
    )


def build_sequential_checkpoint_mcts_pool(
    *,
    specs: dict[str, CheckpointModelSpec],
    device: str,
) -> dict[str, MCTS]:
    from engine.mcts import MCTS
    from model.transformer import AtaxxTransformerNet

    pool: dict[str, MCTS] = {}
    for participant_id, spec in specs.items():
        opponent_model = AtaxxTransformerNet(
            d_model=int(spec.model_cfg["d_model"]),
            nhead=int(spec.model_cfg["nhead"]),
            num_layers=int(spec.model_cfg["num_layers"]),
            value_head_depth=int(spec.model_cfg.get("value_head_depth", 1)),
            count_head_enabled=bool(spec.model_cfg.get("count_head_enabled", 0.0)),
            transformer_pre_ln=bool(spec.model_cfg.get("transformer_pre_ln", 0.0)),
            pos_embed_2d=bool(spec.model_cfg.get("pos_embed_2d", 0.0)),
            patch_embed_conv=bool(spec.model_cfg.get("patch_embed_conv", 0.0)),
            num_input_channels=int(spec.model_cfg.get("num_input_channels", 11.0)),
            dim_feedforward=int(spec.model_cfg["dim_feedforward"]),
            dropout=float(spec.model_cfg["dropout"]),
        )
        opponent_model.load_state_dict(spec.state_dict)
        opponent_model.eval()
        pool[participant_id] = MCTS(
            model=opponent_model,
            c_puct=cfg_float("c_puct"),
            n_simulations=cfg_int("mcts_sims"),
            device=device,
            use_amp=cfg_bool("mcts_use_amp"),
            cache_size=max(0, cfg_int("mcts_cache_size")),
            leaf_batch_size=max(1, cfg_int("mcts_leaf_batch_size")),
            dirichlet_alpha=cfg_float("mcts_dirichlet_alpha"),
            dirichlet_frac=cfg_float("mcts_dirichlet_frac"),
            fpu_reduction=cfg_float("mcts_fpu_reduction"),
            virtual_loss=cfg_float("mcts_virtual_loss"),
            prior_uniform_mix=cfg_float("mcts_prior_uniform_mix"),
            forced_playout_k=cfg_float("mcts_forced_playout_k"),
            policy_target_prune_forced=cfg_bool("mcts_policy_target_prune_forced"),
        )
    return pool


def init_selfplay_process_worker(
    model_state_dict: dict[str, torch.Tensor],
    model_cfg: dict[str, int | float],
    c_puct: float,
    sims: int,
    opponent_specs: dict[str, tuple[dict[str, torch.Tensor], dict[str, int | float]]],
    worker_devices: tuple[str, ...],
) -> None:
    global _WORKER_MCTS
    global _WORKER_OPPONENT_MCTS_POOL
    from engine.mcts import MCTS
    from model.transformer import AtaxxTransformerNet

    model = AtaxxTransformerNet(
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        value_head_depth=int(model_cfg.get("value_head_depth", 1)),
        count_head_enabled=bool(model_cfg.get("count_head_enabled", 0.0)),
        transformer_pre_ln=bool(model_cfg.get("transformer_pre_ln", 0.0)),
        pos_embed_2d=bool(model_cfg.get("pos_embed_2d", 0.0)),
        patch_embed_conv=bool(model_cfg.get("patch_embed_conv", 0.0)),
        num_input_channels=int(model_cfg.get("num_input_channels", 11.0)),
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    worker_device = _resolve_selfplay_worker_device(worker_devices)
    use_amp = cfg_bool("mcts_use_amp") and worker_device.startswith("cuda")
    _WORKER_MCTS = MCTS(
        model=model,
        c_puct=c_puct,
        n_simulations=sims,
        device=worker_device,
        use_amp=use_amp,
        cache_size=max(0, cfg_int("mcts_cache_size")),
        leaf_batch_size=max(1, cfg_int("mcts_leaf_batch_size")),
        dirichlet_alpha=cfg_float("mcts_dirichlet_alpha"),
        dirichlet_frac=cfg_float("mcts_dirichlet_frac"),
        fpu_reduction=cfg_float("mcts_fpu_reduction"),
        prior_uniform_mix=cfg_float("mcts_prior_uniform_mix"),
        forced_playout_k=cfg_float("mcts_forced_playout_k"),
        policy_target_prune_forced=cfg_bool("mcts_policy_target_prune_forced"),
    )
    _WORKER_OPPONENT_MCTS_POOL = {}
    for participant_id, (opponent_state_dict, opponent_cfg) in opponent_specs.items():
        opponent_model = AtaxxTransformerNet(
            d_model=int(opponent_cfg["d_model"]),
            nhead=int(opponent_cfg["nhead"]),
            num_layers=int(opponent_cfg["num_layers"]),
            dim_feedforward=int(opponent_cfg["dim_feedforward"]),
            dropout=float(opponent_cfg["dropout"]),
            value_head_depth=int(opponent_cfg.get("value_head_depth", 1)),
            count_head_enabled=bool(opponent_cfg.get("count_head_enabled", 0.0)),
            transformer_pre_ln=bool(opponent_cfg.get("transformer_pre_ln", 0.0)),
            pos_embed_2d=bool(opponent_cfg.get("pos_embed_2d", 0.0)),
            patch_embed_conv=bool(opponent_cfg.get("patch_embed_conv", 0.0)),
            num_input_channels=int(opponent_cfg.get("num_input_channels", 11.0)),
        )
        opponent_model.load_state_dict(opponent_state_dict)
        opponent_model.eval()
        _WORKER_OPPONENT_MCTS_POOL[participant_id] = MCTS(
            model=opponent_model,
            c_puct=c_puct,
            n_simulations=sims,
            device=worker_device,
            use_amp=use_amp,
            cache_size=max(0, cfg_int("mcts_cache_size")),
            leaf_batch_size=max(1, cfg_int("mcts_leaf_batch_size")),
            dirichlet_alpha=cfg_float("mcts_dirichlet_alpha"),
            dirichlet_frac=cfg_float("mcts_dirichlet_frac"),
            fpu_reduction=cfg_float("mcts_fpu_reduction"),
            virtual_loss=cfg_float("mcts_virtual_loss"),
            prior_uniform_mix=cfg_float("mcts_prior_uniform_mix"),
            forced_playout_k=cfg_float("mcts_forced_playout_k"),
            policy_target_prune_forced=cfg_bool("mcts_policy_target_prune_forced"),
        )


def run_episode_in_process_worker(
    payload: tuple[int, str, str, int, bool, int, str],
) -> tuple[list[HistoryEntry], int, int, bool, int, int]:
    global _WORKER_MCTS
    global _WORKER_OPPONENT_MCTS_POOL
    if _WORKER_MCTS is None:
        raise RuntimeError("Worker MCTS is not initialized.")
    worker_mcts = cast("MCTS", _WORKER_MCTS)
    episode_seed, opponent_type, heuristic_level, model_player, add_noise, temp_threshold, checkpoint_id = payload
    rng = np.random.default_rng(seed=episode_seed)
    opponent_checkpoint_mcts = None
    if opponent_type == "checkpoint":
        pool = _WORKER_OPPONENT_MCTS_POOL or {}
        opponent_checkpoint_mcts = cast("MCTS | None", pool.get(checkpoint_id))
    return play_episode(
        mcts=worker_mcts,
        add_noise=add_noise,
        temp_threshold=temp_threshold,
        rng=rng,
        opponent_type=opponent_type,
        opponent_heuristic_level=heuristic_level,
        model_player=model_player,
        opponent_checkpoint_mcts=opponent_checkpoint_mcts,
    )


def update_stats(stats: dict[str, float | int], winner: int, turn_idx: int) -> None:
    stats["total_turns"] = int(stats["total_turns"]) + turn_idx
    if winner == 1:
        stats["wins_p1"] = int(stats["wins_p1"]) + 1
        return
    if winner == -1:
        stats["wins_p2"] = int(stats["wins_p2"]) + 1
        return
    stats["draws"] = int(stats["draws"]) + 1


def handle_parallel_selfplay_failure(exc: Exception) -> None:
    if cfg_bool("fail_on_selfplay_parallel_error"):
        raise RuntimeError(
            "Process self-play failed with parallel workers. "
            "Aborting instead of silently falling back to sequential mode.",
        ) from exc
    log(f"  Process self-play failed, falling back to sequential mode: {exc}")


__all__ = [
    "build_sequential_checkpoint_mcts_pool",
    "compute_action_probs",
    "handle_parallel_selfplay_failure",
    "init_selfplay_process_worker",
    "play_episode",
    "run_episode_in_process_worker",
    "select_action_idx",
    "update_stats",
]

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from game.actions import ACTION_SPACE
from game.board import AtaxxBoard
from game.types import Move


def model_move(board: AtaxxBoard, mcts: object | None) -> Move | None:
    from engine.mcts import MCTS

    if not isinstance(mcts, MCTS):
        raise RuntimeError("Model agent selected but MCTS is not initialized.")
    probs = mcts.run(board=board, add_dirichlet_noise=False, temperature=0.0)
    action_idx = int(np.argmax(probs))
    return ACTION_SPACE.decode(action_idx)


def model_move_with_diagnostics(
    board: AtaxxBoard,
    mcts: object | None,
) -> tuple[Move | None, dict[str, Any]]:
    """Return (move, diagnostics) so the arena HUD can render what MCTS saw.

    `diagnostics` keys:
      - top_moves: list[(action_idx, visits, value, prior)] sorted by visits desc.
      - root_value: float in [-1, 1], MCTS estimate for the player to move.
    """
    from engine.mcts import MCTS

    if not isinstance(mcts, MCTS):
        raise RuntimeError("Model agent selected but MCTS is not initialized.")
    brain = _model_brain_diagnostics(board=board, mcts=mcts)
    probs, root = mcts.run_with_root(
        board=board,
        root=None,
        add_dirichlet_noise=False,
        temperature=0.0,
    )
    action_idx = int(np.argmax(probs))
    move = ACTION_SPACE.decode(action_idx)
    diagnostics: dict[str, Any] = {
        "top_moves": MCTS.top_n_actions(root, n=3),
        "root_value": float(root.value()) if root is not None else 0.0,
        # Visit distribution sobre ACTION_SPACE para que el ReplayRecorder la
        # capture como policy target (mismo formato que self-play).
        "policy": probs.astype(np.float32, copy=False),
        "action_idx": action_idx,
        "brain": brain,
    }
    return move, diagnostics


def _legal_action_indices(board: AtaxxBoard) -> np.ndarray:
    valid_moves = board.get_valid_moves()
    if len(valid_moves) == 0:
        return np.array([ACTION_SPACE.pass_index], dtype=np.int64)
    return np.fromiter(
        (ACTION_SPACE.encode(move) for move in valid_moves),
        dtype=np.int64,
        count=len(valid_moves),
    )


def _model_brain_diagnostics(board: AtaxxBoard, mcts: object) -> dict[str, Any]:
    """Small, honest model introspection for the arena HUD.

    The heatmap is the L2 norm of final board-token activations after the
    transformer encoder, normalized to [0, 1]. It is not a causal explanation,
    but it shows which cells carry stronger internal signal on this position.
    """
    mcts_obj = cast(Any, mcts)
    model = mcts_obj.model
    device = mcts_obj.device
    use_amp = bool(mcts_obj.use_amp) and torch.device(device).type == "cuda"
    model.eval()

    obs = torch.from_numpy(board.get_observation()).unsqueeze(0).to(device)
    legal_indices = _legal_action_indices(board)
    action_mask = torch.zeros(
        (1, ACTION_SPACE.num_actions),
        device=device,
        dtype=obs.dtype,
    )
    action_mask[0, torch.from_numpy(legal_indices).to(device)] = 1.0

    activation: dict[str, torch.Tensor] = {}

    def capture_encoder(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> None:
        activation["encoded"] = output.detach()

    handle = model.encoder.register_forward_hook(capture_encoder)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else torch.no_grad()
    )
    try:
        with torch.inference_mode(), amp_ctx:
            policy_logits, value_tensor = model(obs, action_mask=action_mask)
    finally:
        handle.remove()

    legal_idx_tensor = torch.from_numpy(legal_indices).to(device)
    legal_policy = torch.softmax(policy_logits[0].index_select(0, legal_idx_tensor), dim=0)
    top_count = min(3, int(legal_indices.size))
    top_probs, top_offsets = torch.topk(legal_policy, k=top_count)
    top_policy = [
        (int(legal_indices[int(offset)]), float(prob))
        for prob, offset in zip(top_probs.detach().cpu(), top_offsets.detach().cpu(), strict=True)
    ]

    encoded = activation.get("encoded")
    if encoded is None:
        heatmap_np = np.zeros((7, 7), dtype=np.float32)
    else:
        board_tokens = encoded[0, 1:]
        heatmap = torch.linalg.vector_norm(board_tokens.float(), dim=1).reshape(7, 7)
        heatmap = heatmap - heatmap.min()
        denom = torch.clamp(heatmap.max(), min=1e-6)
        heatmap_np = (heatmap / denom).detach().cpu().numpy().astype(np.float32)

    return {
        "policy_top": top_policy,
        "value": float(value_tensor.squeeze().detach().float().cpu()),
        "activation_heatmap": heatmap_np,
        "architecture": {
            "tokens": 50,
            "layers": int(getattr(model.encoder, "num_layers", 0)),
            "heads": int(getattr(model.encoder.layers[0].self_attn, "num_heads", 0)),
            "d_model": int(model.pos_embed.shape[-1]),
        },
    }

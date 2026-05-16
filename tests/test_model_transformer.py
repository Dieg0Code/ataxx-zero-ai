from __future__ import annotations

import torch

from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE, OBSERVATION_CHANNELS
from model.transformer import AtaxxTransformerNet


def test_default_arch_forward_returns_policy_and_value() -> None:
    """Legacy arch: value_head_depth=1, count_head_enabled=False.

    Forward sigue devolviendo (policy, value) — back-compat con MCTS y agents.
    """
    model = AtaxxTransformerNet()
    x = torch.zeros(2, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy, value = model(x)
    assert policy.shape == (2, ACTION_SPACE.num_actions)
    assert value.shape == (2, 1)


def test_legacy_value_head_has_one_hidden_layer() -> None:
    model = AtaxxTransformerNet(value_head_depth=1)
    linear_layers = [m for m in model.value_head if isinstance(m, torch.nn.Linear)]
    # 1 oculta + 1 output = 2 linears.
    assert len(linear_layers) == 2


def test_deep_value_head_has_two_hidden_layers() -> None:
    model = AtaxxTransformerNet(value_head_depth=2)
    linear_layers = [m for m in model.value_head if isinstance(m, torch.nn.Linear)]
    # 2 ocultas + 1 output = 3 linears.
    assert len(linear_layers) == 3


def test_count_head_disabled_returns_zeros() -> None:
    model = AtaxxTransformerNet(count_head_enabled=False)
    x = torch.zeros(3, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    _, _, count = model.forward_with_count(x)
    assert count.shape == (3, 1)
    assert torch.equal(count, torch.zeros(3, 1))


def test_count_head_enabled_produces_nonzero_predictions() -> None:
    torch.manual_seed(0)
    model = AtaxxTransformerNet(count_head_enabled=True)
    # Initialize biases nonzero so count head produces nontrivial output.
    for module in model.count_head.modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            torch.nn.init.normal_(module.bias, std=0.5)
    x = torch.randn(4, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    _, _, count = model.forward_with_count(x)
    assert count.shape == (4, 1)
    assert not torch.equal(count, torch.zeros(4, 1))


def test_v11_capacity_config_constructs_and_forwards() -> None:
    """d_model=192, num_layers=8, dim_feedforward=768 con value_head_depth=2 y count_head."""
    model = AtaxxTransformerNet(
        d_model=192,
        nhead=8,
        num_layers=8,
        dim_feedforward=768,
        value_head_depth=2,
        count_head_enabled=True,
    )
    x = torch.zeros(2, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy, value, count = model.forward_with_count(x)
    assert policy.shape == (2, ACTION_SPACE.num_actions)
    assert value.shape == (2, 1)
    assert count.shape == (2, 1)


def test_action_mask_zeros_illegal_actions() -> None:
    model = AtaxxTransformerNet()
    x = torch.randn(1, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    mask = torch.zeros(1, ACTION_SPACE.num_actions)
    mask[0, 0] = 1.0
    policy_logits, _ = model(x, action_mask=mask)
    # Indices != 0 should be saturated negative (very low logits).
    illegal_max = float(policy_logits[0, 1:].max())
    assert illegal_max < -1e30

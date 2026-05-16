from __future__ import annotations

import numpy as np
import pytest
import torch

from data.symmetry import (
    D4_ELEMENTS,
    FORWARD_PERMUTATIONS,
    INVERSE_PERMUTATIONS,
    build_action_permutation,
    random_d4_augment_batch,
    transform_cell,
    transform_obs,
    transform_policy,
)
from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE, OBSERVATION_CHANNELS


def test_identity_is_noop_on_obs_and_policy() -> None:
    obs = torch.randn(2, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy = torch.randn(2, ACTION_SPACE.num_actions)
    assert torch.equal(transform_obs(obs, "identity"), obs)
    assert torch.equal(transform_policy(policy, "identity"), policy)


def test_rot90_four_times_returns_to_identity_obs() -> None:
    obs = torch.randn(1, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    result = obs
    for _ in range(4):
        result = transform_obs(result, "rot90")
    assert torch.allclose(result, obs)


def test_rot90_four_times_returns_to_identity_policy() -> None:
    policy = torch.randn(1, ACTION_SPACE.num_actions)
    result = policy
    for _ in range(4):
        result = transform_policy(result, "rot90")
    assert torch.allclose(result, policy)


def test_flip_h_twice_returns_to_identity() -> None:
    obs = torch.randn(1, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    result = transform_obs(transform_obs(obs, "flip_h"), "flip_h")
    assert torch.equal(result, obs)


def test_flip_v_twice_returns_to_identity() -> None:
    policy = torch.randn(1, ACTION_SPACE.num_actions)
    result = transform_policy(transform_policy(policy, "flip_v"), "flip_v")
    assert torch.allclose(result, policy)


@pytest.mark.parametrize("element", D4_ELEMENTS)
def test_inverse_permutation_is_inverse_of_forward(element: str) -> None:
    fwd = FORWARD_PERMUTATIONS[element]
    inv = INVERSE_PERMUTATIONS[element]
    n = ACTION_SPACE.num_actions
    composed_a = inv[fwd]
    composed_b = fwd[inv]
    expected = torch.arange(n, dtype=torch.long)
    assert torch.equal(composed_a, expected)
    assert torch.equal(composed_b, expected)


@pytest.mark.parametrize("element", D4_ELEMENTS)
def test_pass_action_is_fixed_point(element: str) -> None:
    fwd = build_action_permutation(element)
    assert int(fwd[ACTION_SPACE.pass_index]) == ACTION_SPACE.pass_index


@pytest.mark.parametrize("element", D4_ELEMENTS)
def test_cell_map_is_bijection(element: str) -> None:
    seen: set[tuple[int, int]] = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            seen.add(transform_cell(r, c, element))
    assert len(seen) == BOARD_SIZE * BOARD_SIZE


def test_rot90_cw_top_left_goes_to_top_right() -> None:
    # board CW rotation: (0,0) should end at (0, N-1).
    assert transform_cell(0, 0, "rot90") == (0, BOARD_SIZE - 1)
    # (1, 0) should end at (0, N-2): top row, second to last column.
    assert transform_cell(1, 0, "rot90") == (0, BOARD_SIZE - 2)


def test_flip_h_swaps_left_right() -> None:
    assert transform_cell(0, 0, "flip_h") == (0, BOARD_SIZE - 1)
    assert transform_cell(3, 2, "flip_h") == (3, BOARD_SIZE - 1 - 2)


def test_flip_d_is_transpose() -> None:
    assert transform_cell(0, 3, "flip_d") == (3, 0)
    assert transform_cell(2, 5, "flip_d") == (5, 2)


@pytest.mark.parametrize("element", D4_ELEMENTS)
def test_policy_argmax_tracks_cell_transform(element: str) -> None:
    """Si la mejor accion era (r1,c1,r2,c2), la accion mejor tras el
    transform debe ser (g(r1,c1), g(r2,c2)) -- la jugada fisica
    equivalente bajo la rotacion/flip."""
    rng = np.random.default_rng(seed=42)
    policy = torch.from_numpy(rng.random(ACTION_SPACE.num_actions).astype(np.float32))
    policy = policy.unsqueeze(0)
    best_a = int(policy.argmax(dim=-1).item())
    best_move = ACTION_SPACE.decode(best_a)

    transformed = transform_policy(policy, element)
    new_best_a = int(transformed.argmax(dim=-1).item())
    new_best_move = ACTION_SPACE.decode(new_best_a)

    if best_move is None:
        assert new_best_move is None
        return
    r1, c1, r2, c2 = best_move
    expected = (
        *transform_cell(r1, c1, element),
        *transform_cell(r2, c2, element),
    )
    assert new_best_move == expected


def test_obs_and_policy_transforms_are_consistent_for_one_hot() -> None:
    """One-hot policy: si la accion `a` codifica una jugada cuya celda
    src es (r, c), entonces tras transform la policy debe tener mass
    en la accion cuya celda src es g(r, c). El obs marca la celda src
    con un 1 en el channel 0; tras el transform, ese 1 debe estar en
    g(r, c) del obs transformado."""
    rng = np.random.default_rng(seed=123)
    a = int(rng.integers(0, ACTION_SPACE.num_actions - 1))  # not pass
    move = ACTION_SPACE.decode(a)
    assert move is not None
    r1, c1, r2, c2 = move

    obs = torch.zeros(1, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    obs[0, 0, r1, c1] = 1.0  # mark src in channel 0
    obs[0, 2, r2, c2] = 1.0  # mark dst in channel 2 (empty/target)
    policy = torch.zeros(1, ACTION_SPACE.num_actions)
    policy[0, a] = 1.0

    for element in D4_ELEMENTS:
        obs_t = transform_obs(obs, element)
        policy_t = transform_policy(policy, element)

        r1p, c1p = transform_cell(r1, c1, element)
        r2p, c2p = transform_cell(r2, c2, element)
        assert obs_t[0, 0, r1p, c1p].item() == pytest.approx(1.0), element
        assert obs_t[0, 2, r2p, c2p].item() == pytest.approx(1.0), element

        new_a = int(policy_t.argmax(dim=-1).item())
        new_move = ACTION_SPACE.decode(new_a)
        assert new_move == (r1p, c1p, r2p, c2p), element


def test_random_d4_augment_preserves_shape() -> None:
    obs = torch.randn(16, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy = torch.randn(16, ACTION_SPACE.num_actions)
    generator = torch.Generator().manual_seed(7)
    obs_aug, policy_aug = random_d4_augment_batch(obs, policy, generator=generator)
    assert obs_aug.shape == obs.shape
    assert policy_aug.shape == policy.shape


def test_random_d4_augment_empty_batch_is_noop() -> None:
    obs = torch.zeros(0, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy = torch.zeros(0, ACTION_SPACE.num_actions)
    obs_aug, policy_aug = random_d4_augment_batch(obs, policy)
    assert obs_aug.shape == obs.shape
    assert policy_aug.shape == policy.shape


def test_random_d4_augment_preserves_per_example_consistency() -> None:
    """Cada ejemplo recibe un elemento; si forzamos identity para todos
    (via stub), salida == entrada. No podemos forzar identity facil, pero
    podemos verificar que con un solo ejemplo, las normas se conservan."""
    rng = np.random.default_rng(seed=99)
    obs = torch.from_numpy(rng.random((4, OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE)).astype(np.float32))
    policy = torch.from_numpy(rng.random((4, ACTION_SPACE.num_actions)).astype(np.float32))
    obs_aug, policy_aug = random_d4_augment_batch(obs, policy)
    # Frobenius norms invariant per-example.
    obs_norms = torch.linalg.vector_norm(obs.flatten(1), dim=-1)
    obs_aug_norms = torch.linalg.vector_norm(obs_aug.flatten(1), dim=-1)
    assert torch.allclose(obs_norms, obs_aug_norms, atol=1e-5)
    pol_norms = torch.linalg.vector_norm(policy, dim=-1)
    pol_aug_norms = torch.linalg.vector_norm(policy_aug, dim=-1)
    assert torch.allclose(pol_norms, pol_aug_norms, atol=1e-5)

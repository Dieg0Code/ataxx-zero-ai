"""D4 symmetry augmentation for Ataxx (board 7x7).

El tablero es invariante bajo el grupo dihedral D4 (8 elementos: 4
rotaciones + 4 reflexiones). Aplicar augmentation por elemento de D4
multiplica la diversidad efectiva de un batch sin recolectar mas data.

Solo `obs` y `policy` se transforman. `value` y `count` son escalares
invariantes bajo D4 (cantidad de piezas no depende de orientacion).

Para el policy se precompute una permutacion de acciones por cada
elemento: si `g` mapea celdas `(r,c) -> (r',c')`, una accion
`(r1,c1,r2,c2)` se mapea a `(g(r1,c1), g(r2,c2))`. La accion de pass
queda fija.

La permutacion `forward_perm[a] = g(a)` indica donde "termina" cada
accion. Para aplicar el reordenamiento via gather (`out = in[idx]`)
se usa la inversa: `inverse_perm[g(a)] = a` y luego
`policy_new = policy.index_select(-1, inverse_perm)`.
"""
from __future__ import annotations

import numpy as np
import torch

from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE

D4_ELEMENTS: tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "flip_d",
    "flip_ad",
)


def transform_cell(r: int, c: int, element: str) -> tuple[int, int]:
    """Aplica un elemento de D4 a la celda (r, c) en un board NxN."""
    n = BOARD_SIZE - 1
    if element == "identity":
        return (r, c)
    if element == "rot90":
        return (c, n - r)
    if element == "rot180":
        return (n - r, n - c)
    if element == "rot270":
        return (n - c, r)
    if element == "flip_h":
        return (r, n - c)
    if element == "flip_v":
        return (n - r, c)
    if element == "flip_d":
        return (c, r)
    if element == "flip_ad":
        return (n - c, n - r)
    raise ValueError(f"Unknown D4 element: {element}")


def build_action_permutation(element: str) -> np.ndarray:
    """Permutacion forward: `perm[a] = encode(g(decode(a)))`.

    Es la posicion donde cada accion `a` se mapea bajo el elemento g.
    Pass se mantiene en `pass_index`.
    """
    num_actions = ACTION_SPACE.num_actions
    perm = np.zeros(num_actions, dtype=np.int64)
    for a in range(num_actions):
        move = ACTION_SPACE.decode(a)
        if move is None:
            perm[a] = ACTION_SPACE.pass_index
            continue
        r1, c1, r2, c2 = move
        r1p, c1p = transform_cell(r1, c1, element)
        r2p, c2p = transform_cell(r2, c2, element)
        perm[a] = ACTION_SPACE.encode((r1p, c1p, r2p, c2p))
    return perm


def _build_permutation_tables() -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    forward: dict[str, torch.Tensor] = {}
    inverse: dict[str, torch.Tensor] = {}
    for element in D4_ELEMENTS:
        fwd_np = build_action_permutation(element)
        fwd = torch.from_numpy(fwd_np).to(dtype=torch.long)
        inv = torch.empty_like(fwd)
        inv[fwd] = torch.arange(fwd.numel(), dtype=torch.long)
        forward[element] = fwd
        inverse[element] = inv
    return forward, inverse


FORWARD_PERMUTATIONS, INVERSE_PERMUTATIONS = _build_permutation_tables()


def transform_obs(obs: torch.Tensor, element: str) -> torch.Tensor:
    """Aplica el elemento espacial a un tensor (..., C, H, W).

    Usa primitivas de torch (`rot90`, `flip`, `transpose`) que ya
    expresan correctamente cada transformacion.
    """
    if element == "identity":
        return obs
    if element == "rot90":
        return torch.rot90(obs, k=-1, dims=(-2, -1))
    if element == "rot180":
        return torch.rot90(obs, k=2, dims=(-2, -1))
    if element == "rot270":
        return torch.rot90(obs, k=1, dims=(-2, -1))
    if element == "flip_h":
        return torch.flip(obs, dims=(-1,))
    if element == "flip_v":
        return torch.flip(obs, dims=(-2,))
    if element == "flip_d":
        return obs.transpose(-2, -1).contiguous()
    if element == "flip_ad":
        return torch.flip(obs.transpose(-2, -1).contiguous(), dims=(-1, -2))
    raise ValueError(f"Unknown D4 element: {element}")


def transform_policy(policy: torch.Tensor, element: str) -> torch.Tensor:
    """Reordena la policy bajo el elemento.

    `policy_new[forward_perm[a]] = policy[a]` se implementa via gather
    con la inversa: `policy_new = policy[..., inverse_perm]`.
    """
    if element == "identity":
        return policy
    inv = INVERSE_PERMUTATIONS[element].to(policy.device)
    return policy.index_select(-1, inv)


def random_d4_augment_batch(
    obs: torch.Tensor,
    policy: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica un elemento D4 aleatorio independiente a cada ejemplo del batch.

    Para mantener simplicidad, agrupa el batch por elemento y procesa
    cada grupo de una sola vez. Si el batch es grande, todas las 8
    operaciones suelen aparecer; si es pequeno, solo las que tocaron.
    """
    if obs.shape[0] != policy.shape[0]:
        raise ValueError("obs and policy batch sizes must match")
    batch_size = obs.shape[0]
    if batch_size == 0:
        return obs, policy

    if generator is None:
        choices = torch.randint(0, len(D4_ELEMENTS), (batch_size,))
    else:
        choices = torch.randint(
            0,
            len(D4_ELEMENTS),
            (batch_size,),
            generator=generator,
        )

    obs_out = obs.clone()
    policy_out = policy.clone()
    for elem_idx, element in enumerate(D4_ELEMENTS):
        if element == "identity":
            continue
        mask = choices == elem_idx
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        obs_out[idx] = transform_obs(obs[idx], element)
        policy_out[idx] = transform_policy(policy[idx], element)
    return obs_out, policy_out


__all__ = [
    "D4_ELEMENTS",
    "FORWARD_PERMUTATIONS",
    "INVERSE_PERMUTATIONS",
    "build_action_permutation",
    "random_d4_augment_batch",
    "transform_cell",
    "transform_obs",
    "transform_policy",
]

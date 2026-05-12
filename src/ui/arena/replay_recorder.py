"""Captura partidas humano-vs-IA como training examples y las persiste a NPZ.

El formato es identico al `buffer_iter_*.npz` que genera self-play, asi que el
training pipeline puede consumir estas partidas como imitation/contrastive data.

Por cada partida se escriben dos archivos al mismo directorio:
- `<stem>.npz`   con keys `observations`, `policies`, `values`.
- `<stem>.json`  sidecar con `ReplayMetadata` + estadisticas de la partida.

`ReplayRecorder` es opt-in: el game loop solo llama `record_move` y `finalize`
si recibio una instancia. No hay efectos colaterales globales.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from game.board import AtaxxBoard

# La tupla cruda que vamos juntando — coincide con `HistoryEntry` (4-tuple) del
# training pipeline para poder pasarla directo a `history_to_examples`.
_HistoryEntry = tuple[np.ndarray, np.ndarray, int, float]


@dataclass
class ReplayMetadata:
    """Contexto de la partida — se serializa al JSON sidecar."""

    mode: str                       # "tournament" | "play"
    player_ai: str                  # codename del modelo (ej. "liga")
    starter: str                    # "human" | "ai"
    mcts_sims: int
    tournament_id: str | None = None
    player_human: str | None = None
    round_idx: int | None = None
    match_idx: int | None = None
    is_tiebreak: bool = False
    p1_label: str = ""              # label tal cual lo ve la arena (para auditar)
    p2_label: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ReplayRecorder:
    """Acumula (obs, policy, player) por movimiento y los guarda al finalizar."""

    def __init__(self, save_path: Path, metadata: ReplayMetadata) -> None:
        self._save_path = Path(save_path)
        self._metadata = metadata
        self._history: list[_HistoryEntry] = []

    @property
    def metadata(self) -> ReplayMetadata:
        return self._metadata

    @property
    def n_moves(self) -> int:
        return len(self._history)

    def record_move(
        self,
        *,
        board_before: AtaxxBoard,
        policy: np.ndarray,
        action_idx: int,
        player: int,
    ) -> None:
        """Registra el movimiento que esta a punto de aplicarse.

        `board_before` es el board ANTES del `step()`. La observation se captura
        ahi para que coincida con lo que el modelo vio al decidir. `policy` es la
        distribucion target sobre `ACTION_SPACE.num_actions` (one-hot para humano,
        visit-distribution para IA).
        """
        del action_idx  # se reconstruye al cargar el NPZ si hace falta
        observation = board_before.get_observation().copy()
        policy_arr = np.asarray(policy, dtype=np.float32)
        self._history.append((observation, policy_arr, int(player), 0.0))

    def finalize(self, *, winner: int, forced_draw: bool = False) -> Path | None:
        """Convierte la historia en training examples y persiste a disco.

        Returns:
            Path del NPZ creado, o `None` si no hubo movimientos (forfeit) — en
            ese caso no se escribe nada y el caller debe confiar en el estado
            externo (TournamentState) para registrar el forfeit.
        """
        if not self._history:
            return None

        from training.reward_runtime import history_to_examples

        examples = history_to_examples(
            self._history,
            winner=int(winner),
            forced_draw=bool(forced_draw),
        )
        if not examples:
            return None

        observations = np.stack([e[0] for e in examples]).astype(np.float32, copy=False)
        policies = np.stack([e[1] for e in examples]).astype(np.float32, copy=False)
        values = np.array([e[2] for e in examples], dtype=np.float32)

        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._save_path,
            observations=observations,
            policies=policies,
            values=values,
        )

        sidecar = self._save_path.with_suffix(".json")
        payload = {
            **self._metadata.to_dict(),
            "n_moves": len(self._history),
            "winner": int(winner),
            "forced_draw": bool(forced_draw),
            "value_dtype": "float32",
            "shapes": {
                "observations": list(observations.shape),
                "policies": list(policies.shape),
                "values": list(values.shape),
            },
        }
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self._save_path


def one_hot_policy(action_idx: int, num_actions: int) -> np.ndarray:
    """One-hot policy target — usado para movimientos humanos en imitation."""
    policy = np.zeros(num_actions, dtype=np.float32)
    if 0 <= action_idx < num_actions:
        policy[action_idx] = 1.0
    return policy


__all__ = (
    "ReplayMetadata",
    "ReplayRecorder",
    "one_hot_policy",
)

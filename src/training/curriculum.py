from __future__ import annotations

import numpy as np

from agents.heuristic import HEURISTIC_LEVELS

CurriculumMix = dict[str, float]

# v13 onward: heurísticas se sacan del self-play. La mix solo controla la
# proporción self/heuristic/random del muestreo de oponentes. La porción de
# self se redirige a la league con probabilidad league_selfplay_checkpoint_prob,
# así que "self" termina siendo ~mitad red actual, ~mitad checkpoints viejos.
#
# Los sub-buckets heu_* siguen presentes con valor uniforme dummy para que
# sample_opponent_from_curriculum no rompa si en algún experimento futuro se
# vuelve a subir heuristic > 0 — pero por default la fase >=1 mantiene
# heuristic=0.0 y nunca se samplea.


def _heuristic_dummy_mix() -> dict[str, float]:
    uniform = 1.0 / float(len(HEURISTIC_LEVELS))
    return {f"heu_{level}": uniform for level in HEURISTIC_LEVELS}


def get_curriculum_mix(iteration: int) -> CurriculumMix:
    """Phase-based opponent mix. Heuristics out, league via league_selfplay_checkpoint_prob."""
    if iteration <= 12:
        base = {"self": 0.95, "heuristic": 0.0, "random": 0.05}
    elif iteration <= 30:
        base = {"self": 0.97, "heuristic": 0.0, "random": 0.03}
    elif iteration <= 90:
        base = {"self": 0.98, "heuristic": 0.0, "random": 0.02}
    else:
        base = {"self": 0.99, "heuristic": 0.0, "random": 0.01}
    base.update(_heuristic_dummy_mix())
    return base


def sample_opponent_from_curriculum(
    rng: np.random.Generator,
    iteration: int,
) -> tuple[str, str]:
    mix = get_curriculum_mix(iteration)

    opp_labels = ("self", "heuristic", "random")
    opp_probs = np.asarray([mix["self"], mix["heuristic"], mix["random"]], dtype=np.float64)
    opp_probs = opp_probs / float(np.sum(opp_probs))
    opponent_type = str(rng.choice(opp_labels, p=opp_probs))

    heu_labels = HEURISTIC_LEVELS
    heu_probs = np.asarray([mix[f"heu_{level}"] for level in heu_labels], dtype=np.float64)
    heu_probs = heu_probs / float(np.sum(heu_probs))
    heuristic_level = str(rng.choice(heu_labels, p=heu_probs))
    return opponent_type, heuristic_level

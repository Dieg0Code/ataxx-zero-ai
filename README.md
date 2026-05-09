# ataxx-zero-ai

Motor de IA para Ataxx separado del proyecto web (`ataxx-zero`).

Incluye:

- `src/game/` — reglas del juego (tablero 7x7, acciones, serialización).
- `src/engine/mcts.py` — búsqueda de árbol Monte Carlo.
- `src/model/` — red transformer policy/value (`system.py`, `transformer.py`) y compatibilidad de checkpoints. Postmortems en `src/model/docs/postmortem/`.
- `src/agents/` — agentes (random, heurístico con niveles, model+MCTS) y selector/registry.
- `src/training/` — loop de entrenamiento, self-play, league/curriculum, eval gating, warmup, callbacks.
- `src/inference/` — servicio de inferencia (torch + ONNX), runtimes de duelos y ligas entre checkpoints.
- `src/data/` — dataset y replay buffer.
- `src/ui/arena/` — app Pygame para jugar/espectar localmente.
- `checkpoints/` — modelos guardados.
- `scripts/fetch_run_history.py` — baja metadata de runs en HF Hub a un CSV local para análisis post-mortem.
- `Ataxx_Zero_Colab.ipynb`, `Ataxx_Zero_Kaggle.ipynb` — notebooks.

## Setup

```bash
uv sync                                      # base (numpy + torch + lightning)
uv sync --group train --group dev            # entrenamiento + tests
uv sync --group ui --group dev               # arena Pygame
uv sync --group inference --group dev        # ONNX runtime
uv sync --group export --group dev           # exportar a ONNX
uv sync --all-groups
```

## Entrenamiento

```bash
uv run python train.py --iterations 2 --episodes 8 --epochs 1 --sims 80 --batch-size 64 --save-every 1 --verbose
```

Defaults razonables están en `src/training/config_runtime.py`. Para correr en Kaggle ver el notebook `Ataxx_Zero_Kaggle.ipynb`.

## Arena local (Pygame)

```bash
uv run python scripts/play_pygame.py --mode play --agent1 human --agent2 model --ckpt checkpoints/last.ckpt --sims 220
```

## Tests

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyrefly check
```

## Estado

- Recién separado de `ataxx-zero` (web). Sin historial git heredado.
- El API web quedó temporalmente sin backend de inferencia hasta decidir cómo se conectan los dos proyectos (cliente HTTP, paquete compartido, etc.).

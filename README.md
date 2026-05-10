# ataxx-zero-ai

Motor de IA para Ataxx (juego tipo Reversi en tablero 7×7). Entrenamiento estilo AlphaZero (transformer policy/value + MCTS + self-play) y arena local para jugar / mirar partidas.

---

## Setup rápido

```bash
uv sync --all-groups
```

Si solo te interesa una parte:

```bash
uv sync --group ui                     # solo arena (jugar / espectar)
uv sync --group train --group dev      # entrenamiento + tests
uv sync --group inference              # ONNX runtime (inferencia rápida)
```

---

## Jugar y mirar partidas (arena Pygame)

Lo entretenido primero. Todos estos comandos abren una ventana gráfica.

### Vos contra una heurística

```bash
uv run python scripts/play_pygame.py --mode play --opponent heuristic --level hard
```

Niveles disponibles: `easy, normal, hard, apex, gambit, sentinel` (de menos a más fuerte).

### Vos contra un modelo entrenado

```bash
uv run python scripts/play_pygame.py --mode play --opponent model --ckpt checkpoints/policy_spatial_v6_iter_180.pt --sims 200
```

`--sims` controla cuán fuerte juega el modelo (más simulaciones MCTS = juego más profundo, más lento). Razonable: 100-400.

### Elegir de qué lado jugás

```bash
uv run python scripts/play_pygame.py --mode play --opponent model --ckpt checkpoints/policy_spatial_v6_iter_180.pt --human-side p2
```

`p1` = juego rojo (empieza), `p2` = juego azul.

### Mirar dos modelos jugar entre sí (spectate)

```bash
uv run python scripts/play_pygame.py --mode spectate \
  --p1-agent model --ckpt1 checkpoints/policy_spatial_v6_iter_180.pt \
  --p2-agent model --ckpt2 checkpoints/policy_spatial_v7_iter_140.pt \
  --sims 200
```

### Mirar modelo vs heurística

```bash
uv run python scripts/play_pygame.py --mode spectate \
  --p1-agent model --ckpt1 checkpoints/policy_spatial_v6_iter_180.pt \
  --p2-agent heuristic --level2 sentinel --sims 200
```

### Mirar dos heurísticas pelearse

```bash
uv run python scripts/play_pygame.py --mode spectate \
  --p1-agent heuristic --level1 hard \
  --p2-agent heuristic --level2 sentinel
```

### Mirar random vs heurística (sanity check)

```bash
uv run python scripts/play_pygame.py --mode spectate \
  --p1-agent random \
  --p2-agent heuristic --level2 hard
```

---

## Evaluar checkpoints (sin UI, headless)

Para medir performance objetiva. Devuelven W/L/D, scores y opcionalmente JSON.

### Modelo vs heurísticas (varios niveles, batch)

```bash
uv run python scripts/eval_checkpoint_vs_heuristic.py \
  --checkpoint checkpoints/policy_spatial_v6_iter_180.pt \
  --levels easy,normal,hard,apex,gambit,sentinel \
  --games 64 --sims 160
```

Da el perfil completo del modelo. Si el score varía mucho entre niveles (alto en uno, bajo en otro de dificultad parecida), es señal de **opponent exploitation** — ver `src/model/docs/postmortem/05/`.

### Modelo A vs Modelo B (head-to-head)

```bash
uv run python scripts/compare_checkpoints.py \
  --checkpoint-a checkpoints/policy_spatial_v6_iter_180.pt \
  --checkpoint-b checkpoints/policy_spatial_v7_iter_140.pt \
  --games 32 --sims 160
```

Útil para responder "¿cuál es genuinamente más fuerte?" — independiente de heurísticas, no se puede sobreajustar.

---

## Entrenar

### Smoke test local (rápido, para verificar que todo arranca)

```bash
uv run python train.py --iterations 2 --episodes 8 --epochs 1 --sims 80 --batch-size 64 --save-every 1 --verbose
```

### Run real

Defaults razonables están en `src/training/config_runtime.py`. Para correr en serio se recomienda Kaggle (T4×2 o P100 gratis, 4-6h por sesión):

```bash
# Opción A: Kaggle vía CLI (sube y encola un run)
PYTHONUTF8=1 kaggle kernels push -p .

# Opción B: subir Ataxx_Zero_Kaggle.ipynb a kaggle.com manualmente,
# configurar GPU + Secret HF_TOKEN, Run All.
```

El notebook es la única "config" — todos los hiperparámetros viven en su primera celda.

### Override de hiperparámetros vía CLI

```bash
uv run python train.py \
  --hf --hf-repo-id usuario/repo --hf-run-id mi_run \
  --iterations 180 --episodes 20 --sims 160 \
  --eval-every 6 --eval-games 64 \
  --opp-self 0.65 --opp-heuristic 0.30 --opp-random 0.05
```

O vía JSON file (lo que usa el notebook):

```bash
uv run python train.py --config-json mi_config.json --hf
```

`uv run python train.py --help` lista todos los flags.

---

## Análisis post-mortem de runs

Si entrenaste con HF persistence (recomendado), todos los `metadata_iter_*.json` quedan en HuggingFace Hub:

```bash
uv run python scripts/fetch_run_history.py policy_spatial_v6
# Genera runs_history/policy_spatial_v6/policy_spatial_v6_history.csv
```

El CSV trae todas las métricas por iteración: scores de eval por nivel, train loss/lr/policy_accuracy, replay buffer size, etc. Útil para graficar curvas.

---

## Exportar a ONNX (inferencia rápida fuera de Python)

```bash
uv run python scripts/export_model_onnx.py \
  --checkpoint checkpoints/policy_spatial_v6_iter_180.pt \
  --output ataxx_v6.onnx
```

```bash
uv run python scripts/check_onnx_parity.py \
  --checkpoint checkpoints/policy_spatial_v6_iter_180.pt \
  --onnx ataxx_v6.onnx
```

---

## Tests y dev

```bash
uv run pytest                          # toda la suite (~15s)
uv run pytest tests/test_agents_*.py   # solo agentes
uv run ruff check . && uv run ruff format --check .
uv run pyrefly check
```

---

## Estructura del repo

```
src/
  game/        reglas Ataxx (tablero, movimientos, serialización)
  engine/      MCTS
  model/       transformer policy/value + Lightning module
  agents/      human, random, heurísticas, modelo+MCTS
  training/    self-play, eval gating, league, warmup, callbacks
  inference/   torch + ONNX, duelos entre checkpoints
  ui/arena/    arena Pygame
  data/        replay buffer + dataset

scripts/       entry points (play, eval, export, compare, fetch_history)
checkpoints/   modelos guardados (no en git)
tests/         pytest
src/model/docs/postmortem/   análisis de runs anteriores (PM01-05)
```

---

## Recursos / contexto

- **Postmortems**: cada run que falló o reveló algo útil tiene su análisis en `src/model/docs/postmortem/`. PM05 es el más reciente y explica el problema de **opponent exploitation** que descubrimos en v6.
- **Notebook de producción**: `Ataxx_Zero_Kaggle.ipynb`. La primera celda es la única que tocás entre runs.

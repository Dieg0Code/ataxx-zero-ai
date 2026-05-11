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

> Cada generación de modelo tiene un apodo (`liga`, `centinela`, `bogo`, …) que
> se puede usar en `--ckpt` directamente. Ver [`docs/MODELS.md`](docs/MODELS.md)
> para el catálogo completo y `uv run python scripts/list_models.py` para el
> ranking actual.

### Tú contra una heurística

```bash
uv run python scripts/play_pygame.py --mode play --opponent heuristic --level hard
```

Niveles disponibles: `easy, normal, hard, apex, gambit, sentinel` (de menos a más fuerte).

### Tú contra un modelo entrenado

```bash
uv run python scripts/play_pygame.py --mode play --opponent model --ckpt liga --sims 200
```

`--ckpt` acepta apodo (`liga`, `centinela`, `bogo`…), versión (`v8`), alias
(`latest`, `best`) o un path absoluto. `--sims` controla cuán fuerte juega el
modelo (más simulaciones MCTS = juego más profundo, más lento). Razonable: 100-400.

### Elegir de qué lado juegas

```bash
uv run python scripts/play_pygame.py --mode play --opponent model --ckpt liga --human-side p2
```

`p1` = juego rojo (empieza), `p2` = juego azul.

### Hot-seat: dos humanos en la misma pantalla

```bash
uv run python scripts/play_pygame.py --mode play --p1-agent human --p2-agent human
```

Cada jugador clickea su jugada por turno. Útil para enseñar las reglas o jugar con alguien al lado.

### Mirar dos modelos jugar entre sí (spectate)

```bash
uv run python scripts/play_pygame.py --mode spectate \
  --p1-agent model --ckpt1 liga \
  --p2-agent model --ckpt2 centinela \
  --sims 200
```

### Mirar modelo vs heurística

```bash
uv run python scripts/play_pygame.py --mode spectate \
  --p1-agent model --ckpt1 liga \
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

### Atajos de teclado en la arena

| Tecla       | Acción                                                                |
|-------------|-----------------------------------------------------------------------|
| `space`     | Pausar / reanudar                                                     |
| `s`         | Avanzar una jugada (solo cuando está pausado, modo step)              |
| `1` `2` `4` | Velocidad 1× / 2× / 4× (acelera los delays de la IA)                  |
| `p`         | Captura de pantalla a `arena_screenshots/arena_<ms>.png`              |
| `r`         | Reiniciar partida                                                     |
| `q`         | Salir                                                                 |

El HUD lateral muestra en tiempo real las top-3 jugadas que considera la IA con barras de visitas MCTS, la probabilidad de victoria de ROJO en porcentaje grande, historial de jugadas y un mini-gráfico de evaluación. El récord W/L/D persiste entre sesiones en `~/.ataxx_arena_stats.json`.

---

## Evaluar checkpoints (sin UI, headless)

Para medir performance objetiva. Devuelven W/L/D, scores y opcionalmente JSON.

### Modelo vs heurísticas (varios niveles, batch)

```bash
uv run python scripts/eval_checkpoint_vs_heuristic.py \
  --checkpoint centinela \
  --levels easy,normal,hard,apex,gambit,sentinel \
  --games 64 --sims 160
```

Da el perfil completo del modelo. Si el score varía mucho entre niveles (alto en uno, bajo en otro de dificultad parecida), es señal de **opponent exploitation** — ver `src/model/docs/postmortem/05/`.

### Modelo A vs Modelo B (head-to-head)

```bash
uv run python scripts/compare_checkpoints.py \
  --checkpoint-a centinela \
  --checkpoint-b amnesia \
  --games 32 --sims 160
```

Útil para responder "¿cuál es genuinamente más fuerte?" — independiente de heurísticas, no se puede sobreajustar.

### Round-robin entre todas las generaciones

```bash
uv run python scripts/round_robin.py --games 8 --sims 80
```

Enfrenta cada par del registry (21 pares con 7 generaciones por default — excluye `aprendiz-tardio` para no duplicar identidad). Persiste el score head-to-head en `checkpoints/registry.json` y computa un `round_robin.score` agregado por modelo. La métrica `rr` es la más honesta porque no se puede sobreajustar — un modelo solo gana si genuinamente juega mejor que los otros.

### Listar y rankear todas las generaciones

```bash
uv run python scripts/list_models.py                  # tabla rankeada (default: combined)
uv run python scripts/list_models.py --metric rr      # solo head-to-head
uv run python scripts/list_models.py --metric composite  # solo vs heurísticas
uv run python scripts/list_models.py --full           # con lore + hparams
uv run python scripts/eval_all_checkpoints.py         # poblar evals faltantes vs heurísticas
```

Ver el catálogo completo de generaciones (apodos + historia + lore) en [`docs/MODELS.md`](docs/MODELS.md).

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
  --checkpoint checkpoints/policy_spatial_v8_iter_180.pt \
  --output ataxx_liga.onnx
```

```bash
uv run python scripts/check_onnx_parity.py \
  --checkpoint checkpoints/policy_spatial_v8_iter_180.pt \
  --onnx ataxx_liga.onnx
```

(Estos scripts todavía esperan path explícito; si querés que acepten codename del registry, es un PR de 2 líneas.)

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
  model/       transformer policy/value, Lightning, registry de generaciones
  agents/      human, random, heurísticas, modelo+MCTS
  training/    self-play, eval gating, league, warmup, callbacks
  inference/   torch + ONNX, duelos entre checkpoints
  ui/arena/    arena Pygame (HUD táctico, postfx CRT, fonts retro, stats)
  data/        replay buffer + dataset

scripts/
  play_pygame.py             arena (jugar / espectar)
  eval_checkpoint_vs_heuristic.py   eval contra heurísticas
  compare_checkpoints.py     duelo head-to-head A vs B
  list_models.py             tabla rankeada de generaciones
  eval_all_checkpoints.py    gauntlet automatizado para todos los modelos
  round_robin.py             round-robin head-to-head entre generaciones
  fetch_run_history.py       baja metadata de runs desde HF Hub
  export_model_onnx.py       export para inferencia rápida
  check_onnx_parity.py       verifica que ONNX y torch dan los mismos outputs

checkpoints/   modelos guardados + registry.json (catálogo con apodos)
docs/MODELS.md catálogo humano-legible de las generaciones
tests/         pytest
src/model/docs/postmortem/   análisis de runs anteriores (PM01-05)
```

---

## Recursos / contexto

- **Postmortems**: cada run que falló o reveló algo útil tiene su análisis en `src/model/docs/postmortem/`. PM05 es el más reciente y explica el problema de **opponent exploitation** que descubrimos en v6.
- **Notebook de producción**: `Ataxx_Zero_Kaggle.ipynb`. La primera celda es la única que se toca entre runs.

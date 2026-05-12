# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Lo que es este repo

Motor de IA para Ataxx (7×7, estilo Reversi). Entrenamiento AlphaZero (transformer policy/value + MCTS + self-play), arena Pygame local, registry de generaciones con apodos (`bogo`, `centinela`, `liga`…). Python 3.10+, `uv` para deps, training en Kaggle T4×2.

Idioma: **español neutro** para docs, postmortems y commits. Sin voseo. Inglés solo en código (identificadores, etc.).

## Gates obligatorias antes de pushear (training scope)

CI corre `.github/workflows/ci-train.yml` cuando se modifica cualquier path en: `train.py`, `src/{training,engine,model,game,data}/**`, `scripts/**`, `tests/test_mcts_numerics.py`, `tests/test_training_*.py`, `pyproject.toml`, `uv.lock`, `pyrefly.toml`.

**Antes de `git push`**, correr las 4 gates locales en este orden:

```bash
# 1. Política de longitud (max 500 líneas por archivo en training scope)
uv run python scripts/check_python_max_lines.py --max-lines 500 \
  --path train.py --path src/training --path src/engine --path src/model \
  --path src/game --path src/data \
  --path tests/test_mcts_numerics.py \
  --path tests/test_training_bootstrap.py \
  --path tests/test_training_checkpointing.py \
  --path tests/test_training_curriculum.py \
  --path tests/test_training_monitor.py \
  --path tests/test_training_step_numerics.py \
  --path scripts/export_model_onnx.py --path scripts/check_onnx_parity.py

# 2. Linter
uv run ruff check train.py src/engine src/model src/game src/data tests scripts

# 3. Type-checker
uv run pyrefly check train.py src tests

# 4. Tests del training scope (rápidos, no full suite)
uv run pytest -q \
  tests/test_mcts_numerics.py tests/test_training_bootstrap.py \
  tests/test_training_checkpointing.py tests/test_training_config_validation_runtime.py \
  tests/test_training_curriculum.py tests/test_training_monitor.py \
  tests/test_training_selfplay_runtime.py tests/test_training_step_numerics.py \
  tests/test_training_trainer_runtime.py
```

Después de `git push`:
```bash
gh run list --limit 3                    # ver corridas recientes
gh run view <id> --log-failed            # diagnóstico si falla
```

**Solo después de CI verde**, recién entonces dar luz verde a un training run.

Si `ruff check --fix` modifica archivos, los autofixes **no se stagean solos** — `git add` después o el siguiente push vuelve a fallar.

Para scripts ad-hoc (CLIs, generadores PBIR, validadores) hay `[tool.ruff.lint.per-file-ignores]` que relaja annotations en `scripts/*.py`. No agregar nuevas reglas a la blacklist sin justificación — los scripts críticos (`export_model_onnx.py`, `check_onnx_parity.py`) están en el length policy y deberían respetar todas las reglas.

## Comandos comunes

### Setup
```bash
uv sync --all-groups                    # todo
uv sync --group ui                      # solo arena
uv sync --group train --group dev       # training + tests
```

### Training
```bash
# Smoke test local (verifica que arranca)
uv run python train.py --iterations 2 --episodes 8 --epochs 1 --sims 80 --batch-size 64 --save-every 1 --verbose

# Producción en Kaggle (config en Ataxx_Zero_Kaggle.ipynb, celda 1)
PYTHONUTF8=1 kaggle kernels push -p .
```

### Tests
```bash
uv run pytest                           # toda la suite (~20s, 185 tests)
uv run pytest tests/test_agents_*.py    # filtrado por glob
uv run pytest tests/test_engine_mcts.py::test_top_n_actions  # un test específico
```

### Arena (Pygame UI)
```bash
uv run python scripts/play_pygame.py --mode play --opponent model --ckpt liga --sims 200
uv run python scripts/play_pygame.py --mode spectate --p1-agent model --ckpt1 liga --p2-agent model --ckpt2 centinela
uv run python scripts/play_pygame.py --mode play --p1-agent human --p2-agent human   # hot-seat
```

`--ckpt` acepta apodo (`liga`, `centinela`, `bogo`…), version (`v8`), alias (`latest`, `best`), o path.

### Eval / ranking
```bash
uv run python scripts/list_models.py                    # tabla rankeada
uv run python scripts/eval_checkpoint_vs_heuristic.py --checkpoint liga --games 24
uv run python scripts/compare_checkpoints.py --checkpoint-a liga --checkpoint-b centinela --games 32
uv run python scripts/round_robin.py --games 8 --sims 80
uv run python scripts/eval_all_checkpoints.py           # gauntlet completo
```

### Análisis post-run
```bash
uv run python scripts/fetch_run_history.py policy_spatial_v9   # baja metadata de HF Hub
uv run python scripts/build_master_csv.py                       # consolida CSV para Power BI
uv run python scripts/validate_powerbi.py                       # valida visuals PBIR antes de abrir
uv run python scripts/simulate_powerquery.py                    # simula conversion de tipos
```

## Arquitectura

### Pipeline de training (`train.py` + `src/training/`)

`train.py` orquesta el loop principal. Cada iteración:
1. **Self-play** (`selfplay_runtime.execute_self_play`) — genera N episodios usando MCTS guiado por el modelo actual + mezcla de oponentes (`opponent_self/heuristic/random_prob` + league).
2. **Training** (`trainer_runtime`) — entrena el transformer con replay buffer sobre los episodios nuevos.
3. **Eval gating** (`eval_gating.compute_regression_gate`) — cada `eval_every` iters evalúa contra heurísticas. Si baja más de `eval_regression_delta` por `eval_regression_patience` iters, restaura el best checkpoint (`restore_best_on_regression`).
4. **HF persist** — checkpoint + metadata se suben a HF Hub si `hf_enabled`.

Config canónica está en `src/training/config_runtime.py:DEFAULTS`. Override vía CLI flags o `--config-json file.json`. La validación bloquea combinaciones inseguras (ver `config_validation_runtime.py`).

**Hparams críticos** (modificar con cuidado, leer postmortems primero):
- `opponent_heuristic_prob` alto → opponent exploitation (PM05). Bajar de 0.5 hacia 0 para self-play puro.
- `hf_reset_iteration=True` + `warmup_games=0` → regresión catastrófica (PM04). El validador lo bloquea.
- `league_selfplay_checkpoint_prob` — % de self-play contra checkpoints viejos del propio modelo. Menos explotable que heurísticas.

### Engine MCTS (`src/engine/mcts.py`)
Estándar AlphaZero — PUCT, expansión por leaf evaluation, virtual loss off. Postmortem 02 documenta el bug de desempate determinista que se fixeó. `top_n_actions()` expone visitas/value/prior para el HUD.

### Model (`src/model/`)
- `transformer.py`: arquitectura policy + value heads, espacial (output policy con src/dst projections, no MLP plana). Bogo (v1) usa policy MLP legacy — `checkpoint_compat.py:strip_orig_mod_prefix + has_legacy_flat_policy_head + drop_legacy_policy_head` maneja la compatibilidad.
- `registry.py`: catálogo de generaciones en `checkpoints/registry.json`. Acepta resolución por codename, version, filename, o aliases (`latest`, `best`, `default`).
- `system.py`: Lightning module que envuelve el transformer.

### Agents (`src/agents/`)
`human`, `random`, `heuristic` (6 niveles: easy/normal/hard/apex/gambit/sentinel), `model` (MCTS + red). El factory `selector.py` mapea config → agent instance.

### Arena (`src/ui/arena/`)
Pygame app con HUD táctico (top-3 MCTS, win prob, eval timeline), modo spectator, sprites pixelados opcional, CRT overlay opcional, atajos de teclado, stats persistentes. Window 1280×720 auto-scaled para pantallas pequeñas.

### Registry de generaciones

`checkpoints/registry.json` (gitignored) es la fuente única de verdad. Cada entrada:
```json
{
  "codename": "liga", "version": "v8", "file": "policy_spatial_v8_iter_180.pt",
  "lore": "<1-2 frases con gancho a lo que pasó>",
  "postmortem": "src/model/docs/postmortem/05/README.md",
  "eval": {"composite": 0.667, "vs_heuristic": {...}, "round_robin": {...}, "head_to_head": {...}}
}
```

Apodos asignados después del run, no antes — el nombre captura lo que pasó. Postmortems están en `src/model/docs/postmortem/0{1..5}/README.md`. PM05 es la lectura más importante para entender por qué mucho composite no implica modelo fuerte (opponent exploitation).

## Reglas no-obvias

- **`checkpoints/`, `runs_history/`, `hf_checkpoints/`, `kaggle_logs/`, `.claude/` están gitignored.** Los checkpoints y CSVs viven solo localmente o en HF Hub. El `registry.json` también es gitignored (no commitearlo).
- **El notebook `Ataxx_Zero_Kaggle.ipynb` se sube a Kaggle via `kaggle kernels push -p .`** (CLI separada de git). El notebook clona el repo en Kaggle desde GitHub — por eso pushear código antes de correr es obligatorio.
- **El validador de bootstrap** (`config_validation_runtime.validate_bootstrap_warmup_config`) bloquea `hf_reset_iteration=true + warmup_games=0`. Si querés bootstrap canónico: `reset_iteration=True + warmup_games>=320`.
- **Auto-memory en `.claude/projects/.../memory/`** — preferencias y feedback del usuario persisten entre sesiones. Diego prefiere español neutro en docs.
- **PowerBI dashboards en `runs_history/ataxx_zero.pbip`** — generados por `scripts/build_powerbi_pages.py`. Validar con `scripts/validate_powerbi.py` antes de abrir el .pbip (visualtype whitelist + campos prohibidos en `objects.*`).
- **MCP de Power BI Modeling disponible** para editar el semantic model live (medidas, columnas, partition M). Después de cambios via MCP, `ExportToTmdlFolder` para persistir a disco.
- **Pyrefly tiene un shim de compatibilidad para Windows** en CI (`.venv/Scripts/python.exe` linkea a `bin/python`).

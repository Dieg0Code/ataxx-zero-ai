# Handoff — sesión Claude Code

Documento para arrancar la próxima sesión sin re-leer todo. Generado al separar el motor de IA del proyecto web `ataxx-zero`.

## 1. Qué se hizo en la sesión anterior

### Separación de proyectos
- **Web app (queda en)** `C:\Users\Diego Obando\ai-lab\ataxx-zero` — `src/api/`, `web/`, `alembic/`, `supabase/`, `Dockerfile`, `docker-compose.yml`, tests del API, scripts `bootstrap_*`, `db.ps1`, workflow de Railway.
- **AI engine (este proyecto)** `C:\Users\Diego Obando\dev\ataxx-zero-ai` — todo lo de IA: `src/{agents,engine,model,training,inference,data,game}`, `src/ui/arena/`, `train.py`, `train_improved.py`, notebooks Colab/Kaggle, `Dockerfile.train`, scripts AI (`play_pygame`, `check_onnx_parity`, `export_model_onnx`, `compare_checkpoints`), tests AI, `infra/runpod-train/`, `checkpoints/`, postmortems en `src/model/docs/postmortem/`, workflows de RunPod, `.tmp_*` (scratch de Kaggle/HF).
- Git: repo nuevo (sin historial heredado), 2 commits iniciales.
- **API web quedó roto** a propósito — sus módulos `gameplay/` y `matches/` aún importan `agents`, `inference`, `game` (que ya no están). Cuando volvamos al web hay que decidir: cliente HTTP al AI, paquete compartido, o API "thin" sin AI.

### Postmortems leídos
Los 4 están en `src/model/docs/postmortem/0{1,2,3,4}/README.md`. Recap en sección 3.

## 2. Tarea abierta: investigar por qué `policy_spatial_v7` empeoró respecto de `v6`

PM04 ya tiene una **hipótesis sólida** ("regresión post-bootstrap"), pero falta corroborar contra código y metadatos. Plan acordado, en orden:

1. **Verificar configs reales de v6 vs v7** comparando los `*.metadata.json` en `checkpoints/`. ¿De verdad v7 tenía `warmup_games=0`, `hf_reset_iteration=true`? ¿Qué otros params cambiaron?
2. **Revisar el código actual** de los 3 puntos críticos que PM04 menciona:
   - `src/training/checkpointing.py` → lógica de `hf_reset_iteration`
   - `src/training/curriculum.py` → `get_curriculum_mix()` y la fase temprana (iter ≤12: 88% heuristic / 10% self / 2% random)
   - `src/training/loop_runtime.py` → `sample_recent_mix(...)` con `train_recent_fraction=0.7`
3. **Buscar diferencias no mencionadas en PM04** entre v6 y v7: cambios de arquitectura, hyperparams del optimizador, MCTS, eval. (Sin historial git heredado en el repo nuevo, hay que ir contra el repo viejo `C:\Users\Diego Obando\ai-lab\ataxx-zero` para `git log` entre fechas, o contra el commit hash conservado en metadata si lo tienen.)
4. **Si hay tensorboard logs de v7**, mirar curvas: ¿value loss disparado? ¿entropía de policy colapsó? ¿`eval_composite` bajó suave o hubo knee point?

**Siguiente paso concreto sugerido**: arrancar por (1)+(2) — leer los `.metadata.json` y los tres archivos clave, traer evidencia concreta, después decidir si se profundiza en (3) o (4).

## 3. Recap de los 4 postmortems (compacto)

| PM | Modelo | Síntoma | Causas raíz (resumen) |
|---|---|---|---|
| **01** | iter ~40 inicial | Solo aprendió a oscilar piezas | 12 bugs acumulados: sin detección de repetición, cold-start con 80% self-play, Adam reseteado por iteración, leak train/val, `loss = loss_v + loss_pi` (sin balance), máscara legal solo en inferencia, sin canal `half_moves`, `temp_threshold=15` (greedy demasiado pronto), Dirichlet alpha fijo 0.3, batch MCTS sin virtual loss, scheduler coseno reseteado, post-LN. Todos arreglados. |
| **02** | v2 / v4 | Apertura colapsaba a una sola línea; "siempre gana azul" hasta en self-duel | Desempates determinísticos en MCTS: `_select_child` con `>` (no reemplazaba al primero), `_get_action_probs(temp=0)` con `argmax`. Más policy de root casi plana → siempre primer índice del action space. **Fix**: desempates aleatorios neutros entre opciones empatadas. |
| **03** | hasta v5 | Loop "estable" pero modelos no mejoraban; `eval_composite=0.0`, head-to-head contaminado | 9 fugas: cache MCTS sin `half_moves`, presión de repetición no en obs, heurísticas no enseñaban policy (solo turnos del modelo en buffer), training sin máscara legal, self-play cayendo a CPU, `sample_recent_mix` con reemplazo agresivo, checkpoints rivales no enseñaban policy, observación pobre (expandida a **11 canales**: piezas, vacíos, progreso, repetición, destinos clone/jump propios+rivales, piezas activas), formato eval/league inconsistente. **v6_iter_180** = primer salto real (40-0 vs v2, 24-0 vs v4, `best_eval_score=0.806`). |
| **04** | **v7 vs v6** | v7 partió de v6 pero terminó peor: `0.6875` vs `0.806`, benchmark directo `16-23-1` para v7 | **No es catastrophic forgetting puro**, es **regresión por reset de distribución post-bootstrap**: pesos buenos + `hf_reset_iteration=true` (replay vaciado) + `warmup_games=0`, `warmup_epochs=0` (v6 tenía 320/4) + curriculum reseteado a iteración 0 (fase de 88% heurístico que no le pega a un modelo maduro) + `train_recent_fraction=0.7` amplificó deriva. **Lección**: un checkpoint fuerte no se define solo por sus pesos — también por la distribución que lo formó. |

## 4. Reglas operativas que salieron de PM04 (no repetir)

1. **No combinar por defecto**: `hf_bootstrap_run_id != ""` + `hf_reset_iteration=true` + `warmup_games=0` + `warmup_epochs=0`. Esa combinación deja al modelo sin red de seguridad.
2. **Si querés continuar una corrida buena**: heredá también iteración + replay + opponent mix maduro. No abras `run_id` nuevo si no es necesario.
3. **Si abrís branch nuevo**: warmup ≠ 0, idealmente con mezcla de teachers/ejemplos heredados del padre al inicio.
4. **Benchmark obligatorio contra el padre** cada N iteraciones. Si no lo supera o cae por debajo, hay regresión, no falta de iteraciones.

## 5. Estado del usuario y contexto humano

- Diego está aprendiendo IA con este proyecto.
- Antes usaba Codex 5.4; el último entrenamiento empeoró y nunca lograron self-play desde cero "limpio".
- Tono: español, directo, prefiere respuestas concisas.
- Quiere mejorar el motor de IA, no la app web (por eso la separación).

## 6. Comandos útiles para arrancar

```bash
uv sync --group train --group dev            # entrenamiento + tests
uv sync --group inference --group dev        # ONNX runtime
uv run pytest                                # correr tests
uv run pytest tests/test_training_curriculum.py -q
```

Notebooks Colab/Kaggle: `Ataxx_Zero_Colab.ipynb`, `Ataxx_Zero_Kaggle.ipynb`.
Postmortems: `src/model/docs/postmortem/0{1,2,3,4}/README.md`.
Checkpoints + metadata: `checkpoints/`.

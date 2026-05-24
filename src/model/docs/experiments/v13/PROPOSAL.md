# v13 — Proposal

**Estado:** pre-registrado antes del primer eval.
**Fecha de registro:** 2026-05-24.
**Run id (HF):** `policy_spatial_v13`.
**Commit del código:** `f352223`.
**Lanzado en Kaggle:** sí, kernel version 83.

> Documento pre-registrado. Las predicciones cuantitativas y los
> criterios de falsación se acuerdan **antes** de tener resultados,
> para evitar sesgo de confirmación. Cuando llegue `RESULTS.md`, se
> comparan predicciones vs observaciones sin retocar la hipótesis.

---

## Hipótesis principal

Sacar las heurísticas del self-play (`opponent_heuristic_prob = 0.0`)
elimina la causa principal del techo histórico del repo — opponent
exploitation — y permite que un modelo con capacidad suficiente
(d_model=384, 12 capas) construya juego posicional generalizable
en lugar de patrones tácticos contra bots fijos.

La predicción operativa: **v13 debería pasar de manera sostenida
los peaks de v11/v12 en h2h vs `liga` y mostrar progreso en el nivel
táctico `gambit` que el curriculum estático nunca movió** (gambit
clavado en 0.156 en v12 a pesar de subir su weight al 0.20).

## Justificación

- PM01-PM10 documentan opponent exploitation como falla recurrente
  desde la generación 2. Cambiar las proporciones del curriculum
  no alcanzó (v12 falló a pesar del rebalanceo).
- AlphaZero canon es 100% self-play. Diego le ganó 43-6 al iter
  126 de v12 que era el peak h2h del repo — evidencia directa de
  que las métricas internas contra heurísticas no transfieren a
  humanos.
- Training loss seguía bajando en v12 (2.74 → 2.38) mientras eval
  oscilaba plano: la red sí aprendía, pero aprendía lo equivocado.
  Conclusion: el problema es **señal de entrenamiento**, no
  optimización ni capacidad.

## Predicciones cuantitativas

Todas en términos de métricas que ya emitimos a wandb/HF (no requiere
instrumentación nueva).

### Loss (señal temprana de salud)
- **iter 30:** `train/loss` < 2.5. (Modelo grande aprende algo; si
  no, hay bug.)
- **iter 60:** `train/loss` < 2.2. (Comparable a v12 al mismo paso,
  ajustando por que v13 es más grande.)
- **iter 120:** `train/loss` < 1.95.

### h2h vs `liga` (señal de fuerza relativa real)
- **iter 100:** ≥ 0.45 en al menos 1 eval.
- **iter 200:** ≥ 0.55 sostenido (2 evals consecutivas).
- **iter 300:** ≥ 0.65 sostenido (3 evals consecutivas).
- **iter 600 (final):** ≥ 0.70 peak en al menos 1 eval, con
  estabilidad ≥ 0.65 en las últimas 5 evals.

### Per-level diagnóstico (las heurísticas que v12 nunca dominó)
- **iter 300, vs gambit:** ≥ 0.30. (v12 clavado en 0.156; subir
  a 0.30 sería evidencia de mejora estratégica genuina, no solo
  ganarle a `liga`.)
- **iter 300, vs normal:** ≥ 0.40. (v12 oscilaba 0.26-0.28.)
- **iter 600, gambit:** ≥ 0.45.
- **iter 600, normal:** ≥ 0.55.

### Eval humano (señal final, no automatizable)
- En algún iter ≥ 300, Diego juega 10 partidas casuales contra el
  candidate actual: **gana ≤ 4**.
- En iter ≥ 500: **gana ≤ 2**.
- Éxito de proyecto: Diego pierde mayoría, partidas se sienten
  cerradas.

## Criterios de falsación

Cualquiera de estos mata la hipótesis principal y obliga a un
postmortem que reconsidere el diagnóstico de raíz:

1. **iter 60 con train/loss ≥ 2.5 sostenido** → la arquitectura
   grande no está aprendiendo, hay bug (no es problema de
   curriculum). Verificar config y MCTS targets antes de seguir.
2. **iter 200 con h2h vs liga < 0.30 en 3 evals** → la league
   está dominando, algo del setup de oponentes está mal o pure
   self-play colapsó. Diagnóstico inmediato, posible abort.
3. **iter 600 con gambit ≤ 0.20** → el opponent exploitation
   contra heurísticas no era la causa raíz real. Hay otra
   limitación (capacidad, MCTS depth, board representation) que
   v13 no tocó. Reconsiderar la teoría base.
4. **Diego pierde < 50% de partidas en iter ≥ 500** → métricas
   numéricas mejoraron pero el gap heurística-humano persiste.
   Sugiere que las heurísticas Y la league comparten algún sesgo
   que el self-play replica.

## Criterios de aborto temprano (interrupción del run)

Distintos a los de falsación: estos cortan el run para no quemar
compute en algo claramente roto.

- iter 30: `train/loss` > 2.8 (no debe estar peor que arranque).
- iter 60: 0 partidas ganadas en ningún eval (forced draws o
  losses totales).
- 3 iteraciones consecutivas con OOM o crash en self-play.

## Variables cambiadas (confounders conocidos)

8 cambios simultáneos respecto a v12. **No podremos atribuir el
resultado a una sola palanca a partir de este run solo.**

| Palanca | v12 | v13 |
|---|---|---|
| `opponent_heuristic_prob` | 0.5 | 0.0 |
| `league_selfplay_checkpoint_prob` | 0.35 | 0.55 |
| `d_model` × `num_layers` × `nhead` | 192×8×8 | 384×12×12 |
| `dim_feedforward` | 768 | 1536 |
| `mcts_sims` | 320 | 800 |
| `c_puct` | 1.5 | 1.25 |
| Dirichlet α | 0.3 (hardcoded) | 0.10 (config) |
| `restore_best_on_regression` | true | false |
| `baseline_h2h_min_score` | 0.40 | 0.25 |
| `eval_absolute_min_iteration` | 80 | 150 |
| `eval_composite_uses_h2h_only` | (no existía) | true |

**Plan B (ablations) si v13 funciona:** correr v13.1 aislando
**pure self-play** (volviendo a arch v12, mcts_sims=320) para
confirmar que esa palanca, no la arquitectura ni más sims, es la
que movió la aguja. Costo: una sesión Kaggle adicional.

**Plan B si v13 falla:** ver `RESULTS.md` y `JOURNAL.md` para
identificar el primer signo de divergencia y formar la siguiente
hipótesis. NO escribir el postmortem antes de revisar los datos
crudos.

## Variables controladas (idénticas a v12)

- Pretrain humano (`pretrain_dataset_path`, `pretrain_epochs=3`).
- `human_batch_fraction = 0.20`, `human_value_mask = true`.
- `symmetry_augmentation = true`.
- `value_head_depth = 2`, `count_head_enabled = true`,
  `count_loss_coeff = 0.1`.
- `eval_every = 6`, `eval_games = 64`, `eval_sims = 256`.
- `eval_heuristic_levels = "easy,normal,hard,apex,gambit,sentinel"`.
- `seed = 42`.

## Diseño del experimento

- **Tipo:** observacional con grupo de control histórico (v11.2/v12
  como referencia).
- **N:** 1 run, 600 iters, 1 seed. Estamos limitados por compute,
  no podemos hacer múltiples seeds.
- **Power:** baja para detectar efectos sutiles. Diseño solo
  detecta efectos grandes (que es lo que esperamos — si v13 es
  marginalmente mejor que v12, no concluímos nada).
- **Stopping rule:** 600 iters totales o aborto por criterios
  arriba, lo que ocurra primero.

## Riesgos al diseño del experimento

- **Comparación con v11.2/v12 NO es ablation real** — el código
  cambió en muchos sitios entre runs (curator, monitor, eval
  flow). Si v13 mejora, parte podría ser por bugfixes ocultos.
- **`seed = 42` único** — un solo seed no captura la varianza
  natural del run. Una sesión "afortunada" puede falsear
  positivo.
- **Pretrain humano agrega data leak parcial** — el modelo arranca
  con información que v11/v12 también tenían pero no de forma
  controlada. No es tabula rasa estricto.
- **Las heurísticas (oponentes de eval) son las mismas que se
  sacaron del training** — entonces eval contra ellas en v13 mide
  generalización OOD, no in-distribution. Esto es intencional pero
  hay que recordarlo al interpretar per-level scores.

## Plan de monitoreo durante el run

Anotar en `JOURNAL.md` (append-only):
- Cada eval que llega (compositor, h2h, per-level).
- Cualquier crash, OOM, o resume forzado por timeout.
- Anomalías visuales en wandb (loss spikes, gradiente explosivo,
  lr distinto al esperado).
- Decisiones tomadas durante el run con justificación (e.g. "iter
  X bajé tal hparam porque...").

Frecuencia mínima: una entrada por sesión Kaggle (cada ~12h).

## Plan de análisis

Cuando termine el run (o se aborte):

1. Llenar `RESULTS.md` con tabla predicciones vs observado.
2. Marcar cada predicción como: **cumple / no cumple / inconcluso**.
3. Si cumple la mayoría: declarar v13 como nuevo baseline,
   asignar codename de oro, planear ablation v13.1.
4. Si no cumple: postmortem en `src/model/docs/postmortem/12/`
   con foco en QUÉ predicción se quebró primero y qué nos dice
   sobre la hipótesis. Sin retocar las predicciones.

## Datos crudos esperados

- HF: `runs/policy_spatial_v13/model_iter_NNN.pt` y
  `runs/policy_spatial_v13/model_iter_NNN.metadata.json` cada iter.
- wandb: project `ataxx-zero`, run `policy_spatial_v13`.
- Kaggle logs persistidos vía `kaggle_logs/`.
- CSV consolidado vía `scripts/fetch_run_history.py policy_spatial_v13`.

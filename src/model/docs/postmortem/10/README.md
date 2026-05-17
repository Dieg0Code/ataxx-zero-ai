# Postmortem 10 — `umbral` (v11.2, iter 114)

## TL;DR

Segunda generación post-fix del count head (después de asedio). Rompió
oficialmente el plateau histórico de h2h vs liga: pico **0.781** en
iter 84 (asedio máximo fue 0.422; ningún modelo previo había pasado
0.55). Composite pico **0.486** en iter 114 — el último eval antes
de que el kernel de Kaggle muriera por timeout de 12h. La trayectoria
mostró oscilación clara (96→102→108→114: 0.465→0.393→0.328→0.486),
no plateau. El run NO había terminado de subir cuando se cortó.

Pero en h2h local con MCTS 128 sims CPU, iter 96 sacó 0.359 vs liga
y 0.297 vs paralelo — **muy por debajo** del 0.625-0.734 reportado
en kaggle. Discrepancia de ~0.30 puntos entre GPU/AMP/seed kaggle y
CPU/fp32/seed local. iter 114 no se evaluó vs paralelo local.

Cualitativamente, Diego ganó vs umbral en arena pero reportó:
> "Le gané, pero siento que juega algo mejor, nose, es un vibe pero eso."

Mejor sensación que asedio en lo subjetivo, sin alcanzar el nivel
de "amenaza real" que asedio sí dio. Composite per-level revela el
problema raíz: el modelo le gana a hard/sentinel/apex pero pierde a
easy/normal/gambit — opponent exploitation invertido. Apodo: **umbral**.
Cruzó el umbral del plateau histórico (primer modelo de la familia
que pasa h2h vs liga > 0.50 con muestreo amplio) pero quedó parado
en él, sin entrar de lleno al territorio de "bestia".

## Contexto

PM09 cerró `asedio` (v11.1) documentando que la pipeline v11 funciona
pero el absolute gate abortaba prematuro. v11.2 aplicó dos cambios:

```json
"baseline_h2h_min_score": 0.45 → 0.40
"eval_absolute_min_iteration": 36 → 80
```

Y un bugfix crítico del notebook: cell 1 sobreescribía `hf_run_id`
hardcodeando `RUN_NAME = "policy_spatial_v11"`. Sin el fix, los
uploads de umbral habrían caído en la carpeta HF de v11/v11_1
mezclados. El fix (`a42247d`) hizo que cell 6 leyera `hf_run_id`
desde el json — single source of truth.

Mismo config arquitectónico que asedio: d_model=192, 8 capas, value
head profundo, count head, symmetry aug D4, pretrain humano,
replay humano 20%, league seed (paralelo > liga > espejismo >
amnesia > centinela).

## Trayectoria — composite y h2h

| iter | composite | h2h vs liga | notas |
|---|---|---|---|
|   6 | 0.040 | — | arranque |
|  12 | 0.147 | — | aceleración rápida |
|  18 | 0.275 | — | meseta inicial |
|  24 | 0.299 | — | |
|  30 | 0.260 | — | |
|  36 | 0.293 | — | |
|  42 | 0.273 | — | |
|  48 | 0.302 | — | |
|  54 | 0.275 | — | |
|  60 | 0.311 | — | |
|  66 | 0.299 | — | |
|  72 | 0.324 | — | |
|  78 | 0.444 | — | ★ salto cualitativo |
|  84 | 0.430 | **0.781** | ★ pico h2h (primer eval del gate) |
|  90 | 0.401 | 0.656 | |
|  96 | 0.465 | 0.625 | |
| 102 | 0.393 | 0.734 | |
| 108 | 0.328 | 0.688 | regresión activa (5 fails consecutivos) |
| 114 | **0.486** | 0.625 | ★ pico composite (último eval antes de timeout) |

Tres observaciones:

1. **El run NO estaba plateauando**. iter 114 estableció nuevo pico
   composite Y nuevo pico hard (0.727 vs 0.594 anterior) Y nuevo pico
   gambit (0.375 vs 0.297 anterior). 5 fails consecutivos (102, 108)
   resultaron ser oscilación, no asíntota.
2. **h2h_fail_count quedó en 0** durante todo el run. El gate vs
   liga (threshold 0.40) nunca estuvo en peligro. Los fails eran
   solo de composite (regresión vs best_eval_score).
3. **Kaggle se murió en el mejor momento** — timeout 12h, no abort de
   gate. Si hubiera tenido otras 12h, probablemente seguía subiendo.

## Eval @ iter 114 (vs 6 niveles × 64 partidas = 384 totales)

| Métrica | umbral @ iter 114 | asedio @ iter 36 | lastre @ iter 18 |
|---|---|---|---|
| composite | **0.486** | 0.236 | 0.044 |
| wins totales | 187 | 89 | 17 |
| eval_score_easy | 0.469 | 0.391 | 0.000 |
| eval_score_normal | 0.266 | 0.219 | 0.000 |
| eval_score_hard | **0.727** | 0.180 | 0.016 |
| eval_score_apex | 0.492 | 0.227 | 0.016 |
| eval_score_gambit | 0.375 | 0.172 | 0.000 |
| eval_score_sentinel | **0.586** | 0.227 | 0.047 |

umbral le **gana** a hard, sentinel, apex (al menos empate).
Pero **pierde** a easy (0.469), normal (0.266) y gambit (0.375).
El patrón es consistente con la asimetría del opponent mix:

```
Heuristic mix en self-play:
  easy=0.00   normal=0.02   hard=0.24
  apex=0.28   gambit=0.18   sentinel=0.28
```

El modelo entrenó ~0% del tiempo vs easy y ~2% vs normal. Es
matemáticamente esperable que pierda contra estilos que casi no vio.

## h2h vs liga local — discrepancia con kaggle

Para validar el peak h2h reportado por el run, evaluamos iter 96
localmente con 64 partidas vs liga y vs paralelo (mismo seed range,
sims=128, CPU/fp32):

| Comparación | Kaggle (T4, fp16, AMP) | Local (CPU, fp32) | Δ |
|---|---|---|---|
| iter 96 vs liga | 0.625 (40W-24L) | 0.359 (23W-41L) | -0.266 |
| iter 102 vs liga | 0.734 (47W-17L) | (no medido) | — |
| iter 96 vs paralelo | (no medido) | 0.297 (19W-45L) | — |
| iter 114 vs liga | 0.625 (40W-24L) | (no medido) | — |

La diferencia de 0.266 puntos entre kaggle y local es enorme — fuera
del IC95% de 64 partidas (±0.12). Posibles causas:

1. **AMP fp16 vs fp32**: el modelo evaluado en GPU con autocast puede
   producir distribuciones de policy ligeramente distintas que en CPU
   fp32. En posiciones de balance fino, eso cambia jugadas.
2. **Seed**: el seed del eval kaggle y el local son distintos, y MCTS
   con 128 sims es ruidoso. Pero ambos son n=64; el ruido no explica
   0.27 puntos.
3. **mcts_use_amp=true**: el config v11.2 tiene `mcts_use_amp=true`
   para self-play, posiblemente también activo en eval. Mi h2h local
   no usa AMP.

Conclusión honesta: **el h2h reportado por kaggle probablemente está
inflado vs lo que el modelo realmente puede entregar fuera del setup
de entrenamiento**. El número del registry queda como 0.625 (el del
kernel) pero hay que tomarlo con suspicacia.

## La anécdota cualitativa

Diego cargó umbral (iter 114) en arena con sims=300 y jugó vs el
modelo. Resultado:

> "Le gané, pero siento que juega algo mejor, nose, es un vibe pero eso."

Comparado con asedio (iter 41) que dio:

> "Perdí una pero porque hice missclick. En el resto, gané. Pero sentí
> la presión."

Asedio fue subjetivamente **más amenazante** que umbral, a pesar de
que umbral tiene composite 2× mejor y mucho mejor h2h vs liga. Eso
sugiere dos cosas:

1. El **vibe humano** no escala linealmente con composite. asedio
   entrenó menos iters (41 vs 114) y quizás juega más "humano" —
   con la inocencia de no haber sobreaprendido patrones específicos.
2. **umbral juega muy "rápido"**: avg_turns 28 contra liga, vs ~36
   de asedio. Cierra partidas en pocos turnos cuando va ganando.
   Eso da menos "presión sostenida" al humano que un modelo que
   complica el medio juego.

Para próximas generaciones, el feedback humano debería ser tracked
junto al composite. Una gauntlet humana de 10 partidas (registrar
W/L/feedback corto por modelo) capturaría dimensión que el composite
no.

## Lecciones aprendidas

1. **El run se cortó en su mejor momento**. El kaggle timeout de 12h
   es un constraint duro; v11.2 estaba haciendo nuevo pico cuando
   murió el kernel. Para v12 hay que considerar `iterations`
   ajustado para terminar dentro de las 12h con margen, o aceptar
   que el segundo run (resume) es parte del workflow.

2. **Composite mean enmascara opponent exploitation**. composite 0.486
   suena "competitivo" pero esconde que el modelo PIERDE contra
   easy, normal y gambit. Un eval gate por `min(per-level)` en lugar
   de `mean` forzaría balance.

3. **El curriculum heurístico estaba desbalanceado**. 0% easy + 2%
   normal = el modelo nunca aprende a defenderse contra esos estilos.
   v12 debería subir easy/normal a ~15% cada uno mínimo. Es probable
   que sea el cambio de mayor bang/buck.

4. **kaggle h2h ≠ h2h local**. ~0.27 puntos de diferencia entre el
   eval del kernel y el eval CPU local sobre iter 96. AMP fp16
   probablemente es responsable. Para validar champions de ahora
   en adelante, h2h local debería ser obligatorio antes de promover
   a champion del registry.

5. **El feedback humano es una métrica auxiliar real**. asedio se
   "sintió" mejor que umbral aunque umbral lo gana en todas las
   métricas numéricas. Formalizar un eval humano (10 partidas, log
   subjetivo) por generación.

6. **5 fails consecutivos puede ser ruido, no asíntota**. La
   regresión vista en iters 102-108 se resolvió en iter 114 con
   nuevo pico. Patience=2 (asedio fue abortado por esto) puede ser
   demasiado severo; v12 podría considerar patience=4-5.

7. **El bugfix del notebook valió oro**. Sin el fix de cell 6 que
   lee `hf_run_id` del json, los uploads de umbral se habrían
   mezclado con asedio en la carpeta HF y reconstruir la trayectoria
   habría sido imposible.

## Apodo

**umbral** — del español: línea límite que marca el paso a otro
estado. umbral cruzó el techo histórico del plateau (primer modelo
del repo en superar h2h vs liga > 0.55 con n=64) pero se quedó
parado ahí, sin entrar de lleno al espacio de "bestia". Composite
0.486 (vs 0.85 que tiene liga en su eval de 3 niveles); pierde a
easy/normal/gambit. Es el techo del plateau visto desde el otro
lado: la prueba de que la familia v11 puede ir más arriba, pero no
sin cambios deliberados en curriculum y exploración.

## Checkpoint persistido

`checkpoints/registered/umbral.pt` — copia del iter 114 (último
checkpoint subido a HF antes del timeout kaggle). 14MB, arch v11
(d_model=192, value_head_depth=2, count_head_enabled=True).
Cargable via `resolve('umbral')` o `--ckpt umbral`.

## Próximo paso — v12

v12 va a apuntar específicamente al techo per-level que dejó umbral.
Tres palancas:

1. **Curriculum balanceado** — subir easy y normal del 0/2% al
   ~15% cada uno. Bajar hard/apex/sentinel al ~15-18%. Esperado:
   el modelo deja de perder contra easy/normal a costa de cesión
   marginal vs hard.

2. **mcts_sims 160 → 320** — duplicar la exploración táctica.
   AlphaZero original usó 800; 160 es bajo. Esperado: cierra el
   agujero de gambit (táctica corta que el modelo no anticipa) y
   mejora calidad general de las jugadas del self-play.

3. **eval gate por min(per-level)** — opcional. Forzaría que el
   modelo no pueda ser "best" mientras le pierda a alguna heurística.
   Más severo, runs más largos, pero garantiza balance.

Costo estimado: 2× sims doble el tiempo de self-play. Compensar con
`episodes_per_iter: 12 → 8`. Net throughput: ~30% más lento. 300
iters ≈ 15-16h, dos sesiones de kaggle.

Hipótesis a validar en v12: si el curriculum balanceado + más sims
no rompen el techo de easy/normal, el problema es arquitectónico
o la cantidad de data humana. En ese caso, v13 considera d_model
más grande o data augmentation humana adicional.

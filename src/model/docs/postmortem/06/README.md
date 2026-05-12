# AtaxxZero — Post-mortem 06: el modelo v9 (espejismo) bootstrappeó desde liga y empeoró

> **Resumen:** `policy_spatial_v9_iter_180` (apodado *espejismo*) arrancó desde
> los pesos de `liga` (v8, nuestro mejor modelo) con la intuición de "subir
> el techo": más self-play, más league play, menos heurísticas, asumiendo que
> liga ya había aprendido lo suficiente del estilo greedy. El resultado es
> peor que la base. Pierde 3-12 head-to-head vs liga, baja el composite de
> 0.94 a 0.69, y replica con detalles distintos el patrón de PM05: domina
> heurísticas duras, se desploma contra simples. La lección es directa: si
> tu base ya es fuerte, el nuevo run tiene que verla más, no menos.

---

## Contexto: por qué se intentó este bootstrap

Después de PM05 quedó claro que `centinela` (v6) sufría de **opponent
exploitation**: ganaba bien contra sentinel/apex pero perdía contra easy,
normal y gambit. `liga` (v8) introdujo el league system y diversificó las
heurísticas en self-play (`opponent_heuristic_prob=0.50` con mezcla
easy/normal/hard), lo que subió el composite a 0.94 vs heurísticas y dejó
record perfecto en round-robin (0.94, 45-3-0).

La hipótesis para v9 fue: *"liga ya internalizó el suelo anti-greedy de las
heurísticas. Si bootstrappeamos desde sus pesos y subimos la fracción de
self-play + league, vamos a refinar el techo táctico sin perder el suelo."*

El plan se materializó en estos cambios de config:

| Hparam                              | v8 (liga) | v9 (espejismo) |
|-------------------------------------|-----------|----------------|
| `hf_bootstrap_run_id`               | `""`      | `policy_spatial_v8` |
| `hf_reset_iteration`                | `false`   | **`true`** |
| `opponent_self_prob`                | 0.45      | **0.60** |
| `opponent_heuristic_prob`           | **0.50**  | 0.25 |
| `opponent_random_prob`              | 0.05      | **0.15** |
| `league_selfplay_checkpoint_prob`   | 0.35      | **0.55** |
| `eval_heuristic_levels`             | hard,apex,sentinel | easy,normal,hard,apex,gambit,sentinel |

Todo lo demás idéntico (transformer 128/6/8, batch 224, lr 3e-4, mcts_sims
160, 180 iters).

## Metodología de evaluación

```bash
uv run python scripts/eval_checkpoint_vs_heuristic.py \
    --checkpoint checkpoints/policy_spatial_v9_iter_180.pt \
    --levels hard,apex,sentinel --games 12 --sims 80

uv run python scripts/compare_checkpoints.py \
    --checkpoint-a checkpoints/policy_spatial_v9_iter_180.pt \
    --checkpoint-b liga --games 16 --mcts-sims 80

uv run python scripts/compare_checkpoints.py \
    --checkpoint-a checkpoints/policy_spatial_v9_iter_180.pt \
    --checkpoint-b centinela --games 16 --mcts-sims 80
```

Eval rápida (12 partidas/nivel, SE≈0.14) suficiente para confirmar el patrón.
Para una eval definitiva con 64 partidas/nivel (SE≈0.06) hay que correr
`eval_all_checkpoints.py`, pero los gaps ya son grandes.

## Resultados

### v9 vs heurísticas (12 games, sims=80)

| Nivel    | W  | L  | D | Score | Lectura |
|----------|----|----|---|-------|---------|
| hard     |  9 |  3 | 0 | 0.750 | sólido |
| apex     |  9 |  3 | 0 | 0.750 | sólido |
| sentinel |  7 |  5 | 0 | 0.583 | bajó |

**Composite (rápido, 3 niveles): 0.694** vs liga 0.94.

### v9 según su propia eval del último iter (64 games/nivel internos al run)

| Nivel    | Score | Lectura |
|----------|-------|---------|
| easy     | 0.492 | **perdió la mitad** |
| normal   | 0.266 | **perdió 73%** |
| hard     | 0.875 | bien |
| apex     | 0.969 | dominante |
| gambit   | 0.328 | **mal** |
| sentinel | 0.922 | dominante |

`best_eval_score` reportado durante el entrenamiento: **0.6875**. Liga había
alcanzado 0.8099. v9 nunca cruzó la marca de su base.

### Head-to-head (16 partidas, sims=80)

| Match               | v9 W | rival W | D | Score v9 |
|---------------------|------|---------|---|----------|
| v9 vs **liga**      |  3   | 12      | 1 | **0.219** |
| v9 vs **centinela** |  9   |  6      | 1 | 0.594 |

v9 queda entre centinela y liga. Pierde feo contra liga, gana cómodo a
centinela. No es regresión total — es regresión parcial al promedio entre
los dos.

## Por qué pasó esto

Cuatro causas independientes, ordenadas por impacto estimado:

### 1. Bajar heurísticas es regalar el suelo anti-greedy

PM05 nos enseñó que las heurísticas son un piso pedagógico: enseñan al
modelo a no caer ante un oponente avaro. Liga vio 50% de heurísticas
(mezcla easy/normal/hard). v9 vio 25%, y encima reasignó esa cuota a random
(que no enseña táctica, solo distorsiona el buffer) y a checkpoints viejos
del propio modelo (que comparten los biases de la familia).

El propio eval de v9 lo grita: `easy=0.49, normal=0.27, gambit=0.33`. Lo
que liga sabía manejar, v9 lo desaprendió.

### 2. Bootstrap con `reset_iteration=true` perturba pesos buenos

`hf_reset_iteration=true` no solo resetea el contador de iteraciones —
también reinicia el scheduler de learning rate. El run arranca con LR
alto (3e-4) sobre los pesos de liga, que ya están en un mínimo razonable.
Las primeras iteraciones empujan esos pesos lejos del óptimo de liga
antes de que el scheduler los frene.

Combinado con la nueva distribución de oponentes (más random, menos
heurísticas), las primeras iteraciones generan datos que **alejan** al
modelo del estilo que le funcionaba.

### 3. Más self-play sobre una base fuerte refuerza biases existentes

Self-play es una herramienta poderosa cuando el modelo está aprendiendo
desde cero — explora distintos estilos contra sí mismo. Cuando la base
ya es fuerte, self-play tiende a **estilizar**: el modelo se especializa
en su propio meta. Si en ese meta no aparecen jugadas greedy, el modelo
nunca las practica.

Liga compensaba esto con 50% de heurísticas. v9, al bajarlas a 25% y
subir self-play a 60%, perdió el contrapeso.

### 4. El eval gate no abortó a tiempo

`eval_regression_delta=0.06` con `eval_regression_patience=2` mira la
regresión respecto al **best del run actual**, no contra liga como
baseline absoluto. Como v9 nunca subió de 0.69 (su best fue ~0.69 a partir
de iter 30), el gate nunca detectó "regresión" — todo el run estuvo
flotando bajo el nivel de liga sin que el sistema lo notara.

Resultado: ~150 iteraciones de Kaggle desperdiciadas en un modelo que ya
era peor desde iter 30.

## El patrón con nombre

Este caso es lo que en RL contra oponentes mixtos se llama **catastrophic
forgetting via curriculum drift**: el modelo no olvida porque cambiaste
el modelo, olvida porque cambiaste la dieta de oponentes. Misma red, mismo
optimizer, mismo dataset structure — distintos sparring partners → distinta
distribución de skill final.

Es prima hermana del fenómeno de PM05 (opponent exploitation): allá el
modelo se especializaba en un oponente específico; acá el modelo se
especializa en **sí mismo**, lo que es funcionalmente equivalente a
especializarse en una distribución estrecha.

## Lo que deberíamos haber hecho (acciones para v10)

Tres palancas, en orden de prioridad:

1. **Mantener o subir la dieta de heurísticas**, no bajarla.
   `opponent_heuristic_prob=0.55` con mezcla más uniforme de niveles
   (`easy_prob=0.25, normal_prob=0.25, hard_prob=0.5`). El suelo
   anti-greedy es lo más caro de aprender y lo más fácil de perder.

2. **No resetear el LR scheduler en un bootstrap.** `hf_reset_iteration=false`
   y `warmup_games=0`. Si la base es buena, el LR ya está en el rango
   adecuado — un warmup nuevo es una agresión a los pesos buenos.
   (El validador de bootstrap pide warmup>=320 si reset_iteration=true, así
   que la combinación correcta es reset=false sin warmup.)

3. **Eval gate con baseline absoluto.** Cambiar el gate para que compare
   contra `best_known_composite` (un número global, no del run), y aborte
   automáticamente si N iteraciones consecutivas no superan a la base.
   Eso le habría ahorrado a este run 140 iters de Kaggle.

Bonus para más adelante: subir `mcts_sims` de self-play de 160 a 240-320
mejora la calidad de los policy targets más que cualquier cambio de
oponente — pero cuesta tiempo. Primero (1) + (2) solos.

## Lo que aprendimos (puntos para el cuaderno)

1. **Bootstrap no es upgrade automático.** Empezar desde una base buena
   con la dieta equivocada es estadísticamente peor que entrenar desde cero
   con la dieta correcta — porque arrancás con expectativas altas y caés.

2. **El eval interno del run miente sin baseline absoluto.** v9 reportó
   `best_eval_score=0.69` durante el entrenamiento sin alarma; comparado
   contra el 0.81 de v8 estaba siempre por debajo. El sistema no lo vió
   porque solo se comparaba consigo mismo.

3. **Las palancas de currículum son acopladas.** No podés bajar
   heurísticas sin compensar; no podés subir self-play sin pagar diversidad
   en otro lado. PM05 mostró el extremo opuesto (demasiada heurística →
   exploits); este post-mortem muestra el otro extremo (poca heurística →
   amnesia anti-greedy). El óptimo está en el medio y no es trivial
   moverse en esa dimensión.

4. **El nombre captura la falla.** *Espejismo* parecía la siguiente
   generación (bootstrap desde el campeón, más self-play maduro) y resultó
   un retroceso encubierto. El nombre queda como recordatorio: lo que
   parece evolución a veces es solo continuidad estética.

## Lugar en el árbol de generaciones

| Codename     | Versión | Composite | H2H vs liga | Lectura |
|--------------|---------|-----------|-------------|---------|
| liga         | v8      | 0.94      | —           | campeón actual |
| centinela    | v6      | 0.81      | 0.13        | overfit a sentinel (PM05) |
| **espejismo**| **v9**  | **0.69**  | **0.22**    | bootstrap fallido por dieta |
| amnesia      | v7      | 0.75      | 0.25        | regresión catastrófica (PM04) |
| aprendiz     | v4      | 0.25      | 0.0         | curriculum prematuro |

Liga sigue siendo el rey. v9 entra como advertencia pedagógica, no como
contendor.

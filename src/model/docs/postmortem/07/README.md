# AtaxxZero — Post-mortem 07: v10 (paralelo) empató con liga por otra ruta

> **Resumen:** `policy_spatial_v10_iter_222` (apodado *paralelo*) fue el primer
> run desde cero con la tubería nueva: pretraining sobre replays humanos
> curados (torneo + sesiones de play + charla de 4to medio), absolute gate
> que aborta si no le gana al baseline en h2h, y replay tagging con
> oversample×4 para señal humana. Cero bootstrap, todo aprendido desde
> ruido inicial. Resultado: **empate técnico con liga** — RR 20-12 (0.625)
> contra liga 20-12 (0.625), h2h directo 15-17 (0.469) sobre 32 partidas,
> internal eval 0.66 contra 0.81 de liga. v10 NO bajó el techo (lo que ya
> sería un logro frente a v9), pero tampoco lo rompió. Mismo destino, otra
> ruta. La lección operativa: cambiar la tubería sin cambiar la capacidad
> del modelo no mueve el techo.

---

## Contexto: por qué se intentó este run

Después de PM06 (espejismo, v9) la conclusión fue clara: bootstrap desde
liga con dieta nueva era peligroso, y el eval gate relativo no detectaba
regresiones porque solo se comparaba contra sí mismo. v10 nació con tres
hipótesis acumuladas a probar de una vez:

1. **Pretraining humano** puede romper el techo porque inyecta posiciones
   que self-play nunca alcanza por su propia cuenta. El torneo de la clase
   02 y las sesiones de play habían dejado 52 partidas humanas (~3220
   ejemplos crudos) en `tournament_replays/`.
2. **Absolute gate** evita el patrón de PM06 (run flotando bajo la base
   sin alarma): si en N evals seguidas el candidato no le gana en h2h al
   baseline absoluto (liga), abortar.
3. **Scratch con dieta correcta** (heurísticas mantenidas, sin bootstrap
   problemático) debería al menos llegar al techo de liga.

El plan se materializó en estos cambios respecto a v8/v9:

| Hparam                              | v8 (liga) | v9 (espejismo) | v10 (paralelo) |
|-------------------------------------|-----------|----------------|----------------|
| `hf_bootstrap_run_id`               | `""`      | `policy_spatial_v8` | **`""`** (scratch) |
| `hf_reset_iteration`                | `false`   | `true`         | `false` |
| `pretrain_dataset_path`             | —         | —              | **`v10_pretrain.npz`** |
| `pretrain_epochs`                   | —         | —              | **3** |
| `warmup_games / warmup_epochs`      | 0/0       | 0/0            | 0/0 |
| `iterations`                        | 180       | 180            | 240 |
| `episodes_per_iter`                 | 24        | 20             | 8 |
| `mcts_sims` (self-play)             | 160       | 160            | 96 |
| `eval_every`                        | 6         | 6              | 6 |
| `eval_games`                        | 64        | 64             | 64 |
| `eval_absolute_action`              | —         | —              | **`abort`** |
| `eval_absolute_abort_mode`          | —         | —              | **`h2h`** |
| `eval_absolute_min_iteration`       | —         | —              | 36 |
| `eval_absolute_patience`            | —         | —              | 2 |
| `baseline_checkpoint`               | —         | —              | **`liga`** |
| `baseline_composite`                | —         | —              | 0.81 |
| `baseline_h2h_min_score`            | —         | —              | 0.45 |

Pipeline nueva (toda gitignored salvo el código):

```
tournament_replays/*.npz + .json (sidecar con quality_tag)
        ↓
scripts/curate_training_data.py --human-oversample 4
        ↓
data/curated/v10_pretrain.npz  (~12.8k ejemplos tras oversample)
        ↓
train.py --pretrain-dataset-path ... --pretrain-epochs 3
        ↓
self-play loop normal (96 sims, 8 ep/iter, 240 iter target)
        ↓
absolute_gate cada 6 iter: ¿h2h vs liga ≥ 0.45? si no, contar fallo
```

## Metodología de evaluación

Tres mediciones independientes:

```bash
# 1. Composite vs heurísticas (rápido, IC ancho)
uv run python scripts/eval_checkpoint_vs_heuristic.py \
    --checkpoint checkpoints/policy_spatial_v10_iter_222.pt \
    --levels hard,apex,sentinel --games 12 --sims 80

# 2. Head-to-head directo vs liga (24 partidas, más sims=80)
uv run python scripts/compare_checkpoints.py \
    --checkpoint-a checkpoints/policy_spatial_v10_iter_222.pt \
    --checkpoint-b liga --games 24 --mcts-sims 80

# 3. Round-robin tier alto (8 partidas por par sobre los 5 mejores)
uv run python scripts/round_robin.py --games 8 --sims 80 \
    --only v10,liga,espejismo,amnesia,centinela
```

También se compararon **iter_222 vs iter_227** (los dos checkpoints
locales del run) para validar que iter_222 era el pico. Resultado: 12-12
(P2 siempre gana — artefacto MCTS determinista a temp=0). Diferenciación
vino del eval vs heurísticas: iter_222 composite 0.694, iter_227 0.639.
Promoción → iter_222.

## Resultados

### v10 (iter_222) vs heurísticas (12 games, sims=80)

| Nivel    | W  | L  | D | Score | Lectura |
|----------|----|----|---|-------|---------|
| hard     |  9 |  3 | 0 | 0.750 | sólido |
| apex     |  9 |  3 | 0 | 0.750 | sólido |
| sentinel |  7 |  5 | 0 | 0.583 | bajó |

**Composite local: 0.694**. Por encima de centinela (0.664), debajo de
liga (0.775 con su gauntlet original).

### v10 según su propia eval del último iter (64 games/nivel, internos al run)

| Nivel    | Score | Lectura |
|----------|-------|---------|
| hard     | 0.6875 | bien |
| apex     | 0.5781 | flojo (liga apex 0.9688) |
| sentinel | 0.7109 | bien |

**`best_eval_score` reportado: 0.6588** (iter 222). Liga había alcanzado
**0.8099**. v10 nunca cruzó la marca de su base. Tampoco la cruzó v9
(0.6875). Tres runs, mismo techo de internal eval ~0.65-0.69.

### Head-to-head vs liga

| Fuente | v10 W | liga W | D | Score v10 |
|--------|-------|--------|---|-----------|
| compare_checkpoints (24g)   | 11 | 13 | 0 | 0.458 |
| round_robin tier alto (8g)  |  4 |  4 | 0 | 0.500 |
| **Total (32g)**             | **15** | **17** | **0** | **0.469** |

IC95% de 32 partidas Bernoulli: aproximadamente [0.30, 0.64]. **Empate
estadístico con liga**. Ni regresión ni salto.

### Round-robin tier alto (8 partidas/par, sims=80)

| Codename   | W-L-D     | Score |
|------------|-----------|-------|
| **liga**       | 20-12-0   | **0.625** |
| **v10**        | 20-12-0   | **0.625** |
| espejismo  | 19-13-0   | 0.594 |
| amnesia    | 15-17-0   | 0.469 |
| centinela  |  6-26-0   | 0.188 |

v10 aplastó a centinela 8-0 (liga hizo 7-1), confirmando que el techo
táctico básico se conserva. v10 vs liga 4-4: empate exacto en partidas
pares.

## Por qué pasó esto: cuatro causas, una conclusión

### 1. El pretraining humano se diluyó

3 epochs sobre ~12.8k ejemplos curados (3220 humanos × oversample×4)
antes del self-play. Self-play después: 240 iter × 8 episodios × ~40
movimientos ≈ **76.8k ejemplos por epoch del replay buffer**. Ratio
~6:1 a favor de self-play **por iteración**, y se acumula durante 240
iter.

El pretraining inicializa pesos. No regulariza. Una vez que el self-play
empieza, los gradientes humanos no aparecen en ninguna loss — desaparecen
en las primeras 30-50 iteraciones. Lo que queda al final del run es
indistinguible de un scratch sin pretrain.

Diego lo nombró cuando me reté: "metimos datos humanos, los del torneo
y la charla de 4to medio, más encima metimos pretraining". Sí, los
metimos. Pero los metimos al lugar equivocado del pipeline.

### 2. Capacidad táctica insuficiente para batir humanos *novatos*

El modelo (d_model=128, 6 layers, 8 heads) tiene techo de representación.
Bate heurísticas porque las heurísticas son determinísticas y
explotables. Empata con liga porque liga comparte arquitectura. Un
humano que juega off-distribution (como Diego en su primera partida)
explota el value head fuera del manifold de self-play.

La diagonal de la matriz de RR lo confirma: todos los modelos modernos
(liga, v10, espejismo) terminan en la franja 0.594-0.625 entre sí.
Mismo techo arquitectónico, distintos métodos.

### 3. menos episodios por iter + menos sims que liga

v10 corrió con `episodes_per_iter=8` y `mcts_sims=96`; liga corrió con
24 y 160. Esto da **3× menos data** y **señal de policy más ruidosa** por
movimiento. Las 240 iter de v10 generaron similar throughput de ejemplos
que las 180 iter de liga, pero con peor SNR por ejemplo.

Era una decisión consciente para meter pretraining en el presupuesto de
GPU sin hacer crashear Kaggle. Costó techo táctico final.

### 4. El absolute gate nunca disparó porque h2h_score ≥ 0.45 era trivial

`baseline_h2h_min_score=0.45` significa "no permitas que el candidato
pierda peor que 0.45 contra liga". En h2h pares pequeños (eval_games=64
en kaggle, sims=96 ahí) el modelo flotaba alrededor de 0.45-0.55 — bajo
para superar pero suficiente para no abortar. El gate hizo lo que tenía
que hacer (no abortar, porque v10 no estaba regresando), pero no movió
el techo. La herramienta funcionó; el problema no era el gate.

Detalle adicional: el run de Kaggle terminó por `BrokenProcessPool` en
iter 227 (no por el gate), pero el peak ya estaba en iter 222. No
perdimos un mejor checkpoint por la caída.

## El patrón con nombre

Esto es **techo arquitectónico**, no falla de pipeline. La pipeline
nueva (pretrain humano + absolute gate + curation + tagging) **funciona**
en el sentido de que produce un modelo competitivo. No produce un modelo
mejor porque la capacidad del transformer ya está exprimida con la
señal que tenemos.

Es la lección dual de PM06: allá cambiar la dieta hizo retroceder; acá
cambiar la dieta sin cambiar la red mantiene el plateau.

## Lo que deberíamos haber hecho (acciones para v11)

Tres palancas independientes. Cualquiera por sí sola puede mover el dial,
las tres juntas casi seguro:

1. **Replay humano en cada iter del training loop, no solo en pretrain.**
   Mezclar el replay buffer humano con el self-play buffer en cada
   training step (proporción 10-20% humano fijo). El pretraining como
   inicializador se evapora; la mezcla como regularizador no. Esto
   requiere modificar `trainer_runtime` para aceptar un segundo buffer
   con peso de muestreo. Costo: 1 día de código + tests.

2. **Solo policy distillation de humanos, no value.** El value target de
   un replay humano es el resultado final de la partida, lo que asume que
   *cada* movimiento del humano fue consistente con ese resultado. Un
   humano malo gana por suerte → propaga `value=+1` a movimientos
   pésimos. Quedarse con la jugada (policy) descarta esa contaminación.
   Cambio puntual en `curation.py`: nullear `values` cuando `is_human=True`
   y enmascarar value loss para esos ejemplos en trainer.

3. **Subir capacidad: d_model=192 o 256, 8 layers.** Esto sí mueve el
   techo arquitectónico. Costo: GPU más cara, ~2× tiempo de iter en T4.
   Vale la pena si las palancas 1+2 quedan exhibidas como insuficientes.

Bonus: **diversidad humana**. 4 humanos distintos (Felipe, Vicente,
Julio, Diego) no representan la varianza de un humano nuevo. Otra clase
o torneo abierto mete personas con estilos no vistos. Más alto bang/buck
que más partidas de los mismos.

## Lo que aprendimos (puntos para el cuaderno)

1. **El pretraining se evapora si no hay regularización persistente.**
   Inicializar pesos con datos humanos y después correr 240 iter de
   self-play sin tocar los datos humanos vuelve al modelo equivalente a
   un scratch run con peor SNR. Si querés que los humanos pesen, tienen
   que estar en *cada* batch.

2. **El absolute gate hizo bien su trabajo, pero no es palanca de
   techo.** Detecta regresión absoluta. No fuerza progreso absoluto.
   Para forzar progreso hay que cambiar capacidad o datos del loop, no
   agregar más gates.

3. **El techo del transformer 128/6/8 está agotado para este juego.**
   Tres modelos distintos (liga, v9, v10) entrenados con dietas distintas
   convergen al mismo techo de RR (0.594-0.625) y de internal eval
   (0.65-0.81). El cuello de botella no es el optimizer ni la dieta:
   es el espacio de representación.

4. **El nombre captura la falla.** *Paralelo* tomó una ruta nueva (pretrain
   humano, absolute gate, scratch limpio, replay tagging) y arribó al
   mismo techo que liga. Curva distinta, asíntota igual.

## Lugar en el árbol de generaciones

| Codename     | Versión | Composite | H2H vs liga | RR tier alto | Lectura |
|--------------|---------|-----------|-------------|--------------|---------|
| liga         | v8      | 0.94 (gauntlet completo) | — | 0.625 | campeón actual |
| **paralelo** | **v10** | **0.694** | **0.469** (32g) | **0.625** | **empate técnico con liga** |
| espejismo    | v9      | 0.694     | 0.219       | 0.594        | bootstrap fallido (PM06) |
| centinela    | v6      | 0.664     | 0.13        | 0.188        | overfit a sentinel (PM05) |
| amnesia      | v7      | 0.75      | 0.25        | 0.469        | regresión catastrófica (PM04) |

Liga sigue siendo el rey por h2h directo (0.531 favor liga sobre 32g).
v10 se sienta justo al lado, no atrás. Es la primera generación que no
retrocede respecto a su intento de superar a liga — pero tampoco la
supera. Sirve como **piso conservado** para que v11 pueda intentar la
palanca de capacidad sin riesgo de volver a empezar desde el suelo.

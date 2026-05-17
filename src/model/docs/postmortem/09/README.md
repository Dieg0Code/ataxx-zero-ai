# Postmortec 09 — `asedio` (v11.1, iter 41)

## TL;DR

Primera generación post-fix del count head normalizado. La pipeline v11
funcionó: composite saltó **5×** respecto a lastre (0.044 → 0.236),
todas las losses se comportaron como se esperaba, y por primera vez en
cuatro generaciones consecutivas (liga, espejismo, paralelo, lastre)
el modelo se sintió **amenazante a un humano en partida real**. Diego
ganó 3 de 4 contra `asedio` pero "sentí la presión" — la primera vez
desde liga que un modelo proyecta amenaza.

Pero asedio NO rompió el techo del absolute gate: h2h vs liga quedó
en **0.422** (umbral 0.45), 2.8pp corto. El gate abortó la corrida
en iter 42 después de patience=2 fallidos. Apodo: **asedio** — puso
cerco al portón de liga sin lograr abrirlo.

## Contexto

PM08 cerró `lastre` (v11) documentando que el count head auxiliar con
target sin normalizar canibalizaba el backbone compartido. El fix fue
una línea en `src/model/system.py`:

```python
target_counts_norm = target_counts.view(-1) / float(BOARD_SIZE * BOARD_SIZE)
```

asedio es el primer run con esa normalización aplicada. Mismo config
arquitectónico que lastre (d_model=192, 8 capas, value head profundo,
count head auxiliar, symmetry aug D4, pretrain humano, replay humano
permanente al 20%), pero ahora con todos los heads en escala
comparable.

## Síntoma observado vs lastre

### Training (de la trayectoria wandb, run id `n0my2lli`)

| Métrica | lastre | asedio | Δ |
|---|---|---|---|
| train/loss_total | 16.78 | **2.77** | ↓ 6× |
| train/loss_count (raw) | 132.7 | **0.053** | ↓ 2500× |
| train/loss_policy | 3.03 | **2.40** | ↓ 21% |
| train/loss_value | 0.96 | **0.74** | ↓ 23% |
| train/policy_accuracy | 14.3% | **23.6%** | ↑ +9pp |
| train/value_mae | 0.89 | **0.74** | ↓ 17% |
| val/policy_accuracy | 19.1% | **22.3%** | ↑ +3pp |

El count head pasó de "loss raw plana 100-180 todo el run" a "decrece
de 3.1 a 0.05 en 26 steps". El value MAE de "tasa cero al iter 18" a
"baja monotónicamente". El policy accuracy de "estancada en 10-14%" a
"trayectoria clara hacia 25%+". La pipeline base estaba sana — el
count loss roto la frenaba.

### Eval @ iter 36 (vs 6 niveles × 64 partidas = 384 totales)

| Métrica | lastre @ iter 18 | asedio @ iter 36 |
|---|---|---|
| composite | 0.044 | **0.236** |
| wins | 17 | **89** |
| losses | 367 | 292 |
| draws | 0 | 3 |
| eval_score_easy | 0.000 | **0.391** |
| eval_score_normal | 0.000 | **0.219** |
| eval_score_hard | 0.016 | **0.180** |
| eval_score_apex | 0.016 | **0.227** |
| eval_score_gambit | 0.000 | **0.172** |
| eval_score_sentinel | 0.047 | **0.227** |

Composite 0.236 ya pone a asedio en el rango de `centinela` (0.43)
pero medido más severo (con 6 niveles en vez de 3) — i.e., asedio es
similar o algo mejor que centinela bajo el eval honesto nuevo.

### h2h vs liga @ iter 36

- **27 wins / 37 losses / 0 draws** sobre 64 partidas = **0.422**.
- Umbral del gate: 0.45.
- `abs_fail = 1, h2h_fail = 1` (primera patience).
- iter 42 (segunda patience) repitió el fallo → abort.

## La anécdota cualitativa

Diego cargó `asedio` (iter 41) en arena y jugó 4 partidas mano-a-mano
sin saber el composite numérico. Resultado:

> "Perdí una pero porque hice missclick. En el resto, gané. Pero sentí
> la presión."

Es la primera vez en cuatro generaciones que el reporte cualitativo
sube de "le gané fácil al modelo, mid-game raro" (lastre) a "le gané
pero estuve cerca de perder, hay amenaza real". Crucialmente, el ELO
humano de Diego está calibrado contra liga/paralelo (los más fuertes
hasta ahora) y asedio se sintió comparable a ellos en lo subjetivo,
incluso con composite vs heurística inferior al de paralelo.

Eso confirma dos cosas:
1. La pipeline humana del v11 (pretrain + human_batch_fraction=0.20 con
   value_mask) **sí inocula estilo humano**: asedio juega contra Diego
   distinto a como jugarían liga/paralelo (que entrenaron 100% vs
   heurísticas y self-play).
2. El composite vs heurística **no captura la dimensión humana**. asedio
   compite contra Diego mejor que contra apex/gambit, lo cual es
   exactamente la asimetría que esperábamos cuando metimos data
   humana en el pipeline.

## Por qué falló el gate aunque la pipeline funcionó

El absolute gate exigía `h2h vs liga ≥ 0.45` con patience=2 a partir
de iter ≥ 36. asedio quedó a 2.8pp del umbral en dos chequeos
consecutivos. Tres lecturas posibles:

1. **El umbral estaba mal calibrado para esta generación**. liga tiene
   composite 0.65 (vs 3 heurísticas duras); asedio tiene 0.24 (vs 6,
   métrica más severa). Que asedio quede 0.42 h2h vs liga sin haber
   "alcanzado" su composite es plausible — necesita más iters de
   self-play maduro para destilar el conocimiento del backbone hacia
   policy/value pulidos.
2. **min_iteration=36 era demasiado temprano**. lastre llegó a iter 18
   con composite 0.04, asedio a iter 36 con 0.24 — 5× en 2× iters.
   Extrapolando, en iter 60-80 podría haber estado en 0.40+ composite
   y h2h 0.50+. El gate no le dio runway.
3. **El techo de liga es real**. Cuarta generación que orbita el
   plateau h2h 0.42-0.55. Quizás la arquitectura/datos topa acá y
   ningún tuning de pipeline mueve la asíntota.

Para v11.2 (próxima corrida) optamos por hipótesis #2: bajamos
`baseline_h2h_min_score` a 0.40 y subimos `eval_absolute_min_iteration`
a 80. Si v11.2 escala h2h por encima de 0.50 en iter 100-200, queda
clarísimo que asedio fue víctima de patience prematura. Si plateauea
en 0.42, vamos a la hipótesis #3 con cambios estructurales (más
mcts_sims, otra arquitectura).

## Anécdotas adicionales

- **El bug del notebook que casi se come la evidencia**: cell 1 tenía
  `RUN_NAME = "policy_spatial_v11"` hardcodeado, y cell 6 hacía
  `run_config["hf_run_id"] = RUN_NAME` sobreescribiendo el json. Cuando
  bumpeé `hf_run_id` en `kaggle/v11_config.json` de `v11` a `v11_1`, el
  notebook lo pisaba al ejecutarse y todos los uploads de asedio
  cayeron en `runs/policy_spatial_v11/` mezclados con lastre. La
  carpeta HF de policy_spatial_v11 ahora tiene 41 archivos: iters
  1-18 son una mezcla de lastre seguida de asedio sobrescribiendo, e
  iters 19-41 son asedio puro. Fix en `a42247d`: cell 6 lee
  hf_run_id del json. Lección: cualquier campo "que se puede
  overridear en N lugares" garantiza que se va a desincronizar; un
  campo, un dueño.
- **El gate previo era v10**: el config v11 heredó `min_iteration=36`
  de v10/paralelo, donde tenía sentido porque paralelo arrancaba con
  bootstrap. asedio arrancó scratch + pretrain humano, lo cual
  requiere más calentamiento. Default a 36 era razonable pero
  conservador en exceso para arch nueva.
- **El seed del league** (`paralelo > liga > espejismo > amnesia >
  centinela`) sí funcionó: durante el run, el opponent mix vio esos
  checkpoints desde iter 1 sin tener que autodescubrirlos. No tenemos
  data fina del impacto pero la trayectoria de loss monotónicamente
  decreciente sugiere que el sparring diverso ayudó.

## Lecciones aprendidas

1. **Un fix bien dirigido vale más que cuatro cambios juntos**. PM08
   identificó count loss canibalizando como el cuello de botella.
   Solo eso, normalizar /49, desbloqueó 5× de mejora en composite.
   El resto de la pipeline v11 (arch nueva, pretrain humano, sym aug,
   replay humano) ya estaba sana — solo necesitaba que el head
   auxiliar no robara gradiente.
2. **Gates absolutas necesitan calibración por generación**. Heredar
   `baseline_h2h_min_score=0.45` de v10 (que bootstrappeó) a v11
   (que arranca scratch) fue ingenuo. Cada generación nueva con arch
   diferente debería tener un primer run con gate desactivado o
   loosen para descubrir su asíntota real, y después un segundo run
   con gate calibrado a +0.05 sobre la asíntota observada.
3. **El composite y el feedback humano miden cosas distintas**. asedio
   tiene composite 0.24 (peor que paralelo 0.69) pero Diego lo
   reportó "más amenazante" que paralelo. Eso confirma que entrenar
   con humanos da ganancia en una dimensión que el composite vs
   heurística NO captura. Para próximas generaciones convendría
   formalizar un eval humano (e.g., gauntlet de 10 partidas Diego vs
   modelo, registrar wins/losses/feedback) como métrica auxiliar al
   composite.
4. **Single source of truth para configs propagados**. El bug del
   notebook sobreescribiendo el json fue innecesario y costó la
   carpeta HF limpia para asedio. Cualquier campo que aparece en >1
   lugar es una bomba de tiempo. Fix permanente en `a42247d`.
5. **El plateau del h2h vs liga es real-or-not real, todavía no
   sabemos**. Cuarta generación que orbita 0.42-0.55. v11.2 con gate
   loosen va a decidir si fue patience prematura (asedio rompe el
   techo dado más tiempo) o si es la arquitectura misma.

## Apodo

**asedio** — del español: cerco prolongado y agresivo a una plaza
fuerte sin necesidad de tomarla por asalto. asedio puso a liga bajo
presión (h2h 0.42) y a Diego también (4 partidas, 1 derrota humana
por missclick, las otras 3 victorias humanas con sensación de
peligro). No abrió el portón pero hizo sentir su presencia. Y dejó
asentadas las bases para que v11.2 lo intente con runway adicional.

## Checkpoint persistido

`checkpoints/registered/asedio.pt` — copia del iter 41 (último
checkpoint que llegó a HF antes del abort). 15MB, arch v11
(d_model=192, value_head_depth=2, count_head_enabled=True).
Cargable via `resolve('asedio')` o `--ckpt asedio`.

## Próximo paso

v11.2 está corriendo con `baseline_h2h_min_score=0.40` y
`eval_absolute_min_iteration=80`. Si esta generación rompe 0.50 h2h
en iter 100+, asedio queda como "víctima del gate". Si plateauea en
0.42, asedio queda como "primera vista honesta de la asíntota de la
arquitectura v11".

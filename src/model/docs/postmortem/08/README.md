# Postmortem 08 — `lastre` (v11, intento abortado en iter 18)

## TL;DR

v11 prometía romper el plateau de `liga`/`paralelo` con arquitectura
nueva (d_model 128→192, 8 capas, value head profundo, count head
auxiliar), pretrain humano, replay humano permanente al 20% por batch,
symmetry augmentation D4 y eval contra los 6 niveles de heurística.
Después de 18 iters quedó claro que **el count head auxiliar no
estaba aprendiendo y, peor, estaba canibalizando el backbone
compartido**. Composite plateau en 0.03-0.04 (literalmente 10-17 wins
sobre 384 partidas).

La culpa no era de la arquitectura ni de los datos: era una decisión
de diseño en el target del head auxiliar. Apodo: **lastre** — el peso
muerto que arrastró el resto.

## Contexto

PM07 cerró `paralelo` (v10) como empate técnico con `liga` (v8) por
tercera vez consecutiva. La hipótesis era que el techo no se movía
cambiando solo la dieta de datos, así que v11 atacó cuatro palancas
en simultáneo:

1. **Arquitectura más grande**: d_model 128→192, 6→8 capas, ff 512→768,
   value head con dos capas ocultas en vez de una.
2. **Count head auxiliar** (nuevo): predice la diferencia de piezas
   finales como tarea regularizadora.
3. **Symmetry augmentation D4**: cada batch transformado por uno de
   8 elementos del grupo dihedral.
4. **Pipeline humano permanente**: pretrain + replay buffer al 20% de
   cada batch con value mask (humanos no contribuyen value loss,
   solo distill policy).

## Síntoma observado

Tres evals durante el run (iter 6, 12, 18, eval cada 6 iters vs 6
niveles × 64 partidas = 384 totales):

| Iter | Composite | Wins | Losses | Draws |
|------|-----------|------|--------|-------|
| 6    | 0.030     | 10   | 371    | 3     |
| 12   | 0.029     | 11   | 373    | 0     |
| 18   | 0.044     | 17   | 367    | 0     |

Composite ~0.03 es bottom-of-barrel; `paralelo` en sus iters tempranos
ya orbitaba 0.20-0.30. Las métricas de training contaban una historia
distinta a la del eval:

| Métrica train | Iter 1 | Iter 18 | Trend |
|---|---|---|---|
| `loss_total` | 12.7 | 16.8 | ↑ (¡subió!) |
| `loss_policy` | 3.01 | 3.03 | → estancado |
| `loss_value` | 1.53 | 0.96 | ↓ aprendiendo |
| `loss_count` (raw) | ~130 | ~130 | → plateau |
| `value_mae` | 1.00 | 0.89 | ↓ aprendiendo |
| `policy_acc` | 10.3% | 14.3% | ↑ muy lento |

Value y policy SÍ progresaban — apenitas. Count head quedó plano todo
el run. Y el loss total no bajaba porque count dominaba la suma.

Anécdota cualitativa: cargado el checkpoint iter_018 en arena, en
partida vs humano (Diego) el modelo perdió pero "en algún momento
sintió que iba a perder" — evidencia de que el policy head había
aprendido suficiente mid-game para generar amenazas plausibles, pero
no remataba endgames y se desplomaba contra heurísticas pulidas.

## Diagnóstico (root cause)

El target del count head era **diferencia cruda de piezas finales**,
construida así en `src/training/reward_runtime.py:114`:

```python
count_diff = final_diff_p1 if player_at_turn == 1 else -final_diff_p1
```

Rango efectivo del target: ~[-30, +30] con desviación estándar ~10-15.
El value target, en cambio, vive en [-1, +1].

Aplicando la loss combinada:

```
loss = loss_pi + 0.5 · loss_value + 0.1 · loss_count
     ≈ 3.0    + 0.5 · 1.0        + 0.1 · 130
     ≈ 3.0    + 0.5              + 13.0
```

Comparación de contribuciones al loss:

| Head  | Target range | MSE inicial | × coeff | Contribución |
|-------|--------------|-------------|---------|--------------|
| value | [-1, +1]     | ~1.0        | 0.5     | 0.5          |
| count | [-30, +30]   | ~130        | 0.1     | **13.0**     |

**El count head contribuía 26× más que el value head al loss total**.
Como todos los heads comparten el mismo backbone (CLS token del
transformer), el gradiente que llegaba al backbone estaba dominado
por el count head. El modelo dedicó capacidad a empujar features
útiles para predecir diferencia de piezas (tarea difícil porque desde
init `Linear(d/2, 1)` produce outputs std ~0.1 y los targets piden
magnitudes 10-15), mientras policy/value quedaron pidiendo migajas
de gradiente.

Resultado: ni count aprendió (output prácticamente constante porque
saltar de 0 a 15 es muy lejos para un MLP fresh), ni policy/value
progresaron al ritmo esperado.

## Por qué no se cazó antes

Tres redes de seguridad que no pescaron el problema:

1. **Tests unitarios**: `test_common_step_count_loss_applies_when_coeff_positive`
   verificaba que `loss_count > 0` y matcheaba el cómputo manual de
   MSE, pero con `target_counts = [5.0, -3.0]` (escala razonable). El
   test pasaba — y debía pasar — porque la implementación era correcta
   *según el spec*. El bug vivía en el spec, no en el código.

2. **Smoke local**: corrí con `selfplay_workers=1`, `iterations=2`,
   `episodes=2`, `mcts_sims=32`. La loss total en smoke (~3.5 después
   de pretrain y self-play) no era llamativa porque solo había 2-3
   batches por iter — no había tiempo para ver el plateau. Los gates
   pasaron porque "los gradientes son finitos y la loss no diverge".

3. **CI**: solo corre los tests, no entrena ni evalúa. Imposible que
   pescara esto.

La lección general: un test que verifica "loss > 0 y matchea fórmula"
no captura "loss está en escala correcta relativa a otras losses". Los
gates del repo no incluían un sanity-check de magnitud relativa
entre losses con coeffs aplicados. Es la primera vez que combinamos
un head auxiliar con target no normalizado, no había precedente.

## Fix aplicado (v11.1, ya pusheado en `c434b69`)

Normalizar el target en el sitio de la loss:

```python
# src/model/system.py
target_counts_norm = target_counts.view(-1) / float(BOARD_SIZE * BOARD_SIZE)  # /49
loss_count = functional.mse_loss(count_pred.view(-1), target_counts_norm)
```

Impacto numérico esperado:

| | Antes | Después |
|---|---|---|
| target range          | [-30, +30] | [-0.6, +0.6] |
| `loss_count` inicial  | ~130       | ~0.05        |
| × coeff 0.1 → al total| **13**     | **0.005**    |
| loss total iter 1     | ~16        | ~3.5         |

Con esto el count head sigue siendo señal auxiliar útil sin
canibalizar el backbone. El value head, que ya estaba aprendiendo,
debería acelerar al recibir gradiente sin competencia.

## Anécdotas adicionales del intento

- **3 bugs satélites pre-fix del count**, todos del mismo plano de
  fractura (arch nueva no propagada a callers que asumían legacy):
  1. `selfplay_runtime` no pasaba `value_head_depth`/`count_head_enabled`
     al `init_selfplay_process_worker` → workers spawn levantaban modelo
     legacy y crasheaban al load_state_dict (fix en `8316a28`).
  2. `_extract_model_cfg` en `league_runtime` solo capturaba 5 hparams
     del checkpoint → opponent_model construido con arch legacy y
     state_dict v11 → mismo crash (fix en `c502373`).
  3. `extract_model_kwargs` en `checkpoint_compat` tampoco capturaba
     los dos campos nuevos → arena/play_pygame no podía cargar el
     checkpoint v11 para jugar (fix en `f841ccd`).

  Los tres bugs salieron en cadena después del primer push: cada vez
  que arreglábamos uno, el siguiente se hacía visible. Lección: cuando
  agregás un campo nuevo a la dataclass de hparams, hacer grep
  global por el dataclass anterior y revisar cada caller.

- **El run sobrevivió a una pausa de 3 horas** entre iter 6 y iter 7
  (resume de Kaggle session) sin perder consistencia. La pipeline de
  HF persist + reload funcionó como se esperaba.

- **6 niveles de heurística en eval**: el cambio respecto de PM05/PM06
  permitió ver desde el primer eval que el modelo perdía contra easy
  (10 wins de 64), no solo contra hard. Si hubiéramos evaluado solo
  contra hard/apex/sentinel como antes, habríamos atribuido el 0.030
  composite a "modelo nuevo, dale tiempo" en lugar de "el modelo no
  juega". El nuevo eval es honesto.

## Lecciones aprendidas

1. **Auxiliary heads con target no normalizado son trampa**: si un
   head produce loss en escala distinta a las otras del backbone
   compartido, su gradiente domina y entrena features ortogonales a
   la tarea principal. Siempre escalar a magnitud comparable al
   value loss (~[-1, +1]).
2. **Coeff de loss combinado no compensa escala de target**: bajar
   `count_loss_coeff` de 0.1 a 0.01 habría reducido la contribución
   al total, pero el gradiente seguiría siendo dominante en magnitud
   relativa porque MSE crece cuadráticamente con el error y el target
   crudo amplifica el error inicial.
3. **El plateau de loss no diverge ≠ el modelo está aprendiendo**.
   Hay que mirar componentes individuales y métricas que no dependan
   del loss (policy_accuracy, value_mae, eval composite).
4. **No subir 4 cambios arquitectónicos juntos sin smoke largo**.
   El plan v11 declaró "agresivo, todo junto" y eso fue una elección
   consciente, pero al final el smoke local de 2 iters no podía
   detectar problemas que requieren ~10 iters de training real para
   manifestarse. Próxima vez con cambios de capa de loss: smoke de
   al menos 6 iters con monitoreo de loss_count individual.
5. **Una victoria humana cualitativa vale como señal**: que Diego
   sintiera que "iba a perder" en mid-game contra iter_018 confirma
   que policy+value estaban en una trayectoria correcta — el count
   head los frenaba, no los rompía. Eso valida que v11.1 (mismo
   diseño + fix de normalización) debería romper el plateau.

## Apodo

**lastre** — del español: peso muerto que se carga para hundir un
buque o que un globo suelta para subir. v11 cargó un head auxiliar
mal calibrado que actuó como lastre sobre el backbone. El resto del
diseño (arch grande, pretrain humano, replay humano permanente,
symmetry aug, eval 6-niveles) sigue siendo sano y se reutiliza tal
cual en v11.1. Lo que se suelta es la normalización rota.

## Próximo paso

v11.1: mismo config, mismo dataset, mismo plan — pero con la
normalización aplicada. Si el composite en iter 18 sube de 0.044 a
algo en el rango 0.30-0.50, confirmamos que el diseño v11 era
correcto. Si sigue bajo, hay segunda hipótesis pendiente:
desactivar el count head completo y correr v11.2 sin él.

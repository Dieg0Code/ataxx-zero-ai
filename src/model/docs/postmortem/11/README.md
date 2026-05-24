# Postmortem 11 — Curriculum estático: por qué v12 no terminó de despegar y la propuesta de curriculum adaptativo para v13

## TL;DR

PM05 y PM10 dejaron en claro que el opponent mix es la palanca que
explica casi todo el techo de generación tras generación. v11.2
(`umbral`) tenía easy=0%, normal=2% en fases tardías — el modelo
nunca aprendía a defenderse de esos niveles. v12 (`fulcro`,
candidato) **rebalanceó el curriculum estáticamente** (easy=12%,
normal=15%) y movió `mcts_sims` de 160 a 320. El resultado parcial
(iter 90, run incompleto):

- `normal` subió +17pp (0.266 → 0.438) ✅ — el fix pegó
- `sentinel` subió +15pp (0.586 → 0.734) ✅
- `h2h vs liga` subió +5pp (0.625 → 0.672) ✅
- `gambit` retrocedió -17pp (0.375 → 0.203) ❌
- `hard` cedió -13pp (0.727 → 0.594) ❌

El curriculum estático le pegó a normal/sentinel pero **rompió hard
y dejó gambit volátil** (0.266 → 0.250 → 0.094 entre evals 96/102/108).
Subir sims no compensó. El patrón sugiere algo más profundo: **un
curriculum fijo no puede adaptarse a lo que el modelo descubre que
es difícil mientras entrena**. Si gambit colapsa en el eval iter 108,
los 5 iters siguientes de self-play vuelven a darle el mismo 20%
ciego — no hay corrección.

Propuesta para v13: **curriculum adaptativo**. El opponent mix del
próximo bloque de self-play se ajusta automáticamente según el último
eval. Cuanto peor el score contra un nivel, más peso recibe ese nivel
en el próximo bloque. Conceptualmente es Prioritized Experience
Replay (Schaul et al., 2015) aplicado a la dimensión de adversarios.

## Diagnóstico: qué tiene de malo el curriculum estático

### Hipótesis original (v12, plan de mayo)

> El curriculum balanceado va a cerrar los gaps de easy y normal sin
> romper hard. Subir mcts_sims va a darle profundidad táctica
> suficiente para mejorar gambit.

Lo que pasó:

| Nivel | umbral | fulcro iter 90 | Δ | hipótesis cumplida |
|---|---|---|---|---|
| easy | 0.469 | 0.469 | = | parcial (no rompe) |
| normal | 0.266 | 0.438 | +17 | **sí** |
| hard | 0.727 | 0.594 | -13 | **no** (rompe lo que andaba) |
| apex | 0.492 | 0.469 | ≈ | parcial |
| gambit | 0.375 | 0.203 | -17 | **no** (peor) |
| sentinel | 0.586 | 0.734 | +15 | **sí** |

Dos hipótesis derrumbadas:
1. **"Solo redistribuir el bucket heurístico no rompe lo que ya
   andaba"** — falsa. Bajar hard de 24% a 18% le costó 13pp.
2. **"Más sims arregla gambit"** — falsa. Pasar de 160 a 320 sims
   no recuperó gambit, lo empeoró. Hipótesis alternativa: gambit
   requiere ver más partidas contra gambit, no más exploración
   por partida.

### El problema raíz del curriculum estático

En training de AlphaZero, el modelo descubre debilidades **mientras
entrena**, no de antemano. Algunas partidas vs gambit pueden
revelar un patrón que el modelo no logra contrarrestar — pero el
self-play del siguiente bloque le da el mismo 20% de gambit. Pierde
otra vez. El loop:

```
self-play → eval (gambit=0.09) → self-play (gambit=20% otra vez) → eval (gambit=0.12) → ...
```

No hay corrección. Es como entrenar a alguien para ajedrez dándole
*la misma proporción* de problemas tácticos en todas las sesiones
sin importar si ya domina tácticas o si pifia mate en 2 constantemente.

Esto fue **previsible**: PM05 ya documentó la misma dinámica con
`centinela`. La conclusión de PM05 fue "rebalancear el curriculum",
pero ese rebalanceo es siempre **a priori** — antes de empezar el
run, no durante.

### Por qué oscila la eval

Composite de v12 entre iters 90-108: 0.484 → 0.496 → 0.392 → 0.411.
Variación >10pp entre evals consecutivos. Con IC95% ±0.12 (n=64),
parte es ruido — pero la dirección sostenida (h2h liga 0.672 → 0.422
→ 0.406 → 0.391 a lo largo de 4 evals) **no es ruido**. Es regresión
real impulsada por que el modelo está aprendiendo cosas que mejoran
unas dimensiones a costa de otras, sin un mecanismo que reconcilie.

## Propuesta: curriculum adaptativo

### Algoritmo

```python
def compute_adaptive_mix(
    base_mix: dict[str, float],
    last_eval: dict[str, float] | None,
    ema_state: dict[str, float],
    target_score: float = 0.55,
    alpha: float = 1.5,
    floor: float = 0.05,
    cap: float = 0.35,
    smoothing: float = 0.5,
) -> tuple[CurriculumMix, dict[str, float]]:
    """
    Ajusta el bucket heurístico de la mix según el último eval per-level.

    base_mix: el mix estático de la fase actual (se preserva self/heuristic/random).
    last_eval: scores per-level del eval más reciente (None en el primer bloque).
    ema_state: EMA acumulada de deficits para suavizar ruido.
    target_score: nivel "aceptable" por debajo del cual subimos peso.
    alpha: cuán reactivo es el boost (1.0 conservador, 2.0 agresivo).
    floor / cap: rango admisible por nivel (5% a 35% del bucket heurístico).
    smoothing: peso del eval nuevo en la EMA (0.5 = mitad nuevo, mitad histórico).
    """
    if last_eval is None:
        return base_mix, ema_state

    new_ema = {}
    for level in HEURISTIC_LEVELS:
        new_deficit = max(0.0, target_score - last_eval[level])
        prev = ema_state.get(level, 0.0)
        new_ema[level] = smoothing * new_deficit + (1.0 - smoothing) * prev

    boosted = {}
    for level in HEURISTIC_LEVELS:
        base_w = base_mix[f"heu_{level}"]
        boost = math.exp(alpha * new_ema[level])
        boosted[level] = base_w * boost

    total = sum(boosted.values())
    heu_bucket_size = sum(base_mix[f"heu_{lvl}"] for lvl in HEURISTIC_LEVELS)
    boosted = {k: v * heu_bucket_size / total for k, v in boosted.items()}

    for level in HEURISTIC_LEVELS:
        boosted[level] = min(cap * heu_bucket_size, max(floor * heu_bucket_size, boosted[level]))

    total = sum(boosted.values())
    boosted = {k: v * heu_bucket_size / total for k, v in boosted.items()}

    return {
        "self": base_mix["self"],
        "heuristic": base_mix["heuristic"],
        "random": base_mix["random"],
        **{f"heu_{lvl}": boosted[lvl] for lvl in HEURISTIC_LEVELS},
    }, new_ema
```

### Ejemplo aplicado a iter 108 de fulcro

Input:
```
base_mix (fase >90): easy=0.12 normal=0.15 hard=0.18 apex=0.18 gambit=0.20 sentinel=0.17
last_eval iter 108:  easy=0.531 normal=0.344 hard=0.453 apex=0.547 gambit=0.094 sentinel=0.500
ema_state previa:    {todos en 0.0 si es primer bloque adaptativo}
target=0.55, alpha=1.5, floor=0.05, cap=0.35, smoothing=1.0 (primera vuelta)
```

Deficits:
- easy = max(0, 0.55-0.531) = 0.019
- normal = 0.206
- hard = 0.097
- apex = 0.003
- gambit = **0.456**
- sentinel = 0.050

Boosts (`exp(1.5 * deficit)`):
- easy ×1.03 → 0.124
- normal ×1.36 → 0.204
- hard ×1.16 → 0.209
- apex ×1.00 → 0.180
- gambit ×**1.98** → 0.396 → clamp a cap 0.35
- sentinel ×1.08 → 0.183

Renormalizado al bucket heurístico (33%):
- easy 9% → normal 16% → hard 16% → apex 14% → **gambit 28%** → sentinel 14%

El modelo recibe ~2.4× más exposición a gambit en los próximos 6 iters
de self-play, lo justo para que el value head tenga señal para
recalibrar. Si en el eval iter 114 gambit sube a 0.30, el deficit
baja, el peso vuelve a su rango normal y el ciclo se autoestabiliza.

### Trayectoria esperada

| eval | mecánica |
|---|---|
| iter 6 | base_mix (no hay eval previo) |
| iter 12 | leve boost para niveles débiles (todo es débil al principio) |
| ... | curriculum converge gradualmente al perfil "óptimo" para este modelo |
| pico | todos los niveles ≥ target → curriculum vuelve cerca de base_mix |
| regresión | si un nivel cae, boost automático en próximo bloque |

## Por qué esto puede funcionar mejor que el estático

1. **Cierra el loop**: eval pasa a ser una señal de control, no solo
   un monitor. El modelo siempre recibe self-play **proporcional a
   su debilidad actual**, no a una proyección estática hecha hace
   un mes.
2. **No reinventa anatomía a priori**: el peso óptimo de cada nivel
   no lo decidimos nosotros leyendo registros. Lo decide el modelo
   con sus propios resultados de eval.
3. **Tolera cambios de régimen**: si entre iter 50 y 80 el modelo
   "abre" su entendimiento de hard y baja la prioridad, el curriculum
   se da cuenta y libera ese 20% para otro nivel que sí lo necesita.
4. **Bajo costo computacional**: el ajuste es ~10 multiplicaciones
   por eval. Cero overhead de runtime.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| **Overfitting al adversario duro**: subir gambit a 35% por muchos iters puede enseñar exploits específicos vs gambit y degradar hard. | `cap=0.35` (no >35% nunca); `floor=0.05` (ningún nivel desaparece); EMA suaviza picos. |
| **Inestabilidad por ruido eval (n=64, IC95% ±0.12)**: un eval con mala suerte puede disparar boost falso. | `smoothing=0.5` en EMA — un eval no domina. Hacen falta 2-3 evals con deficit consistente para boost grande. |
| **Feedback loop oscilatorio**: si boost en iter 6 mejora gambit a 0.55 y luego baja peso, gambit cae otra vez → boost otra vez. | El EMA mantiene memoria. Si gambit oscila, el deficit promedio queda alto → boost sostenido hasta que el modelo realmente domine. |
| **Self-play pierde diversidad**: si easy se cae a 5%, el modelo no entrena lo que ya domina. | El `floor=0.05` lo evita; además los niveles que **ya domina** no necesitan más datos. |
| **Imposible diagnosticar postmortem si los weights efectivos no se guardan**. | **Logging obligatorio** del mix efectivo por iter en HF metadata + CSV. Sin esto el postmortem futuro es ciego. |
| **El cap=35% es arbitrario**. | Tuneable. Si v13 plateauea con cap=0.35, probar cap=0.45 en v14. |

## Comparación con métodos conocidos

### Prioritized Experience Replay (Schaul et al., 2015)

[Paper original (arXiv:1511.05952)](https://arxiv.org/abs/1511.05952).
DeepMind aplicó PER en DQN para Atari. La idea: muestrear transiciones
del replay buffer con probabilidad proporcional al TD-error. Las
transiciones "sorprendentes" (donde la red predice mal) reciben más
atención. Resultados: +40% mediana de score en 49 juegos de Atari.

**Diferencia con lo nuestro**: ellos priorizan transiciones; nosotros
priorizamos **adversarios**. Misma intuición de fondo: dedicar más
capacidad de aprendizaje a lo que cuesta.

### Curriculum Learning (Bengio et al., 2009)

[Paper original](https://www.researchgate.net/publication/221344862_Curriculum_learning).
Idea clásica: empezar fácil, escalar dificultad. Lo que ya hacemos
con `phase-based curriculum` en `src/training/curriculum.py`.

**Diferencia con lo nuestro**: ellos asumen orden conocido a priori
(fácil → difícil). Nosotros relajamos el supuesto: el "qué es difícil"
emerge del modelo mismo.

### Self-Paced Learning (Kumar et al., 2010)

[Paper](https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html).
Variante donde el modelo elige sus ejemplos según confianza: aprende
primero lo "easy" según el modelo mismo, no según un humano.
Conceptualmente cercano. Lo nuestro es la mirada inversa: el modelo
recibe MÁS de lo que NO maneja, no menos.

### League training (DeepMind AlphaStar, 2019)

[Paper en Nature](https://www.nature.com/articles/s41586-019-1724-z).
StarCraft II. DeepMind mantuvo "exploiters" — agentes especializados
en castigar debilidades del agente principal, integrados al pool de
sparring. Cuando un exploiter encontraba un agujero, todos los main
agents tenían que aprender a taparlo.

**Diferencia con lo nuestro**: AlphaStar tenía adversarios entrenados
de novo; nosotros usamos heurísticas fijas. Pero la filosofía es la
misma: el sparring debe **explotar** las debilidades del agente
principal, no entrenarlo solo en lo que ya sabe.

### Auto-Curriculum (Jiang et al., 2021, "Prioritized Level Replay")

[Paper (arXiv:2010.03934)](https://arxiv.org/abs/2010.03934). DeepMind/UCL.
En procedural-generated environments (Procgen), priorizaron niveles
con alto "value loss" para entrenar. Resultados: +50% sample efficiency
en MiniHack y Procgen.

**Más cercano a lo nuestro**: ellos también muestrean tareas
proporcionales a una señal de "esto cuesta". Diferencia: usaban
value loss directo del agente; nosotros usamos win-rate del eval.

### Reference checklist

- [Schaul et al., 2015 — Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Bengio et al., 2009 — Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380)
- [Kumar et al., 2010 — Self-Paced Learning](https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html)
- [Vinyals et al., 2019 — AlphaStar (league training)](https://www.nature.com/articles/s41586-019-1724-z)
- [Jiang et al., 2021 — Prioritized Level Replay](https://arxiv.org/abs/2010.03934)
- PM05 (`centinela`) — primera evidencia del problema en el repo
- PM10 (`umbral`) — confirmación del problema con per-level eval

## Plan de implementación para v13

### Archivos a tocar

1. `src/training/curriculum.py` — agregar `compute_adaptive_mix` y
   modificar `sample_opponent_from_curriculum` para recibir `eval_state`.
2. `src/training/loop_runtime.py` — mantener `ema_state` entre evals,
   pasarlo a `sample_opponent_from_curriculum`.
3. `src/training/monitor.py` — loguear `effective_mix` por iter en
   las stats que se persisten en HF metadata.
4. `train.py` — flag `--adaptive-curriculum {on,off}` para A/B test.
5. `tests/test_training_curriculum.py` — agregar:
   - test que un deficit alto produce boost esperado
   - test que `floor` se respeta
   - test que `cap` se respeta
   - test que la suma del mix sigue siendo 1.0
   - test que EMA converge en N iters

### Hparams para v13

| Param | Valor inicial | Justificación |
|---|---|---|
| `adaptive_target` | 0.55 | "aceptable"; el composite que centinela alcanzó |
| `adaptive_alpha` | 1.5 | gambit a 0.20 → boost ~1.8× (razonable) |
| `adaptive_floor` | 0.05 | ningún nivel desaparece (lección PM05/PM10) |
| `adaptive_cap` | 0.35 | evita collapse a un solo adversario |
| `adaptive_smoothing` | 0.5 | un eval no domina, dos sí |
| `adaptive_enabled_from_iter` | 12 | dejar warm-up de fase 1 antes de adaptar |

### Run v13: gates de éxito

- Composite pico > 0.55 (supera el techo histórico del repo).
- Per-level: ningún nivel < 0.30 al pico (vs gambit 0.094 en fulcro).
- h2h vs liga: > 0.65 sostenido por ≥ 2 evals consecutivos (no flash).
- Diego en arena: "se siente que intenta ganar" (vs vibe-test
  pendiente de fulcro).

### Plan B si v13 no mejora

- Si v13 plateauea cerca de v12: el bottleneck no es curriculum,
  es arquitectura. Subir d_model a 256 o num_layers a 12.
- Si v13 oscila peor: el EMA es muy ágil. Bajar `smoothing` a 0.25,
  o subir `cap` para no clipear deficits reales.

## Decisiones registradas

- v12 (`fulcro`, iter 90 candidato) se completa con su run actual.
  No reemplazamos curriculum a mitad de run; el experimento estático
  vale como baseline limpio.
- PM11 (este doc) queda firmado **antes** de codear v13. Si v13
  funciona, este postmortem es el por-qué. Si no, este postmortem
  es el por-qué-NO y entra plan C.
- Logging del effective mix es **mandatorio** desde día 1 — sin él
  no podemos diagnosticar PM12 si v13 falla.

## Autores y fecha

- Autor: Diego Obando + Claude (Claude Opus 4.7).
- Fecha: 2026-05-19.
- Estado: propuesta — no implementado al cierre de v12.

# Modelos entrenados — generaciones, apodos y ranking

Cada run de entrenamiento dejó un checkpoint con su propia historia. Para no
recordar paths feos como `policy_spatial_v8_iter_180.pt`, cada generación tiene
un apodo en clave que alude a lo que pasó en ese entrenamiento. La fuente única
de verdad es [`checkpoints/registry.json`](../checkpoints/registry.json) — todos
los scripts (`play_pygame.py`, `eval_checkpoint_vs_heuristic.py`,
`compare_checkpoints.py`) aceptan apodos donde antes solo aceptaban paths.

## Cómo usar

```bash
# Ver el ranking (default = combined: composite vs heurísticas + round-robin)
uv run python scripts/list_models.py
uv run python scripts/list_models.py --metric composite   # solo vs heurísticas
uv run python scripts/list_models.py --metric rr          # solo head-to-head
uv run python scripts/list_models.py --metric sentinel    # solo vs sentinel
uv run python scripts/list_models.py --full

# Jugar contra una generación (humano vs IA)
uv run python scripts/play_pygame.py --opponent model --ckpt liga --sims 80
uv run python scripts/play_pygame.py --opponent model --ckpt bogo --sims 80
uv run python scripts/play_pygame.py --opponent model --ckpt latest

# Modo espectador: dos generaciones se enfrentan
uv run python scripts/play_pygame.py --mode spectate \
  --p1-agent model --ckpt1 liga \
  --p2-agent model --ckpt2 centinela

# Evaluar una generación contra heurísticas
uv run python scripts/eval_checkpoint_vs_heuristic.py --checkpoint liga --games 24

# Duelo head-to-head entre dos generaciones
uv run python scripts/compare_checkpoints.py --checkpoint-a liga --checkpoint-b amnesia --games 32

# Re-poblar el registry con evals frescas
uv run python scripts/eval_all_checkpoints.py --games 12 --sims 80    # vs heurísticas
uv run python scripts/round_robin.py --games 8 --sims 80              # head-to-head
```

Aliases especiales: `latest` (más reciente), `best` (mejor score composite).

## Métricas del ranking

| Métrica     | Qué mide                                                       |
|---           |---                                                              |
| `composite` | Score promedio vs heurísticas (hard, apex, sentinel) — perfil  |
| `rr`        | Round-robin head-to-head entre todas las generaciones — fuerza |
| `combined`  | Promedio de las dos anteriores. **Default**.                   |
| `<level>`   | Score solo contra una heurística específica                    |

`composite` puede engañar (ej. centinela tenía buen composite pero era overfit a sentinel — PM05). `rr` es la métrica honesta: enfrenta a los modelos entre sí, no se puede sobreajustar. `combined` los promedia para que ambos contribuyan al ranking final.

## Sobre HF Hub y checkpoints intermedios

El registry solo lista los **campeones** (último iter de cada run) que están en `checkpoints/`. En HF Hub (`dieg0code/ataxx-zero`) hay snapshots intermedios cada `save_every` iters de cada run — son la misma identidad evolucionando, redundantes para el ranking. Si querés comparar evolución dentro de un run, descargá manualmente el iter que te interese a `checkpoints/` y agregalo al registry.

## Generaciones

### bogo (v1) — `model_iter_039.pt`
**1 de marzo 2026 · iter 39 · arquitectura legacy**

La primera generación. Solo aprendió a oscilar piezas (mover adelante y atrás
indefinidamente). Bogo sort hecho red neuronal: gesto sin dirección.
Arquitectura distinta a las demás (`policy_head` MLP plana, 3 canales de
observación) — al cargarla en la arena dispara automáticamente el shim de
compatibilidad legacy.

📄 Postmortem: [`src/model/docs/postmortem/01/README.md`](../src/model/docs/postmortem/01/README.md)

---

### reflejo (v2) — `policy_spatial_v2_iter_093.pt`
**4 de marzo 2026 · iter 93 · primera arquitectura espacial**

Pareció jugar bien hasta que descubrimos que el MCTS rompía empates eligiendo
siempre la primera jugada legal del action space. Jugaba por reflejo, no por
análisis: política plana en la apertura + bias determinista contaminando
self-play, eval y matches locales.

📄 Postmortem: [`src/model/docs/postmortem/02/README.md`](../src/model/docs/postmortem/02/README.md)

---

### chispazo (v3) — `policy_spatial_v3_iter_011.pt`
**8 de marzo 2026 · iter 11/220 (abortado)**

Run abortada temprano. Un destello que no prendió mecha. Sirvió como
diagnóstico antes de intentar de nuevo con reward shaping en aprendiz.

---

### aprendiz (v4a, v4b) — `policy_spatial_v4_iter_125.pt` y `_iter_135.pt`
**8 de marzo 2026 · iter 125/135 · primer reward shaping**

Primer intento serio con reward shaping. Tanteando: ya distingue posiciones
buenas de malas pero todavía no aprende a presionar. Era el camino, faltaban
fixes (que llegarían en centinela). Hay dos snapshots del mismo run: `aprendiz`
(iter 125) y `aprendiz-tardio` (iter 135) para medir si el shaping seguía
mejorando o ya había saturado.

📄 Postmortem: [`src/model/docs/postmortem/03/README.md`](../src/model/docs/postmortem/03/README.md)

---

### centinela (v6) — `policy_spatial_v6_iter_180.pt`
**17 de marzo 2026 · iter 180 · primer despegue real**

El primer despegue real del proyecto: juntó suficientes fixes correctos para
producir una mejora medible. Pero la victoria tenía truco: en el postmortem 05
descubrimos que **no aprendió Ataxx, aprendió a ganarle a sentinel** (la
heurística que más vio en entrenamiento). Domina al sparring conocido y pierde
contra heurísticas que nunca enfrentó. El nombre captura las dos caras:
vigilante poderoso, pero solo en su torre.

📄 Postmortems:
[`src/model/docs/postmortem/03/README.md`](../src/model/docs/postmortem/03/README.md) (despegue),
[`src/model/docs/postmortem/05/README.md`](../src/model/docs/postmortem/05/README.md) (overfit a sentinel)

---

### amnesia (v7) — `policy_spatial_v7_iter_140.pt`
**18 de marzo 2026 · iter 140 · bootstrap fallido desde centinela**

Bootstrap desde centinela que terminó peor que su origen. La causa no fue
catastrophic forgetting puro: fue reiniciar el loop con pesos buenos pero sin
el contexto que los hacía buenos — replay buffer limpio, iteración en 0,
curriculum reseteado y sin warmup. Olvidó cómo aprender de sí mismo.

📄 Postmortem: [`src/model/docs/postmortem/04/README.md`](../src/model/docs/postmortem/04/README.md)

---

### liga (v8) — `policy_spatial_v8_iter_180.pt`
**10 de mayo 2026 · iter 180 · league system + reward shaping consolidado**

Primera generación entrenada con league system: el sparring se diversifica
entre múltiples versiones del propio modelo y heurísticas variadas, en lugar de
overfittear a una sola como hizo centinela. Más balanceado contra todas las
heurísticas, gana 62.5% en duelos directos vs amnesia. Sigue siendo modesto
contra apex y hard, pero el techo subió.

---

## Mantener actualizado el registry

- Cuando termine una nueva run (v9, v10…), agregar una entrada al final de
  `checkpoints/registry.json` con codename, lore, hparams clave y eval
  placeholder en `"source": "pending"`.
- Para llenar evals reales:
  `uv run python scripts/eval_all_checkpoints.py --games 24 --sims 80`.
  Solo procesa generaciones con `source == "pending"` salvo que pases
  `--overwrite`.
- Si un apodo no termina de cuajar, basta editar el `codename` en el JSON: los
  scripts y la doc se actualizan solos.

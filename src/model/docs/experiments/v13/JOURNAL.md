# v13 — Journal (append-only)

Una entrada por sesión Kaggle o evento relevante. No editar entradas
viejas; corregir agregando entrada nueva con `[FIX-YYYY-MM-DD]`.

Formato sugerido por entrada:
```
## YYYY-MM-DD HH:MM UTC — [tipo: launch/eval/anomaly/decision/abort]

Qué pasó. Métricas si aplica. Decisión tomada con justificación.
```

---

## 2026-05-24 ~04:00 UTC — launch

Kernel Kaggle version 83 lanzado tras CI verde (`f352223`).
Diego frenó el run automático del scheduler y cambió accelerator
a T4×2. Run efectivo arrancando ahora con config `v13_config.json`.

PROPOSAL.md pre-registrado antes del primer eval. Próximo eval
esperado: iter 6 (~30 min después de iter 1).

## 2026-05-24 ~05:00 UTC — verificación setup

Logs de iter 2 confirman pipeline correcto:
- `Opponent mix: self=0.95, heuristic=0.00, random=0.05` (curriculum).
- `Checkpoint pool: umbral:1380, paralelo:1350, liga:1320` (top-3
  por rating del seed v13). League seed se materializó OK desde
  `seeds/league_v13_seed.json`.
- mcts_sims=800, episodes_per_iter=6, selfplay_workers=2.

Con `league_selfplay_checkpoint_prob=0.55`: estadísticamente cada
iter juega ~3.3 episodes vs league (umbral/paralelo/liga ponderados
por rating) y ~2.7 vs sí misma. Bootstrap operando como AlphaZero
canon.

## 2026-05-24 ~05:30 UTC — discusión: la inversión easy-vs-hard

Diego planteó la pregunta recurrente: ¿por qué los modelos
anteriores le ganaban a las heurísticas más duras (sentinel,
gambit, apex) pero perdían a las más débiles (easy, normal)? Es
contraintuitivo y vale anclarlo acá porque la respuesta es la
raíz teórica de por qué v13 abandona el curriculum heurístico.

### El fenómeno

Datos crudos de `umbral` (v11.2, iter 114), el peak histórico h2h
del repo:

| Nivel | umbral | Predicción "ingenua" |
|---|---|---|
| sentinel | 0.586 | ≤ apex (más fuerte que apex) |
| hard | 0.727 | ≥ easy (más débil que easy) |
| apex | 0.492 | ≥ sentinel |
| easy | 0.469 | el más alto |
| gambit | 0.375 | ≥ normal |
| normal | 0.266 | ≥ easy (porque normal < easy en ranking de "fuerza") |

La intuición lineal — *más fácil → más score* — falla en cinco de
los seis pares. El orden empírico no respeta el ranking de
"dificultad" que asumimos al diseñar las heurísticas.

### Por qué pasa: tres mecanismos superpuestos

**1. Distribution shift puro.**
En v12 el modelo veía hard/apex/sentinel ~70% del self-play y
easy ~0%. Los board states que produce easy son posiciones que el
modelo nunca observó durante entrenamiento — out-of-distribution.
El value head devuelve estimaciones esencialmente random ahí.
No es que el modelo *juegue mal* contra easy; es que **no sabe
evaluar las posiciones que easy produce**.

**2. Predictibilidad vs ruido.**
hard sigue una heurística determinística rígida (maximizar
material/movilidad con weights fijos). Esa predictibilidad es un
patrón explotable: el modelo aprende "si pongo la pieza en X,
hard responde Y, entonces puedo armar la trampa Z". Es
contrabatir un decision tree fijo, no jugar Ataxx.

easy es semi-aleatorio sin regla fuerte. No hay patrón que
aprender a explotar. El modelo no puede converger a una estrategia
óptima contra una política estocástica débil, porque la
"respuesta correcta" cambia entre partidas. Esto es teóricamente
predecible: la varianza del oponente sube → el regret minimization
requiere más muestras → con eval de 64 partidas el score parece
peor de lo que sería con N→∞.

**3. El modelo aprende a vencer al oponente, no a jugar el juego.**
Esta es la lección central de PM05 generalizada. Optimizar contra
una política específica produce un agente que es **bueno
contra-esa-política**, no que entiende la estructura del juego.
"Ganarle a sentinel" no implica "saber jugar Ataxx"; implica
"haber memorizado las debilidades sistemáticas de la política
sentinel". Cambiar el oponente expone que no había generalización.

### El principio general que subyace

Los seis niveles (easy → sentinel) **no son una escala lineal de
fuerza, son seis estilos distintos** con sus propios sesgos.
Tratarlos como puntos en un eje "difícil → fácil" es la falacia
de colapsar dos dimensiones (estilo, fuerza) en una. El modelo
no aprende a "subir de nivel"; aprende contraestrategias estilo
por estilo.

Esto tiene primos en otros dominios:
- Un boxeador entrenado contra zurdos puede perder contra un
  amateur diestro impredecible. No es porque el amateur sea mejor;
  es porque el boxeador entrenó una distribución estrecha.
- Un debate team que practica solo contra un estilo formal puede
  ahogarse contra alguien que argumenta desordenadamente. La
  "calidad" del oponente no captura la dimensión que importa.
- Bias cognitivo general: confundir *fluidez/predictibilidad* del
  oponente con *competencia*. El que es fácil de leer parece más
  débil aunque no lo sea, y viceversa.

### Por qué pure self-play debería romperlo

En self-play vía league no hay "nivel". La distribución de
oponentes es **el conjunto de versiones pasadas del propio
modelo**, que cubre un espectro continuo de styles que cambia a
medida que el modelo evoluciona. No hay una política fija que
memorizar ni una distribución específica a dominar.

La predicción operativa, ya en PROPOSAL.md: si v13 funciona, no
solo debería ganar a la league (señal de fuerza relativa), sino
recuperar los niveles que v12 nunca dominó — gambit ≥ 0.30,
normal ≥ 0.40 a iter 300. Esa recuperación sería evidencia de
generalización real, no exploitation de un nuevo conjunto de
patrones.

Si pasa lo contrario (gambit sigue clavado en 0.156 al iter 600
a pesar de pure self-play), la teoría de opponent exploitation
como causa raíz se cae y hay que mirar más profundo: capacidad
del modelo, profundidad del MCTS, representación del board, o
algo que no estamos viendo.

### Anclaje para futuros postmortems

Si v13 sale bien: este journal entry queda como evidencia del
mecanismo propuesto, y RESULTS.md compara contra estas
predicciones específicas.

Si v13 sale mal: el postmortem en `postmortem/12/` tiene que
explicar **por qué este mecanismo no aplicaba** o **qué
mecanismo adicional lo enmascaró**. No retroceder a "el
curriculum era el problema, ya lo arreglamos" sin evidencia
nueva — esa narrativa ya falló en v8, v10, v12.

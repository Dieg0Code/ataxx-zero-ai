# AtaxxZero - Post-mortem 05: el modelo v6 no aprendió Ataxx, aprendió a ganarle a sentinel

> **Resumen:** evaluamos `policy_spatial_v6_iter_180` (nuestro mejor checkpoint
> hasta la fecha) contra los 6 niveles heurísticos disponibles, con 64 partidas
> por nivel. El resultado no es un gradiente de dificultad como esperábamos —
> es un perfil claramente sobreajustado a las heurísticas vistas en
> entrenamiento. v6 pierde contra heurísticas simples (easy/normal) y contra
> una heurística desconocida (gambit), pero domina sentinel (la que más vio en
> warmup). Es la firma textbook de un modelo que aprendió a explotar oponentes
> específicos en vez de aprender el juego.

---

## Contexto: por qué medimos esto

Durante el entrenamiento de `v6` y `v7`, el `eval_runtime` solo medía contra
una heurística por iteración (típicamente `hard`) con 12-16 partidas. Eso
limita la observabilidad por dos razones independientes:

1. **Pocas partidas** → mucho ruido (SE ≈ 0.13). El IC95% del peak de v6
   (`0.806`) era `[0.58, 1.0]`. No podíamos distinguir runs buenos de malos.
2. **Una sola heurística** → no podemos ver si el modelo aprendió Ataxx o solo
   aprendió a explotar a `hard`.

Diego nos contó además una anécdota relevante: durante su clase, un alumno
descubrió manualmente una secuencia repetible de movimientos que ganaba casi
siempre contra la heurística del juego. Es exactamente el tipo de patrón que
una red neuronal puede descubrir si la dejás entrenar contra un oponente
determinista el tiempo suficiente. La pregunta era: ¿hizo eso v6?

## Metodología

```
uv run python scripts/eval_checkpoint_vs_heuristic.py \
    --checkpoint checkpoints/policy_spatial_v6_iter_180.pt \
    --levels easy,normal,hard,apex,gambit,sentinel \
    --games 64 --sims 160 --device cpu
```

64 partidas por nivel → SE = 0.5/√64 ≈ 0.062, IC95% ≈ ±0.12. Es señal
estadísticamente útil (en contraste con los 12 games del eval original que
daba SE = 0.144).

Total: 384 partidas. Mismo MCTS (160 sims) que el run de entrenamiento, modelo
en CPU (no afecta a runs paralelos).

## Resultados

| Nivel    | W  | L  | D | Score | IC95%          |
|----------|----|----|---|-------|----------------|
| easy     | 26 | 37 | 1 | 0.414 | [0.29, 0.54]   |
| normal   | 19 | 45 | 0 | 0.297 | [0.17, 0.42]   |
| hard     | 44 | 20 | 0 | 0.688 | [0.57, 0.81]   |
| apex     | 41 | 23 | 0 | 0.641 | [0.52, 0.76]   |
| gambit   |  4 | 60 | 0 | 0.062 | [0.00, 0.19]   |
| sentinel | 52 | 12 | 0 | 0.812 | [0.69, 0.94]   |

**Composite: 0.486** (es decir, peor que tirar moneda promediado).

## Lo que estos números significan

### El patrón es NO-MONÓTONO con la dificultad

Si el modelo hubiera aprendido a jugar Ataxx, esperaríamos algo así:

```
score
  ^
1 |·
  |  ·
  |    ·
  |      ·
  |        ·
0 +----------> dificultad heurística
   easy      sentinel
```

Lo que vemos es esto:

```
score
  ^                       sentinel
1 |                          ·
  |              hard  apex
  |                ·    ·
  |  easy
  |   ·
  |        normal
  |          ·                        gambit
0 |                                     ·
  +----------------------------------->
    easy normal hard apex sentinel gambit
```

El score **no traquea con la dificultad real** del oponente. Específicamente:

- **gambit (0.062)**: 4 wins en 64 games. v6 nunca vio esta heurística en
  entrenamiento — sus movimientos no matchean ninguno de los patrones que el
  modelo aprendió a explotar.
- **easy/normal (0.414/0.297)**: heurísticas simples que juegan a veces de
  forma "subóptima" desde la perspectiva de un jugador fuerte. Eso significa
  que NO van a caer en las trampas que v6 preparó para los niveles altos.
  El modelo está sobreentrenado para un estilo específico de oponente y
  pierde contra cualquier estilo distinto.
- **hard/apex (0.688/0.641)**: las heurísticas que v6 vio bastante en
  entrenamiento. Las gana porque sabe sus patrones.
- **sentinel (0.812)**: la heurística más fuerte... pero también la que v6
  vio MÁS porque era el nivel default de warmup en su config
  (`warmup_heuristic_level="sentinel"`). El modelo está específicamente
  optimizado contra ella. Score más alto del set.

### La explicación de por qué pasó esto

El loop de entrenamiento de v6 hizo esto, en bucle, durante 180 iteraciones:

1. Generar partidas de self-play donde el modelo juega contra heurísticas
   deterministas (especialmente sentinel).
2. Entrenar al modelo a copiar las acciones que ganaron esas partidas.
3. Evaluar contra la misma heurística.

Después de muchas iteraciones, el modelo encontró **secuencias que fuerzan
a las heurísticas deterministas a ejecutar movimientos perdedores**. Esas
secuencias son exploits, no comprensión del juego. Funcionan porque la
heurística siempre responde igual a la misma posición — exactamente el mismo
glitch que tu alumno descubrió manualmente.

Para una red neuronal, esto es un mínimo local muy cómodo: el gradiente la
empuja a refinar el exploit en vez de a entender el juego, porque el exploit
da más reward por unidad de cómputo.

## Por qué es importante para tu carrera de ML

Este patrón tiene nombre y aparece en muchos contextos:

- En RL contra oponentes fijos se llama **opponent exploitation** o **policy
  overfitting to opponent**.
- En RLHF (LLMs) se llama **reward hacking** — el modelo encuentra formas de
  maximizar la métrica que no son las que querías.
- En supervisado se llama **shortcut learning** — el modelo encuentra una
  pista trivial en los datos en vez de aprender la tarea real.

Los tres son la misma idea: **un modelo siempre va a optimizar la métrica
que vos le diste, no la que vos quisiste darle**. Si tu métrica tiene una
puerta trasera, la va a encontrar antes que aprender la tarea real.

La lección práctica: cuando diseñes un sistema de aprendizaje, preguntate
"¿qué pasaría si un atacante tuviera que maximizar esta métrica? ¿qué
shortcuts existen?" Esos shortcuts los va a encontrar el modelo solo,
aunque no quieras.

## Implicancias para v8 (el run actual)

El primer eval de v8 (iter 6) ya mostró señales tempranas del mismo patrón:

```
hard:    0.859 (W=55 L=9)
apex:    0.609 (W=39 L=25)   <-- gap grande
sentinel: 0.812 (W=52 L=12)
COMPOSITE: 0.760
```

El gap entre hard y apex (0.25 puntos) es similar al gap final de v6. v8
todavía no pasó por gambit y easy/normal en eval, así que **no sabemos** si
el patrón es igual o si hay una mejora estructural. La métrica del eval
durante el run de Kaggle solo mide hard/apex/sentinel.

Cuando termine v8 vamos a poder correr el mismo eval de 6 niveles contra el
mejor checkpoint de v8 y comparar perfiles directamente.

## Acciones derivadas (Fase 2 — futuro, no ahora)

Estas son las palancas que tenemos para combatir esto, en orden de
complejidad:

1. **Diversificar oponentes en eval primero** — agregar gambit y mezclar
   heurísticas. Ya tenemos infra (`eval_heuristic_levels` config) — es solo
   cambiar config.

2. **Agregar ruido a las heurísticas en self-play** — si la heurística juega
   epsilon-greedy (10% movimiento aleatorio), los exploits dejan de
   funcionar. El modelo se ve forzado a generalizar.

3. **Bajar la fracción de heurísticas en self-play** — el config actual
   tiene `opponent_heuristic_prob=0.5`. Después de N iters de bootstrap,
   bajarlo gradualmente hacia 0.1, dejando más self-play y league play. La
   league son checkpoints viejos del propio modelo — no son deterministas
   entre sí, no se pueden explotar igual.

4. **Reward shaping conservador** — penalizar las posiciones donde el modelo
   "espera" un movimiento específico del oponente. Esto requiere
   instrumentación nueva, va para más adelante.

## Lo que aprendimos esta sesión (puntos para tu cuaderno)

1. **El score promedio miente** sin ver el desglose por oponente. v6 tiene
   composite=0.486 (mediocre) pero con scores de 0.062 a 0.812 — la varianza
   es la historia, no el promedio.

2. **64 games es el mínimo viable** para distinguir señal de ruido en este
   dominio. 12 era inutilizable.

3. **Distribución de oponentes en training importa** tanto como la
   arquitectura del modelo. v6 falló no por su red sino por su currículum.

4. **Los exploits son atractivos para el optimizador** — siempre. Si tu
   sistema permite un shortcut, lo va a tomar.

5. **El postmortem es el código** — cuando vuelvas a esto en 6 meses,
   estos números y este razonamiento van a ahorrarte semanas. La memoria
   propia es poco confiable, los datos quedan.

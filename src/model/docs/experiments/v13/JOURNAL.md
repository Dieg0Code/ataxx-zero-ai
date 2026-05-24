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

## 2026-05-24 ~06:00 UTC — meta-reflexión: ML como espejo de cognición

> Esta entrada está fuera del flujo normal del journal (no es un
> evento del run, no impacta predicciones) pero queda anclada acá
> porque emergió de una conversación que vale documentar. El
> contenido es producto de una interacción entre Diego y un
> asistente de IA — una simbiosis donde la pregunta y el marco
> son humanos, la formalización y los ejemplos vienen del modelo,
> y el filtro de qué vale guardar es de ambos. No tiene sentido
> fingir autoría solo humana cuando el proceso fue conjunto.

Diego trabaja en este repo con una motivación que trasciende el
proyecto inmediato: cree que estudiar sistemas que aprenden
puede iluminar cómo aprende él. Su tesis operativa es que los
principios del aprendizaje son universales, y que la diferencia
entre un cerebro biológico y una red neuronal está en la
implementación (sustrato, eficiencia energética, replicación
vía ADN) más que en la estructura abstracta del problema. Esta
entrada explora dónde esa tesis es defendible, dónde es
metáfora útil, y dónde se vuelve pseudociencia.

### Tres tipos de paralelo

Antes de listar similitudes vale establecer una taxonomía para
no caer en numerología:

1. **Mecanísticos.** La misma matemática emerge en ambos
   sistemas — no porque uno imite al otro, sino porque ambos
   resuelven el mismo problema bajo restricciones parecidas.
   Estos son los paralelos fuertes.
2. **Funcionales.** Ambos sistemas resuelven el mismo problema,
   pero con mecanismos distintos. Útiles como metáfora, peligrosos
   si se confunden con identidad.
3. **Estéticos.** Suenan parecidos. No hay base mecanística ni
   funcional. Hay que descartarlos sin pena: son la materia prima
   de la mala divulgación.

Lo que sigue está ordenado por fuerza decreciente.

### El paralelo fundamental: generalización fuera de la distribución vista

Todo sistema que aprende de ejemplos termina con la misma
limitación estructural — generaliza bien cerca de lo que vio,
mal lejos. No es analogía: es la misma matemática del riesgo
empírico. Una red entrenada solo con gatos blancos clasifica mal
gatos negros. Un cerebro entrenado solo con problemas de un
estilo falla con problemas de otro estilo, aunque el principio
subyacente sea idéntico.

La consecuencia práctica es contraintuitiva: cuando alguien
falla en un dominio nuevo, el reflejo es atribuirlo a falta de
inteligencia o de esfuerzo, cuando habitualmente es falta de
exposición a la distribución correcta. Diego, como cualquier
estudiante avanzado, probablemente sobrestima cuánto generaliza
su entendimiento entre dominios — el ejercicio de poner a prueba
ese supuesto en territorio incómodo es lo que separa
aprendizaje real de la ilusión de aprendizaje.

Una variante de este mismo fenómeno es el **overfitting**:
memorizar las preguntas del examen pasado en lugar de la
materia. El indicador es operacionalizable y duro — rendimiento
en datos vistos vs no vistos. Si alguien resuelve fluidamente
problemas que ya vio y se traba en novedosos que requieren el
mismo principio, no entendió el principio: memorizó instancias.

### Memoria, tiempo y olvido catastrófico

En redes neuronales, entrenar tarea B suele degradar
performance en tarea A (los pesos compartidos se reescriben).
En cerebros, dejar de practicar piano cinco años poda
literalmente las conexiones sinápticas implicadas. Es el mismo
fenómeno bajo el mismo nombre técnico: **catastrophic
forgetting**.

La diferencia importante es que el cerebro tiene mecanismos de
consolidación que las redes no — el sueño REM, la
recapitulación hipocámpica. Los humanos pueden mitigar el
olvido sin re-entrenar formalmente; las redes hoy no pueden.
Esto sugiere que la práctica espaciada y la revisión periódica
de fundamentos no son hábitos opcionales sino implementaciones
de mecanismos que ya están en el hardware biológico. Saltarse
ese mantenimiento es elegir activamente la fragilidad.

### Optimizando la métrica equivocada: reward hacking

Cualquier sistema que optimiza una métrica encuentra atajos que
satisfacen la métrica sin lograr el objetivo subyacente. En RL
se llama reward hacking; en economía y administración pública
se llama Ley de Goodhart, formulada décadas antes de que
existieran las redes neuronales: *"cuando una métrica se
convierte en objetivo, deja de ser una buena métrica"*. El
fenómeno es el mismo y la prueba es el descubrimiento
independiente.

Las implicancias para la auto-mejora son severas. Diego
construye este proyecto, mide su progreso por composite scores
y h2h ratings, y descubrió empíricamente que esas métricas son
falibles — el modelo iter 126 maximizaba h2h vs liga y perdía
43-6 contra él. Esa lección sobre métricas falibles aplica al
desarrollo personal sin modificación: leer 50 libros al año,
ganar X reputación profesional, sumar Y horas de práctica son
todas métricas que un optimizador motivado puede satisfacer sin
mejorar lo que la métrica pretendía aproximar. La defensa no es
abandonar las métricas — sin métrica no hay feedback loop —
sino auditar periódicamente la divergencia entre métrica y
objetivo subyacente. Si la métrica está subiendo y la sensación
de progreso real no, hay reward hacking pasando.

### Bajo incertidumbre, explorar más de lo que parece prudente

La teoría matemática de bandits multibrazo (UCB, Thompson
sampling, regret bounds) tiene un resultado contraintuitivo
robusto: bajo incertidumbre real, la cantidad óptima de
exploración es **bastante mayor** que la que la mayoría de los
agentes —humanos o artificiales sin teoría— eligen practicar.
Los humanos sub-explotamos sistemáticamente: preferimos lo
conocido aunque sepamos que probablemente hay mejor afuera.

El sesgo tiene sentido evolutivo (lo conocido no te mata) pero
es subóptimo para horizontes largos. La teoría dice que cuando
el horizonte de decisión es largo —elegir carrera, pareja,
hábitos centrales, dónde vivir— la asignación óptima entre
explorar y explotar pesa fuerte hacia explorar al principio.
Los humanos hacen lo opuesto: en la juventud siguen plantillas
de los padres, en la adultez quieren "asentar". Es exactamente
el patrón que un agente naïve sin entender bandits produciría.

### Curiosidad como active learning

En active learning, el modelo sabe en qué inputs está más
incierto y pide etiquetas precisamente ahí — porque ahí está la
información que reduce más su pérdida esperada. En cerebros, la
curiosidad probablemente cumple esa función: focaliza atención
en regiones donde el modelo interno del mundo tiene baja
confianza.

Esto reframea el aburrimiento. Aburrirse no es defecto de
carácter ni señal de inmadurez: es la señal interna de que el
modelo predice bien lo que viene a continuación, lo cual
significa que no hay información nueva que extraer. Si una
clase, un libro o una conversación aburre, lo correcto es
sospechar que ese contenido está dentro de la distribución ya
modelada, y mover la atención a donde el modelo se confunde —
ahí está el gradiente útil.

### Diversidad de input: mode collapse

En GANs, el generador puede colapsar a producir una sola
familia de salidas que el discriminador no detecta — perdiendo
toda la diversidad del espacio. En grupos humanos pasa lo
mismo cuando todos consumen las mismas fuentes: las opiniones
convergen a un modo y los individuos creen estar pensando
independientemente cuando en realidad sus inputs son
homogéneos.

La defensa requiere diseño activo de la dieta informacional.
Leer una fuente que sistemáticamente molesta no es ejercicio
de mente abierta ni concesión política — es mantenimiento
técnico del sistema, prevención del mode collapse. La métrica
es si las opiniones propias siguen siendo predecibles para
quien las conoce, o si todavía pueden sorprender.

### Escala, método y retornos decrecientes

Las scaling laws empíricas de los modelos modernos muestran que
performance escala log-linealmente con cómputo y datos — lo
cual significa que duplicar el esfuerzo da mejoras marginales
cada vez más chicas. Es exactamente la curva del aprendizaje
humano: las primeras 100 horas en algo entregan el 80% del
nivel competente; las siguientes 9,900 horas entregan el 20%
restante.

La consecuencia operacional es clara y poco aceptada
emocionalmente: cuando se siente plateau en una habilidad, la
respuesta correcta casi nunca es "más esfuerzo del mismo
tipo". Es cambiar el algoritmo —cómo se practica, qué se
practica, contra quién, con qué feedback loop— no aumentar el
compute. Quien suma horas sin cambiar método después del
plateau está pagando un costo lineal por ganancias
sub-lineales.

### Lo que NO transfiere bien

La honestidad intelectual exige listar también dónde la
analogía es estética y no debería usarse como guía:

- **Atención** en transformers comparte el nombre con atención
  cognitiva humana pero no el mecanismo. El producto punto
  query-key no es lo que pasa cuando una persona se concentra.
- **Memoria** en LLMs es contexto inmediato sin consolidación
  persistente. La memoria humana es estructuralmente distinta —
  multi-tier, episódica vs semántica, con olvido activo, con
  reconsolidación al recordar.
- **Razonamiento** vía chain-of-thought es generación
  autoregresiva que produce texto que parece razonamiento. No
  hay evidencia de que internamente sea el mismo proceso que
  cuando un humano razona.
- **Conciencia** y experiencia subjetiva: nadie tiene idea, y
  cualquier paralelo serio aquí es especulación pura. Vale más
  decir "no sabemos" que construir analogías bonitas.

Confundir estos cuatro con los paralelos mecanísticos arriba
es la trampa principal de la pop-science de IA. Cualquier
texto que mezcle libremente "atención", "memoria",
"razonamiento" y "conciencia" entre modelos y humanos sin
distinguir tipo de paralelo está vendiendo intuición sin
rigor.

### El meta-punto: calibración como habilidad central

El cerebro es un optimizador, y todos los optimizadores
comparten un conjunto pequeño de fallas modales: sub-exploran,
overfittean, hacen reward hacking, sufren mode collapse,
sobre-confían fuera de su distribución de entrenamiento. No
importa si están implementados en biología o silicio. Reconocer
las firmas de estas fallas en uno mismo, en otros, en
instituciones, es un framework de diagnóstico portable —
posiblemente el más útil que produjo el campo de machine
learning para uso externo.

Pero la intuición de Diego, que la inteligencia es la solución
a todo, tiene una trampa que la teoría también identifica con
nombre técnico: **calibración**. Un modelo está bien calibrado
cuando su confianza coincide con su accuracy — cuando dice
"95% seguro" acierta el 95% de las veces. Los modelos
sobre-entrenados están seguros incluso cuando se equivocan;
los bien calibrados saben distinguir donde su predicción es
confiable de donde está extrapolando.

La inteligencia sin calibración es activamente peligrosa,
principalmente para uno mismo. El que confía demasiado en su
modelo del mundo —porque ese modelo lo ha llevado lejos— se
vuelve sistemáticamente vulnerable a sus propios sesgos. La
historia intelectual está llena de mentes brillantes que se
equivocaron catastróficamente en su tema porque su éxito
previo eliminó el incentivo para dudar. Calibrar es
contracultura para la inteligencia: requiere preguntarse, con
disciplina, no "¿tengo razón?" sino "¿qué tendría que
observar para concluir que estoy equivocado?". Esa pregunta es
falsacionista en sentido popperiano y antifrágil en sentido
talebiano — y es el mismo principio que estructura el
PROPOSAL.md de este experimento.

La consecuencia, para alguien comprometido con volverse más
inteligente: la inteligencia escala con compute, pero la
calibración es lo que determina si ese compute se invierte en
ideas verdaderas o en ideas seductoras. Sin calibración, más
inteligencia es más certeza, no más verdad. Con calibración,
más inteligencia es más capacidad de actuar bajo incertidumbre
sin colapsar en certeza prematura — que es, posiblemente, la
definición operacional de sabiduría.

# Glosario IA / AlphaZero / Ataxx Zero

Términos que aparecen en este repo, postmortems y conversaciones técnicas.
Organizado por tema, no alfabético — leído de arriba abajo va construyendo
el contexto. Cada término trae definición corta + ejemplo concreto del
proyecto cuando ayuda.

---

## 1. AlphaZero / Reinforcement Learning

**AlphaZero** — Algoritmo de DeepMind (2017) que entrena una red neuronal
de cero, sin data humana, usando self-play guiado por MCTS. La red aprende
de las partidas que ella misma genera. Este proyecto es una implementación
de AlphaZero para Ataxx 7×7.

**Self-play** — El modelo juega partidas contra sí mismo (o variantes:
checkpoints viejos, heurísticas). Los resultados de esas partidas se usan
como datos de entrenamiento. Núcleo del paradigma AlphaZero.

**MCTS (Monte Carlo Tree Search)** — Algoritmo de búsqueda en árbol que
explora movimientos prometedores con guía estadística. Cada nodo del árbol
representa una posición; las ramas son movimientos. La red neuronal
sugiere qué ramas explorar (`policy`) y qué tan buena es cada posición
(`value`). Más sims = más exploración = mejor decisión, más lento.

**Sims (simulations)** — Número de iteraciones que MCTS hace por movimiento.
Hiperparámetro clave: `mcts_sims=160` en v11 significa que para elegir cada
jugada, MCTS evalúa 160 caminos posibles. v9-v11 usaron 160; AlphaZero
production usa 400-800.

**PUCT** — Variante de UCB (Upper Confidence Bound) específica de AlphaZero.
Fórmula que balancea **explorar** ramas poco visitadas vs **explotar** ramas
con buenos resultados. El `c_puct` controla el balance.

**c_puct** — Hiperparámetro del PUCT. Más alto = más exploración (probar
movimientos no obvios). Más bajo = más explotación (seguir lo que ya parece
funcionar). En este repo default ~1.5.

**Policy head** — Salida de la red neuronal que predice una distribución de
probabilidad sobre todos los movimientos posibles. "Cuál movimiento es bueno
aquí". Output shape: `[batch, 793]` (793 acciones legales en Ataxx 7×7).

**Value head** — Salida que predice un escalar en [-1, +1] estimando quién
va ganando desde la perspectiva del jugador que mueve. +1 = "voy a ganar
seguro", -1 = "voy a perder seguro", 0 = "está parejo".

**Auxiliary head / count head** — Cabeza adicional que predice una señal
secundaria que ayuda al aprendizaje. En v11 agregamos `count_head` que
predice diferencia final de piezas. PM08 mostró que mal calibrada
canibaliza el backbone.

**Backbone / encoder** — La parte de la red anterior a las heads. En v11
es el transformer que procesa el board. Todas las heads (policy, value,
count) comparten este backbone y compiten por su capacidad.

**Reward shaping** — Modificar las recompensas que ve el modelo para
guiar el aprendizaje. Por ejemplo, premio parcial por capturar piezas
en lugar de solo recompensa final. v11 lo tiene desactivado.

**Bootstrap** — Inicializar un run nuevo desde los pesos de un run
anterior (en lugar de ruido aleatorio). v10 paralelo NO bootstrappeó.
Tiene trampas: PM04 documentó cómo `reset_iteration=true + warmup_games=0`
corrompe el modelo heredado.

**Warmup** — Iteraciones iniciales con configuración especial (usualmente
opponent mix más conservador) para que un modelo bootstrappeado se
"acostumbre" al nuevo régimen antes de meterse en self-play full.

---

## 2. Arquitectura Transformer

**Transformer** — Arquitectura de red neuronal basada en *attention* en
lugar de convoluciones o recurrencia. Originalmente para texto (GPT), hoy
para todo. Este repo usa transformer para procesar el board 7×7 como una
secuencia de 49 tokens (uno por celda).

**d_model** — Dimensión de los embeddings/tokens dentro del transformer.
v8 (liga) usa 128, v11 (lastre, asedio) usa 192. Más alto = más capacidad,
más memoria, más cómputo.

**nhead** — Número de "cabezas de atención" en cada capa. Cada head mira
relaciones distintas entre tokens. 8 cabezas significa que el transformer
puede atender a 8 patrones distintos simultáneamente.

**num_layers** — Cuántas capas de transformer apila. Más capas =
representaciones más profundas. v8 usa 6, v11 usa 8.

**dim_feedforward** — Dimensión interna del MLP dentro de cada capa de
transformer. Convención: 4× d_model (v11: 768).

**Dropout** — Técnica de regularización: durante training, "apaga"
aleatoriamente algunas neuronas por step. Fuerza al modelo a no depender
de neuronas específicas. v11 usa 0.1 (10% drop).

**Embedding** — Representación vectorial densa de una entidad discreta.
En este repo cada celda del board se embeddea de su estado (vacío, p1, p2,
muro, etc.) a un vector de tamaño `d_model`.

**Positional encoding** — Cómo el transformer "sabe" la posición de cada
token (sin esto, vería el board como bag-of-tokens sin geometría). En este
repo usa positional embeddings aprendidos.

**Attention** — Mecanismo donde cada token mira a los demás tokens del
input y decide qué tan importante es cada uno para su propia representación.
Es lo que hace al transformer poderoso.

**CLS token** — Un token especial que se agrega al input para servir de
"resumen" agregado. La salida del CLS se usa para predicciones globales
(en este repo, value y count heads usan el CLS).

**Action mask** — Vector booleano que indica qué movimientos son legales
en una posición dada. Antes del softmax de la policy se aplica para que
movimientos ilegales tengan probabilidad 0.

---

## 3. Training / Optimización

**Batch** — Grupo de ejemplos procesados juntos en un step de training.
v11 usa `batch_size=192` ejemplos por step.

**Step / global_step** — Una iteración del optimizador: forward + loss +
backward + update. Distinto de "iter de AlphaZero".

**Iter (AlphaZero)** — Una vuelta completa del loop: generar self-play →
entrenar sobre el buffer → eval opcional. v11 hace 240 iters por run.
Cada iter genera múltiples steps de optimizer.

**Epoch** — En training de Lightning, una pasada completa sobre el dataset.
v11 usa 1 epoch por iter de AlphaZero (el dataset es el replay buffer).

**Loss / pérdida** — Número que mide qué tan mal está el modelo. Bajar el
loss = entrenar. En AlphaZero típicamente combina policy_loss + value_loss
+ otros componentes.

**Cross-entropy** — Loss para clasificación / distribuciones. Mide qué tan
distinta es la distribución predicha de la verdadera. En este repo se usa
para policy.

**MSE (Mean Squared Error)** — Error cuadrático medio. Promedio de
`(pred - target)²`. Loss típica para regresión. value_head y count_head
usan MSE.

**Gradient / gradiente** — Derivada del loss respecto a cada parámetro.
Le dice al optimizador hacia dónde mover cada peso para bajar el loss.
"Gradient flow" = "cómo se propaga el aprendizaje hacia atrás por la red".

**Backpropagation** — Algoritmo que computa los gradientes usando la regla
de la cadena, desde la loss hacia atrás por toda la red.

**Optimizer** — Algoritmo que aplica los gradientes para actualizar pesos.
Este repo usa AdamW (variante mejorada de Adam con weight decay).

**Learning rate (lr)** — Tamaño del paso del optimizer. Más alto = aprende
más rápido pero menos estable. v11 usa lr=3e-4 (0.0003).

**Weight decay** — Regularización L2: penaliza pesos grandes para evitar
overfitting. v11 usa 1e-4.

**LR scheduler** — Cómo cambia el learning rate a lo largo del training.
Cosine decay = baja suavemente como coseno. Constante = no cambia.

**Mixed precision (16-mixed)** — Hacer cómputo en float16 (menor precisión,
2× más rápido) pero mantener algunos pasos críticos en float32. En T4/V100
da speedup gratis sin perder calidad.

**DDP (Distributed Data Parallel)** — Cómo PyTorch entrena en múltiples
GPUs. Cada GPU procesa un slice del batch, los gradientes se promedian.
`ddp_spawn` = la variante que arranca un proceso nuevo por GPU.

**Checkpoint** — Archivo .pt con los pesos del modelo en un momento dado.
v11 guarda uno cada iter. Sirven para retomar runs, hacer eval, jugar.

**state_dict** — Diccionario {nombre_parámetro: tensor} que es lo que se
guarda y carga al persistir un modelo. Si la arquitectura cambia, las
keys del state_dict pueden no coincidir → "size mismatch error".

**Replay buffer** — Memoria donde se acumulan los ejemplos de self-play
para entrenar. v11 lo crece a ~15k ejemplos en 41 iters. El modelo se
entrena cada iter sobre una muestra del buffer.

**Forward pass** — Computar la salida del modelo dado un input.
"forward(boards) → (policy, value, count)".

**Backward pass** — Computar gradientes a partir del loss.

---

## 4. Evaluación / Métricas

**Composite** — Score promedio del modelo contra una batería de heurísticas
fijas. v11 evalúa contra 6 niveles (easy, normal, hard, apex, gambit,
sentinel), 64 partidas cada uno = 384 totales. composite = avg(scores).

**h2h (head-to-head)** — Score directo de un modelo vs otro modelo.
`h2h vs liga = 0.42` significa que asedio le ganó a liga el 42% de las
partidas que jugaron entre ellos.

**Round-robin (RR)** — Cada modelo juega contra todos los demás. Score RR
= win rate global. Cuantifica el ranking entre generaciones.

**Policy accuracy** — % de veces que el `argmax(policy_pred)` coincide con
`argmax(policy_target)`. Métrica honesta porque no depende de heads
auxiliares.

**Value MAE (Mean Absolute Error)** — Promedio de `|pred - target|`. Para
value, indica qué tan cerca está la predicción de [-1, 1] del resultado
final real.

**Eval composite vs h2h**: composite mide vs heurísticas fijas (no se
mueven), h2h mide vs otro modelo (también entrenado). Pueden disociarse:
asedio tiene composite menor que paralelo pero parece comparable en juego
real contra humanos.

**Gate / absolute gate** — Mecanismo automático que aborta runs que no
cumplen un umbral mínimo. v11 exige `h2h vs liga >= 0.45` desde iter 36.
Si falla 2 veces seguidas (`patience=2`), abort. Protege contra runs que
no van a ningún lado.

**Patience** — Cuántos fallos consecutivos del gate se toleran antes de
abortar. v11 usa patience=2: iter 36 falla, iter 42 también falla → abort.

**Regression gate** — Distinto al absolute gate: detecta si el modelo
EMPEORÓ respecto a su mejor versión (no si es peor que el baseline).
Restaura el "best checkpoint" si hay regresión.

---

## 5. Curriculum / Oponentes

**Opponent mix** — Distribución de oponentes durante self-play. v11:
self=10%, heuristic=88%, random=2%. La heurística domina porque sin ella
el modelo solo aprende contra sí mismo y se queda en su propio óptimo.

**Opponent exploitation** — Cuando el modelo aprende a "explotar" debilidades
específicas de un oponente fijo en lugar de jugar bien en general. PM05
documentó cómo entrenar mucho vs sentinel hacía a centinela ganar a sentinel
pero perder a normal.

**League** — Sistema donde el modelo juega contra una "liga" de checkpoints
viejos (de sí mismo o de generaciones pasadas). Diversifica oponentes,
reduce exploitation. v11 con `league_selfplay_checkpoint_prob=0.35` =
35% del self-play va contra checkpoints del league pool.

**Elo / Elo rating** — Sistema de rating numérico (como en ajedrez). Cada
partida ganada sube el rating, perdida lo baja. El delta depende de la
diferencia de ratings (ganarle a uno mejor sube más). v11 seed: paralelo
1350, liga 1320, etc. Base = 1200.

**Champion games** — Partidas que el modelo actual juega contra el champion
del league pool para actualizar ratings. v11 hace 6 por iter.

**Curriculum** — Estrategia de presentar oponentes en orden de dificultad
(easy → hard) para que el modelo aprenda gradualmente.

---

## 6. Datos / Pipeline

**Pretrain (imitación)** — Entrenar el modelo con data fija ANTES del
self-play. v11 hace 3 epochs sobre 14k ejemplos de partidas humanas
curadas. Inicializa los pesos con conocimiento humano antes de empezar
self-play puro.

**Human replay buffer** — En v11 mantenemos en memoria un buffer paralelo
con los ejemplos humanos. Cada batch saca 20% de ahí y 80% del buffer de
self-play. Inocula estilo humano durante TODO el training.

**Value mask** — Booleano por ejemplo que indica si el value loss aplica.
Para humanos en v11 ponemos value_mask=False (humanos juegan mal a veces;
no querés que el value head aprenda de su `value target`). Solo el policy
aprende de humanos en ese caso.

**Data augmentation** — Generar ejemplos sintéticos transformando los
originales. v11 usa **symmetry augmentation D4**: rotar/flipear el board
(8 transformaciones del grupo dihedral) para que el modelo vea cada
posición 8 veces.

**D4 (grupo dihedral)** — Las 8 simetrías del cuadrado: identidad, 3
rotaciones (90°, 180°, 270°) y 4 flips. Ataxx 7×7 es invariante bajo D4:
si rotás el board, el "buen movimiento" rota también. Aprovechamos eso
para multiplicar el dataset efectivo 8×.

**Curation** — Proceso de filtrar/limpiar/oversamplear data antes de
entrenar. `scripts/curate_training_data.py` toma replays raw y produce
NPZs con ejemplos balanceados por fase (apertura/medio/final).

---

## 7. Conceptos generales de ML

**Plateau / asíntota** — Cuando una métrica deja de mejorar a pesar de
seguir entrenando. Puede ser real (límite arquitectónico/de datos) o
aparente (necesita más iters / mejor config). Cuatro generaciones de este
proyecto plateauearon en h2h vs liga ~0.42-0.55.

**Overfitting** — Cuando el modelo memoriza el training set pero falla
en data nueva. Síntomas: train_loss baja pero val_loss sube.

**Distribution shift** — Cuando los datos de entrenamiento son distintos
de los de inferencia (o entre iters consecutivas). PM04 documentó
distribution shift cuando bootstrap + reset_iteration sin warmup metió
un modelo maduro al curriculum temprano.

**Regularization** — Técnicas para evitar overfitting: dropout, weight
decay, data augmentation, etc.

**Ablation study** — Experimento donde apagás UN componente para ver
cuánto importa. "v11 sin count head" sería una ablation del count head.

**A/B testing** — Comparar dos versiones que difieren en UNA cosa para
aislar el efecto de ese cambio. Lección clave: si cambiás 4 cosas a la
vez y mejora, no sabés cuál fue.

**Sample efficiency** — Cuánto aprende el modelo por unidad de data.
Pretrain + symmetry aug + human replay buffer buscan subir sample
efficiency.

**Knowledge distillation** — Entrenar un modelo "estudiante" para imitar
las predicciones de un modelo "maestro". El pretrain humano del v11 es
una forma de distillation (estudiante = red v11, maestro = jugadores
humanos).

**Inference** — Usar el modelo entrenado para hacer predicciones (jugar,
evaluar). Distinto de training. Inference es más rápido y no necesita
gradientes.

**Transfer learning** — Tomar un modelo entrenado en una tarea y
adaptarlo a otra. Bootstrap entre runs es una forma simple de transfer.

**Hyperparameter (hparam)** — Valor que se elige antes del training y no
se aprende (ej: lr, batch_size, num_layers). Opuesto a "parámetros" del
modelo (los pesos, sí se aprenden).

**Gradient explosion / vanishing** — Cuando los gradientes se hacen muy
grandes (explosión) o muy chicos (vanishing) durante backprop, rompen el
training. LayerNorm y residual connections en el transformer mitigan esto.

**LayerNorm** — Normaliza features por capa para estabilizar el training.
v11 lo usa antes de cada head y dentro del transformer.

**GELU** — Función de activación (no-lineal) común en transformers,
suaviza el ReLU. Usado en el feedforward del transformer y en las heads.

**Softmax / log-softmax** — Convierten logits (números reales) en
distribución de probabilidad (positivos, suman 1). log-softmax es la
versión log, más estable numéricamente. policy_head devuelve logits;
applicar log-softmax + cross-entropy es el flujo estándar.

**Temperature (softmax)** — Parámetro que controla qué tan "picuda" o
"plana" es la distribución. Alta T → más uniforme (exploración). Baja T
→ más concentrada en el argmax (explotación). MCTS de AlphaZero
modula la temperatura: alta en aperturas, baja en endgame.

**Tanh** — Activación que mapea a [-1, +1]. Usada en value_head porque el
target value está en ese rango.

**MoE (Mixture of Experts)** — *No usado en este repo*. Arquitectura donde
distintos "expertos" se activan según el input. Aclaro porque mencioné el
término en un mensaje pasado sin contexto.

---

## 8. Específicos de este repo

**Codename / generación** — Apodo que cada modelo recibe después de
entrenado, capturando lo que pasó en el run. bogo, reflejo, chispazo,
aprendiz, centinela, amnesia, liga, espejismo, paralelo, lastre, asedio.
Ver `checkpoints/registry.json`.

**Registry** — `checkpoints/registry.json`. Catálogo único de
generaciones con su config, eval, postmortem. `scripts/list_models.py`
los rankea.

**Postmortem (PM)** — Documento que se escribe al cerrar un run.
`src/model/docs/postmortem/0N/README.md`. Captura qué pasó, qué se
aprendió, qué cambia para la próxima. Llevamos PM01-PM09 hasta ahora.

**Lore** — La narrativa de cada codename en el registry. Cuenta la
historia humana del modelo.

**Round-robin (RR) interno** — `scripts/round_robin.py` enfrenta a todos
los codenames entre sí. Mide ranking absoluto.

**Internal eval vs external eval** — Internal eval = el que corre el
trainer cada `eval_every` iters. External eval = `scripts/eval_all_*`
corridos después del run, controlados manualmente.

**Absolute gate** — Ver "gate" arriba. Específico de este repo: aborta
si h2h vs liga < umbral por patience iters.

**Replay tagging** — Marcar cada ejemplo con su origen
(`is_human_source`). v11 usa el tag para aplicar value_mask y para
balancear el batch.

---

## 9. Términos de Kaggle / infraestructura

**Kaggle kernel** — Notebook que corre en GPU gratis (12h/sesión).
Este repo usa Kaggle para training porque T4×2 es suficiente.

**T4 / P100 / V100 / A100** — GPUs de NVIDIA. T4 = 16GB, GTX moderna.
T4×2 = dos T4 en paralelo. P100 = más vieja, 16GB también. v11 fue
diseñado para T4×2 con ddp_spawn.

**HF / HuggingFace Hub** — Servicio de hosting de modelos. Este repo
sube los checkpoints y metadata de cada run a `dieg0code/ataxx-zero`.

**WandB (Weights & Biases)** — Servicio para visualizar curvas de
training en tiempo real. v11 está logueando a `wandb.ai/dieg0code-ai/
ataxx-zero`.

**Pretrain NPZ** — Archivo NumPy comprimido con los ejemplos curados de
humanos. v11 lo genera en cada run desde `tournament_replays/`.

**uv** — Package manager Python que usa este repo, alternativa rápida a
pip. `uv sync --group train` instala dependencias del grupo train.

**Pyrefly** — Type checker estricto (alternativa a mypy). Corre como
gate antes de push.

**Ruff** — Linter rápido (alternativa a flake8). Corre como gate antes
de push.

---

## 10. Mini-glosario de jerga conversacional

**"El modelo canibaliza"** — Un componente del modelo (head auxiliar)
domina los gradientes y le roba capacidad a los componentes principales.

**"Saturar"** — Un output llega al límite de su rango (tanh saturado en
+1 o -1) y deja de aprender porque el gradiente se hace ~0.

**"Romper el techo"** — Que una nueva generación supere significativamente
a las anteriores. v11 intenta romper el techo de liga (composite 0.65).

**"Plateau-ear"** — Quedarse en una asíntota sin mejorar.

**"Heads"** — Las cabezas de salida del modelo (policy, value, count).

**"Iterar"** — Hacer una vuelta completa del loop AlphaZero.

**"Tirar un run"** — Lanzar un training completo en Kaggle.

**"Smoke test"** — Test rápido localmente (1-2 iters) para verificar que
todo arranca antes de gastar GPU horas.

**"Cazar un bug"** — Detectar y arreglar un bug.

**"Pisar (en HF)"** — Sobrescribir archivos en HF Hub (lo que pasó cuando
el notebook bumbo `RUN_NAME`).

---

## Referencias para profundizar

- [DeepMind AlphaZero paper](https://arxiv.org/abs/1712.01815) — el paper
  original (Silver et al., 2017).
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) — paper
  del Transformer original.
- [Lil'Log: A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
  — overview accesible de RL.
- [Spinning Up (OpenAI)](https://spinningup.openai.com/) — tutorial de RL
  con código.
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  — implementación de transformer desde cero, muy didáctico.

Si aparece un término en una conversación que no está acá, decime y lo
agrego.

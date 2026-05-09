# AtaxxZero - Post-mortem 04: por que `v7` empeoro despues de bootstrapping desde `v6`

> **Resumen:** `policy_spatial_v6` fue el primer checkpoint claramente bueno del
> proyecto. La corrida `policy_spatial_v7` partio desde esos pesos, pero termino
> peor. La explicacion mas precisa no es "catastrophic forgetting" puro, sino
> una regresion provocada por reiniciar el loop con pesos buenos pero sin el
> contexto de entrenamiento que hacia buenos a esos pesos: replay buffer limpio,
> iteracion en `0`, curriculum reiniciado y sin warmup.

---

## Sintoma

La secuencia observada fue esta:

- `policy_spatial_v6_iter_180` se convirtio en el primer checkpoint realmente
  fuerte del proyecto;
- luego se uso como bootstrap para `policy_spatial_v7`;
- pero `policy_spatial_v7_iter_140` termino por debajo de `v6` en benchmark
  directo y tambien con peor `best_eval_score`.

La evidencia concreta:

- `v6`: `best_eval_score = 0.8055555555555555`
- `v7`: `best_eval_score = 0.6875`
- benchmark directo `v7 vs v6`: score `0.4125` para `v7`
  - `16` wins
  - `23` losses
  - `1` draw

Eso ya descarta la idea comoda de:

- "como partio desde un modelo mejor, por definicion tenia que mejorar"

No. El bootstrap le dio a `v7` una mejor condicion inicial de pesos, pero no le
garantizo preservar la calidad del regimen de entrenamiento que hizo bueno a
`v6`.

---

## Hipotesis inicial

La intuicion natural fue:

- "capaz que esto sea catastrophic forgetting"

Esa hipotesis no esta totalmente mal, pero es incompleta.

Si uno mira solo el resultado final, si hubo una forma de olvido o regresion:
un modelo que partio bueno termino jugando peor que su padre.

Pero la causa operativa no parece ser una patologia abstracta del optimizador ni
un colapso misterioso de la red. El problema esta mucho mas cerca del pipeline.

---

## Diagnostico correcto

`v7` no fracaso porque "bootstrapping sea malo".

`v7` fracaso porque el bootstrap se uso junto con un reset demasiado agresivo
del loop:

1. se cargaron pesos buenos;
2. se limpio el replay buffer;
3. se reinicio la iteracion a `0`;
4. se reinicio el curriculum temprano;
5. y no hubo warmup para re-anclar el modelo.

En otras palabras:

> Se preservo el checkpoint, pero no se preservo el contexto estadistico que lo
> sostenia.

Eso es mas parecido a una **regresion por cambio de distribucion despues del
bootstrap** que a catastrophic forgetting clasico.

---

## Causas raiz

### 1. `hf_reset_iteration` limpio justo la memoria que mas importaba

En `src/training/checkpointing.py`, cuando `hf_reset_iteration` es `true`, el
sistema hace esto:

- carga los pesos desde Hugging Face;
- no carga el replay buffer;
- limpia explicitamente el buffer;
- y devuelve iteracion `0`.

La logica es esta:

> "keep learned weights but rebuild replay from scratch"

Eso puede servir si se quiere lanzar una corrida nueva con curriculum otra vez.
Pero tambien rompe una propiedad importantisima:

- los pesos arrancan maduros;
- el buffer arranca vacio;
- el loop deja de entrenar sobre la distribucion que hizo buenos a esos pesos.

Si el padre era bueno en parte porque habia acumulado una mezcla amplia de
posiciones, teachers y estados duros, borrar eso deja al hijo sin memoria
estructural del regimen que lo formo.

---

### 2. `v7` se reinicio sin warmup

La diferencia entre `v6` y `v7` aqui es critica:

- `v6`
  - `warmup_games = 320`
  - `warmup_epochs = 4`
- `v7`
  - `warmup_games = 0`
  - `warmup_epochs = 0`

O sea:

- `v6` tuvo una fase de anclaje supervisado/imitation contra heuristicas fuertes;
- `v7` no.

Eso importa mucho despues de limpiar el buffer.

Sin warmup, el primer tramo de `v7` dependio casi por completo de self-play y
curriculum temprano para reconstruir datos desde cero. Pero ese nuevo buffer ya
no estaba siendo filtrado por el mismo andamiaje que ayudo a estabilizar `v6`.

---

### 3. El curriculum volvio a una fase temprana que no coincide con un modelo maduro

En `src/training/curriculum.py`, las iteraciones tempranas usan una mezcla muy
dominada por heuristicas:

- iteraciones `<= 12`
  - `self = 0.10`
  - `heuristic = 0.88`
  - `random = 0.02`

Eso esta bien para modelos debiles o recien nacidos.

Pero cuando un modelo ya venia maduro desde `v6`, reiniciarlo a iteracion `0`
lo fuerza a volver a una fase del curriculum pensada para otro tipo de politica.

Eso no necesariamente "olvida" por si solo, pero si cambia el entorno de
aprendizaje de forma brusca:

- el modelo ya no esta aprendiendo como un checkpoint tardio;
- esta siendo tratado como si recien estuviera despegando.

---

### 4. El sampler recent-heavy pudo amplificar la deriva

En `src/training/loop_runtime.py`, el train set se re-samplea con
`sample_recent_mix(...)`.

Y en configuracion:

- `train_recent_fraction = 0.7`
- `train_recent_window_fraction = 0.4`

Eso significa que el training da bastante peso a experiencia reciente.

Ese sesgo es razonable cuando la distribucion reciente representa una mejora
real del modelo.

Pero despues de:

- limpiar buffer;
- reiniciar curriculum;
- y omitir warmup;

la experiencia reciente deja de ser "lo mejor del sistema" y pasa a ser "lo que
alcance a reconstruir despues del reset".

Entonces el sesgo a lo reciente puede acelerar justo la deriva que querias
evitar.

---

### 5. El benchmark contra el padre confirma regresion real, no solo ruido de eval

Podria haberse dicho:

- "quizas `best_eval_score` bajo un poco por varianza"

Pero el benchmark directo contra `v6` muestra una derrota clara.

`v7` no solo no supero al padre: quedo por debajo en juego real.

Eso importa porque evita una lectura demasiado optimista del tipo:

- "el hijo era distinto, no necesariamente peor"

No. En la comparacion directa, fue peor.

---

## Por que esto no es catastrophic forgetting puro

Si uno usa el termino de forma amplia, se puede decir que si:

- el modelo olvido parte de lo que sabia hacer bien

Pero no parece un caso canonico de continual learning donde tareas nuevas
machacan representaciones viejas por si solas.

La cadena causal aqui parece mucho mas concreta:

- bootstrap desde pesos fuertes;
- reset del loop;
- replay vacio;
- curriculum temprano;
- cero warmup;
- re-entrenamiento sobre una distribucion nueva y mas estrecha.

Entonces el nombre mas util es:

- regresion post-bootstrap
- o forgetting inducido por reset de distribucion

Eso apunta mucho mejor al fix real.

---

## Leccion correcta

Un checkpoint fuerte no se define solo por sus pesos.

Tambien depende de:

- la distribucion de posiciones que lo formo;
- el tipo de teachers que moldearon la policy;
- la mezcla de opponents del curriculum;
- el replay acumulado;
- y el regimen de validacion/regresion que lo sostuvo.

Por eso, bootstrapping no debe leerse como:

- "copio el `.pt` y sigo"

Sino como:

- "decido que heredo del padre y que cosa quiero reiniciar sin destruir lo que
  ya estaba bien"

---

## Reglas operativas para no repetirlo

### 1. No combinar por defecto:

- `hf_bootstrap_run_id != ""`
- `hf_reset_iteration = true`
- `warmup_games = 0`
- `warmup_epochs = 0`

Esa combinacion deja al modelo sin red de seguridad.

---

### 2. Si el objetivo es continuar una corrida buena, heredar tambien el contexto

Si la idea real era continuar `v6`, entonces lo mas sano no era crear una
corrida nueva vaciando memoria, sino reanudar preservando:

- iteracion;
- replay buffer;
- y regimen de opponent mix ya maduro.

---

### 3. Si el objetivo es abrir una corrida nueva, usar warmup de re-anclaje

Si por razones de orden o experimentacion igual quieres un `run_id` nuevo,
entonces hace falta un bootstrap menos violento:

- pesos del padre;
- warmup no cero;
- y, si es posible, mezcla de ejemplos/teachers heredados del padre al inicio.

No para "volver al pasado", sino para evitar que el hijo derive antes de
reconstruir contexto suficiente.

---

### 4. El padre debe ser benchmark obligatorio, no solo referencia emocional

Cada corrida hija de un checkpoint fuerte deberia medirse periodicamente contra
su padre.

Si no lo supera o empieza a caer claramente debajo:

- no hay mejora acumulativa;
- hay regresion;
- y el loop necesita correccion, no solo mas iteraciones.

---

## Checklist practico para el siguiente intento

- decidir si la siguiente corrida es **continuacion** o **branch**
- si es continuacion, no limpiar replay ni iteracion
- si es branch, no dejar warmup en `0`
- bajar el riesgo de deriva temprana
- mantener benchmark recurrente contra `v6`
- desconfiar de cualquier corrida que "se vea viva" pero no gane al padre

---

## Conclusiones

`policy_spatial_v6` no fue solo un buen archivo de pesos. Fue el producto de un
regimen de entrenamiento ya suficientemente sano.

`policy_spatial_v7` mostro que:

- copiar pesos buenos no basta;
- un reset mal calibrado puede degradar un modelo fuerte;
- y el bootstrap tiene que heredar mas que el checkpoint si se quiere conservar
  fuerza real.

La idea correcta para adelante no es:

- "bootstrapping funciona o no funciona"

Sino esta:

> Un bootstrap bueno tiene que preservar o reconstruir cuidadosamente el
> contexto estadistico que hacia bueno al modelo padre.


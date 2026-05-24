# Experimentos — pre-registration discipline

Esta carpeta existe para forzar disciplina experimental: **no se
escribe postmortem sin PROPOSAL.md previo**.

## Estructura por experimento

Cada experimento (v13, v14, etc.) tiene su propia subcarpeta con:

| Archivo | Cuándo | Qué contiene |
|---|---|---|
| `PROPOSAL.md` | **ANTES del run, commiteado antes del push a Kaggle** | Hipótesis falsificable, predicciones cuantitativas con números, criterios de aborto/falsación, variables cambiadas vs controladas, plan de análisis |
| `JOURNAL.md` | **DURANTE el run, append-only** | Una entrada por evento relevante (launch, eval, anomalía, decisión). Sin reescribir el pasado |
| `RESULTS.md` | **DESPUÉS del run** | Tabla predicho vs observado, marcar cada predicción cumple/no cumple/inconcluso, sin retocar el PROPOSAL |

Si el experimento falla, el postmortem va en
`src/model/docs/postmortem/NN/` con referencia explícita a qué
predicción del PROPOSAL se quebró primero y qué nos dice sobre
la hipótesis. **El postmortem analiza el experimento, no lo
reemplaza.**

## Por qué

PM01-PM10 documentaron 10 generaciones de runs sin pre-registration.
Toda la "lección aprendida" se construyó post-hoc, vulnerable a
sesgo de confirmación. El cambio cultural: predicciones explícitas
antes del compute, comparación rigurosa después.

## Plantilla mínima de PROPOSAL.md

```markdown
## Hipótesis principal
Una frase falsificable. Si no se puede falsificar, no es hipótesis.

## Predicciones cuantitativas
- iter X: métrica Y < Z
- iter X: per-level vs gambit ≥ Z

## Criterios de falsación
- Cualquiera de estos mata la hipótesis: ...

## Variables cambiadas vs controladas
| Palanca | baseline | experimento |
| ... | ... | ... |

## Plan B
Si la hipótesis se confirma → próximo experimento (ablation).
Si se falsea → qué otra hipótesis explorar.
```

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

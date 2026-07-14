# Plan 3 detail_v2 — smoke CPU

Necesidad de GPU: **No requerida para este smoke**; obligatoria para el entrenamiento completo.

## Ejecución real

- Dataset: manifests smoke de `coinrun_teams_v1`
- Train: 8 episodios, 179 frames
- Validación: 8 episodios, 361 frames
- Pasos: 100
- Dispositivo: CPU, sin AMP
- Arquitectura: `detail_v2`, 3,204,035 parámetros, latente 128
- Tests del repositorio: 57/57 correctos

## Resultado diagnóstico

| Métrica | Valor |
|---|---:|
| PSNR | 12.4213 dB |
| SSIM | 0.42949 |
| Frames con evidencia amarilla | 43 |
| Recall amarillo por píxel | 24.21% |
| Precisión amarilla por píxel | 1.01% |
| Píxeles amarillos objetivo | 190 |
| Píxeles amarillos predichos | 4,560 |

El resultado es **NO-GO como modelo**, pero **GO como smoke de implementación**. Las nuevas
compuertas detectan los falsos positivos y la auditoría independiente reproduce las métricas.
La calidad solo puede decidirse después del entrenamiento completo en GPU.

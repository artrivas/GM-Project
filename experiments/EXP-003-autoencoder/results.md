# Resultados de Plan 3

## Resultado cuantitativo

| Métrica de validación | Autoencoder | Baseline promedio |
|---|---:|---:|
| MAE global | 0.015913 | 0.165804 |
| MSE global | 0.001757 | 0.046213 |
| PSNR | 27.5526 dB | 13.3524 dB |
| SSIM | 0.904375 | 0.450565 |

El autoencoder supera las puertas globales: +14.20 dB de PSNR y +0.4538 de SSIM frente al baseline.

## Moneda

Validación contiene un único frame positivo (`pilot_sticky-sticky-s0-001`, frame 299), con cinco píxeles de moneda y bbox `[59,39,64,46]`.

| Métrica localizada | Autoencoder | Baseline promedio |
|---|---:|---:|
| MAE en ROI 5×7 | 0.153186 | 0.377554 |
| MAE en los cinco píxeles de moneda | 0.433566 | 0.367756 |
| Retención de píxeles amarillos | 0% | No aplica |

La ROI contextual mejora 59.43%, pero la moneda propiamente dicha es peor que el baseline y pierde su color distintivo.

## Ajuste experimental

Se probó el ajuste previsto `coin_roi_weight=10` durante cinco epochs adicionales. Mejoró PSNR global a 28.673 dB y el MAE contextual a 0.134153, pero empeoró el MAE exacto de moneda a 0.582652 y mantuvo recall amarillo en 0%. El ajuste fue rechazado; sus métricas, checkpoint, configuración e imagen se conservan con sufijo `coin_weighted_failed`.

El entrenamiento inicialmente sufrió un solapamiento breve de procesos causado por el timeout del wrapper externo. Los checkpoints se escribían de forma atómica. Se detuvieron ambos procesos, se reanudó desde un checkpoint íntegro y se reconstruyó `training_clean.jsonl` con la cadena seleccionada. `training.log` se conserva como evidencia del incidente, pero no es la fuente de métricas finales.

## Pruebas

- Tests específicos: 11/11 correctos.
- Suite completa: 43/43 correctos.
- Checkpoint seleccionado: epoch 11, step 936, SHA-256 `598a1fb086db5f881454029b7de2f48bb8106159720b77ff6274591c815131bc`.

## Decisión

**NO-GO para Plan 4.** Personaje y plataformas son reconocibles y las métricas globales son fuertes, pero la moneda no supera el criterio localizado y solo existe un ejemplo positivo en validación. Tampoco se ha demostrado de forma independiente la preservación de enemigos.

No debe seguirse ajustando contra ese único frame. La siguiente acción es recolectar/repartir más trayectorias con moneda visible y anotar moneda/enemigo en al menos diez frames de validación; después se puede evaluar latent 128 o reducir un downsampling sin contaminar el único ejemplo actual.

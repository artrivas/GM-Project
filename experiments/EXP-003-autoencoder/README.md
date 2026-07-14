# EXP-003 — Autoencoder de CoinRun

## Objetivo

Entrenar en CPU un autoencoder convolucional continuo de 805,251 parámetros sobre los manifests de Plan 2 y medir reconstrucción global y localizada en la moneda.

## Modelo seleccionado

- Latente continuo: 64.
- Canales base: 32.
- Pérdida: `0.8 × L1 + 0.2 × DSSIM`.
- Train: 2,485 frames; val: 600 frames.
- 12 epochs, 936 actualizaciones, batch 32.
- Dispositivo CPU; sin CUDA ni AMP.

`checkpoints/best.pt` es el checkpoint seleccionado. El experimento ponderado por ROI se conserva con sufijo `coin_weighted_failed` y no es el modelo principal.

## Evidencia

- `metrics.json`: métricas finales del checkpoint seleccionado.
- `figures/reconstructions.png`: originales, reconstrucciones, errores y baseline.
- `test_output.log`: tests específicos de Plan 3.
- `full_test_output.log`: suite completa del repositorio.
- `training_clean.jsonl`: cadena seleccionada de epochs 0–11.
- `checkpoint.sha256`: integridad del checkpoint.

La detección de moneda se realiza únicamente sobre el ground truth mediante sus píxeles amarillos. Validación contiene un solo frame positivo, por lo que no puede sostener una conclusión robusta de preservación.

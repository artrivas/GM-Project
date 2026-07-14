# Smoke test del dataset externo

Necesidad de GPU: **No requerida** para este smoke test. La ejecución usó PyTorch 2.2.2 CPU;
`torch.cuda.is_available()` devolvió `False` y `nvidia-smi` no está instalado en el entorno.

## Alcance ejecutado

- train: 8 episodios, 179 frames
- val: 8 episodios, 361 frames
- entrenamiento: CPU, 20 pasos, 4 épocas parciales
- tiempo: 6.69 s
- checkpoint: `checkpoints/best.pt`
- mejor pérdida de validación L1+SSIM: 0.16125

El smoke confirma lectura NPZ, normalización, forward/backward, validación, guardado de
checkpoint y evaluación. No es un entrenamiento de calidad: sus criterios finales de
reconstrucción no se usan como Go/No-Go del autoencoder completo.

## Métricas diagnósticas

- MAE global del modelo: 0.13326
- PSNR del modelo: 14.16 dB
- frames con región amarilla detectada: 43
- MAE localizada en ROI: 0.33240
- recall de color de moneda: 0.0

El recall nulo es esperable tras solo 20 pasos y confirma que aún hace falta el entrenamiento
completo en GPU antes de emitir el veredicto del Plan 3.

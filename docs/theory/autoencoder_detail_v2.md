# Corrección Plan 3 — autoencoder detail_v2

## Diagnóstico comprobado

El autoencoder `compact_v1` obtuvo buenas métricas globales sobre el dataset externo completo
(PSNR 22.97 dB y SSIM 0.815), pero recall de moneda 0.0 en 5,208 frames. La auditoría visual
confirmó que la moneda desaparecía en lugar de sufrir únicamente un cambio de color.

El intento legado `coin_roi_weight=10` no era evidencia suficiente para descartar una pérdida
localizada: se ejecutó sobre el dataset antiguo, con un único frame positivo, y la región seguía
formando parte de un promedio global de píxeles.

## Cambios

- **[INGENIERÍA] Menos downsampling:** `detail_v2` reduce 64×64 a 8×8 mediante tres etapas, en
  vez de llegar a 4×4 con cuatro etapas. Esto evita que un objeto de 2–6 píxeles atraviese una
  cuarta reducción espacial antes del bottleneck.
- **[INGENIERÍA] Latente 128:** duplica la capacidad respecto del baseline de 64, dentro del
  rango previsto para una GPU de 12–16 GB. Se mantiene un vector, compatible con el futuro RSSM.
- **[INGENIERÍA] Sin conexiones skip:** el decoder recibe exclusivamente el vector latente. Un
  U-Net podría mejorar reconstrucción copiando features del encoder, pero esas features no
  existirían durante un rollout imaginado del RSSM.
- **[INGENIERÍA] Bloques residuales:** refinan features sin reducir nuevamente la resolución.
- **[INGENIERÍA] Pérdidas localizadas normalizadas:** MAE exacto de píxel de moneda y MAE de una
  ROI 7×7 se calculan como términos separados; ya no desaparecen dentro de los 12,288 valores de
  un frame RGB.
- **[INGENIERÍA] Máscara amarilla diferenciable:** aproxima los mismos umbrales usados en
  evaluación. Incluye hard-negative mining sobre el 1% de píxeles negativos más difíciles para
  impedir que el modelo obtenga recall pintando amarillo difuso.
- **[INGENIERÍA] Precision gate:** la evaluación exige simultáneamente recall y precisión de
  píxeles amarillos ≥90%. El criterio anterior solo medía recall y podía aceptar falsos positivos.

## Evidencia CPU

- Modelo: 3,204,035 parámetros, latente `[N,128]`, salida `[N,3,64,64]`.
- 17 tests específicos pasan, incluido un overfit sintético que conserva objetos amarillos de
  2×2 con recall ≥75%.
- Smoke real: 8 episodios de train, 8 de validación y 100 pasos en CPU.
- El hard-negative mining redujo los píxeles amarillos predichos de 111,863 a 9,195 a igualdad
  de 20 pasos frente a la primera variante localizada.
- A 100 pasos el smoke todavía no es un modelo válido: PSNR 12.42 dB, recall 24.2% y precisión
  1.01%. Sirve únicamente para comprobar integración y dirección de los falsos positivos.

## Criterio pendiente de GPU

**[REQUIERE VALIDACIÓN EXPERIMENTAL]** Solo el entrenamiento completo puede determinar GO:

- superar el baseline promedio por ≥3 dB de PSNR y ≥0.10 de SSIM;
- al menos 10 frames con evidencia de moneda;
- mejorar MAE de ROI de moneda ≥25%;
- recall amarillo ≥90%;
- precisión amarilla ≥90%;
- revisión independiente reproducible de al menos 10 frames.

El RSSM permanece bloqueado hasta que todos los criterios pasen y la inspección visual confirme
que personaje, moneda y otros objetos pequeños siguen presentes.

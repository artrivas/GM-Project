# Auditoría independiente del autoencoder

Necesidad de GPU: **Opcional**. El recálculo se ejecutó en CPU porque los splits revisados son
pequeños. PyTorch 2.2.2 es CPU-only en este entorno, `torch.cuda.is_available()` es `False` y
`nvidia-smi` no está instalado.

## Veredicto

**NO-GO para iniciar el Plan 4.** No existe todavía un checkpoint completo entrenado sobre el
dataset externo activo y los dos checkpoints disponibles eliminan sistemáticamente la moneda.
El checkpoint de 20 pasos es solo una prueba de integración, no un modelo candidato a dinámica.

## Hallazgo previo de procedencia

La ruta solicitada `experiments/EXP-003-autoencoder/checkpoints/best.pt` existe, pero corresponde
al dataset HDF5 legado: fue entrenada durante 936 pasos con 2,485 frames de train y 600 de
validación. Su manifest de validación contiene únicamente un frame detectado con moneda, por lo
que no permite satisfacer la revisión obligatoria de 10 ejemplos.

El único checkpoint que consume el dataset externo activo es
`external-dataset-smoke/checkpoints/best.pt`: 20 pasos, 179 frames de train y 361 de validación.
Este split sí contiene 43 frames con moneda y se usó únicamente como evidencia auxiliar para
completar la inspección independiente de 10 casos.

## Método independiente

`src.evaluation.independent_reconstruction_review` no importa el cargador ni las métricas de la
implementación. Lee directamente los HDF5/NPZ listados en cada manifest, reconstruye con el
checkpoint, vuelve a calcular PSNR y SSIM desde las estadísticas locales y detecta la región
amarilla de la moneda desde el frame original. La selección visual es determinista y usa
muestreo por distancia de apariencia; no reutiliza las imágenes elegidas por la implementación.

Además de los 10 casos con moneda, se generó otro montaje de 10 frames diversos para observar al
personaje en distintas posiciones y objetos pequeños del escenario.

## Resultados recalculados

| Evidencia | Checkpoint legado completo | Smoke dataset activo |
|---|---:|---:|
| Pasos de entrenamiento | 936 | 20 |
| Frames de validación recontados | 600 | 361 |
| Frames con moneda | 1 | 43 |
| Casos independientes solicitados/seleccionados | 10/1 | 10/10 |
| PSNR global | 27.55263 dB | 14.15773 dB |
| SSIM global | 0.904375 | 0.453606 |
| MAE de ROI de moneda | 0.153186 | 0.332405 |
| MAE en píxeles de moneda | 0.433566 | 0.329982 |
| Recall del color de moneda | **0.0** | **0.0** |
| Coincidencia con `metrics.json` | exacta (≤1e-6) | exacta (≤1e-6) |

El recálculo reproduce exactamente las cuatro métricas contrastadas de cada `metrics.json`:
PSNR, SSIM, MAE localizada y recall de moneda. Por tanto, el hallazgo no es una discrepancia del
reporte anterior; es un fallo real de preservación localizado que las métricas globales ocultan.

## Inspección visual

En los 10 casos con moneda del dataset activo, provenientes de tres episodios y tres seeds, la
moneda desaparece en todas las reconstrucciones. Los recortes originales contienen entre 2 y 6
píxeles amarillos y ninguno conserva píxeles que superen el umbral de color en la salida.

El montaje contextual muestra varias posiciones y apariencias del personaje. El personaje, las
monedas, cajas pequeñas y varios objetos/obstáculos de plataforma se convierten en fondos
borrosos. El dataset no incluye bounding boxes ni etiquetas de clase para enemigos, por lo que
no es posible aislar cuantitativamente enemigos de otros sprites pequeños sin inventar etiquetas.
Esta limitación no cambia el veredicto: la pérdida sistemática de la moneda ya cumple por sí sola
el criterio de NO-GO.

Evidencia visual:

- `independent_review/external_smoke_aux/coin_samples.png`
- `independent_review/external_smoke_aux/context_samples.png`
- `independent_review/legacy_requested/coin_samples.png`

## Respuestas obligatorias

- **¿Se preservan los objetos pequeños?** No. La moneda se pierde sistemáticamente y otros
  sprites pequeños también desaparecen visualmente.
- **¿Las métricas originales son reproducibles?** Sí. Todas las métricas contrastadas coinciden
  exactamente dentro de una tolerancia de 1e-6.
- **¿Puede construirse dinámica temporal encima?** No con los checkpoints actuales. Primero debe
  completarse el Plan 3 sobre los splits externos completos y repetirse esta auditoría.

## Comandos ejecutados

```bash
source .venv/bin/activate

python -m src.evaluation.independent_reconstruction_review \
  --checkpoint experiments/EXP-003-autoencoder/checkpoints/best.pt \
  --eval-dir data/splits/val --n-samples 10 \
  --out experiments/EXP-003-autoencoder/independent_review/legacy_requested/

python -m src.evaluation.independent_reconstruction_review \
  --checkpoint experiments/EXP-003-autoencoder/external-dataset-smoke/checkpoints/best.pt \
  --eval-dir data/datasets/coinrun_teams_v1/smoke/val --n-samples 10 \
  --out experiments/EXP-003-autoencoder/independent_review/external_smoke_aux/
```

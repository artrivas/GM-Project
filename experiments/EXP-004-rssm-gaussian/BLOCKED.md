# Plan 4 — Preflight bloqueado

Necesidad de GPU: **Obligatoria para el entrenamiento completo**. El dataset activo contiene
163,746 frames y el RSSM requiere entrenamiento secuencial. No se inició entrenamiento porque
fallaron prerrequisitos anteriores a cualquier uso de GPU.

## Veredicto

**NO-GO. No construir ni entrenar el RSSM todavía.**

## Evidencia

### 1. Autoencoder no aprobado

La auditoría independiente del Plan 3 terminó explícitamente en `NO-GO para iniciar el Plan 4`.

- El checkpoint completo de `experiments/EXP-003-autoencoder/checkpoints/best.pt` fue entrenado
  sobre el dataset HDF5 legado, no sobre el dataset externo activo.
- El único checkpoint que consume el dataset activo es un smoke de 20 pasos sobre 179 frames de
  entrenamiento.
- El recálculo independiente obtuvo recall de color de moneda 0.0 tanto en el checkpoint legado
  como en el smoke activo.
- Las métricas independientes coincidieron exactamente con los reportes originales, por lo que
  no se trata de un error de medición.

Fuente: `experiments/EXP-003-autoencoder/independent_review.md`.

### 2. Dataset no aprobado para dinámica condicionada por acción

`data/datasets/coinrun_teams_v1/summary.json` declara el dataset activo para `autoencoder` y
`visual_reconstruction`, pero lo mantiene fuera de `rssm` y `action_conditioned_dynamics`.
La alineación temporal entre `observations` y `actions` todavía requiere una auditoría
independiente.

Entrenar antes de comprobar esa alineación podría enseñar transiciones con la acción desplazada
un paso, invalidando teacher forcing y la comparación contra el baseline naive.

### 3. Hardware local

- PyTorch: 2.2.2+cpu
- `torch.cuda.is_available()`: `False`
- `nvidia-smi`: no instalado

No se intentó ni se simuló entrenamiento GPU.

## Acciones deliberadamente no ejecutadas

- No se creó `src/models/rssm.py`.
- No se creó `src/models/heads.py`.
- No se creó `src/training/train_rssm_step.py`.
- No se fijó un valor de free bits/KL balance.
- No se creó un checkpoint RSSM.
- No se reportaron métricas PSNR/SSIM inexistentes.

Esto cumple la Sección E del Plan 4 — Implementación: detenerse si el autoencoder no obtuvo GO o
GO CONDICIONAL.

## Condiciones concretas para desbloquear Plan 4

1. Entrenar el autoencoder sobre los splits completos de
   `data/datasets/coinrun_teams_v1/splits/` en un entorno GPU.
2. Repetir la auditoría independiente con al menos 10 frames con moneda y obtener GO o GO
   CONDICIONAL.
3. Auditar y documentar la correspondencia exacta entre `observation[t]`, `action[t]`, reward y
   siguiente observación en los NPZ externos.
4. Cambiar explícitamente la readiness del dataset a `rssm: true` solamente si la auditoría de
   transiciones pasa.
5. Reanudar Plan 4 y entonces verificar en la fuente el valor inicial de free bits antes de
   escribir `configs/rssm_gaussian.yaml`.

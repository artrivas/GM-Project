# EXP-002 — Auditoría y split del dataset

## Objetivo

Auditar todos los episodios crudos de Plan 1, detectar secuencias rotas y contenido duplicado, y producir manifests train/val/test sin fuga por episodio ni por trayectoria idéntica.

## Entrada

El patrón normalizado es `data/raw/*/episode_*.h5`. Los artefactos `smoke_*` y las sesiones humanas inactivas se excluyen explícitamente; no se borran.

## Salida

- `quality_report.json`: reporte completo y reproducible.
- `episodes.jsonl` y `rejected_episodes.jsonl`: inventario aceptado y rechazado.
- `figures/`: cobertura de acciones, longitudes y eventos raros.
- `samples/random_episode_sample.png` y `visual_review.json`: evidencia de inspección manual.
- `split_summary.json`: copia de las comprobaciones y conteos de `data/splits/summary.json`.
- `test_output.log`: salida íntegra de las pruebas específicas de Plan 2.

Los manifests producidos son diagnósticos: `data/splits/DO_NOT_USE.json` impide tratarlos como entrada de entrenamiento mientras las puertas estrictas estén incumplidas.

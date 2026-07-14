# Dataset activo CoinRun teams v1

## Estado

**Activo para autoencoder y reconstrucción visual mediante manifests versionados.**

Los archivos NPZ originales permanecen sin modificar en
`data/incoming/coinrun_teams_v1/original/`. El flujo activo no descubre esa carpeta de forma
implícita: consume únicamente los manifests auditados de
`data/datasets/coinrun_teams_v1/`. De este modo se usa la entrega recibida sin mezclarla con el
dataset HDF5 legado de `data/raw/` ni sobrescribir los splits anteriores de `data/splits/`.

## Inventario real

| Métrica | Valor |
|---|---:|
| Equipos | 6 |
| Sesiones | 21 |
| Episodios NPZ válidos | 1,899 |
| Frames | 163,746 |
| Seeds únicos | 1,899 |
| Victorias | 1,421 |
| No ganadores terminales | 468 |
| Truncados | 10 |
| Referencias a archivos ausentes | 5 |

La sesión `Team5/session_20260524_125456` declara 80 episodios, pero contiene 75. Los cinco
archivos ausentes se documentan en el inventario y no entran en ningún split.

## Splits activos sin fuga

La división retiene equipos completos para evitar que episodios correlacionados del mismo
contribuidor aparezcan tanto en entrenamiento como en evaluación.

| Split | Equipos | Episodios | Frames | Victorias | No victoria | Truncados |
|---|---|---:|---:|---:|---:|---:|
| train | Team1–Team4 | 1,277 | 107,353 | 983 | 288 | 6 |
| val | Team5 | 281 | 30,679 | 186 | 94 | 1 |
| test | Team6 | 341 | 25,714 | 252 | 86 | 3 |

Las comprobaciones automáticas confirman conservación de los 1,899 episodios y 163,746 frames,
sin solapamiento de equipos, seeds ni IDs de episodio entre splits.

## Muestra para smoke tests

`data/datasets/coinrun_teams_v1/smoke/` contiene 8 episodios por split. La muestra conserva
casos `easy`/`hard`, victorias, no victorias y al menos un truncado, pero solo sirve para validar
integración y tiempos; nunca reemplaza los manifests completos para métricas finales.

## Límites de uso

- `source_policy` no está presente y se conserva honestamente como `unknown_external`.
- Los manifests están aprobados para reconstrucción visual y entrenamiento del autoencoder.
- Antes de entrenar el RSSM se debe auditar de forma independiente la alineación temporal entre
  `observations` y `actions`; por ahora el dataset no está aprobado para dinámica condicionada
  por acciones.
- El inventario reproducible se genera con `src.data_prep.inventory_external_coinrun` y la
  promoción con `src.data_prep.promote_external_coinrun`.

## Comandos reproducibles

Smoke test corto en CPU:

```bash
source .venv/bin/activate
python -m src.training.train_autoencoder \
  --config configs/autoencoder.yaml \
  --train-dir data/datasets/coinrun_teams_v1/smoke/train \
  --val-dir data/datasets/coinrun_teams_v1/smoke/val \
  --device cpu --max-steps 20 \
  --out experiments/EXP-003-autoencoder/external-dataset-smoke/
```

Entrenamiento completo, después de confirmar una GPU y ajustar el batch a la VRAM medida:

```bash
source .venv/bin/activate
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"
python -m src.training.train_autoencoder \
  --config configs/autoencoder.yaml \
  --train-dir data/datasets/coinrun_teams_v1/splits/train \
  --val-dir data/datasets/coinrun_teams_v1/splits/val \
  --device cuda \
  --out experiments/EXP-003-autoencoder/external-dataset-full/
```

Reanudación exacta desde el último checkpoint:

```bash
python -m src.training.train_autoencoder \
  --config configs/autoencoder.yaml \
  --train-dir data/datasets/coinrun_teams_v1/splits/train \
  --val-dir data/datasets/coinrun_teams_v1/splits/val \
  --device cuda \
  --resume experiments/EXP-003-autoencoder/external-dataset-full/checkpoints/last.pt \
  --out experiments/EXP-003-autoencoder/external-dataset-full/
```

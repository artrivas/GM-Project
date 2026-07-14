#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/data/raw/exp001_reproduction}"
cd "$ROOT"
source .venv/bin/activate

python -m src.data_collection.run_collector --config configs/data_collection.yaml \
  --policy random --seed 0 --episodes 5 --out "$OUT_ROOT/pilot_random"
python -m src.data_collection.run_collector --config configs/data_collection.yaml \
  --policy sticky --seed 0 --episodes 5 --out "$OUT_ROOT/pilot_sticky"
python -m src.data_collection.run_collector --config configs/data_collection.yaml \
  --policy scripted --seed 0 --episodes 5 --distribution-mode hard \
  --out "$OUT_ROOT/pilot_scripted_hard"
python -m src.data_collection.run_collector --config configs/data_collection.yaml \
  --policy scripted --seed 0 --episodes 5 --distribution-mode easy \
  --out "$OUT_ROOT/pilot_scripted_easy"

python -m src.data_collection.quick_report \
  --path "$OUT_ROOT/pilot_*/episode_*.h5"
pytest tests/test_data_collection.py -v

# Ejecutar por separado con una persona frente al teclado:
# scripts/run_human_capture.sh --config configs/data_collection.yaml \
#   --seed 0 --out data/raw/pilot_human --max-minutes 2

#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python -m src.data.frame_dataset \
  --train-manifest data/splits/train/manifest.jsonl \
  --val-manifest data/splits/val/manifest.jsonl \
  --audit-only

python -m src.models.autoencoder \
  --config configs/autoencoder.yaml \
  --device cpu \
  --smoke-test

python -m src.training.train_autoencoder \
  --config configs/autoencoder.yaml \
  --train-dir data/splits/train \
  --val-dir data/splits/val \
  --out experiments/EXP-003-autoencoder/

python -m src.evaluation.reconstruction_metrics \
  --checkpoint experiments/EXP-003-autoencoder/checkpoints/best.pt \
  --eval-dir data/splits/val \
  --out experiments/EXP-003-autoencoder/metrics.json

pytest tests/test_autoencoder.py -v \
  | tee experiments/EXP-003-autoencoder/test_output.log

pip freeze > environment.txt
git rev-parse HEAD > git_commit.txt
nvidia-smi > experiments/EXP-003-autoencoder/hardware.txt 2>/dev/null \
  || echo "sin GPU" > experiments/EXP-003-autoencoder/hardware.txt

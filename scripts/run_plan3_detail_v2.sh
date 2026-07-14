#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:-experiments/EXP-003-autoencoder/detail-v2-full}"
CONFIG="configs/autoencoder_detail_v2.yaml"
TRAIN="data/datasets/coinrun_teams_v1/splits/train"
VAL="data/datasets/coinrun_teams_v1/splits/val"
SMOKE_TRAIN="data/datasets/coinrun_teams_v1/smoke/train"
SMOKE_VAL="data/datasets/coinrun_teams_v1/smoke/val"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("DETENER: el entrenamiento completo requiere una GPU CUDA")
print("gpu:", torch.cuda.get_device_name(0))
PY

mkdir -p "$OUT"
nvidia-smi > "$OUT/hardware.txt"
python -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))' \
  >> "$OUT/hardware.txt"

# CPU first: five optimizer steps over the audited smoke manifests.
python -m src.training.train_autoencoder \
  --config "$CONFIG" \
  --train-dir "$SMOKE_TRAIN" --val-dir "$SMOKE_VAL" \
  --device cpu --max-steps 5 \
  --out /tmp/coinrun-autoencoder-detail-v2-smoke

RESUME_ARGS=()
if [[ -f "$OUT/checkpoints/last.pt" ]]; then
  echo "Reanudando desde $OUT/checkpoints/last.pt"
  RESUME_ARGS=(--resume "$OUT/checkpoints/last.pt")
fi

python -m src.training.train_autoencoder \
  --config "$CONFIG" \
  --train-dir "$TRAIN" --val-dir "$VAL" \
  --device cuda --out "$OUT" "${RESUME_ARGS[@]}" \
  2>&1 | tee "$OUT/training.log"

python -m src.evaluation.reconstruction_metrics \
  --checkpoint "$OUT/checkpoints/best.pt" \
  --eval-dir "$VAL" --device cuda \
  --out "$OUT/metrics.json"

python -m src.evaluation.independent_reconstruction_review \
  --checkpoint "$OUT/checkpoints/best.pt" \
  --eval-dir "$VAL" --n-samples 10 --device cuda \
  --out "$OUT/independent_review"

pytest \
  tests/test_autoencoder.py \
  tests/test_external_coinrun_dataset.py \
  tests/test_independent_reconstruction_review.py \
  -v | tee "$OUT/test_output.log"

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = json.loads((root / "metrics.json").read_text())
review = json.loads((root / "independent_review" / "report.json").read_text())
summary = {
    "go": metrics["go"],
    "psnr_db": metrics["global"]["model_psnr_db"],
    "ssim": metrics["global"]["model_ssim"],
    "yellow_recall": metrics["coin"]["yellow_pixel_recall"],
    "yellow_precision": metrics["coin"]["yellow_pixel_precision"],
    "independent_metrics_reproduced": review["reported_comparison"]["reproduced_within_1e_6"],
    "independent_coin_samples": review["selected_coin_samples"],
}
print(json.dumps(summary, indent=2))
if not metrics["go"]:
    raise SystemExit("NO-GO: no construir el RSSM; revisar metrics.json")
if not review["reported_comparison"]["reproduced_within_1e_6"]:
    raise SystemExit("NO-GO: la auditoría independiente no reproduce las métricas")
PY

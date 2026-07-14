#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

pytest tests/test_data_collection.py -v \
  | tee experiments/EXP-002-dataset-audit/upstream_plan1_test_output.log

python -m src.data_prep.audit_dataset \
  --path 'data/raw/*/episode\_*.h5' \
  --out experiments/EXP-002-dataset-audit/quality_report.json \
  > experiments/EXP-002-dataset-audit/audit_stdout.log

set +e
python -m src.data_prep.split_dataset \
  --path 'data/raw/*/episode\_*.h5' \
  --strategy by_episode_within_seed \
  --out data/splits/ \
  > experiments/EXP-002-dataset-audit/split_stdout.log 2>&1
split_exit_code=$?
set -e
printf '%s\n' "$split_exit_code" > experiments/EXP-002-dataset-audit/split_exit_code.txt
if [[ "$split_exit_code" -ne 0 && "$split_exit_code" -ne 2 ]]; then
  exit "$split_exit_code"
fi

cp data/splits/summary.json experiments/EXP-002-dataset-audit/split_summary.json
pytest tests/test_dataset_prep.py -v \
  | tee experiments/EXP-002-dataset-audit/test_output.log

pip freeze > environment.txt
git rev-parse HEAD > git_commit.txt

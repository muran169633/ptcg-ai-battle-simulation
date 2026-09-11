#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="/home/xxc/miniconda3/envs/my_project_env/bin/python"
data_dir="$repo_dir/data/retrain_top50_recent14_alltrain_20260814_v2/daily_parquet_v5"
scratch_dir="$repo_dir/artifacts/retrain_top50_recent14_alltrain_20260814_v2/general_bc_v5_scratch8_seed2026081408"
tail_dir="$repo_dir/artifacts/retrain_top50_recent14_alltrain_20260814_v2/general_bc_v5_tail4_lr3e5_frome8_seed2026081409"

while pgrep -f "tools/train_bc_orbit.py.*general_bc_v5_scratch8_seed2026081408" >/dev/null; do
  sleep 30
done

"$python_bin" - "$scratch_dir" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
summary = json.loads((run / "summary.json").read_text())
if len(metrics) != 8 or metrics[-1].get("epoch") != 8:
    raise SystemExit("scratch BC did not complete all 8 epochs")
if summary.get("best_epoch") != 8 or not (run / "best.pt").is_file():
    raise SystemExit("scratch BC final checkpoint audit failed")
PY

if [[ -e "$tail_dir" ]]; then
  echo "Refusing to reuse tail BC output: $tail_dir" >&2
  exit 1
fi

"$python_bin" tools/train_bc_orbit.py \
  --data "$data_dir" \
  --output-dir "$tail_dir" \
  --epochs 4 \
  --batch-size 256 \
  --workers 8 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --target-accuracy 0.80 \
  --feature-version ptcg-bc-orbit-entity-transformer-v5 \
  --expected-train-rows 3984602 \
  --init-checkpoint "$scratch_dir/best.pt" \
  --split-mode all_train \
  --skip-valid \
  --skip-test \
  --train-shuffle-buffer-rows-per-worker 4096 \
  --seed 2026081409 \
  --device cuda

"$python_bin" tools/run_meta_weighted_ppo_routes.py \
  --route dragapult \
  --bc-checkpoint "$tail_dir/best.pt"

"$python_bin" tools/run_meta_weighted_ppo_routes.py \
  --route rmy \
  --bc-checkpoint "$tail_dir/best.pt"

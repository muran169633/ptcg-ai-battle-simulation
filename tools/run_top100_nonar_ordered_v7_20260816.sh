#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/xxc/miniconda3/envs/my_project_env/bin/python"
data_root="$repo_root/data/retrain_top100_recent14_nonar_order_end0813_20260816_v1"
parquet_output="$data_root/daily_parquet_split90_10"
artifact_output="$repo_root/artifacts/bc_top100_recent14_nonar_order_v7_end0813_b2048_20260816_v1"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing my_project_env Python: $python_bin" >&2
  exit 2
fi

daily_inputs=(
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-07-31.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-01.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-02.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-03.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-04.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-05.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-06.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-07.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-08.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-09.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-10.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-11.zip"
  "$repo_root/data/retrain_top100_recent14_alltrain_20260813_v1/daily_archives/2026-08-12.zip"
  "$repo_root/data/retrain_top100_refresh_20260814_v1/daily_archives/2026-08-13.zip"
)
for source_path in "${daily_inputs[@]}"; do
  if [[ ! -f "$source_path" ]]; then
    echo "Missing daily Top100 source: $source_path" >&2
    exit 2
  fi
done

mkdir -p "$data_root/logs"
cd "$repo_root"

if [[ ! -f "$parquet_output/manifest.json" ]]; then
  if [[ -e "$parquet_output" ]]; then
    echo "Incomplete Parquet output already exists: $parquet_output" >&2
    exit 2
  fi
  converter_args=()
  for source_path in "${daily_inputs[@]}"; do
    converter_args+=(--input "$source_path")
  done
  PYTHONPATH=tools PTCG_BC_FEATURE_MODULE=bc_nonar_v7 \
    "$python_bin" tools/convert_bc_daily_zips_to_split_parquet.py \
      "${converter_args[@]}" \
      --output-dir "$parquet_output" \
      --workers 8 \
      --rows-per-group 4096 \
      --split-seed 20260815 \
      --train-fraction 0.90 \
      2>&1 | tee "$data_root/logs/convert.log"
fi

if [[ -e "$artifact_output" ]]; then
  echo "Training output already exists; refusing to overwrite: $artifact_output" >&2
  exit 2
fi

"$python_bin" tools/train_bc_nonar_v7.py \
  --data "$parquet_output" \
  --output-dir "$artifact_output" \
  --epochs 12 \
  --batch-size 2048 \
  --workers 8 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --categorical-dim 64 \
  --model-dim 192 \
  --layers 4 \
  --heads 6 \
  --dropout 0.05 \
  --set-bce-weight 0.25 \
  --count-loss-weight 0.05 \
  --value-loss-weight 0.05 \
  --seed 20260816 \
  --target-accuracy 0.80 \
  --train-shuffle-buffer-rows-per-worker 65536 \
  --daily-equal-weighting \
  --device cuda \
  2>&1 | tee "$data_root/logs/train_12epochs.log"

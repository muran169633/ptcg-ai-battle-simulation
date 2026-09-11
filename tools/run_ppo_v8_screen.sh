#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
output_dir="artifacts/ppo_marnie_v8_screen_u455"

candidate="artifacts/ppo_marnie_v8_lastblockheads_lr12e6_u455/checkpoints/update-0455.pt"
bc_checkpoint="artifacts/bc_marnie_top50_train23_valid24_orbit_v2/best.pt"
gate_archive="data/bc_marnie_valid24_hash75_25_seed20260820.zip"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
v3_u440="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
v3_u570="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0570.pt"
kang_checkpoint="artifacts/bc_kangaskhan_orbit_v1/best.pt"
kang_deck="data/decks/kangaskhan_standard.csv"

if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"

"$python_bin" tools/evaluate_policy_bc.py \
  --checkpoint "$candidate" \
  --data "$gate_archive" \
  --split valid \
  --split-mode archive \
  --split-seed 20260723 \
  --batch-size 256 \
  --workers 8 \
  --prediction-order policy \
  --device cuda \
  --compact \
  --json-output "$output_dir/gate25_behavior.json"

run_match() {
  local label="$1"
  local opponent="$2"
  local opponent_deck="$3"
  local games="$4"
  local seed="$5"
  shift 5
  "$python_bin" tools/evaluate_ppo_head_to_head.py \
    --candidate "$candidate" \
    --opponent "$opponent" \
    --bc-checkpoint "$bc_checkpoint" \
    --candidate-deck "$marnie_deck" \
    --opponent-deck "$opponent_deck" \
    --games "$games" \
    --environments 16 \
    --max-game-decisions 1000 \
    --seed "$seed" \
    --device cuda \
    "$@" \
    --output "$output_dir/$label.json"
}

run_match vs_v3u440_256_block1 \
  "$v3_u440" "$marnie_deck" 256 20260912
run_match vs_v3u570_128 \
  "$v3_u570" "$marnie_deck" 128 20260913
run_match vs_kang128 \
  "$kang_checkpoint" "$kang_deck" 128 20260914 \
  --opponent-canonical-order
run_match vs_v3u440_256_block2 \
  "$v3_u440" "$marnie_deck" 256 20260915

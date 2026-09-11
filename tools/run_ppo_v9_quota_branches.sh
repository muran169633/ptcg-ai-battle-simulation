#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
requested_branch="${1:-all}"

if [[ "$requested_branch" != "all" && "$requested_branch" != "fixed" && "$requested_branch" != "adaptive" ]]; then
  echo "Usage: $0 [all|fixed|adaptive]" >&2
  exit 2
fi

bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
v3_checkpoint="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
v1_name="update-0200@marnie_grimmsnarl_froslass_luca"
v3_name="update-0440@marnie_grimmsnarl_froslass_luca"
v4_name="alpha-075@marnie_grimmsnarl_froslass_luca"
drag_name="best@dragapult_lumen"
kang_name="best@kangaskhan_standard"

common_args=(
  --bc-checkpoint "$bc_checkpoint"
  --kl-reference-checkpoint "$v3_checkpoint"
  --deck "$marnie_deck"
  --extra-opponent
    artifacts/ppo_marnie_terminal01_v1/checkpoints/update-0200.pt
    "$marnie_deck"
  --extra-opponent
    "$v3_checkpoint"
    "$marnie_deck"
  --extra-opponent
    artifacts/ppo_marnie_v4_soups_u80_u100/alpha-075.pt
    "$marnie_deck"
  --extra-opponent
    artifacts/bc_dragapult_lumen_orbit_v1/best.pt
    data/decks/dragapult_lumen.csv
  --extra-opponent
    artifacts/bc_kangaskhan_orbit_v1/best.pt
    data/decks/kangaskhan_standard.csv
  --updates 455
  --environments 16
  --games-per-update 96
  --ppo-epochs 4
  --minibatch-size 1024
  --learning-rate 3e-6
  --value-learning-rate 1.5e-5
  --weight-decay 1e-4
  --learning-rate-schedule constant
  --schedule-start-update 441
  --gamma 1.0
  --gae-lambda 0.97
  --clip-ratio 0.15
  --value-coefficient 0.5
  --entropy-coefficient 0.003
  --max-grad-norm 0.5
  --trainable-scope full
  --bc-kl-start 0.002
  --bc-kl-end 0.002
  --target-kl 0.008
  --league-probability 1.0
  --opponent-sampling per_game
  --snapshot-interval 1000
  --max-pool-size 6
  --eval-interval 15
  --eval-games 64
  --eval-all-permanent-opponents
  --selection-aggregation min
  --checkpoint-interval 15
  --max-game-decisions 1000
  --seed 20260929
  --resume "$v3_checkpoint"
  --reset-optimizer-on-resume
  --skip-initial-eval
  --device cuda
)

run_fixed() {
  local output_dir="artifacts/ppo_marnie_v9_FQ_fixedquota_u455"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    exit 1
  fi
  "$python_bin" tools/train_ppo.py \
    "${common_args[@]}" \
    --output-dir "$output_dir" \
    --opponent-quota-mode fixed \
    --opponent-base-quota bc 12 \
    --opponent-base-quota "$v1_name" 12 \
    --opponent-base-quota "$v3_name" 36 \
    --opponent-base-quota "$v4_name" 12 \
    --opponent-base-quota "$drag_name" 12 \
    --opponent-base-quota "$kang_name" 12
}

run_adaptive() {
  local output_dir="artifacts/ppo_marnie_v9_UQ_betaucb_u455"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    exit 1
  fi
  "$python_bin" tools/train_ppo.py \
    "${common_args[@]}" \
    --output-dir "$output_dir" \
    --opponent-quota-mode adaptive \
    --opponent-quota-refresh-updates 3 \
    --opponent-base-quota bc 8 \
    --opponent-base-quota "$v1_name" 8 \
    --opponent-base-quota "$v3_name" 32 \
    --opponent-base-quota "$v4_name" 8 \
    --opponent-base-quota "$drag_name" 8 \
    --opponent-base-quota "$kang_name" 8 \
    --opponent-cap bc 24 \
    --opponent-cap "$v1_name" 24 \
    --opponent-cap "$v3_name" 48 \
    --opponent-cap "$v4_name" 24 \
    --opponent-cap "$drag_name" 24 \
    --opponent-cap "$kang_name" 24 \
    --opponent-audit bc 32 32 0 \
    --opponent-audit "$v1_name" 38 26 0 \
    --opponent-audit "$v3_name" 31 33 0 \
    --opponent-audit "$v4_name" 41 23 0 \
    --opponent-audit "$drag_name" 61 3 0 \
    --opponent-audit "$kang_name" 45 19 0
}

if [[ "$requested_branch" == "all" || "$requested_branch" == "fixed" ]]; then
  run_fixed
fi
if [[ "$requested_branch" == "all" || "$requested_branch" == "adaptive" ]]; then
  run_adaptive
fi

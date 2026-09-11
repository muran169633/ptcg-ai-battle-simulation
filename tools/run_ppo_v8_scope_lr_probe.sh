#!/usr/bin/env bash
set -euo pipefail

# Calibrate last_block_heads actor LR against the first three-update KL of the
# v7 full-model control.  No head-to-head result is used to choose the LR.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"

bc_checkpoint="artifacts/bc_marnie_top50_train23_valid24_orbit_v2/best.pt"
v3_checkpoint="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"

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
  --updates 443
  --environments 16
  --games-per-update 96
  --ppo-epochs 4
  --minibatch-size 1024
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
  --trainable-scope last_block_heads
  --bc-kl-start 0.002
  --bc-kl-end 0.002
  --target-kl 0.008
  --league-probability 1.0
  --opponent-sampling per_game
  --bc-opponent-probability 0.15
  --snapshot-interval 1000
  --max-pool-size 6
  --eval-interval 1000
  --eval-games 2
  --checkpoint-interval 3
  --max-game-decisions 1000
  --seed 20260910
  --resume "$v3_checkpoint"
  --reset-optimizer-on-resume
  --skip-initial-eval
  --device cuda
)

run_probe() {
  local label="$1"
  local actor_lr="$2"
  local output_dir="artifacts/ppo_marnie_v8_scope_probe_${label}"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    return 1
  fi
  "$python_bin" tools/train_ppo.py \
    "${common_args[@]}" \
    --learning-rate "$actor_lr" \
    --output-dir "$output_dir"
}

if (($# == 0)); then
  set -- lr6e6 lr9e6 lr12e6
fi

for label in "$@"; do
  case "$label" in
    lr6e6)
      run_probe "$label" 6e-6
      ;;
    lr9e6)
      run_probe "$label" 9e-6
      ;;
    lr12e6)
      run_probe "$label" 1.2e-5
      ;;
    *)
      echo "Unknown LR probe: $label" >&2
      exit 2
      ;;
  esac
done

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"

bc_checkpoint="artifacts/bc_marnie_top50_train23_valid24_orbit_v2/best.pt"
v3_checkpoint="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
output_dir="artifacts/ppo_marnie_v8_lastblockheads_lr12e6_u455"

if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi

"$python_bin" tools/train_ppo.py \
  --bc-checkpoint "$bc_checkpoint" \
  --kl-reference-checkpoint "$v3_checkpoint" \
  --deck "$marnie_deck" \
  --extra-opponent \
    artifacts/ppo_marnie_terminal01_v1/checkpoints/update-0200.pt \
    "$marnie_deck" \
  --extra-opponent \
    "$v3_checkpoint" \
    "$marnie_deck" \
  --extra-opponent \
    artifacts/ppo_marnie_v4_soups_u80_u100/alpha-075.pt \
    "$marnie_deck" \
  --extra-opponent \
    artifacts/bc_dragapult_lumen_orbit_v1/best.pt \
    data/decks/dragapult_lumen.csv \
  --extra-opponent \
    artifacts/bc_kangaskhan_orbit_v1/best.pt \
    data/decks/kangaskhan_standard.csv \
  --output-dir "$output_dir" \
  --updates 455 \
  --environments 16 \
  --games-per-update 96 \
  --ppo-epochs 4 \
  --minibatch-size 1024 \
  --learning-rate 1.2e-5 \
  --value-learning-rate 1.5e-5 \
  --weight-decay 1e-4 \
  --learning-rate-schedule constant \
  --schedule-start-update 441 \
  --gamma 1.0 \
  --gae-lambda 0.97 \
  --clip-ratio 0.15 \
  --value-coefficient 0.5 \
  --entropy-coefficient 0.003 \
  --max-grad-norm 0.5 \
  --trainable-scope last_block_heads \
  --bc-kl-start 0.002 \
  --bc-kl-end 0.002 \
  --target-kl 0.008 \
  --league-probability 1.0 \
  --opponent-sampling per_game \
  --bc-opponent-probability 0.15 \
  --snapshot-interval 1000 \
  --max-pool-size 6 \
  --eval-interval 15 \
  --eval-games 64 \
  --eval-all-permanent-opponents \
  --selection-aggregation min \
  --checkpoint-interval 15 \
  --max-game-decisions 1000 \
  --seed 20260901 \
  --resume "$v3_checkpoint" \
  --reset-optimizer-on-resume \
  --skip-initial-eval \
  --device cuda

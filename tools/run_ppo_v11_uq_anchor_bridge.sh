#!/usr/bin/env bash
set -euo pipefail

# Matched v10 ablation: the sole training change is the KL reference.  v11
# anchors to the successful v9 UQ start instead of pulling back toward v3.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
output_dir="artifacts/ppo_marnie_v11_UQanchor_bcbridge_u461"
bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
resume_checkpoint="artifacts/ppo_marnie_v9_UQ_betaucb_u455/checkpoints/update-0455.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
v1_name="update-0200@marnie_grimmsnarl_froslass_luca"
v3_name="update-0440@marnie_grimmsnarl_froslass_luca"
v4_name="alpha-075@marnie_grimmsnarl_froslass_luca"
drag_name="best@dragapult_lumen"
kang_name="best@kangaskhan_standard"

for path in \
  "$bc_checkpoint" \
  "$resume_checkpoint" \
  "$marnie_deck" \
  artifacts/ppo_marnie_terminal01_v1/checkpoints/update-0200.pt \
  artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt \
  artifacts/ppo_marnie_v4_soups_u80_u100/alpha-075.pt \
  artifacts/bc_dragapult_lumen_orbit_v1/best.pt \
  artifacts/bc_kangaskhan_orbit_v1/best.pt \
  data/decks/dragapult_lumen.csv \
  data/decks/kangaskhan_standard.csv; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi

"$python_bin" tools/train_ppo.py \
  --bc-checkpoint "$bc_checkpoint" \
  --kl-reference-checkpoint "$resume_checkpoint" \
  --deck "$marnie_deck" \
  --extra-opponent \
    artifacts/ppo_marnie_terminal01_v1/checkpoints/update-0200.pt \
    "$marnie_deck" \
  --extra-opponent \
    artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt \
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
  --updates 461 \
  --environments 16 \
  --games-per-update 96 \
  --ppo-epochs 4 \
  --minibatch-size 1024 \
  --learning-rate 3e-6 \
  --value-learning-rate 1.5e-5 \
  --weight-decay 1e-4 \
  --learning-rate-schedule constant \
  --schedule-start-update 456 \
  --gamma 1.0 \
  --gae-lambda 0.97 \
  --clip-ratio 0.15 \
  --value-coefficient 0.5 \
  --entropy-coefficient 0.003 \
  --max-grad-norm 0.5 \
  --trainable-scope full \
  --bc-kl-start 0.004 \
  --bc-kl-end 0.004 \
  --target-kl 0.008 \
  --league-probability 1.0 \
  --opponent-sampling per_game \
  --opponent-quota-mode fixed \
  --opponent-base-quota bc 32 \
  --opponent-base-quota "$v1_name" 4 \
  --opponent-base-quota "$v3_name" 48 \
  --opponent-base-quota "$v4_name" 4 \
  --opponent-base-quota "$drag_name" 4 \
  --opponent-base-quota "$kang_name" 4 \
  --snapshot-interval 1000 \
  --max-pool-size 6 \
  --eval-interval 461 \
  --eval-games 64 \
  --eval-all-permanent-opponents \
  --selection-aggregation min \
  --checkpoint-interval 461 \
  --max-game-decisions 1000 \
  --seed 20261001 \
  --resume "$resume_checkpoint" \
  --reset-optimizer-on-resume \
  --reset-opponent-quota-on-resume \
  --skip-initial-eval \
  --device cuda

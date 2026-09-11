#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"

# Resume the strongest v1 policy at update 200 and add 400 more updates.
# The frozen v1 checkpoint is a permanent incumbent; checkpoint selection uses
# the worst win rate across v1 and the original BC anchor.
exec "$python_bin" tools/train_ppo.py \
  --bc-checkpoint artifacts/bc_marnie_luca_orbit_v5/best.pt \
  --deck data/decks/marnie_grimmsnarl_froslass_luca.csv \
  --extra-opponent \
    artifacts/ppo_marnie_terminal01_v1/checkpoints/update-0200.pt \
    data/decks/marnie_grimmsnarl_froslass_luca.csv \
  --output-dir artifacts/ppo_marnie_v3_incumbent600 \
  --updates 600 \
  --environments 32 \
  --games-per-update 128 \
  --ppo-epochs 4 \
  --minibatch-size 1024 \
  --learning-rate 5e-6 \
  --value-learning-rate 2e-5 \
  --weight-decay 1e-4 \
  --gamma 1.0 \
  --gae-lambda 0.95 \
  --clip-ratio 0.2 \
  --value-coefficient 0.5 \
  --entropy-coefficient 0.005 \
  --max-grad-norm 0.5 \
  --bc-kl-start 0.01 \
  --bc-kl-end 0.001 \
  --target-kl 0.015 \
  --league-probability 1.0 \
  --opponent-sampling per_game \
  --bc-opponent-probability 0.10 \
  --snapshot-interval 10 \
  --max-pool-size 4 \
  --eval-interval 10 \
  --eval-games 128 \
  --eval-all-permanent-opponents \
  --selection-aggregation min \
  --checkpoint-interval 10 \
  --max-game-decisions 1000 \
  --seed 20260726 \
  --resume artifacts/ppo_marnie_terminal01_v1/checkpoints/update-0200.pt \
  --device cuda

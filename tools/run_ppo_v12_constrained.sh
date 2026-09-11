#!/usr/bin/env bash
set -euo pipefail

# Structural v12: repair the fresh-BC weakness from the v9 UQ endpoint while
# explicitly guarding the v3 surrogate. The full run is a locked 12-update
# endpoint; "smoke" exercises the same mechanisms on one small update.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
mode="${1:-full}"
bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
resume_checkpoint="artifacts/ppo_marnie_v9_UQ_betaucb_u455/checkpoints/update-0455.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
v1_name="update-0200@marnie_grimmsnarl_froslass_luca"
v3_name="update-0440@marnie_grimmsnarl_froslass_luca"
v4_name="alpha-075@marnie_grimmsnarl_froslass_luca"
drag_name="best@dragapult_lumen"
kang_name="best@kangaskhan_standard"

case "$mode" in
  full)
    output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v12_constrained_seatbalanced_u467}"
    updates=467
    games_per_update=96
    environments=16
    ppo_epochs=4
    minibatch_size=1024
    eval_games=64
    seed=20261021
    bc_quota=40
    v3_quota=40
    auxiliary_quota=4
    ;;
  smoke)
    output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v12_constrained_smoke_u456}"
    updates=456
    games_per_update=16
    environments=8
    ppo_epochs=2
    minibatch_size=256
    eval_games=8
    seed=20261020
    bc_quota=4
    v3_quota=4
    auxiliary_quota=2
    ;;
  full_smoke)
    output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v12_constrained_fullbatch_smoke_u456}"
    updates=456
    games_per_update=96
    environments=16
    ppo_epochs=4
    minibatch_size=1024
    eval_games=8
    seed=20261023
    bc_quota=40
    v3_quota=40
    auxiliary_quota=4
    ;;
  *)
    echo "Usage: $0 [full|smoke|full_smoke]" >&2
    exit 2
    ;;
esac

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
  --updates "$updates" \
  --environments "$environments" \
  --games-per-update "$games_per_update" \
  --ppo-epochs "$ppo_epochs" \
  --minibatch-size "$minibatch_size" \
  --learning-rate 3e-6 \
  --value-learning-rate 1.5e-5 \
  --weight-decay 1e-4 \
  --learning-rate-schedule constant \
  --schedule-start-update 456 \
  --gamma 1.0 \
  --gae-lambda 0.97 \
  --advantage-normalization per_opponent \
  --clip-ratio 0.15 \
  --value-coefficient 0.5 \
  --entropy-coefficient 0.003 \
  --max-grad-norm 0.5 \
  --trainable-scope full \
  --bc-kl-start 0.002 \
  --bc-kl-end 0.002 \
  --target-kl 0.008 \
  --league-probability 1.0 \
  --opponent-sampling per_game \
  --opponent-quota-mode fixed \
  --opponent-quota-seat-balance \
  --opponent-base-quota bc "$bc_quota" \
  --opponent-base-quota "$v1_name" "$auxiliary_quota" \
  --opponent-base-quota "$v3_name" "$v3_quota" \
  --opponent-base-quota "$v4_name" "$auxiliary_quota" \
  --opponent-base-quota "$drag_name" "$auxiliary_quota" \
  --opponent-base-quota "$kang_name" "$auxiliary_quota" \
  --ppo-objective constrained \
  --primary-opponent-name bc \
  --guard-opponent-name "$v3_name" \
  --primary-policy-weight 1.0 \
  --guard-policy-weight 0.25 \
  --auxiliary-policy-weight 0.10 \
  --guard-surrogate-floor -0.002 \
  --constraint-dual-initial 1.0 \
  --constraint-dual-lr 0.05 \
  --constraint-dual-max 10.0 \
  --opponent-loss-weight bc 0.45 \
  --opponent-loss-weight "$v1_name" 0.025 \
  --opponent-loss-weight "$v3_name" 0.45 \
  --opponent-loss-weight "$v4_name" 0.025 \
  --opponent-loss-weight "$drag_name" 0.025 \
  --opponent-loss-weight "$kang_name" 0.025 \
  --snapshot-interval 1000 \
  --max-pool-size 6 \
  --eval-interval "$updates" \
  --eval-games "$eval_games" \
  --eval-all-permanent-opponents \
  --selection-aggregation min \
  --checkpoint-interval "$updates" \
  --max-game-decisions 1000 \
  --seed "$seed" \
  --resume "$resume_checkpoint" \
  --reset-optimizer-on-resume \
  --reset-opponent-quota-on-resume \
  --skip-initial-eval \
  --device cuda

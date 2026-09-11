#!/usr/bin/env bash
set -euo pipefail

# Deployment-aligned v14: continue from the v9 UQ endpoint with the v12
# opponent-balanced constrained objective, but collect and optimize learner
# actions at temperature 0.5. Submission inference is greedy, so this reduces
# the train/deploy distribution gap while retaining stochastic exploration.

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
    output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v14_lowtemp_constrained_u463}"
    updates=463
    games_per_update=128
    environments=16
    ppo_epochs=4
    minibatch_size=1024
    eval_games=64
    seed=20261111
    bc_quota=52
    v3_quota=52
    auxiliary_quota=6
    ;;
  smoke)
    output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v14_lowtemp_smoke_u456}"
    updates=456
    games_per_update=16
    environments=8
    ppo_epochs=2
    minibatch_size=256
    eval_games=8
    seed=20261110
    bc_quota=4
    v3_quota=4
    auxiliary_quota=2
    ;;
  *)
    echo "Usage: $0 [full|smoke]" >&2
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
  --learning-rate 1.5e-6 \
  --value-learning-rate 1.5e-5 \
  --weight-decay 1e-4 \
  --learning-rate-schedule constant \
  --schedule-start-update 456 \
  --gamma 1.0 \
  --gae-lambda 0.97 \
  --advantage-normalization per_opponent \
  --clip-ratio 0.15 \
  --value-coefficient 0.5 \
  --entropy-coefficient 0.0015 \
  --max-grad-norm 0.5 \
  --policy-temperature 0.5 \
  --trainable-scope full \
  --bc-kl-start 0.004 \
  --bc-kl-end 0.004 \
  --target-kl 0.006 \
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
  --guard-policy-weight 0.5 \
  --auxiliary-policy-weight 0.05 \
  --guard-surrogate-floor 0.0 \
  --constraint-dual-initial 1.0 \
  --constraint-dual-lr 0.10 \
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

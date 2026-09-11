#!/usr/bin/env bash
set -euo pipefail

# Projected-head v15: train only the action/count and value heads from the
# locked v9 UQ endpoint. BC and v3 are the only rollout tasks; conflicting
# policy gradients are symmetrically projected before every optimizer step.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
mode="${1:-full}"
bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
resume_checkpoint="artifacts/ppo_marnie_v9_UQ_betaucb_u455/checkpoints/update-0455.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
v3_checkpoint="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
v3_name="update-0440@marnie_grimmsnarl_froslass_luca"

case "$mode" in
  full)
    output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v15_pcgrad_heads_u461}"
    updates=461
    games_per_update=128
    environments=16
    ppo_epochs=2
    minibatch_size=1024
    eval_games=64
    seed=20261121
    core_quota=64
    ;;
  smoke)
    output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v15_pcgrad_heads_smoke_u456}"
    updates=456
    games_per_update=16
    environments=8
    ppo_epochs=2
    minibatch_size=256
    eval_games=8
    seed=20261120
    core_quota=8
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
  "$v3_checkpoint"; do
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
  --extra-opponent "$v3_checkpoint" "$marnie_deck" \
  --output-dir "$output_dir" \
  --updates "$updates" \
  --environments "$environments" \
  --games-per-update "$games_per_update" \
  --ppo-epochs "$ppo_epochs" \
  --minibatch-size "$minibatch_size" \
  --learning-rate 1.2e-5 \
  --value-learning-rate 1.5e-5 \
  --weight-decay 0 \
  --learning-rate-schedule constant \
  --schedule-start-update 456 \
  --gamma 1.0 \
  --gae-lambda 0.97 \
  --advantage-normalization per_opponent \
  --clip-ratio 0.10 \
  --value-coefficient 0.5 \
  --entropy-coefficient 0 \
  --max-grad-norm 0.5 \
  --policy-temperature 1.0 \
  --trainable-scope heads \
  --bc-kl-start 0.01 \
  --bc-kl-end 0.01 \
  --target-kl 0.003 \
  --league-probability 1.0 \
  --opponent-sampling per_game \
  --opponent-quota-mode fixed \
  --opponent-quota-seat-balance \
  --opponent-base-quota bc "$core_quota" \
  --opponent-base-quota "$v3_name" "$core_quota" \
  --ppo-objective constrained \
  --constrained-gradient-mode pcgrad \
  --primary-opponent-name bc \
  --guard-opponent-name "$v3_name" \
  --primary-policy-weight 1.0 \
  --guard-policy-weight 1.0 \
  --auxiliary-policy-weight 0 \
  --guard-surrogate-floor 0 \
  --constraint-dual-initial 0 \
  --constraint-dual-lr 0 \
  --constraint-dual-max 10 \
  --opponent-loss-weight bc 0.5 \
  --opponent-loss-weight "$v3_name" 0.5 \
  --snapshot-interval 1000 \
  --max-pool-size 2 \
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
  --device cuda

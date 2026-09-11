#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="/home/xxc/miniconda3/envs/my_project_env/bin/python"
feature_source="$repo_dir/frozen_behavior_source__sha256_f6c92841f2f6974408be05600b80f532b099a816a40ef2a5b2168fe45dc713ab/train_bc_orbit.py"
feature_sha256="ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
pool_dir="$repo_dir/data/current_top23_20260814_v3"
pool_manifest="$pool_dir/manifest.json"
pool_manifest_sha256="5aac436db172010eea4b66cb027e155cdf5970f507a963727aa16799b0acf3e7"
general_bc="$repo_dir/artifacts/retrain_top50_recent14_alltrain_20260813_v1/general_bc_v5_tail4_lr3e5_frome8_seed2026081314/best.pt"
parent_ppo="$repo_dir/artifacts/dragapult_standard_goldpush_20260813_v1/ppo_matched4_guard1_block1_g2048_lastblock_lr7e6_seed2026081321/soups/alpha_0p50.pt"
learner_deck="$repo_dir/data/top100_proxy_recent14_20260811_v1/decks/07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725.csv"
bc_replay="$repo_dir/data/top100_proxy_recent14_20260811_v1/archives/dragapult_ex.zip"
output_dir="$repo_dir/artifacts/dragapult_standard_goldpush_20260813_v1/ppo_currenttop23_90pool10self_gate58_mb2048_u4_seed2026081403"

export PTCG_BC_FEATURE_SOURCE="$feature_source"
export PTCG_BC_FEATURE_SOURCE_SHA256="$feature_sha256"

actual_manifest_sha256="$(sha256sum "$pool_manifest" | cut -d' ' -f1)"
if [[ "$actual_manifest_sha256" != "$pool_manifest_sha256" ]]; then
  echo "Top23 manifest hash mismatch: $actual_manifest_sha256" >&2
  exit 1
fi
if [[ -e "$output_dir" ]]; then
  echo "Refusing to reuse output directory: $output_dir" >&2
  exit 1
fi

mapfile -d '' deck_paths < <(find "$pool_dir/decks" -maxdepth 1 -type f -name 'rank*.csv' -print0 | sort -z)
if [[ "${#deck_paths[@]}" -ne 23 ]]; then
  echo "Expected 23 frozen leaderboard deck slots, found ${#deck_paths[@]}" >&2
  exit 1
fi

args=(
  tools/train_ppo.py
  --bc-checkpoint "$general_bc"
  --kl-reference-checkpoint "$parent_ppo"
  --deck "$learner_deck"
  --output-dir "$output_dir"
  --updates 4
  --environments 64
  --games-per-update 1024
  --ppo-epochs 4
  --minibatch-size 2048
  --learning-rate 3e-6
  --value-learning-rate 1.5e-6
  --weight-decay 0
  --gamma 1
  --gae-lambda 1
  --advantage-normalization per_opponent
  --clip-ratio 0.10
  --value-coefficient 0.2
  --value-trunk-gradient-scale 0
  --entropy-coefficient 0.0005
  --max-grad-norm 0.25
  --policy-temperature 0.8
  --trainable-scope last_block_heads
  --learning-rate-schedule constant
  --schedule-start-update 1
  --bc-kl-start 0.04
  --bc-kl-end 0.04
  --target-kl 0.0005
  --league-probability 0.90
  --opponent-sampling per_game
  --opponent-weight bc 0
  --history-opponent-weight 0
  --opponent-quota-mode legacy
  --ppo-objective standard
  --actor-reduction episode_mean
  --actor-value-gradient-mode scalar
  --snapshot-interval 1000000
  --max-pool-size 24
  --bc-replay-data "$bc_replay"
  --bc-replay-split train
  --bc-replay-batches 32
  --bc-replay-batch-size 64
  --bc-replay-workers 4
  --bc-replay-steps 64
  --bc-replay-lr-scale 0.15
  --bc-replay-loss hybrid_ordered
  --eval-interval 2
  --eval-games 32
  --eval-all-permanent-opponents
  --selection-aggregation mean
  --champion-gate-interval 2
  --champion-gate-games 200
  --champion-gate-min-win-rate 0.58
  --checkpoint-interval 1
  --max-game-decisions 1000
  --seed 2026081403
  --resume "$parent_ppo"
  --resume-learner-weights resume
  --reset-optimizer-on-resume
  --skip-initial-eval
  --device cuda
)

for deck_path in "${deck_paths[@]}"; do
  deck_stem="$(basename "$deck_path" .csv)"
  args+=(--extra-opponent "$general_bc" "$deck_path")
  args+=(--opponent-weight "best@$deck_stem" 1)
done

exec "$python_bin" "${args[@]}"

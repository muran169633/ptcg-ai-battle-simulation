#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"

bc_checkpoint="artifacts/bc_marnie_top50_train23_valid24_orbit_v2/best.pt"
resume_checkpoint="artifacts/ppo_marnie_v6_phaseA_A00_newbc_uniform/checkpoints/update-0450.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
replay_archive="data/bc_marnie_valid24_hash75_25_seed20260820.zip"

common_args=(
  --bc-checkpoint "$bc_checkpoint"
  --deck "$marnie_deck"
  --extra-opponent
    artifacts/ppo_marnie_terminal01_v1/checkpoints/update-0200.pt
    "$marnie_deck"
  --extra-opponent
    artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt
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
  --updates 465
  --environments 16
  --games-per-update 96
  --ppo-epochs 4
  --minibatch-size 1024
  --learning-rate 3e-6
  --value-learning-rate 1.5e-5
  --weight-decay 1e-4
  --learning-rate-schedule constant
  --schedule-start-update 451
  --gamma 1.0
  --clip-ratio 0.15
  --value-coefficient 0.5
  --entropy-coefficient 0.003
  --max-grad-norm 0.5
  --bc-kl-start 0.002
  --bc-kl-end 0.002
  --target-kl 0.008
  --league-probability 1.0
  --opponent-sampling per_game
  --bc-opponent-probability 0.15
  --snapshot-interval 1000
  --max-pool-size 6
  --eval-interval 15
  --eval-games 64
  --eval-all-permanent-opponents
  --selection-aggregation min
  --checkpoint-interval 5
  --max-game-decisions 1000
  --seed 20260830
  --resume "$resume_checkpoint"
  --reset-optimizer-on-resume
  --skip-initial-eval
  --device cuda
)

run_one() {
  local run_name="$1"
  local replay_kind="$2"
  local gae_lambda="$3"
  local output_dir="artifacts/ppo_marnie_v6_phaseB_${run_name}"
  local replay_args=()

  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    return 1
  fi

  if [[ "$replay_kind" == "light" ]]; then
    replay_args=(
      --bc-replay-data "$replay_archive"
      --bc-replay-split train
      --bc-replay-batches 72
      --bc-replay-batch-size 256
      --bc-replay-workers 8
      --bc-replay-steps 1
      --bc-replay-lr-scale 0.10
    )
  elif [[ "$replay_kind" != "none" ]]; then
    echo "Unknown replay kind: $replay_kind" >&2
    return 1
  fi

  "$python_bin" tools/train_ppo.py \
    "${common_args[@]}" \
    "${replay_args[@]}" \
    --gae-lambda "$gae_lambda" \
    --output-dir "$output_dir"
}

if (($# == 0)); then
  set -- B0_noreplay_full B1_lightreplay_full B2_lambda100_noreplay_full
fi

for run_name in "$@"; do
  case "$run_name" in
    B0_noreplay_full)
      run_one "$run_name" none 0.97
      ;;
    B1_lightreplay_full)
      run_one "$run_name" light 0.97
      ;;
    B2_lambda100_noreplay_full)
      run_one "$run_name" none 1.0
      ;;
    *)
      echo "Unknown Phase B run: $run_name" >&2
      exit 2
      ;;
  esac
done

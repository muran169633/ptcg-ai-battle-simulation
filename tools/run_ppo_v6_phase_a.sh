#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"

bc_checkpoint="artifacts/bc_marnie_top50_train23_valid24_orbit_v2/best.pt"
v3_checkpoint="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"

common_args=(
  --bc-checkpoint "$bc_checkpoint"
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
  --updates 450
  --environments 16
  --games-per-update 64
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
  --bc-kl-start 0.002
  --bc-kl-end 0.002
  --target-kl 0.008
  --league-probability 1.0
  --opponent-sampling per_game
  --snapshot-interval 1000
  --max-pool-size 6
  --eval-interval 10
  --eval-games 64
  --eval-all-permanent-opponents
  --selection-aggregation min
  --checkpoint-interval 5
  --max-game-decisions 1000
  --seed 20260820
  --resume "$v3_checkpoint"
  --reset-optimizer-on-resume
  --skip-initial-eval
  --device cuda
)

run_one() {
  local run_name="$1"
  local reference_kind="$2"
  local sampling_kind="$3"
  local output_dir="artifacts/ppo_marnie_v6_phaseA_${run_name}"
  local reference_args=()
  local sampling_args=()

  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing output: $output_dir" >&2
    return 1
  fi

  if [[ "$reference_kind" == "v3" ]]; then
    reference_args=(--kl-reference-checkpoint "$v3_checkpoint")
  elif [[ "$reference_kind" != "newbc" ]]; then
    echo "Unknown reference kind: $reference_kind" >&2
    return 1
  fi

  if [[ "$sampling_kind" == "weighted" ]]; then
    sampling_args=(
      --opponent-weight bc 10
      --opponent-weight update-0200@marnie_grimmsnarl_froslass_luca 10
      --opponent-weight update-0440@marnie_grimmsnarl_froslass_luca 45
      --opponent-weight alpha-075@marnie_grimmsnarl_froslass_luca 10
      --opponent-weight best@dragapult_lumen 5
      --opponent-weight best@kangaskhan_standard 20
    )
  elif [[ "$sampling_kind" == "uniform" ]]; then
    # Reproduce v5's base-BC 15%, with the remaining mass split uniformly.
    sampling_args=(--bc-opponent-probability 0.15)
  else
    echo "Unknown sampling kind: $sampling_kind" >&2
    return 1
  fi

  "$python_bin" tools/train_ppo.py \
    "${common_args[@]}" \
    "${reference_args[@]}" \
    "${sampling_args[@]}" \
    --output-dir "$output_dir"
}

if (($# == 0)); then
  set -- \
    A00_newbc_uniform \
    A01_v3_uniform \
    A10_newbc_weighted \
    A11_v3_weighted
fi

for run_name in "$@"; do
  case "$run_name" in
    A00_newbc_uniform)
      run_one "$run_name" newbc uniform
      ;;
    A01_v3_uniform)
      run_one "$run_name" v3 uniform
      ;;
    A10_newbc_weighted)
      run_one "$run_name" newbc weighted
      ;;
    A11_v3_weighted)
      run_one "$run_name" v3 weighted
      ;;
    *)
      echo "Unknown Phase A run: $run_name" >&2
      exit 2
      ;;
  esac
done

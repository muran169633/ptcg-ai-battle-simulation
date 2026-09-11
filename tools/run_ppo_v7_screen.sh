#!/usr/bin/env bash
set -euo pipefail

# Screen the three locked v7 update-0455 checkpoints with the same evaluation
# geometry used by the v6 Phase B screen.  This script intentionally does not
# select or promote a winner; it only writes one immutable JSON report per run.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
resume_existing="${RESUME_EXISTING:-0}"
output_dir="artifacts/ppo_marnie_v7_screen_u455"

bc_checkpoint="artifacts/bc_marnie_top50_train23_valid24_orbit_v2/best.pt"
gate_archive="data/bc_marnie_valid24_hash75_25_seed20260820.zip"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
kang_deck="data/decks/kangaskhan_standard.csv"
v3_u440="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
v3_u570="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0570.pt"
kang_checkpoint="artifacts/bc_kangaskhan_orbit_v1/best.pt"

candidate_ids=(
  R0H0_full_noreplay_u455
  R1H0_full_hybridreplay_u455
  R0H1_lastblockheads_noreplay_u455
)
candidate_checkpoints=(
  artifacts/ppo_marnie_v7_R0H0_full_noreplay_u455/checkpoints/update-0455.pt
  artifacts/ppo_marnie_v7_R1H0_full_hybridreplay_u455/checkpoints/update-0455.pt
  artifacts/ppo_marnie_v7_R0H1_lastblockheads_noreplay_u455/checkpoints/update-0455.pt
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

checkpoint_for() {
  local candidate_id="$1"
  local index
  for index in "${!candidate_ids[@]}"; do
    if [[ "${candidate_ids[$index]}" == "$candidate_id" ]]; then
      printf '%s\n' "${candidate_checkpoints[$index]}"
      return 0
    fi
  done
  echo "Unknown candidate: $candidate_id" >&2
  return 2
}

behavior_output() {
  printf '%s/%s_gate25_behavior.json\n' "$output_dir" "$1"
}

match_output() {
  printf '%s/%s_%s.json\n' "$output_dir" "$1" "$2"
}

run_behavior() {
  local candidate_id="$1"
  local checkpoint
  local output
  checkpoint="$(checkpoint_for "$candidate_id")"
  output="$(behavior_output "$candidate_id")"
  if [[ -f "$output" && "$resume_existing" == "1" ]]; then
    echo "Skipping existing report: $output"
    return 0
  fi
  echo "gate25 behavior: $candidate_id"
  "$python_bin" tools/evaluate_policy_bc.py \
    --checkpoint "$checkpoint" \
    --data "$gate_archive" \
    --split valid \
    --split-mode archive \
    --split-seed 20260723 \
    --batch-size 256 \
    --workers 8 \
    --prediction-order policy \
    --device "$device" \
    --compact \
    --json-output "$output"
}

run_match() {
  local candidate_id="$1"
  local label="$2"
  local opponent="$3"
  local opponent_deck="$4"
  local games="$5"
  local seed="$6"
  local opponent_order="$7"
  local checkpoint
  local output
  local order_args=()

  checkpoint="$(checkpoint_for "$candidate_id")"
  output="$(match_output "$candidate_id" "$label")"
  if [[ -f "$output" && "$resume_existing" == "1" ]]; then
    echo "Skipping existing report: $output"
    return 0
  fi
  if [[ "$opponent_order" == "canonical" ]]; then
    order_args+=(--opponent-canonical-order)
  elif [[ "$opponent_order" != "policy" ]]; then
    echo "Unknown opponent order: $opponent_order" >&2
    return 2
  fi

  echo "$label: $candidate_id"
  "$python_bin" tools/evaluate_ppo_head_to_head.py \
    --candidate "$checkpoint" \
    --opponent "$opponent" \
    --bc-checkpoint "$bc_checkpoint" \
    --candidate-deck "$marnie_deck" \
    --opponent-deck "$opponent_deck" \
    --games "$games" \
    --environments 16 \
    --max-game-decisions 1000 \
    --seed "$seed" \
    --device "$device" \
    "${order_args[@]}" \
    --output "$output"
}

for path in \
  "$bc_checkpoint" \
  "$gate_archive" \
  "$marnie_deck" \
  "$kang_deck" \
  "$v3_u440" \
  "$v3_u570" \
  "$kang_checkpoint" \
  "${candidate_checkpoints[@]}"; do
  require_file "$path"
done

planned_outputs=()
for candidate_id in "${candidate_ids[@]}"; do
  planned_outputs+=(
    "$(behavior_output "$candidate_id")"
    "$(match_output "$candidate_id" vs_v3u440_256_block1)"
    "$(match_output "$candidate_id" vs_v3u440_256_block2)"
    "$(match_output "$candidate_id" vs_v3u570_128)"
    "$(match_output "$candidate_id" vs_kang128)"
  )
done
if [[ "$resume_existing" != "1" ]]; then
  for path in "${planned_outputs[@]}"; do
    if [[ -e "$path" ]]; then
      echo "Refusing to overwrite existing report: $path" >&2
      echo "Set RESUME_EXISTING=1 to skip completed reports." >&2
      exit 1
    fi
  done
fi
mkdir -p "$output_dir"

# Behavior evaluation is deterministic over the episode-disjoint gate25 split.
for candidate_id in "${candidate_ids[@]}"; do
  run_behavior "$candidate_id"
done

# Interleave candidates rather than finishing one candidate at a time.  The
# second u440 block reverses the first block's order.  Across all four H2H
# rounds, each candidate occupies each temporal position at least once.
run_match R0H0_full_noreplay_u455 \
  vs_v3u440_256_block1 "$v3_u440" "$marnie_deck" 256 20260831 policy
run_match R1H0_full_hybridreplay_u455 \
  vs_v3u440_256_block1 "$v3_u440" "$marnie_deck" 256 20260831 policy
run_match R0H1_lastblockheads_noreplay_u455 \
  vs_v3u440_256_block1 "$v3_u440" "$marnie_deck" 256 20260831 policy

run_match R1H0_full_hybridreplay_u455 \
  vs_v3u570_128 "$v3_u570" "$marnie_deck" 128 20260901 policy
run_match R0H1_lastblockheads_noreplay_u455 \
  vs_v3u570_128 "$v3_u570" "$marnie_deck" 128 20260901 policy
run_match R0H0_full_noreplay_u455 \
  vs_v3u570_128 "$v3_u570" "$marnie_deck" 128 20260901 policy

run_match R0H1_lastblockheads_noreplay_u455 \
  vs_kang128 "$kang_checkpoint" "$kang_deck" 128 20260902 canonical
run_match R0H0_full_noreplay_u455 \
  vs_kang128 "$kang_checkpoint" "$kang_deck" 128 20260902 canonical
run_match R1H0_full_hybridreplay_u455 \
  vs_kang128 "$kang_checkpoint" "$kang_deck" 128 20260902 canonical

run_match R0H1_lastblockheads_noreplay_u455 \
  vs_v3u440_256_block2 "$v3_u440" "$marnie_deck" 256 20260903 policy
run_match R1H0_full_hybridreplay_u455 \
  vs_v3u440_256_block2 "$v3_u440" "$marnie_deck" 256 20260903 policy
run_match R0H0_full_noreplay_u455 \
  vs_v3u440_256_block2 "$v3_u440" "$marnie_deck" 256 20260903 policy

echo "Completed v7 u455 screen: $output_dir"

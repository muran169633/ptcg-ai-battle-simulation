#!/usr/bin/env bash
set -euo pipefail

# Immutable, interleaved external screen for the two locked v9 update-0455
# checkpoints.  This script evaluates only; it never packages or submits a
# model.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
output_dir="artifacts/ppo_marnie_v9_screen_u455"

fq_checkpoint="artifacts/ppo_marnie_v9_FQ_fixedquota_u455/checkpoints/update-0455.pt"
uq_checkpoint="artifacts/ppo_marnie_v9_UQ_betaucb_u455/checkpoints/update-0455.pt"
bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
gate_archive="data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
kang_deck="data/decks/kangaskhan_standard.csv"
v3_u440="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
v3_u570="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0570.pt"
v4_soup75="artifacts/ppo_marnie_v4_soups_u80_u100/alpha-075.pt"
kang_checkpoint="artifacts/bc_kangaskhan_orbit_v1/best.pt"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

checkpoint_for() {
  case "$1" in
    FQ) printf '%s\n' "$fq_checkpoint" ;;
    UQ) printf '%s\n' "$uq_checkpoint" ;;
    *)
      echo "Unknown candidate: $1" >&2
      return 2
      ;;
  esac
}

run_behavior() {
  local candidate_id="$1"
  local checkpoint
  checkpoint="$(checkpoint_for "$candidate_id")"
  echo "behavior valid25-hash20: $candidate_id"
  "$python_bin" tools/evaluate_policy_bc.py \
    --checkpoint "$checkpoint" \
    --data "$gate_archive" \
    --split valid \
    --split-mode archive \
    --batch-size 256 \
    --workers 8 \
    --prediction-order auto \
    --device "$device" \
    --compact \
    --progress-interval 0 \
    --json-output "$output_dir/${candidate_id}_behavior.json" \
    >"$output_dir/${candidate_id}_behavior.stdout.log"
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
  local order_args=()

  checkpoint="$(checkpoint_for "$candidate_id")"
  if [[ "$opponent_order" == "canonical" ]]; then
    order_args+=(--opponent-canonical-order)
  elif [[ "$opponent_order" != "policy" ]]; then
    echo "Unknown opponent order: $opponent_order" >&2
    return 2
  fi

  echo "$label: $candidate_id ($games games)"
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
    --output "$output_dir/${candidate_id}_${label}.json" \
    >"$output_dir/${candidate_id}_${label}.stdout.log"
}

for path in \
  "$fq_checkpoint" \
  "$uq_checkpoint" \
  "$bc_checkpoint" \
  "$gate_archive" \
  "$marnie_deck" \
  "$kang_deck" \
  "$v3_u440" \
  "$v3_u570" \
  "$v4_soup75" \
  "$kang_checkpoint"; do
  require_file "$path"
done

if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"

run_behavior FQ
run_behavior UQ

# Interleave the candidates.  The second u440 block reverses candidate order.
run_match FQ vs_v3u440_256_block1 "$v3_u440" "$marnie_deck" 256 20260931 policy
run_match UQ vs_v3u440_256_block1 "$v3_u440" "$marnie_deck" 256 20260932 policy

run_match FQ vs_v3u570_128 "$v3_u570" "$marnie_deck" 128 20260933 policy
run_match UQ vs_v3u570_128 "$v3_u570" "$marnie_deck" 128 20260934 policy

run_match UQ vs_kang128 "$kang_checkpoint" "$kang_deck" 128 20260935 canonical
run_match FQ vs_kang128 "$kang_checkpoint" "$kang_deck" 128 20260936 canonical

run_match FQ vs_v4soup75_128 "$v4_soup75" "$marnie_deck" 128 20260937 policy
run_match UQ vs_v4soup75_128 "$v4_soup75" "$marnie_deck" 128 20260938 policy

run_match UQ vs_freshbc128 "$bc_checkpoint" "$marnie_deck" 128 20260939 canonical
run_match FQ vs_freshbc128 "$bc_checkpoint" "$marnie_deck" 128 20260940 canonical

run_match UQ vs_v3u440_256_block2 "$v3_u440" "$marnie_deck" 256 20260941 policy
run_match FQ vs_v3u440_256_block2 "$v3_u440" "$marnie_deck" 256 20260942 policy

"$python_bin" - "$output_dir" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


output_dir = Path(sys.argv[1])
thresholds = {
    "behavior": {
        "set_exact_accuracy_min": 0.6980750789381359,
        "ordered_exact_accuracy_min": 0.6853865045023974,
        "value_win_accuracy_min": 0.6631364752660507,
        "count_accuracy_min": 0.99,
    },
    "head_to_head_min_wins": {
        "vs_v3u440_512": 269,
        "vs_v3u570_128": 57,
        "vs_kang128": 82,
        "vs_v4soup75_128": 64,
        "vs_freshbc128": 64,
    },
    "invalid_games_max": 0,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_evaluation(path: Path) -> dict:
    evaluation = read_json(path)["evaluation"]
    return {
        key: evaluation[key]
        for key in (
            "valid_games",
            "wins",
            "losses",
            "draws",
            "invalid_games",
            "win_rate",
            "wilson_95_low",
            "wilson_95_high",
        )
    }


def combine_evaluations(first: dict, second: dict) -> dict:
    valid = first["valid_games"] + second["valid_games"]
    wins = first["wins"] + second["wins"]
    losses = first["losses"] + second["losses"]
    draws = first["draws"] + second["draws"]
    invalid = first["invalid_games"] + second["invalid_games"]
    return {
        "valid_games": valid,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "invalid_games": invalid,
        "win_rate": wins / valid if valid else None,
        "blocks": [first, second],
    }


candidates = {}
for candidate_id in ("FQ", "UQ"):
    behavior_result = read_json(output_dir / f"{candidate_id}_behavior.json")
    metrics = behavior_result["metrics"]
    behavior = {
        "rows": metrics["rows"],
        "set_exact_accuracy": metrics["set_exact_accuracy"],
        "ordered_exact_accuracy": metrics["ordered_exact_accuracy"],
        "hybrid_order_exact_accuracy": metrics["hybrid_order_exact_accuracy"],
        "value_win_accuracy": metrics["value_win_accuracy"],
        "count_accuracy": metrics["count_accuracy"],
    }
    behavior_gates = {
        "set_exact_accuracy": (
            behavior["set_exact_accuracy"]
            >= thresholds["behavior"]["set_exact_accuracy_min"]
        ),
        "ordered_exact_accuracy": (
            behavior["ordered_exact_accuracy"]
            >= thresholds["behavior"]["ordered_exact_accuracy_min"]
        ),
        "value_win_accuracy": (
            behavior["value_win_accuracy"]
            >= thresholds["behavior"]["value_win_accuracy_min"]
        ),
        "count_accuracy": (
            behavior["count_accuracy"]
            >= thresholds["behavior"]["count_accuracy_min"]
        ),
    }

    v3_block1 = compact_evaluation(
        output_dir / f"{candidate_id}_vs_v3u440_256_block1.json"
    )
    v3_block2 = compact_evaluation(
        output_dir / f"{candidate_id}_vs_v3u440_256_block2.json"
    )
    matches = {
        "vs_v3u440_512": combine_evaluations(v3_block1, v3_block2),
        "vs_v3u570_128": compact_evaluation(
            output_dir / f"{candidate_id}_vs_v3u570_128.json"
        ),
        "vs_kang128": compact_evaluation(
            output_dir / f"{candidate_id}_vs_kang128.json"
        ),
        "vs_v4soup75_128": compact_evaluation(
            output_dir / f"{candidate_id}_vs_v4soup75_128.json"
        ),
        "vs_freshbc128": compact_evaluation(
            output_dir / f"{candidate_id}_vs_freshbc128.json"
        ),
    }
    match_gates = {
        label: result["wins"]
        >= thresholds["head_to_head_min_wins"][label]
        for label, result in matches.items()
    }
    complete_requested_games = (
        matches["vs_v3u440_512"]["valid_games"] == 512
        and all(
            matches[label]["valid_games"] == 128
            for label in (
                "vs_v3u570_128",
                "vs_kang128",
                "vs_v4soup75_128",
                "vs_freshbc128",
            )
        )
    )
    all_invalid_zero = all(
        result["invalid_games"] == 0 for result in matches.values()
    )
    all_absolute_gates = (
        all(behavior_gates.values())
        and all(match_gates.values())
        and complete_requested_games
        and all_invalid_zero
    )
    candidates[candidate_id] = {
        "checkpoint": behavior_result["checkpoint"],
        "checkpoint_sha256": behavior_result["checkpoint_sha256"],
        "behavior": behavior,
        "behavior_gates": behavior_gates,
        "matches": matches,
        "match_gates": match_gates,
        "complete_requested_games": complete_requested_games,
        "all_invalid_zero": all_invalid_zero,
        "all_absolute_gates": all_absolute_gates,
        "sampler_mechanism_gate": (
            {
                "applicable": False,
                "passed": None,
                "reason": "fixed-quota control",
            }
            if candidate_id == "FQ"
            else {
                "applicable": True,
                "passed": False,
                "observed_max_half_l1_shift": 4,
                "required_shift": 6,
                "reason": "adaptive allocation did not move enough quota",
            }
        ),
    }

paired_needed = any(
    candidate["all_absolute_gates"] for candidate in candidates.values()
)
summary = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "screen": "v9 update-0455 immutable interleaved absolute screen",
    "thresholds": thresholds,
    "candidates": candidates,
    "paired_needed": paired_needed,
    "paired_rule": {
        "games": 512,
        "candidate": "UQ",
        "opponent": "FQ",
        "lock_uq_min_wins": 269,
        "lock_fq_max_uq_wins": 243,
        "uncertain_uq_wins": [244, 268],
        "lock_fq_requires_fq_absolute_gate": True,
    },
    "submission_or_packaging_performed": False,
}
(output_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Completed v9 update-0455 screen: $output_dir"

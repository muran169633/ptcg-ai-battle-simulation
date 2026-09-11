#!/usr/bin/env bash
set -euo pipefail

# Pre-registered v13 diagnostic: interpolate the complementary v9 UQ and FQ
# update-0455 endpoints at three fixed coefficients, then apply the existing
# absolute behavior/H2H gates. This script never packages or submits a model.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v13_uq_fq_soup_screen_u455}"
checkpoint_dir="$output_dir/checkpoints"

uq_checkpoint="artifacts/ppo_marnie_v9_UQ_betaucb_u455/checkpoints/update-0455.pt"
fq_checkpoint="artifacts/ppo_marnie_v9_FQ_fixedquota_u455/checkpoints/update-0455.pt"
bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
gate_archive="data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
kang_deck="data/decks/kangaskhan_standard.csv"
v3_u440="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
v3_u570="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0570.pt"
v4_soup75="artifacts/ppo_marnie_v4_soups_u80_u100/alpha-075.pt"
kang_checkpoint="artifacts/bc_kangaskhan_orbit_v1/best.pt"

candidate_ids=(fq025 fq050 fq075)
candidate_alphas=(0.25 0.50 0.75)

for path in \
  "$uq_checkpoint" \
  "$fq_checkpoint" \
  "$bc_checkpoint" \
  "$gate_archive" \
  "$marnie_deck" \
  "$kang_deck" \
  "$v3_u440" \
  "$v3_u570" \
  "$v4_soup75" \
  "$kang_checkpoint"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi
mkdir -p "$checkpoint_dir"

checkpoint_for() {
  local candidate_id="$1"
  printf '%s/%s.pt\n' "$checkpoint_dir" "$candidate_id"
}

for index in "${!candidate_ids[@]}"; do
  candidate_id="${candidate_ids[$index]}"
  alpha="${candidate_alphas[$index]}"
  checkpoint="$(checkpoint_for "$candidate_id")"
  echo "build soup: $candidate_id (FQ weight $alpha)"
  "$python_bin" tools/interpolate_ppo_checkpoints.py \
    --checkpoint-a "$uq_checkpoint" \
    --checkpoint-b "$fq_checkpoint" \
    --alpha "$alpha" \
    --output "$checkpoint" \
    >"$output_dir/${candidate_id}_build.json"
done

# This manifest is written before any candidate evaluation. It freezes the
# candidate grid, thresholds, cascade, and all evaluation seeds.
"$python_bin" - "$output_dir" "$uq_checkpoint" "$fq_checkpoint" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(sys.argv[1])
uq = Path(sys.argv[2])
fq = Path(sys.argv[3])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "v13 UQ/FQ update-0455 pre-registered policy-soup screen",
    "formula": "(1 - alpha) * UQ + alpha * FQ",
    "candidates": {
        "fq025": {"alpha_fq": 0.25},
        "fq050": {"alpha_fq": 0.50},
        "fq075": {"alpha_fq": 0.75},
    },
    "sources": {
        "UQ": {"path": str(uq.resolve()), "sha256": sha256(uq)},
        "FQ": {"path": str(fq.resolve()), "sha256": sha256(fq)},
    },
    "thresholds": {
        "behavior": {
            "set_exact_accuracy_min": 0.6980750789381359,
            "ordered_exact_accuracy_min": 0.6853865045023974,
            "value_win_accuracy_min": 0.6631364752660507,
            "count_accuracy_min": 0.99,
        },
        "head_to_head_min_wins": {
            "vs_freshbc128": 64,
            "vs_v3u440_512": 269,
            "vs_v3u570_128": 57,
            "vs_kang128": 82,
            "vs_v4soup75_128": 64,
        },
        "invalid_games_max": 0,
    },
    "cascade": {
        "early": [
            "behavior_valid25_hash20",
            "vs_freshbc128",
        ],
        "broad_only_if_early_passes": [
            "vs_v3u440_256_block1",
            "vs_v3u570_128",
            "vs_kang128",
            "vs_v4soup75_128",
            "vs_v3u440_256_block2",
        ],
        "strict_pass_action": (
            "independent confirmation only; no automatic packaging or submission"
        ),
    },
    "seeds": {
        "fq025": {
            "freshbc": 20261031,
            "v3u440_block1": 20261041,
            "v3u570": 20261051,
            "kang": 20261061,
            "v4": 20261071,
            "v3u440_block2": 20261081,
        },
        "fq050": {
            "freshbc": 20261032,
            "v3u440_block1": 20261042,
            "v3u570": 20261052,
            "kang": 20261062,
            "v4": 20261072,
            "v3u440_block2": 20261082,
        },
        "fq075": {
            "freshbc": 20261033,
            "v3u440_block1": 20261043,
            "v3u570": 20261053,
            "kang": 20261063,
            "v4": 20261073,
            "v3u440_block2": 20261083,
        },
    },
    "selection_rule": (
        "Every absolute gate must pass. If more than one candidate passes, "
        "do not select on these results; run a fresh independent confirmation."
    ),
    "packaging_or_submission_authorized": False,
}
(root / "preregistration.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

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

for candidate_id in "${candidate_ids[@]}"; do
  run_behavior "$candidate_id"
done

for index in "${!candidate_ids[@]}"; do
  candidate_id="${candidate_ids[$index]}"
  seed=$((20261031 + index))
  run_match \
    "$candidate_id" vs_freshbc128 "$bc_checkpoint" "$marnie_deck" \
    128 "$seed" canonical
done

"$python_bin" - "$output_dir" "${candidate_ids[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
candidate_ids = sys.argv[2:]
manifest = json.loads((root / "preregistration.json").read_text())
thresholds = manifest["thresholds"]
candidates = {}
for candidate_id in candidate_ids:
    behavior_doc = json.loads(
        (root / f"{candidate_id}_behavior.json").read_text()
    )
    metrics = behavior_doc["metrics"]
    match = json.loads(
        (root / f"{candidate_id}_vs_freshbc128.json").read_text()
    )["evaluation"]
    behavior = {
        key: metrics[key]
        for key in (
            "rows",
            "set_exact_accuracy",
            "ordered_exact_accuracy",
            "hybrid_order_exact_accuracy",
            "value_win_accuracy",
            "count_accuracy",
        )
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
    fresh_bc_gate = (
        match["valid_games"] == 128
        and match["wins"]
        >= thresholds["head_to_head_min_wins"]["vs_freshbc128"]
        and match["invalid_games"] <= thresholds["invalid_games_max"]
    )
    candidates[candidate_id] = {
        "checkpoint": behavior_doc["checkpoint"],
        "checkpoint_sha256": behavior_doc["checkpoint_sha256"],
        "behavior": behavior,
        "behavior_gates": behavior_gates,
        "fresh_bc": match,
        "fresh_bc_gate": fresh_bc_gate,
        "passed": all(behavior_gates.values()) and fresh_bc_gate,
    }
summary = {
    "stage": "early_behavior_and_fresh_bc",
    "thresholds": thresholds,
    "candidates": candidates,
}
(root / "early_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

eligible=()
for candidate_id in "${candidate_ids[@]}"; do
  if "$python_bin" - "$output_dir/early_summary.json" "$candidate_id" <<'PY'
import json
import sys

summary = json.loads(open(sys.argv[1], encoding="utf-8").read())
raise SystemExit(0 if summary["candidates"][sys.argv[2]]["passed"] else 1)
PY
  then
    eligible+=("$candidate_id")
  fi
done

if [[ "${#eligible[@]}" -eq 0 ]]; then
  "$python_bin" - "$output_dir" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
early = json.loads((root / "early_summary.json").read_text())
summary = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "screen": "v13 UQ/FQ policy-soup cascaded absolute screen",
    "status": "all_candidates_rejected_at_early_gate",
    "early": early,
    "candidates": {
        candidate_id: {
            **candidate,
            "broad_screen_run": False,
            "all_absolute_gates": False,
        }
        for candidate_id, candidate in early["candidates"].items()
    },
    "strict_passes": [],
    "independent_confirmation_required": False,
    "submission_or_packaging_performed": False,
}
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
  echo "All v13 soups stopped at the pre-registered early gate"
  exit 0
fi

# Interleave all eligible candidates by opponent and put the second v3 block
# last, matching the established v9 screen structure.
for index in "${!eligible[@]}"; do
  candidate_id="${eligible[$index]}"
  case "$candidate_id" in
    fq025) seed=20261041 ;;
    fq050) seed=20261042 ;;
    fq075) seed=20261043 ;;
    *) echo "Unknown eligible candidate: $candidate_id" >&2; exit 2 ;;
  esac
  run_match \
    "$candidate_id" vs_v3u440_256_block1 "$v3_u440" "$marnie_deck" \
    256 "$seed" policy
done

for candidate_id in "${eligible[@]}"; do
  case "$candidate_id" in
    fq025) seed=20261051 ;;
    fq050) seed=20261052 ;;
    fq075) seed=20261053 ;;
    *) echo "Unknown eligible candidate: $candidate_id" >&2; exit 2 ;;
  esac
  run_match \
    "$candidate_id" vs_v3u570_128 "$v3_u570" "$marnie_deck" \
    128 "$seed" policy
done

for candidate_id in "${eligible[@]}"; do
  case "$candidate_id" in
    fq025) seed=20261061 ;;
    fq050) seed=20261062 ;;
    fq075) seed=20261063 ;;
    *) echo "Unknown eligible candidate: $candidate_id" >&2; exit 2 ;;
  esac
  run_match \
    "$candidate_id" vs_kang128 "$kang_checkpoint" "$kang_deck" \
    128 "$seed" canonical
done

for candidate_id in "${eligible[@]}"; do
  case "$candidate_id" in
    fq025) seed=20261071 ;;
    fq050) seed=20261072 ;;
    fq075) seed=20261073 ;;
    *) echo "Unknown eligible candidate: $candidate_id" >&2; exit 2 ;;
  esac
  run_match \
    "$candidate_id" vs_v4soup75_128 "$v4_soup75" "$marnie_deck" \
    128 "$seed" policy
done

for candidate_id in "${eligible[@]}"; do
  case "$candidate_id" in
    fq025) seed=20261081 ;;
    fq050) seed=20261082 ;;
    fq075) seed=20261083 ;;
    *) echo "Unknown eligible candidate: $candidate_id" >&2; exit 2 ;;
  esac
  run_match \
    "$candidate_id" vs_v3u440_256_block2 "$v3_u440" "$marnie_deck" \
    256 "$seed" policy
done

"$python_bin" - "$output_dir" "${eligible[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(sys.argv[1])
eligible = set(sys.argv[2:])
early = json.loads((root / "early_summary.json").read_text())
thresholds = early["thresholds"]


def evaluation(candidate_id: str, label: str) -> dict:
    return json.loads(
        (root / f"{candidate_id}_{label}.json").read_text()
    )["evaluation"]


def compact(result: dict) -> dict:
    return {
        key: result[key]
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


candidates = {}
strict_passes = []
for candidate_id, early_candidate in early["candidates"].items():
    if candidate_id not in eligible:
        candidates[candidate_id] = {
            **early_candidate,
            "broad_screen_run": False,
            "all_absolute_gates": False,
        }
        continue

    block1 = compact(evaluation(candidate_id, "vs_v3u440_256_block1"))
    block2 = compact(evaluation(candidate_id, "vs_v3u440_256_block2"))
    v3_u440 = {
        "valid_games": block1["valid_games"] + block2["valid_games"],
        "wins": block1["wins"] + block2["wins"],
        "losses": block1["losses"] + block2["losses"],
        "draws": block1["draws"] + block2["draws"],
        "invalid_games": block1["invalid_games"] + block2["invalid_games"],
        "win_rate": (
            (block1["wins"] + block2["wins"])
            / (block1["valid_games"] + block2["valid_games"])
        ),
        "blocks": [block1, block2],
    }
    matches = {
        "vs_freshbc128": early_candidate["fresh_bc"],
        "vs_v3u440_512": v3_u440,
        "vs_v3u570_128": compact(evaluation(candidate_id, "vs_v3u570_128")),
        "vs_kang128": compact(evaluation(candidate_id, "vs_kang128")),
        "vs_v4soup75_128": compact(
            evaluation(candidate_id, "vs_v4soup75_128")
        ),
    }
    requested_games = {
        "vs_freshbc128": 128,
        "vs_v3u440_512": 512,
        "vs_v3u570_128": 128,
        "vs_kang128": 128,
        "vs_v4soup75_128": 128,
    }
    match_gates = {
        label: (
            result["valid_games"] == requested_games[label]
            and result["wins"]
            >= thresholds["head_to_head_min_wins"][label]
            and result["invalid_games"] <= thresholds["invalid_games_max"]
        )
        for label, result in matches.items()
    }
    all_absolute_gates = (
        early_candidate["passed"] and all(match_gates.values())
    )
    candidates[candidate_id] = {
        **early_candidate,
        "broad_screen_run": True,
        "matches": matches,
        "match_gates": match_gates,
        "all_absolute_gates": all_absolute_gates,
    }
    if all_absolute_gates:
        strict_passes.append(candidate_id)

summary = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "screen": "v13 UQ/FQ policy-soup cascaded absolute screen",
    "status": (
        "strict_pass_needs_independent_confirmation"
        if strict_passes
        else "all_candidates_rejected"
    ),
    "thresholds": thresholds,
    "candidates": candidates,
    "strict_passes": strict_passes,
    "independent_confirmation_required": bool(strict_passes),
    "selection_rule": (
        "No candidate is promoted from this screen alone. A strict pass "
        "requires a fresh independent confirmation; multiple passes are not "
        "ranked using these screen results."
    ),
    "submission_or_packaging_performed": False,
}
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Completed v13 UQ/FQ soup screen: $output_dir"

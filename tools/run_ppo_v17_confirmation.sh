#!/usr/bin/env bash
set -euo pipefail

# Complete independent confirmation for the single locked v17 endpoint.
# Apart from the behavior safety gate, there is no preliminary performance
# screen. All eight core blocks run before their results are aggregated.
# This script never packages or submits a model.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
train_dir="artifacts/ppo_marnie_v17_guard_pcgrad_varred_u458"
candidate="$train_dir/checkpoints/update-0458.pt"
candidate_lock="$train_dir/candidate_lock.json"
output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v17_confirmation_u458}"
bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
gate_archive="data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
kang_deck="data/decks/kangaskhan_standard.csv"
v3_u440="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
v3_u570="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0570.pt"
v4_soup75="artifacts/ppo_marnie_v4_soups_u80_u100/alpha-075.pt"
kang_checkpoint="artifacts/bc_kangaskhan_orbit_v1/best.pt"

for path in \
  "$candidate" \
  "$candidate_lock" \
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
mkdir -p "$output_dir"

"$python_bin" - "$candidate" "$candidate_lock" "$output_dir" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


candidate = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
root = Path(sys.argv[3])
lock = json.loads(lock_path.read_text(encoding="utf-8"))
actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
if Path(lock["candidate"]).resolve() != candidate.resolve():
    raise RuntimeError("Candidate path does not match the post-training lock")
if lock["candidate_sha256"] != actual:
    raise RuntimeError(
        "Candidate SHA does not match the post-training lock: "
        f"locked={lock['candidate_sha256']} actual={actual}"
    )
manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "candidate": str(candidate.resolve()),
    "candidate_sha256": actual,
    "selection_rule": (
        "Only the behavior safety gate may stop evaluation. If it passes, all "
        "eight interleaved core blocks run before aggregation; no block result "
        "selects a checkpoint or changes the remaining evaluation."
    ),
    "behavior_thresholds": {
        "set_exact_accuracy_min": 0.6980750789381359,
        "ordered_exact_accuracy_min": 0.6853865045023974,
        "value_win_accuracy_min": 0.6631364752660507,
        "count_accuracy_min": 0.99,
    },
    "core_blocks": [
        {"label": "vs_v3u440_block1", "games": 512, "seed": 20261161},
        {"label": "vs_freshbc_block1", "games": 512, "seed": 20261162},
        {"label": "vs_v3u440_block2", "games": 512, "seed": 20261163},
        {"label": "vs_freshbc_block2", "games": 512, "seed": 20261164},
        {"label": "vs_v3u440_block3", "games": 512, "seed": 20261165},
        {"label": "vs_freshbc_block3", "games": 512, "seed": 20261166},
        {"label": "vs_v3u440_block4", "games": 512, "seed": 20261167},
        {"label": "vs_freshbc_block4", "games": 512, "seed": 20261168},
    ],
    "core_thresholds": {
        "valid_games_each": 2048,
        "minimum_wins_each": 1076,
        "wilson_95_low_strictly_above": 0.5,
        "invalid_games_max": 0,
    },
    "secondary_blocks_if_core_passes": [
        {
            "label": "vs_v3u570_256",
            "games": 256,
            "seed": 20261169,
            "minimum_wins": 114,
        },
        {
            "label": "vs_kang_256",
            "games": 256,
            "seed": 20261170,
            "minimum_wins": 164,
        },
        {
            "label": "vs_v4soup75_256",
            "games": 256,
            "seed": 20261171,
            "minimum_wins": 128,
        },
    ],
    "submission_or_packaging_authorized": False,
}
(root / "preregistration.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "behavior safety gate"
"$python_bin" tools/evaluate_policy_bc.py \
  --checkpoint "$candidate" \
  --data "$gate_archive" \
  --split valid \
  --split-mode archive \
  --batch-size 256 \
  --workers 8 \
  --prediction-order auto \
  --device "$device" \
  --compact \
  --progress-interval 0 \
  --json-output "$output_dir/behavior.json" \
  >"$output_dir/behavior.stdout.log"

if ! "$python_bin" - "$output_dir" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(sys.argv[1])
manifest = json.loads((root / "preregistration.json").read_text())
doc = json.loads((root / "behavior.json").read_text())
metrics = doc["metrics"]
thresholds = manifest["behavior_thresholds"]
gates = {
    "set": (
        metrics["set_exact_accuracy"]
        >= thresholds["set_exact_accuracy_min"]
    ),
    "ordered": (
        metrics["ordered_exact_accuracy"]
        >= thresholds["ordered_exact_accuracy_min"]
    ),
    "value": (
        metrics["value_win_accuracy"]
        >= thresholds["value_win_accuracy_min"]
    ),
    "count": (
        metrics["count_accuracy"]
        >= thresholds["count_accuracy_min"]
    ),
}
result = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "candidate": manifest["candidate"],
    "candidate_sha256": manifest["candidate_sha256"],
    "status": "behavior_passed" if all(gates.values()) else "behavior_failed",
    "behavior": {
        key: metrics[key]
        for key in (
            "rows",
            "set_exact_accuracy",
            "ordered_exact_accuracy",
            "hybrid_order_exact_accuracy",
            "value_win_accuracy",
            "count_accuracy",
        )
    },
    "behavior_gates": gates,
    "passed": all(gates.values()),
    "submission_or_packaging_performed": False,
}
(root / "behavior_gate.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
if not result["passed"]:
    (root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["passed"] else 1)
PY
then
  echo "v17 stopped at the predeclared behavior safety gate"
  exit 0
fi

run_match() {
  local label="$1"
  local opponent="$2"
  local opponent_deck="$3"
  local games="$4"
  local seed="$5"
  shift 5
  echo "$label ($games games)"
  "$python_bin" tools/evaluate_ppo_head_to_head.py \
    --candidate "$candidate" \
    --opponent "$opponent" \
    --bc-checkpoint "$bc_checkpoint" \
    --candidate-deck "$marnie_deck" \
    --opponent-deck "$opponent_deck" \
    --games "$games" \
    --environments 16 \
    --max-game-decisions 1000 \
    --seed "$seed" \
    --device "$device" \
    "$@" \
    --output "$output_dir/$label.json" \
    >"$output_dir/$label.stdout.log"
}

run_match vs_v3u440_block1 "$v3_u440" "$marnie_deck" 512 20261161
run_match vs_freshbc_block1 "$bc_checkpoint" "$marnie_deck" 512 20261162 \
  --opponent-canonical-order
run_match vs_v3u440_block2 "$v3_u440" "$marnie_deck" 512 20261163
run_match vs_freshbc_block2 "$bc_checkpoint" "$marnie_deck" 512 20261164 \
  --opponent-canonical-order
run_match vs_v3u440_block3 "$v3_u440" "$marnie_deck" 512 20261165
run_match vs_freshbc_block3 "$bc_checkpoint" "$marnie_deck" 512 20261166 \
  --opponent-canonical-order
run_match vs_v3u440_block4 "$v3_u440" "$marnie_deck" 512 20261167
run_match vs_freshbc_block4 "$bc_checkpoint" "$marnie_deck" 512 20261168 \
  --opponent-canonical-order

if "$python_bin" - "$output_dir" <<'PY'
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(sys.argv[1])
manifest = json.loads((root / "preregistration.json").read_text())
behavior = json.loads((root / "behavior_gate.json").read_text())


def evaluation(label: str) -> dict:
    return json.loads((root / f"{label}.json").read_text())["evaluation"]


def combine(labels: list[str]) -> dict:
    blocks = [evaluation(label) for label in labels]
    valid = sum(block["valid_games"] for block in blocks)
    wins = sum(block["wins"] for block in blocks)
    losses = sum(block["losses"] for block in blocks)
    draws = sum(block["draws"] for block in blocks)
    invalid = sum(block["invalid_games"] for block in blocks)
    z = 1.959963984540054
    proportion = wins / valid
    denominator = 1.0 + z * z / valid
    center = (proportion + z * z / (2.0 * valid)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / valid
            + z * z / (4.0 * valid * valid)
        )
        / denominator
    )
    return {
        "valid_games": valid,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "invalid_games": invalid,
        "win_rate": proportion,
        "wilson_95_low": center - margin,
        "wilson_95_high": center + margin,
        "blocks": blocks,
    }


v3 = combine([f"vs_v3u440_block{index}" for index in range(1, 5)])
bc = combine([f"vs_freshbc_block{index}" for index in range(1, 5)])
thresholds = manifest["core_thresholds"]


def gates(result: dict) -> dict:
    return {
        "complete": result["valid_games"] == thresholds["valid_games_each"],
        "wins": result["wins"] >= thresholds["minimum_wins_each"],
        "wilson": (
            result["wilson_95_low"]
            > thresholds["wilson_95_low_strictly_above"]
        ),
        "invalid": (
            result["invalid_games"] <= thresholds["invalid_games_max"]
        ),
    }


core_gates = {
    "vs_v3u440": gates(v3),
    "vs_freshbc": gates(bc),
}
core_passed = all(
    all(group.values())
    for group in core_gates.values()
)
summary = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "candidate": manifest["candidate"],
    "candidate_sha256": manifest["candidate_sha256"],
    "behavior": behavior["behavior"],
    "behavior_gates": behavior["behavior_gates"],
    "core_matches": {
        "vs_v3u440_2048": v3,
        "vs_freshbc_2048": bc,
    },
    "core_gates": core_gates,
    "core_passed": core_passed,
    "passed": False,
    "status": (
        "core_passed_secondary_pending"
        if core_passed
        else "core_confirmation_failed"
    ),
    "submission_or_packaging_performed": False,
}
(root / "core_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
if not core_passed:
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if core_passed else 1)
PY
then
  core_status=0
else
  core_status=$?
fi

if [[ "$core_status" -ne 0 ]]; then
  echo "v17 stopped after the complete predeclared core confirmation"
  exit 0
fi

run_match vs_v3u570_256 "$v3_u570" "$marnie_deck" 256 20261169
run_match vs_kang_256 "$kang_checkpoint" "$kang_deck" 256 20261170 \
  --opponent-canonical-order
run_match vs_v4soup75_256 "$v4_soup75" "$marnie_deck" 256 20261171

"$python_bin" - "$output_dir" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(sys.argv[1])
manifest = json.loads((root / "preregistration.json").read_text())
summary = json.loads((root / "core_summary.json").read_text())


def evaluation(label: str) -> dict:
    return json.loads((root / f"{label}.json").read_text())["evaluation"]


matches = {
    "vs_v3u570_256": evaluation("vs_v3u570_256"),
    "vs_kang_256": evaluation("vs_kang_256"),
    "vs_v4soup75_256": evaluation("vs_v4soup75_256"),
}
minimum_wins = {
    item["label"]: item["minimum_wins"]
    for item in manifest["secondary_blocks_if_core_passes"]
}
gates = {
    label: {
        "complete": result["valid_games"] == 256,
        "wins": result["wins"] >= minimum_wins[label],
        "invalid": result["invalid_games"] == 0,
    }
    for label, result in matches.items()
}
passed = summary["core_passed"] and all(
    all(group.values())
    for group in gates.values()
)
summary.update(
    {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "secondary_matches": matches,
        "secondary_gates": gates,
        "passed": passed,
        "status": (
            "independent_confirmation_passed"
            if passed
            else "secondary_confirmation_failed"
        ),
    }
)
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Completed v17 independent confirmation: $output_dir"

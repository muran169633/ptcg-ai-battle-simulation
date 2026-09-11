#!/usr/bin/env bash
set -euo pipefail

# Independent confirmation for the single locked v16 strict-screen passer.
# The screen games are not reused. Two fresh 512-game blocks are interleaved
# for each core opponent. This script never packages or submits a model.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v16_confirmation_u458}"
candidate="artifacts/ppo_marnie_v16_guard_pcgrad_heads_u458/checkpoints/update-0458.pt"
expected_sha256="f2e7009fa619e46911c581b1dc54734566061c8b55c7ee0a2894de0f4475b8ba"
bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
v3_checkpoint="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"

for path in \
  "$candidate" \
  "$bc_checkpoint" \
  "$v3_checkpoint" \
  "$marnie_deck"; do
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

"$python_bin" - "$candidate" "$expected_sha256" "$output_dir" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


candidate = Path(sys.argv[1])
expected = sys.argv[2]
root = Path(sys.argv[3])
actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError(
        f"Locked candidate SHA mismatch: expected={expected}, actual={actual}"
    )
manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "candidate": str(candidate.resolve()),
    "candidate_sha256": actual,
    "screen": (
        "artifacts/ppo_marnie_v16_screen_u458/summary.json"
    ),
    "blocks": [
        {
            "label": "vs_v3u440_block1",
            "games": 512,
            "seed": 20261141,
            "opponent_order": "policy",
        },
        {
            "label": "vs_freshbc_block1",
            "games": 512,
            "seed": 20261142,
            "opponent_order": "canonical",
        },
        {
            "label": "vs_v3u440_block2",
            "games": 512,
            "seed": 20261143,
            "opponent_order": "policy",
        },
        {
            "label": "vs_freshbc_block2",
            "games": 512,
            "seed": 20261144,
            "opponent_order": "canonical",
        },
    ],
    "thresholds": {
        "vs_v3u440_1024_min_wins": 538,
        "vs_freshbc_1024_min_wins": 512,
        "invalid_games_max": 0,
        "valid_games_each": 1024,
    },
    "selection_rule": (
        "Both independent absolute gates must pass. Screen games are excluded."
    ),
    "submission_or_packaging_authorized": False,
}
(root / "preregistration.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

run_match() {
  local label="$1"
  local opponent="$2"
  local seed="$3"
  local opponent_order="$4"
  local order_args=()
  if [[ "$opponent_order" == "canonical" ]]; then
    order_args+=(--opponent-canonical-order)
  elif [[ "$opponent_order" != "policy" ]]; then
    echo "Unknown opponent order: $opponent_order" >&2
    return 2
  fi
  echo "$label (512 games)"
  "$python_bin" tools/evaluate_ppo_head_to_head.py \
    --candidate "$candidate" \
    --opponent "$opponent" \
    --bc-checkpoint "$bc_checkpoint" \
    --candidate-deck "$marnie_deck" \
    --opponent-deck "$marnie_deck" \
    --games 512 \
    --environments 16 \
    --max-game-decisions 1000 \
    --seed "$seed" \
    --device "$device" \
    "${order_args[@]}" \
    --output "$output_dir/$label.json" \
    >"$output_dir/$label.stdout.log"
}

run_match vs_v3u440_block1 "$v3_checkpoint" 20261141 policy
run_match vs_freshbc_block1 "$bc_checkpoint" 20261142 canonical
run_match vs_v3u440_block2 "$v3_checkpoint" 20261143 policy
run_match vs_freshbc_block2 "$bc_checkpoint" 20261144 canonical

"$python_bin" - "$output_dir" <<'PY'
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(sys.argv[1])
manifest = json.loads((root / "preregistration.json").read_text())


def evaluation(label: str) -> dict:
    return json.loads((root / f"{label}.json").read_text())["evaluation"]


def combine(first: dict, second: dict) -> dict:
    valid = first["valid_games"] + second["valid_games"]
    wins = first["wins"] + second["wins"]
    losses = first["losses"] + second["losses"]
    draws = first["draws"] + second["draws"]
    invalid = first["invalid_games"] + second["invalid_games"]
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
        "blocks": [first, second],
    }


v3 = combine(
    evaluation("vs_v3u440_block1"),
    evaluation("vs_v3u440_block2"),
)
bc = combine(
    evaluation("vs_freshbc_block1"),
    evaluation("vs_freshbc_block2"),
)
thresholds = manifest["thresholds"]
gates = {
    "v3_complete": v3["valid_games"] == thresholds["valid_games_each"],
    "v3_wins": v3["wins"] >= thresholds["vs_v3u440_1024_min_wins"],
    "v3_invalid": (
        v3["invalid_games"] <= thresholds["invalid_games_max"]
    ),
    "freshbc_complete": bc["valid_games"] == thresholds["valid_games_each"],
    "freshbc_wins": (
        bc["wins"] >= thresholds["vs_freshbc_1024_min_wins"]
    ),
    "freshbc_invalid": (
        bc["invalid_games"] <= thresholds["invalid_games_max"]
    ),
}
summary = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "candidate": manifest["candidate"],
    "candidate_sha256": manifest["candidate_sha256"],
    "thresholds": thresholds,
    "matches": {
        "vs_v3u440_1024": v3,
        "vs_freshbc_1024": bc,
    },
    "gates": gates,
    "passed": all(gates.values()),
    "status": (
        "independent_confirmation_passed"
        if all(gates.values())
        else "independent_confirmation_failed"
    ),
    "submission_or_packaging_performed": False,
}
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Completed v16 independent confirmation: $output_dir"

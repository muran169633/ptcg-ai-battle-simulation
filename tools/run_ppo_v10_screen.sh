#!/usr/bin/env bash
set -euo pipefail

# Cascaded immutable screen for one locked bridge update-0461 candidate.
# Behavior and the targeted fresh-BC repair are checked first; the broader
# screen runs only if both early gates pass.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v10_screen_u461}"
candidate="${CANDIDATE:-artifacts/ppo_marnie_v10_UQ_bcbridge_u461/checkpoints/update-0461.pt}"
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

echo "behavior valid25-hash20"
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

run_match vs_freshbc128 "$bc_checkpoint" "$marnie_deck" 128 20261011 \
  --opponent-canonical-order

if ! "$python_bin" - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
m = json.loads((root / "behavior.json").read_text())["metrics"]
e = json.loads((root / "vs_freshbc128.json").read_text())["evaluation"]
gates = {
    "behavior_set": m["set_exact_accuracy"] >= 0.6980750789381359,
    "behavior_ordered": m["ordered_exact_accuracy"] >= 0.6853865045023974,
    "behavior_value": m["value_win_accuracy"] >= 0.6631364752660507,
    "behavior_count": m["count_accuracy"] >= 0.99,
    "fresh_bc": (
        e["valid_games"] == 128
        and e["wins"] >= 64
        and e["invalid_games"] == 0
    ),
}
result = {
    "stage": "early_behavior_and_fresh_bc",
    "behavior": {
        key: m[key]
        for key in (
            "rows",
            "set_exact_accuracy",
            "ordered_exact_accuracy",
            "hybrid_order_exact_accuracy",
            "value_win_accuracy",
            "count_accuracy",
        )
    },
    "fresh_bc": e,
    "gates": gates,
    "passed": all(gates.values()),
}
(root / "early_gate.json").write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
PY
then
  echo "v10 stopped at the predeclared early gate"
  exit 0
fi

run_match vs_v3u440_256_block1 "$v3_u440" "$marnie_deck" 256 20261012
run_match vs_v3u570_128 "$v3_u570" "$marnie_deck" 128 20261013
run_match vs_kang128 "$kang_checkpoint" "$kang_deck" 128 20261014 \
  --opponent-canonical-order
run_match vs_v4soup75_128 "$v4_soup75" "$marnie_deck" 128 20261015
run_match vs_v3u440_256_block2 "$v3_u440" "$marnie_deck" 256 20261016

"$python_bin" - "$output_dir" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(sys.argv[1])


def read(name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def evaluation(name: str) -> dict:
    return read(name)["evaluation"]


behavior = read("behavior.json")
early = read("early_gate.json")
block1 = evaluation("vs_v3u440_256_block1.json")
block2 = evaluation("vs_v3u440_256_block2.json")
v3 = {
    "valid_games": block1["valid_games"] + block2["valid_games"],
    "wins": block1["wins"] + block2["wins"],
    "losses": block1["losses"] + block2["losses"],
    "draws": block1["draws"] + block2["draws"],
    "invalid_games": block1["invalid_games"] + block2["invalid_games"],
    "blocks": [block1, block2],
}
matches = {
    "vs_v3u440_512": v3,
    "vs_v3u570_128": evaluation("vs_v3u570_128.json"),
    "vs_kang128": evaluation("vs_kang128.json"),
    "vs_v4soup75_128": evaluation("vs_v4soup75_128.json"),
    "vs_freshbc128": early["fresh_bc"],
}
minimum_wins = {
    "vs_v3u440_512": 269,
    "vs_v3u570_128": 57,
    "vs_kang128": 82,
    "vs_v4soup75_128": 64,
    "vs_freshbc128": 64,
}
match_gates = {
    label: (
        result["wins"] >= minimum_wins[label]
        and result["invalid_games"] == 0
        and result["valid_games"] == (512 if label == "vs_v3u440_512" else 128)
    )
    for label, result in matches.items()
}
checkpoint = Path(behavior["checkpoint"])
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
summary = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "candidate": str(checkpoint),
    "checkpoint_sha256": digest,
    "behavior": early["behavior"],
    "behavior_gates": {
        key: value
        for key, value in early["gates"].items()
        if key.startswith("behavior_")
    },
    "matches": matches,
    "match_gates": match_gates,
    "all_absolute_gates": (
        all(early["gates"].values()) and all(match_gates.values())
    ),
    "submission_or_packaging_performed": False,
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
PY

echo "Completed v10 update-0461 screen: $output_dir"

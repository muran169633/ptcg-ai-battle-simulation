#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"

output_dir="artifacts/ppo_marnie_v9_shared_audit_20260726"
candidate="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"

if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"

run_match() {
  local label="$1"
  local opponent="$2"
  local opponent_deck="$3"
  local seed="$4"
  shift 4
  "$python_bin" tools/evaluate_ppo_head_to_head.py \
    --candidate "$candidate" \
    --opponent "$opponent" \
    --bc-checkpoint "$bc_checkpoint" \
    --candidate-deck "$marnie_deck" \
    --opponent-deck "$opponent_deck" \
    --games 64 \
    --environments 16 \
    --max-game-decisions 1000 \
    --seed "$seed" \
    --device cuda \
    "$@" \
    --output "$output_dir/$label.json"
}

run_match bc \
  "$bc_checkpoint" "$marnie_deck" 20260923 \
  --opponent-canonical-order
run_match v1_u200 \
  artifacts/ppo_marnie_terminal01_v1/checkpoints/update-0200.pt \
  "$marnie_deck" 20260924
run_match v3_u440 \
  "$candidate" "$marnie_deck" 20260925
run_match v4_soup75 \
  artifacts/ppo_marnie_v4_soups_u80_u100/alpha-075.pt \
  "$marnie_deck" 20260926
run_match dragapult \
  artifacts/bc_dragapult_lumen_orbit_v1/best.pt \
  data/decks/dragapult_lumen.csv 20260927 \
  --opponent-canonical-order
run_match kangaskhan \
  artifacts/bc_kangaskhan_orbit_v1/best.pt \
  data/decks/kangaskhan_standard.csv 20260928 \
  --opponent-canonical-order

"$python_bin" - "$output_dir" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
summary = {}
for path in sorted(output_dir.glob("*.json")):
    if path.name == "summary.json":
        continue
    result = json.loads(path.read_text(encoding="utf-8"))
    evaluation = result["evaluation"]
    summary[path.stem] = {
        "games": int(evaluation["valid_games"]),
        **{
            key: int(evaluation[key])
            for key in ("wins", "losses", "draws", "invalid_games")
        },
    }
payload = {
    "candidate": "v3 update 440",
    "games_per_opponent": 64,
    "learner_perspective": True,
    "results": summary,
}
(output_dir / "summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

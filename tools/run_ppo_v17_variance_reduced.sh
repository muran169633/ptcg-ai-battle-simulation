#!/usr/bin/env bash
set -euo pipefail

# Variance-reduced v17: reproduce the locked v16 guard-priority mechanism from
# the UQ anchor, but collect twice as many independent games per update and
# reuse each rollout for one PPO epoch instead of two. There is one fixed
# endpoint and no intermediate-checkpoint selection.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
mode="${1:-full}"
bc_checkpoint="artifacts/bc_marnie_train24_day25hash80_orbit_v3/best.pt"
resume_checkpoint="artifacts/ppo_marnie_v9_UQ_betaucb_u455/checkpoints/update-0455.pt"
marnie_deck="data/decks/marnie_grimmsnarl_froslass_luca.csv"
v3_checkpoint="artifacts/ppo_marnie_v3_incumbent600/checkpoints/update-0440.pt"
v3_name="update-0440@marnie_grimmsnarl_froslass_luca"

case "$mode" in
  full)
    output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v17_guard_pcgrad_varred_u458}"
    updates=458
    games_per_update=256
    environments=16
    minibatch_size=1024
    eval_games=64
    seed=20261151
    core_quota=128
    ;;
  smoke)
    output_dir="${OUTPUT_DIR:-artifacts/ppo_marnie_v17_guard_pcgrad_varred_smoke_u456}"
    updates=456
    games_per_update=32
    environments=8
    minibatch_size=256
    eval_games=8
    seed=20261150
    core_quota=16
    ;;
  *)
    echo "Usage: $0 [full|smoke]" >&2
    exit 2
    ;;
esac

for path in \
  "$bc_checkpoint" \
  "$resume_checkpoint" \
  "$marnie_deck" \
  "$v3_checkpoint"; do
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

"$python_bin" - "$output_dir" "$mode" "$resume_checkpoint" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(sys.argv[1])
mode = sys.argv[2]
resume = Path(sys.argv[3])
manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "v17_variance_reduced_guard_pcgrad",
    "mode": mode,
    "start_checkpoint": str(resume.resolve()),
    "start_checkpoint_sha256": hashlib.sha256(resume.read_bytes()).hexdigest(),
    "single_locked_endpoint": "update-0458.pt" if mode == "full" else "update-0456.pt",
    "selection_rule": (
        "No intermediate checkpoint selection. The fixed endpoint receives "
        "only a behavior safety gate, followed directly by the complete "
        "independent confirmation."
    ),
    "mechanism_change_from_v16": {
        "games_per_update": {"v16": 128, "v17": 256},
        "ppo_epochs": {"v16": 2, "v17": 1},
        "purpose": (
            "Approximately preserve optimizer steps while doubling independent "
            "games and eliminating second-epoch rollout reuse."
        ),
    },
    "submission_or_packaging_authorized": False,
}
(root / "preregistration.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

"$python_bin" tools/train_ppo.py \
  --bc-checkpoint "$bc_checkpoint" \
  --kl-reference-checkpoint "$resume_checkpoint" \
  --deck "$marnie_deck" \
  --extra-opponent "$v3_checkpoint" "$marnie_deck" \
  --output-dir "$output_dir" \
  --updates "$updates" \
  --environments "$environments" \
  --games-per-update "$games_per_update" \
  --ppo-epochs 1 \
  --minibatch-size "$minibatch_size" \
  --learning-rate 1e-5 \
  --value-learning-rate 1.5e-5 \
  --weight-decay 0 \
  --learning-rate-schedule constant \
  --schedule-start-update 456 \
  --gamma 1.0 \
  --gae-lambda 0.97 \
  --advantage-normalization per_opponent \
  --clip-ratio 0.10 \
  --value-coefficient 0.5 \
  --entropy-coefficient 0 \
  --max-grad-norm 0.5 \
  --policy-temperature 1.0 \
  --trainable-scope heads \
  --bc-kl-start 0.05 \
  --bc-kl-end 0.05 \
  --target-kl 0.002 \
  --league-probability 1.0 \
  --opponent-sampling per_game \
  --opponent-quota-mode fixed \
  --opponent-quota-seat-balance \
  --opponent-base-quota bc "$core_quota" \
  --opponent-base-quota "$v3_name" "$core_quota" \
  --ppo-objective constrained \
  --constrained-gradient-mode guard_pcgrad \
  --primary-opponent-name bc \
  --guard-opponent-name "$v3_name" \
  --primary-policy-weight 1.0 \
  --guard-policy-weight 1.25 \
  --auxiliary-policy-weight 0 \
  --guard-surrogate-floor 0 \
  --constraint-dual-initial 0 \
  --constraint-dual-lr 0 \
  --constraint-dual-max 10 \
  --opponent-loss-weight bc 0.5 \
  --opponent-loss-weight "$v3_name" 0.5 \
  --snapshot-interval 1000 \
  --max-pool-size 2 \
  --eval-interval "$updates" \
  --eval-games "$eval_games" \
  --eval-all-permanent-opponents \
  --selection-aggregation min \
  --checkpoint-interval "$updates" \
  --max-game-decisions 1000 \
  --seed "$seed" \
  --resume "$resume_checkpoint" \
  --reset-optimizer-on-resume \
  --reset-opponent-quota-on-resume \
  --device cuda

candidate="$output_dir/checkpoints/update-$(printf '%04d' "$updates").pt"
if [[ ! -f "$candidate" ]]; then
  echo "Missing fixed endpoint: $candidate" >&2
  exit 1
fi
"$python_bin" - "$output_dir" "$candidate" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(sys.argv[1])
candidate = Path(sys.argv[2])
lock = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "candidate": str(candidate.resolve()),
    "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "selected_intermediate_checkpoint": False,
    "submission_or_packaging_performed": False,
}
(root / "candidate_lock.json").write_text(
    json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(lock, ensure_ascii=False, indent=2))
PY


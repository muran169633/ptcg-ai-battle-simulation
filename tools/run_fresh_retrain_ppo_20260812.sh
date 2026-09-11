#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-/home/xxc/miniconda3/envs/my_project_env/bin/python}"
bc_checkpoint="artifacts/retrain_top50_recent7_fresh_20260812_v1/marnie_specialist_soups/a50_b50.pt"
marnie_anchor="artifacts/gold8_recent7_20260808/marnie/specialist_bc/best.pt"
alakazam_anchor="artifacts/top100_proxy_recent14_20260811_v1/alakazam_control/best.pt"
marnie_deck="data/gold_push_marnie_top50_recent7_20260811_v1/decks/c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af.csv"
alakazam_deck="data/top100_proxy_recent14_20260811_v1/decks/3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf.csv"
bc_replay="data/retrain_top50_recent7_fresh_20260812_v1/marnie_specialist_latesttrain.zip"
output_dir="artifacts/retrain_top50_recent7_fresh_20260812_v1/marnie_ppo_seed202608802"

marnie_anchor_name="best@c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
alakazam_anchor_name="best@3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"

for path in \
  "$python_bin" \
  "$bc_checkpoint" \
  "$marnie_anchor" \
  "$alakazam_anchor" \
  "$marnie_deck" \
  "$alakazam_deck" \
  "$bc_replay"; do
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

"$python_bin" - "$output_dir" "$python_bin" "$bc_checkpoint" \
  "$marnie_anchor" "$alakazam_anchor" "$marnie_deck" \
  "$alakazam_deck" "$bc_replay" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
named_paths = {
    "python": Path(sys.argv[2]),
    "bc_checkpoint": Path(sys.argv[3]),
    "marnie_anchor": Path(sys.argv[4]),
    "alakazam_anchor": Path(sys.argv[5]),
    "marnie_deck": Path(sys.argv[6]),
    "alakazam_deck": Path(sys.argv[7]),
    "bc_replay": Path(sys.argv[8]),
}
manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "fresh_top50_recent7_marnie_bc_then_ppo",
    "submission_authorized": False,
    "freshness": {
        "general_bc_random_initialization": True,
        "specialist_initialized_only_from_fresh_general_bc": True,
        "ppo_optimizer_state": "fresh",
        "old_marnie_checkpoint_role": "frozen_opponent_only",
    },
    "bc_gate": {
        "split": "2026-08-09 validation",
        "rows": 44486,
        "exact_action_set_accuracy": 0.7712089196601178,
        "threshold": 0.77,
        "passed": True,
    },
    "reward": {"win": 1.0, "loss": 0.0, "draw": 0.0},
    "ppo_design": {
        "updates": 6,
        "games_per_update": 384,
        "fixed_games_per_opponent_per_update": 128,
        "ppo_epochs": 2,
        "trainable_scope": "last_block_heads",
        "actor_learning_rate": 4e-6,
        "target_kl": 3e-4,
        "bc_anchor_kl": 0.03,
        "advantage_normalization": "per_opponent",
        "actor_reduction": "episode_mean",
        "bc_replay_steps_per_update": 2,
    },
    "inputs": {},
}
for name, path in named_paths.items():
    resolved = path.resolve()
    manifest["inputs"][name] = {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "bytes": resolved.stat().st_size,
    }
(output / "preregistration.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

"$python_bin" tools/train_ppo.py \
  --bc-checkpoint "$bc_checkpoint" \
  --kl-reference-checkpoint "$bc_checkpoint" \
  --deck "$marnie_deck" \
  --extra-opponent "$marnie_anchor" "$marnie_deck" \
  --extra-opponent "$alakazam_anchor" "$alakazam_deck" \
  --output-dir "$output_dir" \
  --updates 6 \
  --environments 16 \
  --games-per-update 384 \
  --ppo-epochs 2 \
  --minibatch-size 1024 \
  --learning-rate 4e-6 \
  --value-learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --gamma 1.0 \
  --gae-lambda 0.97 \
  --advantage-normalization per_opponent \
  --clip-ratio 0.10 \
  --value-coefficient 0.5 \
  --value-trunk-gradient-scale 0.25 \
  --entropy-coefficient 0.001 \
  --max-grad-norm 0.5 \
  --policy-temperature 1.0 \
  --trainable-scope last_block_heads \
  --learning-rate-schedule constant \
  --schedule-start-update 1 \
  --bc-kl-start 0.03 \
  --bc-kl-end 0.03 \
  --target-kl 0.0003 \
  --league-probability 1.0 \
  --opponent-sampling per_game \
  --opponent-quota-mode fixed \
  --opponent-quota-seat-balance \
  --opponent-base-quota bc 128 \
  --opponent-base-quota "$marnie_anchor_name" 128 \
  --opponent-base-quota "$alakazam_anchor_name" 128 \
  --ppo-objective standard \
  --actor-reduction episode_mean \
  --snapshot-interval 1000 \
  --max-pool-size 3 \
  --bc-replay-data "$bc_replay" \
  --bc-replay-split train \
  --bc-replay-batches 12 \
  --bc-replay-batch-size 256 \
  --bc-replay-workers 8 \
  --bc-replay-steps 2 \
  --bc-replay-lr-scale 0.25 \
  --bc-replay-loss set \
  --eval-interval 6 \
  --eval-games 96 \
  --eval-all-permanent-opponents \
  --selection-aggregation min \
  --checkpoint-interval 6 \
  --max-game-decisions 1000 \
  --seed 202608802 \
  --device cuda 2>&1 | tee "$output_dir/training.log"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation"
PYTHON="/home/xxc/miniconda3/envs/my_project_env/bin/python"
CANDIDATE="$ROOT/artifacts/bc_top100_recent14_mode_ar_v7_end0813_b1024_20260815_v1/best.pt"
METRICS="$ROOT/artifacts/bc_top100_recent14_mode_ar_v7_end0813_b1024_20260815_v1/metrics.jsonl"
BASELINE="$ROOT/artifacts/rmy_ogerpon_hydrapple_goldpush_20260814_v1/ppo_currenttop23_90pool10self_gate58_mb4096_u120_seed2026081406/fixed_gate54_g200_v1/champions/update-0110.pt"
DECK="$ROOT/data/current_top23_20260814_v3/decks/rank06_16425135_7e3984370203.csv"
ARCHIVE="$ROOT/submissions/ptcg_ppo_rmy_ogerpon_hydrapple_u110_20260814.tar.gz"
MANIFEST="$ROOT/submissions/ptcg_ppo_rmy_ogerpon_hydrapple_u110_20260814.tar.gz.manifest.json"
CONTRACT="$ROOT/artifacts/rmy_ogerpon_hydrapple_goldpush_20260814_v1/submission_raw_u110_v1/deployment_contract.json"
VALIDATION="$ROOT/submissions/ptcg_ppo_rmy_ogerpon_hydrapple_u110_20260814.validation.json"
OUTPUT_DIR="$ROOT/artifacts/bc_top100_recent14_mode_ar_v7_end0813_b1024_20260815_v1/h2h_vs_latest_rmy_submission"

COMMON=(
  --candidate "$CANDIDATE"
  --submitted-checkpoint "$BASELINE"
  --deck "$DECK"
  --submission-archive "$ARCHIVE"
  --submission-manifest "$MANIFEST"
  --deployment-contract "$CONTRACT"
  --submission-validation "$VALIDATION"
  --training-metrics "$METRICS"
  --environments 32
  --max-game-decisions 1000
  --loop-diagnostic-tail 8
  --seed 20260815
  --device cuda
)

mkdir -p "$OUTPUT_DIR"

"$PYTHON" "$ROOT/tools/evaluate_mode_ar_vs_submission.py" \
  "${COMMON[@]}" \
  --games 64 \
  --output "$OUTPUT_DIR/smoke_64.json"

"$PYTHON" - "$OUTPUT_DIR/smoke_64.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
evaluation = result["evaluation"]
if evaluation["valid_games"] != 64 or evaluation["invalid_games"] != 0:
    raise SystemExit(f"smoke gate failed: {evaluation}")
if result["match"]["same_deck_mirror"] is not True:
    raise SystemExit("smoke gate failed: comparison is not same-deck mirror")
PY

"$PYTHON" "$ROOT/tools/evaluate_mode_ar_vs_submission.py" \
  "${COMMON[@]}" \
  --games 2048 \
  --output "$OUTPUT_DIR/strict_2048.json"

"$PYTHON" - "$OUTPUT_DIR/strict_2048.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
evaluation = result["evaluation"]
if evaluation["valid_games"] != 2048 or evaluation["invalid_games"] != 0:
    raise SystemExit(f"strict H2H gate failed: {evaluation}")
seats = evaluation["by_candidate_seat"]
if any(seats[str(seat)]["valid_games"] != 1024 for seat in (0, 1)):
    raise SystemExit(f"strict H2H seat gate failed: {seats}")
PY

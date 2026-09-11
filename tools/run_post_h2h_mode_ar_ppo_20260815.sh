#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation"
PYTHON="/home/xxc/miniconda3/envs/my_project_env/bin/python"
BC="$ROOT/artifacts/bc_top100_recent14_mode_ar_v7_end0813_b1024_20260815_v1/best.pt"
H2H="$ROOT/artifacts/bc_top100_recent14_mode_ar_v7_end0813_b1024_20260815_v1/h2h_vs_latest_rmy_submission/strict_2048.json"
META="$ROOT/data/recent_day_meta_pool_20260813_top23_v1/meta_pool.json"
RMY_DECK="$ROOT/data/recent_day_meta_pool_20260813_top23_v1/decks/rank05_7e3984370203.csv"
DRAG_DECK="$ROOT/data/recent_day_meta_pool_20260813_top23_v1/decks/rank02_07bedfffbfad.csv"
BASE="$ROOT/artifacts/mode_ar_ppo_posth2h_20260815_v1"

while [[ ! -f "$H2H" ]]
do
  sleep 30
done

"$PYTHON" - "$H2H" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
evaluation = result["evaluation"]
if evaluation["valid_games"] != 2048 or evaluation["invalid_games"] != 0:
    raise SystemExit(f"strict BC H2H gate failed: {evaluation}")
if result["match"]["same_deck_mirror"] is not True:
    raise SystemExit("strict BC H2H was not a same-deck mirror")
PY

common=(
  --bc-checkpoint "$BC"
  --meta-pool "$META"
  --device cuda
)

"$PYTHON" "$ROOT/tools/train_mode_ar_ppo.py" \
  "${common[@]}" \
  --route rmy \
  --learner-deck "$RMY_DECK" \
  --output-dir "$BASE/smoke_rmy_u1_g16_seed2026081511" \
  --updates 1 \
  --environments 8 \
  --games-per-update 16 \
  --ppo-epochs 1 \
  --minibatch-size 256 \
  --champion-gate-interval 1 \
  --champion-gate-games 16 \
  --bc-eval-games 16 \
  --checkpoint-interval 1 \
  --seed 2026081511

"$PYTHON" "$ROOT/tools/train_mode_ar_ppo.py" \
  "${common[@]}" \
  --route dragapult \
  --learner-deck "$DRAG_DECK" \
  --output-dir "$BASE/smoke_dragapult_u1_g16_seed2026081512" \
  --updates 1 \
  --environments 8 \
  --games-per-update 16 \
  --ppo-epochs 1 \
  --minibatch-size 256 \
  --champion-gate-interval 1 \
  --champion-gate-games 16 \
  --bc-eval-games 16 \
  --checkpoint-interval 1 \
  --seed 2026081512

"$PYTHON" - "$BASE/smoke_rmy_u1_g16_seed2026081511" "$BASE/smoke_dragapult_u1_g16_seed2026081512" <<'PY'
import json
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    run = Path(raw)
    rows = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or rows[0]["update"] != 1:
        raise SystemExit(f"PPO smoke metrics gate failed: {run}")
    if rows[0]["rollout"]["valid_games"] != 16:
        raise SystemExit(f"PPO smoke rollout gate failed: {run}")
    if rows[0]["champion_gate"]["invalid_games"] != 0:
        raise SystemExit(f"PPO smoke champion gate failed: {run}")
    if rows[0]["evaluation_vs_initial_bc"]["invalid_games"] != 0:
        raise SystemExit(f"PPO smoke BC gate failed: {run}")
    if not (run / "checkpoints/update-0001.pt").is_file():
        raise SystemExit(f"PPO smoke checkpoint missing: {run}")
PY

"$PYTHON" "$ROOT/tools/train_mode_ar_ppo.py" \
  "${common[@]}" \
  --route rmy \
  --learner-deck "$RMY_DECK" \
  --output-dir "$BASE/rmy_recent1_invwin200_90pool10self_u120_seed2026081513" \
  --updates 120 \
  --environments 64 \
  --games-per-update 1024 \
  --ppo-epochs 4 \
  --minibatch-size 4096 \
  --champion-gate-interval 10 \
  --champion-gate-games 200 \
  --champion-gate-min-win-rate 0.54 \
  --bc-eval-games 200 \
  --checkpoint-interval 10 \
  --seed 2026081513

"$PYTHON" "$ROOT/tools/train_mode_ar_ppo.py" \
  "${common[@]}" \
  --route dragapult \
  --learner-deck "$DRAG_DECK" \
  --output-dir "$BASE/dragapult_recent1_invwin200_90pool10self_u120_seed2026081514" \
  --updates 120 \
  --environments 64 \
  --games-per-update 1024 \
  --ppo-epochs 4 \
  --minibatch-size 4096 \
  --champion-gate-interval 10 \
  --champion-gate-games 200 \
  --champion-gate-min-win-rate 0.54 \
  --bc-eval-games 200 \
  --checkpoint-interval 10 \
  --seed 2026081514

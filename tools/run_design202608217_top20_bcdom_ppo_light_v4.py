#!/usr/bin/env python3
"""Lightweight PPO continuation from v3 best (top20 two-week BC, legacy BC-first path)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
TRAINER = ROOT / "tools" / "train_ppo.py"
RESUME = (
    ROOT
    / "artifacts/design202608217_s24_bcdom_ppo_light_v3/"
    / "B_gold_league/seed-202608217/best.pt"
)
BC_CHECKPOINT = (
    ROOT
    / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731"
    / "best.pt"
)
REPLAY = (
    ROOT
    / "data/bc_marnie_top20_current14_latesttop50_trainvalid_through0804_design202608194.zip"
)
OUTPUT = (
    ROOT
    / "artifacts/design202608217_s24_bcdom_ppo_light_v4/"
    / "B_gold_league/seed-202608217"
)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"Output exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        "-I",
        "-B",
        str(TRAINER),
        "--bc-checkpoint",
        str(BC_CHECKPOINT),
        "--resume",
        str(RESUME),
        "--resume-learner-weights",
        "resume",
        "--reset-optimizer-on-resume",
        "--skip-initial-eval",
        "--output-dir",
        str(OUTPUT),
        "--updates",
        "560",
        "--seed",
        "202608217",
        "--schedule-start-update",
        "485",
        "--environments",
        "12",
        "--games-per-update",
        "64",
        "--ppo-epochs",
        "2",
        "--minibatch-size",
        "256",
        "--learning-rate",
        "1.8e-06",
        "--value-learning-rate",
        "6e-06",
        "--learning-rate-schedule",
        "constant",
        "--clip-ratio",
        "0.15",
        "--value-coefficient",
        "0.25",
        "--entropy-coefficient",
        "0.001",
        "--max-grad-norm",
        "0.5",
        "--policy-temperature",
        "0.8",
        "--trainable-scope",
        "last_block_heads",
        "--bc-kl-start",
        "0.003",
        "--bc-kl-end",
        "0.003",
        "--target-kl",
        "0.0012",
        "--league-probability",
        "1.0",
        "--opponent-quota-mode",
        "legacy",
        "--eval-interval",
        "8",
        "--eval-games",
        "64",
        "--checkpoint-interval",
        "8",
        "--eval-all-permanent-opponents",
        "--selection-aggregation",
        "mean",
        "--bc-replay-data",
        str(REPLAY),
        "--bc-replay-split",
        "train",
        "--bc-replay-batches",
        "72",
        "--bc-replay-batch-size",
        "256",
        "--bc-replay-workers",
        "8",
        "--bc-replay-steps",
        "1",
        "--bc-replay-lr-scale",
        "0.02",
        "--bc-replay-loss",
        "ordered",
        "--bc-replay-order-context-weight",
        "8.0",
        "--bc-replay-non-context34-fixed-multi-action-order-weight",
        "1.0",
        "--bc-replay-context34-rows-per-batch",
        "4",
        "--deck",
        str(ROOT / "data/decks/marnie_grimmsnarl_froslass_luca.csv"),
    ]
    print("CMD:", " ".join(cmd))
    return subprocess.call(cmd, env=os.environ.copy(), cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

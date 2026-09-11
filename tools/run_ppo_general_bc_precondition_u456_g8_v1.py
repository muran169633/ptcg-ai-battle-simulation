#!/usr/bin/env python3
"""Apply one frozen natural-order G8 general-BC stage to the true U456 anchor.

The audited U464 implementation supplies the fail-closed cache, optimizer,
parameter-scope, finite-value, displacement, and checkpoint checks.  This
specialization binds those checks to the byte-exact U456 incumbent.  The G8
stage continues U456's compatible replay AdamW state for eight steps; the
following PPO stage is separately required to reset PPO, replay, and quota
state and therefore consumes this checkpoint as weights-only lineage.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

sys.dont_write_bytecode = True

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

BASE_IMPL_PATH = TOOLS_ROOT / "run_ppo_general_bc_precondition_u464_g8.py"
BASE_IMPL_SHA256 = "b91e7c86947f7e591cc64820b40efc396c1adb6bc5084d9a7bdac8c92aac227d"
_base_stat = os.lstat(BASE_IMPL_PATH)
if stat.S_ISLNK(_base_stat.st_mode) or not stat.S_ISREG(_base_stat.st_mode):
    raise RuntimeError("authenticated G8 implementation must be a regular file")
if hashlib.sha256(BASE_IMPL_PATH.read_bytes()).hexdigest() != BASE_IMPL_SHA256:
    raise RuntimeError("authenticated G8 implementation SHA-256 mismatch")

import run_ppo_general_bc_precondition_u464_g8 as base  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
base.__file__ = str(Path(__file__).resolve())
base.DESIGN_ID = 202608130
base.G8_SEED = 202608013
base.G8_BATCH_INDICES = (1, 3, 4, 5, 6, 7, 8, 11)
base.G8_DISPLACEMENT_MIN = 0.00100
base.G8_DISPLACEMENT_MAX = 0.00450

base.PARENT_PATH = REPO_ROOT / (
    "artifacts/ppo_bc28init_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_u448meta_to_u456_seed20260736/"
    "B_gold_league/seed-20260736/checkpoints/update-0456.pt"
)
base.PARENT_SHA256 = (
    "b7ed9580543e4a2374ffcc618bb2eed74b90e762117a587768c90daf0019c83b"
)
base.PARENT_UPDATE = 456
base.PARENT_MODEL_SHA256 = (
    "e9baf1917d24aeaab66341835daf86105d071caab9b2985082f921611db18b2c"
)
base.PARENT_PPO_SHA256 = (
    "9871a48963689affb08eb02a13388e7498a54b35155f23aa6d86ed87c14717b6"
)
base.PARENT_REPLAY_SHA256 = (
    "777082c5d53b15f086095c55ddd27d40f7afc97e1db7c14201078d55635e827e"
)
base.PARENT_QUOTA_SHA256 = (
    "caea4b3bfbed2bfe6ed1467aa181cdb641ef5d62796544b6f785f77637c3c4fd"
)
base.PARENT_PPO_STEP = 208
base.PARENT_REPLAY_STEP = 16
base.GENERAL_CACHE_SHA256 = (
    "7a9f7d44d45f472e31c762500c8ca8b3464fa4cd0c315e1049e4295835ee0f08"
)

base.EXPECTED_PARENT_CONFIG = {
    "updates": 456,
    "minibatch_size": 512,
    "actor_reduction": "episode_mean",
    "trainable_scope": "last_block_heads",
    "learning_rate": 3.6e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 0.5,
    "policy_temperature": 0.8,
    "bc_replay_data": str(base.GENERAL_REPLAY_PATH),
    "bc_replay_split": "train",
    "bc_replay_batches": 72,
    "bc_replay_batch_size": 256,
    "bc_replay_workers": 8,
    "bc_replay_steps": 2,
    "bc_replay_lr_scale": 0.05,
    "bc_replay_loss": "ordered",
    "bc_replay_order_context_weight": 8.0,
    "bc_replay_context34_rows_per_batch": 4,
    "bc_replay_non_context34_fixed_multi_action_order_weight": 1.0,
    "seed": 20260736,
}

if __name__ == "__main__":
    raise SystemExit(base.main())

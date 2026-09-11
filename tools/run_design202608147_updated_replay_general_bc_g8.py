#!/usr/bin/env python3
"""Apply the frozen updated-replay G8 transport to guarded U468."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path


sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
BASE_PATH = TOOLS / "run_ppo_general_bc_precondition_u464_g8.py"
BASE_SHA256 = "b91e7c86947f7e591cc64820b40efc396c1adb6bc5084d9a7bdac8c92aac227d"
info = os.lstat(BASE_PATH)
if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
    raise RuntimeError("authenticated G8 base must be a regular file")
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("authenticated G8 base SHA-256 mismatch")

import run_ppo_general_bc_precondition_u464_g8 as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
base.__file__ = str(Path(__file__).resolve())
base.DESIGN_ID = 202608147
base.G8_SEED = 202608147
base.G8_BATCH_INDICES = (1, 3, 4, 5, 6, 7, 8, 11)
base.G8_LR_SCALE = 0.075
base.G8_LEARNING_RATE = 2.4e-5 * 0.075
base.G8_DISPLACEMENT_MIN = 0.00100
base.G8_DISPLACEMENT_MAX = 0.00450

base.PARENT_PATH = ROOT / (
    "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141/"
    "ppo_stage/block3/B_gold_league/seed-202608141/checkpoints/update-0468.pt"
)
base.PARENT_SHA256 = "a9290745b8eb58704c4617594ac5ab49500c6e65ecd9f273a8285c12547fbb83"
base.PARENT_UPDATE = 468
base.PARENT_MODEL_SHA256 = "832c724277d6c2263d76ddbf41622f4ba06621ec3155fa8b7ae2591203db1c5e"
base.PARENT_PPO_SHA256 = "93f4bfc722eb1d950805674484e057ac620e020658832fddb60cd0babedf6c9d"
base.PARENT_REPLAY_SHA256 = "40d9104d155153781ef0e2da006deb4a66f9b5b1b7a3792ef34dc104dfd5742c"
base.PARENT_QUOTA_SHA256 = "2800876baf7f5d2b89971e2494e389a153b8acf09c8677cee010ef652423e851"
base.PARENT_PPO_STEP = 468
base.PARENT_REPLAY_STEP = 24

base.GENERAL_REPLAY_PATH = ROOT / (
    "data/bc_marnie_top50_current14_timeforward_"
    "train0802_valid0803_design202608147.zip"
)
base.GENERAL_REPLAY_SHA256 = "bf01cfc7e9c0f616f157ae36ed68c76619ae45448cdc17562befe889a4d5ae93"
base.GENERAL_CACHE_SHA256 = "676ca38d51dfeb0b3d6dc78d4e7ba77eeba2d532277adce2a56395a83ffc6d78"

# These values authenticate the held U468 checkpoint's original config.
# The cache path and deterministic cache seed are changed only after this
# validation, leaving all other PPO and replay fields byte-for-byte equal.
base.EXPECTED_PARENT_CONFIG = {
    "updates": 468,
    "minibatch_size": 512,
    "actor_reduction": "episode_mean",
    "trainable_scope": "last_block_heads",
    "learning_rate": 2.4e-5,
    "value_learning_rate": 7.5e-6,
    "weight_decay": 1e-4,
    "max_grad_norm": 0.5,
    "policy_temperature": 0.8,
    "bc_checkpoint": str(base.GENERAL_BC_PATH),
    "bc_replay_data": str(
        ROOT / "data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip"
    ),
    "bc_replay_split": "train",
    "bc_replay_batches": 72,
    "bc_replay_batch_size": 256,
    "bc_replay_workers": 8,
    "bc_replay_steps": 2,
    "bc_replay_lr_scale": 0.075,
    "bc_replay_loss": "ordered",
    "bc_replay_order_context_weight": 8.0,
    "bc_replay_context34_rows_per_batch": 4,
    "bc_replay_non_context34_fixed_multi_action_order_weight": 1.0,
    "seed": 202608141,
}

_validate_parent_config = base.validate_parent_config


def validate_and_transport_config(raw: object):
    config = _validate_parent_config(raw)
    config.bc_replay_data = str(base.GENERAL_REPLAY_PATH)
    config.seed = base.G8_SEED
    return config


base.validate_parent_config = validate_and_transport_config


if __name__ == "__main__":
    raise SystemExit(base.main())

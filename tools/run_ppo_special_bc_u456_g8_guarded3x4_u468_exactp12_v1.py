#!/usr/bin/env python3
"""Run the frozen guarded-3x4 U468 exact-historical-P12 special-BC stage.

This wrapper authenticates the frozen S32 executor, applies only explicit
contextual transformations needed by the separately preregistered P12 route,
and replaces every historical parent/design/output binding before delegating
to the inherited fail-closed executor.  The inherited executor retains its
held-descriptor authentication, zero-step audit, gradient guard, optimizer
round-trip checks, and single-terminal-checkpoint publication protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

SOURCE_EXECUTOR_PATH = (
    REPO_ROOT / "tools/run_ppo_special_bc_u468_actorheadonly_s32eqp12_v3.py"
)
SOURCE_EXECUTOR_SHA256 = (
    "3c398bcec22f4d379cb8b49ba683b9e2bfb959d21dec483de1c11601e37b40df"
)

BRANCH = "ppo_u456_g8_guarded3x4x96_p12_design202608141"
DESIGN_PATH = REPO_ROOT / f"artifacts/{BRANCH}.p12_design_preregistration.json"
DESIGN_SHA256 = "6b6da1d3b2896e09bcf0a7915e981cbeaaf36cf0a70be7b7b75193484f60bb1f"
MASTER_PATH = REPO_ROOT / f"artifacts/{BRANCH}.master_preregistration.json"
MASTER_SHA256 = "e5dd52ba5e28f5672ea0199ba3b55c4da3fd8a6f45ebd6062b77fca41a5d37c7"
AUTHORIZATION_PATH = (
    REPO_ROOT / f"artifacts/{BRANCH}.block3_training_integrity_decision.json"
)
AUTHORIZATION_SHA256 = (
    "d57954cdaeed5d0a2ff5c2a8d459253ba03f1b9a0f8dd1b5c760b23fc6f026bf"
)
PARENT_PATH = REPO_ROOT / (
    f"artifacts/{BRANCH}/ppo_stage/block3/B_gold_league/seed-202608141/"
    "checkpoints/update-0468.pt"
)
PARENT_SHA256 = "a9290745b8eb58704c4617594ac5ab49500c6e65ecd9f273a8285c12547fbb83"
PARENT_UPDATE = 468
PARENT_MODEL_SHA256 = (
    "832c724277d6c2263d76ddbf41622f4ba06621ec3155fa8b7ae2591203db1c5e"
)
PARENT_MODEL_NESTED_SHA256 = (
    "04c977f141e5b6e1e01646831b39153b5e0133e3ebb6da295b621efe526f122e"
)
PARENT_PPO_SHA256 = (
    "93f4bfc722eb1d950805674484e057ac620e020658832fddb60cd0babedf6c9d"
)
PARENT_REPLAY_SHA256 = (
    "40d9104d155153781ef0e2da006deb4a66f9b5b1b7a3792ef34dc104dfd5742c"
)
PARENT_QUOTA_SHA256 = (
    "2800876baf7f5d2b89971e2494e389a153b8acf09c8677cee010ef652423e851"
)
PARENT_PPO_STATE_COUNT = 28
PARENT_PPO_STEP = 468
PARENT_REPLAY_STATE_COUNT = 24
PARENT_REPLAY_STEP = 24
PARENT_QUOTA_GAMES = 1152
PARENT_QUOTA_REFRESH = 466

PREFLIGHT_DIR = REPO_ROOT / f"artifacts/{BRANCH}.p12_preflight_v1"
PREFLIGHT_PATH = PREFLIGHT_DIR / "preflight_audit.json"
FORMAL_DIR = REPO_ROOT / f"artifacts/{BRANCH}/special_stage"
CHECKPOINT_PATH = (
    FORMAL_DIR / "special-bc-actorheadonly-pokemonfan-exactp12-0012.pt"
)
MANIFEST_PATH = FORMAL_DIR / "special_bc_manifest.json"
ATTEMPT_MARKER = REPO_ROOT / (
    ".ptcg-u456-g8-guarded3x4-u468-exactp12-specialbc-"
    "attempt-202608012-202608141.json"
)

SPECIAL_SEED = 202608012
BATCH_INDICES = (0, 1, 2, 7, 8, 9, 10, 14, 15, 16, 18, 19)
STEPS = 12
SPECIAL_LR_SCALE = 0.075
SPECIAL_LEARNING_RATE = 1.8e-6
L2_MIN = 0.00177912
L2_MAX = 0.00266868

EXPECTED_PARENT_CONFIG: dict[str, Any] = {
    "updates": 468,
    "minibatch_size": 512,
    "actor_reduction": "episode_mean",
    "trainable_scope": "last_block_heads",
    "learning_rate": 2.4e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 0.5,
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
}


def require_regular_bytes(path: Path, expected: str, label: str) -> bytes:
    current = os.lstat(path)
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise ValueError(f"{label} is not a regular non-symlink file")
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {observed}"
        )
    return raw


def require_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    raw = require_regular_bytes(path, expected, label)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def validate_control_bindings() -> None:
    design = require_json(DESIGN_PATH, DESIGN_SHA256, "guarded P12 design")
    if (
        design.get("status")
        != "authoritative_design_locked_after_block3_GO_P12_and_before_wrapper_creation_or_preflight_attempt"
        or design.get("only_parent", {}).get("sha256") != PARENT_SHA256
        or design.get("authorization", {}).get("sha256") != AUTHORIZATION_SHA256
    ):
        raise ValueError("guarded P12 design identity is invalid")

    master = require_json(MASTER_PATH, MASTER_SHA256, "guarded master design")
    if (
        master.get("branch") != BRANCH
        or master.get("special_bc_if_all_three_blocks_pass", {}).get("recipe")
        != "exact_historical_P12"
    ):
        raise ValueError("guarded master does not authorize exact historical P12")

    authorization = require_json(
        AUTHORIZATION_PATH, AUTHORIZATION_SHA256, "Block3 GO_P12 authorization"
    )
    terminal = authorization.get("terminal_identity")
    expected_terminal = {
        "update": PARENT_UPDATE,
        "runtime_model_state_sha256": PARENT_MODEL_SHA256,
        "model_state_nested_sha256": PARENT_MODEL_NESTED_SHA256,
        "ppo_optimizer_nested_sha256": PARENT_PPO_SHA256,
        "ppo_optimizer_state_count": PARENT_PPO_STATE_COUNT,
        "ppo_optimizer_step": PARENT_PPO_STEP,
        "bc_replay_optimizer_nested_sha256": PARENT_REPLAY_SHA256,
        "bc_replay_optimizer_state_count": PARENT_REPLAY_STATE_COUNT,
        "bc_replay_optimizer_step": PARENT_REPLAY_STEP,
        "opponent_quota_nested_sha256": PARENT_QUOTA_SHA256,
        "opponent_quota_observed_games": PARENT_QUOTA_GAMES,
        "opponent_quota_last_refresh_update": PARENT_QUOTA_REFRESH,
        "all_model_and_optimizer_values_finite": True,
    }
    if authorization.get("status") != "GO_P12" or terminal != expected_terminal:
        raise ValueError("Block3 authorization or terminal identity drifted")
    effect = authorization.get("decision_effect")
    if (
        not isinstance(effect, dict)
        or effect.get("block3_terminal_is_sole_PPO_candidate") is not True
        or effect.get("exact_p12_authorized_after_new_zero_step_preflight") is not True
        or effect.get("transport_checkpoint_selection_forbidden") is not True
    ):
        raise ValueError("Block3 decision effect does not authorize this P12 route")


def replace_exactly_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"source transform {label!r} count is {count}, expected 1")
    return source.replace(old, new, 1)


def transformed_executor_source(raw: bytes) -> str:
    source = raw.decode("utf-8")
    replacements = (
        (
            "if tuple(BATCH_INDICES) != tuple(range(32)) or STEPS != 32:",
            "if tuple(BATCH_INDICES) != "
            "(0, 1, 2, 7, 8, 9, 10, 14, 15, 16, 18, 19) or STEPS != 12:",
            "P12 batch cardinality",
        ),
        (
            "if 3.6e-5 * SPECIAL_LR_SCALE != SPECIAL_LEARNING_RATE:",
            "if 2.4e-5 * SPECIAL_LR_SCALE != SPECIAL_LEARNING_RATE:",
            "guarded actor learning-rate identity",
        ),
        (
            'integrity["optimizer_steps"] == 32,',
            'integrity["optimizer_steps"] == 12,',
            "optimizer step gate",
        ),
        (
            'integrity["rows"] == 8192,',
            'integrity["rows"] == 3072,',
            "row gate",
        ),
        (
            'integrity["context34_rows"] == 32,',
            'integrity["context34_rows"] == 12,',
            "context34 row gate",
        ),
        (
            '"ptcg-u468-actorheadonly-s32eqp12-execution-v3"',
            '"ptcg-u456-g8-guarded3x4-u468-exacthistoricalp12-execution-v1"',
            "execution schema",
        ),
        (
            '"u468_actorheadonly_s32eqp12_attempt_consumed"',
            '"u456_g8_guarded3x4_u468_exacthistoricalp12_attempt_consumed"',
            "attempt event",
        ),
        (
            '"actorheadonly_s32eqp12_completed"',
            '"u456_g8_guarded3x4_u468_exacthistoricalp12_completed"',
            "terminal status",
        ),
    )
    for old, new, label in replacements:
        source = replace_exactly_once(source, old, new, label)

    batch_key = '"batch_order_exact_natural_0_to_31"'
    if source.count(batch_key) != 2:
        raise RuntimeError("source batch-order key count is not exactly 2")
    source = source.replace(batch_key, '"batch_order_exact_historical_p12"')

    preflight_label = '"S32 v3 preflight audit"'
    if source.count(preflight_label) != 2:
        raise RuntimeError("source preflight binding-label count is not exactly 2")
    source = source.replace(preflight_label, '"guarded exact-P12 preflight audit"')

    old_bindings = '''    binding_specs = [
        (tool_path, args.expected_tool_sha256, "executor"),
        (DESIGN_V1_PATH, DESIGN_V1_SHA256, "S32 training design v1"),
        (DESIGN_V2_PATH, DESIGN_V2_SHA256, "S32 transport correction v2"),
        (DESIGN_PATH, DESIGN_SHA256, "S32 descriptor correction v3"),
        (V1_REJECTION_PATH, V1_REJECTION_SHA256, "rejected v1 decision"),
        (V2_REJECTION_PATH, V2_REJECTION_SHA256, "rejected v2 decision"),
        (MASTER_PATH, MASTER_SHA256, "master design"),
        (AUTHORIZATION_PATH, AUTHORIZATION_SHA256, "Stage2 authorization"),
        (PARENT_PATH, PARENT_SHA256, "U468 parent"),
        (GENERAL_BC_PATH, GENERAL_BC_SHA256, "general BC"),
        (SPECIAL_DATA_PATH, SPECIAL_DATA_SHA256, "PokemonFan archive"),
        (TRAIN_PPO_PATH, TRAIN_PPO_SHA256, "trainer"),
        (TRAIN_BC_PATH, TRAIN_BC_SHA256, "BC dependency"),
        (REPAIR_PATH, REPAIR_SHA256, "audit dependency"),
        (CG_INIT_PATH, CG_INIT_SHA256, "cg package dependency"),
        (CG_SIM_PATH, CG_SIM_SHA256, "sim dependency"),
        (CG_LIB_PATH, CG_LIB_SHA256, "native dependency"),
    ]'''
    new_bindings = '''    binding_specs = [
        (tool_path, args.expected_tool_sha256, "executor"),
        (SOURCE_EXECUTOR_PATH, SOURCE_EXECUTOR_SHA256, "authenticated source executor"),
        (DESIGN_PATH, DESIGN_SHA256, "guarded exact-P12 design"),
        (MASTER_PATH, MASTER_SHA256, "guarded master design"),
        (AUTHORIZATION_PATH, AUTHORIZATION_SHA256, "Block3 GO_P12 authorization"),
        (PARENT_PATH, PARENT_SHA256, "U468 parent"),
        (GENERAL_BC_PATH, GENERAL_BC_SHA256, "general BC"),
        (SPECIAL_DATA_PATH, SPECIAL_DATA_SHA256, "PokemonFan archive"),
        (TRAIN_PPO_PATH, TRAIN_PPO_SHA256, "trainer"),
        (TRAIN_BC_PATH, TRAIN_BC_SHA256, "BC dependency"),
        (REPAIR_PATH, REPAIR_SHA256, "audit dependency"),
        (CG_INIT_PATH, CG_INIT_SHA256, "cg package dependency"),
        (CG_SIM_PATH, CG_SIM_SHA256, "sim dependency"),
        (CG_LIB_PATH, CG_LIB_SHA256, "native dependency"),
    ]'''
    source = replace_exactly_once(
        source, old_bindings, new_bindings, "guarded binding set"
    )

    old_common = '''        "training_design_v1": {
            "path": str(DESIGN_V1_PATH),
            "sha256": DESIGN_V1_SHA256,
        },
        "transport_design_v2": {
            "path": str(DESIGN_V2_PATH),
            "sha256": DESIGN_V2_SHA256,
        },
        "design": {"path": str(DESIGN_PATH), "sha256": DESIGN_SHA256},
        "v1_rejection": {
            "path": str(V1_REJECTION_PATH),
            "sha256": V1_REJECTION_SHA256,
        },
        "v2_rejection": {
            "path": str(V2_REJECTION_PATH),
            "sha256": V2_REJECTION_SHA256,
        },
        "master": {"path": str(MASTER_PATH), "sha256": MASTER_SHA256},'''
    new_common = '''        "source_executor": {
            "path": str(SOURCE_EXECUTOR_PATH),
            "sha256": SOURCE_EXECUTOR_SHA256,
        },
        "design": {"path": str(DESIGN_PATH), "sha256": DESIGN_SHA256},
        "master": {"path": str(MASTER_PATH), "sha256": MASTER_SHA256},'''
    source = replace_exactly_once(
        source, old_common, new_common, "guarded provenance metadata"
    )

    old_parent_gate = '''    if int(parent.get("update", -1)) != PARENT_UPDATE:
        raise ValueError("parent update is not U468")
    config = validate_parent_config(parent.get("config"))'''
    new_parent_gate = '''    if int(parent.get("update", -1)) != PARENT_UPDATE:
        raise ValueError("parent update is not U468")
    if repair.nested_sha256(parent.get("model_state_dict")) != PARENT_MODEL_NESTED_SHA256:
        raise ValueError("parent nested model state hash mismatch")
    config = validate_parent_config(parent.get("config"))'''
    source = replace_exactly_once(
        source, old_parent_gate, new_parent_gate, "nested parent model gate"
    )
    source = replace_exactly_once(
        source,
        '            "model_state_sha256": model_hash_before,',
        '            "model_state_sha256": model_hash_before,\n'
        '            "model_state_nested_sha256": PARENT_MODEL_NESTED_SHA256,',
        "nested parent model provenance",
    )
    return source


def main() -> int:
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError("current working directory must be the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("wrapper must use my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("wrapper requires outer Python flags -I -B")
    validate_control_bindings()
    source_raw = require_regular_bytes(
        SOURCE_EXECUTOR_PATH, SOURCE_EXECUTOR_SHA256, "authenticated source executor"
    )
    transformed = transformed_executor_source(source_raw)

    runtime = types.ModuleType("_ptcg_guarded3x4_u468_exactp12_runtime")
    runtime.__file__ = str(Path(__file__).resolve())
    runtime.__package__ = None
    exec(compile(transformed, str(Path(__file__).resolve()), "exec"), runtime.__dict__)

    runtime.BRANCH = BRANCH
    runtime.SOURCE_EXECUTOR_PATH = SOURCE_EXECUTOR_PATH
    runtime.SOURCE_EXECUTOR_SHA256 = SOURCE_EXECUTOR_SHA256
    runtime.DESIGN_PATH = DESIGN_PATH
    runtime.DESIGN_SHA256 = DESIGN_SHA256
    runtime.MASTER_PATH = MASTER_PATH
    runtime.MASTER_SHA256 = MASTER_SHA256
    runtime.AUTHORIZATION_PATH = AUTHORIZATION_PATH
    runtime.AUTHORIZATION_SHA256 = AUTHORIZATION_SHA256
    runtime.PARENT_PATH = PARENT_PATH
    runtime.PARENT_SHA256 = PARENT_SHA256
    runtime.PARENT_UPDATE = PARENT_UPDATE
    runtime.PARENT_MODEL_SHA256 = PARENT_MODEL_SHA256
    runtime.PARENT_MODEL_NESTED_SHA256 = PARENT_MODEL_NESTED_SHA256
    runtime.PARENT_PPO_SHA256 = PARENT_PPO_SHA256
    runtime.PARENT_REPLAY_SHA256 = PARENT_REPLAY_SHA256
    runtime.PARENT_QUOTA_SHA256 = PARENT_QUOTA_SHA256
    runtime.PARENT_PPO_STATE_COUNT = PARENT_PPO_STATE_COUNT
    runtime.PARENT_PPO_STEP = PARENT_PPO_STEP
    runtime.PARENT_REPLAY_STATE_COUNT = PARENT_REPLAY_STATE_COUNT
    runtime.PARENT_REPLAY_STEP = PARENT_REPLAY_STEP
    runtime.PARENT_QUOTA_GAMES = PARENT_QUOTA_GAMES
    runtime.PARENT_QUOTA_REFRESH = PARENT_QUOTA_REFRESH
    runtime.EXPECTED_PARENT_CONFIG = dict(EXPECTED_PARENT_CONFIG)
    runtime.PREFLIGHT_DIR = PREFLIGHT_DIR
    runtime.PREFLIGHT_PATH = PREFLIGHT_PATH
    runtime.FORMAL_DIR = FORMAL_DIR
    runtime.CHECKPOINT_PATH = CHECKPOINT_PATH
    runtime.MANIFEST_PATH = MANIFEST_PATH
    runtime.ATTEMPT_MARKER = ATTEMPT_MARKER
    runtime.SPECIAL_SEED = SPECIAL_SEED
    runtime.STEPS = STEPS
    runtime.BATCH_INDICES = BATCH_INDICES
    runtime.SPECIAL_LR_SCALE = SPECIAL_LR_SCALE
    runtime.SPECIAL_LEARNING_RATE = SPECIAL_LEARNING_RATE
    runtime.L2_MIN = L2_MIN
    runtime.L2_MAX = L2_MAX
    return int(runtime.main())


if __name__ == "__main__":
    raise SystemExit(main())

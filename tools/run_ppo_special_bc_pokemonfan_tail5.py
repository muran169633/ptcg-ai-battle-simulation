#!/usr/bin/env python3
"""Continue the frozen PokemonFan32 actor with five fresh tail batches.

This is a standalone executor, not a constant-mutating wrapper around
``run_ppo_special_bc``.  It independently binds its own bytes, the reference
base executor, every imported training dependency, the exact PokemonFan32
parent, and the original specialist archive.  Audit-only mode constructs and
hashes the complete replay cache but performs zero optimizer steps and writes
no checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import run_ppo_bc_repair as repair_audit
import train_ppo as ppo


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SCHEMA = "ptcg-pokemonfan-fresh-tail5-archive-v1"
OUTPUT_SCHEMA = "ptcg-u456-post-pokemonfan32-tail5-v1"
ROW_SCHEMA = "ptcg-bc-visible-decisions-v1"
DECISION_KEY_FORMAT = "dataset_date|episode_id|seat|action_step_index"

SPECIAL_STEPS = 5
SPECIAL_BATCH_SIZE = 256
SPECIAL_ROWS = SPECIAL_STEPS * SPECIAL_BATCH_SIZE
SPECIAL_CONTEXT34_ROWS_PER_BATCH = 1
SPECIAL_CONTEXT34_ROWS = SPECIAL_STEPS
SPECIAL_ORDINARY_ROWS = SPECIAL_ROWS - SPECIAL_CONTEXT34_ROWS
SPECIAL_LEARNING_RATE = 1.8e-6
P32_STEPS = 32
CUMULATIVE_SPECIAL_STEPS = P32_STEPS + SPECIAL_STEPS
EXPECTED_REPLAY_STEP_BEFORE = 48
EXPECTED_REPLAY_STEP_AFTER = 53
EXPECTED_ACTOR_TENSORS = 24

EXPECTED_PARENT_PATH = (
    REPO_ROOT
    / "artifacts/ppo_u456inc_generalbc_ppo_specialbc_"
    "pokemonfan_timeforward_actoronly32_seed202608012/"
    "special_stage/special-bc-0032.pt"
)
EXPECTED_PARENT_SHA256 = (
    "2f8c612807a991ee8f1280d1238a88063412ad64118da1f3f6ce3744c146960f"
)
EXPECTED_PARENT_MODEL_SHA256 = (
    "cc6648210a24ae36842d8ecf0b568a9f76e620c8abc718d61a48ac82390a4a00"
)
EXPECTED_PARENT_REPLAY_SHA256 = (
    "a10d996400083c5f437e53fb7affba1011e0c24f325d97950cfb047d16a4c1ab"
)
EXPECTED_PARENT_PPO_SHA256 = (
    "9871a48963689affb08eb02a13388e7498a54b35155f23aa6d86ed87c14717b6"
)
EXPECTED_PARENT_PROVENANCE_SCHEMA = "ptcg-u456-post-special-bc32-v1"

EXPECTED_ANCESTOR_PATH = (
    REPO_ROOT
    / "artifacts/ppo_bc28init_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_u448meta_to_u456_seed20260736/B_gold_league/"
    "seed-20260736/checkpoints/update-0456.pt"
)
EXPECTED_ANCESTOR_SHA256 = (
    "b7ed9580543e4a2374ffcc618bb2eed74b90e762117a587768c90daf0019c83b"
)
EXPECTED_GENERAL_BC_PATH = (
    REPO_ROOT
    / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_"
    "seed20260922_20260731/best.pt"
)
EXPECTED_GENERAL_BC_SHA256 = (
    "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
)
EXPECTED_SOURCE_ARCHIVE_PATH = (
    REPO_ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_"
    "valid29_special_20260801.zip"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "884f6ef27847b9f333c66a2b888e8708cf1139e61d3c667ce6d2bc3d9f3c9b6c"
)
EXPECTED_SOURCE_TRAIN_MEMBER = "train/part-00000.jsonl"
EXPECTED_SOURCE_TRAIN_MEMBER_SHA256 = (
    "1cd6e76e7e60446bccb2966d5d63bf44c30eb73956cc3986721351b6e966dbde"
)
EXPECTED_SOURCE_ROWS = 9487
EXPECTED_SOURCE_CONTEXT34_ROWS = 38
EXPECTED_P32_PREFIX_ROWS = 8192
EXPECTED_P32_PREFIX_ORDINARY_ROWS = 8160
EXPECTED_P32_PREFIX_CONTEXT34_ROWS = 32
EXPECTED_P32_DECISION_KEYS_DIGEST = (
    "0aa87ca9da7ed669f359706f3ed6e96dcc5c11d86df349e9f16493806074a636"
)
EXPECTED_TAIL_DECISION_KEYS_DIGEST = (
    "5a0d3dc6987edb4def344054f6ce7deb554ebc6d48006f6726aaa5dea4d62837"
)
EXPECTED_COMBINED_DECISION_KEYS_DIGEST = (
    "d1a43d1f113c67133bc17de09fa782665badb5d58924ff5744368b10f341dde7"
)
EXPECTED_RAW_SELECTED_CONTENT_SHA256 = (
    "505fca08a6ecccca3e7bb46f165d8bc26c8efd442a00ace3b387b45f429c54e7"
)
EXPECTED_UNUSED_ROWS = 15
EXPECTED_UNUSED_ORDINARY_ROWS = 14
EXPECTED_UNUSED_CONTEXT34_ROWS = 1

EXPECTED_TAIL_ARCHIVE_PATH = (
    REPO_ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_"
    "valid29_special_tail5_after_p32_20260801.zip"
)
EXPECTED_TAIL_ARCHIVE_SHA256 = (
    "b014c6bc6ef656314e2b679ae34f7cfe927d647add65b6e69024bdba13f65c79"
)
EXPECTED_TAIL_MANIFEST_SHA256 = (
    "98366492785537b0ab1eafa7c7649efae4fc7c7a26b09a16845bf1339066b603"
)
EXPECTED_TRAINING_SEED = 202608030

EXPECTED_P32_EXECUTOR_PATH = REPO_ROOT / "tools/run_ppo_special_bc_32.py"
EXPECTED_P32_EXECUTOR_SHA256 = (
    "daaca8902819b56a3aff2c8a1cab802dc537c0cec9df59569c1e118ee801d661"
)
BASE_EXECUTOR_PATH = REPO_ROOT / "tools/run_ppo_special_bc.py"
REPAIR_AUDIT_PATH = REPO_ROOT / "tools/run_ppo_bc_repair.py"
TRAIN_PPO_PATH = REPO_ROOT / "tools/train_ppo.py"
EXPECTED_BASE_EXECUTOR_SHA256 = (
    "84698c85effa5e22f6de421ec07011a85fc4ede2a7c14de73f61f0f26b615ef6"
)
EXPECTED_REPAIR_AUDIT_SHA256 = (
    "6035e023c760dd23fb426a9b363cd59f4a8bf71e14534eb19c0a4e2e73aa9966"
)
EXPECTED_TRAIN_PPO_SHA256 = (
    "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
)

EXPECTED_PARENT_CONFIG: dict[str, Any] = {
    "actor_reduction": "episode_mean",
    "trainable_scope": "last_block_heads",
    "learning_rate": 3.6e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 0.5,
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
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    return repair_audit.file_sha256(path)


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_lines(values: list[str] | set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_sha256(value: str, label: str) -> str:
    if len(value) != 64:
        raise ValueError(f"{label} is not a 64-character SHA256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value


def validate_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is absent or symlinked: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def resolve_repo_path(raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return value


def stable_decision_key(row: dict[str, Any], label: str) -> str:
    dataset_date = row.get("dataset_date")
    episode_id = row.get("episode_id")
    if (
        not isinstance(dataset_date, str)
        or not dataset_date
        or "|" in dataset_date
        or not isinstance(episode_id, (str, int))
        or not str(episode_id)
        or "|" in str(episode_id)
    ):
        raise ValueError(f"{label}: invalid dataset/episode identity")
    seat = require_integer(row.get("seat"), f"{label}: seat")
    step = require_integer(
        row.get("action_step_index"),
        f"{label}: action_step_index",
    )
    if seat < 0 or step < 0:
        raise ValueError(f"{label}: seat/action step must be non-negative")
    return f"{dataset_date}|{episode_id}|{seat}|{step}"


def row_context(row: dict[str, Any], label: str) -> int:
    observation = row.get("observation")
    select = observation.get("select") if isinstance(observation, dict) else None
    if not isinstance(select, dict):
        raise ValueError(f"{label}: observation.select is absent")
    try:
        observed = int(select.get("context", 0) or 0)
        declared = int(row.get("select_context"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid select context") from error
    if observed != declared:
        raise ValueError(f"{label}: select_context disagrees with observation")
    return observed


def validate_row(row: Any, label: str) -> tuple[str, bool]:
    if not isinstance(row, dict):
        raise ValueError(f"{label}: row is not an object")
    if row.get("schema_version") != ROW_SCHEMA or row.get("split") != "train":
        raise ValueError(f"{label}: row schema/split drifted")
    if row.get("team_name") != "Pokemon Fan":
        raise ValueError(f"{label}: row is not Pokemon Fan")
    if row.get("deck_hash") != (
        "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
    ):
        raise ValueError(f"{label}: Pokemon Fan deck hash drifted")
    if "visualize" in row or (
        isinstance(row.get("observation"), dict)
        and "visualize" in row["observation"]
    ):
        raise ValueError(f"{label}: hidden visualize payload is forbidden")
    weight = row.get("sample_weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) <= 0.0
    ):
        raise ValueError(f"{label}: sample_weight must be finite and positive")
    return (
        stable_decision_key(row, label),
        row_context(row, label) == ppo.SKILL_ORDER_CONTEXT,
    )


def parse_rows(raw_lines: list[bytes], label: str) -> tuple[list[str], list[bool]]:
    keys: list[str] = []
    contexts: list[bool] = []
    for index, raw_line in enumerate(raw_lines, start=1):
        if not raw_line:
            raise ValueError(f"{label}:{index}: blank row")
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{label}:{index}: invalid JSON") from error
        key, is_context34 = validate_row(row, f"{label}:{index}")
        keys.append(key)
        contexts.append(is_context34)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label}: duplicate decision keys")
    return keys, contexts


def validate_parent_config(raw_config: Any) -> ppo.PPOConfig:
    if not isinstance(raw_config, dict):
        raise ValueError("Parent checkpoint has no PPO config")
    for name, expected in EXPECTED_PARENT_CONFIG.items():
        actual = raw_config.get(name)
        if actual != expected:
            raise ValueError(
                f"Parent config {name!r} mismatch: expected {expected!r}, "
                f"got {actual!r}"
            )
    config = ppo.PPOConfig(**raw_config)
    if not math.isclose(
        config.learning_rate * config.bc_replay_lr_scale,
        SPECIAL_LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Parent replay learning rate drifted")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue the exact PokemonFan32 checkpoint with exactly five "
            "fresh, source-order PokemonFan tail batches."
        )
    )
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--expected-parent-update", type=int, default=456)
    parser.add_argument("--general-bc-checkpoint", type=Path)
    parser.add_argument("--expected-general-bc-sha256", required=True)
    parser.add_argument("--special-data", type=Path, required=True)
    parser.add_argument("--expected-special-data-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-raw-selected-content-sha256", required=True)
    parser.add_argument("--expected-decision-keys-digest", required=True)
    parser.add_argument("--expected-combined-decision-keys-digest", required=True)
    parser.add_argument("--expected-replay-cache-sha256", required=True)
    parser.add_argument("--expected-executor-sha256", required=True)
    parser.add_argument("--expected-base-executor-sha256", required=True)
    parser.add_argument("--expected-repair-audit-sha256", required=True)
    parser.add_argument("--expected-train-ppo-sha256", required=True)
    parser.add_argument(
        "--expected-replay-state-step",
        type=int,
        default=EXPECTED_REPLAY_STEP_BEFORE,
    )
    parser.add_argument("--seed", "--special-seed", dest="seed", type=int, required=True)
    parser.add_argument(
        "--steps",
        "--repair-steps",
        dest="steps",
        type=int,
        default=SPECIAL_STEPS,
    )
    parser.add_argument(
        "--batch-indices",
        type=int,
        nargs="+",
        default=list(range(SPECIAL_STEPS)),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def validate_request(args: argparse.Namespace) -> None:
    if args.steps != SPECIAL_STEPS:
        raise ValueError("This protocol requires exactly five additional steps")
    if args.batch_indices != list(range(SPECIAL_STEPS)):
        raise ValueError("This protocol requires batch order 0 1 2 3 4")
    if args.expected_parent_update != 456:
        raise ValueError("The parent must retain checkpoint update label 456")
    if args.expected_replay_state_step != EXPECTED_REPLAY_STEP_BEFORE:
        raise ValueError("The replay AdamW must resume at step 48")
    if args.seed != EXPECTED_TRAINING_SEED:
        raise ValueError("This protocol requires training seed 202608030")
    if args.parent_checkpoint.is_symlink():
        raise ValueError("The frozen PokemonFan32 parent may not be a symlink")
    if args.parent_checkpoint.resolve() != EXPECTED_PARENT_PATH.resolve():
        raise ValueError("This protocol requires the exact PokemonFan32 parent path")
    if args.expected_parent_sha256 != EXPECTED_PARENT_SHA256:
        raise ValueError("This protocol requires the exact PokemonFan32 parent SHA256")
    if args.expected_general_bc_sha256 != EXPECTED_GENERAL_BC_SHA256:
        raise ValueError("The general BC anchor SHA256 drifted")
    if args.special_data.is_symlink():
        raise ValueError("The frozen tail5 archive may not be a symlink")
    if args.special_data.resolve() != EXPECTED_TAIL_ARCHIVE_PATH.resolve():
        raise ValueError("This protocol requires the exact tail5 archive path")
    if args.expected_special_data_sha256 != EXPECTED_TAIL_ARCHIVE_SHA256:
        raise ValueError("The frozen tail5 archive SHA256 drifted")
    if args.expected_manifest_sha256 != EXPECTED_TAIL_MANIFEST_SHA256:
        raise ValueError("The frozen tail5 manifest SHA256 drifted")
    if args.expected_base_executor_sha256 != EXPECTED_BASE_EXECUTOR_SHA256:
        raise ValueError("The base executor dependency SHA256 drifted")
    if args.expected_repair_audit_sha256 != EXPECTED_REPAIR_AUDIT_SHA256:
        raise ValueError("The repair-audit dependency SHA256 drifted")
    if args.expected_train_ppo_sha256 != EXPECTED_TRAIN_PPO_SHA256:
        raise ValueError("The train_ppo dependency SHA256 drifted")
    if args.expected_raw_selected_content_sha256 != (
        EXPECTED_RAW_SELECTED_CONTENT_SHA256
    ):
        raise ValueError("The selected raw tail content SHA256 drifted")
    if args.expected_decision_keys_digest != EXPECTED_TAIL_DECISION_KEYS_DIGEST:
        raise ValueError("The five-batch tail decision-key digest drifted")
    if args.expected_combined_decision_keys_digest != (
        EXPECTED_COMBINED_DECISION_KEYS_DIGEST
    ):
        raise ValueError("The P32-plus-tail decision-key digest drifted")
    if args.output_dir.is_symlink():
        raise ValueError("Output directory may not be a symlink")
    for name in (
        "expected_parent_sha256",
        "expected_general_bc_sha256",
        "expected_special_data_sha256",
        "expected_manifest_sha256",
        "expected_raw_selected_content_sha256",
        "expected_decision_keys_digest",
        "expected_combined_decision_keys_digest",
        "expected_replay_cache_sha256",
        "expected_executor_sha256",
        "expected_base_executor_sha256",
        "expected_repair_audit_sha256",
        "expected_train_ppo_sha256",
    ):
        require_sha256(str(getattr(args, name)), f"--{name.replace('_', '-')}")


def validate_parent_lineage(parent: dict[str, Any]) -> dict[str, Any]:
    provenance = parent.get("post_ppo_special_bc")
    if not isinstance(provenance, dict):
        raise ValueError("PokemonFan32 parent has no special-BC provenance")
    expected_header = {
        "schema_version": EXPECTED_PARENT_PROVENANCE_SCHEMA,
        "mode": "special_bc",
        "status": "special_bc_completed",
        "checkpoint_update_label": 456,
        "not_a_new_ppo_update": True,
    }
    if any(provenance.get(key) != value for key, value in expected_header.items()):
        raise ValueError("PokemonFan32 provenance header drifted")

    special = provenance.get("special_bc")
    optimizer = provenance.get("optimizer")
    integrity = provenance.get("integrity")
    sources = provenance.get("sources")
    ancestor = provenance.get("parent")
    if not all(isinstance(value, dict) for value in (
        special,
        optimizer,
        integrity,
        sources,
        ancestor,
    )):
        raise ValueError("PokemonFan32 provenance is incomplete")
    if (
        special.get("steps_requested") != P32_STEPS
        or special.get("rows_per_batch") != SPECIAL_BATCH_SIZE
        or special.get("context34_rows_per_batch") != 1
        or special.get("loss") != "ordered"
        or special.get("order_context_weight") != 8.0
        or special.get("trainable_scope") != "last_block_heads"
        or len(special.get("batch_indices", [])) != P32_STEPS
        or set(special.get("batch_indices", [])) != set(range(P32_STEPS))
    ):
        raise ValueError("PokemonFan32 specialization protocol drifted")
    if (
        integrity.get("optimizer_steps") != P32_STEPS
        or integrity.get("rows") != EXPECTED_P32_PREFIX_ROWS
        or integrity.get("context34_rows") != EXPECTED_P32_PREFIX_CONTEXT34_ROWS
        or integrity.get("model_state_sha256_after")
        != EXPECTED_PARENT_MODEL_SHA256
        or integrity.get("replay_state_sha256_after")
        != EXPECTED_PARENT_REPLAY_SHA256
        or integrity.get("ppo_state_sha256_after") != EXPECTED_PARENT_PPO_SHA256
        or set(integrity.get("replay_steps_after", []))
        != {EXPECTED_REPLAY_STEP_BEFORE}
        or integrity.get("changed_parameters_subset_of_actor") is not True
        or integrity.get("value_head_parameters_unchanged") is not True
        or integrity.get("ppo_optimizer_state_unchanged") is not True
        or integrity.get("parent_checkpoint_unchanged") is not True
    ):
        raise ValueError("PokemonFan32 training-integrity provenance drifted")
    actor_names = optimizer.get("actor_parameter_names")
    if (
        not isinstance(actor_names, list)
        or len(actor_names) != EXPECTED_ACTOR_TENSORS
        or set(integrity.get("changed_parameter_names", [])) != set(actor_names)
    ):
        raise ValueError("PokemonFan32 actor manifest drifted")
    if (
        Path(str(ancestor.get("path", ""))).resolve()
        != EXPECTED_ANCESTOR_PATH.resolve()
        or ancestor.get("sha256") != EXPECTED_ANCESTOR_SHA256
        or ancestor.get("update") != 456
    ):
        raise ValueError("PokemonFan32 U456 ancestor binding drifted")

    expected_sources = {
        "general_bc_checkpoint": (
            EXPECTED_GENERAL_BC_PATH,
            EXPECTED_GENERAL_BC_SHA256,
        ),
        "special_bc_archive": (
            EXPECTED_SOURCE_ARCHIVE_PATH,
            EXPECTED_SOURCE_ARCHIVE_SHA256,
        ),
        "tool": (EXPECTED_P32_EXECUTOR_PATH, EXPECTED_P32_EXECUTOR_SHA256),
        "train_ppo": (TRAIN_PPO_PATH, sources["train_ppo"].get("sha256")),
    }
    frozen_files: dict[str, str] = {
        str(EXPECTED_PARENT_PATH.resolve()): EXPECTED_PARENT_SHA256,
        str(EXPECTED_ANCESTOR_PATH.resolve()): EXPECTED_ANCESTOR_SHA256,
    }
    report: dict[str, Any] = {}
    for name, (path, sha256) in expected_sources.items():
        record = sources.get(name)
        if (
            not isinstance(record, dict)
            or Path(str(record.get("path", ""))).resolve() != path.resolve()
            or record.get("sha256") != sha256
        ):
            raise ValueError(f"PokemonFan32 source binding {name!r} drifted")
        actual = validate_sha(path, str(sha256), f"PokemonFan32 {name}")
        frozen_files[str(path.resolve())] = actual
        report[name] = {"path": str(path.resolve()), "sha256": actual}
    validate_sha(
        EXPECTED_ANCESTOR_PATH,
        EXPECTED_ANCESTOR_SHA256,
        "U456 ancestor",
    )
    if repair_audit.nested_sha256(
        parent.get("bc_replay_optimizer_state_dict")
    ) != EXPECTED_PARENT_REPLAY_SHA256:
        raise ValueError("Embedded replay optimizer state hash drifted")
    if repair_audit.nested_sha256(
        parent.get("optimizer_state_dict")
    ) != EXPECTED_PARENT_PPO_SHA256:
        raise ValueError("Embedded PPO optimizer state hash drifted")
    return {
        "validated": True,
        "sources": report,
        "frozen_files": frozen_files,
        "parent_model_state_sha256": EXPECTED_PARENT_MODEL_SHA256,
        "parent_replay_state_sha256": EXPECTED_PARENT_REPLAY_SHA256,
        "parent_ppo_state_sha256": EXPECTED_PARENT_PPO_SHA256,
        "replay_adamw_step": EXPECTED_REPLAY_STEP_BEFORE,
    }


def source_tail_audit() -> tuple[list[bytes], dict[str, Any], dict[str, str]]:
    source_hash = validate_sha(
        EXPECTED_SOURCE_ARCHIVE_PATH,
        EXPECTED_SOURCE_ARCHIVE_SHA256,
        "original PokemonFan archive",
    )
    with zipfile.ZipFile(EXPECTED_SOURCE_ARCHIVE_PATH) as archive:
        if archive.namelist().count("manifest.json") != 1:
            raise ValueError("Original PokemonFan archive manifest is not unique")
        if archive.namelist().count(EXPECTED_SOURCE_TRAIN_MEMBER) != 1:
            raise ValueError("Original PokemonFan train member is not unique")
        manifest_payload = archive.read("manifest.json")
        source_payload = archive.read(EXPECTED_SOURCE_TRAIN_MEMBER)
    if bytes_sha256(manifest_payload) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("Original PokemonFan manifest SHA256 drifted")
    if bytes_sha256(source_payload) != EXPECTED_SOURCE_TRAIN_MEMBER_SHA256:
        raise ValueError("Original PokemonFan train member SHA256 drifted")
    source_lines = source_payload.splitlines()
    if len(source_lines) != EXPECTED_SOURCE_ROWS:
        raise ValueError("Original PokemonFan train row count drifted")
    keys, contexts = parse_rows(source_lines, "original PokemonFan train")
    if sum(contexts) != EXPECTED_SOURCE_CONTEXT34_ROWS:
        raise ValueError("Original PokemonFan context-34 count drifted")

    prefix_lines = source_lines[:EXPECTED_P32_PREFIX_ROWS]
    prefix_keys = keys[:EXPECTED_P32_PREFIX_ROWS]
    prefix_contexts = contexts[:EXPECTED_P32_PREFIX_ROWS]
    if (
        sum(prefix_contexts) != EXPECTED_P32_PREFIX_CONTEXT34_ROWS
        or len(prefix_contexts) - sum(prefix_contexts)
        != EXPECTED_P32_PREFIX_ORDINARY_ROWS
        or digest_lines(prefix_keys) != EXPECTED_P32_DECISION_KEYS_DIGEST
    ):
        raise ValueError("The frozen P32 source prefix drifted")

    selected_lines: list[bytes] = []
    selected_keys: list[str] = []
    selected_contexts: list[bool] = []
    unused_keys: list[str] = []
    ordinary_count = 0
    context_count = 0
    for raw_line, key, is_context34 in zip(
        source_lines[EXPECTED_P32_PREFIX_ROWS:],
        keys[EXPECTED_P32_PREFIX_ROWS:],
        contexts[EXPECTED_P32_PREFIX_ROWS:],
    ):
        take = (
            is_context34 and context_count < SPECIAL_CONTEXT34_ROWS
        ) or (
            not is_context34 and ordinary_count < SPECIAL_ORDINARY_ROWS
        )
        if take:
            selected_lines.append(raw_line)
            selected_keys.append(key)
            selected_contexts.append(is_context34)
            context_count += int(is_context34)
            ordinary_count += int(not is_context34)
        else:
            unused_keys.append(key)
    if (
        len(selected_lines) != SPECIAL_ROWS
        or context_count != SPECIAL_CONTEXT34_ROWS
        or ordinary_count != SPECIAL_ORDINARY_ROWS
        or len(unused_keys) != EXPECTED_UNUSED_ROWS
    ):
        raise ValueError("Fresh-tail source-order selection counts drifted")
    unused_contexts = [
        contexts[keys.index(key)] for key in unused_keys
    ]
    if (
        sum(unused_contexts) != EXPECTED_UNUSED_CONTEXT34_ROWS
        or len(unused_contexts) - sum(unused_contexts)
        != EXPECTED_UNUSED_ORDINARY_ROWS
    ):
        raise ValueError("Unused PokemonFan tail counts drifted")
    tail_digest = digest_lines(selected_keys)
    combined_digest = digest_lines(prefix_keys + selected_keys)
    if tail_digest != EXPECTED_TAIL_DECISION_KEYS_DIGEST:
        raise ValueError("Fresh-tail decision-key digest drifted")
    if combined_digest != EXPECTED_COMBINED_DECISION_KEYS_DIGEST:
        raise ValueError("P32-plus-tail decision-key digest drifted")
    return selected_lines, {
        "source_rows": len(source_lines),
        "source_context34_rows": sum(contexts),
        "p32_prefix_rows": len(prefix_keys),
        "p32_prefix_context34_rows": sum(prefix_contexts),
        "p32_prefix_decision_keys_digest": digest_lines(prefix_keys),
        "tail_rows": len(selected_keys),
        "tail_ordinary_rows": ordinary_count,
        "tail_context34_rows": context_count,
        "tail_decision_keys_digest": tail_digest,
        "combined_rows": len(prefix_keys) + len(selected_keys),
        "combined_decision_keys_digest": combined_digest,
        "unused_rows": len(unused_keys),
        "unused_ordinary_rows": EXPECTED_UNUSED_ORDINARY_ROWS,
        "unused_context34_rows": EXPECTED_UNUSED_CONTEXT34_ROWS,
        "source_order_selection_recomputed": True,
        "source_rows_byte_equal_to_tail_archive": True,
    }, {
        str(EXPECTED_SOURCE_ARCHIVE_PATH.resolve()): source_hash,
    }


def validate_tail_archive(
    archive_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    archive_hash = validate_sha(
        archive_path,
        args.expected_special_data_sha256,
        "PokemonFan tail5 archive",
    )
    expected_lines, source_audit, source_files = source_tail_audit()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Tail5 archive contains duplicate member names")
        if set(names) != {"manifest.json", EXPECTED_SOURCE_TRAIN_MEMBER}:
            raise ValueError("Tail5 archive member set drifted")
        manifest_payload = archive.read("manifest.json")
        train_payload = archive.read(EXPECTED_SOURCE_TRAIN_MEMBER)
    manifest_hash = bytes_sha256(manifest_payload)
    if manifest_hash != args.expected_manifest_sha256:
        raise ValueError("Tail5 manifest SHA256 does not match the request")
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Tail5 manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARCHIVE_SCHEMA:
        raise ValueError("Tail5 archive schema drifted")
    train_lines = train_payload.splitlines()
    if train_lines != expected_lines:
        raise ValueError("Tail5 rows are not the exact fresh source-order selection")
    if bytes_sha256(train_payload) != args.expected_raw_selected_content_sha256:
        raise ValueError("Tail5 raw selected content SHA256 drifted")
    keys, contexts = parse_rows(train_lines, "tail5 train")
    if (
        len(keys) != SPECIAL_ROWS
        or len(keys) != len(set(keys))
        or sum(contexts) != SPECIAL_CONTEXT34_ROWS
        or digest_lines(keys) != args.expected_decision_keys_digest
    ):
        raise ValueError("Tail5 raw row integrity drifted")

    source = manifest.get("source")
    lineage = manifest.get("lineage")
    selection = manifest.get("selection")
    train = manifest.get("train")
    unused = manifest.get("unused_tail")
    decisions = manifest.get("decision_keys")
    if not all(
        isinstance(value, dict)
        for value in (source, lineage, selection, train, unused, decisions)
    ):
        raise ValueError("Tail5 manifest contract is incomplete")
    source_train = source.get("train_member")
    if not isinstance(source_train, dict):
        raise ValueError("Tail5 source train-member binding is absent")
    if (
        resolve_repo_path(source.get("path"), "source.path")
        != EXPECTED_SOURCE_ARCHIVE_PATH.resolve()
        or source.get("sha256") != EXPECTED_SOURCE_ARCHIVE_SHA256
        or source.get("manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or source_train.get("name") != EXPECTED_SOURCE_TRAIN_MEMBER
        or source_train.get("sha256") != EXPECTED_SOURCE_TRAIN_MEMBER_SHA256
        or source_train.get("rows") != EXPECTED_SOURCE_ROWS
        or source_train.get("ordinary_rows")
        != EXPECTED_SOURCE_ROWS - EXPECTED_SOURCE_CONTEXT34_ROWS
        or source_train.get("context34_rows") != EXPECTED_SOURCE_CONTEXT34_ROWS
    ):
        raise ValueError("Tail5 source binding drifted")
    p32_prefix = lineage.get("p32_source_prefix")
    if not isinstance(p32_prefix, dict):
        raise ValueError("Tail5 P32 source-prefix lineage is absent")
    if (
        lineage.get("parent_stage") != "PokemonFan special BC P32"
        or p32_prefix.get("rows") != EXPECTED_P32_PREFIX_ROWS
        or p32_prefix.get("ordinary_rows") != EXPECTED_P32_PREFIX_ORDINARY_ROWS
        or p32_prefix.get("context34_rows")
        != EXPECTED_P32_PREFIX_CONTEXT34_ROWS
        or p32_prefix.get("decision_keys_digest")
        != EXPECTED_P32_DECISION_KEYS_DIGEST
        or lineage.get("p32_tail_decision_key_overlap_count") != 0
    ):
        raise ValueError("Tail5 P32 lineage metadata drifted")
    if (
        selection.get("algorithm")
        != "source_order_remaining_after_exact_p32_prefix"
        or selection.get("stable_decision_key_format") != DECISION_KEY_FORMAT
        or selection.get("rows") != SPECIAL_ROWS
        or selection.get("ordinary_rows") != SPECIAL_ORDINARY_ROWS
        or selection.get("context34_rows") != SPECIAL_CONTEXT34_ROWS
        or selection.get("without_replacement") is not True
        or selection.get("source_order_preserved") is not True
        or selection.get("seed") is not None
    ):
        raise ValueError("Tail5 selection contract drifted")
    train_members = train.get("members")
    if not isinstance(train_members, list) or len(train_members) != 1:
        raise ValueError("Tail5 train member list drifted")
    train_member = train_members[0]
    if not isinstance(train_member, dict):
        raise ValueError("Tail5 train member record is not an object")
    if (
        train_member.get("member") != EXPECTED_SOURCE_TRAIN_MEMBER
        or train_member.get("sha256") != bytes_sha256(train_payload)
        or train_member.get("rows") != SPECIAL_ROWS
        or train_member.get("ordinary_rows") != SPECIAL_ORDINARY_ROWS
        or train_member.get("context34_rows") != SPECIAL_CONTEXT34_ROWS
        or train_member.get("source_order_preserved") is not True
        or train.get("member_count") != 1
        or train.get("total_rows") != SPECIAL_ROWS
        or train.get("ordinary_rows") != SPECIAL_ORDINARY_ROWS
        or train.get("context34_rows") != SPECIAL_CONTEXT34_ROWS
        or train.get("raw_selected_content_sha256")
        != EXPECTED_RAW_SELECTED_CONTENT_SHA256
    ):
        raise ValueError("Tail5 train member metadata drifted")
    if (
        unused.get("rows") != EXPECTED_UNUSED_ROWS
        or unused.get("ordinary_rows") != EXPECTED_UNUSED_ORDINARY_ROWS
        or unused.get("context34_rows") != EXPECTED_UNUSED_CONTEXT34_ROWS
    ):
        raise ValueError("Tail5 unused-tail metadata drifted")
    decision_prefix = decisions.get("p32_prefix")
    decision_combined = decisions.get("combined_p32_plus_tail")
    if not isinstance(decision_prefix, dict) or not isinstance(
        decision_combined,
        dict,
    ):
        raise ValueError("Tail5 nested decision-key lineage is absent")
    if (
        decisions.get("format") != DECISION_KEY_FORMAT
        or decisions.get("count") != SPECIAL_ROWS
        or decisions.get("duplicate_count") != 0
        or decisions.get("digest") != EXPECTED_TAIL_DECISION_KEYS_DIGEST
        or decisions.get("without_replacement") is not True
        or decision_prefix.get("count") != EXPECTED_P32_PREFIX_ROWS
        or decision_prefix.get("digest") != EXPECTED_P32_DECISION_KEYS_DIGEST
        or decision_prefix.get("duplicate_count") != 0
        or decision_combined.get("count")
        != EXPECTED_P32_PREFIX_ROWS + SPECIAL_ROWS
        or decision_combined.get("digest")
        != EXPECTED_COMBINED_DECISION_KEYS_DIGEST
        or decision_combined.get("duplicate_count") != 0
        or decision_combined.get("overlap_count") != 0
    ):
        raise ValueError("Tail5 decision-key manifest drifted")
    byte_preservation = manifest.get("byte_preservation")
    intended_replay = manifest.get("intended_replay")
    if (
        not isinstance(byte_preservation, dict)
        or byte_preservation.get("selected_rows_copied_verbatim") is not True
        or byte_preservation.get("sample_weight_modified") is not False
        or byte_preservation.get("raw_selected_content_sha256")
        != EXPECTED_RAW_SELECTED_CONTENT_SHA256
    ):
        raise ValueError("Tail5 manifest raw-content binding drifted")
    if (
        not isinstance(intended_replay, dict)
        or intended_replay.get("batches") != SPECIAL_STEPS
        or intended_replay.get("rows_per_batch") != SPECIAL_BATCH_SIZE
        or intended_replay.get("context34_rows_per_batch") != 1
        or intended_replay.get("ordinary_rows_per_batch") != 255
    ):
        raise ValueError("Tail5 intended replay contract drifted")
    source_files[str(archive_path.resolve())] = archive_hash
    return manifest, {
        "archive_sha256": archive_hash,
        "manifest_sha256": manifest_hash,
        "raw_selected_content_sha256": bytes_sha256(train_payload),
        "rows": len(keys),
        "ordinary_rows": len(keys) - sum(contexts),
        "context34_rows": sum(contexts),
        "decision_keys_digest": digest_lines(keys),
        "source_reconstruction": source_audit,
    }, source_files


def all_files_unchanged(files: dict[str, str]) -> bool:
    return all(file_sha256(Path(path)) == digest for path, digest in files.items())


def main() -> None:
    args = parse_args()
    validate_request(args)

    executor_path = Path(__file__).resolve()
    parent_path = args.parent_checkpoint.resolve()
    archive_path = args.special_data.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    general_bc_path = (
        args.general_bc_checkpoint.resolve()
        if args.general_bc_checkpoint is not None
        else EXPECTED_GENERAL_BC_PATH.resolve()
    )
    if general_bc_path != EXPECTED_GENERAL_BC_PATH.resolve():
        raise ValueError("This protocol requires the exact general BC anchor path")

    bound_files: dict[str, str] = {}
    for path, expected, label in (
        (executor_path, args.expected_executor_sha256, "tail5 executor"),
        (BASE_EXECUTOR_PATH, args.expected_base_executor_sha256, "base executor"),
        (REPAIR_AUDIT_PATH, args.expected_repair_audit_sha256, "repair audit"),
        (TRAIN_PPO_PATH, args.expected_train_ppo_sha256, "train_ppo"),
        (EXPECTED_P32_EXECUTOR_PATH, EXPECTED_P32_EXECUTOR_SHA256, "P32 executor"),
        (parent_path, args.expected_parent_sha256, "PokemonFan32 parent"),
        (general_bc_path, args.expected_general_bc_sha256, "general BC anchor"),
    ):
        bound_files[str(path.resolve())] = validate_sha(path, expected, label)

    manifest, raw_archive_integrity, archive_files = validate_tail_archive(
        archive_path,
        args,
    )
    bound_files.update(archive_files)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise ValueError("Parent is not a compatible PPO checkpoint")
    if int(parent.get("update", -1)) != args.expected_parent_update:
        raise ValueError("Parent update label drifted")
    config = validate_parent_config(parent.get("config"))
    lineage = validate_parent_lineage(parent)
    bound_files.update(lineage["frozen_files"])

    general_bc = torch.load(general_bc_path, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_checkpoint(parent, general_bc, device)
    model_hash_before = ppo.model_state_sha256(model)
    if model_hash_before != EXPECTED_PARENT_MODEL_SHA256:
        raise ValueError("Instantiated PokemonFan32 model state hash drifted")
    actor_parameters, _, trainable_manifest = ppo.configure_trainable_scope(
        model,
        config.trainable_scope,
    )
    actor_names = list(trainable_manifest["actor_parameter_names"])
    value_names = list(trainable_manifest["value_parameter_names"])
    optimizer_names = parent.get("optimizer_parameter_names")
    if not isinstance(optimizer_names, dict):
        raise ValueError("Parent checkpoint has no optimizer parameter manifest")
    if (
        len(actor_names) != EXPECTED_ACTOR_TENSORS
        or actor_names != optimizer_names.get("actor")
        or value_names != optimizer_names.get("value")
        or set(actor_names) & set(value_names)
    ):
        raise ValueError("Parent actor/value parameter manifest drifted")

    replay_optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=SPECIAL_LEARNING_RATE,
        eps=1e-5,
        weight_decay=config.weight_decay,
    )
    replay_state = parent.get("bc_replay_optimizer_state_dict")
    if not isinstance(replay_state, dict):
        raise ValueError("Parent has no replay optimizer state")
    replay_optimizer.load_state_dict(replay_state)
    replay_steps_before = repair_audit.optimizer_steps(
        replay_optimizer.state_dict()
    )
    replay_hash_before = repair_audit.nested_sha256(
        replay_optimizer.state_dict()
    )
    ppo_hash_before = repair_audit.nested_sha256(parent.get("optimizer_state_dict"))
    if (
        len(replay_steps_before) != EXPECTED_ACTOR_TENSORS
        or set(replay_steps_before) != {EXPECTED_REPLAY_STEP_BEFORE}
        or replay_hash_before != EXPECTED_PARENT_REPLAY_SHA256
        or ppo_hash_before != EXPECTED_PARENT_PPO_SHA256
    ):
        raise ValueError("Parent optimizer state binding drifted")
    if any(
        not math.isclose(
            float(group["lr"]),
            SPECIAL_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for group in replay_optimizer.param_groups
    ):
        raise ValueError("Embedded replay AdamW learning rate drifted")

    model_state_before = repair_audit.clone_model_state(model)
    special_config = copy.deepcopy(config)
    special_config.bc_replay_data = str(archive_path)
    special_config.bc_replay_split = "train"
    special_config.bc_replay_batches = SPECIAL_STEPS
    special_config.bc_replay_batch_size = SPECIAL_BATCH_SIZE
    special_config.bc_replay_workers = 1
    special_config.bc_replay_steps = 1
    special_config.bc_replay_lr_scale = 0.05
    special_config.bc_replay_loss = "ordered"
    special_config.bc_replay_order_context_weight = 8.0
    special_config.bc_replay_context34_rows_per_batch = 1
    special_config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    special_config.seed = args.seed
    replay_batches = ppo.build_bc_replay_batches(
        special_config,
        parent["model_config"],
    )
    if (
        len(replay_batches) != SPECIAL_STEPS
        or any(
            int(batch["action_counts"].shape[0]) != SPECIAL_BATCH_SIZE
            for batch in replay_batches
        )
    ):
        raise ValueError("Tail5 replay cache does not contain five full batches")
    contexts_per_batch = [
        int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum())
        for batch in replay_batches
    ]
    if contexts_per_batch != [1] * SPECIAL_STEPS:
        raise ValueError("Tail5 replay cache context-34 quota drifted")
    if any(
        not torch.equal(
            batch["sample_weights"],
            torch.ones_like(batch["sample_weights"]),
        )
        for batch in replay_batches
    ):
        raise ValueError("Tail5 replay cache sample weights drifted")
    cache_hash, batch_hashes = repair_audit.replay_cache_manifest(replay_batches)
    if cache_hash != args.expected_replay_cache_sha256:
        raise ValueError(
            "Tail5 replay cache SHA256 mismatch: expected "
            f"{args.expected_replay_cache_sha256}, got {cache_hash}"
        )

    common: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "mode": "audit_only" if args.audit_only else "special_bc",
        "parent": {
            "path": str(parent_path),
            "sha256": args.expected_parent_sha256,
            "update": int(parent["update"]),
            "model_state_sha256": model_hash_before,
            "replay_state_sha256": replay_hash_before,
            "ppo_state_sha256": ppo_hash_before,
        },
        "sources": {
            "executor": {
                "path": str(executor_path),
                "sha256": args.expected_executor_sha256,
            },
            "base_executor": {
                "path": str(BASE_EXECUTOR_PATH.resolve()),
                "sha256": args.expected_base_executor_sha256,
                "executed": False,
            },
            "repair_audit": {
                "path": str(REPAIR_AUDIT_PATH.resolve()),
                "sha256": args.expected_repair_audit_sha256,
            },
            "train_ppo": {
                "path": str(TRAIN_PPO_PATH.resolve()),
                "sha256": args.expected_train_ppo_sha256,
            },
            "general_bc_checkpoint": {
                "path": str(general_bc_path),
                "sha256": args.expected_general_bc_sha256,
            },
            "special_bc_archive": {
                "path": str(archive_path),
                "sha256": args.expected_special_data_sha256,
                "manifest_sha256": args.expected_manifest_sha256,
                "raw_selected_content_sha256": (
                    args.expected_raw_selected_content_sha256
                ),
                "decision_keys_digest": args.expected_decision_keys_digest,
                "combined_p32_plus_tail_digest": (
                    args.expected_combined_decision_keys_digest
                ),
            },
        },
        "special_bc": {
            "seed": args.seed,
            "additional_steps": SPECIAL_STEPS,
            "cumulative_special_bc_steps": CUMULATIVE_SPECIAL_STEPS,
            "batch_indices": args.batch_indices,
            "batch_selection": (
                "all_five_fresh_tail_batches_in_deterministic_cache_order"
            ),
            "batch_sha256": batch_hashes,
            "rows_per_batch": SPECIAL_BATCH_SIZE,
            "context34_rows_per_batch": 1,
            "loss": special_config.bc_replay_loss,
            "order_context_weight": special_config.bc_replay_order_context_weight,
            "learning_rate": SPECIAL_LEARNING_RATE,
            "max_grad_norm": special_config.max_grad_norm,
            "trainable_scope": special_config.trainable_scope,
        },
        "replay_cache": {
            "sha256": cache_hash,
            "batches": len(replay_batches),
            "rows": SPECIAL_ROWS,
            "context34_rows": SPECIAL_CONTEXT34_ROWS,
            "context34_rows_by_batch": contexts_per_batch,
            "all_selected_source_rows_consumed_once_before_stratification": True,
        },
        "raw_archive_integrity": raw_archive_integrity,
        "archive_manifest": manifest,
        "pokemonfan32_parent_lineage": {
            key: value for key, value in lineage.items() if key != "frozen_files"
        },
        "optimizer": {
            "actor_parameter_names": actor_names,
            "value_parameter_names": value_names,
            "replay_state_tensor_count": len(replay_steps_before),
            "replay_steps_before": replay_steps_before,
            "replay_state_sha256_before": replay_hash_before,
            "ppo_state_sha256_before": ppo_hash_before,
            "replay_state_loaded_from_parent": True,
        },
        "determinism": {
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "python_seed": args.seed,
            "torch_seed": args.seed,
        },
    }

    output_dir.mkdir(parents=False, exist_ok=False)
    if args.audit_only:
        result = common | {
            "status": "audit_passed",
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "model_state_unchanged": (
                ppo.model_state_sha256(model) == model_hash_before
            ),
            "replay_optimizer_state_unchanged": (
                repair_audit.nested_sha256(replay_optimizer.state_dict())
                == replay_hash_before
            ),
            "ppo_optimizer_state_unchanged": (
                repair_audit.nested_sha256(parent.get("optimizer_state_dict"))
                == ppo_hash_before
            ),
            "all_frozen_files_unchanged": all_files_unchanged(bound_files),
        }
        if not all(
            result[key]
            for key in (
                "model_state_unchanged",
                "replay_optimizer_state_unchanged",
                "ppo_optimizer_state_unchanged",
                "all_frozen_files_unchanged",
            )
        ):
            raise RuntimeError("Audit-only mode mutated a frozen input/state")
        write_json(output_dir / "preflight_audit.json", result)
        if sorted(path.name for path in output_dir.iterdir()) != [
            "preflight_audit.json"
        ]:
            raise RuntimeError("Audit-only output contains an unexpected file")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return

    per_step: list[dict[str, Any]] = []
    for special_step, batch_index in enumerate(args.batch_indices, start=1):
        metrics = ppo.bc_replay_update(
            model,
            replay_optimizer,
            [replay_batches[batch_index]],
            special_config,
            device,
            config.learning_rate,
        )
        if (
            not isinstance(metrics, dict)
            or metrics.get("steps") != 1
            or int(metrics.get("rows", -1)) != SPECIAL_BATCH_SIZE
            or int(metrics.get("context_34_rows", -1)) != 1
            or metrics.get("selected_batch_indices") != [0]
        ):
            raise RuntimeError("Tail5 special-BC step integrity failed")
        if not repair_audit.finite_nested(metrics):
            raise FloatingPointError("Tail5 special-BC metric is non-finite")
        per_step.append({
            "additional_step": special_step,
            "cumulative_special_bc_step": P32_STEPS + special_step,
            "batch_index": batch_index,
            "batch_sha256": batch_hashes[batch_index],
            "metrics": metrics,
        })

    model_state_after = repair_audit.clone_model_state(model)
    model_hash_after = ppo.model_state_sha256(model)
    if not repair_audit.finite_nested(model_state_after):
        raise FloatingPointError("Tail5 model contains non-finite tensors")
    changed_names = repair_audit.changed_tensor_names(
        model_state_before,
        model_state_after,
    )
    if (
        len(changed_names) != EXPECTED_ACTOR_TENSORS
        or set(changed_names) != set(actor_names)
    ):
        raise RuntimeError("Tail5 did not change exactly the 24 frozen actor tensors")
    if any(
        not torch.equal(model_state_before[name], model_state_after[name])
        for name in value_names
    ):
        raise RuntimeError("Tail5 changed value-head tensors")

    replay_state_after = replay_optimizer.state_dict()
    replay_steps_after = repair_audit.optimizer_steps(replay_state_after)
    replay_hash_after = repair_audit.nested_sha256(replay_state_after)
    if (
        len(replay_steps_after) != EXPECTED_ACTOR_TENSORS
        or set(replay_steps_after) != {EXPECTED_REPLAY_STEP_AFTER}
    ):
        raise RuntimeError("Replay AdamW did not advance exactly from 48 to 53")
    if repair_audit.nested_sha256(parent.get("optimizer_state_dict")) != ppo_hash_before:
        raise RuntimeError("PPO optimizer state changed during tail5 BC")
    if not all_files_unchanged(bound_files):
        raise RuntimeError("A frozen executor, dependency, archive, or parent changed")

    integrity = {
        "optimizer_steps": len(per_step),
        "additional_steps": SPECIAL_STEPS,
        "cumulative_special_bc_steps": CUMULATIVE_SPECIAL_STEPS,
        "rows": SPECIAL_ROWS,
        "context34_rows": SPECIAL_CONTEXT34_ROWS,
        "all_sample_weights_one": True,
        "all_metrics_and_model_tensors_finite": True,
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "model_state_changed": model_hash_after != model_hash_before,
        "changed_parameter_names": changed_names,
        "changed_exactly_24_actor_parameters": True,
        "value_head_parameters_unchanged": True,
        "replay_steps_after": replay_steps_after,
        "replay_state_sha256_after": replay_hash_after,
        "ppo_state_sha256_after": ppo_hash_before,
        "ppo_optimizer_state_unchanged": True,
        "parent_checkpoint_sha256_after": file_sha256(parent_path),
        "parent_checkpoint_unchanged": True,
        "all_lineage_archives_and_dependencies_unchanged": True,
    }
    provenance = common | {
        "status": "special_bc_completed",
        "per_step": per_step,
        "integrity": integrity,
        "checkpoint_update_label": 456,
        "not_a_new_ppo_update": True,
    }
    output_checkpoint = output_dir / "special-bc-pokemonfan32-tail5-0037.pt"
    payload = copy.deepcopy(parent)
    payload["model_state_dict"] = model_state_after
    payload["bc_replay_optimizer_state_dict"] = replay_state_after
    payload["post_ppo_special_bc"] = provenance
    torch.save(payload, output_checkpoint)
    result = provenance | {
        "checkpoint": {
            "path": str(output_checkpoint),
            "sha256": file_sha256(output_checkpoint),
            "update": 456,
            "additional_special_bc_steps": SPECIAL_STEPS,
            "cumulative_special_bc_steps": CUMULATIVE_SPECIAL_STEPS,
        },
        "checkpoint_writes": 1,
    }
    write_json(output_dir / "special_bc_manifest.json", result)
    if sorted(path.name for path in output_dir.iterdir()) != [
        "special-bc-pokemonfan32-tail5-0037.pt",
        "special_bc_manifest.json",
    ]:
        raise RuntimeError("Tail5 output directory contains unexpected files")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

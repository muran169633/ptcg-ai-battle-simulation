#!/usr/bin/env python3
"""Apply balanced Gold-core BC after the frozen PokemonFan32 checkpoint.

Unlike the general PPO replay path, this executor treats each
``train/part-xxxxx.jsonl`` archive member as one already-balanced optimizer
batch.  It deliberately never calls ``build_bc_replay_batches`` (or any
stratifier/shuffler), so the archive's policy and context quotas are preserved
exactly.  Audit-only mode performs all source, model, optimizer, and batch
checks while taking zero optimizer steps and writing no checkpoint.  The
PokemonFan32 parent provenance and its immutable U456 ancestor are independently
bound before either audit or training is allowed.
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
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import run_ppo_bc_repair as repair_audit
import train_ppo as ppo


ARCHIVE_SCHEMA = "ptcg-balanced-gold-special-archive-v1"
OUTPUT_SCHEMA = "ptcg-u456-post-pokemonfan32-balanced-core5-v1"
SOURCE_SCHEMA = "ptcg-gold-policy-visible-decisions-v1"
ROW_SCHEMA = "ptcg-bc-visible-decisions-v1"
DECISION_KEY_FORMAT = "dataset_date|episode_id|seat|action_step_index"
EPISODE_KEY_FORMAT = "dataset_date|episode_id"
REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_POLICY_IDS = (
    "rank02_dominic",
    "rank04_liam",
    "rank11_luca",
    "rank12_taichicchi",
    "rank19_szlachetny",
)
SPECIAL_STEPS = 20
SPECIAL_BATCH_SIZE = 256
SPECIAL_CONTEXT34_ROWS_PER_BATCH = 1
SPECIAL_ROWS_PER_POLICY = 1024
SPECIAL_CONTEXT34_PER_POLICY = 4
EXPECTED_VALID_ROWS = 8319
EXPECTED_VALID_CONTEXT34_ROWS = 43
EXPECTED_ARCHIVE_SEED = 202608017
EXPECTED_TRAINING_SEED = 202608020
EXPECTED_REPLAY_STEP_BEFORE = 48
EXPECTED_REPLAY_STEP_AFTER = 68
SPECIAL_LEARNING_RATE = 1.8e-6
EXPECTED_PARENT_PATH = (
    REPO_ROOT
    / "artifacts/ppo_u456inc_generalbc_ppo_specialbc_"
    "pokemonfan_timeforward_actoronly32_seed202608012/"
    "special_stage/special-bc-0032.pt"
)
EXPECTED_PARENT_SHA256 = (
    "2f8c612807a991ee8f1280d1238a88063412ad64118da1f3f6ce3744c146960f"
)
EXPECTED_PARENT_PROVENANCE_SCHEMA = "ptcg-u456-post-special-bc32-v1"
EXPECTED_PARENT_MODEL_SHA256 = (
    "cc6648210a24ae36842d8ecf0b568a9f76e620c8abc718d61a48ac82390a4a00"
)
EXPECTED_PARENT_REPLAY_SHA256 = (
    "a10d996400083c5f437e53fb7affba1011e0c24f325d97950cfb047d16a4c1ab"
)
EXPECTED_PARENT_PPO_SHA256 = (
    "9871a48963689affb08eb02a13388e7498a54b35155f23aa6d86ed87c14717b6"
)
EXPECTED_ANCESTOR_PATH = (
    REPO_ROOT
    / "artifacts/ppo_bc28init_currentgold10_tailfocus_replay2_ctx34q4_"
    "episode_mean_actorlr2x_u448meta_to_u456_seed20260736/B_gold_league/"
    "seed-20260736/checkpoints/update-0456.pt"
)
EXPECTED_ANCESTOR_SHA256 = (
    "b7ed9580543e4a2374ffcc618bb2eed74b90e762117a587768c90daf0019c83b"
)
EXPECTED_TRAIN_MEMBERS = tuple(
    f"train/part-{index:05d}.jsonl" for index in range(SPECIAL_STEPS)
)
EXPECTED_VALID_MEMBERS = tuple(
    f"valid/part-{index:05d}.jsonl" for index in range(len(CORE_POLICY_IDS))
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


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return bytes_sha256(payload)


def digest_lines(values: list[str] | set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return value


def stable_decision_identity(
    row: dict[str, Any],
    label: str,
) -> tuple[str, str, str]:
    dataset_date = row.get("dataset_date")
    episode_id = row.get("episode_id")
    if (
        not isinstance(dataset_date, str)
        or not dataset_date
        or "|" in dataset_date
    ):
        raise ValueError(f"{label}: invalid dataset_date {dataset_date!r}")
    if (
        not isinstance(episode_id, (str, int))
        or not str(episode_id)
        or "|" in str(episode_id)
    ):
        raise ValueError(f"{label}: invalid episode_id {episode_id!r}")
    seat = require_integer(row.get("seat"), f"{label}: seat")
    action_step = require_integer(
        row.get("action_step_index"),
        f"{label}: action_step_index",
    )
    if seat < 0 or action_step < 0:
        raise ValueError(f"{label}: seat/action_step_index must be non-negative")
    episode_text = str(episode_id)
    episode_key = f"{dataset_date}|{episode_text}"
    return (
        f"{episode_key}|{seat}|{action_step}",
        episode_text,
        episode_key,
    )


def resolve_source_path(raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Special archive source path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing or symlinked source archive: {path}")
    return path


def validate_sha(path: Path, expected: str, label: str) -> str:
    actual = repair_audit.file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def require_sha256(value: str, label: str) -> str:
    if len(value) != 64:
        raise ValueError(f"{label} is not a 64-character SHA256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value


def validate_parent_config(raw_config: Any) -> ppo.PPOConfig:
    if not isinstance(raw_config, dict):
        raise ValueError("Parent checkpoint has no PPO config")
    for name, expected in EXPECTED_PARENT_CONFIG.items():
        actual = raw_config.get(name)
        if actual != expected:
            raise ValueError(
                f"Parent config {name!r} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
    config = ppo.PPOConfig(**raw_config)
    replay_lr = config.learning_rate * config.bc_replay_lr_scale
    if not math.isclose(
        replay_lr,
        SPECIAL_LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "Parent replay learning rate does not resolve to the frozen "
            f"{SPECIAL_LEARNING_RATE}"
        )
    return config


def validate_request(args: argparse.Namespace) -> None:
    if args.steps != SPECIAL_STEPS:
        raise ValueError(f"This protocol requires exactly {SPECIAL_STEPS} steps")
    if args.expected_parent_update != 456:
        raise ValueError("The PokemonFan32 parent must retain update label 456")
    if args.seed != EXPECTED_TRAINING_SEED:
        raise ValueError(
            "This stacked protocol requires training seed 202608020"
        )
    if args.parent_checkpoint.is_symlink():
        raise ValueError("The frozen PokemonFan32 parent may not be a symlink")
    if args.parent_checkpoint.resolve() != EXPECTED_PARENT_PATH.resolve():
        raise ValueError("This protocol requires the exact PokemonFan32 parent path")
    if args.expected_parent_sha256 != EXPECTED_PARENT_SHA256:
        raise ValueError("This protocol requires the exact PokemonFan32 parent SHA256")
    if args.expected_replay_state_step != EXPECTED_REPLAY_STEP_BEFORE:
        raise ValueError(
            "This protocol must resume the embedded replay AdamW at step 48"
        )
    if args.expected_decision_key_count != (
        SPECIAL_STEPS * SPECIAL_BATCH_SIZE
    ):
        raise ValueError("The frozen decision-key count must be 5120")
    for name in (
        "expected_parent_sha256",
        "expected_general_bc_sha256",
        "expected_special_data_sha256",
        "expected_manifest_sha256",
        "expected_content_sha256",
        "expected_sources_sha256",
        "expected_decision_keys_digest",
        "expected_train_ppo_sha256",
        "expected_tool_sha256",
    ):
        require_sha256(str(getattr(args, name)), f"--{name.replace('_', '-')}")
    if args.expected_replay_cache_sha256 is not None:
        require_sha256(
            args.expected_replay_cache_sha256,
            "--expected-replay-cache-sha256",
        )


def validate_pokemonfan32_provenance(
    parent: dict[str, Any],
) -> dict[str, Any]:
    provenance = parent.get("post_ppo_special_bc")
    if not isinstance(provenance, dict):
        raise ValueError("PokemonFan32 parent has no post-special-BC provenance")
    expected_top_level = {
        "schema_version": EXPECTED_PARENT_PROVENANCE_SCHEMA,
        "mode": "special_bc",
        "status": "special_bc_completed",
        "checkpoint_update_label": 456,
        "not_a_new_ppo_update": True,
    }
    if any(provenance.get(key) != value for key, value in expected_top_level.items()):
        raise ValueError("PokemonFan32 provenance header drifted")

    ancestor = provenance.get("parent")
    if not isinstance(ancestor, dict):
        raise ValueError("PokemonFan32 provenance has no U456 ancestor")
    if (
        Path(str(ancestor.get("path", ""))).resolve()
        != EXPECTED_ANCESTOR_PATH.resolve()
        or ancestor.get("sha256") != EXPECTED_ANCESTOR_SHA256
        or ancestor.get("update") != 456
        or ancestor.get("model_state_sha256")
        != "e9baf1917d24aeaab66341835daf86105d071caab9b2985082f921611db18b2c"
    ):
        raise ValueError("PokemonFan32 U456 ancestor binding drifted")
    ancestor_hash = validate_sha(
        EXPECTED_ANCESTOR_PATH,
        EXPECTED_ANCESTOR_SHA256,
        "PokemonFan32 U456 ancestor",
    )

    special = provenance.get("special_bc")
    if not isinstance(special, dict) or (
        special.get("seed") != 202608012
        or special.get("steps_requested") != 32
        or special.get("rows_per_batch") != 256
        or special.get("context34_rows_per_batch") != 1
        or special.get("loss") != "ordered"
        or special.get("order_context_weight") != 8.0
        or special.get("trainable_scope") != "last_block_heads"
        or not math.isclose(
            float(special.get("learning_rate", float("nan"))),
            SPECIAL_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("PokemonFan32 special-BC protocol drifted")
    batch_indices = special.get("batch_indices")
    if (
        not isinstance(batch_indices, list)
        or len(batch_indices) != 32
        or len(set(batch_indices)) != 32
        or set(batch_indices) != set(range(32))
    ):
        raise ValueError("PokemonFan32 batch sequence is not a 32-batch permutation")

    optimizer = provenance.get("optimizer")
    integrity = provenance.get("integrity")
    if not isinstance(optimizer, dict) or not isinstance(integrity, dict):
        raise ValueError("PokemonFan32 optimizer/integrity provenance is absent")
    replay_steps_before = optimizer.get("replay_steps_before")
    replay_steps_after = integrity.get("replay_steps_after")
    if (
        not isinstance(replay_steps_before, list)
        or len(replay_steps_before) != 24
        or set(replay_steps_before) != {16}
        or not isinstance(replay_steps_after, list)
        or len(replay_steps_after) != 24
        or set(replay_steps_after) != {48}
        or integrity.get("optimizer_steps") != 32
        or integrity.get("rows") != 8192
        or integrity.get("context34_rows") != 32
        or integrity.get("model_state_sha256_after")
        != EXPECTED_PARENT_MODEL_SHA256
        or integrity.get("replay_state_sha256_after")
        != EXPECTED_PARENT_REPLAY_SHA256
        or integrity.get("ppo_state_sha256_after")
        != EXPECTED_PARENT_PPO_SHA256
        or integrity.get("changed_parameters_subset_of_actor") is not True
        or integrity.get("value_head_parameters_unchanged") is not True
        or integrity.get("ppo_optimizer_state_unchanged") is not True
        or integrity.get("parent_checkpoint_unchanged") is not True
        or integrity.get("parent_checkpoint_sha256_after")
        != EXPECTED_ANCESTOR_SHA256
    ):
        raise ValueError("PokemonFan32 training-integrity provenance drifted")
    prefix = integrity.get("frozen_16_step_prefix")
    if not isinstance(prefix, dict) or prefix.get("exact_match") is not True:
        raise ValueError("PokemonFan32 did not retain its frozen 16-step prefix")
    actor_names = optimizer.get("actor_parameter_names")
    if (
        not isinstance(actor_names, list)
        or len(actor_names) != 24
        or sorted(integrity.get("changed_parameter_names", []))
        != sorted(actor_names)
    ):
        raise ValueError("PokemonFan32 changed-actor manifest drifted")

    if repair_audit.nested_sha256(
        parent.get("bc_replay_optimizer_state_dict")
    ) != EXPECTED_PARENT_REPLAY_SHA256:
        raise ValueError("PokemonFan32 embedded replay optimizer is not provenance-bound")
    if repair_audit.nested_sha256(
        parent.get("optimizer_state_dict")
    ) != EXPECTED_PARENT_PPO_SHA256:
        raise ValueError("PokemonFan32 embedded PPO optimizer is not provenance-bound")

    sources = provenance.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("PokemonFan32 provenance has no source bindings")
    expected_sources = {
        "general_bc_checkpoint": (
            REPO_ROOT
            / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_"
            "seed20260922_20260731/best.pt",
            "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb",
        ),
        "special_bc_archive": (
            REPO_ROOT
            / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_"
            "valid29_special_20260801.zip",
            "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
        ),
        "tool": (
            REPO_ROOT / "tools/run_ppo_special_bc_32.py",
            "daaca8902819b56a3aff2c8a1cab802dc537c0cec9df59569c1e118ee801d661",
        ),
        "train_ppo": (
            REPO_ROOT / "tools/train_ppo.py",
            "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3",
        ),
    }
    frozen_files: dict[str, str] = {
        str(EXPECTED_ANCESTOR_PATH.resolve()): ancestor_hash,
    }
    source_report: dict[str, dict[str, str]] = {}
    for name, (expected_path, expected_sha) in expected_sources.items():
        record = sources.get(name)
        if (
            not isinstance(record, dict)
            or Path(str(record.get("path", ""))).resolve()
            != expected_path.resolve()
            or record.get("sha256") != expected_sha
        ):
            raise ValueError(f"PokemonFan32 provenance source {name!r} drifted")
        actual_sha = validate_sha(expected_path, expected_sha, f"PokemonFan32 {name}")
        frozen_files[str(expected_path.resolve())] = actual_sha
        source_report[name] = {
            "path": str(expected_path.resolve()),
            "sha256": actual_sha,
        }
    return {
        "validated": True,
        "schema_version": provenance["schema_version"],
        "parent_checkpoint_sha256": EXPECTED_PARENT_SHA256,
        "parent_model_state_sha256": EXPECTED_PARENT_MODEL_SHA256,
        "replay_adamw_step": 48,
        "replay_state_sha256": EXPECTED_PARENT_REPLAY_SHA256,
        "ppo_state_sha256": EXPECTED_PARENT_PPO_SHA256,
        "ancestor": {
            "path": str(EXPECTED_ANCESTOR_PATH.resolve()),
            "sha256": ancestor_hash,
            "update": 456,
        },
        "provenance_sources": source_report,
        "frozen_files": frozen_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue PokemonFan32's embedded actor BC optimizer on the "
            "frozen, batch-aligned Gold19 core-five archive."
        )
    )
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--expected-parent-update", type=int, default=456)
    parser.add_argument("--general-bc-checkpoint", type=Path)
    parser.add_argument(
        "--expected-general-bc-sha256",
        "--expected-bc-checkpoint-sha256",
        dest="expected_general_bc_sha256",
        required=True,
    )
    parser.add_argument("--special-data", type=Path, required=True)
    parser.add_argument("--expected-special-data-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-content-sha256", required=True)
    parser.add_argument("--expected-sources-sha256", required=True)
    parser.add_argument("--expected-decision-keys-digest", required=True)
    parser.add_argument(
        "--expected-decision-key-count",
        type=int,
        default=SPECIAL_STEPS * SPECIAL_BATCH_SIZE,
    )
    parser.add_argument("--expected-train-ppo-sha256", required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--expected-replay-cache-sha256")
    parser.add_argument(
        "--expected-replay-state-step",
        type=int,
        default=EXPECTED_REPLAY_STEP_BEFORE,
    )
    parser.add_argument(
        "--seed",
        "--special-seed",
        dest="seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--steps",
        "--repair-steps",
        dest="steps",
        type=int,
        default=SPECIAL_STEPS,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def positive_count_map(raw: Any, label: str) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    result: dict[str, int] = {}
    for raw_key, raw_value in raw.items():
        key = str(raw_key)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(f"{label}[{key!r}] must be an integer")
        if raw_value < 0:
            raise ValueError(f"{label}[{key!r}] must be non-negative")
        if raw_value:
            result[key] = raw_value
    return dict(sorted(result.items()))


def member_records(raw: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list")
    records: dict[str, dict[str, Any]] = {}
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError(f"{label} contains a non-object record")
        name = record.get("member")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} record has no member name")
        if name in records:
            raise ValueError(f"{label} contains duplicate member {name!r}")
        records[name] = record
    return records


def archive_content_manifest(
    archive: zipfile.ZipFile,
    jsonl_members: tuple[str, ...],
) -> tuple[str, dict[str, str]]:
    """Hash uncompressed JSONL content using the archive schema contract."""
    content_digest = hashlib.sha256()
    member_hashes: dict[str, str] = {}
    for member in sorted(jsonl_members):
        content_digest.update(member.encode("utf-8"))
        content_digest.update(b"\0")
        member_digest = hashlib.sha256()
        with archive.open(member) as handle:
            while chunk := handle.read(1024 * 1024):
                content_digest.update(chunk)
                member_digest.update(chunk)
        member_hashes[member] = member_digest.hexdigest()
    return content_digest.hexdigest(), member_hashes


def validate_archive_manifest(
    archive_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str]]:
    with zipfile.ZipFile(archive_path) as archive:
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)):
            raise ValueError("Special archive contains duplicate ZIP member names")
        if names.count("manifest.json") != 1:
            raise ValueError("Special archive must contain one manifest.json")
        jsonl_members = tuple(sorted(name for name in names if name.endswith(".jsonl")))
        expected_jsonl = tuple(sorted(EXPECTED_TRAIN_MEMBERS + EXPECTED_VALID_MEMBERS))
        if jsonl_members != expected_jsonl:
            raise ValueError(
                "Special archive JSONL members are not the frozen 20-train/"
                "5-valid layout"
            )

        manifest_bytes = archive.read("manifest.json")
        manifest_hash = bytes_sha256(manifest_bytes)
        if manifest_hash != args.expected_manifest_sha256:
            raise ValueError(
                "Special manifest SHA256 mismatch: expected "
                f"{args.expected_manifest_sha256}, got {manifest_hash}"
            )
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Special archive manifest is not valid JSON") from error
        if not isinstance(manifest, dict):
            raise ValueError("Special archive manifest must be an object")
        if manifest.get("schema_version") != ARCHIVE_SCHEMA:
            raise ValueError("Special archive schema_version mismatch")
        if manifest.get("seed") != EXPECTED_ARCHIVE_SEED:
            raise ValueError(
                "Special archive selection seed is not the frozen 202608017"
            )

        sources = manifest.get("sources")
        if not isinstance(sources, list):
            raise ValueError("Special archive manifest has no sources list")
        source_ids = tuple(
            source.get("policy_id") if isinstance(source, dict) else None
            for source in sources
        )
        if source_ids != CORE_POLICY_IDS:
            raise ValueError("Special archive source policy order/identity drifted")
        for source in sources:
            for key in (
                "policy_id",
                "team_name",
                "source_submission_id",
                "deck_hash",
                "path",
                "sha256",
            ):
                if key not in source:
                    raise ValueError(f"Special archive source lacks {key!r}")
            require_sha256(str(source["sha256"]), "source sha256")
        sources_hash = canonical_json_sha256(sources)
        if sources_hash != args.expected_sources_sha256:
            raise ValueError(
                "Special sources SHA256 mismatch: expected "
                f"{args.expected_sources_sha256}, got {sources_hash}"
            )
        if manifest.get("sources_sha256") != sources_hash:
            raise ValueError("Manifest sources_sha256 does not bind sources")

        content_hash, actual_member_hashes = archive_content_manifest(
            archive,
            expected_jsonl,
        )
        if content_hash != args.expected_content_sha256:
            raise ValueError(
                "Special content SHA256 mismatch: expected "
                f"{args.expected_content_sha256}, got {content_hash}"
            )
        if manifest.get("content_sha256") != content_hash:
            raise ValueError("Manifest content_sha256 does not bind JSONL content")

    train = manifest.get("train")
    valid = manifest.get("valid")
    if not isinstance(train, dict) or not isinstance(valid, dict):
        raise ValueError("Special manifest must contain train and valid objects")
    train_records = member_records(train.get("members"), "train.members")
    valid_records = member_records(valid.get("members"), "valid.members")
    if tuple(sorted(train_records)) != EXPECTED_TRAIN_MEMBERS:
        raise ValueError("Manifest train member sequence/set drifted")
    if tuple(sorted(valid_records)) != EXPECTED_VALID_MEMBERS:
        raise ValueError("Manifest valid member sequence/set drifted")
    for member, record in {**train_records, **valid_records}.items():
        require_sha256(str(record.get("sha256", "")), f"{member} sha256")
        if record["sha256"] != actual_member_hashes[member]:
            raise ValueError(f"Manifest member SHA256 mismatch for {member}")

    decision_keys = manifest.get("decision_keys")
    if not isinstance(decision_keys, dict):
        raise ValueError("Special manifest has no decision_keys object")
    if decision_keys.get("format") != DECISION_KEY_FORMAT:
        raise ValueError("Special decision-key format drifted")
    if decision_keys.get("count") != args.expected_decision_key_count:
        raise ValueError("Special decision-key count drifted")
    if decision_keys.get("duplicate_count") != 0:
        raise ValueError("Special training selection contains duplicate decisions")
    if decision_keys.get("without_replacement") is not True:
        raise ValueError("Special decisions were not selected without replacement")
    if decision_keys.get("digest") != args.expected_decision_keys_digest:
        raise ValueError("Special decision-key digest does not match the request")

    expected_total_team_counts = {
        policy_id: SPECIAL_ROWS_PER_POLICY for policy_id in CORE_POLICY_IDS
    }
    expected_total_context_counts = {
        policy_id: SPECIAL_CONTEXT34_PER_POLICY for policy_id in CORE_POLICY_IDS
    }
    if train.get("total_rows") != SPECIAL_STEPS * SPECIAL_BATCH_SIZE:
        raise ValueError("Manifest train.total_rows must be 5120")
    if train.get("total_context34_rows") != SPECIAL_STEPS:
        raise ValueError("Manifest train.total_context34_rows must be 20")
    if positive_count_map(
        train.get("total_team_counts"), "train.total_team_counts"
    ) != expected_total_team_counts:
        raise ValueError("Manifest total per-policy row quotas drifted")
    if positive_count_map(
        train.get("total_context34_team_counts"),
        "train.total_context34_team_counts",
    ) != expected_total_context_counts:
        raise ValueError("Manifest total per-policy context-34 quotas drifted")
    return manifest, actual_member_hashes


def raw_context(row: dict[str, Any]) -> int:
    observation = row.get("observation") or {}
    select = observation.get("select") if isinstance(observation, dict) else None
    if not isinstance(select, dict):
        raise ValueError("Training row has no observation.select object")
    return int(select.get("context", row.get("select_context", 0)) or 0)


def validate_row_identity(
    row: dict[str, Any],
    source: dict[str, Any],
    split: str,
    label: str,
) -> tuple[str, str, str, bool]:
    expected = {
        "schema_version": ROW_SCHEMA,
        "split": split,
        "policy_id": source["policy_id"],
        "team_name": source["team_name"],
        "deck_hash": source["deck_hash"],
        "source_submission_id": source["source_submission_id"],
    }
    mismatches = [
        f"{key}={row.get(key)!r} expected {wanted!r}"
        for key, wanted in expected.items()
        if row.get(key) != wanted
    ]
    if mismatches:
        raise ValueError(f"{label}: identity mismatch: {'; '.join(mismatches)}")
    if "visualize" in row or (
        isinstance(row.get("observation"), dict)
        and "visualize" in row["observation"]
    ):
        raise ValueError(f"{label}: hidden visualize payload is forbidden")
    decision_key, episode_id, episode_key = stable_decision_identity(row, label)
    context = raw_context(row)
    select_context = row.get("select_context")
    try:
        declared_context = int(select_context)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid select_context") from error
    if declared_context != context:
        raise ValueError(f"{label}: select_context disagrees with observation")
    return decision_key, episode_id, episode_key, context == ppo.SKILL_ORDER_CONTEXT


def validate_sample_weight(row: dict[str, Any]) -> None:
    if "sample_weight" not in row:
        raise ValueError("Training row omits the frozen sample_weight")
    weight = row["sample_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("Training row sample_weight is not numeric")
    if not math.isfinite(float(weight)) or float(weight) != 1.0:
        raise ValueError("Every special-BC sample_weight must equal 1.0")


def source_members(archive: zipfile.ZipFile, split: str) -> list[str]:
    members = sorted(
        name
        for name in archive.namelist()
        if name.startswith(f"{split}/") and name.endswith(".jsonl")
    )
    if not members:
        raise ValueError(f"Source archive has no {split} JSONL members")
    return members


def audit_source_archives(manifest: dict[str, Any]) -> dict[str, Any]:
    """Independently recompute source identities and exclusion statistics."""
    states: list[dict[str, Any]] = []
    global_valid_decisions: set[str] = set()
    global_valid_episode_ids: set[str] = set()
    global_valid_episode_keys: set[str] = set()

    for source in manifest["sources"]:
        path = resolve_source_path(source["path"])
        expected_sha = str(source["sha256"])
        actual_sha = validate_sha(path, expected_sha, f"source {source['policy_id']}")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"Source archive has duplicate members: {path}")
            if names.count("manifest.json") != 1:
                raise ValueError(f"Source archive has no unique manifest: {path}")
            source_manifest_payload = archive.read("manifest.json")
            source_manifest_hash = bytes_sha256(source_manifest_payload)
            if source_manifest_hash != source.get("manifest_sha256"):
                raise ValueError(
                    f"Source manifest SHA256 mismatch for {source['policy_id']}"
                )
            try:
                source_manifest = json.loads(source_manifest_payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid source manifest: {path}") from error
            if source_manifest.get("schema_version") != SOURCE_SCHEMA:
                raise ValueError(f"Source schema drifted: {path}")
            if source.get("schema_version") != SOURCE_SCHEMA:
                raise ValueError(
                    f"Balanced manifest source schema drifted: {source['policy_id']}"
                )
            nested_policy = source.get("policy")
            if not isinstance(nested_policy, dict):
                raise ValueError("Balanced source record has no policy identity")
            expected_identity = {
                "policy_id": source["policy_id"],
                "rank": nested_policy.get("rank"),
                "team_name": source["team_name"],
                "submission_id": source["source_submission_id"],
                "deck_hash": source["deck_hash"],
            }
            if nested_policy != expected_identity:
                raise ValueError(
                    f"Balanced nested policy identity drifted: {source['policy_id']}"
                )
            source_policy = source_manifest.get("policy")
            if not isinstance(source_policy, dict) or {
                key: source_policy.get(key) for key in expected_identity
            } != expected_identity:
                raise ValueError(
                    f"Source manifest policy identity drifted: {source['policy_id']}"
                )
            split_decisions = source_manifest.get("split_decisions")
            episode_ids = source_manifest.get("episode_ids")
            if not isinstance(split_decisions, dict) or not isinstance(
                episode_ids,
                dict,
            ):
                raise ValueError(f"Source split metadata is absent: {path}")
            members = {
                split: source_members(archive, split)
                for split in ("train", "valid")
            }
            valid_rows = 0
            valid_context34 = 0
            valid_episode_ids: set[str] = set()
            valid_episode_keys: set[str] = set()
            valid_decisions: set[str] = set()
            for member in members["valid"]:
                with archive.open(member) as handle:
                    for line_number, raw_line in enumerate(handle, start=1):
                        try:
                            row = json.loads(raw_line)
                        except (UnicodeDecodeError, json.JSONDecodeError) as error:
                            raise ValueError(
                                f"{path}:{member}:{line_number}: invalid JSON"
                            ) from error
                        if not isinstance(row, dict):
                            raise ValueError(
                                f"{path}:{member}:{line_number}: row is not an object"
                            )
                        key, episode_id, episode_key, is_context34 = (
                            validate_row_identity(
                                row,
                                source,
                                "valid",
                                f"{path}:{member}:{line_number}",
                            )
                        )
                        if key in valid_decisions or key in global_valid_decisions:
                            raise ValueError(f"Duplicate source valid decision: {key}")
                        valid_decisions.add(key)
                        global_valid_decisions.add(key)
                        valid_episode_ids.add(episode_id)
                        valid_episode_keys.add(episode_key)
                        global_valid_episode_ids.add(episode_id)
                        global_valid_episode_keys.add(episode_key)
                        valid_rows += 1
                        valid_context34 += int(is_context34)
        expected_valid_ids = {str(value) for value in episode_ids.get("valid", [])}
        if valid_episode_ids != expected_valid_ids:
            raise ValueError(
                f"Source valid episode set drifted: {source['policy_id']}"
            )
        if split_decisions.get("valid") != valid_rows:
            raise ValueError(
                f"Source valid row count drifted: {source['policy_id']}"
            )
        source_counts = source.get("source_counts")
        if not isinstance(source_counts, dict):
            raise ValueError("Balanced source record has no source_counts")
        expected_valid_counts = {
            "valid_rows": valid_rows,
            "valid_episodes": len(valid_episode_ids),
            "valid_context34_rows": valid_context34,
        }
        if any(
            source_counts.get(key) != value
            for key, value in expected_valid_counts.items()
        ):
            raise ValueError(
                f"Balanced source valid counts drifted: {source['policy_id']}"
            )
        states.append(
            {
                "record": source,
                "path": path,
                "sha256": actual_sha,
                "manifest": source_manifest,
                "members": members,
                "valid_rows": valid_rows,
                "valid_context34_rows": valid_context34,
                "valid_episode_ids": valid_episode_ids,
                "valid_episode_keys": valid_episode_keys,
            }
        )

    global_train_decisions: set[str] = set()
    exclusion_by_policy: dict[str, dict[str, int]] = {}
    source_reports: list[dict[str, Any]] = []
    for state in states:
        source = state["record"]
        path = state["path"]
        train_rows = 0
        train_context34 = 0
        train_episode_ids: set[str] = set()
        train_episode_keys: set[str] = set()
        local_train_decisions: set[str] = set()
        excluded_rows = 0
        excluded_context34 = 0
        excluded_episode_ids: set[str] = set()
        eligible_ordinary = 0
        eligible_context34 = 0
        with zipfile.ZipFile(path) as archive:
            for member in state["members"]["train"]:
                with archive.open(member) as handle:
                    for line_number, raw_line in enumerate(handle, start=1):
                        try:
                            row = json.loads(raw_line)
                        except (UnicodeDecodeError, json.JSONDecodeError) as error:
                            raise ValueError(
                                f"{path}:{member}:{line_number}: invalid JSON"
                            ) from error
                        if not isinstance(row, dict):
                            raise ValueError(
                                f"{path}:{member}:{line_number}: row is not an object"
                            )
                        key, episode_id, episode_key, is_context34 = (
                            validate_row_identity(
                                row,
                                source,
                                "train",
                                f"{path}:{member}:{line_number}",
                            )
                        )
                        if (
                            key in local_train_decisions
                            or key in global_train_decisions
                        ):
                            raise ValueError(f"Duplicate source train decision: {key}")
                        local_train_decisions.add(key)
                        global_train_decisions.add(key)
                        train_episode_ids.add(episode_id)
                        train_episode_keys.add(episode_key)
                        train_rows += 1
                        train_context34 += int(is_context34)
                        if episode_id in global_valid_episode_ids:
                            excluded_rows += 1
                            excluded_context34 += int(is_context34)
                            excluded_episode_ids.add(episode_id)
                        elif is_context34:
                            eligible_context34 += 1
                        else:
                            eligible_ordinary += 1

        source_manifest = state["manifest"]
        expected_train_ids = {
            str(value)
            for value in source_manifest["episode_ids"].get("train", [])
        }
        if train_episode_ids != expected_train_ids:
            raise ValueError(
                f"Source train episode set drifted: {source['policy_id']}"
            )
        if source_manifest["split_decisions"].get("train") != train_rows:
            raise ValueError(
                f"Source train row count drifted: {source['policy_id']}"
            )
        if train_episode_ids & state["valid_episode_ids"]:
            raise ValueError(
                f"Source train/valid episodes overlap: {source['policy_id']}"
            )
        source_counts = source["source_counts"]
        if (
            source_counts.get("train_rows") != train_rows
            or source_counts.get("train_episodes") != len(train_episode_ids)
        ):
            raise ValueError(
                f"Balanced source train counts drifted: {source['policy_id']}"
            )
        expected_exclusion = {
            "rows": excluded_rows,
            "ordinary_rows": excluded_rows - excluded_context34,
            "context34_rows": excluded_context34,
            "episodes": len(excluded_episode_ids),
        }
        if source.get("global_valid_exclusion") != expected_exclusion:
            raise ValueError(
                f"Balanced source exclusion counts drifted: {source['policy_id']}"
            )
        expected_eligible = {
            "ordinary_rows": eligible_ordinary,
            "context34_rows": eligible_context34,
        }
        if source.get("eligible_train") != expected_eligible:
            raise ValueError(
                f"Balanced source eligible counts drifted: {source['policy_id']}"
            )
        expected_selected = {
            "ordinary_rows": SPECIAL_STEPS * 51,
            "context34_rows": SPECIAL_CONTEXT34_PER_POLICY,
            "rows": SPECIAL_ROWS_PER_POLICY,
        }
        if source.get("selected_train") != expected_selected:
            raise ValueError(
                f"Balanced source selected counts drifted: {source['policy_id']}"
            )
        exclusion_by_policy[source["policy_id"]] = expected_exclusion
        source_reports.append(
            {
                "policy_id": source["policy_id"],
                "path": str(path),
                "sha256": state["sha256"],
                "train_rows": train_rows,
                "train_episodes": len(train_episode_ids),
                "valid_rows": state["valid_rows"],
                "valid_episodes": len(state["valid_episode_ids"]),
                "valid_context34_rows": state["valid_context34_rows"],
                "excluded": expected_exclusion,
                "eligible_train": expected_eligible,
                "selected_train": expected_selected,
            }
        )

    exclusion = manifest.get("exclusion")
    if not isinstance(exclusion, dict):
        raise ValueError("Balanced manifest has no exclusion audit")
    aggregate_excluded_rows = sum(
        counts["rows"] for counts in exclusion_by_policy.values()
    )
    aggregate_excluded_context = sum(
        counts["context34_rows"] for counts in exclusion_by_policy.values()
    )
    if (
        exclusion.get("global_valid_episode_union_count")
        != len(global_valid_episode_ids)
        or exclusion.get("excluded_train_rows") != aggregate_excluded_rows
        or exclusion.get("excluded_train_context34_rows")
        != aggregate_excluded_context
        or exclusion.get("by_policy") != exclusion_by_policy
    ):
        raise ValueError("Balanced manifest aggregate exclusion audit drifted")
    episode_isolation = manifest.get("episode_isolation")
    if not isinstance(episode_isolation, dict):
        raise ValueError("Balanced manifest has no episode_isolation audit")
    if (
        episode_isolation.get("global_valid_episode_count")
        != len(global_valid_episode_ids)
        or episode_isolation.get("global_valid_episode_digest")
        != digest_lines(global_valid_episode_ids)
    ):
        raise ValueError("Balanced manifest global valid episode audit drifted")
    return {
        "source_files": {
            str(state["path"]): state["sha256"] for state in states
        },
        "global_valid_episode_ids": global_valid_episode_ids,
        "global_valid_episode_keys": global_valid_episode_keys,
        "report": {
            "sources": source_reports,
            "global_valid_episode_union_count": len(global_valid_episode_ids),
            "global_valid_episode_digest": digest_lines(
                global_valid_episode_ids
            ),
            "global_valid_dated_episode_count": len(
                global_valid_episode_keys
            ),
            "global_valid_dated_episode_digest": digest_lines(
                global_valid_episode_keys
            ),
            "excluded_train_rows": aggregate_excluded_rows,
            "excluded_train_context34_rows": aggregate_excluded_context,
            "exclusion_by_policy": exclusion_by_policy,
            "all_source_archives_opened_and_hashed": True,
            "all_source_manifests_and_rows_revalidated": True,
        },
    }


def load_batch_aligned_archive(
    archive_path: Path,
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    source_audit: dict[str, Any],
) -> tuple[
    list[dict[str, torch.Tensor]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Load one archive member into one batch without shuffling/stratifying."""
    sources = {source["policy_id"]: source for source in manifest["sources"]}
    records = member_records(manifest["train"]["members"], "train.members")
    batches: list[dict[str, torch.Tensor]] = []
    metadata: list[dict[str, Any]] = []
    total_team_counts: Counter[str] = Counter()
    total_context_counts: Counter[str] = Counter()
    train_decision_keys: list[str] = []
    train_episode_ids: set[str] = set()
    train_episode_keys: set[str] = set()

    with zipfile.ZipFile(archive_path) as archive:
        for batch_index, member in enumerate(EXPECTED_TRAIN_MEMBERS):
            raw_payload = archive.read(member)
            raw_lines = raw_payload.splitlines()
            if len(raw_lines) != SPECIAL_BATCH_SIZE or any(not line for line in raw_lines):
                raise ValueError(f"{member} must contain exactly 256 non-empty rows")
            features: list[dict[str, Any]] = []
            team_counts: Counter[str] = Counter()
            context_counts: Counter[str] = Counter()
            for line_index, raw_line in enumerate(raw_lines, start=1):
                try:
                    row = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"{member}:{line_index} is not valid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"{member}:{line_index} is not an object")
                validate_sample_weight(row)
                policy_id = row.get("policy_id")
                if policy_id not in sources:
                    raise ValueError(
                        f"{member}:{line_index} has an unknown policy_id"
                    )
                source = sources[policy_id]
                decision_key, episode_id, episode_key, is_context34 = (
                    validate_row_identity(
                        row,
                        source,
                        "train",
                        f"{member}:{line_index}",
                    )
                )
                train_decision_keys.append(decision_key)
                train_episode_ids.add(episode_id)
                train_episode_keys.add(episode_key)
                team_counts[policy_id] += 1
                total_team_counts[policy_id] += 1
                if is_context34:
                    context_counts[policy_id] += 1
                    total_context_counts[policy_id] += 1
                feature = ppo.featurize_row(
                    row,
                    hash_size=int(model_config["hash_size"]),
                    max_state_entities=int(model_config["max_state_entities"]),
                )
                if feature is None:
                    raise ValueError(
                        f"{member}:{line_index} was rejected by PPO featurize_row"
                    )
                if float(feature["sample_weight"]) != 1.0:
                    raise ValueError("PPO featurization changed sample_weight")
                features.append(feature)

            rotating_policy = CORE_POLICY_IDS[batch_index % len(CORE_POLICY_IDS)]
            expected_team_counts = {
                policy_id: 51 + int(policy_id == rotating_policy)
                for policy_id in CORE_POLICY_IDS
            }
            expected_context_counts = {rotating_policy: 1}
            actual_team_counts = dict(sorted(team_counts.items()))
            actual_context_counts = dict(sorted(context_counts.items()))
            if actual_team_counts != expected_team_counts:
                raise ValueError(f"{member} per-policy batch quota drifted")
            if actual_context_counts != expected_context_counts:
                raise ValueError(f"{member} context-34 rotation quota drifted")

            record = records[member]
            if record.get("rows") != SPECIAL_BATCH_SIZE:
                raise ValueError(f"Manifest row count drifted for {member}")
            if record.get("context34_rows") != 1:
                raise ValueError(f"Manifest context-34 count drifted for {member}")
            if positive_count_map(record.get("team_counts"), f"{member}.team_counts") != actual_team_counts:
                raise ValueError(f"Manifest policy counts drifted for {member}")
            if positive_count_map(
                record.get("context34_team_counts"),
                f"{member}.context34_team_counts",
            ) != actual_context_counts:
                raise ValueError(f"Manifest context policy counts drifted for {member}")

            batch = ppo.collate_decisions(
                features,
                max_state_entities=int(model_config["max_state_entities"]),
                entity_fields=int(model_config["entity_fields"]),
                option_fields=int(model_config["option_fields"]),
            )
            if int(batch["action_counts"].shape[0]) != SPECIAL_BATCH_SIZE:
                raise ValueError(f"PPO collate produced a short batch for {member}")
            if int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum()) != 1:
                raise ValueError(f"PPO collate changed the context-34 quota for {member}")
            if not torch.equal(
                batch["sample_weights"],
                torch.ones_like(batch["sample_weights"]),
            ):
                raise ValueError(f"PPO collate changed sample weights for {member}")
            batches.append(batch)
            metadata.append(
                {
                    "batch_index": batch_index,
                    "member": member,
                    "rows": SPECIAL_BATCH_SIZE,
                    "context34_rows": 1,
                    "team_counts": actual_team_counts,
                    "context34_team_counts": actual_context_counts,
                }
            )

    expected_total_counts = Counter(
        {policy_id: SPECIAL_ROWS_PER_POLICY for policy_id in CORE_POLICY_IDS}
    )
    expected_total_contexts = Counter(
        {policy_id: SPECIAL_CONTEXT34_PER_POLICY for policy_id in CORE_POLICY_IDS}
    )
    if total_team_counts != expected_total_counts:
        raise ValueError("Loaded archive did not preserve 1024 rows per policy")
    if total_context_counts != expected_total_contexts:
        raise ValueError("Loaded archive did not preserve four context-34 rows per policy")

    train_unique_keys = set(train_decision_keys)
    train_duplicate_count = len(train_decision_keys) - len(train_unique_keys)
    train_digest = digest_lines(train_decision_keys)
    decision_manifest = manifest["decision_keys"]
    if (
        len(train_decision_keys) != SPECIAL_STEPS * SPECIAL_BATCH_SIZE
        or train_duplicate_count != 0
        or train_digest != decision_manifest.get("digest")
        or train_digest != decision_manifest.get("train", {}).get("digest")
        or len(train_decision_keys)
        != decision_manifest.get("train", {}).get("count")
        or train_duplicate_count
        != decision_manifest.get("train", {}).get("duplicate_count")
    ):
        raise ValueError("Raw train decision-key recomputation failed")

    valid_records = member_records(manifest["valid"]["members"], "valid.members")
    valid_decision_keys: list[str] = []
    valid_episode_ids: set[str] = set()
    valid_episode_keys: set[str] = set()
    valid_team_counts: Counter[str] = Counter()
    valid_context_counts: Counter[str] = Counter()
    valid_member_audit: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for source_index, member in enumerate(EXPECTED_VALID_MEMBERS):
            source = manifest["sources"][source_index]
            expected_policy_id = CORE_POLICY_IDS[source_index]
            if source["policy_id"] != expected_policy_id:
                raise ValueError("Valid member/source policy order drifted")
            member_rows = 0
            member_context34 = 0
            member_decision_keys: list[str] = []
            member_episode_ids: set[str] = set()
            with archive.open(member) as handle:
                for line_index, raw_line in enumerate(handle, start=1):
                    try:
                        row = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(
                            f"{member}:{line_index} is not valid JSON"
                        ) from error
                    if not isinstance(row, dict):
                        raise ValueError(f"{member}:{line_index} is not an object")
                    validate_sample_weight(row)
                    key, episode_id, episode_key, is_context34 = (
                        validate_row_identity(
                            row,
                            source,
                            "valid",
                            f"{member}:{line_index}",
                        )
                    )
                    member_rows += 1
                    member_context34 += int(is_context34)
                    member_decision_keys.append(key)
                    member_episode_ids.add(episode_id)
                    valid_decision_keys.append(key)
                    valid_episode_ids.add(episode_id)
                    valid_episode_keys.add(episode_key)
                    valid_team_counts[expected_policy_id] += 1
                    valid_context_counts[expected_policy_id] += int(is_context34)
            record = valid_records[member]
            expected_member_team_counts = {expected_policy_id: member_rows}
            expected_member_context_counts = (
                {expected_policy_id: member_context34}
                if member_context34
                else {}
            )
            if (
                record.get("policy_id") != expected_policy_id
                or record.get("rows") != member_rows
                or record.get("context34_rows") != member_context34
                or record.get("episodes") != len(member_episode_ids)
                or positive_count_map(
                    record.get("team_counts"),
                    f"{member}.team_counts",
                )
                != expected_member_team_counts
                or positive_count_map(
                    record.get("context34_team_counts"),
                    f"{member}.context34_team_counts",
                )
                != expected_member_context_counts
                or record.get("decision_keys_digest")
                != digest_lines(member_decision_keys)
            ):
                raise ValueError(f"Manifest valid member audit drifted for {member}")
            valid_member_audit.append(
                {
                    "member": member,
                    "policy_id": expected_policy_id,
                    "rows": member_rows,
                    "context34_rows": member_context34,
                    "episodes": len(member_episode_ids),
                    "decision_keys_digest": digest_lines(member_decision_keys),
                }
            )

    valid_unique_keys = set(valid_decision_keys)
    valid_duplicate_count = len(valid_decision_keys) - len(valid_unique_keys)
    valid_digest = digest_lines(valid_decision_keys)
    global_keys = train_unique_keys | valid_unique_keys
    global_duplicate_count = (
        len(train_decision_keys) + len(valid_decision_keys) - len(global_keys)
    )
    global_digest = digest_lines(global_keys)
    if len(valid_decision_keys) != EXPECTED_VALID_ROWS:
        raise ValueError("Raw valid row count is not the frozen 8319")
    if sum(valid_context_counts.values()) != EXPECTED_VALID_CONTEXT34_ROWS:
        raise ValueError("Raw valid context-34 count is not the frozen 43")
    if valid_duplicate_count != 0 or global_duplicate_count != 0:
        raise ValueError("Raw train/valid archive decision keys overlap or repeat")
    if (
        decision_manifest.get("valid", {}).get("digest") != valid_digest
        or decision_manifest.get("valid", {}).get("count")
        != len(valid_decision_keys)
        or decision_manifest.get("valid", {}).get("duplicate_count") != 0
        or decision_manifest.get("global", {}).get("digest") != global_digest
        or decision_manifest.get("global", {}).get("count") != len(global_keys)
        or decision_manifest.get("global", {}).get("duplicate_count") != 0
    ):
        raise ValueError("Raw valid/global decision-key recomputation failed")
    if valid_episode_ids != source_audit["global_valid_episode_ids"]:
        raise ValueError("Output valid episode IDs differ from source valid union")
    if valid_episode_keys != source_audit["global_valid_episode_keys"]:
        raise ValueError("Output dated valid episodes differ from source union")
    episode_overlap = train_episode_keys & valid_episode_keys
    if episode_overlap:
        raise ValueError("Output train/valid dated episodes overlap")

    valid_manifest = manifest["valid"]
    if (
        valid_manifest.get("total_rows") != len(valid_decision_keys)
        or valid_manifest.get("total_context34_rows")
        != sum(valid_context_counts.values())
        or positive_count_map(
            valid_manifest.get("total_team_counts"),
            "valid.total_team_counts",
        )
        != dict(sorted(valid_team_counts.items()))
        or positive_count_map(
            valid_manifest.get("total_context34_team_counts"),
            "valid.total_context34_team_counts",
        )
        != dict(sorted(valid_context_counts.items()))
        or valid_manifest.get("episodes") != len(valid_episode_ids)
    ):
        raise ValueError("Manifest aggregate valid audit drifted")
    split_integrity = manifest.get("split_integrity")
    expected_split_integrity = {
        "episode_key_format": EPISODE_KEY_FORMAT,
        "train_episode_count": len(train_episode_keys),
        "valid_episode_count": len(valid_episode_keys),
        "episode_overlap_count": 0,
        "global_valid_episode_digest": digest_lines(valid_episode_keys),
    }
    if split_integrity != expected_split_integrity:
        raise ValueError("Manifest split_integrity does not match raw rows")
    episode_isolation = manifest["episode_isolation"]
    if (
        episode_isolation.get("selected_train_episode_count")
        != len(train_episode_ids)
        or episode_isolation.get("train_valid_overlap_count")
        != len(train_episode_ids & valid_episode_ids)
    ):
        raise ValueError("Manifest episode_isolation does not match raw rows")
    if train_episode_ids & valid_episode_ids:
        raise ValueError("Output train/valid episode IDs overlap")
    raw_integrity = {
        "train": {
            "rows": len(train_decision_keys),
            "decision_keys_digest": train_digest,
            "decision_key_duplicate_count": train_duplicate_count,
            "episode_ids": len(train_episode_ids),
            "dated_episodes": len(train_episode_keys),
        },
        "valid": {
            "rows": len(valid_decision_keys),
            "context34_rows": sum(valid_context_counts.values()),
            "team_counts": dict(sorted(valid_team_counts.items())),
            "context34_team_counts": dict(sorted(valid_context_counts.items())),
            "decision_keys_digest": valid_digest,
            "decision_key_duplicate_count": valid_duplicate_count,
            "episode_ids": len(valid_episode_ids),
            "dated_episodes": len(valid_episode_keys),
            "members": valid_member_audit,
        },
        "global": {
            "decision_keys_digest": global_digest,
            "decision_key_duplicate_count": global_duplicate_count,
            "train_valid_episode_id_overlap_count": len(
                train_episode_ids & valid_episode_ids
            ),
            "train_valid_dated_episode_overlap_count": len(episode_overlap),
        },
        "decision_keys_recomputed_from_raw_rows": True,
        "split_integrity_recomputed_from_raw_rows": True,
    }
    return batches, metadata, raw_integrity


def train_bc_replay_step(
    model: torch.nn.Module,
    replay_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: ppo.PPOConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Run train_ppo's audited BC core on exactly one supplied batch."""
    metrics = ppo.bc_replay_update(
        model,
        replay_optimizer,
        [batch],
        config,
        device,
        config.learning_rate,
    )
    if not isinstance(metrics, dict):
        raise RuntimeError("train_ppo BC replay core returned no metrics")
    return metrics


def main() -> None:
    args = parse_args()
    validate_request(args)

    tool_path = Path(__file__).resolve()
    train_path = tool_path.with_name("train_ppo.py")
    parent_path = args.parent_checkpoint.resolve()
    archive_path = args.special_data.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing output directory: {output_dir}"
        )

    tool_hash = validate_sha(tool_path, args.expected_tool_sha256, "tool")
    train_hash = validate_sha(
        train_path,
        args.expected_train_ppo_sha256,
        "train_ppo",
    )
    parent_hash_before = validate_sha(
        parent_path,
        args.expected_parent_sha256,
        "parent checkpoint",
    )
    archive_hash_before = validate_sha(
        archive_path,
        args.expected_special_data_sha256,
        "special BC archive",
    )
    manifest, member_hashes = validate_archive_manifest(archive_path, args)
    source_audit = audit_source_archives(manifest)

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
        raise ValueError("Parent checkpoint is not a compatible PPO checkpoint")
    if int(parent.get("update", -1)) != args.expected_parent_update:
        raise ValueError("PokemonFan32 parent update label is not 456")
    lineage_audit = validate_pokemonfan32_provenance(parent)
    config = validate_parent_config(parent.get("config"))

    configured_bc_path = Path(config.bc_checkpoint).resolve()
    if args.general_bc_checkpoint is not None:
        requested_bc_path = args.general_bc_checkpoint.resolve()
        if requested_bc_path != configured_bc_path:
            raise ValueError(
                "--general-bc-checkpoint does not match the parent's BC anchor"
            )
    general_bc_path = configured_bc_path
    general_bc_hash_before = validate_sha(
        general_bc_path,
        args.expected_general_bc_sha256,
        "general BC anchor",
    )
    general_bc = torch.load(
        general_bc_path,
        map_location="cpu",
        weights_only=False,
    )
    model = ppo.instantiate_model_from_checkpoint(parent, general_bc, device)
    if ppo.model_state_sha256(model) != EXPECTED_PARENT_MODEL_SHA256:
        raise ValueError("PokemonFan32 model state does not match its provenance")
    actor_parameters, _, trainable_manifest = ppo.configure_trainable_scope(
        model,
        config.trainable_scope,
    )
    actor_names = list(trainable_manifest["actor_parameter_names"])
    value_names = list(trainable_manifest["value_parameter_names"])
    parent_parameter_names = parent.get("optimizer_parameter_names")
    if not isinstance(parent_parameter_names, dict):
        raise ValueError("Parent checkpoint has no optimizer parameter manifest")
    if actor_names != parent_parameter_names.get("actor"):
        raise ValueError("Actor parameter names do not match the parent optimizer")
    if value_names != parent_parameter_names.get("value"):
        raise ValueError("Value parameter names do not match the parent optimizer")
    if set(actor_names) & set(value_names):
        raise ValueError("Parent actor and value parameter manifests overlap")

    replay_state = parent.get("bc_replay_optimizer_state_dict")
    if not isinstance(replay_state, dict):
        raise ValueError("Parent checkpoint has no replay optimizer state")
    replay_optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=SPECIAL_LEARNING_RATE,
        eps=1e-5,
        weight_decay=config.weight_decay,
    )
    replay_optimizer.load_state_dict(replay_state)
    replay_steps_before = repair_audit.optimizer_steps(
        replay_optimizer.state_dict()
    )
    if len(replay_steps_before) != len(actor_names):
        raise ValueError("Replay optimizer does not cover every actor tensor")
    if set(replay_steps_before) != {EXPECTED_REPLAY_STEP_BEFORE}:
        raise ValueError("Replay AdamW must start at step 48")

    ppo_optimizer_state_before = copy.deepcopy(parent.get("optimizer_state_dict"))
    if not isinstance(ppo_optimizer_state_before, dict):
        raise ValueError("Parent checkpoint has no PPO optimizer state")
    ppo_optimizer_hash_before = repair_audit.nested_sha256(
        ppo_optimizer_state_before
    )
    replay_optimizer_hash_before = repair_audit.nested_sha256(
        replay_optimizer.state_dict()
    )
    model_state_before = repair_audit.clone_model_state(model)
    model_hash_before = ppo.model_state_sha256(model)

    replay_batches, batch_metadata, raw_archive_integrity = (
        load_batch_aligned_archive(
        archive_path,
        manifest,
        parent["model_config"],
        source_audit,
        )
    )
    if len(replay_batches) != SPECIAL_STEPS:
        raise ValueError("Batch-aligned loader did not return exactly 20 batches")
    replay_cache_hash, replay_batch_hashes = repair_audit.replay_cache_manifest(
        replay_batches
    )
    if (
        args.expected_replay_cache_sha256 is not None
        and replay_cache_hash != args.expected_replay_cache_sha256
    ):
        raise ValueError(
            "Replay cache SHA256 mismatch: expected "
            f"{args.expected_replay_cache_sha256}, got {replay_cache_hash}"
        )
    for item, batch_hash in zip(batch_metadata, replay_batch_hashes):
        item["raw_member_sha256"] = member_hashes[item["member"]]
        item["collated_batch_sha256"] = batch_hash

    special_config = copy.deepcopy(config)
    special_config.bc_replay_data = str(archive_path)
    special_config.bc_replay_split = "train"
    special_config.bc_replay_batches = SPECIAL_STEPS
    special_config.bc_replay_batch_size = SPECIAL_BATCH_SIZE
    special_config.bc_replay_workers = 0
    special_config.bc_replay_steps = 1
    special_config.bc_replay_lr_scale = 0.05
    special_config.bc_replay_loss = "ordered"
    special_config.bc_replay_order_context_weight = 8.0
    special_config.bc_replay_context34_rows_per_batch = 1
    special_config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    special_config.seed = args.seed

    common: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "mode": "audit_only" if args.audit_only else "special_bc",
        "parent": {
            "path": str(parent_path),
            "sha256": parent_hash_before,
            "update": int(parent["update"]),
            "model_state_sha256": model_hash_before,
        },
        "sources": {
            "tool": {"path": str(tool_path), "sha256": tool_hash},
            "train_ppo": {"path": str(train_path), "sha256": train_hash},
            "general_bc_checkpoint": {
                "path": str(general_bc_path),
                "sha256": general_bc_hash_before,
            },
            "special_bc_archive": {
                "path": str(archive_path),
                "sha256": archive_hash_before,
                "manifest_sha256": args.expected_manifest_sha256,
                "content_sha256": args.expected_content_sha256,
                "sources_sha256": args.expected_sources_sha256,
                "decision_keys_digest": args.expected_decision_keys_digest,
            },
        },
        "special_bc": {
            "seed": args.seed,
            "archive_selection_seed": EXPECTED_ARCHIVE_SEED,
            "training_seed": args.seed,
            "steps_requested": args.steps,
            "batch_selection": "archive_member_order_no_shuffle_no_stratify",
            "rows_per_batch": SPECIAL_BATCH_SIZE,
            "context34_rows_per_batch": SPECIAL_CONTEXT34_ROWS_PER_BATCH,
            "policy_ids": list(CORE_POLICY_IDS),
            "rows_per_policy": SPECIAL_ROWS_PER_POLICY,
            "context34_rows_per_policy": SPECIAL_CONTEXT34_PER_POLICY,
            "sample_weight": 1.0,
            "loss": special_config.bc_replay_loss,
            "order_context_weight": (
                special_config.bc_replay_order_context_weight
            ),
            "learning_rate": SPECIAL_LEARNING_RATE,
            "max_grad_norm": special_config.max_grad_norm,
            "trainable_scope": special_config.trainable_scope,
            "batches": batch_metadata,
        },
        "replay_cache": {
            "sha256": replay_cache_hash,
            "batches": len(replay_batches),
            "rows": len(replay_batches) * SPECIAL_BATCH_SIZE,
            "context34_rows": SPECIAL_STEPS,
            "loaded_directly_from_batch_aligned_members": True,
            "replay_cache_builder_called": False,
        },
        "raw_archive_integrity": raw_archive_integrity,
        "source_archive_audit": source_audit["report"],
        "pokemonfan32_parent_lineage": {
            key: value
            for key, value in lineage_audit.items()
            if key != "frozen_files"
        },
        "optimizer": {
            "actor_parameter_names": actor_names,
            "value_parameter_names": value_names,
            "replay_state_tensor_count": len(replay_steps_before),
            "replay_steps_before": replay_steps_before,
            "replay_state_sha256_before": replay_optimizer_hash_before,
            "ppo_state_sha256_before": ppo_optimizer_hash_before,
            "replay_state_loaded_from_parent": True,
        },
        "determinism": {
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
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
                == replay_optimizer_hash_before
            ),
            "ppo_optimizer_state_unchanged": (
                repair_audit.nested_sha256(parent.get("optimizer_state_dict"))
                == ppo_optimizer_hash_before
            ),
            "parent_checkpoint_unchanged": (
                repair_audit.file_sha256(parent_path) == parent_hash_before
            ),
            "general_bc_checkpoint_unchanged": (
                repair_audit.file_sha256(general_bc_path)
                == general_bc_hash_before
            ),
            "special_bc_archive_unchanged": (
                repair_audit.file_sha256(archive_path) == archive_hash_before
            ),
            "tool_unchanged": repair_audit.file_sha256(tool_path) == tool_hash,
            "train_ppo_unchanged": (
                repair_audit.file_sha256(train_path) == train_hash
            ),
            "source_archives_unchanged": all(
                repair_audit.file_sha256(Path(path)) == expected_sha
                for path, expected_sha in source_audit["source_files"].items()
            ),
            "pokemonfan32_lineage_files_unchanged": all(
                repair_audit.file_sha256(Path(path)) == expected_sha
                for path, expected_sha in lineage_audit["frozen_files"].items()
            ),
        }
        invariant_keys = (
            "model_state_unchanged",
            "replay_optimizer_state_unchanged",
            "ppo_optimizer_state_unchanged",
            "parent_checkpoint_unchanged",
            "general_bc_checkpoint_unchanged",
            "special_bc_archive_unchanged",
            "tool_unchanged",
            "train_ppo_unchanged",
            "source_archives_unchanged",
            "pokemonfan32_lineage_files_unchanged",
        )
        if not all(result[key] for key in invariant_keys):
            raise RuntimeError("Audit-only mode mutated or lost a frozen input")
        write_json(output_dir / "preflight_audit.json", result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return

    per_step: list[dict[str, Any]] = []
    for special_step, (batch, batch_info) in enumerate(
        zip(replay_batches, batch_metadata),
        start=1,
    ):
        metrics = train_bc_replay_step(
            model,
            replay_optimizer,
            batch,
            special_config,
            device,
        )
        if metrics.get("steps") != 1:
            raise RuntimeError("Special BC core did not execute exactly one step")
        if int(metrics.get("rows", -1)) != SPECIAL_BATCH_SIZE:
            raise RuntimeError("Special BC core consumed an unexpected row count")
        if int(metrics.get("context_34_rows", -1)) != 1:
            raise RuntimeError("Special BC core consumed a wrong context-34 quota")
        if metrics.get("selected_batch_indices") != [0]:
            raise RuntimeError("Special BC core selected outside the supplied batch")
        if not repair_audit.finite_nested(metrics):
            raise FloatingPointError("Non-finite special-BC metric")
        per_step.append(
            {
                "special_step": special_step,
                "member": batch_info["member"],
                "raw_member_sha256": batch_info["raw_member_sha256"],
                "collated_batch_sha256": batch_info[
                    "collated_batch_sha256"
                ],
                "team_counts": batch_info["team_counts"],
                "context34_team_counts": batch_info[
                    "context34_team_counts"
                ],
                "metrics": metrics,
            }
        )

    model_state_after = repair_audit.clone_model_state(model)
    model_hash_after = ppo.model_state_sha256(model)
    if not repair_audit.finite_nested(model_state_after):
        raise FloatingPointError("Special-BC model contains non-finite tensors")
    changed_names = repair_audit.changed_tensor_names(
        model_state_before,
        model_state_after,
    )
    if not changed_names:
        raise RuntimeError("Special BC did not change any model tensor")
    unexpected_changes = sorted(set(changed_names) - set(actor_names))
    if unexpected_changes:
        raise RuntimeError(
            "Special BC changed tensors outside actor scope: "
            + json.dumps(unexpected_changes)
        )
    value_parameters_unchanged = all(
        torch.equal(model_state_before[name], model_state_after[name])
        for name in value_names
    )
    if not value_parameters_unchanged:
        raise RuntimeError("Special BC changed value-head tensors")

    replay_state_after = replay_optimizer.state_dict()
    replay_steps_after = repair_audit.optimizer_steps(replay_state_after)
    if set(replay_steps_after) != {EXPECTED_REPLAY_STEP_AFTER}:
        raise RuntimeError("Replay AdamW did not advance exactly from step 48 to 68")
    if repair_audit.nested_sha256(parent.get("optimizer_state_dict")) != (
        ppo_optimizer_hash_before
    ):
        raise RuntimeError("PPO optimizer state changed during special BC")
    parent_hash_after = repair_audit.file_sha256(parent_path)
    if parent_hash_after != parent_hash_before:
        raise RuntimeError("Parent checkpoint changed during special BC")
    if repair_audit.file_sha256(general_bc_path) != general_bc_hash_before:
        raise RuntimeError("General BC anchor changed during special BC")
    if repair_audit.file_sha256(archive_path) != archive_hash_before:
        raise RuntimeError("Special BC archive changed during training")
    if repair_audit.file_sha256(tool_path) != tool_hash:
        raise RuntimeError("Executor changed during training")
    if repair_audit.file_sha256(train_path) != train_hash:
        raise RuntimeError("train_ppo changed during training")
    if not all(
        repair_audit.file_sha256(Path(path)) == expected_sha
        for path, expected_sha in source_audit["source_files"].items()
    ):
        raise RuntimeError("A source archive changed during training")
    if not all(
        repair_audit.file_sha256(Path(path)) == expected_sha
        for path, expected_sha in lineage_audit["frozen_files"].items()
    ):
        raise RuntimeError("A PokemonFan32 lineage file changed during training")

    integrity = {
        "optimizer_steps": len(per_step),
        "rows": len(per_step) * SPECIAL_BATCH_SIZE,
        "context34_rows": len(per_step),
        "rows_per_policy": {
            policy_id: SPECIAL_ROWS_PER_POLICY for policy_id in CORE_POLICY_IDS
        },
        "context34_rows_per_policy": {
            policy_id: SPECIAL_CONTEXT34_PER_POLICY
            for policy_id in CORE_POLICY_IDS
        },
        "all_sample_weights_one": True,
        "all_metrics_and_model_tensors_finite": True,
        "model_state_sha256_before": model_hash_before,
        "model_state_sha256_after": model_hash_after,
        "model_state_changed": model_hash_after != model_hash_before,
        "changed_parameter_names": changed_names,
        "changed_parameters_subset_of_actor": True,
        "value_head_parameters_unchanged": value_parameters_unchanged,
        "replay_steps_after": replay_steps_after,
        "replay_state_sha256_after": repair_audit.nested_sha256(
            replay_state_after
        ),
        "ppo_state_sha256_after": ppo_optimizer_hash_before,
        "ppo_optimizer_state_unchanged": True,
        "parent_checkpoint_sha256_after": parent_hash_after,
        "parent_checkpoint_unchanged": True,
        "general_bc_checkpoint_unchanged": True,
        "special_bc_archive_unchanged": True,
        "tool_unchanged": True,
        "train_ppo_unchanged": True,
        "source_archives_unchanged": True,
        "pokemonfan32_lineage_files_unchanged": True,
    }
    provenance = common | {
        "status": "special_bc_completed",
        "per_step": per_step,
        "integrity": integrity,
        "checkpoint_update_label": int(parent["update"]),
        "not_a_new_ppo_update": True,
    }
    output_checkpoint = output_dir / "special-bc-pokemonfan32-balanced-core5-0020.pt"
    payload = copy.deepcopy(parent)
    payload["model_state_dict"] = model_state_after
    payload["bc_replay_optimizer_state_dict"] = replay_state_after
    payload["post_ppo_special_bc"] = provenance
    torch.save(payload, output_checkpoint)
    output_checkpoint_hash = repair_audit.file_sha256(output_checkpoint)

    result = provenance | {
        "checkpoint": {
            "path": str(output_checkpoint),
            "sha256": output_checkpoint_hash,
            "update": int(parent["update"]),
        },
        "checkpoint_writes": 1,
    }
    write_json(output_dir / "special_bc_manifest.json", result)
    if sorted(path.name for path in output_dir.iterdir()) != [
        "special-bc-pokemonfan32-balanced-core5-0020.pt",
        "special_bc_manifest.json",
    ]:
        raise RuntimeError("Special BC output directory contains unexpected files")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

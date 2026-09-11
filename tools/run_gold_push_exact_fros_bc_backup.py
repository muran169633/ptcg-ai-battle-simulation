#!/usr/bin/env python3
"""RETIRED, NEVER EXECUTED: obsolete update-0 exact-Fros BC draft.

This launcher is intentionally narrow.  It starts from the frozen update-0
PPO bridge, performs exactly eight ordered-BC optimizer steps with source
order general/Fros/KD/Fros/general/Fros/KD/Fros, and publishes only the E50
and E100 contractions of the resulting actor6 direction.

The exact-Fros archive is a future input.  Dry-run/freeze/execute all require
its reviewed outer-file digest, embedded manifest digest, and an external
canonical allowlist whose bytes must also be present inside the archive.
Dry-run writes nothing.  Execution cannot package, upload, submit, or evaluate
gameplay, and it refuses to reuse an output directory.

This draft was superseded before its first dry-run by the S8 anti-KD repair
route.  ``main`` is permanently fail-closed so this old lineage cannot be
mistakenly frozen or executed.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import random
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import torch  # noqa: E402
import interpolate_ppo_checkpoints as contraction_core  # noqa: E402
import run_gold_push_postppo_tail_repair as training_core  # noqa: E402


ppo = training_core.ppo

SCHEMA_VERSION = "ptcg-gold-push-exact-fros-bc-backup-v1"
RETIRED_NEVER_EXECUTED = True
RETIREMENT_REASON = "superseded_before_first_dry_run_by_s8_anti_kd_repair"
ALLOWLIST_SCHEMA_VERSION = (
    "ptcg-marnie-vs-exact-fros-trainwin-allowlist-v1"
)
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
EXPECTED_ENV_PREFIX = Path("/home/xxc/miniconda3/envs/my_project_env")

PARENT = ROOT / "artifacts/gold_push_20260810_v1/ppo_marnie_tail32_v1/best.pt"
PARENT_SHA256 = (
    "a205210bbe201f88eb4942d29c0c840799047df38b9f6abb24c1959e0b683300"
)
BC_ARCHITECTURE = (
    ROOT / "artifacts/gold_push_20260810_v1/bc/"
    "marnie_trainwins_seed1011/best.pt"
)
BC_ARCHITECTURE_SHA256 = (
    "dda68d51d9b922526709149143aa8409ba287fb8f31ddbb2293f0ea543ef01a0"
)

ARCHIVE_ROOT = ROOT / "data/gold_push_recent7_20260810_v1/archives"
GENERAL_ARCHIVE = ARCHIVE_ROOT / "marnie_trainwins.zip"
GENERAL_ARCHIVE_SHA256 = (
    "7cc2a4cb38b857ccdacf9cc84fe4610400ca3295c9ab1dbefbcb75e4341e8eab"
)
GENERAL_MANIFEST_SHA256 = (
    "217d62d390b5efad4368227e693784abbc89a2e4ac5ac54a814d56ba7c3959e2"
)
KD_ARCHIVE = ARCHIVE_ROOT / "marnie_kdcyberdude_trainwins.zip"
KD_ARCHIVE_SHA256 = (
    "24fd53320193ec299282b5dd8e609486d565c35fa1ee7b7febe698f872c360c5"
)
KD_MANIFEST_SHA256 = (
    "0abdc8feafff54a2fa8f71d3640e4f5fef593166d7907b1c55f07387f6d8ad0b"
)

MARNIE_DECK_HASH = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
FROS_DECK_HASH = (
    "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc"
)
FROS_ALLOWLIST_MEMBER = "exact_episode_allowlist.json"
EXPECTED_EPISODES = 104
EXPECTED_DECISION_ROWS = 10_614
EXPECTED_DATE_EPISODE_COUNTS = {
    "2026-08-02": 26,
    "2026-08-03": 11,
    "2026-08-04": 23,
    "2026-08-05": 29,
    "2026-08-06": 15,
}

OUTPUT_DIR = (
    ROOT / "artifacts/gold_push_20260810_v1/exact_fros_bc_backup_v1"
)

SEED = 2026081051
TRAIN_BATCHES_PER_SOURCE = 32
BATCH_SIZE = 256
CONTEXT34_ROWS_PER_BATCH = 1
ORDER_CONTEXT_WEIGHT = 8.0
RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT = 2.0
LEARNING_RATE = 4e-7
WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-5
MAX_GRAD_NORM = 1.0

GENERAL_BATCH_INDICES = (7, 19)
FROS_BATCH_INDICES = (11, 5, 23, 1)
KD_BATCH_INDICES = (3, 27)
STEP_SCHEDULE = (
    ("general", GENERAL_BATCH_INDICES[0]),
    ("fros", FROS_BATCH_INDICES[0]),
    ("kd", KD_BATCH_INDICES[0]),
    ("fros", FROS_BATCH_INDICES[1]),
    ("general", GENERAL_BATCH_INDICES[1]),
    ("fros", FROS_BATCH_INDICES[2]),
    ("kd", KD_BATCH_INDICES[1]),
    ("fros", FROS_BATCH_INDICES[3]),
)
CONTRACTIONS = (("E50", 0.50), ("E100", 1.00))
ACTOR6 = training_core.ACTOR6


@dataclass(frozen=True)
class ArchiveRecord:
    path: Path
    archive_sha256: str
    manifest_sha256: str
    manifest: dict[str, Any]
    member_names: tuple[str, ...]
    embedded_allowlist: bytes | None = None


@dataclass
class PreparedRun:
    manifest: dict[str, Any]
    manifest_sha256: str
    parent: dict[str, Any]
    bc_checkpoint: dict[str, Any]
    configs: dict[str, argparse.Namespace]
    caches: dict[str, list[dict[str, torch.Tensor]]]
    batch_sha256: dict[str, list[str]]
    fros_archive: Path
    allowlist_path: Path


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_sha256(path: Path) -> str:
    return training_core.file_sha256(path)


def validate_sha256(value: str, label: str) -> str:
    training_core.validate_sha256_text(value, label)
    return value


def require_regular_file(path: Path, expected_sha256: str, label: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    training_core.require_regular_file(path, expected_sha256, label)
    return path


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def validate_exact_allowlist(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("Exact-Fros allowlist root must be an object")
    expected_scalars = {
        "schema_version": ALLOWLIST_SCHEMA_VERSION,
        "learner_deck_hash": MARNIE_DECK_HASH,
        "opponent_deck_hash": FROS_DECK_HASH,
        "split": "train",
        "terminal_reward": "win",
        "episode_count": EXPECTED_EPISODES,
        "decision_rows": EXPECTED_DECISION_ROWS,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Exact-Fros allowlist {key} mismatch: "
                f"expected {expected!r}, got {payload.get(key)!r}"
            )
    if payload.get("date_episode_counts") != EXPECTED_DATE_EPISODE_COUNTS:
        raise RuntimeError("Exact-Fros allowlist date_episode_counts mismatch")

    episode_ids = payload.get("episode_ids")
    episodes = payload.get("episodes")
    if not isinstance(episode_ids, list) or not isinstance(episodes, list):
        raise TypeError("Exact-Fros allowlist requires episode_ids and episodes lists")
    if len(episode_ids) != EXPECTED_EPISODES or len(episodes) != EXPECTED_EPISODES:
        raise RuntimeError("Exact-Fros allowlist must contain exactly 104 episodes")
    if any(not isinstance(value, str) or not value for value in episode_ids):
        raise RuntimeError("Exact-Fros episode_ids must be nonempty strings")
    if len(set(episode_ids)) != EXPECTED_EPISODES:
        raise RuntimeError("Exact-Fros episode_ids must be unique")

    normalized: list[tuple[str, str, int]] = []
    for index, record in enumerate(episodes):
        if not isinstance(record, dict):
            raise TypeError(f"Exact-Fros episodes[{index}] must be an object")
        episode_id = record.get("episode_id")
        date = record.get("date")
        rows = _int(record.get("decision_rows"), f"episodes[{index}].decision_rows")
        if not isinstance(episode_id, str) or not episode_id:
            raise RuntimeError(f"Exact-Fros episodes[{index}] has invalid episode_id")
        if date not in EXPECTED_DATE_EPISODE_COUNTS:
            raise RuntimeError(f"Exact-Fros episodes[{index}] has invalid date")
        if rows <= 0:
            raise RuntimeError(f"Exact-Fros episodes[{index}] has no decision rows")
        normalized.append((date, episode_id, rows))

    record_ids = [episode_id for _, episode_id, _ in normalized]
    if record_ids != episode_ids:
        raise RuntimeError("episode_ids must exactly match ordered episodes records")
    if len(set(record_ids)) != EXPECTED_EPISODES:
        raise RuntimeError("Exact-Fros episodes records contain duplicate IDs")
    if normalized != sorted(normalized, key=lambda row: (row[0], int(row[1]))):
        raise RuntimeError("Exact-Fros episodes must be sorted by date and numeric ID")
    if sum(rows for _, _, rows in normalized) != EXPECTED_DECISION_ROWS:
        raise RuntimeError("Exact-Fros per-episode decision row total mismatch")
    observed_dates = {
        date: sum(1 for row_date, _, _ in normalized if row_date == date)
        for date in EXPECTED_DATE_EPISODE_COUNTS
    }
    if observed_dates != EXPECTED_DATE_EPISODE_COUNTS:
        raise RuntimeError("Exact-Fros per-episode date counts mismatch")
    return payload


def load_allowlist(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes, str]:
    path = require_regular_file(path, expected_sha256, "exact-Fros allowlist")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Exact-Fros allowlist is not valid UTF-8 JSON") from error
    payload = validate_exact_allowlist(payload)
    canonical = canonical_json_bytes(payload)
    if raw != canonical:
        raise RuntimeError("Exact-Fros allowlist must use canonical JSON bytes")
    canonical_sha256 = sha256_bytes(canonical)
    if canonical_sha256 != expected_sha256:
        raise RuntimeError("Exact-Fros raw and canonical allowlist hashes differ")
    return payload, raw, canonical_sha256


def _validate_zip_member(info: zipfile.ZipInfo, label: str) -> None:
    member = PurePosixPath(info.filename)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise RuntimeError(f"Unsafe {label} ZIP member: {info.filename!r}")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise RuntimeError(f"Symlink {label} ZIP member is forbidden: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise RuntimeError(f"Encrypted {label} ZIP member is forbidden: {info.filename!r}")


def load_archive(
    path: Path,
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    label: str,
    embedded_allowlist_raw: bytes | None = None,
) -> ArchiveRecord:
    path = require_regular_file(path, expected_archive_sha256, label)
    validate_sha256(expected_manifest_sha256, f"{label} manifest SHA-256")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise RuntimeError(f"{label} ZIP contains duplicate member names")
            for info in infos:
                _validate_zip_member(info, label)
            if names.count("manifest.json") != 1:
                raise RuntimeError(f"{label} ZIP must contain exactly one manifest.json")
            manifest_raw = archive.read("manifest.json")
            observed_manifest_sha256 = sha256_bytes(manifest_raw)
            if observed_manifest_sha256 != expected_manifest_sha256:
                raise RuntimeError(
                    f"{label} embedded manifest SHA-256 mismatch: expected "
                    f"{expected_manifest_sha256}, got {observed_manifest_sha256}"
                )
            try:
                manifest = json.loads(manifest_raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"{label} manifest is not valid UTF-8 JSON") from error
            embedded = None
            if embedded_allowlist_raw is not None:
                if names.count(FROS_ALLOWLIST_MEMBER) != 1:
                    raise RuntimeError(
                        f"{label} ZIP must contain exactly one {FROS_ALLOWLIST_MEMBER}"
                    )
                embedded = archive.read(FROS_ALLOWLIST_MEMBER)
                if embedded != embedded_allowlist_raw:
                    raise RuntimeError(
                        "Exact-Fros external and embedded allowlist bytes differ"
                    )
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"{label} is not a valid ZIP archive") from error
    if not isinstance(manifest, dict):
        raise TypeError(f"{label} manifest root must be an object")
    return ArchiveRecord(
        path=path,
        archive_sha256=expected_archive_sha256,
        manifest_sha256=expected_manifest_sha256,
        manifest=manifest,
        member_names=tuple(names),
        embedded_allowlist=embedded,
    )


def validate_standard_archive(record: ArchiveRecord, source: str) -> None:
    manifest = record.manifest
    if manifest.get("schema_version") != "ptcg-bc-visible-decisions-v1":
        raise RuntimeError(f"{source} archive schema mismatch")
    if manifest.get("deck_hash_filter") != MARNIE_DECK_HASH:
        raise RuntimeError(f"{source} archive learner deck mismatch")
    reward_filter = manifest.get("terminal_reward_filter")
    if not isinstance(reward_filter, dict) or reward_filter.get("mode") != "wins":
        raise RuntimeError(f"{source} archive is not train-win filtered")
    if reward_filter.get("splits") != "train":
        raise RuntimeError(f"{source} archive reward-filter split mismatch")
    split_rows = manifest.get("split_decisions")
    if not isinstance(split_rows, dict) or _int(
        split_rows.get("train"), f"{source}.split_decisions.train"
    ) < TRAIN_BATCHES_PER_SOURCE * BATCH_SIZE:
        raise RuntimeError(f"{source} archive has insufficient frozen train rows")
    if source == "kd" and manifest.get("team_name_filter") != "@kdcyberdude":
        raise RuntimeError("KD archive team filter mismatch")


def validate_fros_archive(
    record: ArchiveRecord,
    *,
    allowlist_sha256: str,
    allowlist_canonical_sha256: str,
) -> None:
    manifest = record.manifest
    expected = {
        "schema_version": "ptcg-bc-visible-decisions-v1",
        "learner_deck_hash": MARNIE_DECK_HASH,
        "opponent_deck_hash": FROS_DECK_HASH,
        "split": "train",
        "terminal_reward": "win",
        "episode_count": EXPECTED_EPISODES,
        "decision_rows": EXPECTED_DECISION_ROWS,
        "date_episode_counts": EXPECTED_DATE_EPISODE_COUNTS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Exact-Fros archive manifest {key} mismatch: "
                f"expected {value!r}, got {manifest.get(key)!r}"
            )
    split_episodes = manifest.get("split_episodes")
    split_decisions = manifest.get("split_decisions")
    if not isinstance(split_episodes, dict) or split_episodes.get("train") != EXPECTED_EPISODES:
        raise RuntimeError("Exact-Fros archive train episode count mismatch")
    if not isinstance(split_decisions, dict) or split_decisions.get("train") != EXPECTED_DECISION_ROWS:
        raise RuntimeError("Exact-Fros archive train decision row count mismatch")
    reference = manifest.get("exact_episode_allowlist")
    if not isinstance(reference, dict):
        raise RuntimeError("Exact-Fros archive lacks exact_episode_allowlist binding")
    expected_reference = {
        "member": FROS_ALLOWLIST_MEMBER,
        "schema_version": ALLOWLIST_SCHEMA_VERSION,
        "sha256": allowlist_sha256,
        "canonical_sha256": allowlist_canonical_sha256,
        "episodes": EXPECTED_EPISODES,
        "decision_rows": EXPECTED_DECISION_ROWS,
    }
    for key, value in expected_reference.items():
        if reference.get(key) != value:
            raise RuntimeError(f"Exact-Fros archive allowlist {key} mismatch")


def validate_protocol_constants() -> None:
    if tuple(source for source, _ in STEP_SCHEDULE) != (
        "general",
        "fros",
        "kd",
        "fros",
        "general",
        "fros",
        "kd",
        "fros",
    ):
        raise RuntimeError("Exact-Fros eight-step source order drifted")
    if len(STEP_SCHEDULE) != 8 or len(set(STEP_SCHEDULE)) != 8:
        raise RuntimeError("Exact-Fros schedule must contain eight unique batches")
    if CONTRACTIONS != (("E50", 0.50), ("E100", 1.00)):
        raise RuntimeError("Only E50/E100 contractions are allowed")
    expected_core = {
        "BATCH_SIZE": BATCH_SIZE,
        "ORDER_CONTEXT_WEIGHT": ORDER_CONTEXT_WEIGHT,
        "RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT": RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
    }
    for name, expected in expected_core.items():
        if getattr(training_core, name) != expected:
            raise RuntimeError(f"Training core {name} is incompatible")
    if training_core.ACTOR6 != ACTOR6:
        raise RuntimeError("Training core actor6 scope drifted")
    if contraction_core.PPO_FEATURE_VERSION != ppo.PPO_FEATURE_VERSION:
        raise RuntimeError("Contraction and PPO feature versions differ")
    if PARENT_SHA256 == BC_ARCHITECTURE_SHA256:
        raise RuntimeError("Parent and architecture checkpoint identities collided")


def replay_config(
    parent_config: dict[str, Any], archive: Path, source_offset: int
) -> argparse.Namespace:
    return training_core.replay_config(
        parent_config,
        archive=archive,
        split="train",
        batches=TRAIN_BATCHES_PER_SOURCE,
        context34_rows_per_batch=CONTEXT34_ROWS_PER_BATCH,
        seed=SEED + source_offset,
    )


def _scheduled_indices(source: str) -> list[int]:
    return [index for item_source, index in STEP_SCHEDULE if item_source == source]


def _cache_record(
    source: str,
    batches: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, Any], list[str]]:
    profile = training_core.cache_profile(
        batches,
        expected_batches=TRAIN_BATCHES_PER_SOURCE,
        expected_context34_rows_per_batch=CONTEXT34_ROWS_PER_BATCH,
    )
    cache_sha256, per_batch = training_core.audit.replay_cache_manifest(batches)
    indices = _scheduled_indices(source)
    fixed_multi_rows = []
    for index in indices:
        batch = batches[index]
        rows = int(
            (
                (batch["contexts"] != ppo.SKILL_ORDER_CONTEXT)
                & (batch["min_counts"] == batch["max_counts"])
                & (batch["action_counts"] > 1)
            ).sum()
        )
        if rows <= 0:
            raise RuntimeError(
                f"Scheduled {source} batch {index} lacks fixed multi-action rows"
            )
        fixed_multi_rows.append(rows)
    record = {
        **profile,
        "sha256": cache_sha256,
        "batch_sha256": per_batch,
        "split": "train",
        "used_for_optimizer_steps": True,
        "scheduled_batch_indices": indices,
        "scheduled_batch_sha256": [per_batch[index] for index in indices],
        "scheduled_non_context34_fixed_multi_action_rows": fixed_multi_rows,
    }
    return record, per_batch


def _absolute(path: Path) -> str:
    return str(Path(os.path.abspath(os.fspath(path))))


def build_manifest_payload(
    *,
    parent: dict[str, Any],
    parent_model_sha256: str,
    archives: dict[str, ArchiveRecord],
    fros_expected_archive_sha256: str,
    fros_expected_manifest_sha256: str,
    allowlist_path: Path,
    allowlist_sha256: str,
    allowlist_canonical_sha256: str,
    allowlist_payload: dict[str, Any],
    cache_records: dict[str, dict[str, Any]],
    actor_shapes: dict[str, list[int]],
    dependency_hashes: dict[str, str],
    device: str,
) -> dict[str, Any]:
    archive_inputs = {}
    for source, record in archives.items():
        archive_inputs[source] = {
            "path": _absolute(record.path),
            "sha256": record.archive_sha256,
            "embedded_manifest_sha256": record.manifest_sha256,
        }
    # Keep the reviewed CLI values explicit in addition to the normalized
    # ArchiveRecord so accidental argument rewiring changes the manifest.
    archive_inputs["fros"]["reviewed_archive_sha256"] = fros_expected_archive_sha256
    archive_inputs["fros"]["reviewed_manifest_sha256"] = fros_expected_manifest_sha256
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "frozen_before_training",
        "inputs": {
            "parent_checkpoint": {
                "path": _absolute(PARENT),
                "sha256": PARENT_SHA256,
                "update": int(parent.get("update", -1)),
                "feature_version": parent.get("feature_version"),
                "model_state_sha256": parent_model_sha256,
            },
            "bc_architecture_checkpoint": {
                "path": _absolute(BC_ARCHITECTURE),
                "sha256": BC_ARCHITECTURE_SHA256,
            },
            "archives": archive_inputs,
            "exact_fros_allowlist": {
                "path": _absolute(allowlist_path),
                "sha256": allowlist_sha256,
                "canonical_sha256": allowlist_canonical_sha256,
                "embedded_member": FROS_ALLOWLIST_MEMBER,
                "embedded_bytes_identical": True,
                "schema_version": allowlist_payload["schema_version"],
                "episode_count": allowlist_payload["episode_count"],
                "decision_rows": allowlist_payload["decision_rows"],
                "date_episode_counts": allowlist_payload["date_episode_counts"],
            },
            "dependencies": {
                name: {"path": path, "sha256": dependency_hashes[name]}
                for name, path in (
                    ("launcher", _absolute(Path(__file__).resolve())),
                    ("training_core", _absolute(Path(training_core.__file__).resolve())),
                    ("trainer_module", _absolute(training_core.TRAINER)),
                    ("contraction_core", _absolute(Path(contraction_core.__file__).resolve())),
                )
            },
            "python": {
                "invocation_path": str(EXPECTED_PYTHON),
                "resolved_path": str(EXPECTED_PYTHON.resolve()),
                "sha256": dependency_hashes["python"],
                "environment_prefix": str(EXPECTED_ENV_PREFIX),
            },
        },
        "lineage": {
            "learner_deck_hash": parent.get("learner_deck_hash"),
            "required_learner_deck_hash": MARNIE_DECK_HASH,
            "exact_fros_opponent_deck_hash": FROS_DECK_HASH,
            "parent_update_must_equal": 0,
            "parent_config_sha256": sha256_json(parent.get("config")),
            "parent_model_config_sha256": sha256_json(parent.get("model_config")),
        },
        "runtime": {
            "python_invocation": str(EXPECTED_PYTHON),
            "python_resolved": str(EXPECTED_PYTHON.resolve()),
            "environment_prefix": str(EXPECTED_ENV_PREFIX),
            "isolated": True,
            "dont_write_bytecode": True,
            "device": device,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "deterministic_algorithms_required": True,
            "cublas_workspace_config": ":4096:8",
            "tf32_allowed": False,
        },
        "protocol": {
            "seed": SEED,
            "optimizer": "fresh_adamw",
            "optimizer_state_from_parent": False,
            "steps": 8,
            "step_schedule": [
                {"step": step, "source": source, "batch_index": batch_index}
                for step, (source, batch_index) in enumerate(STEP_SCHEDULE, 1)
            ],
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_eps": ADAM_EPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "loss": "ordered",
            "context34_order_weight": ORDER_CONTEXT_WEIGHT,
            "non_context34_fixed_multi_action_order_weight": (
                RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT
            ),
            "batch_size": BATCH_SIZE,
            "cache_batches_per_source": TRAIN_BATCHES_PER_SOURCE,
            "context34_rows_per_batch": CONTEXT34_ROWS_PER_BATCH,
            "trainable_scope": "actor6",
            "trainable_parameter_names": list(ACTOR6),
            "trainable_parameter_shapes": actor_shapes,
            "contractions": [
                {"name": name, "alpha": alpha} for name, alpha in CONTRACTIONS
            ],
            "only_published_endpoints": [name for name, _ in CONTRACTIONS],
        },
        "replay_caches": cache_records,
        "guards": {
            "exact_fros_allowlist_external_equals_embedded": True,
            "exact_fros_episode_count": EXPECTED_EPISODES,
            "exact_fros_decision_rows": EXPECTED_DECISION_ROWS,
            "exact_fros_date_episode_counts": EXPECTED_DATE_EPISODE_COUNTS,
            "all_non_actor6_tensors_must_be_bit_identical": True,
            "each_actor6_tensor_must_move": True,
            "E50_actor6_l2_ratio_to_E100_min": 0.49,
            "E50_actor6_l2_ratio_to_E100_max": 0.51,
            "no_intermediate_training_checkpoint": True,
            "output_directory_must_be_absent": True,
            "inputs_must_be_unchanged_after_training": True,
        },
        "output": {
            "directory": _absolute(OUTPUT_DIR),
            "must_not_exist_before_execute": True,
            "checkpoints": ["exact-fros-bc-E50.pt", "exact-fros-bc-E100.pt"],
            "evaluation_only": True,
            "resume_forbidden": True,
        },
        "scope": {
            "training": True,
            "gameplay_evaluation": False,
            "package": False,
            "upload": False,
            "submission": False,
            "local_only": True,
        },
    }


def prepare_run(
    *,
    fros_archive: Path,
    expected_fros_archive_sha256: str,
    expected_fros_manifest_sha256: str,
    allowlist_path: Path,
    expected_allowlist_sha256: str,
    device: str,
) -> PreparedRun:
    validate_protocol_constants()
    training_core.assert_output_absent(OUTPUT_DIR)
    validate_sha256(expected_fros_archive_sha256, "Fros archive SHA-256")
    validate_sha256(expected_fros_manifest_sha256, "Fros manifest SHA-256")
    validate_sha256(expected_allowlist_sha256, "allowlist SHA-256")

    require_regular_file(PARENT, PARENT_SHA256, "update-0 parent checkpoint")
    require_regular_file(
        BC_ARCHITECTURE,
        BC_ARCHITECTURE_SHA256,
        "BC architecture checkpoint",
    )
    allowlist_path = Path(os.path.abspath(os.fspath(allowlist_path)))
    allowlist, allowlist_raw, allowlist_canonical_sha256 = load_allowlist(
        allowlist_path, expected_allowlist_sha256
    )
    archives = {
        "general": load_archive(
            GENERAL_ARCHIVE,
            expected_archive_sha256=GENERAL_ARCHIVE_SHA256,
            expected_manifest_sha256=GENERAL_MANIFEST_SHA256,
            label="general train-win archive",
        ),
        "kd": load_archive(
            KD_ARCHIVE,
            expected_archive_sha256=KD_ARCHIVE_SHA256,
            expected_manifest_sha256=KD_MANIFEST_SHA256,
            label="KD train-win archive",
        ),
        "fros": load_archive(
            fros_archive,
            expected_archive_sha256=expected_fros_archive_sha256,
            expected_manifest_sha256=expected_fros_manifest_sha256,
            label="exact-Fros train-win archive",
            embedded_allowlist_raw=allowlist_raw,
        ),
    }
    validate_standard_archive(archives["general"], "general")
    validate_standard_archive(archives["kd"], "kd")
    validate_fros_archive(
        archives["fros"],
        allowlist_sha256=expected_allowlist_sha256,
        allowlist_canonical_sha256=allowlist_canonical_sha256,
    )

    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(
        BC_ARCHITECTURE, map_location="cpu", weights_only=False
    )
    if not isinstance(parent, dict) or not isinstance(bc_checkpoint, dict):
        raise TypeError("Parent and BC architecture checkpoints must be dictionaries")
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise RuntimeError("Parent checkpoint PPO feature version mismatch")
    if parent.get("update") != 0:
        raise RuntimeError("Backup route must start from the update-0 bridge")
    if parent.get("learner_deck_hash") != MARNIE_DECK_HASH:
        raise RuntimeError("Parent checkpoint Marnie deck binding mismatch")
    if not isinstance(parent.get("config"), dict):
        raise RuntimeError("Parent checkpoint lacks PPO config")
    if not isinstance(parent.get("model_config"), dict):
        raise RuntimeError("Parent checkpoint lacks model_config")
    if not isinstance(parent.get("model_state_dict"), dict):
        raise RuntimeError("Parent checkpoint lacks model_state_dict")
    if bc_checkpoint.get("feature_version") != ppo.BC_FEATURE_VERSION:
        raise RuntimeError("BC architecture checkpoint feature version mismatch")
    bc_config = bc_checkpoint.get("config")
    if not isinstance(bc_config, dict):
        raise RuntimeError("BC architecture checkpoint lacks config")
    architecture_mismatches = {
        name: {"parent": expected, "bc": bc_config.get(name)}
        for name, expected in parent["model_config"].items()
        if bc_config.get(name) != expected
    }
    if architecture_mismatches:
        raise RuntimeError(
            f"Parent/BC architecture mismatch: {architecture_mismatches}"
        )

    verification_model = ppo.instantiate_model_from_checkpoint(
        parent, bc_checkpoint, torch.device("cpu")
    )
    training_core.configure_actor6(verification_model)
    actor_shapes = {
        name: list(dict(verification_model.named_parameters())[name].shape)
        for name in ACTOR6
    }
    del verification_model

    configs = {
        "general": replay_config(parent["config"], archives["general"].path, 101),
        "fros": replay_config(parent["config"], archives["fros"].path, 202),
        "kd": replay_config(parent["config"], archives["kd"].path, 303),
    }
    cache_log = io.StringIO()
    with contextlib.redirect_stdout(cache_log):
        caches = {
            source: ppo.build_bc_replay_batches(config, parent["model_config"])
            for source, config in configs.items()
        }
    cache_records: dict[str, dict[str, Any]] = {}
    batch_sha256: dict[str, list[str]] = {}
    for source in ("general", "fros", "kd"):
        cache_records[source], batch_sha256[source] = _cache_record(
            source, caches[source]
        )

    dependency_paths = {
        "launcher": Path(__file__).resolve(),
        "training_core": Path(training_core.__file__).resolve(),
        "trainer_module": training_core.TRAINER,
        "contraction_core": Path(contraction_core.__file__).resolve(),
        "python": EXPECTED_PYTHON.resolve(),
    }
    dependency_hashes = {
        name: file_sha256(path) for name, path in dependency_paths.items()
    }
    manifest = build_manifest_payload(
        parent=parent,
        parent_model_sha256=training_core.audit.nested_sha256(
            parent["model_state_dict"]
        ),
        archives=archives,
        fros_expected_archive_sha256=expected_fros_archive_sha256,
        fros_expected_manifest_sha256=expected_fros_manifest_sha256,
        allowlist_path=allowlist_path,
        allowlist_sha256=expected_allowlist_sha256,
        allowlist_canonical_sha256=allowlist_canonical_sha256,
        allowlist_payload=allowlist,
        cache_records=cache_records,
        actor_shapes=actor_shapes,
        dependency_hashes=dependency_hashes,
        device=device,
    )
    return PreparedRun(
        manifest=manifest,
        manifest_sha256=sha256_json(manifest),
        parent=parent,
        bc_checkpoint=bc_checkpoint,
        configs=configs,
        caches=caches,
        batch_sha256=batch_sha256,
        fros_archive=archives["fros"].path,
        allowlist_path=allowlist_path,
    )


def frozen_manifest_envelope(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION + "-frozen-manifest-envelope",
        "manifest_sha256": sha256_json(manifest),
        "manifest": manifest,
    }


def dry_run_bytes(manifest: dict[str, Any]) -> bytes:
    return canonical_json_bytes(frozen_manifest_envelope(manifest))


def load_frozen_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    training_core.require_unhashed_regular_file(path, "frozen training manifest")
    validate_sha256(expected_sha256, "expected manifest SHA-256")
    try:
        envelope = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Frozen training manifest is not valid UTF-8 JSON") from error
    if not isinstance(envelope, dict) or not isinstance(envelope.get("manifest"), dict):
        raise RuntimeError("Frozen training manifest envelope is malformed")
    expected_schema = SCHEMA_VERSION + "-frozen-manifest-envelope"
    if envelope.get("schema_version") != expected_schema:
        raise RuntimeError("Frozen training manifest envelope schema mismatch")
    manifest = envelope["manifest"]
    observed = sha256_json(manifest)
    if envelope.get("manifest_sha256") != observed:
        raise RuntimeError("Frozen training manifest embedded digest is invalid")
    if observed != expected_sha256:
        raise RuntimeError(
            f"Frozen training manifest SHA-256 mismatch: expected "
            f"{expected_sha256}, got {observed}"
        )
    return manifest


def actor6_movement(
    parent_state: Mapping[str, torch.Tensor],
    endpoint_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    per_tensor = {}
    total_square = 0.0
    for name in ACTOR6:
        delta = endpoint_state[name].double() - parent_state[name].double()
        l2 = math.sqrt(float(delta.square().sum()))
        max_abs = float(delta.abs().max())
        if not math.isfinite(l2) or not math.isfinite(max_abs) or l2 <= 0.0:
            raise RuntimeError(f"Actor6 tensor did not move finitely: {name}")
        per_tensor[name] = {"l2": l2, "max_abs": max_abs}
        total_square += l2 * l2
    total_l2 = math.sqrt(total_square)
    if not math.isfinite(total_l2) or total_l2 <= 0.0:
        raise RuntimeError("Actor6 direction has no finite movement")
    return {"l2": total_l2, "per_tensor": per_tensor}


def build_contractions(
    parent_state: dict[str, torch.Tensor],
    trained_state: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, Any]]]:
    trained_integrity = training_core.validate_endpoint_state(
        parent_state, trained_state
    )
    states = {}
    audits = {}
    for name, alpha in CONTRACTIONS:
        state = contraction_core.interpolate_state_dict(
            parent_state, trained_state, alpha
        )
        integrity = training_core.validate_endpoint_state(parent_state, state)
        movement = actor6_movement(parent_state, state)
        states[name] = state
        audits[name] = {
            "alpha": alpha,
            "integrity": integrity,
            "movement": movement,
        }
    for tensor_name in parent_state:
        if tensor_name not in ACTOR6:
            if not torch.equal(states["E50"][tensor_name], parent_state[tensor_name]):
                raise RuntimeError(f"E50 changed non-actor6 tensor {tensor_name}")
            if not torch.equal(states["E100"][tensor_name], parent_state[tensor_name]):
                raise RuntimeError(f"E100 changed non-actor6 tensor {tensor_name}")
        if not torch.equal(states["E100"][tensor_name], trained_state[tensor_name]):
            raise RuntimeError(f"E100 differs from trained state at {tensor_name}")
    ratio = audits["E50"]["movement"]["l2"] / audits["E100"]["movement"]["l2"]
    if not 0.49 <= ratio <= 0.51:
        raise RuntimeError(f"E50 actor6 movement ratio is not one half: {ratio}")
    audits["E50"]["actor6_l2_ratio_to_E100"] = ratio
    audits["trained_direction"] = trained_integrity
    return states, audits


def endpoint_payload(
    *,
    parent: dict[str, Any],
    state: dict[str, torch.Tensor],
    manifest_sha256: str,
    endpoint: str,
    alpha: float,
    audit: dict[str, Any],
) -> dict[str, Any]:
    retained = {}
    for key in (
        "feature_version",
        "bc_feature_version",
        "config",
        "model_config",
        "learner_deck_hash",
        "reward",
        "value_trunk_gradient",
        "actor_value_gradient",
        "action_distribution",
    ):
        if key in parent:
            retained[key] = copy.deepcopy(parent[key])
    retained.update(
        {
            "model_state_dict": state,
            "update": 0,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_states_omitted": [
                "optimizer_state_dict",
                "bc_replay_optimizer_state_dict",
                "opponent_quota_state",
            ],
            "exact_fros_ordered_bc": {
                "schema_version": SCHEMA_VERSION,
                "frozen_manifest_sha256": manifest_sha256,
                "optimizer_steps": 8,
                "endpoint": endpoint,
                "contraction_alpha": alpha,
                "trainable_scope": "actor6",
                "source_order": [source for source, _ in STEP_SCHEDULE],
                "audit": audit,
                "downstream_gameplay_evaluation_required": True,
                "promotion_eligible": False,
                "submission_authorized": False,
            },
        }
    )
    return retained


def _current_input_hashes(prepared: PreparedRun) -> dict[str, str]:
    return {
        "parent": file_sha256(PARENT),
        "bc_architecture": file_sha256(BC_ARCHITECTURE),
        "general_archive": file_sha256(GENERAL_ARCHIVE),
        "kd_archive": file_sha256(KD_ARCHIVE),
        "fros_archive": file_sha256(prepared.fros_archive),
        "allowlist": file_sha256(prepared.allowlist_path),
        "launcher": file_sha256(Path(__file__).resolve()),
        "training_core": file_sha256(Path(training_core.__file__).resolve()),
        "trainer_module": file_sha256(training_core.TRAINER),
        "contraction_core": file_sha256(Path(contraction_core.__file__).resolve()),
        "python": file_sha256(EXPECTED_PYTHON.resolve()),
    }


def _expected_input_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    inputs = manifest["inputs"]
    dependencies = inputs["dependencies"]
    return {
        "parent": inputs["parent_checkpoint"]["sha256"],
        "bc_architecture": inputs["bc_architecture_checkpoint"]["sha256"],
        "general_archive": inputs["archives"]["general"]["sha256"],
        "kd_archive": inputs["archives"]["kd"]["sha256"],
        "fros_archive": inputs["archives"]["fros"]["sha256"],
        "allowlist": inputs["exact_fros_allowlist"]["sha256"],
        "launcher": dependencies["launcher"]["sha256"],
        "training_core": dependencies["training_core"]["sha256"],
        "trainer_module": dependencies["trainer_module"]["sha256"],
        "contraction_core": dependencies["contraction_core"]["sha256"],
        "python": inputs["python"]["sha256"],
    }


def execute(prepared: PreparedRun, device_name: str) -> int:
    training_core.assert_output_absent(OUTPUT_DIR)
    device = torch.device(device_name)
    if device.type != "cuda" or device.index not in (None, 0):
        raise RuntimeError("Formal backup training requires cuda or cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = ppo.instantiate_model_from_checkpoint(
        prepared.parent, prepared.bc_checkpoint, device
    )
    parameters = training_core.configure_actor6(model)
    parent_state = training_core.clone_model_state(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=ADAM_EPS,
    )

    trace = []
    for step, (source, batch_index) in enumerate(STEP_SCHEDULE, 1):
        metrics = training_core.actor6_optimizer_step(
            model=model,
            parameters=parameters,
            optimizer=optimizer,
            cpu_batch=prepared.caches[source][batch_index],
            config=prepared.configs[source],
            device=device,
        )
        trace.append(
            {
                "step": step,
                "source": source,
                "batch_index": batch_index,
                "batch_sha256": prepared.batch_sha256[source][batch_index],
                "metrics": metrics,
            }
        )
    trained_state = training_core.clone_model_state(model)
    states, audits = build_contractions(parent_state, trained_state)

    expected_hashes = _expected_input_hashes(prepared.manifest)
    current_hashes = _current_input_hashes(prepared)
    if current_hashes != expected_hashes:
        raise RuntimeError(
            f"Frozen input drift detected after training: expected="
            f"{expected_hashes}, observed={current_hashes}"
        )

    checkpoint_bytes: dict[str, bytes] = {}
    checkpoint_records: dict[str, Any] = {}
    for endpoint, alpha in CONTRACTIONS:
        payload = endpoint_payload(
            parent=prepared.parent,
            state=states[endpoint],
            manifest_sha256=prepared.manifest_sha256,
            endpoint=endpoint,
            alpha=alpha,
            audit=audits[endpoint],
        )
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        raw = buffer.getvalue()
        filename = f"exact-fros-bc-{endpoint}.pt"
        checkpoint_bytes[filename] = raw
        checkpoint_records[endpoint] = {
            "filename": filename,
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
            "alpha": alpha,
            "audit": audits[endpoint],
            "evaluation_only": True,
            "promotion_eligible": False,
        }
    if set(checkpoint_bytes) != {
        "exact-fros-bc-E50.pt",
        "exact-fros-bc-E100.pt",
    }:
        raise RuntimeError("Published endpoint set drifted from E50/E100")

    result = {
        "schema_version": SCHEMA_VERSION + "-result",
        "status": "training_integrity_pass_evaluation_required",
        "frozen_manifest_sha256": prepared.manifest_sha256,
        "optimizer_steps": 8,
        "source_order": [source for source, _ in STEP_SCHEDULE],
        "training_trace": trace,
        "endpoints": checkpoint_records,
        "trained_direction_integrity": audits["trained_direction"],
        "inputs_unchanged_after_training": True,
        "observed_input_hashes": current_hashes,
        "only_E50_E100_published": True,
        "gameplay_evaluation_performed": False,
        "submission_authorized": False,
        "scope": prepared.manifest["scope"],
    }

    OUTPUT_DIR.mkdir(mode=0o700, parents=False, exist_ok=False)
    training_core.write_exclusive_bytes(
        OUTPUT_DIR / "frozen_training_manifest.json",
        dry_run_bytes(prepared.manifest),
    )
    for filename, raw in checkpoint_bytes.items():
        training_core.write_exclusive_bytes(OUTPUT_DIR / filename, raw)
    training_core.write_exclusive_json(OUTPUT_DIR / "training_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def enforce_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from repository root: {ROOT}")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"Wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )
    if Path(sys.prefix).resolve() != EXPECTED_ENV_PREFIX.resolve():
        raise RuntimeError(
            f"Wrong environment prefix: {sys.prefix}; expected {EXPECTED_ENV_PREFIX}"
        )
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("Run with my_project_env Python flags -I -B")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the deterministic manifest and perform no writes/training.",
    )
    mode.add_argument(
        "--freeze-manifest",
        type=Path,
        metavar="PATH",
        help="Write one exclusive manifest and perform no training.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Train only after a reviewed frozen manifest is supplied.",
    )
    parser.add_argument("--fros-archive", type=Path, required=True)
    parser.add_argument("--expected-fros-archive-sha256", required=True)
    parser.add_argument("--expected-fros-manifest-sha256", required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--expected-allowlist-sha256", required=True)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def validate_mode_args(args: argparse.Namespace) -> None:
    if args.execute:
        if args.frozen_manifest is None or not args.expected_manifest_sha256:
            raise ValueError(
                "--execute requires --frozen-manifest and "
                "--expected-manifest-sha256"
            )
    elif args.frozen_manifest is not None or args.expected_manifest_sha256:
        raise ValueError(
            "--frozen-manifest/--expected-manifest-sha256 are execute-only"
        )


def main(argv: Sequence[str] | None = None) -> int:
    if RETIRED_NEVER_EXECUTED:
        raise RuntimeError(
            "RETIRED_NEVER_EXECUTED: this update-0 draft may not be dry-run, "
            "frozen, or executed; use the S8 anti-KD repair launcher"
        )
    args = parse_args(argv)
    validate_mode_args(args)
    enforce_runtime()
    prepared = prepare_run(
        fros_archive=args.fros_archive,
        expected_fros_archive_sha256=args.expected_fros_archive_sha256,
        expected_fros_manifest_sha256=args.expected_fros_manifest_sha256,
        allowlist_path=args.allowlist,
        expected_allowlist_sha256=args.expected_allowlist_sha256,
        device=args.device,
    )
    if args.dry_run:
        sys.stdout.buffer.write(dry_run_bytes(prepared.manifest))
        return 0
    if args.freeze_manifest is not None:
        manifest_path = Path(os.path.abspath(os.fspath(args.freeze_manifest)))
        if manifest_path == OUTPUT_DIR:
            raise ValueError("Frozen manifest path must differ from output directory")
        if manifest_path.exists() or manifest_path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")
        if not manifest_path.parent.is_dir() or manifest_path.parent.is_symlink():
            raise FileNotFoundError(
                f"Manifest parent must be an existing non-symlink directory: "
                f"{manifest_path.parent}"
            )
        training_core.write_exclusive_bytes(
            manifest_path, dry_run_bytes(prepared.manifest)
        )
        print(prepared.manifest_sha256)
        return 0

    frozen = load_frozen_manifest(
        Path(os.path.abspath(os.fspath(args.frozen_manifest))),
        args.expected_manifest_sha256,
    )
    if frozen != prepared.manifest or sha256_json(frozen) != prepared.manifest_sha256:
        raise RuntimeError("Current inputs/protocol differ from reviewed manifest")
    return execute(prepared, args.device)


if __name__ == "__main__":
    raise SystemExit(main())

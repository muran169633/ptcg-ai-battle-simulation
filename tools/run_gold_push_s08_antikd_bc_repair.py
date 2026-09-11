#!/usr/bin/env python3
"""Freeze or run the S8 anti-KD actor6 ordered-BC micro-repair.

The route is deliberately closed:

* parent: the frozen S8 checkpoint;
* eight batches: antiKD/general/antiKD/exactFros, repeated twice;
* one fresh AdamW optimizer over actor6 only;
* only S8-relative E50 and E100 contractions are published;
* offline behavior gates are evaluated against the contemporaneous S8 parent;
* gameplay evaluation, packaging, upload, and submission are out of scope.

Dry-run performs the full input/archive/cache audit but no optimizer step and
no write.  Execute requires an exclusive frozen manifest and reviewed digest.
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
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import orjson  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import evaluate_policy_bc as behavior_core  # noqa: E402
import interpolate_ppo_checkpoints as contraction_core  # noqa: E402
import run_gold_push_postppo_tail_repair as training_core  # noqa: E402
import build_marnie_froslass_exact_wins as fros_train_builder  # noqa: E402
import build_marnie_exact_anti_kd as anti_kd_builder  # noqa: E402
import build_marnie_exact_fros_valid_view as fros_valid_builder  # noqa: E402
import build_marnie_general_anchor_exclusion as general_anchor_builder  # noqa: E402


ppo = training_core.ppo

SCHEMA_VERSION = "ptcg-gold-push-s08-antikd-bc-repair-v1"
DATA_SCHEMA_VERSION = "ptcg-bc-visible-decisions-v1"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
EXPECTED_ENV_PREFIX = Path("/home/xxc/miniconda3/envs/my_project_env")

S8_PARENT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/postppo_tail_repair_r2best_v1/"
    "postppo-special-bc-s08.pt"
)
S8_PARENT_SHA256 = (
    "d7443bda57cb89a5c12d9d776710d1b8b2c01151541573e01f7e04b032ea25e9"
)
BC_ARCHITECTURE = (
    ROOT / "artifacts/gold_push_20260810_v1/bc/"
    "marnie_trainwins_seed1011/best.pt"
)
BC_ARCHITECTURE_SHA256 = (
    "dda68d51d9b922526709149143aa8409ba287fb8f31ddbb2293f0ea543ef01a0"
)
GENERAL_BEHAVIOR_ARCHIVE = (
    ROOT / "data/gold_push_recent7_20260810_v1/archives/marnie.zip"
)
GENERAL_BEHAVIOR_ARCHIVE_SHA256 = (
    "6b3873b28bfad70377d0a3b22fe1163516a3d40b5af429885c0ce15f75b402a0"
)
GENERAL_BEHAVIOR_MANIFEST_SHA256 = (
    "0839b0fd98ceaaa3b66c92c46899e62e8e3318998d6bba0c841536205879ecdb"
)
OUTPUT_DIR = (
    ROOT / "artifacts/gold_push_20260810_v1/s08_antikd_bc_repair_v1"
)

MARNIE_DECK_HASH = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
FROS_DECK_HASH = (
    "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc"
)

FROS_TRAIN_ARCHIVE_SCHEMA = fros_train_builder.SCHEMA_VERSION
FROS_TRAIN_ALLOWLIST_SCHEMA = fros_train_builder.ALLOWLIST_SCHEMA_VERSION
ANTIKD_ARCHIVE_SCHEMA = anti_kd_builder.ARCHIVE_SCHEMA_VERSION
ANTIKD_ALLOWLIST_SCHEMA = anti_kd_builder.ALLOWLIST_SCHEMA_VERSION
FROS_VALID_ARCHIVE_SCHEMA = fros_valid_builder.ARCHIVE_SCHEMA_VERSION
FROS_VALID_ALLOWLIST_SCHEMA = fros_valid_builder.ALLOWLIST_SCHEMA_VERSION
GENERAL_ANCHOR_ARCHIVE_SCHEMA = general_anchor_builder.ARCHIVE_SCHEMA_VERSION
ALLOWLIST_MEMBER = "exact_episode_allowlist.json"

FROS_TRAIN_EPISODES = 104
FROS_TRAIN_ROWS = 10_614
FROS_TRAIN_DATE_COUNTS = {
    "2026-08-02": 26,
    "2026-08-03": 11,
    "2026-08-04": 23,
    "2026-08-05": 29,
    "2026-08-06": 15,
}
ANTIKD_EPISODES = 43
ANTIKD_ROWS = 4_873
ANTIKD_DATE_COUNTS = {
    "2026-08-02": 27,
    "2026-08-03": 5,
    "2026-08-04": 6,
    "2026-08-05": 4,
    "2026-08-06": 1,
}
ANTIKD_TRAIN_EPISODES = 35
ANTIKD_DEV_EPISODES = 8
ANTIKD_TRAIN_ROWS = 3_905
ANTIKD_DEV_ROWS = 968
ANTIKD_SPLIT_DOMAIN = anti_kd_builder.SPLIT_DOMAIN_SEPARATOR
FROS_VALID_CONTRACT = fros_valid_builder.DEFAULT_CONTRACT
FROS_VALID_EPISODES = FROS_VALID_CONTRACT.expected_episodes
FROS_VALID_ROWS = FROS_VALID_CONTRACT.expected_rows
FROS_VALID_WINS = FROS_VALID_CONTRACT.expected_wins
FROS_VALID_LOSSES = FROS_VALID_CONTRACT.expected_losses
FROS_VALID_DATE_COUNTS = {
    FROS_VALID_CONTRACT.dataset_date: FROS_VALID_EPISODES
}
GENERAL_SOURCE_EPISODES = 1_909
GENERAL_SOURCE_ROWS = 187_611
GENERAL_ANCHOR_EPISODES = 1_762
GENERAL_ANCHOR_ROWS = 172_124
GENERAL_BEHAVIOR_DATE = FROS_VALID_CONTRACT.dataset_date
GENERAL_BEHAVIOR_EPISODES = FROS_VALID_CONTRACT.source_valid_episodes
GENERAL_BEHAVIOR_VIEWS = FROS_VALID_CONTRACT.source_valid_views
GENERAL_BEHAVIOR_MIRROR_EPISODES = FROS_VALID_CONTRACT.source_mirror_episodes
GENERAL_BEHAVIOR_MIRROR_IDS = FROS_VALID_CONTRACT.source_mirror_episode_ids
GENERAL_BEHAVIOR_ROWS = FROS_VALID_CONTRACT.source_valid_rows
GENERAL_BEHAVIOR_VIEW_OUTCOMES = {"loss": 138, "win": 161}

SEED = 2026081052
CACHE_BATCHES_PER_SOURCE = 8
BATCH_SIZE = 256
CONTEXT34_ROWS_PER_BATCH = 1
ORDER_CONTEXT_WEIGHT = 8.0
RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT = 2.0
LEARNING_RATE = 2e-7
WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-5
MAX_GRAD_NORM = 1.0
MAX_FULL_ACTOR6_L2 = 2.5e-4

ANTIKD_BATCH_INDICES = (0, 2, 4, 6)
GENERAL_BATCH_INDICES = (1, 5)
FROS_BATCH_INDICES = (3, 7)
STEP_SCHEDULE = (
    ("anti_kd", ANTIKD_BATCH_INDICES[0]),
    ("general", GENERAL_BATCH_INDICES[0]),
    ("anti_kd", ANTIKD_BATCH_INDICES[1]),
    ("fros", FROS_BATCH_INDICES[0]),
    ("anti_kd", ANTIKD_BATCH_INDICES[2]),
    ("general", GENERAL_BATCH_INDICES[1]),
    ("anti_kd", ANTIKD_BATCH_INDICES[3]),
    ("fros", FROS_BATCH_INDICES[1]),
)
CONTRACTIONS = (("E50", 0.50), ("E100", 1.00))
ACTOR6 = training_core.ACTOR6


@dataclass(frozen=True)
class AllowlistRecord:
    path: Path
    raw: bytes
    raw_sha256: str
    canonical_sha256: str
    payload: dict[str, Any]
    episode_keys: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class ArchiveRecord:
    path: Path
    archive_sha256: str
    manifest_sha256: str
    manifest: dict[str, Any]
    members: tuple[str, ...]


@dataclass(frozen=True)
class SplitAudit:
    split: str
    rows: int
    episode_keys: frozenset[tuple[str, str]]
    view_keys: frozenset[tuple[str, str, int, str]]
    date_episode_counts: dict[str, int]
    date_view_counts: dict[str, int]
    outcome_view_counts: dict[str, int]
    view_rewards: dict[tuple[str, str, int, str], float]
    view_opponents: dict[tuple[str, str, int, str], str]
    view_row_counts: dict[tuple[str, str, int, str], int]
    decision_keys_sha256: str


@dataclass
class PreparedRun:
    manifest: dict[str, Any]
    manifest_sha256: str
    parent: dict[str, Any]
    bc_checkpoint: dict[str, Any]
    configs: dict[str, argparse.Namespace]
    caches: dict[str, list[dict[str, torch.Tensor]]]
    batch_sha256: dict[str, list[str]]
    paths: dict[str, Path]


def canonical_json_bytes(value: Any) -> bytes:
    # This is the shared builder serialization, deliberately without LF.
    return fros_train_builder.canonical_json_bytes(value)


def canonical_line_bytes(value: Any) -> bytes:
    # Every reviewed builder defines canonical document bytes as minified,
    # sorted JSON plus one LF.  Raw and canonical digests remain separately
    # named even when the deterministic output makes them equal.
    return canonical_json_bytes(value) + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_line_bytes(value))


def file_sha256(path: Path) -> str:
    return training_core.file_sha256(path)


def validate_sha256(value: str, label: str) -> str:
    training_core.validate_sha256_text(value, label)
    return value


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def require_regular_file(path: Path, expected_sha256: str, label: str) -> Path:
    validate_sha256(expected_sha256, f"{label} SHA-256")
    path = fros_train_builder.audit_path_components(
        path,
        label,
        must_exist=True,
        kind="file",
    )
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return path


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{label} must be a positive integer")
    return value


def _episode_key(record: Mapping[str, Any], label: str) -> tuple[str, str]:
    date = record.get("date", record.get("dataset_date"))
    episode_id = record.get("episode_id")
    if not isinstance(date, str) or not date:
        raise RuntimeError(f"{label} lacks a nonempty date")
    if not isinstance(episode_id, str) or not episode_id:
        raise RuntimeError(f"{label} lacks a nonempty episode_id")
    return date, episode_id


def anti_kd_split_score(date: str, episode_id: str) -> str:
    return anti_kd_builder.split_score(date, episode_id, ANTIKD_SPLIT_DOMAIN)


def expected_anti_kd_dev_keys(
    episode_keys: Sequence[tuple[str, str]],
    *,
    contract: anti_kd_builder.AntiKDContract = anti_kd_builder.DEFAULT_CONTRACT,
) -> frozenset[tuple[str, str]]:
    expected_episodes = contract.expected_episodes
    if (
        len(episode_keys) != expected_episodes
        or len(set(episode_keys)) != expected_episodes
    ):
        raise RuntimeError(
            "anti-KD split requires exactly "
            f"{expected_episodes} unique episodes"
        )
    ordered = sorted(
        episode_keys,
        key=lambda key: (
            anti_kd_builder.split_score(
                key[0], key[1], contract.split_domain_separator
            ),
            key[0],
            int(key[1]),
        ),
    )
    return frozenset(ordered[: contract.dev_episode_count])


def _validate_common_allowlist(
    payload: Any,
    *,
    schema: str,
    episodes: int,
    rows: int,
    date_counts: dict[str, int],
    split: str,
    learner_deck_hash: str = MARNIE_DECK_HASH,
) -> tuple[dict[str, Any], frozenset[tuple[str, str]]]:
    if not isinstance(payload, dict):
        raise TypeError("allowlist root must be an object")
    expected = {
        "schema_version": schema,
        "learner_deck_hash": learner_deck_hash,
        "split": split,
        "episode_count": episodes,
        "decision_rows": rows,
        "date_episode_counts": date_counts,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"allowlist {schema} field {key} mismatch: "
                f"expected {value!r}, got {payload.get(key)!r}"
            )
    episode_ids = payload.get("episode_ids")
    records = payload.get("episodes")
    if not isinstance(episode_ids, list) or not isinstance(records, list):
        raise TypeError("allowlist requires episode_ids and episodes lists")
    if len(episode_ids) != episodes or len(records) != episodes:
        raise RuntimeError("allowlist episode list length mismatch")
    if any(not isinstance(value, str) or not value for value in episode_ids):
        raise RuntimeError("allowlist episode_ids must be nonempty strings")
    if len(set(episode_ids)) != episodes:
        raise RuntimeError("allowlist episode_ids must be unique")
    keys: list[tuple[str, str]] = []
    row_sum = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"allowlist episodes[{index}] must be an object")
        key = _episode_key(record, f"allowlist episodes[{index}]")
        if key[0] not in date_counts:
            raise RuntimeError("allowlist episode date is outside the frozen dates")
        row_sum += _positive_int(
            record.get("decision_rows"),
            f"allowlist episodes[{index}].decision_rows",
        )
        keys.append(key)
    if [episode_id for _, episode_id in keys] != episode_ids:
        raise RuntimeError("allowlist episode_ids do not match episode records")
    if len(set(keys)) != episodes:
        raise RuntimeError("allowlist episode records are not unique")
    if row_sum != rows:
        raise RuntimeError("allowlist per-episode row total mismatch")
    observed_dates = Counter(date for date, _ in keys)
    if dict(sorted(observed_dates.items())) != date_counts:
        raise RuntimeError("allowlist per-episode date counts mismatch")
    return payload, frozenset(keys)


def validate_fros_train_allowlist(
    payload: Any,
    *,
    contract: fros_train_builder.BuildContract = fros_train_builder.DEFAULT_CONTRACT,
) -> tuple[dict[str, Any], frozenset[tuple[str, str]]]:
    shared = fros_train_builder.validate_allowlist_document(
        canonical_line_bytes(payload),
        contract,
    )
    if shared != payload:
        raise RuntimeError("shared Fros train validator changed the document")
    payload, keys = _validate_common_allowlist(
        payload,
        schema=FROS_TRAIN_ALLOWLIST_SCHEMA,
        episodes=contract.expected_selected_episodes,
        rows=contract.expected_selected_rows,
        date_counts=dict(contract.expected_date_episodes),
        split="train",
        learner_deck_hash=contract.learner_deck_hash,
    )
    if payload.get("opponent_deck_hash") != contract.opponent_deck_hash:
        raise RuntimeError("Fros train allowlist opponent deck hash mismatch")
    if payload.get("terminal_reward") != "win":
        raise RuntimeError("Fros train allowlist is not win-only")
    return payload, keys


def validate_anti_kd_allowlist(
    payload: Any,
    *,
    contract: anti_kd_builder.AntiKDContract = anti_kd_builder.DEFAULT_CONTRACT,
) -> tuple[dict[str, Any], frozenset[tuple[str, str]]]:
    shared = anti_kd_builder.validate_allowlist_document(
        canonical_line_bytes(payload),
        contract,
    )
    if shared != payload:
        raise RuntimeError("shared anti-KD validator changed the document")
    payload, keys = _validate_common_allowlist(
        payload,
        schema=ANTIKD_ALLOWLIST_SCHEMA,
        episodes=contract.expected_episodes,
        rows=contract.expected_rows,
        date_counts=dict(contract.expected_date_episodes),
        split="train",
        learner_deck_hash=contract.base.learner_deck_hash,
    )
    if payload.get("learner_deck_hash") != contract.base.learner_deck_hash:
        raise RuntimeError("anti-KD allowlist learner deck hash mismatch")
    if payload.get("opponent_deck_hash") != contract.base.opponent_deck_hash:
        raise RuntimeError("anti-KD allowlist opponent deck hash mismatch")
    if payload.get("opponent_team_name") != contract.kd_team_name:
        raise RuntimeError("anti-KD allowlist opponent team mismatch")
    if payload.get("terminal_reward") != "win":
        raise RuntimeError("anti-KD allowlist is not win-only")
    if payload.get("source_split") != "train":
        raise RuntimeError("anti-KD source_split must be train")
    if payload.get("partition_role") != "train_dev":
        raise RuntimeError("anti-KD partition_role must be train_dev")
    rule = payload.get("split_rule")
    expected_rule = {
        "algorithm": anti_kd_builder.SPLIT_ALGORITHM,
        "domain_separator": contract.split_domain_separator,
        "payload": anti_kd_builder.SPLIT_PAYLOAD,
        "order": anti_kd_builder.SPLIT_ORDER,
        "dev_count": contract.dev_episode_count,
    }
    if rule != expected_rule:
        raise RuntimeError("anti-KD split_rule mismatch")
    if payload.get("train_episode_count") != contract.train_episode_count:
        raise RuntimeError("anti-KD train episode count mismatch")
    if payload.get("dev_episode_count") != contract.dev_episode_count:
        raise RuntimeError("anti-KD dev episode count mismatch")
    expected_dev = expected_anti_kd_dev_keys(list(keys), contract=contract)
    observed_dev = frozenset(
        _episode_key(record, "anti-KD episode")
        for record in payload["episodes"]
        if record.get("derived_split") == "dev"
    )
    observed_train = frozenset(
        _episode_key(record, "anti-KD episode")
        for record in payload["episodes"]
        if record.get("derived_split") == "train"
    )
    if observed_dev != expected_dev:
        raise RuntimeError("anti-KD dev split does not match preregistered hashes")
    if observed_train != keys - expected_dev:
        raise RuntimeError("anti-KD train split does not complement dev")
    return payload, keys


def validate_fros_valid_allowlist(
    payload: Any,
) -> tuple[dict[str, Any], frozenset[tuple[str, str]]]:
    shared = fros_valid_builder.validate_allowlist_document(
        canonical_line_bytes(payload)
    )
    if shared != payload:
        raise RuntimeError("shared Fros-valid validator changed the document")
    payload, keys = _validate_common_allowlist(
        payload,
        schema=FROS_VALID_ALLOWLIST_SCHEMA,
        episodes=FROS_VALID_EPISODES,
        rows=FROS_VALID_ROWS,
        date_counts=FROS_VALID_DATE_COUNTS,
        split="valid",
    )
    if payload.get("opponent_deck_hash") != FROS_DECK_HASH:
        raise RuntimeError("Fros-valid allowlist opponent deck hash mismatch")
    if payload.get("terminal_reward") != "all":
        raise RuntimeError("Fros-valid allowlist must contain all outcomes")
    if payload.get("win_episodes") != FROS_VALID_WINS:
        raise RuntimeError("Fros-valid allowlist win count mismatch")
    if payload.get("loss_episodes") != FROS_VALID_LOSSES:
        raise RuntimeError("Fros-valid allowlist loss count mismatch")
    return payload, keys


def validate_general_anchor_manifest(
    payload: Any,
    *,
    expected_fros_allowlist_sha256: str,
    expected_anti_allowlist_sha256: str,
    contract: general_anchor_builder.GeneralContract = (
        general_anchor_builder.DEFAULT_CONTRACT
    ),
    expected_source_train_members: Sequence[tuple[str, int, str]] | None = None,
) -> dict[str, Any]:
    """Bind a builder-validated general manifest to exact source members."""

    document = general_anchor_builder.validate_manifest_document(
        payload,
        contract,
        expected_fros_allowlist_sha256=expected_fros_allowlist_sha256,
        expected_anti_allowlist_sha256=expected_anti_allowlist_sha256,
    )
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError("general anchor manifest lacks source lineage")
    if source.get("logical_path") != contract.source.source_logical_path:
        raise RuntimeError("general anchor source logical path mismatch")
    members = source.get("train_members")
    if not isinstance(members, list):
        raise RuntimeError("general anchor source train members are missing")
    try:
        observed = tuple(
            (str(item["member"]), int(item["bytes"]), str(item["sha256"]))
            for item in members
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "general anchor source train member identity is malformed"
        ) from error
    expected = tuple(
        contract.source_train_member_audit
        if expected_source_train_members is None
        else expected_source_train_members
    )
    if observed != expected:
        raise RuntimeError(
            "general anchor source train member name/size/SHA mismatch"
        )
    return document


def load_allowlist(
    path: Path,
    *,
    expected_raw_sha256: str,
    expected_canonical_sha256: str,
    label: str,
    validator: Any,
) -> AllowlistRecord:
    path = require_regular_file(path, expected_raw_sha256, label)
    validate_sha256(expected_canonical_sha256, f"{label} canonical SHA-256")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    payload, keys = validator(payload)
    canonical_sha256 = sha256_bytes(canonical_line_bytes(payload))
    if canonical_sha256 != expected_canonical_sha256:
        raise RuntimeError(
            f"{label} canonical SHA-256 mismatch: expected "
            f"{expected_canonical_sha256}, got {canonical_sha256}"
        )
    if raw != canonical_line_bytes(payload):
        raise RuntimeError(f"{label} bytes are not canonical JSON plus one LF")
    return AllowlistRecord(
        path=path,
        raw=raw,
        raw_sha256=expected_raw_sha256,
        canonical_sha256=expected_canonical_sha256,
        payload=payload,
        episode_keys=keys,
    )


def _validate_zip_member(info: zipfile.ZipInfo, label: str) -> None:
    member = PurePosixPath(info.filename)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise RuntimeError(f"Unsafe {label} ZIP member: {info.filename!r}")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise RuntimeError(f"Symlink {label} ZIP member is forbidden")
    if info.flag_bits & 0x1:
        raise RuntimeError(f"Encrypted {label} ZIP member is forbidden")


def load_archive(
    path: Path,
    *,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    expected_schema: str,
    label: str,
    allowlist: AllowlistRecord | None = None,
    manifest_validator: Any | None = None,
) -> ArchiveRecord:
    path = require_regular_file(path, expected_archive_sha256, label)
    validate_sha256(expected_manifest_sha256, f"{label} manifest SHA-256")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            members = [info.filename for info in infos]
            if len(members) != len(set(members)):
                raise RuntimeError(f"{label} contains duplicate ZIP members")
            for info in infos:
                _validate_zip_member(info, label)
            if members.count("manifest.json") != 1:
                raise RuntimeError(f"{label} requires exactly one manifest.json")
            manifest_raw = archive.read("manifest.json")
            if sha256_bytes(manifest_raw) != expected_manifest_sha256:
                raise RuntimeError(f"{label} embedded manifest digest mismatch")
            manifest = json.loads(manifest_raw)
            if allowlist is not None:
                if members.count(ALLOWLIST_MEMBER) != 1:
                    raise RuntimeError(
                        f"{label} requires exactly one {ALLOWLIST_MEMBER}"
                    )
                if archive.read(ALLOWLIST_MEMBER) != allowlist.raw:
                    raise RuntimeError(f"{label} embedded allowlist bytes differ")
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not a valid bound ZIP archive") from error
    if not isinstance(manifest, dict):
        raise TypeError(f"{label} manifest root must be an object")
    if manifest_validator is not None:
        shared_manifest = manifest_validator(manifest_raw)
        if shared_manifest != manifest:
            raise RuntimeError(f"{label} shared validator changed the manifest")
    if manifest.get("schema_version") != expected_schema:
        raise RuntimeError(
            f"{label} top-level schema mismatch: expected {expected_schema!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    observed_data_schema = manifest.get("data_schema_version")
    if observed_data_schema is None and expected_schema == DATA_SCHEMA_VERSION:
        # The frozen pre-derived Aug7 archive predates the two-level schema
        # convention; its exact top-level schema is the row data schema.
        observed_data_schema = manifest.get("schema_version")
    if observed_data_schema != DATA_SCHEMA_VERSION:
        raise RuntimeError(f"{label} data_schema_version mismatch")
    if allowlist is not None:
        reference = manifest.get("exact_episode_allowlist")
        if not isinstance(reference, dict):
            raise RuntimeError(f"{label} lacks exact_episode_allowlist binding")
        expected_reference = {
            "member": ALLOWLIST_MEMBER,
            "schema_version": allowlist.payload["schema_version"],
            "sha256": allowlist.raw_sha256,
            "canonical_sha256": allowlist.canonical_sha256,
            "episodes": allowlist.payload["episode_count"],
            "decision_rows": allowlist.payload["decision_rows"],
        }
        for key, value in expected_reference.items():
            if reference.get(key) != value:
                raise RuntimeError(f"{label} allowlist reference {key} mismatch")
    return ArchiveRecord(
        path=path,
        archive_sha256=expected_archive_sha256,
        manifest_sha256=expected_manifest_sha256,
        manifest=manifest,
        members=tuple(members),
    )


def _view_key(
    row: Mapping[str, Any],
    label: str,
) -> tuple[str, str, int, str]:
    date, episode_id = _episode_key(row, label)
    seat = row.get("seat")
    if isinstance(seat, bool) or not isinstance(seat, int) or seat not in (0, 1):
        raise RuntimeError(f"{label} has invalid seat")
    team_name = row.get("team_name")
    if not isinstance(team_name, str) or not team_name:
        raise RuntimeError(f"{label} has empty team_name")
    return date, episode_id, seat, team_name


def _mirror_episode_keys(
    audit: SplitAudit,
) -> tuple[tuple[str, str], ...]:
    counts = Counter((date, episode_id) for date, episode_id, _seat, _team in audit.view_keys)
    return tuple(
        sorted(
            (key for key, count in counts.items() if count > 1),
            key=lambda key: (key[0], int(key[1])),
        )
    )


def _view_count_histogram(audit: SplitAudit) -> dict[int, int]:
    counts = Counter((date, episode_id) for date, episode_id, _seat, _team in audit.view_keys)
    return dict(sorted(Counter(counts.values()).items()))


def _validate_mirror_views(audit: SplitAudit, label: str) -> None:
    by_episode: dict[tuple[str, str], list[tuple[str, str, int, str]]] = {}
    for view_key in audit.view_keys:
        by_episode.setdefault(view_key[:2], []).append(view_key)
    for episode_key, views in by_episode.items():
        if len(views) == 1:
            continue
        if len(views) != 2:
            raise RuntimeError(f"{label} mirror {episode_key} must have exactly two views")
        if {view[2] for view in views} != {0, 1}:
            raise RuntimeError(f"{label} mirror {episode_key} must cover seats 0 and 1")
        teams = {view[3] for view in views}
        if len(teams) != 2:
            raise RuntimeError(f"{label} mirror {episode_key} repeats learner team")
        first, second = sorted(views, key=lambda view: view[2])
        if (
            audit.view_opponents[first] != second[3]
            or audit.view_opponents[second] != first[3]
        ):
            raise RuntimeError(f"{label} mirror {episode_key} is not reverse-routed")
        if {audit.view_rewards[first], audit.view_rewards[second]} != {-1.0, 1.0}:
            raise RuntimeError(f"{label} mirror {episode_key} lacks +/-1 rewards")


def _validate_single_view_split(
    audit: SplitAudit,
    *,
    label: str,
    expected_rows: int,
    expected_episodes: int,
    expected_outcomes: dict[str, int],
) -> None:
    if audit.rows != expected_rows:
        raise RuntimeError(f"{label} row count mismatch")
    if len(audit.episode_keys) != expected_episodes:
        raise RuntimeError(f"{label} episode count mismatch")
    if len(audit.view_keys) != expected_episodes:
        raise RuntimeError(f"{label} must contain exactly one view per episode")
    if _mirror_episode_keys(audit):
        raise RuntimeError(f"{label} unexpectedly contains mirror episodes")
    if audit.outcome_view_counts != expected_outcomes:
        raise RuntimeError(f"{label} outcome-view counts mismatch")


def validate_general_behavior_split(audit: SplitAudit) -> None:
    fros_valid_builder.validate_contract(FROS_VALID_CONTRACT)
    _validate_mirror_views(audit, "Aug7 general behavior")
    expected_mirrors = tuple(
        (GENERAL_BEHAVIOR_DATE, episode_id)
        for episode_id in GENERAL_BEHAVIOR_MIRROR_IDS
    )
    checks = {
        "rows": (audit.rows, GENERAL_BEHAVIOR_ROWS),
        "episodes": (len(audit.episode_keys), GENERAL_BEHAVIOR_EPISODES),
        "views": (len(audit.view_keys), GENERAL_BEHAVIOR_VIEWS),
        "mirror episodes": (
            len(_mirror_episode_keys(audit)),
            GENERAL_BEHAVIOR_MIRROR_EPISODES,
        ),
        "mirror IDs": (_mirror_episode_keys(audit), expected_mirrors),
        "view-count histogram": (
            _view_count_histogram(audit),
            {1: GENERAL_BEHAVIOR_EPISODES - GENERAL_BEHAVIOR_MIRROR_EPISODES, 2: 4},
        ),
        "date episode counts": (
            audit.date_episode_counts,
            {GENERAL_BEHAVIOR_DATE: GENERAL_BEHAVIOR_EPISODES},
        ),
        "date view counts": (
            audit.date_view_counts,
            {GENERAL_BEHAVIOR_DATE: GENERAL_BEHAVIOR_VIEWS},
        ),
        "outcome-view counts": (
            audit.outcome_view_counts,
            GENERAL_BEHAVIOR_VIEW_OUTCOMES,
        ),
    }
    failures = [
        f"{name}: {actual!r} != {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise RuntimeError("Aug7 general behavior contract drift: " + "; ".join(failures))


def scan_archive_split(record: ArchiveRecord, split: str) -> SplitAudit:
    rows = 0
    view_rewards: dict[tuple[str, str, int, str], float] = {}
    view_opponents: dict[tuple[str, str, int, str], str] = {}
    view_row_counts: Counter[tuple[str, str, int, str]] = Counter()
    decision_keys: set[tuple[str, str, int, str, int]] = set()
    with zipfile.ZipFile(record.path) as archive:
        members = sorted(
            name
            for name in record.members
            if name.startswith(f"{split}/") and name.endswith(".jsonl")
        )
        if not members:
            raise RuntimeError(f"{record.path.name} has no {split} JSONL members")
        for member in members:
            with archive.open(member) as stream:
                for line_number, raw in enumerate(stream, 1):
                    try:
                        row = orjson.loads(raw)
                    except orjson.JSONDecodeError as error:
                        raise RuntimeError(
                            f"Invalid JSON row {member}:{line_number}"
                        ) from error
                    if not isinstance(row, dict):
                        raise RuntimeError(f"Non-object row {member}:{line_number}")
                    label = f"{member}:{line_number}"
                    view_key = _view_key(row, label)
                    opponent = row.get("opponent_team_name")
                    if not isinstance(opponent, str) or not opponent:
                        raise RuntimeError(f"{label} has empty opponent_team_name")
                    reward = row.get("terminal_reward")
                    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                        raise RuntimeError(f"Invalid terminal_reward at {label}")
                    reward = float(reward)
                    if not math.isfinite(reward) or reward not in (-1.0, 0.0, 1.0):
                        raise RuntimeError(f"Invalid terminal_reward at {label}")
                    previous_reward = view_rewards.setdefault(view_key, reward)
                    previous_opponent = view_opponents.setdefault(view_key, opponent)
                    if previous_reward != reward or previous_opponent != opponent:
                        raise RuntimeError(
                            f"Composite view identity/reward drift within {view_key}"
                        )
                    action_step = row.get("action_step_index")
                    if (
                        isinstance(action_step, bool)
                        or not isinstance(action_step, int)
                        or action_step <= 0
                    ):
                        raise RuntimeError(f"{label} has invalid action_step_index")
                    decision_key = (*view_key, action_step)
                    if decision_key in decision_keys:
                        raise RuntimeError(f"Duplicate composite decision {decision_key}")
                    decision_keys.add(decision_key)
                    view_row_counts[view_key] += 1
                    rows += 1
    outcome_counts = Counter(
        "win" if reward > 0 else "loss" if reward < 0 else "draw"
        for reward in view_rewards.values()
    )
    episode_keys = frozenset(view_key[:2] for view_key in view_rewards)
    view_keys = frozenset(view_rewards)
    date_episode_counts = Counter(date for date, _episode_id in episode_keys)
    date_view_counts = Counter(date for date, _episode_id, _seat, _team in view_keys)
    return SplitAudit(
        split=split,
        rows=rows,
        episode_keys=episode_keys,
        view_keys=view_keys,
        date_episode_counts=dict(sorted(date_episode_counts.items())),
        date_view_counts=dict(sorted(date_view_counts.items())),
        outcome_view_counts=dict(sorted(outcome_counts.items())),
        view_rewards=view_rewards,
        view_opponents=view_opponents,
        view_row_counts=dict(view_row_counts),
        decision_keys_sha256=sha256_json(sorted(decision_keys)),
    )


def validate_general_anchor_archive(record: ArchiveRecord) -> None:
    expected_members = (
        general_anchor_builder.TRAIN_MEMBER,
        general_anchor_builder.MANIFEST_MEMBER,
    )
    if record.members != expected_members:
        raise RuntimeError("general anchor archive member/order mismatch")
    declarations = record.manifest.get("members")
    if not isinstance(declarations, list) or len(declarations) != 1:
        raise RuntimeError("general anchor archive lacks one train declaration")
    declaration = declarations[0]
    if not isinstance(declaration, Mapping):
        raise RuntimeError("general anchor train declaration is malformed")
    if declaration.get("member") != general_anchor_builder.TRAIN_MEMBER:
        raise RuntimeError("general anchor declared train member mismatch")
    with zipfile.ZipFile(record.path) as archive:
        raw = archive.read(general_anchor_builder.TRAIN_MEMBER)
    if declaration.get("bytes") != len(raw):
        raise RuntimeError("general anchor train member size mismatch")
    if declaration.get("sha256") != sha256_bytes(raw):
        raise RuntimeError("general anchor train member SHA mismatch")


def validate_archive_relationships(
    *,
    anti_allowlist: AllowlistRecord,
    anti_archive: ArchiveRecord,
    fros_allowlist: AllowlistRecord,
    fros_archive: ArchiveRecord,
    fros_valid_allowlist: AllowlistRecord,
    fros_valid_archive: ArchiveRecord,
    general_anchor: ArchiveRecord,
    general_behavior: ArchiveRecord,
) -> dict[str, SplitAudit]:
    if anti_allowlist.episode_keys & fros_allowlist.episode_keys:
        raise RuntimeError("anti-KD and Fros training episodes overlap")
    anti_train = scan_archive_split(anti_archive, "train")
    anti_dev = scan_archive_split(anti_archive, "dev")
    fros_train = scan_archive_split(fros_archive, "train")
    fros_valid = scan_archive_split(fros_valid_archive, "valid")
    general_train = scan_archive_split(general_anchor, "train")
    general_valid = scan_archive_split(general_behavior, "valid")

    anti_dev_expected = expected_anti_kd_dev_keys(
        list(anti_allowlist.episode_keys)
    )
    if anti_dev.episode_keys != anti_dev_expected:
        raise RuntimeError("anti-KD archive dev episodes differ from hash split")
    if anti_train.episode_keys != anti_allowlist.episode_keys - anti_dev_expected:
        raise RuntimeError("anti-KD archive train episodes differ from hash split")
    if anti_train.rows + anti_dev.rows != ANTIKD_ROWS:
        raise RuntimeError("anti-KD archive total rows mismatch")
    _validate_single_view_split(
        anti_train,
        label="anti-KD train",
        expected_rows=ANTIKD_TRAIN_ROWS,
        expected_episodes=ANTIKD_TRAIN_EPISODES,
        expected_outcomes={"win": ANTIKD_TRAIN_EPISODES},
    )
    _validate_single_view_split(
        anti_dev,
        label="anti-KD dev",
        expected_rows=ANTIKD_DEV_ROWS,
        expected_episodes=ANTIKD_DEV_EPISODES,
        expected_outcomes={"win": ANTIKD_DEV_EPISODES},
    )

    if fros_train.episode_keys != fros_allowlist.episode_keys:
        raise RuntimeError("Fros train archive episodes differ from allowlist")
    _validate_single_view_split(
        fros_train,
        label="Fros train",
        expected_rows=FROS_TRAIN_ROWS,
        expected_episodes=FROS_TRAIN_EPISODES,
        expected_outcomes={"win": FROS_TRAIN_EPISODES},
    )
    if fros_valid.episode_keys != fros_valid_allowlist.episode_keys:
        raise RuntimeError("Fros-valid archive episodes differ from allowlist")
    _validate_single_view_split(
        fros_valid,
        label="Fros-valid",
        expected_rows=FROS_VALID_ROWS,
        expected_episodes=FROS_VALID_EPISODES,
        expected_outcomes={"loss": FROS_VALID_LOSSES, "win": FROS_VALID_WINS},
    )
    _validate_single_view_split(
        general_train,
        label="general anchor",
        expected_rows=GENERAL_ANCHOR_ROWS,
        expected_episodes=GENERAL_ANCHOR_EPISODES,
        expected_outcomes={"win": GENERAL_ANCHOR_EPISODES},
    )
    validate_general_behavior_split(general_valid)

    excluded = anti_allowlist.episode_keys | fros_allowlist.episode_keys
    if general_train.episode_keys & excluded:
        raise RuntimeError("general anchor leaked anti-KD/Fros episodes")
    if sum(map(len, (general_train.episode_keys, excluded))) != GENERAL_SOURCE_EPISODES:
        raise RuntimeError("general anchor plus exclusions do not reconstruct source episodes")
    if general_train.rows + ANTIKD_ROWS + FROS_TRAIN_ROWS != GENERAL_SOURCE_ROWS:
        raise RuntimeError("general anchor plus exclusions do not reconstruct source rows")

    exclusion_sources = general_anchor.manifest.get("exclusion_sources")
    if not isinstance(exclusion_sources, dict):
        raise RuntimeError("general anchor manifest lacks exclusion_sources")
    for name, allowlist, expected_episodes, expected_rows in (
        ("fros", fros_allowlist, FROS_TRAIN_EPISODES, FROS_TRAIN_ROWS),
        ("anti_kd", anti_allowlist, ANTIKD_EPISODES, ANTIKD_ROWS),
    ):
        reference = exclusion_sources.get(name)
        if not isinstance(reference, dict):
            raise RuntimeError(f"general anchor lacks {name} exclusion binding")
        expected = {
            "raw_sha256": allowlist.raw_sha256,
            "canonical_sha256": allowlist.canonical_sha256,
            "schema_version": allowlist.payload["schema_version"],
            "episodes": expected_episodes,
            "rows": expected_rows,
        }
        for key, value in expected.items():
            if reference.get(key) != value:
                raise RuntimeError(f"general anchor {name}.{key} mismatch")
    return {
        "anti_kd_train": anti_train,
        "anti_kd_dev": anti_dev,
        "fros_train": fros_train,
        "fros_valid": fros_valid,
        "general_train": general_train,
        "general_valid": general_valid,
    }


def validate_protocol_constants() -> None:
    if tuple(source for source, _ in STEP_SCHEDULE) != (
        "anti_kd",
        "general",
        "anti_kd",
        "fros",
    ) * 2:
        raise RuntimeError("S8 repair source order drifted")
    if len(STEP_SCHEDULE) != 8 or len(set(STEP_SCHEDULE)) != 8:
        raise RuntimeError("S8 repair must use eight unique scheduled batches")
    if CONTRACTIONS != (("E50", 0.50), ("E100", 1.00)):
        raise RuntimeError("Only E50/E100 contractions are permitted")
    expected_core = {
        "BATCH_SIZE": BATCH_SIZE,
        "ORDER_CONTEXT_WEIGHT": ORDER_CONTEXT_WEIGHT,
        "RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT": RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
    }
    for name, expected in expected_core.items():
        if getattr(training_core, name) != expected:
            raise RuntimeError(f"training core {name} is incompatible")
    if contraction_core.PPO_FEATURE_VERSION != ppo.PPO_FEATURE_VERSION:
        raise RuntimeError("contraction core PPO feature version mismatch")


def replay_config(
    parent_config: dict[str, Any], archive: Path, source_offset: int
) -> argparse.Namespace:
    return training_core.replay_config(
        parent_config,
        archive=archive,
        split="train",
        batches=CACHE_BATCHES_PER_SOURCE,
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
        expected_batches=CACHE_BATCHES_PER_SOURCE,
        expected_context34_rows_per_batch=CONTEXT34_ROWS_PER_BATCH,
    )
    cache_sha256, per_batch = training_core.audit.replay_cache_manifest(batches)
    indices = _scheduled_indices(source)
    fixed_multi_rows = []
    for index in indices:
        batch = batches[index]
        count = int(
            (
                (batch["contexts"] != ppo.SKILL_ORDER_CONTEXT)
                & (batch["min_counts"] == batch["max_counts"])
                & (batch["action_counts"] > 1)
            ).sum()
        )
        if count <= 0:
            raise RuntimeError(
                f"Scheduled {source} batch {index} lacks fixed multi rows"
            )
        fixed_multi_rows.append(count)
    return (
        {
            **profile,
            "sha256": cache_sha256,
            "batch_sha256": per_batch,
            "split": "train",
            "scheduled_batch_indices": indices,
            "scheduled_batch_sha256": [per_batch[index] for index in indices],
            "scheduled_non_context34_fixed_multi_action_rows": fixed_multi_rows,
        },
        per_batch,
    )


def _split_audit_json(audit: SplitAudit) -> dict[str, Any]:
    mirror_keys = _mirror_episode_keys(audit)
    row_count_records = sorted(
        [*view_key, count]
        for view_key, count in audit.view_row_counts.items()
    )
    return {
        "split": audit.split,
        "rows": audit.rows,
        "episodes": len(audit.episode_keys),
        "views": len(audit.view_keys),
        "episode_keys_sha256": sha256_json(sorted(audit.episode_keys)),
        "view_keys_sha256": sha256_json(sorted(audit.view_keys)),
        "decision_keys_sha256": audit.decision_keys_sha256,
        "view_row_counts_sha256": sha256_json(row_count_records),
        "date_episode_counts": audit.date_episode_counts,
        "date_view_counts": audit.date_view_counts,
        "outcome_view_counts": audit.outcome_view_counts,
        "view_count_histogram": {
            str(count): episodes
            for count, episodes in _view_count_histogram(audit).items()
        },
        "mirror_episodes": len(mirror_keys),
        "mirror_episode_keys": [list(key) for key in mirror_keys],
        "mirror_episode_keys_sha256": sha256_json(mirror_keys),
    }


def _path_record(record: ArchiveRecord) -> dict[str, Any]:
    return {
        "path": str(record.path),
        "sha256": record.archive_sha256,
        "embedded_manifest_sha256": record.manifest_sha256,
        "schema_version": record.manifest["schema_version"],
        "data_schema_version": record.manifest["data_schema_version"],
    }


def _allowlist_record(record: AllowlistRecord) -> dict[str, Any]:
    return {
        "path": str(record.path),
        "raw_sha256": record.raw_sha256,
        "canonical_sha256": record.canonical_sha256,
        "schema_version": record.payload["schema_version"],
        "episodes": len(record.episode_keys),
        "rows": record.payload["decision_rows"],
    }


def build_manifest_payload(
    *,
    parent: dict[str, Any],
    archives: dict[str, ArchiveRecord],
    allowlists: dict[str, AllowlistRecord],
    split_audits: dict[str, SplitAudit],
    cache_records: dict[str, dict[str, Any]],
    actor_shapes: dict[str, list[int]],
    dependency_hashes: dict[str, str],
    device: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "frozen_before_training",
        "inputs": {
            "s8_parent": {
                "path": str(S8_PARENT),
                "sha256": S8_PARENT_SHA256,
                "update": parent.get("update"),
                "model_state_sha256": training_core.audit.nested_sha256(
                    parent["model_state_dict"]
                ),
            },
            "bc_architecture": {
                "path": str(BC_ARCHITECTURE),
                "sha256": BC_ARCHITECTURE_SHA256,
            },
            "archives": {
                name: _path_record(record) for name, record in archives.items()
            },
            "allowlists": {
                name: _allowlist_record(record)
                for name, record in allowlists.items()
            },
            "split_audits": {
                name: _split_audit_json(audit)
                for name, audit in split_audits.items()
            },
            "dependencies": {
                "launcher": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": dependency_hashes["launcher"],
                },
                "training_core": {
                    "path": str(Path(training_core.__file__).resolve()),
                    "sha256": dependency_hashes["training_core"],
                },
                "trainer_module": {
                    "path": str(training_core.TRAINER),
                    "sha256": dependency_hashes["trainer_module"],
                },
                "contraction_core": {
                    "path": str(Path(contraction_core.__file__).resolve()),
                    "sha256": dependency_hashes["contraction_core"],
                },
                "behavior_core": {
                    "path": str(Path(behavior_core.__file__).resolve()),
                    "sha256": dependency_hashes["behavior_core"],
                },
                "fros_train_builder": {
                    "path": str(Path(fros_train_builder.__file__).resolve()),
                    "sha256": dependency_hashes["fros_train_builder"],
                },
                "anti_kd_builder": {
                    "path": str(Path(anti_kd_builder.__file__).resolve()),
                    "sha256": dependency_hashes["anti_kd_builder"],
                },
                "fros_valid_builder": {
                    "path": str(Path(fros_valid_builder.__file__).resolve()),
                    "sha256": dependency_hashes["fros_valid_builder"],
                },
                "general_anchor_builder": {
                    "path": str(Path(general_anchor_builder.__file__).resolve()),
                    "sha256": dependency_hashes["general_anchor_builder"],
                },
            },
            "python": {
                "invocation_path": str(EXPECTED_PYTHON),
                "resolved_path": str(EXPECTED_PYTHON.resolve()),
                "environment_prefix": str(EXPECTED_ENV_PREFIX),
                "sha256": dependency_hashes["python"],
            },
        },
        "lineage": {
            "parent": "S8",
            "parent_sha256": S8_PARENT_SHA256,
            "parent_update": 0,
            "learner_deck_hash": MARNIE_DECK_HASH,
            "parent_post_ppo_special_bc_sha256": sha256_json(
                parent.get("post_ppo_special_bc")
            ),
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
                {"step": step, "source": source, "batch_index": index}
                for step, (source, index) in enumerate(STEP_SCHEDULE, 1)
            ],
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_eps": ADAM_EPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "batch_size": BATCH_SIZE,
            "loss": "ordered",
            "context34_order_weight": ORDER_CONTEXT_WEIGHT,
            "non_context34_fixed_multi_action_order_weight": (
                RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT
            ),
            "trainable_scope": "actor6",
            "trainable_parameter_names": list(ACTOR6),
            "trainable_parameter_shapes": actor_shapes,
            "full_actor6_l2_max": MAX_FULL_ACTOR6_L2,
            "contractions_relative_to": "S8",
            "contractions": [
                {"name": name, "alpha": alpha} for name, alpha in CONTRACTIONS
            ],
            "only_published_endpoints": ["E50", "E100"],
            "three_training_source_episode_sets_pairwise_disjoint": True,
        },
        "replay_caches": cache_records,
        "behavior_gates": {
            "baseline": "contemporaneous_S8_same_rows",
            "Aug7_general": {
                "split": "valid",
                "rows": GENERAL_BEHAVIOR_ROWS,
                "set_accuracy_drop_max": 0.001,
                "hybrid_accuracy_drop_max": 0.001,
            },
            "Aug7_exact_Fros": {
                "split": "valid",
                "rows": FROS_VALID_ROWS,
                "hybrid_accuracy_must_not_drop": True,
                "ordered_bc_loss_must_not_increase": True,
            },
            "anti_KD_dev": {
                "split": "dev",
                "episodes": ANTIKD_DEV_EPISODES,
                "ordered_bc_loss_must_strictly_decrease": True,
            },
            "count_prediction_digest_must_be_identical_on_all_views": True,
            "count_head_must_be_bit_identical": True,
            "all_non_actor6_tensors_must_be_bit_identical": True,
            "failed_endpoint_not_H2H_eligible": True,
        },
        "output": {
            "directory": str(OUTPUT_DIR),
            "must_not_exist_before_execute": True,
            "checkpoints": ["s08-antikd-E50.pt", "s08-antikd-E100.pt"],
            "evaluation_only": True,
            "resume_forbidden": True,
        },
        "scope": {
            "training": True,
            "offline_behavior_evaluation": True,
            "gameplay_evaluation": False,
            "package": False,
            "upload": False,
            "submission": False,
            "local_only": True,
        },
    }


def prepare_run(args: argparse.Namespace) -> PreparedRun:
    validate_protocol_constants()
    fros_train_builder.audit_path_components(
        OUTPUT_DIR,
        "S8 anti-KD repair output",
        must_exist=False,
        kind="output",
    )
    training_core.assert_output_absent(OUTPUT_DIR)
    require_regular_file(S8_PARENT, S8_PARENT_SHA256, "S8 parent")
    require_regular_file(
        BC_ARCHITECTURE, BC_ARCHITECTURE_SHA256, "BC architecture checkpoint"
    )

    fros_allowlist = load_allowlist(
        args.fros_allowlist,
        expected_raw_sha256=args.expected_fros_allowlist_sha256,
        expected_canonical_sha256=args.expected_fros_allowlist_canonical_sha256,
        label="Fros train allowlist",
        validator=validate_fros_train_allowlist,
    )
    anti_allowlist = load_allowlist(
        args.anti_kd_allowlist,
        expected_raw_sha256=args.expected_anti_kd_allowlist_sha256,
        expected_canonical_sha256=args.expected_anti_kd_allowlist_canonical_sha256,
        label="anti-KD allowlist",
        validator=validate_anti_kd_allowlist,
    )
    fros_valid_allowlist = load_allowlist(
        args.fros_valid_allowlist,
        expected_raw_sha256=args.expected_fros_valid_allowlist_sha256,
        expected_canonical_sha256=(
            args.expected_fros_valid_allowlist_canonical_sha256
        ),
        label="Fros-valid allowlist",
        validator=validate_fros_valid_allowlist,
    )

    archives = {
        "anti_kd": load_archive(
            args.anti_kd_archive,
            expected_archive_sha256=args.expected_anti_kd_archive_sha256,
            expected_manifest_sha256=args.expected_anti_kd_manifest_sha256,
            expected_schema=ANTIKD_ARCHIVE_SCHEMA,
            label="anti-KD archive",
            allowlist=anti_allowlist,
        ),
        "fros_train": load_archive(
            args.fros_archive,
            expected_archive_sha256=args.expected_fros_archive_sha256,
            expected_manifest_sha256=args.expected_fros_manifest_sha256,
            expected_schema=FROS_TRAIN_ARCHIVE_SCHEMA,
            label="Fros train archive",
            allowlist=fros_allowlist,
        ),
        "fros_valid": load_archive(
            args.fros_valid_archive,
            expected_archive_sha256=args.expected_fros_valid_archive_sha256,
            expected_manifest_sha256=args.expected_fros_valid_manifest_sha256,
            expected_schema=FROS_VALID_ARCHIVE_SCHEMA,
            label="Fros-valid archive",
            allowlist=fros_valid_allowlist,
        ),
        "general_anchor": load_archive(
            args.general_anchor_archive,
            expected_archive_sha256=args.expected_general_anchor_archive_sha256,
            expected_manifest_sha256=args.expected_general_anchor_manifest_sha256,
            expected_schema=GENERAL_ANCHOR_ARCHIVE_SCHEMA,
            label="general anchor archive",
            manifest_validator=partial(
                validate_general_anchor_manifest,
                expected_fros_allowlist_sha256=fros_allowlist.raw_sha256,
                expected_anti_allowlist_sha256=anti_allowlist.raw_sha256,
            ),
        ),
        "general_behavior": load_archive(
            GENERAL_BEHAVIOR_ARCHIVE,
            expected_archive_sha256=GENERAL_BEHAVIOR_ARCHIVE_SHA256,
            expected_manifest_sha256=GENERAL_BEHAVIOR_MANIFEST_SHA256,
            expected_schema=DATA_SCHEMA_VERSION,
            label="Aug7 general behavior archive",
        ),
    }
    validate_general_anchor_archive(archives["general_anchor"])
    # The legacy general behavior archive uses its top-level data schema as
    # schema_version; normalize its explicit data schema only after exact hash
    # verification so the manifest record has the same two-field interface.
    if "data_schema_version" not in archives["general_behavior"].manifest:
        normalized = dict(archives["general_behavior"].manifest)
        normalized["data_schema_version"] = DATA_SCHEMA_VERSION
        archives["general_behavior"] = ArchiveRecord(
            path=archives["general_behavior"].path,
            archive_sha256=archives["general_behavior"].archive_sha256,
            manifest_sha256=archives["general_behavior"].manifest_sha256,
            manifest=normalized,
            members=archives["general_behavior"].members,
        )

    split_audits = validate_archive_relationships(
        anti_allowlist=anti_allowlist,
        anti_archive=archives["anti_kd"],
        fros_allowlist=fros_allowlist,
        fros_archive=archives["fros_train"],
        fros_valid_allowlist=fros_valid_allowlist,
        fros_valid_archive=archives["fros_valid"],
        general_anchor=archives["general_anchor"],
        general_behavior=archives["general_behavior"],
    )

    parent = torch.load(S8_PARENT, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(
        BC_ARCHITECTURE, map_location="cpu", weights_only=False
    )
    if not isinstance(parent, dict) or not isinstance(bc_checkpoint, dict):
        raise TypeError("S8 parent and BC architecture must be checkpoint dictionaries")
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise RuntimeError("S8 parent PPO feature version mismatch")
    if parent.get("update") != 0:
        raise RuntimeError("S8 parent update must remain zero")
    if parent.get("learner_deck_hash") != MARNIE_DECK_HASH:
        raise RuntimeError("S8 parent learner deck hash mismatch")
    special = parent.get("post_ppo_special_bc")
    if not isinstance(special, dict) or special.get("steps") != 8:
        raise RuntimeError("Parent is not the frozen S8 special-BC endpoint")
    if bc_checkpoint.get("feature_version") != ppo.BC_FEATURE_VERSION:
        raise RuntimeError("BC architecture feature version mismatch")
    if not isinstance(parent.get("config"), dict) or not isinstance(
        parent.get("model_config"), dict
    ):
        raise RuntimeError("S8 parent lacks frozen PPO/model config")
    bc_config = bc_checkpoint.get("config")
    if not isinstance(bc_config, dict):
        raise RuntimeError("BC architecture checkpoint lacks config")
    mismatches = {
        key: {"parent": value, "bc": bc_config.get(key)}
        for key, value in parent["model_config"].items()
        if bc_config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"S8/BC architecture mismatch: {mismatches}")

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
        "anti_kd": replay_config(parent["config"], archives["anti_kd"].path, 101),
        "general": replay_config(
            parent["config"], archives["general_anchor"].path, 202
        ),
        "fros": replay_config(parent["config"], archives["fros_train"].path, 303),
    }
    cache_log = io.StringIO()
    with contextlib.redirect_stdout(cache_log):
        caches = {
            source: ppo.build_bc_replay_batches(config, parent["model_config"])
            for source, config in configs.items()
        }
    cache_records = {}
    batch_sha256 = {}
    for source in ("anti_kd", "general", "fros"):
        cache_records[source], batch_sha256[source] = _cache_record(
            source, caches[source]
        )

    dependency_paths = {
        "launcher": Path(__file__).resolve(),
        "training_core": Path(training_core.__file__).resolve(),
        "trainer_module": training_core.TRAINER,
        "contraction_core": Path(contraction_core.__file__).resolve(),
        "behavior_core": Path(behavior_core.__file__).resolve(),
        "fros_train_builder": Path(fros_train_builder.__file__).resolve(),
        "anti_kd_builder": Path(anti_kd_builder.__file__).resolve(),
        "fros_valid_builder": Path(fros_valid_builder.__file__).resolve(),
        "general_anchor_builder": Path(general_anchor_builder.__file__).resolve(),
        "python": EXPECTED_PYTHON.resolve(),
    }
    dependency_hashes = {
        name: file_sha256(path) for name, path in dependency_paths.items()
    }
    allowlists = {
        "anti_kd": anti_allowlist,
        "fros_train": fros_allowlist,
        "fros_valid": fros_valid_allowlist,
    }
    manifest = build_manifest_payload(
        parent=parent,
        archives=archives,
        allowlists=allowlists,
        split_audits=split_audits,
        cache_records=cache_records,
        actor_shapes=actor_shapes,
        dependency_hashes=dependency_hashes,
        device=args.device,
    )
    paths = {
        "s8_parent": S8_PARENT,
        "bc_architecture": BC_ARCHITECTURE,
        **{f"archive_{name}": record.path for name, record in archives.items()},
        **{f"allowlist_{name}": record.path for name, record in allowlists.items()},
        **dependency_paths,
    }
    return PreparedRun(
        manifest=manifest,
        manifest_sha256=sha256_json(manifest),
        parent=parent,
        bc_checkpoint=bc_checkpoint,
        configs=configs,
        caches=caches,
        batch_sha256=batch_sha256,
        paths=paths,
    )


def frozen_manifest_envelope(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION + "-frozen-manifest-envelope",
        "manifest_sha256": sha256_json(manifest),
        "manifest": manifest,
    }


def dry_run_bytes(manifest: dict[str, Any]) -> bytes:
    return canonical_line_bytes(frozen_manifest_envelope(manifest))


def load_frozen_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = fros_train_builder.audit_path_components(
        path,
        "frozen manifest",
        must_exist=True,
        kind="file",
    )
    validate_sha256(expected_sha256, "expected frozen manifest SHA-256")
    envelope = json.loads(path.read_bytes())
    if not isinstance(envelope, dict) or not isinstance(envelope.get("manifest"), dict):
        raise RuntimeError("Frozen manifest envelope is malformed")
    if envelope.get("schema_version") != SCHEMA_VERSION + "-frozen-manifest-envelope":
        raise RuntimeError("Frozen manifest envelope schema mismatch")
    manifest = envelope["manifest"]
    observed = sha256_json(manifest)
    if envelope.get("manifest_sha256") != observed:
        raise RuntimeError("Frozen manifest embedded digest is invalid")
    if observed != expected_sha256:
        raise RuntimeError("Frozen manifest differs from reviewed digest")
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
    return {"l2": total_l2, "per_tensor": per_tensor}


def build_contractions(
    parent_state: dict[str, torch.Tensor],
    trained_state: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    trained_integrity = training_core.validate_endpoint_state(
        parent_state, trained_state
    )
    full_movement = actor6_movement(parent_state, trained_state)
    if full_movement["l2"] > MAX_FULL_ACTOR6_L2:
        raise RuntimeError(
            f"Full actor6 correction exceeds {MAX_FULL_ACTOR6_L2}: "
            f"{full_movement['l2']}"
        )
    states = {}
    audits: dict[str, Any] = {
        "trained_direction": {
            "integrity": trained_integrity,
            "movement": full_movement,
            "l2_cap": MAX_FULL_ACTOR6_L2,
        }
    }
    for endpoint, alpha in CONTRACTIONS:
        state = contraction_core.interpolate_state_dict(
            parent_state, trained_state, alpha
        )
        integrity = training_core.validate_endpoint_state(parent_state, state)
        movement = actor6_movement(parent_state, state)
        states[endpoint] = state
        audits[endpoint] = {
            "alpha": alpha,
            "integrity": integrity,
            "movement": movement,
        }
    for name in parent_state:
        if name not in ACTOR6:
            if not torch.equal(states["E50"][name], parent_state[name]):
                raise RuntimeError(f"E50 changed non-actor6 tensor {name}")
            if not torch.equal(states["E100"][name], parent_state[name]):
                raise RuntimeError(f"E100 changed non-actor6 tensor {name}")
        if not torch.equal(states["E100"][name], trained_state[name]):
            raise RuntimeError(f"E100 differs from full correction at {name}")
    ratio = audits["E50"]["movement"]["l2"] / audits["E100"]["movement"]["l2"]
    if not 0.49 <= ratio <= 0.51:
        raise RuntimeError(f"E50 movement ratio is not one half: {ratio}")
    audits["E50"]["actor6_l2_ratio_to_E100"] = ratio
    return states, audits


class BoundMemberDataset(behavior_core.OrderedZipDecisionDataset):
    """Use the archive member prefix as split authority.

    Anti-KD dev rows are copied verbatim from source train rows; the immutable
    row-level ``split`` field therefore remains ``train``.  The bound archive
    member and allowlist-derived split, both hash audited by this launcher,
    are the authority for this specialized view.
    """

    def row_split(self, row: dict[str, Any]) -> str:
        return self.split


def behavior_loader(
    *,
    archive: Path,
    split: str,
    model_config: dict[str, Any],
    device: torch.device,
) -> DataLoader:
    dataset = BoundMemberDataset(
        archive_path=archive,
        split=split,
        split_mode="archive",
        split_seed=SEED,
        hash_size=int(model_config["hash_size"]),
        max_state_entities=int(model_config["max_state_entities"]),
        deck_hashes=(),
        team_names=(),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=partial(
            behavior_core.collate_ordered,
            max_state_entities=int(model_config["max_state_entities"]),
            entity_fields=int(model_config["entity_fields"]),
            option_fields=int(model_config["option_fields"]),
        ),
        pin_memory=device.type == "cuda",
    )


@torch.inference_mode()
def evaluate_behavior_view(
    *,
    model: torch.nn.Module,
    archive: Path,
    split: str,
    expected_rows: int,
    model_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    accumulator = behavior_core.MetricAccumulator()
    count_digest = hashlib.sha256()
    loss_totals: dict[str, float] = {}
    rows = 0
    model.eval()
    for cpu_batch in behavior_loader(
        archive=archive,
        split=split,
        model_config=model_config,
        device=device,
    ):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in cpu_batch.items()
        }
        outputs = ppo.model_forward(model, batch, device)
        policy_actions, _, _, _ = ppo.sample_ordered_actions(
            outputs,
            batch,
            deterministic=True,
            canonicalize_order=False,
        )
        accumulator.update(batch, outputs, policy_actions, policy_actions)
        count_predictions = torch.tensor(
            [len(action) for action in policy_actions], dtype=torch.int16
        )
        count_digest.update(count_predictions.numpy().tobytes())
        loss_batch = dict(batch)
        loss_batch["action_sequences"] = batch["expert_ordered_actions"]
        loss, parts = ppo.bc_expert_actor_loss(
            outputs,
            loss_batch,
            loss_mode="ordered",
            order_context_weight=ORDER_CONTEXT_WEIGHT,
            non_context34_fixed_multi_action_order_weight=(
                RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT
            ),
        )
        batch_rows = int(batch["action_counts"].shape[0])
        rows += batch_rows
        loss_totals["ordered_bc_loss"] = (
            loss_totals.get("ordered_bc_loss", 0.0)
            + float(loss.detach()) * batch_rows
        )
        for name, value in parts.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value) * batch_rows
    if rows != expected_rows:
        raise RuntimeError(f"Behavior view rows mismatch: {rows} != {expected_rows}")
    result = {
        "rows": rows,
        "metrics": accumulator.summary(),
        "losses": {name: value / rows for name, value in loss_totals.items()},
        "count_prediction_sha256": count_digest.hexdigest(),
    }
    if not training_core.audit.finite_nested(result):
        raise FloatingPointError("Behavior view produced non-finite metrics")
    return result


def behavior_gate(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    general_base = baseline["general"]
    general_candidate = candidate["general"]
    fros_base = baseline["fros_valid"]
    fros_candidate = candidate["fros_valid"]
    anti_base = baseline["anti_kd_dev"]
    anti_candidate = candidate["anti_kd_dev"]
    gates = {
        "general_set_drop_at_most_0_001": (
            general_candidate["metrics"]["set_exact_accuracy"]
            >= general_base["metrics"]["set_exact_accuracy"] - 0.001
        ),
        "general_hybrid_drop_at_most_0_001": (
            general_candidate["metrics"]["hybrid_order_exact_accuracy"]
            >= general_base["metrics"]["hybrid_order_exact_accuracy"] - 0.001
        ),
        "fros_hybrid_non_drop": (
            fros_candidate["metrics"]["hybrid_order_exact_accuracy"]
            >= fros_base["metrics"]["hybrid_order_exact_accuracy"]
        ),
        "fros_ordered_loss_non_increase": (
            fros_candidate["losses"]["ordered_bc_loss"]
            <= fros_base["losses"]["ordered_bc_loss"]
        ),
        "anti_kd_dev_ordered_loss_strict_decrease": (
            anti_candidate["losses"]["ordered_bc_loss"]
            < anti_base["losses"]["ordered_bc_loss"]
        ),
        "count_predictions_identical_all_views": all(
            candidate[view]["count_prediction_sha256"]
            == baseline[view]["count_prediction_sha256"]
            for view in ("general", "fros_valid", "anti_kd_dev")
        ),
        "count_correct_identical_all_views": all(
            candidate[view]["metrics"]["count_correct"]
            == baseline[view]["metrics"]["count_correct"]
            for view in ("general", "fros_valid", "anti_kd_dev")
        ),
    }
    return {
        "gates": gates,
        "pass": all(gates.values()),
        "deltas": {
            "general_set_accuracy": (
                general_candidate["metrics"]["set_exact_accuracy"]
                - general_base["metrics"]["set_exact_accuracy"]
            ),
            "general_hybrid_accuracy": (
                general_candidate["metrics"]["hybrid_order_exact_accuracy"]
                - general_base["metrics"]["hybrid_order_exact_accuracy"]
            ),
            "fros_hybrid_accuracy": (
                fros_candidate["metrics"]["hybrid_order_exact_accuracy"]
                - fros_base["metrics"]["hybrid_order_exact_accuracy"]
            ),
            "fros_ordered_bc_loss": (
                fros_candidate["losses"]["ordered_bc_loss"]
                - fros_base["losses"]["ordered_bc_loss"]
            ),
            "anti_kd_dev_ordered_bc_loss": (
                anti_candidate["losses"]["ordered_bc_loss"]
                - anti_base["losses"]["ordered_bc_loss"]
            ),
        },
    }


def evaluate_all_views(
    model: torch.nn.Module,
    prepared: PreparedRun,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    return {
        "general": evaluate_behavior_view(
            model=model,
            archive=prepared.paths["archive_general_behavior"],
            split="valid",
            expected_rows=GENERAL_BEHAVIOR_ROWS,
            model_config=prepared.parent["model_config"],
            device=device,
        ),
        "fros_valid": evaluate_behavior_view(
            model=model,
            archive=prepared.paths["archive_fros_valid"],
            split="valid",
            expected_rows=FROS_VALID_ROWS,
            model_config=prepared.parent["model_config"],
            device=device,
        ),
        "anti_kd_dev": evaluate_behavior_view(
            model=model,
            archive=prepared.paths["archive_anti_kd"],
            split="dev",
            expected_rows=prepared.manifest["inputs"]["split_audits"][
                "anti_kd_dev"
            ]["rows"],
            model_config=prepared.parent["model_config"],
            device=device,
        ),
    }


def endpoint_payload(
    *,
    parent: dict[str, Any],
    state: dict[str, torch.Tensor],
    manifest_sha256: str,
    endpoint: str,
    alpha: float,
    integrity: dict[str, Any],
    behavior: dict[str, Any],
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
            "s8_anti_kd_bc_repair": {
                "schema_version": SCHEMA_VERSION,
                "frozen_manifest_sha256": manifest_sha256,
                "parent": "S8",
                "parent_sha256": S8_PARENT_SHA256,
                "optimizer_steps": 8,
                "endpoint": endpoint,
                "contraction_alpha": alpha,
                "trainable_scope": "actor6",
                "integrity": integrity,
                "behavior_gate": behavior,
                "H2H_eligible": bool(behavior["pass"]),
                "H2H_not_performed": True,
                "promotion_eligible": False,
                "submission_authorized": False,
            },
        }
    )
    return retained


def _current_hashes(prepared: PreparedRun) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in prepared.paths.items()}


def _expected_hashes(prepared: PreparedRun) -> dict[str, str]:
    inputs = prepared.manifest["inputs"]
    expected = {
        "s8_parent": inputs["s8_parent"]["sha256"],
        "bc_architecture": inputs["bc_architecture"]["sha256"],
        **{
            f"archive_{name}": record["sha256"]
            for name, record in inputs["archives"].items()
        },
        **{
            f"allowlist_{name}": record["raw_sha256"]
            for name, record in inputs["allowlists"].items()
        },
        **{
            name: record["sha256"]
            for name, record in inputs["dependencies"].items()
        },
        "python": inputs["python"]["sha256"],
    }
    return expected


def execute(prepared: PreparedRun, device_name: str) -> int:
    fros_train_builder.audit_path_components(
        OUTPUT_DIR,
        "S8 anti-KD repair output",
        must_exist=False,
        kind="output",
    )
    training_core.assert_output_absent(OUTPUT_DIR)
    device = torch.device(device_name)
    if device.type != "cuda" or device.index not in (None, 0):
        raise RuntimeError("Formal S8 repair requires cuda or cuda:0")
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
    baseline_behavior = evaluate_all_views(model, prepared, device)
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
    states, integrity = build_contractions(parent_state, trained_state)

    endpoint_behavior = {}
    behavior_decisions = {}
    for endpoint, _ in CONTRACTIONS:
        model.load_state_dict(states[endpoint], strict=True)
        endpoint_behavior[endpoint] = evaluate_all_views(model, prepared, device)
        behavior_decisions[endpoint] = behavior_gate(
            baseline_behavior, endpoint_behavior[endpoint]
        )

    current_hashes = _current_hashes(prepared)
    expected_hashes = _expected_hashes(prepared)
    if current_hashes != expected_hashes:
        raise RuntimeError("A frozen input changed during S8 repair")

    checkpoint_bytes = {}
    endpoint_records = {}
    for endpoint, alpha in CONTRACTIONS:
        payload = endpoint_payload(
            parent=prepared.parent,
            state=states[endpoint],
            manifest_sha256=prepared.manifest_sha256,
            endpoint=endpoint,
            alpha=alpha,
            integrity=integrity[endpoint],
            behavior=behavior_decisions[endpoint],
        )
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        raw = buffer.getvalue()
        filename = f"s08-antikd-{endpoint}.pt"
        checkpoint_bytes[filename] = raw
        endpoint_records[endpoint] = {
            "filename": filename,
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
            "alpha": alpha,
            "integrity": integrity[endpoint],
            "behavior": endpoint_behavior[endpoint],
            "behavior_gate": behavior_decisions[endpoint],
            "H2H_eligible": bool(behavior_decisions[endpoint]["pass"]),
            "H2H_performed": False,
        }
    if set(checkpoint_bytes) != {"s08-antikd-E50.pt", "s08-antikd-E100.pt"}:
        raise RuntimeError("Endpoint publication set drifted from E50/E100")

    eligible = [
        endpoint
        for endpoint, record in endpoint_records.items()
        if record["H2H_eligible"]
    ]
    result = {
        "schema_version": SCHEMA_VERSION + "-result",
        "status": (
            "behavior_pass_H2H_required" if eligible else "behavior_rejected"
        ),
        "frozen_manifest_sha256": prepared.manifest_sha256,
        "optimizer_steps": 8,
        "training_trace": trace,
        "trained_direction_integrity": integrity["trained_direction"],
        "baseline_behavior": baseline_behavior,
        "endpoints": endpoint_records,
        "H2H_eligible_endpoints": eligible,
        "H2H_performed": False,
        "inputs_unchanged_after_training": True,
        "observed_input_hashes": current_hashes,
        "only_E50_E100_published": True,
        "package_performed": False,
        "upload_performed": False,
        "submission_performed": False,
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
    return 0 if eligible else 42


def enforce_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from repository root: {ROOT}")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"Wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )
    if Path(sys.prefix).resolve() != EXPECTED_ENV_PREFIX.resolve():
        raise RuntimeError("Wrong Python environment prefix")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("Run with my_project_env Python flags -I -B")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--freeze-manifest", type=Path, metavar="PATH")
    mode.add_argument("--execute", action="store_true")

    def bound_input(prefix: str, label: str, *, allowlist: bool = False) -> None:
        parser.add_argument(f"--{prefix}", type=Path, required=True, help=label)
        parser.add_argument(f"--expected-{prefix}-sha256", required=True)
        if allowlist:
            parser.add_argument(
                f"--expected-{prefix}-canonical-sha256", required=True
            )
        else:
            manifest_prefix = prefix.removesuffix("-archive")
            parser.add_argument(
                f"--expected-{manifest_prefix}-manifest-sha256", required=True
            )

    bound_input("anti-kd-archive", "exact anti-KD archive")
    bound_input("anti-kd-allowlist", "exact anti-KD allowlist", allowlist=True)
    bound_input("fros-archive", "exact Fros train archive")
    bound_input("fros-allowlist", "exact Fros train allowlist", allowlist=True)
    bound_input("fros-valid-archive", "Aug7 exact Fros behavior archive")
    bound_input(
        "fros-valid-allowlist", "Aug7 exact Fros behavior allowlist", allowlist=True
    )
    bound_input("general-anchor-archive", "general exclusion anchor archive")
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
    args = parse_args(argv)
    validate_mode_args(args)
    enforce_runtime()
    prepared = prepare_run(args)
    if args.dry_run:
        sys.stdout.buffer.write(dry_run_bytes(prepared.manifest))
        return 0
    if args.freeze_manifest is not None:
        path = fros_train_builder.audit_path_components(
            args.freeze_manifest,
            "frozen manifest output",
            must_exist=False,
            kind="output",
        )
        if path == OUTPUT_DIR:
            raise ValueError("Frozen manifest path must differ from output directory")
        if not path.parent.is_dir():
            raise FileNotFoundError(
                f"Frozen manifest parent must be an existing non-symlink: "
                f"{path.parent}"
            )
        training_core.write_exclusive_bytes(path, dry_run_bytes(prepared.manifest))
        print(prepared.manifest_sha256)
        return 0
    frozen = load_frozen_manifest(
        args.frozen_manifest, args.expected_manifest_sha256
    )
    if frozen != prepared.manifest or sha256_json(frozen) != prepared.manifest_sha256:
        raise RuntimeError("Current inputs/protocol differ from reviewed manifest")
    return execute(prepared, args.device)


if __name__ == "__main__":
    raise SystemExit(main())

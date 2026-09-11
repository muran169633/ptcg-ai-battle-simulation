#!/usr/bin/env python3
"""Build the Marnie general-anchor train archive after exact exclusions.

The builder consumes the frozen ``marnie_trainwins`` source plus reviewed
exact-Fros and exact-anti-KD allowlists.  It requires the two episode sets to
be disjoint, verifies their per-episode row counts against the source, and
copies every remaining source train row byte for byte.  The output bundle is
published as one atomic no-replace directory.

Default mode is a write-free dry run.  Execute requires reviewed hashes for
both allowlists and for the plan/archive/manifest payloads.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import orjson


def _load_local_module(module_name: str, filename: str) -> Any:
    """Load one audited sibling module without cwd/PYTHONPATH dependence."""

    script_dir = Path(os.path.abspath(os.fspath(Path(__file__).parent)))
    module_path = Path(os.path.abspath(os.fspath(script_dir / filename)))
    if module_path.parent != script_dir:
        raise RuntimeError(f"local module escapes tools directory: {module_path}")
    current = Path(module_path.anchor)
    for part in module_path.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise RuntimeError(f"local module path is missing: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"local module path traverses symlink: {current}")
        if current == module_path:
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"local module is not a regular file: {current}")
        elif not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"local module ancestor is not a directory: {current}")
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(os.path.abspath(existing_file)) != module_path:
            raise RuntimeError(f"unexpected preloaded module identity: {module_name}")
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create local module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


fros = _load_local_module(
    "build_marnie_froslass_exact_wins",
    "build_marnie_froslass_exact_wins.py",
)
anti = _load_local_module(
    "build_marnie_exact_anti_kd",
    "build_marnie_exact_anti_kd.py",
)


REPO_ROOT = Path(os.path.abspath(os.fspath(Path(__file__).parent))).parent
ARCHIVE_SCHEMA_VERSION = "ptcg-marnie-general-anchor-exclusion-archive-v1"
TRAIN_MEMBER = "train/part-00000.jsonl"
MANIFEST_MEMBER = "manifest.json"
DEFAULT_FROS_ALLOWLIST = (
    REPO_ROOT
    / "data/gold_push_marnie_fros_exact_20260810_v1/exact_episode_allowlist.json"
)
DEFAULT_ANTI_ALLOWLIST = anti.DEFAULT_ALLOWLIST
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data/gold_push_marnie_general_anchor_20260810_v1"
)
DEFAULT_ARCHIVE = DEFAULT_OUTPUT_ROOT / "marnie_general_anchor_train.zip"
DEFAULT_EXTERNAL_MANIFEST = DEFAULT_OUTPUT_ROOT / MANIFEST_MEMBER

# Exact uncompressed train-member identities of the frozen source archive.
# These values are redundant with the reviewed outer ZIP hash by design: they
# let downstream consumers prove that the derived archive carries the exact
# source-member lineage instead of accepting arbitrary well-formed digests.
SOURCE_TRAIN_MEMBER_AUDIT = (
    (
        "train/part-00000.jsonl",
        151_765_245,
        "04f79003fc6193e606e2c314c6f7107f33bcd27e8359f8b981d33cdaaf0d573d",
    ),
    (
        "train/part-00001.jsonl",
        150_689_905,
        "61de6533b8be4d05e2bee2ecd929579c443137a79142bf652bf7190a54f0ff41",
    ),
    (
        "train/part-00002.jsonl",
        150_689_039,
        "93eefa74d18d5dbe98e741013dd58dcb9e323a9626ef99f676c0e27d183294ec",
    ),
    (
        "train/part-00003.jsonl",
        149_424_256,
        "d481d3e4863236d28e8cb5256180217b9c07d0e2e950b5d5b17217dbca6f37d3",
    ),
    (
        "train/part-00004.jsonl",
        151_377_452,
        "78b9fd5c7709370846f5eecfda968a9462854a3c64f2487e30462fb7cfb47d78",
    ),
    (
        "train/part-00005.jsonl",
        150_673_569,
        "e07e056c527fb7acc3293719214fded7c68cf8a9e06aa88ac8ab06a5440e338c",
    ),
    (
        "train/part-00006.jsonl",
        151_519_286,
        "9e18b33f6dfc11ea025d77bea0c73699a32f03ea78b2a4a4ddc599968f9f24b7",
    ),
    (
        "train/part-00007.jsonl",
        75_830_765,
        "f147a6958a5b89cb1096ed5c2ada53597c56441efcda78fd94542f3e1c77cf23",
    ),
)


@dataclass(frozen=True)
class GeneralContract:
    source: fros.BuildContract
    fros_allowlist_contract: fros.BuildContract
    anti_allowlist_contract: anti.AntiKDContract
    expected_excluded_episodes: int
    expected_excluded_rows: int
    expected_output_episodes: int
    expected_output_rows: int
    source_train_member_audit: tuple[tuple[str, int, str], ...]


DEFAULT_CONTRACT = GeneralContract(
    source=fros.DEFAULT_CONTRACT,
    fros_allowlist_contract=fros.DEFAULT_CONTRACT,
    anti_allowlist_contract=anti.DEFAULT_CONTRACT,
    expected_excluded_episodes=147,
    expected_excluded_rows=15_487,
    expected_output_episodes=1_762,
    expected_output_rows=172_124,
    source_train_member_audit=SOURCE_TRAIN_MEMBER_AUDIT,
)


@dataclass(frozen=True)
class ExclusionInput:
    label: str
    path: Path
    raw_sha256: str
    canonical_sha256: str
    schema_version: str
    episodes: dict[tuple[str, str], int]
    date_episode_counts: dict[str, int]
    date_decision_rows: dict[str, int]


@dataclass(frozen=True)
class GeneralAudit:
    source: fros.SourceAudit
    fros_exclusion: ExclusionInput
    anti_exclusion: ExclusionInput
    output_episodes: tuple[fros.SelectedEpisode, ...]
    output_lines: tuple[bytes, ...]
    output_content_sha256: str
    output_decision_keys_digest: str
    source_date_episode_counts: dict[str, int]
    source_date_decision_rows: dict[str, int]
    output_date_episode_counts: dict[str, int]
    output_date_decision_rows: dict[str, int]


def validate_contract(contract: GeneralContract) -> None:
    source = contract.source
    source_lineage = (
        source.source_sha256,
        source.source_manifest_sha256,
        source.expected_source_train_episodes,
        source.expected_source_train_rows,
        source.source_members,
        source.source_train_members,
        source.source_logical_path,
        source.learner_deck_hash,
    )
    for label, lineage_contract in (
        ("Fros", contract.fros_allowlist_contract),
        ("anti-KD", contract.anti_allowlist_contract.base),
    ):
        lineage = (
            lineage_contract.source_sha256,
            lineage_contract.source_manifest_sha256,
            lineage_contract.expected_source_train_episodes,
            lineage_contract.expected_source_train_rows,
            lineage_contract.source_members,
            lineage_contract.source_train_members,
            lineage_contract.source_logical_path,
            lineage_contract.learner_deck_hash,
        )
        if lineage != source_lineage:
            raise RuntimeError(f"general source and {label} source lineage differ")
    if contract.expected_excluded_episodes != (
        contract.fros_allowlist_contract.expected_selected_episodes
        + contract.anti_allowlist_contract.expected_episodes
    ):
        raise RuntimeError("excluded episode algebra drift")
    if contract.expected_excluded_rows != (
        contract.fros_allowlist_contract.expected_selected_rows
        + contract.anti_allowlist_contract.expected_rows
    ):
        raise RuntimeError("excluded row algebra drift")
    if source.expected_source_train_episodes - contract.expected_excluded_episodes != (
        contract.expected_output_episodes
    ):
        raise RuntimeError("output episode subtraction drift")
    if source.expected_source_train_rows - contract.expected_excluded_rows != (
        contract.expected_output_rows
    ):
        raise RuntimeError("output row subtraction drift")
    expected_names = tuple(value[0] for value in contract.source_train_member_audit)
    if expected_names != source.source_train_members:
        raise RuntimeError("source train-member audit order/name drift")
    for index, (_name, byte_count, digest) in enumerate(
        contract.source_train_member_audit
    ):
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise RuntimeError(f"source train-member audit size drift at {index}")
        fros.require_sha256(digest, f"source_train_member_audit[{index}].sha256")


def _load_allowlist(
    *,
    label: str,
    path: Path,
    expected_sha256: str,
    validator: Any,
    validator_contract: Any,
) -> ExclusionInput:
    fros.require_sha256(expected_sha256, f"expected_{label}_allowlist_sha256")
    raw = path.read_bytes()
    raw_sha = fros.sha256_bytes(raw)
    if raw_sha != expected_sha256:
        raise RuntimeError(
            f"{label} allowlist SHA mismatch: expected {expected_sha256}, got {raw_sha}"
        )
    document = validator(raw, validator_contract)
    canonical = fros.canonical_json_bytes(document) + b"\n"
    if canonical != raw:
        raise RuntimeError(f"{label} allowlist is not canonical JSON+LF")
    episodes: dict[tuple[str, str], int] = {}
    date_episodes: Counter[str] = Counter()
    date_rows: Counter[str] = Counter()
    for value in document["episodes"]:
        date = str(value["date"])
        episode_id = str(value["episode_id"])
        key = (date, episode_id)
        if key in episodes:
            raise RuntimeError(f"{label} allowlist repeats episode key {key!r}")
        rows = int(value["decision_rows"])
        if rows <= 0:
            raise RuntimeError(f"{label} allowlist has non-positive episode rows")
        episodes[key] = rows
        date_episodes[date] += 1
        date_rows[date] += rows
    return ExclusionInput(
        label=label,
        path=path,
        raw_sha256=raw_sha,
        canonical_sha256=fros.sha256_bytes(canonical),
        schema_version=str(document["schema_version"]),
        episodes=episodes,
        date_episode_counts=dict(sorted(date_episodes.items())),
        date_decision_rows=dict(sorted(date_rows.items())),
    )


def _selected_episode_from_source(
    value: fros.EpisodeSourceAudit,
) -> fros.SelectedEpisode:
    return fros.SelectedEpisode(
        dataset_date=value.dataset_date,
        episode_id=value.episode_id,
        replay_member="not_used_for_general_anchor",
        replay_sha256="0" * 64,
        seat=value.seat,
        team_name=value.team_name,
        opponent_team_name=value.opponent_team_name,
        replay_rewards=(value.terminal_reward, -value.terminal_reward),
        rows=value.rows,
        raw_rows_sha256=value.raw_rows_sha256.hexdigest(),
        decision_keys_digest=fros.digest_lines(value.decision_keys),
    )


def audit_inputs(
    source_path: Path,
    fros_allowlist_path: Path,
    anti_allowlist_path: Path,
    expected_fros_allowlist_sha256: str,
    expected_anti_allowlist_sha256: str,
    contract: GeneralContract,
) -> GeneralAudit:
    validate_contract(contract)
    source = fros.scan_source(source_path, contract.source)
    observed_source_members = tuple(
        (
            name,
            dict(source.train_member_bytes)[name],
            digest,
        )
        for name, digest in source.train_member_sha256
    )
    if observed_source_members != contract.source_train_member_audit:
        raise RuntimeError("general source train-member exact identity drift")
    fros_exclusion = _load_allowlist(
        label="fros",
        path=fros_allowlist_path,
        expected_sha256=expected_fros_allowlist_sha256,
        validator=fros.validate_allowlist_document,
        validator_contract=contract.fros_allowlist_contract,
    )
    anti_exclusion = _load_allowlist(
        label="anti_kd",
        path=anti_allowlist_path,
        expected_sha256=expected_anti_allowlist_sha256,
        validator=anti.validate_allowlist_document,
        validator_contract=contract.anti_allowlist_contract,
    )
    fros_keys = set(fros_exclusion.episodes)
    anti_keys = set(anti_exclusion.episodes)
    overlap = fros_keys & anti_keys
    if overlap:
        raise RuntimeError(f"Fros and anti-KD exclusion sets overlap: {sorted(overlap)!r}")
    excluded = fros_keys | anti_keys
    if len(excluded) != contract.expected_excluded_episodes:
        raise RuntimeError("general exclusion episode union count drift")
    if sum(fros_exclusion.episodes.values()) + sum(
        anti_exclusion.episodes.values()
    ) != contract.expected_excluded_rows:
        raise RuntimeError("general exclusion row union count drift")
    missing = excluded - set(source.episodes)
    if missing:
        raise RuntimeError(f"general exclusions missing from source: {sorted(missing)!r}")
    for key, expected_rows in {
        **fros_exclusion.episodes,
        **anti_exclusion.episodes,
    }.items():
        if source.episodes[key].rows != expected_rows:
            raise RuntimeError(
                f"general exclusion source row mismatch for {key!r}: "
                f"source={source.episodes[key].rows}, allowlist={expected_rows}"
            )
    output_values = [
        value for key, value in source.episodes.items() if key not in excluded
    ]
    output_values.sort(key=lambda value: (value.dataset_date, int(value.episode_id)))
    if len(output_values) != contract.expected_output_episodes or sum(
        value.rows for value in output_values
    ) != contract.expected_output_rows:
        raise RuntimeError(
            "general output count drift: "
            f"episodes={len(output_values)}, rows={sum(value.rows for value in output_values)}"
        )
    selected = tuple(_selected_episode_from_source(value) for value in output_values)
    materialize_contract = dataclasses.replace(
        contract.source,
        expected_selected_episodes=contract.expected_output_episodes,
        expected_selected_rows=contract.expected_output_rows,
    )
    lines, content_sha, keys_digest, output_date_rows_tuple = (
        fros.materialize_selected_lines(
            source_path,
            source,
            selected,
            materialize_contract,
        )
    )
    source_date_episodes = Counter(value.dataset_date for value in source.episodes.values())
    source_date_rows: Counter[str] = Counter()
    output_date_episodes = Counter(value.dataset_date for value in output_values)
    for value in source.episodes.values():
        source_date_rows[value.dataset_date] += value.rows
    output_date_rows = dict(output_date_rows_tuple)
    for date in contract.source.dates:
        expected_episode_subtraction = (
            source_date_episodes[date]
            - fros_exclusion.date_episode_counts.get(date, 0)
            - anti_exclusion.date_episode_counts.get(date, 0)
        )
        expected_row_subtraction = (
            source_date_rows[date]
            - fros_exclusion.date_decision_rows.get(date, 0)
            - anti_exclusion.date_decision_rows.get(date, 0)
        )
        if output_date_episodes[date] != expected_episode_subtraction:
            raise RuntimeError(f"general date episode subtraction drift for {date}")
        if output_date_rows[date] != expected_row_subtraction:
            raise RuntimeError(f"general date row subtraction drift for {date}")
    return GeneralAudit(
        source=source,
        fros_exclusion=fros_exclusion,
        anti_exclusion=anti_exclusion,
        output_episodes=selected,
        output_lines=lines,
        output_content_sha256=content_sha,
        output_decision_keys_digest=keys_digest,
        source_date_episode_counts={
            date: source_date_episodes[date] for date in contract.source.dates
        },
        source_date_decision_rows={
            date: source_date_rows[date] for date in contract.source.dates
        },
        output_date_episode_counts={
            date: output_date_episodes[date] for date in contract.source.dates
        },
        output_date_decision_rows={
            date: output_date_rows[date] for date in contract.source.dates
        },
    )


def plan_document(
    contract: GeneralContract,
    expected_fros_allowlist_sha256: str,
    expected_anti_allowlist_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "data_schema_version": fros.ROW_SCHEMA_VERSION,
        "source": {
            "logical_path": contract.source.source_logical_path,
            "sha256": contract.source.source_sha256,
            "manifest_sha256": contract.source.source_manifest_sha256,
            "episodes": contract.source.expected_source_train_episodes,
            "rows": contract.source.expected_source_train_rows,
            "train_members": [
                {"member": name, "bytes": byte_count, "sha256": digest}
                for name, byte_count, digest in contract.source_train_member_audit
            ],
        },
        "exclusions": {
            "fros": {
                "schema_version": fros.ALLOWLIST_SCHEMA_VERSION,
                "raw_sha256": expected_fros_allowlist_sha256,
                "canonical_sha256": expected_fros_allowlist_sha256,
                "episodes": contract.fros_allowlist_contract.expected_selected_episodes,
                "rows": contract.fros_allowlist_contract.expected_selected_rows,
            },
            "anti_kd": {
                "schema_version": anti.ALLOWLIST_SCHEMA_VERSION,
                "raw_sha256": expected_anti_allowlist_sha256,
                "canonical_sha256": expected_anti_allowlist_sha256,
                "episodes": contract.anti_allowlist_contract.expected_episodes,
                "rows": contract.anti_allowlist_contract.expected_rows,
            },
            "sets_must_be_disjoint": True,
        },
        "output": {
            "episodes": contract.expected_output_episodes,
            "rows": contract.expected_output_rows,
            "member": TRAIN_MEMBER,
            "raw_source_rows": True,
        },
    }


def manifest_document(
    audit: GeneralAudit,
    contract: GeneralContract,
    plan_sha256: str,
) -> dict[str, Any]:
    date_lineage = {}
    for date in contract.source.dates:
        date_lineage[date] = {
            "source_episodes": audit.source_date_episode_counts[date],
            "source_rows": audit.source_date_decision_rows[date],
            "fros_excluded_episodes": audit.fros_exclusion.date_episode_counts.get(date, 0),
            "fros_excluded_rows": audit.fros_exclusion.date_decision_rows.get(date, 0),
            "anti_kd_excluded_episodes": audit.anti_exclusion.date_episode_counts.get(date, 0),
            "anti_kd_excluded_rows": audit.anti_exclusion.date_decision_rows.get(date, 0),
            "output_episodes": audit.output_date_episode_counts[date],
            "output_rows": audit.output_date_decision_rows[date],
        }
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "data_schema_version": fros.ROW_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "learner_deck_hash": contract.source.learner_deck_hash,
        "source_split": "train",
        "episode_count": len(audit.output_episodes),
        "decision_rows": len(audit.output_lines),
        "split_episodes": {"train": len(audit.output_episodes)},
        "split_decisions": {"train": len(audit.output_lines)},
        "source": {
            "logical_path": contract.source.source_logical_path,
            "sha256": contract.source.source_sha256,
            "manifest_sha256": contract.source.source_manifest_sha256,
            "episodes": len(audit.source.episodes),
            "rows": audit.source.train_rows,
            "train_members": [
                {
                    "member": name,
                    "bytes": dict(audit.source.train_member_bytes)[name],
                    "sha256": digest,
                }
                for name, digest in audit.source.train_member_sha256
            ],
        },
        "exclusion_sources": {
            "fros": {
                "path": str(audit.fros_exclusion.path),
                "schema_version": audit.fros_exclusion.schema_version,
                "raw_sha256": audit.fros_exclusion.raw_sha256,
                "canonical_sha256": audit.fros_exclusion.canonical_sha256,
                "source_sha256": contract.fros_allowlist_contract.source_sha256,
                "source_manifest_sha256": (
                    contract.fros_allowlist_contract.source_manifest_sha256
                ),
                "episodes": len(audit.fros_exclusion.episodes),
                "rows": sum(audit.fros_exclusion.episodes.values()),
            },
            "anti_kd": {
                "path": str(audit.anti_exclusion.path),
                "schema_version": audit.anti_exclusion.schema_version,
                "raw_sha256": audit.anti_exclusion.raw_sha256,
                "canonical_sha256": audit.anti_exclusion.canonical_sha256,
                "source_sha256": contract.anti_allowlist_contract.base.source_sha256,
                "source_manifest_sha256": (
                    contract.anti_allowlist_contract.base.source_manifest_sha256
                ),
                "episodes": len(audit.anti_exclusion.episodes),
                "rows": sum(audit.anti_exclusion.episodes.values()),
            },
        },
        "exclusion_union": {
            "sets_disjoint": True,
            "overlap_episodes": 0,
            "episodes": contract.expected_excluded_episodes,
            "rows": contract.expected_excluded_rows,
        },
        "date_lineage": date_lineage,
        "members": [
            {
                "member": TRAIN_MEMBER,
                "episodes": len(audit.output_episodes),
                "rows": len(audit.output_lines),
                "bytes": sum(len(value) for value in audit.output_lines),
                "sha256": audit.output_content_sha256,
            }
        ],
        "byte_preservation": {
            "remaining_source_train_rows_copied_verbatim": True,
            "selected_content_sha256": audit.output_content_sha256,
            "decision_keys_digest": audit.output_decision_keys_digest,
            "visualize_copied": False,
        },
        "zip": {
            "timestamp": list(fros.ZIP_TIMESTAMP),
            "compression": "deflate",
            "compresslevel": fros.ZIP_COMPRESSLEVEL,
            "member_order": [TRAIN_MEMBER, MANIFEST_MEMBER],
            "double_serialization_required": True,
        },
    }


def validate_manifest_document(
    payload: Any,
    contract: GeneralContract = DEFAULT_CONTRACT,
    *,
    expected_fros_allowlist_sha256: str | None = None,
    expected_anti_allowlist_sha256: str | None = None,
) -> dict[str, Any]:
    """Pure shared consumer validator for the general-anchor manifest."""

    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        document = orjson.loads(raw)
        if fros.canonical_json_bytes(document) + b"\n" != raw:
            raise RuntimeError("general manifest is not canonical JSON+LF")
    else:
        document = payload
    if not isinstance(document, dict):
        raise RuntimeError("general manifest is not an object")
    checks = {
        "schema_version": (document.get("schema_version"), ARCHIVE_SCHEMA_VERSION),
        "data_schema_version": (
            document.get("data_schema_version"),
            fros.ROW_SCHEMA_VERSION,
        ),
        "learner_deck_hash": (
            document.get("learner_deck_hash"),
            contract.source.learner_deck_hash,
        ),
        "source_split": (document.get("source_split"), "train"),
        "episode_count": (document.get("episode_count"), contract.expected_output_episodes),
        "decision_rows": (document.get("decision_rows"), contract.expected_output_rows),
        "split_episodes": (
            document.get("split_episodes"),
            {"train": contract.expected_output_episodes},
        ),
        "split_decisions": (
            document.get("split_decisions"),
            {"train": contract.expected_output_rows},
        ),
    }
    failures = [
        f"{name}={actual!r} expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    source = document.get("source")
    if not isinstance(source, Mapping):
        failures.append("source missing")
    else:
        source_checks = {
            "logical_path": contract.source.source_logical_path,
            "sha256": contract.source.source_sha256,
            "manifest_sha256": contract.source.source_manifest_sha256,
            "episodes": contract.source.expected_source_train_episodes,
            "rows": contract.source.expected_source_train_rows,
        }
        for name, expected in source_checks.items():
            if source.get(name) != expected:
                failures.append(f"source.{name} drift")
        train_members = source.get("train_members")
        if not isinstance(train_members, list):
            failures.append("source.train_members missing")
        else:
            try:
                observed_members = tuple(
                    (
                        str(value["member"]),
                        fros.require_integer(
                            value["bytes"],
                            f"source.train_members[{index}].bytes",
                        ),
                        str(value["sha256"]),
                    )
                    for index, value in enumerate(train_members)
                )
            except (KeyError, TypeError, ValueError, RuntimeError):
                failures.append("source.train_members malformed")
            else:
                if observed_members != contract.source_train_member_audit:
                    failures.append("source.train_members exact name/size/SHA drift")
    union = document.get("exclusion_union")
    expected_union = {
        "sets_disjoint": True,
        "overlap_episodes": 0,
        "episodes": contract.expected_excluded_episodes,
        "rows": contract.expected_excluded_rows,
    }
    if union != expected_union:
        failures.append("exclusion_union drift")
    exclusions = document.get("exclusion_sources")
    if not isinstance(exclusions, Mapping):
        failures.append("exclusion_sources missing")
    else:
        expected_hashes = {
            "fros": expected_fros_allowlist_sha256,
            "anti_kd": expected_anti_allowlist_sha256,
        }
        for label, expected_hash in expected_hashes.items():
            item = exclusions.get(label)
            if not isinstance(item, Mapping):
                failures.append(f"{label} exclusion missing")
                continue
            if item.get("raw_sha256") != item.get("canonical_sha256"):
                failures.append(f"{label} raw/canonical SHA differ")
            if expected_hash is not None and item.get("raw_sha256") != expected_hash:
                failures.append(f"{label} expected SHA mismatch")
            expected_contract = (
                contract.fros_allowlist_contract
                if label == "fros"
                else contract.anti_allowlist_contract.base
            )
            expected_schema = (
                fros.ALLOWLIST_SCHEMA_VERSION
                if label == "fros"
                else anti.ALLOWLIST_SCHEMA_VERSION
            )
            expected_episodes = (
                contract.fros_allowlist_contract.expected_selected_episodes
                if label == "fros"
                else contract.anti_allowlist_contract.expected_episodes
            )
            expected_rows = (
                contract.fros_allowlist_contract.expected_selected_rows
                if label == "fros"
                else contract.anti_allowlist_contract.expected_rows
            )
            item_checks = {
                "schema_version": expected_schema,
                "source_sha256": expected_contract.source_sha256,
                "source_manifest_sha256": expected_contract.source_manifest_sha256,
                "episodes": expected_episodes,
                "rows": expected_rows,
            }
            for name, expected in item_checks.items():
                if item.get(name) != expected:
                    failures.append(f"{label}.{name} drift")
    members = document.get("members")
    if not isinstance(members, list) or len(members) != 1:
        failures.append("output members drift")
    else:
        member = members[0]
        expected_member = {
            "member": TRAIN_MEMBER,
            "episodes": contract.expected_output_episodes,
            "rows": contract.expected_output_rows,
        }
        for name, expected in expected_member.items():
            if member.get(name) != expected:
                failures.append(f"members[0].{name} drift")
        digest = member.get("sha256")
        if not isinstance(digest, str) or not fros.SHA256_RE.fullmatch(digest):
            failures.append("members[0].sha256 drift")
        if not isinstance(member.get("bytes"), int) or member["bytes"] <= 0:
            failures.append("members[0].bytes drift")
    byte_preservation = document.get("byte_preservation")
    if not isinstance(byte_preservation, Mapping):
        failures.append("byte_preservation missing")
    else:
        if byte_preservation.get("remaining_source_train_rows_copied_verbatim") is not True:
            failures.append("byte preservation flag drift")
        if byte_preservation.get("visualize_copied") is not False:
            failures.append("visualize copy flag drift")
        if isinstance(members, list) and len(members) == 1 and (
            byte_preservation.get("selected_content_sha256")
            != members[0].get("sha256")
        ):
            failures.append("member/byte-preservation digest drift")
        decision_digest = byte_preservation.get("decision_keys_digest")
        if not isinstance(decision_digest, str) or not fros.SHA256_RE.fullmatch(
            decision_digest
        ):
            failures.append("decision_keys_digest drift")
    if failures:
        raise RuntimeError("general manifest contract drift: " + "; ".join(failures))
    lineage = document.get("date_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != set(contract.source.dates):
        raise RuntimeError("general date lineage coverage drift")
    for date, row in lineage.items():
        if row["source_episodes"] - row["fros_excluded_episodes"] - row[
            "anti_kd_excluded_episodes"
        ] != row["output_episodes"]:
            raise RuntimeError(f"general date episode algebra drift for {date}")
        if row["source_rows"] - row["fros_excluded_rows"] - row[
            "anti_kd_excluded_rows"
        ] != row["output_rows"]:
            raise RuntimeError(f"general date row algebra drift for {date}")
    return document


def create_archive(audit: GeneralAudit, manifest: dict[str, Any]) -> bytes:
    train_payload = b"".join(audit.output_lines)
    manifest_payload = fros.canonical_json_bytes(manifest) + b"\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=fros.ZIP_COMPRESSION,
        compresslevel=fros.ZIP_COMPRESSLEVEL,
        allowZip64=True,
    ) as archive:
        archive.writestr(fros.zip_info(TRAIN_MEMBER), train_payload)
        archive.writestr(fros.zip_info(MANIFEST_MEMBER), manifest_payload)
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != [TRAIN_MEMBER, MANIFEST_MEMBER]:
            raise RuntimeError("general archive member order drift")
        if archive.testzip() is not None:
            raise RuntimeError("general archive CRC failure")
        if archive.read(TRAIN_MEMBER) != train_payload:
            raise RuntimeError("general archive train bytes changed")
        if archive.read(MANIFEST_MEMBER) != manifest_payload:
            raise RuntimeError("general archive manifest bytes changed")
    return payload


def audit_and_serialize(
    source: Path,
    fros_allowlist: Path,
    anti_allowlist: Path,
    expected_fros_allowlist_sha256: str,
    expected_anti_allowlist_sha256: str,
    contract: GeneralContract,
) -> tuple[GeneralAudit, bytes, bytes, dict[str, str]]:
    audit = audit_inputs(
        source,
        fros_allowlist,
        anti_allowlist,
        expected_fros_allowlist_sha256,
        expected_anti_allowlist_sha256,
        contract,
    )
    plan = plan_document(
        contract,
        expected_fros_allowlist_sha256,
        expected_anti_allowlist_sha256,
    )
    plan_sha = fros.sha256_bytes(fros.canonical_json_bytes(plan))
    manifest = manifest_document(audit, contract, plan_sha)
    manifest_payload = fros.canonical_json_bytes(manifest) + b"\n"
    validate_manifest_document(
        manifest_payload,
        contract,
        expected_fros_allowlist_sha256=expected_fros_allowlist_sha256,
        expected_anti_allowlist_sha256=expected_anti_allowlist_sha256,
    )
    first = create_archive(audit, manifest)
    second = create_archive(audit, manifest)
    if first != second:
        raise RuntimeError("general archive double serialization differs")
    return audit, first, manifest_payload, {
        "plan_sha256": plan_sha,
        "archive_sha256": fros.sha256_bytes(first),
        "manifest_sha256": fros.sha256_bytes(manifest_payload),
    }


def build(
    *,
    source: Path,
    fros_allowlist: Path,
    anti_allowlist: Path,
    archive_path: Path,
    external_manifest_path: Path,
    expected_fros_allowlist_sha256: str,
    expected_anti_allowlist_sha256: str,
    contract: GeneralContract = DEFAULT_CONTRACT,
    execute: bool,
    expected_plan_sha256: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    source = fros.audit_path_components(
        source, "general source", must_exist=True, kind="file"
    )
    fros_allowlist = fros.audit_path_components(
        fros_allowlist, "Fros exclusion allowlist", must_exist=True, kind="file"
    )
    anti_allowlist = fros.audit_path_components(
        anti_allowlist, "anti-KD exclusion allowlist", must_exist=True, kind="file"
    )
    archive_path = fros.audit_path_components(
        archive_path, "general archive output", must_exist=False, kind="output"
    )
    external_manifest_path = fros.audit_path_components(
        external_manifest_path,
        "general external manifest output",
        must_exist=False,
        kind="output",
    )
    if archive_path.parent != external_manifest_path.parent:
        raise RuntimeError("general bundle outputs must share a directory")
    if os.path.lexists(archive_path.parent):
        raise FileExistsError(
            f"general bundle directory must be absent: {archive_path.parent}"
        )
    if execute and not all(
        (expected_plan_sha256, expected_archive_sha256, expected_manifest_sha256)
    ):
        raise RuntimeError("execute requires reviewed plan/archive/manifest SHA gates")
    for value, label in (
        (expected_fros_allowlist_sha256, "expected_fros_allowlist_sha256"),
        (expected_anti_allowlist_sha256, "expected_anti_allowlist_sha256"),
    ):
        fros.require_sha256(value, label)
    if execute:
        for value, label in (
            (str(expected_plan_sha256), "expected_plan_sha256"),
            (str(expected_archive_sha256), "expected_archive_sha256"),
            (str(expected_manifest_sha256), "expected_manifest_sha256"),
        ):
            fros.require_sha256(value, label)
    inputs = (source, fros_allowlist, anti_allowlist)
    before = {str(path): fros.sha256_file(path) for path in inputs}
    audit, archive_payload, manifest_payload, metadata = audit_and_serialize(
        source,
        fros_allowlist,
        anti_allowlist,
        expected_fros_allowlist_sha256,
        expected_anti_allowlist_sha256,
        contract,
    )
    after = {str(path): fros.sha256_file(path) for path in inputs}
    if before != after:
        raise RuntimeError("general builder inputs changed during audit")
    if execute:
        expected = {
            "plan_sha256": expected_plan_sha256,
            "archive_sha256": expected_archive_sha256,
            "manifest_sha256": expected_manifest_sha256,
        }
        if any(metadata[name] != value for name, value in expected.items()):
            raise RuntimeError("reviewed general SHA gate mismatch")
        anti.publish_bundle_no_clobber(
            archive_path,
            external_manifest_path,
            archive_payload,
            manifest_payload,
        )
        if fros.sha256_file(archive_path) != metadata["archive_sha256"]:
            raise RuntimeError("published general archive SHA mismatch")
        if fros.sha256_file(external_manifest_path) != metadata["manifest_sha256"]:
            raise RuntimeError("published general manifest SHA mismatch")
        with zipfile.ZipFile(archive_path) as published:
            if published.read(MANIFEST_MEMBER) != external_manifest_path.read_bytes():
                raise RuntimeError("published general embedded/external manifests differ")
    return {
        "status": "built" if execute else "dry_run_passed",
        "dry_run": not execute,
        "output_written": execute,
        "archive": str(archive_path),
        "external_manifest": str(external_manifest_path),
        "outer_zip_sha256": metadata["archive_sha256"],
        "archive_sha256": metadata["archive_sha256"],
        "embedded_manifest_raw_sha256": metadata["manifest_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "plan_sha256": metadata["plan_sha256"],
        "source_episodes": len(audit.source.episodes),
        "source_rows": audit.source.train_rows,
        "excluded_episodes": contract.expected_excluded_episodes,
        "excluded_rows": contract.expected_excluded_rows,
        "output_episodes": len(audit.output_episodes),
        "output_rows": len(audit.output_lines),
        "output_date_episode_counts": audit.output_date_episode_counts,
        "output_date_decision_rows": audit.output_date_decision_rows,
        "exclusion_sets_disjoint": True,
        "deterministic_rebuild_match": True,
        "rows_copied_verbatim": True,
        "training_started": False,
        "evaluation_started": False,
        "submission_started": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=fros.DEFAULT_SOURCE)
    parser.add_argument("--fros-allowlist", type=Path, default=DEFAULT_FROS_ALLOWLIST)
    parser.add_argument("--anti-allowlist", type=Path, default=DEFAULT_ANTI_ALLOWLIST)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--external-manifest", type=Path, default=DEFAULT_EXTERNAL_MANIFEST
    )
    parser.add_argument("--expected-fros-allowlist-sha256", required=True)
    parser.add_argument("--expected-anti-allowlist-sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--expected-manifest-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(
        source=args.source,
        fros_allowlist=args.fros_allowlist,
        anti_allowlist=args.anti_allowlist,
        archive_path=args.archive,
        external_manifest_path=args.external_manifest,
        expected_fros_allowlist_sha256=args.expected_fros_allowlist_sha256,
        expected_anti_allowlist_sha256=args.expected_anti_allowlist_sha256,
        execute=args.execute,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

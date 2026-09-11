#!/usr/bin/env python3
"""Deterministically rewrite train-only BC sample weights.

The tool is intentionally narrow: it preserves every valid/test member byte for
byte and changes only ``sample_weight`` in train rows.  The default rule boosts
one recent, single-action option-type bucket and one flexible-cardinality
context bucket while keeping every weight in the existing ``(0, 1]`` audit
range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import orjson


MEMBER_RE = re.compile(r"^(train|valid|test)/part-(\d{5})\.jsonl$")
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = stat.S_IFREG << 16 | 0o644 << 16
    return info


def strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must not be boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be an integer, got {value!r}") from error
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"{label} must be an integer, got {value!r}")
    return parsed


def row_context(row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    observation = row.get("observation")
    if not isinstance(observation, dict):
        raise TypeError("observation must be an object")
    select = observation.get("select")
    if not isinstance(select, dict):
        raise TypeError("observation.select must be an object")
    context = strict_int(
        select.get("context", row.get("select_context")),
        "select.context",
    )
    return context, select


def validated_action(row: dict[str, Any], select: dict[str, Any]) -> list[int]:
    options = select.get("option")
    action = row.get("action")
    if not isinstance(options, list) or not options:
        raise TypeError("observation.select.option must be a non-empty list")
    if not isinstance(action, list):
        raise TypeError("action must be a list")
    indices = [strict_int(value, "action index") for value in action]
    if len(indices) != len(set(indices)):
        raise ValueError("action contains duplicate indices")
    if any(index < 0 or index >= len(options) for index in indices):
        raise ValueError("action index is outside the option array")
    return indices


def decision_key(row: dict[str, Any]) -> str:
    fields = (
        str(row.get("dataset_date") or ""),
        str(row.get("episode_id") or ""),
        str(row.get("seat") if row.get("seat") is not None else ""),
        str(
            row.get("action_step_index")
            if row.get("action_step_index") is not None
            else ""
        ),
    )
    if any(value == "" for value in fields):
        raise ValueError(f"incomplete decision key: {fields!r}")
    return "|".join(fields)


def canonical_without_weight(row: dict[str, Any]) -> bytes:
    stable = dict(row)
    stable.pop("sample_weight", None)
    return orjson.dumps(stable, option=orjson.OPT_SORT_KEYS)


def validate_weight(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{label} must be finite and in (0, 1], got {value!r}")
    return value


def classify_train_row(
    row: dict[str, Any],
    *,
    target_date: str,
    primary_context: int,
    primary_option_types: frozenset[int],
    secondary_context: int,
) -> str:
    context, select = row_context(row)
    action = validated_action(row, select)
    if str(row.get("dataset_date")) != target_date:
        return "base"

    if context == primary_context and len(action) == 1:
        # BC targets are positions in the current option array.  The option's
        # own ``index`` field identifies a card/area and is not the target
        # position (see train_bc_orbit.featurize_row).
        options = select["option"]
        selected = options[action[0]]
        if not isinstance(selected, dict):
            raise TypeError("selected option must be an object")
        option_type = strict_int(selected.get("type"), "selected option.type")
        if option_type in primary_option_types:
            return "primary"

    if context == secondary_context:
        min_count = strict_int(
            select.get("minCount", row.get("min_count")),
            "select.minCount",
        )
        max_count = strict_int(
            select.get("maxCount", row.get("max_count")),
            "select.maxCount",
        )
        if min_count != max_count:
            return "secondary"
    return "base"


def sorted_data_members(archive: zipfile.ZipFile) -> list[tuple[str, str, int]]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("input archive has duplicate member names")
    if names.count("manifest.json") != 1:
        raise ValueError("input archive must contain exactly one manifest.json")
    members: list[tuple[str, str, int]] = []
    for name in names:
        if name == "manifest.json":
            continue
        match = MEMBER_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"unexpected input member: {name!r}")
        members.append((name, match.group(1), int(match.group(2))))
    members.sort(key=lambda item: (SPLIT_ORDER[item[1]], item[2]))
    for split in SPLIT_ORDER:
        indices = [index for _, value, index in members if value == split]
        if not indices or indices != list(range(len(indices))):
            raise ValueError(f"{split} shards must be non-empty and contiguous")
    return members


def rewrite_archive(
    *,
    input_path: Path,
    output_path: Path,
    expected_input_sha256: str,
    target_date: str,
    primary_context: int,
    primary_option_types: frozenset[int],
    secondary_context: int,
    base_weight: float,
    secondary_weight: float,
    primary_weight: float,
    expected_train_rows: int | None = None,
    expected_valid_rows: int | None = None,
    expected_test_rows: int | None = None,
    expected_primary_rows: int | None = None,
    expected_secondary_rows: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file() or input_path.is_symlink():
        raise FileNotFoundError(f"input must be a regular non-symlink file: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    if partial_path.exists() or partial_path.is_symlink():
        raise FileExistsError(partial_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_weight = validate_weight(base_weight, "base_weight")
    secondary_weight = validate_weight(secondary_weight, "secondary_weight")
    primary_weight = validate_weight(primary_weight, "primary_weight")
    if not base_weight < secondary_weight < primary_weight:
        raise ValueError("weights must satisfy base < secondary < primary")
    if not primary_option_types:
        raise ValueError("at least one primary option type is required")

    input_sha256 = sha256_file(input_path)
    if input_sha256 != expected_input_sha256.lower():
        raise ValueError(
            f"input SHA-256 mismatch: expected {expected_input_sha256}, got {input_sha256}"
        )

    counts: Counter[str] = Counter()
    team_counts: Counter[str] = Counter()
    seen_keys: set[str] = set()
    key_digest_rows: list[str] = []
    source_nonweight_digest = hashlib.sha256()
    output_nonweight_digest = hashlib.sha256()
    member_audit: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(input_path) as source:
            bad_member = source.testzip()
            if bad_member is not None:
                raise ValueError(f"input CRC failure: {bad_member}")
            members = sorted_data_members(source)
            source_manifest_bytes = source.read("manifest.json")
            source_manifest = orjson.loads(source_manifest_bytes)
            if not isinstance(source_manifest, dict):
                raise TypeError("source manifest must be an object")

            with zipfile.ZipFile(
                partial_path,
                "w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as target:
                for name, split, _ in members:
                    raw = source.read(name)
                    source_member_sha = sha256_bytes(raw)
                    if split != "train":
                        target.writestr(fixed_zip_info(name), raw)
                        for line_number, line in enumerate(raw.splitlines(), 1):
                            if not line.strip():
                                raise ValueError(f"blank row in {name}:{line_number}")
                            row = orjson.loads(line)
                            if not isinstance(row, dict):
                                raise TypeError(f"row must be object in {name}:{line_number}")
                            if str(row.get("split")) != split:
                                raise ValueError(f"row split mismatch in {name}:{line_number}")
                            team_name = str(row.get("team_name") or "")
                            if not team_name:
                                raise ValueError(f"missing team_name in {name}:{line_number}")
                            team_counts[team_name] += 1
                            key = decision_key(row)
                            if key in seen_keys:
                                raise ValueError(f"duplicate decision key: {key}")
                            seen_keys.add(key)
                            key_digest_rows.append(key)
                            counts[f"{split}_rows"] += 1
                        output_member_sha = source_member_sha
                    else:
                        output_member_digest = hashlib.sha256()
                        info = fixed_zip_info(name)
                        with target.open(info, "w", force_zip64=True) as writer:
                            for line_number, line in enumerate(raw.splitlines(), 1):
                                if not line.strip():
                                    raise ValueError(f"blank row in {name}:{line_number}")
                                row = orjson.loads(line)
                                if not isinstance(row, dict):
                                    raise TypeError(f"row must be object in {name}:{line_number}")
                                if str(row.get("split")) != "train":
                                    raise ValueError(f"row split mismatch in {name}:{line_number}")
                                team_name = str(row.get("team_name") or "")
                                if not team_name:
                                    raise ValueError(
                                        f"missing team_name in {name}:{line_number}"
                                    )
                                team_counts[team_name] += 1
                                key = decision_key(row)
                                if key in seen_keys:
                                    raise ValueError(f"duplicate decision key: {key}")
                                seen_keys.add(key)
                                key_digest_rows.append(key)
                                stable = canonical_without_weight(row)
                                source_nonweight_digest.update(stable + b"\n")
                                bucket = classify_train_row(
                                    row,
                                    target_date=target_date,
                                    primary_context=primary_context,
                                    primary_option_types=primary_option_types,
                                    secondary_context=secondary_context,
                                )
                                row["sample_weight"] = {
                                    "base": base_weight,
                                    "secondary": secondary_weight,
                                    "primary": primary_weight,
                                }[bucket]
                                rewritten = (
                                    orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b"\n"
                                )
                                writer.write(rewritten)
                                output_member_digest.update(rewritten)
                                output_nonweight_digest.update(
                                    canonical_without_weight(row) + b"\n"
                                )
                                counts["train_rows"] += 1
                                counts[f"train_{bucket}_rows"] += 1
                        output_member_sha = output_member_digest.hexdigest()
                    member_audit.append(
                        {
                            "name": name,
                            "split": split,
                            "source_sha256": source_member_sha,
                            "output_sha256": output_member_sha,
                            "byte_identity_preserved": split != "train",
                        }
                    )

                if source_nonweight_digest.digest() != output_nonweight_digest.digest():
                    raise RuntimeError("a train field other than sample_weight changed")

                expected_by_split = {
                    "train": expected_train_rows,
                    "valid": expected_valid_rows,
                    "test": expected_test_rows,
                }
                for split, expected in expected_by_split.items():
                    actual = counts[f"{split}_rows"]
                    if expected is not None and actual != expected:
                        raise ValueError(
                            f"{split} row count mismatch: expected {expected}, got {actual}"
                        )
                for bucket, expected in (
                    ("primary", expected_primary_rows),
                    ("secondary", expected_secondary_rows),
                ):
                    actual = counts[f"train_{bucket}_rows"]
                    if actual == 0:
                        raise ValueError(f"target bucket {bucket!r} is empty")
                    if expected is not None and actual != expected:
                        raise ValueError(
                            f"{bucket} row count mismatch: expected {expected}, got {actual}"
                        )

                key_digest = hashlib.sha256(
                    ("\n".join(sorted(key_digest_rows)) + "\n").encode()
                ).hexdigest()
                manifest = dict(source_manifest)
                manifest["compression"] = "stored"
                source_team_filter = source_manifest.get("team_filter")
                manifest["team_filter"] = {
                    **(
                        source_team_filter
                        if isinstance(source_team_filter, dict)
                        else {}
                    ),
                    "all_teams": False,
                    "global_team_count": len(team_counts),
                    "dated_team_counts": {},
                    "display_names": sorted(team_counts),
                }
                manifest["team_decisions"] = dict(sorted(team_counts.items()))
                manifest["sample_reweighting"] = {
                    "schema_version": "ptcg-bc-sample-reweight-v1",
                    "source_path": str(input_path.resolve()),
                    "source_sha256": input_sha256,
                    "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
                    "scope": "train_only",
                    "valid_test_member_bytes_preserved": True,
                    "mutated_fields_only": ["sample_weight"],
                    "decision_key": {
                        "format": "dataset_date|episode_id|seat|action_step_index",
                        "count": len(seen_keys),
                        "sha256_sorted_lf": key_digest,
                    },
                    "rule": {
                        "target_date": target_date,
                        "primary": {
                            "context": primary_context,
                            "single_action": True,
                            "selected_option_types": sorted(primary_option_types),
                            "weight": primary_weight,
                        },
                        "secondary": {
                            "context": secondary_context,
                            "flexible_cardinality": True,
                            "weight": secondary_weight,
                        },
                        "base_weight": base_weight,
                        "relative_weights_before_common_scale": {
                            "base": 1.0,
                            "secondary": secondary_weight / base_weight,
                            "primary": primary_weight / base_weight,
                        },
                    },
                    "counts": {
                        "train": counts["train_rows"],
                        "valid": counts["valid_rows"],
                        "test": counts["test_rows"],
                        "train_base": counts["train_base_rows"],
                        "train_secondary": counts["train_secondary_rows"],
                        "train_primary": counts["train_primary_rows"],
                    },
                    "observed_team_decisions": dict(sorted(team_counts.items())),
                    "train_non_sample_weight_sha256": source_nonweight_digest.hexdigest(),
                    "members": member_audit,
                    "deterministic_zip": {
                        "member_order": "train,valid,test,manifest; shard ascending",
                        "timestamp": "1980-01-01T00:00:00",
                        "compression": "stored",
                    },
                    "required_training_flags": [
                        "--use-trajectory-weights",
                        "--trajectory-weight-scope=policy_only",
                    ],
                }
                manifest_bytes = (
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
                target.writestr(fixed_zip_info("manifest.json"), manifest_bytes)

        with partial_path.open("rb") as handle:
            os.fsync(handle.fileno())
        with zipfile.ZipFile(partial_path) as audit:
            bad_member = audit.testzip()
            if bad_member is not None:
                raise RuntimeError(f"output CRC failure: {bad_member}")
            output_manifest = orjson.loads(audit.read("manifest.json"))
            audit_counts = output_manifest["sample_reweighting"]["counts"]
            if audit_counts["train"] != counts["train_rows"]:
                raise RuntimeError("output manifest train count mismatch")
        os.replace(partial_path, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    return {
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "input_sha256": input_sha256,
        "counts": dict(counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--primary-context", type=int, default=0)
    parser.add_argument("--primary-option-type", type=int, action="append", required=True)
    parser.add_argument("--secondary-context", type=int, default=7)
    parser.add_argument("--base-weight", type=float, default=2.0 / 3.0)
    parser.add_argument("--secondary-weight", type=float, default=5.0 / 6.0)
    parser.add_argument("--primary-weight", type=float, default=1.0)
    parser.add_argument("--expected-train-rows", type=int)
    parser.add_argument("--expected-valid-rows", type=int)
    parser.add_argument("--expected-test-rows", type=int)
    parser.add_argument("--expected-primary-rows", type=int)
    parser.add_argument("--expected-secondary-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = rewrite_archive(
        input_path=args.input,
        output_path=args.output,
        expected_input_sha256=args.expected_input_sha256,
        target_date=args.target_date,
        primary_context=args.primary_context,
        primary_option_types=frozenset(args.primary_option_type),
        secondary_context=args.secondary_context,
        base_weight=args.base_weight,
        secondary_weight=args.secondary_weight,
        primary_weight=args.primary_weight,
        expected_train_rows=args.expected_train_rows,
        expected_valid_rows=args.expected_valid_rows,
        expected_test_rows=args.expected_test_rows,
        expected_primary_rows=args.expected_primary_rows,
        expected_secondary_rows=args.expected_secondary_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

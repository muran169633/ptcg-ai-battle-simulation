#!/usr/bin/env python3
"""Fresh-cache and real-main audit for the frozen PPO/BC dual-anchor profile.

This audit reruns the hash-bound train-only profiler into a second initially
absent cache, compares a cache-location/statistics-independent semantic profile,
independently recomputes all BC/PPO classifications, and calls the real frozen
hybrid ``main.py`` entrypoints for both anchors on all 61,600 train decisions.

Only fixed ``train/*.jsonl`` members are opened.  No model, candidate, holdout,
upload, or submission is produced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import orjson
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
DUAL_PROFILER_PATH = ROOT / "tools/profile_yanz_ppo_dual_anchor_deployment_v1.py"
ACCEPTED_PROFILE = ROOT / (
    "data/yanz_alakazam_ppo_dual_anchor_20260810_v1/train_only_profile.json"
)

SCHEMA = "ptcg-yanz-ppo-dual-anchor-independent-audit-v1"
DUAL_PROFILER_SHA256 = (
    "25ace3c610ea587a5e3dc6de1848f6e277bf833a4d3c58237ee490fc79393b65"
)
ACCEPTED_PROFILE_SHA256 = (
    "e10e2e6d4ee1b1225b82c1e43af089969bbdad61dfd70e2c42d151e85015b7b0"
)
EXPECTED_ROWS = {"yanz": 1725, "old": 59875}
EXPECTED_SHARDS = {"yanz": 4, "old": 118}
CATEGORIES = (
    "ppo_correct_protection",
    "bc_correct_ppo_wrong_recovery_target",
    "ppo_only_repair_protection",
    "both_wrong",
)
EXPECTED_COUNTS: dict[str, dict[str, Any]] = {
    "yanz": {
        "bc": {"hybrid_correct": 1421, "set_correct": 1422, "count_correct": 1723},
        "ppo": {"hybrid_correct": 1413, "set_correct": 1414, "count_correct": 1723},
        "hybrid_categories": {
            "ppo_correct_protection": 1406,
            "bc_correct_ppo_wrong_recovery_target": 15,
            "ppo_only_repair_protection": 7,
            "both_wrong": 297,
        },
        "set_categories": {
            "ppo_correct_protection": 1407,
            "bc_correct_ppo_wrong_recovery_target": 15,
            "ppo_only_repair_protection": 7,
            "both_wrong": 296,
        },
        "count_categories": {
            "ppo_correct_protection": 1723,
            "bc_correct_ppo_wrong_recovery_target": 0,
            "ppo_only_repair_protection": 0,
            "both_wrong": 2,
        },
    },
    "old": {
        "bc": {"hybrid_correct": 56241, "set_correct": 56482, "count_correct": 59864},
        "ppo": {"hybrid_correct": 56070, "set_correct": 56303, "count_correct": 59864},
        "hybrid_categories": {
            "ppo_correct_protection": 55798,
            "bc_correct_ppo_wrong_recovery_target": 443,
            "ppo_only_repair_protection": 272,
            "both_wrong": 3362,
        },
        "set_categories": {
            "ppo_correct_protection": 56035,
            "bc_correct_ppo_wrong_recovery_target": 447,
            "ppo_only_repair_protection": 268,
            "both_wrong": 3125,
        },
        "count_categories": {
            "ppo_correct_protection": 59864,
            "bc_correct_ppo_wrong_recovery_target": 0,
            "ppo_only_repair_protection": 0,
            "both_wrong": 11,
        },
    },
}


class AuditError(RuntimeError):
    """A fail-closed dual-anchor audit contract violation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_json_strict(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise AuditError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def write_new_json(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(canonical_json_bytes(dict(payload)))
        if path.exists():
            raise FileExistsError(f"audit JSON appeared during write: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_dual_profiler() -> ModuleType:
    module_name = "ptcg_frozen_yanz_ppo_dual_anchor_profiler_v1_for_audit"
    spec = importlib.util.spec_from_file_location(module_name, DUAL_PROFILER_PATH)
    if spec is None or spec.loader is None:
        raise AuditError("cannot construct frozen dual-anchor profiler import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dual = _load_dual_profiler()
frozen = dual.frozen


def assert_file_sha(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise AuditError(f"{label} SHA drift: {actual} != {expected}")


def independent_category(bc_correct: bool, ppo_correct: bool) -> str:
    if bc_correct and ppo_correct:
        return "ppo_correct_protection"
    if bc_correct:
        return "bc_correct_ppo_wrong_recovery_target"
    if ppo_correct:
        return "ppo_only_repair_protection"
    return "both_wrong"


def validate_frozen_bindings() -> dict[str, Any]:
    assert_file_sha(DUAL_PROFILER_PATH, DUAL_PROFILER_SHA256, "dual profiler")
    assert_file_sha(ACCEPTED_PROFILE, ACCEPTED_PROFILE_SHA256, "accepted profile")
    static = dual.validate_static_bindings()
    expected = {
        "ppo_parent_checkpoint": dual.PPO_PARENT_SHA256,
        "pure_bc_checkpoint": dual.PURE_BC_SHA256,
        "yanz_train_archive": dual.YANZ_SHA256,
        "old_train_archive": dual.OLD_SHA256,
        "hybrid_main": dual.HYBRID_MAIN_SHA256,
        "policy_runtime": dual.RUNTIME_SHA256,
        "main_runtime_anchor": dual.RUNTIME_SHA256,
        "template_contract": dual.CONTRACT_SHA256,
        "deck_csv": dual.DECK_CSV_SHA256,
        "frozen_profiler_dependency": dual.FROZEN_PROFILER_SHA256,
    }
    for key, sha256 in expected.items():
        binding = static.get(key)
        if not isinstance(binding, Mapping) or binding.get("sha256") != sha256:
            raise AuditError(f"static binding drift: {key}")
    semantics = static.get("runtime_semantics")
    required_semantics = {
        "device": "cpu",
        "dtype": "torch.float32",
        "batch_size": 1,
        "autocast": False,
        "inference_mode": True,
        "torch_num_threads": 1,
        "count_classes": 61,
        "models_per_row": 2,
        "model_evaluation_order": ["pure_bc_teacher", "ppo_parent"],
        "opened_archive_members": "train/*.jsonl only",
    }
    if not isinstance(semantics, Mapping):
        raise AuditError("runtime semantics missing")
    for key, expected_value in required_semantics.items():
        if semantics.get(key) != expected_value:
            raise AuditError(f"runtime semantics drift: {key}")
    return static


def validate_profile_contract(profile: Mapping[str, Any], *, accepted: bool) -> None:
    if profile.get("schema_version") != dual.SCHEMA:
        raise AuditError("dual profile schema drift")
    if profile.get("status") != "frozen_complete_train_only_dual_anchor_deployment_profile":
        raise AuditError("dual profile status drift")
    bindings = profile.get("bindings")
    if not isinstance(bindings, Mapping):
        raise AuditError("dual profile bindings missing")
    profiler_binding = bindings.get("profiler")
    if not isinstance(profiler_binding, Mapping) or profiler_binding.get(
        "sha256"
    ) != DUAL_PROFILER_SHA256:
        raise AuditError("dual profile profiler binding drift")
    anchors = bindings.get("anchors")
    if not isinstance(anchors, Mapping):
        raise AuditError("dual profile anchor audit missing")
    bc_anchor = anchors.get("pure_bc_teacher")
    ppo_anchor = anchors.get("ppo_parent")
    if not isinstance(bc_anchor, Mapping) or bc_anchor.get(
        "expanded_model_state_sha256"
    ) != dual.PURE_BC_EXPANDED61_BITWISE_STATE_SHA256:
        raise AuditError("pure-BC anchor bitwise binding drift")
    if not isinstance(ppo_anchor, Mapping) or ppo_anchor.get(
        "model_state_sha256"
    ) != dual.PPO_PARENT_BITWISE_STATE_SHA256:
        raise AuditError("PPO anchor bitwise binding drift")
    if bc_anchor.get("raw_model_state_sha256") != dual.PURE_BC_RAW_BITWISE_STATE_SHA256:
        raise AuditError("pure-BC raw bitwise binding drift")
    source_counts = profile.get("source_counts")
    if not isinstance(source_counts, Mapping):
        raise AuditError("dual profile source counts missing")
    for source in ("yanz", "old"):
        counts = source_counts.get(source)
        if not isinstance(counts, Mapping):
            raise AuditError(f"dual profile source missing: {source}")
        if int(counts.get("raw_rows", -1)) != EXPECTED_ROWS[source]:
            raise AuditError(f"dual profile row count drift: {source}")
        if int(counts.get("accepted_rows", -1)) != EXPECTED_ROWS[source]:
            raise AuditError(f"dual profile accepted row drift: {source}")
        if int(counts.get("cache_shards", -1)) != EXPECTED_SHARDS[source]:
            raise AuditError(f"dual profile source shard drift: {source}")
        for key in ("bc", "ppo", "hybrid_categories", "set_categories", "count_categories"):
            if counts.get(key) != EXPECTED_COUNTS[source][key]:
                raise AuditError(f"dual profile count drift: {source}.{key}")
    cache = profile.get("cache")
    if not isinstance(cache, Mapping):
        raise AuditError("dual profile cache manifest missing")
    shards = cache.get("immutable_shards")
    expected_shards = sum(EXPECTED_SHARDS.values())
    if not isinstance(shards, list) or len(shards) != expected_shards:
        raise AuditError("dual profile cache shard count drift")
    if int(cache.get("shard_count", -1)) != expected_shards:
        raise AuditError("dual profile declared shard count drift")
    scope = profile.get("scope")
    if not isinstance(scope, Mapping) or scope.get("train_members_only") is not True:
        raise AuditError("dual profile train-only scope drift")
    for key in (
        "holdout_members_opened",
        "model_artifact_written",
        "submission_performed",
        "external_upload_performed",
    ):
        if scope.get(key) is not False:
            raise AuditError(f"dual profile forbidden scope drift: {key}")
    if accepted and file_sha256(ACCEPTED_PROFILE) != ACCEPTED_PROFILE_SHA256:
        raise AuditError("accepted profile changed during validation")


def semantic_projection(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only cache location and cache execution counters."""

    projected = copy.deepcopy(dict(profile))
    counts = projected.get("source_counts")
    if not isinstance(counts, dict):
        raise AuditError("semantic source counts missing")
    for source in ("yanz", "old"):
        source_counts = counts.get(source)
        if not isinstance(source_counts, dict):
            raise AuditError(f"semantic source missing: {source}")
        for key in ("cache_hits", "cache_writes"):
            if key not in source_counts:
                raise AuditError(f"semantic execution field missing: {source}.{key}")
            source_counts.pop(key)
    cache = projected.get("cache")
    if not isinstance(cache, dict) or not isinstance(cache.get("immutable_shards"), list):
        raise AuditError("semantic cache manifest missing")
    for index, shard in enumerate(cache["immutable_shards"]):
        if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
            raise AuditError(f"semantic shard path missing: {index}")
        shard.pop("path")
    return projected


def semantic_profile_sha256(profile: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(semantic_projection(profile))).hexdigest()


def manifest_projection(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for shard in profile["cache"]["immutable_shards"]:
        if not isinstance(shard, Mapping):
            raise AuditError("manifest shard is not an object")
        item = dict(shard)
        if not isinstance(item.pop("path", None), str):
            raise AuditError("manifest shard path missing")
        output.append(item)
    return output


def assert_fresh_targets(cache_dir: Path, fresh_profile: Path, output: Path) -> None:
    if cache_dir.exists():
        raise FileExistsError(f"second cache must be initially absent: {cache_dir}")
    for label, path in (("fresh profile", fresh_profile), ("audit output", output)):
        if path.exists():
            raise FileExistsError(f"{label} already exists: {path}")
    accepted_cache = ACCEPTED_PROFILE.parent / "cache"
    if cache_dir.resolve() == accepted_cache.resolve():
        raise AuditError("second cache cannot be the accepted cache")


def verify_manifest_files(
    profile: Mapping[str, Any], *, required_root: Path | None
) -> dict[str, Any]:
    rows = 0
    by_source: Counter[str] = Counter()
    digest = hashlib.sha256()
    digest.update(b"ptcg-dual-anchor-cache-manifest-semantic-v1\0")
    for item in profile["cache"]["immutable_shards"]:
        path = Path(str(item["path"])).resolve()
        if required_root is not None:
            try:
                path.relative_to(required_root.resolve())
            except ValueError as error:
                raise AuditError(f"fresh shard escaped second cache: {path}") from error
        assert_file_sha(path, str(item["sha256"]), "cache shard")
        count = int(item.get("record_count", -1))
        if count < 1:
            raise AuditError(f"cache shard record count drift: {path}")
        source = str(item.get("source", ""))
        rows += count
        by_source[source] += count
        semantic_item = dict(item)
        semantic_item.pop("path", None)
        digest.update(canonical_json_bytes(semantic_item))
    if dict(by_source) != EXPECTED_ROWS or rows != sum(EXPECTED_ROWS.values()):
        raise AuditError(f"cache manifest row/source drift: {dict(by_source)}")
    return {
        "shards": sum(EXPECTED_SHARDS.values()),
        "rows": rows,
        "by_source": dict(by_source),
        "semantic_manifest_sha256": digest.hexdigest(),
    }


def _int_action(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise AuditError(f"invalid action field: {label}")
    return list(value)


def _new_source_audit() -> dict[str, Any]:
    return {
        "rows": 0,
        "bc": Counter(),
        "ppo": Counter(),
        "hybrid_categories": Counter(),
        "set_categories": Counter(),
        "count_categories": Counter(),
        "bc_ppo_policy_action_equal": 0,
        "bc_ppo_hybrid_action_equal": 0,
        "bc_ppo_predicted_count_equal": 0,
        "both_wrong_different_hybrid_action": 0,
    }


def load_and_recompute_fresh_records(
    profile: Mapping[str, Any], cache_dir: Path
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Independently validate records, categories, indexes, and summary counts."""

    records: dict[tuple[str, str], dict[str, Any]] = {}
    audits = {"yanz": _new_source_audit(), "old": _new_source_audit()}
    indexes = {
        source: {category: [] for category in CATEGORIES}
        for source in ("yanz", "old")
    }
    stream_digests = {source: hashlib.sha256() for source in ("yanz", "old")}
    for digest in stream_digests.values():
        digest.update(b"ptcg-yanz-ppo-dual-anchor-record-stream-v1\0")
    cache_contract = profile["bindings"]["cache_contract"]

    for manifest in profile["cache"]["immutable_shards"]:
        path = Path(str(manifest["path"])).resolve()
        try:
            path.relative_to(cache_dir.resolve())
        except ValueError as error:
            raise AuditError(f"fresh shard escaped second cache: {path}") from error
        payload = load_json_strict(path)
        if payload.get("schema_version") != dual.CACHE_SCHEMA:
            raise AuditError(f"fresh cache schema drift: {path}")
        if payload.get("status") != "complete_immutable_train_only_dual_anchor_chunk":
            raise AuditError(f"fresh cache status drift: {path}")
        if payload.get("cache_bindings") != cache_contract:
            raise AuditError(f"fresh cache binding drift: {path}")
        rows = payload.get("records")
        if not isinstance(rows, list) or int(payload.get("record_count", -1)) != len(rows):
            raise AuditError(f"fresh cache records drift: {path}")
        records_sha = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
        if records_sha != payload.get("records_sha256") or records_sha != manifest.get(
            "records_sha256"
        ):
            raise AuditError(f"fresh cache record hash drift: {path}")
        for raw_record in rows:
            if not isinstance(raw_record, Mapping):
                raise AuditError(f"fresh cache record is not an object: {path}")
            record = dict(raw_record)
            if record.get("record_schema") != dual.RECORD_SCHEMA:
                raise AuditError(f"fresh record schema drift: {path}")
            source = str(record.get("source", ""))
            line_sha = str(record.get("line_sha256", ""))
            if source not in audits or len(line_sha) != 64:
                raise AuditError(f"fresh record identity drift: {path}")
            key = (source, line_sha)
            if key in records:
                raise AuditError(f"duplicate fresh record: {source}:{line_sha}")
            expert = _int_action(record.get("expert_action_order"), "expert")
            bc_policy = _int_action(record.get("bc_policy_action"), "bc_policy")
            ppo_policy = _int_action(record.get("ppo_policy_action"), "ppo_policy")
            bc_hybrid = _int_action(record.get("bc_hybrid_action"), "bc_hybrid")
            ppo_hybrid = _int_action(record.get("ppo_hybrid_action"), "ppo_hybrid")
            if not isinstance(record.get("context"), int) or isinstance(
                record.get("context"), bool
            ):
                raise AuditError(f"fresh record context drift: {source}:{line_sha}")
            for prefix, policy, hybrid in (
                ("bc", bc_policy, bc_hybrid),
                ("ppo", ppo_policy, ppo_hybrid),
            ):
                predicted_count = record.get(f"{prefix}_predicted_count")
                if not isinstance(predicted_count, int) or isinstance(predicted_count, bool):
                    raise AuditError(f"fresh predicted count drift: {source}:{line_sha}")
                expected_bools = {
                    f"{prefix}_hybrid_correct": hybrid == expert,
                    f"{prefix}_set_correct": sorted(policy) == sorted(expert),
                    f"{prefix}_count_correct": predicted_count == len(expert),
                }
                for field, expected in expected_bools.items():
                    if record.get(field) is not expected:
                        raise AuditError(f"fresh correctness drift {field}: {source}:{line_sha}")
                    if expected:
                        audits[source][prefix][field.removeprefix(f"{prefix}_")] += 1
            for basis in ("hybrid", "set", "count"):
                category = independent_category(
                    bool(record[f"bc_{basis}_correct"]),
                    bool(record[f"ppo_{basis}_correct"]),
                )
                if record.get(f"{basis}_category") != category:
                    raise AuditError(f"fresh {basis} category drift: {source}:{line_sha}")
                audits[source][f"{basis}_categories"][category] += 1
                if basis == "hybrid":
                    indexes[source][category].append(line_sha)
            equality_checks = {
                "bc_ppo_policy_action_equal": bc_policy == ppo_policy,
                "bc_ppo_hybrid_action_equal": bc_hybrid == ppo_hybrid,
                "bc_ppo_predicted_count_equal": (
                    int(record["bc_predicted_count"]) == int(record["ppo_predicted_count"])
                ),
            }
            for field, expected in equality_checks.items():
                if record.get(field) is not expected:
                    raise AuditError(f"fresh equality field drift {field}: {source}:{line_sha}")
                audits[source][field] += int(expected)
            if record["hybrid_category"] == "both_wrong" and bc_hybrid != ppo_hybrid:
                audits[source]["both_wrong_different_hybrid_action"] += 1
            audits[source]["rows"] += 1
            stream_digests[source].update(canonical_json_bytes(record))
            records[key] = {
                "bc_action": bc_hybrid,
                "ppo_action": ppo_hybrid,
                "context": int(record["context"]),
            }

    if len(records) != sum(EXPECTED_ROWS.values()):
        raise AuditError(f"fresh unique record count drift: {len(records)}")
    profile_indexes = profile.get("hybrid_category_index")
    if not isinstance(profile_indexes, Mapping):
        raise AuditError("fresh category index missing")
    index_payload = {"yanz": indexes["yanz"], "old": indexes["old"]}
    if profile_indexes.get("yanz") != indexes["yanz"] or profile_indexes.get(
        "old"
    ) != indexes["old"]:
        raise AuditError("fresh category index differs from cache records")
    index_sha = hashlib.sha256(canonical_json_bytes(index_payload)).hexdigest()
    if profile_indexes.get("sha256") != index_sha:
        raise AuditError("fresh category index SHA drift")

    serialisable: dict[str, Any] = {}
    for source in ("yanz", "old"):
        audit = audits[source]
        if audit["rows"] != EXPECTED_ROWS[source]:
            raise AuditError(f"fresh recomputed row drift: {source}")
        reduced = {
            "bc": {
                "hybrid_correct": int(audit["bc"]["hybrid_correct"]),
                "set_correct": int(audit["bc"]["set_correct"]),
                "count_correct": int(audit["bc"]["count_correct"]),
            },
            "ppo": {
                "hybrid_correct": int(audit["ppo"]["hybrid_correct"]),
                "set_correct": int(audit["ppo"]["set_correct"]),
                "count_correct": int(audit["ppo"]["count_correct"]),
            },
            "hybrid_categories": {
                category: int(audit["hybrid_categories"][category])
                for category in CATEGORIES
            },
            "set_categories": {
                category: int(audit["set_categories"][category])
                for category in CATEGORIES
            },
            "count_categories": {
                category: int(audit["count_categories"][category])
                for category in CATEGORIES
            },
        }
        for key, expected in EXPECTED_COUNTS[source].items():
            if reduced[key] != expected:
                raise AuditError(f"fresh independently recomputed count drift: {source}.{key}")
            if profile["source_counts"][source][key] != expected:
                raise AuditError(f"fresh profile summarized count drift: {source}.{key}")
        stream_sha = stream_digests[source].hexdigest()
        if profile["source_counts"][source].get("record_stream_sha256") != stream_sha:
            raise AuditError(f"fresh record stream SHA drift: {source}")
        serialisable[source] = {
            "rows": int(audit["rows"]),
            **reduced,
            "bc_ppo_policy_action_equal": int(audit["bc_ppo_policy_action_equal"]),
            "bc_ppo_hybrid_action_equal": int(audit["bc_ppo_hybrid_action_equal"]),
            "bc_ppo_predicted_count_equal": int(
                audit["bc_ppo_predicted_count_equal"]
            ),
            "both_wrong_different_hybrid_action": int(
                audit["both_wrong_different_hybrid_action"]
            ),
            "record_stream_sha256": stream_sha,
        }
    serialisable["hybrid_category_index_sha256"] = index_sha
    return records, serialisable


def iter_train_observations(
    archive_path: Path, source: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Independent train-only archive reader used by real-main conformance."""

    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not members:
            raise AuditError(f"no train JSONL members: {archive_path}")
        for member in members:
            with archive.open(member) as handle:
                for line in handle:
                    row = orjson.loads(line)
                    if not isinstance(row, dict) or row.get("split") != "train":
                        raise AuditError(f"non-train row in train member: {source}:{member}")
                    if row.get("deck_hash") != dual.DECK_HASH:
                        raise AuditError(f"deck drift: {source}:{member}")
                    observation = row.get("observation")
                    if not isinstance(observation, dict) or not isinstance(
                        observation.get("select"), dict
                    ):
                        raise AuditError(f"selection observation missing: {source}:{member}")
                    line_sha = hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()
                    yield line_sha, observation


def load_actual_hybrid_main(
    *,
    label: str,
    runtime: ModuleType,
    model: torch.nn.Module,
    config: Mapping[str, int | float],
) -> ModuleType:
    """Import the real frozen main into an anchor-specific module instance."""

    assert_file_sha(dual.HYBRID_MAIN, dual.HYBRID_MAIN_SHA256, "hybrid main")
    assert_file_sha(dual.RUNTIME, dual.RUNTIME_SHA256, "policy runtime")
    assert_file_sha(dual.MAIN_RUNTIME_ANCHOR, dual.RUNTIME_SHA256, "runtime anchor")
    assert_file_sha(
        dual.MAIN_RUNTIME_ANCHOR.parent / "deck.csv",
        dual.DECK_CSV_SHA256,
        "runtime-anchor deck",
    )
    module_name = f"ptcg_dual_anchor_real_hybrid_main_{label}"
    spec = importlib.util.spec_from_file_location(module_name, dual.HYBRID_MAIN)
    if spec is None or spec.loader is None:
        raise AuditError(f"cannot construct real hybrid main import: {label}")
    module = importlib.util.module_from_spec(spec)
    previous_alias = sys.modules.get("policy_runtime")
    previous_runtime_file = runtime.__file__
    try:
        runtime.__file__ = str(dual.MAIN_RUNTIME_ANCHOR)
        sys.modules["policy_runtime"] = runtime
        spec.loader.exec_module(module)
    finally:
        runtime.__file__ = previous_runtime_file
        if previous_alias is None:
            sys.modules.pop("policy_runtime", None)
        else:
            sys.modules["policy_runtime"] = previous_alias
    module._MODEL = model
    module._MODEL_CONFIG = dict(config)
    loaded_model, loaded_config = module._load_model()
    if loaded_model is not model or loaded_config != dict(config):
        raise AuditError(f"real main did not retain injected anchor: {label}")
    if module.DECK_HASH != dual.DECK_HASH or module.SKILL_ORDER_CONTEXT != 34:
        raise AuditError(f"real main constants drift: {label}")
    for function_name in ("_policy_action", "agent"):
        function = getattr(module, function_name)
        if Path(function.__code__.co_filename).resolve() != dual.HYBRID_MAIN.resolve():
            raise AuditError(f"real main function source drift: {label}.{function_name}")
    return module


def _cpu_autocast_enabled() -> bool:
    try:
        return bool(torch.is_autocast_enabled("cpu"))
    except TypeError:  # pragma: no cover - older torch fallback
        return bool(torch.is_autocast_cpu_enabled())


def validate_loaded_anchors(
    bc_model: torch.nn.Module, ppo_model: torch.nn.Module
) -> dict[str, Any]:
    bc_sha = frozen.bitwise_model_state_sha256(bc_model.state_dict())
    ppo_sha = frozen.bitwise_model_state_sha256(ppo_model.state_dict())
    if bc_sha != dual.PURE_BC_EXPANDED61_BITWISE_STATE_SHA256:
        raise AuditError("loaded pure-BC expanded61 bitwise state drift")
    if ppo_sha != dual.PPO_PARENT_BITWISE_STATE_SHA256:
        raise AuditError("loaded PPO bitwise state drift")
    for label, model in (("pure_bc", bc_model), ("ppo", ppo_model)):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise AuditError(f"loaded anchor mode drift: {label}")
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise AuditError(f"loaded anchor device drift: {label}")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise AuditError(f"loaded anchor dtype drift: {label}")
    if torch.get_num_threads() != 1:
        raise AuditError("runtime is not one CPU thread")
    if torch.is_autocast_enabled() or _cpu_autocast_enabled():
        raise AuditError("autocast unexpectedly enabled")
    return {
        "pure_bc_expanded61_bitwise_state_sha256": bc_sha,
        "ppo_parent_bitwise_state_sha256": ppo_sha,
        "both_models_cpu": True,
        "both_models_float32": True,
        "both_models_eval": True,
        "both_models_require_grad_false": True,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_autocast_enabled": False,
        "cuda_autocast_enabled": False,
    }


def run_full_real_main_conformance(
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    runtime: ModuleType,
    bc_model: torch.nn.Module,
    ppo_model: torch.nn.Module,
    config: Mapping[str, int | float],
) -> dict[str, Any]:
    bc_main = load_actual_hybrid_main(
        label="pure_bc", runtime=runtime, model=bc_model, config=config
    )
    ppo_main = load_actual_hybrid_main(
        label="ppo_parent", runtime=runtime, model=ppo_model, config=config
    )
    for label, module in (("pure_bc", bc_main), ("ppo_parent", ppo_main)):
        deck_action = module.agent({})
        if deck_action != list(module.DECK) or len(deck_action) != 60:
            raise AuditError(f"real main deck branch drift: {label}")

    seen: set[tuple[str, str]] = set()
    by_source: Counter[str] = Counter()
    by_context: Counter[int] = Counter()
    digest = hashlib.sha256()
    digest.update(b"ptcg-dual-anchor-real-main-conformance-v1\0")
    with torch.inference_mode():
        for source, archive in (("yanz", dual.YANZ), ("old", dual.OLD)):
            for line_sha, observation in iter_train_observations(archive, source):
                key = (source, line_sha)
                if key in seen:
                    raise AuditError(f"duplicate source row: {source}:{line_sha}")
                record = expected.get(key)
                if record is None:
                    raise AuditError(f"fresh cache record missing: {source}:{line_sha}")
                bc_policy = bc_main._policy_action(dict(observation))
                bc_agent = bc_main.agent(dict(observation))
                ppo_policy = ppo_main._policy_action(dict(observation))
                ppo_agent = ppo_main.agent(dict(observation))
                wanted_bc = record["bc_action"]
                wanted_ppo = record["ppo_action"]
                if bc_policy != wanted_bc or bc_agent != wanted_bc:
                    raise AuditError(f"real main pure-BC mismatch: {source}:{line_sha}")
                if ppo_policy != wanted_ppo or ppo_agent != wanted_ppo:
                    raise AuditError(f"real main PPO mismatch: {source}:{line_sha}")
                if bc_policy != bc_agent or ppo_policy != ppo_agent:
                    raise AuditError(f"real main entrypoint disagreement: {source}:{line_sha}")
                raw_context = observation["select"].get("context", -1)
                context = -1 if raw_context is None else int(raw_context)
                if context != int(record["context"]):
                    raise AuditError(f"real main context drift: {source}:{line_sha}")
                digest.update(source.encode("ascii"))
                digest.update(b"\0")
                digest.update(line_sha.encode("ascii"))
                digest.update(b"\0")
                digest.update(canonical_json_bytes({"bc": bc_policy, "ppo": ppo_policy}))
                seen.add(key)
                by_source[source] += 1
                by_context[context] += 1
    if seen != set(expected):
        raise AuditError(f"source/fresh identity set mismatch: {len(seen)}")
    if dict(by_source) != EXPECTED_ROWS:
        raise AuditError(f"real main source count drift: {dict(by_source)}")
    rows = len(seen)
    return {
        "rows_checked": rows,
        "pure_bc_policy_action_matches_fresh_cache": rows,
        "pure_bc_agent_matches_fresh_cache": rows,
        "ppo_policy_action_matches_fresh_cache": rows,
        "ppo_agent_matches_fresh_cache": rows,
        "pure_bc_policy_action_matches_agent": rows,
        "ppo_policy_action_matches_agent": rows,
        "by_source": dict(by_source),
        "by_context": {str(key): value for key, value in sorted(by_context.items())},
        "both_deck_branches_match_bound_deck": True,
        "dual_anchor_real_main_action_ledger_sha256": digest.hexdigest(),
        "hybrid_main_path": str(dual.HYBRID_MAIN.resolve()),
    }


def run(cache_dir: Path, fresh_profile_path: Path, output: Path) -> dict[str, Any]:
    assert_fresh_targets(cache_dir, fresh_profile_path, output)
    startup_script_sha = file_sha256(SCRIPT)
    static = validate_frozen_bindings()
    accepted = load_json_strict(ACCEPTED_PROFILE)
    validate_profile_contract(accepted, accepted=True)
    accepted_manifest = verify_manifest_files(accepted, required_root=None)

    build_result = dual.build(cache_dir, fresh_profile_path)
    if int(build_result.get("cache_shards", -1)) != sum(EXPECTED_SHARDS.values()):
        raise AuditError("fresh dual-profiler shard count drift")
    fresh = load_json_strict(fresh_profile_path)
    validate_profile_contract(fresh, accepted=False)
    for source in ("yanz", "old"):
        counts = fresh["source_counts"][source]
        if int(counts.get("cache_hits", -1)) != 0:
            raise AuditError(f"second cache was not empty: {source}")
        if int(counts.get("cache_writes", -1)) != EXPECTED_SHARDS[source]:
            raise AuditError(f"second cache write count drift: {source}")

    accepted_semantic_sha = semantic_profile_sha256(accepted)
    fresh_semantic_sha = semantic_profile_sha256(fresh)
    if semantic_projection(accepted) != semantic_projection(fresh):
        raise AuditError("fresh semantic projection differs from accepted")
    if accepted_semantic_sha != fresh_semantic_sha:
        raise AuditError("fresh semantic profile SHA differs from accepted")
    if manifest_projection(accepted) != manifest_projection(fresh):
        raise AuditError("fresh semantic shard manifest differs from accepted")
    fresh_manifest = verify_manifest_files(fresh, required_root=cache_dir)
    if accepted_manifest["semantic_manifest_sha256"] != fresh_manifest[
        "semantic_manifest_sha256"
    ]:
        raise AuditError("fresh semantic shard manifest SHA differs from accepted")

    fresh_records, classification_audit = load_and_recompute_fresh_records(
        fresh, cache_dir
    )
    runtime = frozen.load_bound_runtime()
    bc_model, ppo_model, config, anchor_load_audit = dual.load_bound_models(runtime)
    exact_runtime_audit = validate_loaded_anchors(bc_model, ppo_model)
    main_conformance = run_full_real_main_conformance(
        fresh_records,
        runtime=runtime,
        bc_model=bc_model,
        ppo_model=ppo_model,
        config=config,
    )
    after_anchor_audit = validate_loaded_anchors(bc_model, ppo_model)
    if exact_runtime_audit != after_anchor_audit:
        raise AuditError("anchor/runtime audit changed during real-main conformance")

    if file_sha256(SCRIPT) != startup_script_sha:
        raise AuditError("audit tool changed during run")
    validate_frozen_bindings()
    if file_sha256(fresh_profile_path) != build_result.get("sha256"):
        raise AuditError("fresh profile changed after build")

    audit = {
        "schema_version": SCHEMA,
        "status": "passed_full_train_fresh_rerun_classification_and_dual_real_main",
        "bindings": {
            "audit_tool": {"path": str(SCRIPT.resolve()), "sha256": startup_script_sha},
            "dual_profiler": {
                "path": str(DUAL_PROFILER_PATH.resolve()),
                "sha256": DUAL_PROFILER_SHA256,
            },
            "accepted_profile": {
                "path": str(ACCEPTED_PROFILE.resolve()),
                "sha256": ACCEPTED_PROFILE_SHA256,
            },
            "fresh_profile": {
                "path": str(fresh_profile_path.resolve()),
                "sha256": file_sha256(fresh_profile_path),
            },
            "ppo_parent_checkpoint": static["ppo_parent_checkpoint"],
            "pure_bc_checkpoint": static["pure_bc_checkpoint"],
            "yanz_train_archive": static["yanz_train_archive"],
            "old_train_archive": static["old_train_archive"],
            "hybrid_main": static["hybrid_main"],
            "policy_runtime": static["policy_runtime"],
            "frozen_profiler_dependency": static["frozen_profiler_dependency"],
            "runtime_semantics": static["runtime_semantics"],
        },
        "fresh_rerun": {
            "second_cache_initially_absent": True,
            "second_cache": str(cache_dir.resolve()),
            "cache_hits": {"yanz": 0, "old": 0},
            "cache_writes": dict(EXPECTED_SHARDS),
            "accepted_profile_raw_sha256": ACCEPTED_PROFILE_SHA256,
            "fresh_profile_raw_sha256": file_sha256(fresh_profile_path),
            "semantic_hash_algorithm": (
                "sha256(canonical-json-v1 after removing cache shard absolute path "
                "and source cache_hits/cache_writes)"
            ),
            "semantic_hash_excluded_fields": [
                "cache.immutable_shards[*].path",
                "source_counts.yanz.cache_hits",
                "source_counts.yanz.cache_writes",
                "source_counts.old.cache_hits",
                "source_counts.old.cache_writes",
            ],
            "accepted_semantic_sha256": accepted_semantic_sha,
            "fresh_semantic_sha256": fresh_semantic_sha,
            "semantic_projection_exact_match": True,
            "cache_manifest_projection_exact_match": True,
            "accepted_cache_manifest": accepted_manifest,
            "fresh_cache_manifest": fresh_manifest,
        },
        "independent_classification_audit": classification_audit,
        "anchor_load_audit": anchor_load_audit,
        "exact_runtime_audit": exact_runtime_audit,
        "real_hybrid_main_conformance": main_conformance,
        "scope": {
            "opened_archive_members": "train/*.jsonl only",
            "holdout_opened": False,
            "model_artifact_written": False,
            "training_performed": False,
            "candidate_written": False,
            "external_upload_performed": False,
            "submission_performed": False,
        },
    }
    if file_sha256(SCRIPT) != startup_script_sha:
        raise AuditError("audit tool changed before output")
    write_new_json(audit, output)
    if file_sha256(SCRIPT) != startup_script_sha:
        raise AuditError("audit tool changed after output")
    return {
        "status": audit["status"],
        "output": str(output.resolve()),
        "sha256": file_sha256(output),
        "fresh_profile": str(fresh_profile_path.resolve()),
        "fresh_profile_sha256": file_sha256(fresh_profile_path),
        "semantic_sha256": fresh_semantic_sha,
        "rows_checked": main_conformance["rows_checked"],
        "pure_bc_matches": main_conformance[
            "pure_bc_policy_action_matches_fresh_cache"
        ],
        "ppo_matches": main_conformance["ppo_policy_action_matches_fresh_cache"],
        "model_artifact_written": False,
        "candidate_written": False,
        "submission_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--fresh-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run(args.cache_dir, args.fresh_profile, args.output)
    sys.stdout.buffer.write(canonical_json_bytes(result))


if __name__ == "__main__":
    main()

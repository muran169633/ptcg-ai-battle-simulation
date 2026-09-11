#!/usr/bin/env python3
"""Independently audit the frozen actor6 deployment-semantics v2 profile.

The audit has two deliberately separate checks over train-only data:

1. Run the hash-bound v2 profiler into a second, initially absent cache and
   compare a cache-location/statistics-independent semantic hash with the
   accepted profile.
2. Import the real hash-bound hybrid ``main.py`` once, inject the exact frozen
   expanded61 source model, and call both ``_policy_action`` and ``agent`` for
   every one of the 61,600 train decisions.  Both entrypoints must equal the
   freshly generated record for every row.

No checkpoint, candidate, evaluation split, upload, or submission is produced.
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
PROFILER = ROOT / "tools/profile_yanz_source_error_actor6_deployment_v2.py"


def _load_frozen_profiler() -> ModuleType:
    module_name = "ptcg_frozen_actor6_deployment_profiler_v2_for_audit"
    spec = importlib.util.spec_from_file_location(module_name, PROFILER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct frozen profiler import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


profiler = _load_frozen_profiler()


ACCEPTED_PROFILE = ROOT / (
    "data/yanz_alakazam_actor6_pcgrad_specialbc_20260810_v2_"
    "deployment_policyboundary/train_only_profile.json"
)

SCHEMA = "ptcg-yanz-actor6-deployment-v2-independent-audit-v1"
PROFILER_SHA256 = (
    "386d23e280f42225b1752e987b1eb92df8b30b41a1be5ebbd9260b8f6388927e"
)
ACCEPTED_PROFILE_SHA256 = (
    "87891f8985d5f23b8114e5aff11426a4c80c717dc2782eae4cb9cd38d71eb529"
)
EXPECTED_TOTAL_ROWS = profiler.EXPECTED_YANZ_RAW_ROWS + profiler.EXPECTED_OLD_RAW_ROWS
EXPECTED_SHARDS_BY_SOURCE = {"yanz": 4, "old": 118}
EXPECTED_SHARDS = sum(EXPECTED_SHARDS_BY_SOURCE.values())
SEMANTIC_HASH_ALGORITHM = (
    "sha256(canonical-json-v1 after removing cache shard absolute path and "
    "source cache_hits/cache_writes)"
)


class AuditError(RuntimeError):
    """A fail-closed independent-audit contract violation."""


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
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise AuditError(f"duplicate JSON key {key!r}: {path}")
            output[key] = value
        return output

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


def assert_file_sha(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise AuditError(f"{label} SHA drift: {actual} != {expected}")


def validate_frozen_bindings() -> dict[str, Any]:
    assert_file_sha(PROFILER, PROFILER_SHA256, "profiler")
    assert_file_sha(ACCEPTED_PROFILE, ACCEPTED_PROFILE_SHA256, "accepted profile")
    static = profiler.validate_static_bindings()
    expected = {
        "source_checkpoint": profiler.SOURCE_SHA256,
        "yanz_train_archive": profiler.YANZ_SHA256,
        "old_train_archive": profiler.OLD_SHA256,
        "hybrid_main": profiler.HYBRID_MAIN_SHA256,
        "policy_runtime": profiler.RUNTIME_SHA256,
        "template_contract": profiler.CONTRACT_SHA256,
        "deck_csv": profiler.DECK_CSV_SHA256,
    }
    for key, sha256 in expected.items():
        binding = static.get(key)
        if not isinstance(binding, Mapping) or binding.get("sha256") != sha256:
            raise AuditError(f"profiler static binding drift: {key}")
    return static


def validate_profile_contract(
    profile: Mapping[str, Any], *, accepted: bool
) -> None:
    if profile.get("schema_version") != profiler.SCHEMA:
        raise AuditError("profile schema drift")
    if profile.get("status") != "frozen_train_only_deployment_semantics_row_selection":
        raise AuditError("profile status drift")
    bindings = profile.get("bindings")
    if not isinstance(bindings, Mapping):
        raise AuditError("profile bindings missing")
    profile_profiler = bindings.get("profiler")
    if not isinstance(profile_profiler, Mapping) or profile_profiler.get(
        "sha256"
    ) != PROFILER_SHA256:
        raise AuditError("profile profiler binding drift")
    expected = {
        "source_checkpoint": profiler.SOURCE_SHA256,
        "yanz_train_archive": profiler.YANZ_SHA256,
        "old_train_archive": profiler.OLD_SHA256,
        "hybrid_main": profiler.HYBRID_MAIN_SHA256,
        "policy_runtime": profiler.RUNTIME_SHA256,
        "template_contract": profiler.CONTRACT_SHA256,
        "deck_csv": profiler.DECK_CSV_SHA256,
    }
    for key, sha256 in expected.items():
        value = bindings.get(key)
        if not isinstance(value, Mapping) or value.get("sha256") != sha256:
            raise AuditError(f"profile frozen binding drift: {key}")
    counts = profile.get("source_counts")
    if not isinstance(counts, Mapping):
        raise AuditError("profile source counts missing")
    expected_counts = {
        ("yanz", "raw_rows"): profiler.EXPECTED_YANZ_RAW_ROWS,
        ("old", "raw_rows"): profiler.EXPECTED_OLD_RAW_ROWS,
        ("source_counts", "treatment_selection_only"): 302,
        ("source_counts", "retention_selected"): 467,
    }
    for (section, key), expected_count in expected_counts.items():
        container: Mapping[str, Any]
        if section == "source_counts":
            container = counts
        else:
            value = counts.get(section)
            if not isinstance(value, Mapping):
                raise AuditError(f"profile count section missing: {section}")
            container = value
        if int(container.get(key, -1)) != expected_count:
            raise AuditError(f"profile count drift: {section}.{key}")
    cache = profile.get("cache")
    if not isinstance(cache, Mapping):
        raise AuditError("profile cache manifest missing")
    shards = cache.get("immutable_shards")
    if not isinstance(shards, list) or len(shards) != EXPECTED_SHARDS:
        raise AuditError("profile cache shard count drift")
    if int(cache.get("shard_count", -1)) != EXPECTED_SHARDS:
        raise AuditError("profile cache declared shard count drift")
    if accepted and file_sha256(ACCEPTED_PROFILE) != ACCEPTED_PROFILE_SHA256:
        raise AuditError("accepted profile changed during validation")


def semantic_projection(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Drop only execution-local cache fields, retaining all semantic fields."""

    projected = copy.deepcopy(dict(profile))
    counts = projected.get("source_counts")
    if not isinstance(counts, dict):
        raise AuditError("semantic projection source counts missing")
    for source in ("yanz", "old"):
        source_counts = counts.get(source)
        if not isinstance(source_counts, dict):
            raise AuditError(f"semantic projection source missing: {source}")
        for key in ("cache_hits", "cache_writes"):
            if key not in source_counts:
                raise AuditError(f"semantic projection field missing: {source}.{key}")
            source_counts.pop(key)
    cache = projected.get("cache")
    if not isinstance(cache, dict):
        raise AuditError("semantic projection cache missing")
    shards = cache.get("immutable_shards")
    if not isinstance(shards, list):
        raise AuditError("semantic projection shard manifest missing")
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or "path" not in shard:
            raise AuditError(f"semantic projection shard path missing: {index}")
        shard.pop("path")
    return projected


def semantic_profile_sha256(profile: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(semantic_projection(profile))).hexdigest()


def cache_manifest_projection(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    cache = profile.get("cache")
    if not isinstance(cache, Mapping):
        raise AuditError("cache manifest missing")
    shards = cache.get("immutable_shards")
    if not isinstance(shards, list):
        raise AuditError("cache shards missing")
    projected: list[dict[str, Any]] = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise AuditError("cache shard is not an object")
        item = dict(shard)
        if not isinstance(item.pop("path", None), str):
            raise AuditError("cache shard path missing")
        projected.append(item)
    return projected


def assert_fresh_targets(cache_dir: Path, fresh_profile: Path, output: Path) -> None:
    if cache_dir.exists():
        raise FileExistsError(
            f"second cache must be initially absent, found: {cache_dir}"
        )
    for label, path in (("fresh profile", fresh_profile), ("audit output", output)):
        if path.exists():
            raise FileExistsError(f"{label} already exists: {path}")
    accepted_cache = ACCEPTED_PROFILE.parent / "cache"
    if cache_dir.resolve() == accepted_cache.resolve():
        raise AuditError("fresh cache cannot be the accepted cache")


def verify_manifest_files(
    profile: Mapping[str, Any], *, required_root: Path | None
) -> dict[str, Any]:
    cache = profile["cache"]
    shards = cache["immutable_shards"]
    rows = 0
    by_source: Counter[str] = Counter()
    manifest_digest = hashlib.sha256()
    manifest_digest.update(b"ptcg-cache-manifest-semantic-v1\0")
    for item in shards:
        path = Path(str(item["path"])).resolve()
        if required_root is not None:
            try:
                path.relative_to(required_root.resolve())
            except ValueError as error:
                raise AuditError(f"fresh shard escaped second cache: {path}") from error
        actual = file_sha256(path)
        if actual != item.get("sha256"):
            raise AuditError(f"cache shard SHA drift: {path}")
        count = int(item.get("record_count", -1))
        if count < 1:
            raise AuditError(f"invalid cache shard record count: {path}")
        rows += count
        by_source[str(item.get("source"))] += count
        semantic_item = dict(item)
        semantic_item.pop("path", None)
        manifest_digest.update(canonical_json_bytes(semantic_item))
    if rows != EXPECTED_TOTAL_ROWS:
        raise AuditError(f"cache manifest row drift: {rows}")
    expected_by_source = {
        "yanz": profiler.EXPECTED_YANZ_RAW_ROWS,
        "old": profiler.EXPECTED_OLD_RAW_ROWS,
    }
    if dict(by_source) != expected_by_source:
        raise AuditError(f"cache manifest source counts drift: {dict(by_source)}")
    return {
        "shards": len(shards),
        "rows": rows,
        "by_source": dict(sorted(by_source.items())),
        "semantic_manifest_sha256": manifest_digest.hexdigest(),
    }


def load_fresh_expected_actions(
    profile: Mapping[str, Any], cache_dir: Path
) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    declared_cache_bindings = profile["bindings"]["cache_contract"]
    for item in profile["cache"]["immutable_shards"]:
        shard = Path(str(item["path"])).resolve()
        try:
            shard.relative_to(cache_dir.resolve())
        except ValueError as error:
            raise AuditError(f"fresh shard escaped second cache: {shard}") from error
        payload = load_json_strict(shard)
        if payload.get("schema_version") != profiler.CACHE_SCHEMA:
            raise AuditError(f"fresh cache schema drift: {shard}")
        if payload.get("status") != "complete_immutable_train_chunk":
            raise AuditError(f"fresh cache status drift: {shard}")
        if payload.get("cache_bindings") != declared_cache_bindings:
            raise AuditError(f"fresh cache binding drift: {shard}")
        rows = payload.get("records")
        if not isinstance(rows, list) or int(payload.get("record_count", -1)) != len(
            rows
        ):
            raise AuditError(f"fresh cache record list drift: {shard}")
        records_sha = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
        if records_sha != payload.get("records_sha256"):
            raise AuditError(f"fresh cache records SHA drift: {shard}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise AuditError(f"fresh cache record is not an object: {shard}")
            source = str(row.get("source", ""))
            line_sha = str(row.get("line_sha256", ""))
            action = row.get("source_hybrid_action")
            context = row.get("context")
            if source not in {"yanz", "old"} or len(line_sha) != 64:
                raise AuditError(f"fresh cache row identity drift: {shard}")
            if not isinstance(action, list) or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in action
            ):
                raise AuditError(f"fresh cache action drift: {shard}")
            if not isinstance(context, int) or isinstance(context, bool):
                raise AuditError(f"fresh cache context drift: {shard}")
            key = (source, line_sha)
            if key in records:
                raise AuditError(f"duplicate fresh cache row: {source}:{line_sha}")
            records[key] = {
                "action": list(action),
                "context": int(context),
            }
    if len(records) != EXPECTED_TOTAL_ROWS:
        raise AuditError(f"fresh cache unique record drift: {len(records)}")
    return records


def iter_train_observations(
    archive_path: Path, source: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Independent reader that never opens a non-train archive member."""

    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not members:
            raise AuditError(f"no train members in {archive_path}")
        for member in members:
            with archive.open(member) as handle:
                for line in handle:
                    value = orjson.loads(line)
                    if not isinstance(value, dict):
                        raise AuditError(f"non-object train row: {source}:{member}")
                    if value.get("split") != "train":
                        raise AuditError(f"non-train row in train member: {source}:{member}")
                    if value.get("deck_hash") != profiler.DECK_HASH:
                        raise AuditError(f"deck drift: {source}:{member}")
                    observation = value.get("observation")
                    if not isinstance(observation, dict) or not isinstance(
                        observation.get("select"), dict
                    ):
                        raise AuditError(f"selection observation missing: {source}:{member}")
                    yield hashlib.sha256(line.rstrip(b"\r\n")).hexdigest(), observation


def load_actual_hybrid_main(
    runtime: ModuleType,
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> ModuleType:
    """Import the frozen real main once and inject the exact expanded61 source."""

    assert_file_sha(profiler.HYBRID_MAIN, profiler.HYBRID_MAIN_SHA256, "hybrid main")
    assert_file_sha(profiler.RUNTIME, profiler.RUNTIME_SHA256, "policy runtime")
    assert_file_sha(
        profiler.MAIN_RUNTIME_ANCHOR, profiler.RUNTIME_SHA256, "main runtime anchor"
    )
    anchor_deck = profiler.MAIN_RUNTIME_ANCHOR.parent / "deck.csv"
    assert_file_sha(anchor_deck, profiler.DECK_CSV_SHA256, "main deck anchor")
    module_name = "ptcg_actor6_deployment_v2_independent_audit_main"
    if module_name in sys.modules:
        raise AuditError("actual hybrid main audit module was already imported")
    spec = importlib.util.spec_from_file_location(module_name, profiler.HYBRID_MAIN)
    if spec is None or spec.loader is None:
        raise AuditError("cannot construct actual hybrid main import")
    module = importlib.util.module_from_spec(spec)
    previous_alias = sys.modules.get("policy_runtime")
    previous_runtime_file = runtime.__file__
    try:
        runtime.__file__ = str(profiler.MAIN_RUNTIME_ANCHOR)
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
        raise AuditError("actual main did not retain injected expanded61 model")
    if module.DECK_HASH != profiler.DECK_HASH or module.SKILL_ORDER_CONTEXT != 34:
        raise AuditError("actual main constants drift")
    if module._policy_action.__code__.co_filename != str(profiler.HYBRID_MAIN):
        raise AuditError("_policy_action is not executing the frozen main.py")
    if module.agent.__code__.co_filename != str(profiler.HYBRID_MAIN):
        raise AuditError("agent is not executing the frozen main.py")
    return module


def run_full_main_conformance(
    profile: Mapping[str, Any], cache_dir: Path
) -> dict[str, Any]:
    expected = load_fresh_expected_actions(profile, cache_dir)
    runtime = profiler.load_bound_runtime()
    _, model, config, model_audit = profiler.instantiate_deployment_source(runtime)
    main_module = load_actual_hybrid_main(runtime, model, config)
    deck_action = main_module.agent({})
    if deck_action != list(main_module.DECK) or len(deck_action) != 60:
        raise AuditError("actual agent deck-return branch drift")

    by_source: Counter[str] = Counter()
    by_context: Counter[int] = Counter()
    seen: set[tuple[str, str]] = set()
    action_digest = hashlib.sha256()
    action_digest.update(b"ptcg-real-main-policy-agent-conformance-v1\0")
    sources = (("yanz", profiler.YANZ), ("old", profiler.OLD))
    with torch.inference_mode():
        for source, archive in sources:
            for line_sha, observation in iter_train_observations(archive, source):
                key = (source, line_sha)
                if key in seen:
                    raise AuditError(f"duplicate source row: {source}:{line_sha}")
                record = expected.get(key)
                if record is None:
                    raise AuditError(f"fresh record missing: {source}:{line_sha}")
                policy_action = main_module._policy_action(dict(observation))
                agent_action = main_module.agent(dict(observation))
                wanted = record["action"]
                if policy_action != wanted:
                    raise AuditError(f"_policy_action mismatch: {source}:{line_sha}")
                if agent_action != wanted:
                    raise AuditError(f"agent mismatch: {source}:{line_sha}")
                if policy_action != agent_action:
                    raise AuditError(f"entrypoint disagreement: {source}:{line_sha}")
                context_raw = observation["select"].get("context", -1)
                context = -1 if context_raw is None else int(context_raw)
                if context != record["context"]:
                    raise AuditError(f"context disagreement: {source}:{line_sha}")
                action_digest.update(source.encode("ascii"))
                action_digest.update(b"\0")
                action_digest.update(line_sha.encode("ascii"))
                action_digest.update(b"\0")
                action_digest.update(canonical_json_bytes(policy_action))
                seen.add(key)
                by_source[source] += 1
                by_context[context] += 1
    if seen != set(expected):
        raise AuditError(f"source/fresh cache identity set mismatch: {len(seen)}")
    expected_sources = {
        "old": profiler.EXPECTED_OLD_RAW_ROWS,
        "yanz": profiler.EXPECTED_YANZ_RAW_ROWS,
    }
    if dict(sorted(by_source.items())) != expected_sources:
        raise AuditError(f"actual-main source count drift: {dict(by_source)}")
    return {
        "rows_checked": len(seen),
        "policy_action_matches_fresh_record": len(seen),
        "agent_matches_fresh_record": len(seen),
        "policy_action_matches_agent": len(seen),
        "by_source": dict(sorted(by_source.items())),
        "by_context": {str(key): value for key, value in sorted(by_context.items())},
        "deck_branch_matches_bound_deck": True,
        "deck_branch_cards": len(deck_action),
        "real_main_action_ledger_sha256": action_digest.hexdigest(),
        "hybrid_main_policy_action_code_file": str(
            Path(main_module._policy_action.__code__.co_filename).resolve()
        ),
        "hybrid_main_agent_code_file": str(
            Path(main_module.agent.__code__.co_filename).resolve()
        ),
        "expanded_source": model_audit,
    }


def run(cache_dir: Path, fresh_profile_path: Path, output: Path) -> dict[str, Any]:
    assert_fresh_targets(cache_dir, fresh_profile_path, output)
    startup_script_sha = file_sha256(SCRIPT)
    static = validate_frozen_bindings()
    accepted = load_json_strict(ACCEPTED_PROFILE)
    validate_profile_contract(accepted, accepted=True)
    accepted_manifest = verify_manifest_files(accepted, required_root=None)

    build_result = profiler.build(cache_dir, fresh_profile_path)
    if int(build_result.get("cache_shards", -1)) != EXPECTED_SHARDS:
        raise AuditError("fresh profiler build shard count drift")
    fresh = load_json_strict(fresh_profile_path)
    validate_profile_contract(fresh, accepted=False)
    fresh_counts = fresh["source_counts"]
    expected_writes = dict(EXPECTED_SHARDS_BY_SOURCE)
    for source, writes in expected_writes.items():
        if int(fresh_counts[source].get("cache_hits", -1)) != 0:
            raise AuditError(f"second cache was not empty for {source}")
        if int(fresh_counts[source].get("cache_writes", -1)) != writes:
            raise AuditError(f"second cache write count drift for {source}")

    accepted_semantic = semantic_profile_sha256(accepted)
    fresh_semantic = semantic_profile_sha256(fresh)
    if semantic_projection(accepted) != semantic_projection(fresh):
        raise AuditError("fresh profile semantic projection differs from accepted")
    if accepted_semantic != fresh_semantic:
        raise AuditError("fresh profile semantic hash differs from accepted")
    accepted_manifest_projection = cache_manifest_projection(accepted)
    fresh_manifest_projection = cache_manifest_projection(fresh)
    if accepted_manifest_projection != fresh_manifest_projection:
        raise AuditError("fresh cache semantic manifest differs from accepted")
    fresh_manifest = verify_manifest_files(fresh, required_root=cache_dir)
    if accepted_manifest["semantic_manifest_sha256"] != fresh_manifest[
        "semantic_manifest_sha256"
    ]:
        raise AuditError("fresh cache semantic manifest hash differs from accepted")

    conformance = run_full_main_conformance(fresh, cache_dir)
    if file_sha256(SCRIPT) != startup_script_sha:
        raise AuditError("audit script changed during run")
    validate_frozen_bindings()
    if file_sha256(fresh_profile_path) != build_result.get("sha256"):
        raise AuditError("fresh profile changed after build")

    audit = {
        "schema_version": SCHEMA,
        "status": "passed_full_train_fresh_rerun_and_real_main_conformance",
        "bindings": {
            "audit_tool": {
                "path": str(SCRIPT.resolve()),
                "sha256": startup_script_sha,
            },
            "profiler": {
                "path": str(PROFILER.resolve()),
                "sha256": PROFILER_SHA256,
            },
            "accepted_profile": {
                "path": str(ACCEPTED_PROFILE.resolve()),
                "sha256": ACCEPTED_PROFILE_SHA256,
            },
            "fresh_profile": {
                "path": str(fresh_profile_path.resolve()),
                "sha256": file_sha256(fresh_profile_path),
            },
            "source_checkpoint": static["source_checkpoint"],
            "hybrid_main": static["hybrid_main"],
            "policy_runtime": static["policy_runtime"],
            "yanz_train_archive": static["yanz_train_archive"],
            "old_train_archive": static["old_train_archive"],
            "template_contract": static["template_contract"],
            "deck_csv": static["deck_csv"],
            "runtime_semantics": static["runtime_semantics"],
        },
        "fresh_rerun": {
            "second_cache_initially_absent": True,
            "second_cache": str(cache_dir.resolve()),
            "cache_hits": {"yanz": 0, "old": 0},
            "cache_writes": expected_writes,
            "accepted_profile_raw_sha256": ACCEPTED_PROFILE_SHA256,
            "fresh_profile_raw_sha256": file_sha256(fresh_profile_path),
            "semantic_hash_algorithm": SEMANTIC_HASH_ALGORITHM,
            "semantic_hash_excluded_fields": [
                "cache.immutable_shards[*].path",
                "source_counts.yanz.cache_hits",
                "source_counts.yanz.cache_writes",
                "source_counts.old.cache_hits",
                "source_counts.old.cache_writes",
            ],
            "accepted_semantic_sha256": accepted_semantic,
            "fresh_semantic_sha256": fresh_semantic,
            "semantic_projection_exact_match": True,
            "accepted_cache_manifest": accepted_manifest,
            "fresh_cache_manifest": fresh_manifest,
            "cache_manifest_projection_exact_match": True,
        },
        "real_hybrid_main_conformance": conformance,
        "scope": {
            "opened_archive_members": "train/*.jsonl only",
            "holdout_opened": False,
            "model_artifact_written": False,
            "training_performed": False,
            "candidate_written": False,
            "submission_performed": False,
        },
    }
    if file_sha256(SCRIPT) != startup_script_sha:
        raise AuditError("audit script changed before output")
    write_new_json(audit, output)
    if file_sha256(SCRIPT) != startup_script_sha:
        raise AuditError("audit script changed after output")
    return {
        "status": audit["status"],
        "output": str(output.resolve()),
        "sha256": file_sha256(output),
        "fresh_profile": str(fresh_profile_path.resolve()),
        "fresh_profile_sha256": file_sha256(fresh_profile_path),
        "semantic_sha256": fresh_semantic,
        "rows_checked": conformance["rows_checked"],
        "policy_agent_matches": conformance["policy_action_matches_agent"],
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
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

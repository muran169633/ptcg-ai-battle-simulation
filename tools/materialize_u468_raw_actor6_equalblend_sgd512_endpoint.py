#!/usr/bin/env python3
"""Materialize the authorized raw-U468 equal-blend SGD512 endpoint once.

This runner authenticates the completed GO shadow chain before importing its
implementation.  It also authenticates the separately frozen full-train
promotion-gate tool and its unarmed contract, which must predate endpoint
materialization.  Static, cache, and materialization-contract modes are
strictly zero-write.  The CUDA-only actual mode requires an exact external
O_EXCL preregistration and consumes a one-shot O_EXCL attempt marker before a
model or optimizer exists.  It then reproduces the same direct512 update once,
requires the complete training/evaluation/displacement record to equal the
shadow evidence, and publishes exactly one evaluation-only, resume-forbidden
checkpoint plus an immutable manifest and result seal.  It has no validation,
submission, or resumable-training path.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

SHADOW_RUNNER = TOOLS / "run_u468_raw_actor6_equalblend_sgd512_shadow.py"
SHADOW_RUNNER_SHA256 = (
    "e04f7b7579ef42d0c6f643db833779fb837ae86e3205d948b5564d08f8b30f8e"
)
SHADOW_RESULT = ROOT / (
    "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_shadow_"
    "design202608115.result.json"
)
SHADOW_RESULT_SHA256 = (
    "a35d2050e92a9f0035847d3438241555573c053c4f99c0c4269a39bcd80bfe45"
)
SHADOW_PREREGISTRATION = ROOT / (
    "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_shadow_"
    "design202608115.preregistration.json"
)
SHADOW_PREREGISTRATION_SHA256 = (
    "961ed51abcc5e502f603e5fc73385c64f9bc7fd0408154825920ed48002cb6da"
)
SHADOW_ATTEMPT_MARKER = ROOT / (
    ".ptcg-ppo_u468_raw_actor6_equalblend_sgd512_shadow_"
    "design202608115-attempt.json"
)
SHADOW_ATTEMPT_MARKER_SHA256 = (
    "42a14c657366cc51d104954dd00f1774af859850ef75ce304c629438ab22cbde"
)
PROMOTION_GATE_TOOL = TOOLS / "run_u468_raw_equalblend_train_only_promotion_gate.py"
PROMOTION_GATE_TOOL_SHA256 = (
    "4f36bd0f86cbb811d8142f155027445b4bfb43a1b3e55b2fbf3b5787981aeae0"
)
PROMOTION_GATE_CONTRACT = ROOT / (
    "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_contract_"
    "design202608116.json"
)
PROMOTION_GATE_CONTRACT_SHA256 = (
    "e0d80b72aed237c3de4b643157b058acf2bff513d11420dd7579db0308b970c2"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(
    path: Path,
    expected_sha256: str | None,
    label: str,
    *,
    expected_mode: int | None = None,
) -> dict[str, Any]:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise RuntimeError(f"{label} is not a single-link regular file")
    mode = observed.st_mode & 0o777
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": oct(mode),
        "nlink": observed.st_nlink,
    }


# The shadow runner is authenticated before any of its code executes here.
SHADOW_RUNNER_EVIDENCE = require_regular(
    SHADOW_RUNNER,
    SHADOW_RUNNER_SHA256,
    "frozen equalblend shadow runner",
    expected_mode=0o555,
)
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_SHADOW_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_equalblend_shadow_e04f7b75", SHADOW_RUNNER
)
if _SHADOW_SPEC is None or _SHADOW_SPEC.loader is None:
    raise RuntimeError("cannot construct frozen shadow-runner import spec")
shadow: ModuleType = importlib.util.module_from_spec(_SHADOW_SPEC)
_SHADOW_SPEC.loader.exec_module(shadow)

# The contract-only gate tool is likewise authenticated before its code runs.
PROMOTION_GATE_TOOL_EVIDENCE = require_regular(
    PROMOTION_GATE_TOOL,
    PROMOTION_GATE_TOOL_SHA256,
    "frozen full-train promotion-gate tool",
    expected_mode=0o555,
)
_GATE_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_equalblend_promotion_gate_4f36bd0f", PROMOTION_GATE_TOOL
)
if _GATE_SPEC is None or _GATE_SPEC.loader is None:
    raise RuntimeError("cannot construct frozen promotion-gate import spec")
promotion_gate: ModuleType = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(promotion_gate)

torch = shadow.torch
ppo = shadow.ppo
repair = shadow.repair
frozen = shadow.frozen
aggregate = shadow.aggregate


SCHEMA = "ptcg-u468-raw-actor6-equalblend-sgd512-materializer-v1"
PREREGISTRATION_SCHEMA = (
    "ptcg-u468-raw-actor6-equalblend-sgd512-materialization-preregistration-v1"
)
ATTEMPT_SCHEMA = (
    "ptcg-u468-raw-actor6-equalblend-sgd512-materialization-attempt-v1"
)
MANIFEST_SCHEMA = (
    "ptcg-u468-raw-actor6-equalblend-sgd512-materialization-manifest-v1"
)
RESULT_SCHEMA = (
    "ptcg-u468-raw-actor6-equalblend-sgd512-materialization-result-v1"
)
BRANCH = "ppo_u468_raw_actor6_equalblend_sgd512_materialize_design202608116"
EXECUTION_SEED = 202608115
PREREGISTRATION = ROOT / f"artifacts/{BRANCH}.preregistration.json"
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"
OUTPUT_ROOT = ROOT / "artifacts" / BRANCH
ENDPOINT = OUTPUT_ROOT / "equalblend-sgd512-eval-only.pt"
MANIFEST = OUTPUT_ROOT / "materialization_manifest.json"
RESULT = OUTPUT_ROOT / "materialization_result.json"

EXPECTED_DIRECT_CACHE_SHA256 = shadow.EXPECTED_DIRECT_CACHE_SHA256
EXPECTED_DIRECT_BATCH_SHA256 = shadow.EXPECTED_DIRECT_BATCH_SHA256
EXPECTED_FLAT_IDENTITY_SHA256 = shadow.EXPECTED_FLAT_IDENTITY_SHA256
EXPECTED_EQUALBLEND_GRADIENT_VECTOR_SHA256_FLOAT64 = (
    "47521ab0de80e6ec25f88a5704a7d9c185edf45197b25031e8b0c3e08be59e36"
)
ACTOR_NAMES = tuple(shadow.ACTOR_NAMES)
REQUIRED_SLIM_KEYS = (
    "feature_version",
    "bc_feature_version",
    "config",
    "model_config",
    "learner_deck_hash",
    "reward",
    "action_distribution",
)
FORBIDDEN_RESUME_KEYS = {
    "optimizer_state_dict",
    "bc_replay_optimizer_state_dict",
    "opponent_quota_state",
    "fresh_special_optimizer_state_dict",
}
EXPECTED_SHADOW_GATE_KEYS = {
    "aggregate_retention_improves_at_least_1e-6",
    "all_finite",
    "changed_scope_exact_actor6_frozen_exact",
    "critical_line_pinned_correct",
    "displacement_l2_in_closed_interval",
    "displacement_max_abs_at_most_3e-6",
    "endpoint_retention_288_context34_6_zero_flips",
    "gradient_scope_exact_actor6_all_six_nonzero",
    "one_forward_four_scalars_one_backward_clip_step",
    "plain_sgd_state_empty",
    "three_hard_each_improve_at_least_1e-5",
    "three_source_all_each_improve_at_least_1e-8",
    "three_source_retention_each_improve_at_least_1e-8",
    "union_improves_at_least_1e-5",
}


def canonical_json(value: Any) -> bytes:
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise RuntimeError(f"{label} contains nonfinite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")


def normalize_repo_path(path: str | Path, label: str) -> Path:
    result = Path(path)
    if not result.is_absolute():
        result = ROOT / result
    result = Path(os.path.abspath(os.fspath(result)))
    try:
        result.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"{label} must remain inside repository") from error
    return result


def require_absent(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    if not path.parent.is_dir() or path.parent.resolve() != path.parent:
        raise RuntimeError(f"{label} parent must be an existing real directory")


def publish_o_excl(path: Path, payload: bytes, mode: int) -> dict[str, Any]:
    path = normalize_repo_path(path, "publication target")
    require_absent(path, "publication target")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(fd, view[offset:])
            if count <= 0:
                raise RuntimeError("short O_EXCL publication write")
            offset += count
        os.fsync(fd)
        observed = os.fstat(fd)
        visible = os.lstat(path)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_ISLNK(visible.st_mode)
            or (observed.st_dev, observed.st_ino, observed.st_size)
            != (visible.st_dev, visible.st_ino, visible.st_size)
            or observed.st_size != len(payload)
        ):
            raise RuntimeError("unsafe O_EXCL publication")
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        reloaded = b"".join(chunks)
        if reloaded != payload:
            raise RuntimeError("descriptor-held publication reload drift")
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_bytes(reloaded),
            "bytes": len(reloaded),
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mode": oct(observed.st_mode & 0o777),
            "nlink": observed.st_nlink,
            "descriptor_held_exact_reload": True,
        }
    finally:
        os.close(fd)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def self_evidence() -> dict[str, Any]:
    return require_regular(
        normalize_repo_path(__file__, "materializer runner"),
        None,
        "materializer runner",
    )


def verify_shadow_chain() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence = {
        "runner": require_regular(
            SHADOW_RUNNER,
            SHADOW_RUNNER_SHA256,
            "shadow runner",
            expected_mode=0o555,
        ),
        "result": require_regular(
            SHADOW_RESULT,
            SHADOW_RESULT_SHA256,
            "shadow result",
            expected_mode=0o444,
        ),
        "preregistration": require_regular(
            SHADOW_PREREGISTRATION,
            SHADOW_PREREGISTRATION_SHA256,
            "shadow preregistration",
            expected_mode=0o444,
        ),
        "attempt_marker": require_regular(
            SHADOW_ATTEMPT_MARKER,
            SHADOW_ATTEMPT_MARKER_SHA256,
            "shadow attempt marker",
            expected_mode=0o444,
        ),
        "v2_probe_result": require_regular(
            shadow.V2_PROBE_RESULT,
            shadow.V2_PROBE_RESULT_SHA256,
            "v2 probe result authorized by shadow",
            expected_mode=0o444,
        ),
        "v2_probe_tool": require_regular(
            shadow.V2_PROBE_TOOL,
            shadow.V2_PROBE_TOOL_SHA256,
            "v2 probe tool authorized by shadow",
            expected_mode=0o555,
        ),
    }
    if evidence["runner"]["sha256"] != SHADOW_RUNNER_EVIDENCE["sha256"]:
        raise RuntimeError("shadow runner changed after hash-bound import")
    result = strict_json(SHADOW_RESULT, "shadow result")
    preregistration = strict_json(SHADOW_PREREGISTRATION, "shadow preregistration")
    marker = strict_json(SHADOW_ATTEMPT_MARKER, "shadow attempt marker")
    probe_result = strict_json(shadow.V2_PROBE_RESULT, "v2 probe result")
    probe_evidence, probe_authorization = shadow.require_probe_authorization()
    candidate = result.get("candidate")
    contract = result.get("shadow_contract")
    gates = candidate.get("gate_checks") if isinstance(candidate, dict) else None
    probe_payload = probe_result.get("probe")
    selected_direction = (
        probe_payload.get("selection", {}).get("selected_direction")
        if isinstance(probe_payload, dict)
        else None
    )
    selected_direction_report = (
        probe_payload.get("directions", {}).get(selected_direction)
        if isinstance(probe_payload, dict)
        else None
    )
    if (
        result.get("schema_version") != shadow.SCHEMA
        or result.get("status")
        != "completed_actual_ram_only_equalblend_sgd512_step"
        or result.get("decision") != "GO"
        or result.get("single_candidate_only") is not True
        or not isinstance(candidate, dict)
        or candidate.get("candidate_count") != 1
        or candidate.get("fully_passes") is not True
        or not isinstance(gates, dict)
        or set(gates) != EXPECTED_SHADOW_GATE_KEYS
        or not all(value is True for value in gates.values())
        or result.get("validation_member_payloads_opened") is not False
        or result.get("checkpoint_writes") != 0
        or result.get("model_artifact_writes") != 0
        or result.get("optimizer_artifact_writes") != 0
        or result.get("training_artifact_writes") != 0
        or result.get("submission_performed") is not False
        or result.get("runner", {}).get("path")
        != str(SHADOW_RUNNER.relative_to(ROOT))
        or result.get("runner", {}).get("sha256") != SHADOW_RUNNER_SHA256
        or result.get("preregistration", {}).get("path")
        != str(SHADOW_PREREGISTRATION.relative_to(ROOT))
        or result.get("preregistration", {}).get("sha256")
        != SHADOW_PREREGISTRATION_SHA256
        or result.get("attempt_marker", {}).get("path")
        != str(SHADOW_ATTEMPT_MARKER.relative_to(ROOT))
        or result.get("attempt_marker", {}).get("sha256")
        != SHADOW_ATTEMPT_MARKER_SHA256
        or result.get("probe", {}).get("path")
        != str(shadow.V2_PROBE_RESULT.relative_to(ROOT))
        or result.get("probe", {}).get("sha256")
        != shadow.V2_PROBE_RESULT_SHA256
        or result.get("probe_authorization") != probe_authorization
        or not isinstance(contract, dict)
        or contract.get("runner", {}).get("sha256") != SHADOW_RUNNER_SHA256
        or contract.get("result_output") != str(SHADOW_RESULT.relative_to(ROOT))
        or contract.get("attempt_marker")
        != str(SHADOW_ATTEMPT_MARKER.relative_to(ROOT))
        or contract.get("scope", {}).get("validation") is not False
        or contract.get("scope", {}).get("checkpoint_write") is not False
        or contract.get("scope", {}).get("model_artifact_write") is not False
        or contract.get("scope", {}).get("optimizer_artifact_write") is not False
        or contract.get("direct512", {}).get("cache_sha256")
        != EXPECTED_DIRECT_CACHE_SHA256
        or contract.get("direct512", {}).get("batch_sha256")
        != EXPECTED_DIRECT_BATCH_SHA256
        or contract.get("direct512", {}).get("flat_identity_sha256")
        != EXPECTED_FLAT_IDENTITY_SHA256
        or contract.get("v2_probe", {}).get("result", {}).get("sha256")
        != shadow.V2_PROBE_RESULT_SHA256
        or contract.get("v2_probe", {}).get("tool", {}).get("sha256")
        != shadow.V2_PROBE_TOOL_SHA256
        or contract.get("v2_probe", {}).get("authorization")
        != probe_authorization
        or candidate.get("training", {}).get("operation_counts")
        != {
            "native_bf16_gradient_forward_calls": 1,
            "complete_ordered_composite_scalar_calls": 4,
            "combined_scalar_backward_calls": 1,
            "gradient_clip_calls": 1,
            "optimizer_step_calls": 1,
        }
        or candidate.get("training", {}).get("optimizer_state_count_before_step")
        != 0
        or candidate.get("training", {}).get("optimizer_state_count_after_step")
        != 0
        or candidate.get("frozen_tensors_unchanged") is not True
        or candidate.get("retention_flips_from_raw", {}).get(
            "total_correct_to_wrong"
        )
        != 0
    ):
        raise RuntimeError("shadow result does not authorize materialization")
    if (
        probe_evidence["sha256"] != shadow.V2_PROBE_RESULT_SHA256
        or not isinstance(probe_payload, dict)
        or probe_payload.get("decision") != "DIRECTION_FOUND"
        or selected_direction != shadow.V2_PROBE_EXPECTED_SELECTED_DIRECTION
        or not isinstance(selected_direction_report, dict)
        or selected_direction_report.get("all_gate_objectives_robust_descent")
        is not True
        or selected_direction_report.get("direction", {}).get(
            "vector_sha256_float64"
        )
        != EXPECTED_EQUALBLEND_GRADIENT_VECTOR_SHA256_FLOAT64
    ):
        raise RuntimeError("v2 probe equalblend direction binding mismatch")
    if (
        preregistration.get("schema_version") != shadow.PREREGISTRATION_SCHEMA
        or preregistration.get("status") != "locked_before_actual"
        or preregistration.get("shadow_contract") != contract
        or marker.get("schema_version") != shadow.ATTEMPT_SCHEMA
        or marker.get("status")
        != "actual_attempt_consumed_before_model_or_optimizer_creation"
        or marker.get("shadow_contract") != contract
        or marker.get("runner", {}).get("sha256") != SHADOW_RUNNER_SHA256
        or marker.get("preregistration", {}).get("path")
        != str(SHADOW_PREREGISTRATION.relative_to(ROOT))
        or marker.get("preregistration", {}).get("sha256")
        != SHADOW_PREREGISTRATION_SHA256
        or marker.get("validation_member_payloads_opened") is not False
        or marker.get("checkpoint_writes_before_lock") != 0
    ):
        raise RuntimeError("shadow preregistration/attempt chain mismatch")

    reference_hashes = {
        "candidate_canonical_sha256": sha256_bytes(canonical_json(candidate)),
        "training_canonical_sha256": sha256_bytes(
            canonical_json(candidate["training"])
        ),
        "evaluations_canonical_sha256": sha256_bytes(
            canonical_json(candidate["evaluations"])
        ),
        "displacement_canonical_sha256": sha256_bytes(
            canonical_json(candidate["displacement_from_raw"])
        ),
        "improvements_canonical_sha256": sha256_bytes(
            canonical_json(candidate["improvements"])
        ),
    }
    authorization = {
        "status": result["status"],
        "decision": "GO",
        "single_candidate_only": True,
        "all_gate_checks_true": True,
        "validation_member_payloads_opened": False,
        "shadow_checkpoint_writes": 0,
        "shadow_model_artifact_writes": 0,
        "shadow_optimizer_artifact_writes": 0,
        "shadow_training_artifact_writes": 0,
        "v2_probe_result_sha256": shadow.V2_PROBE_RESULT_SHA256,
        "v2_probe_tool_sha256": shadow.V2_PROBE_TOOL_SHA256,
        "equalblend_gradient_vector_sha256_float64": (
            EXPECTED_EQUALBLEND_GRADIENT_VECTOR_SHA256_FLOAT64
        ),
        "reference_hashes": reference_hashes,
        "shadow_endpoint_model_state_sha256_present": False,
        "endpoint_state_identity_strategy": (
            "exact deterministic reproduction of the full shadow candidate "
            "record, then pre-serialization and reload model-state SHA equality"
        ),
    }
    return evidence, authorization, result


def verify_promotion_gate_precommit() -> dict[str, Any]:
    tool = require_regular(
        PROMOTION_GATE_TOOL,
        PROMOTION_GATE_TOOL_SHA256,
        "full-train promotion-gate tool",
        expected_mode=0o555,
    )
    if tool["sha256"] != PROMOTION_GATE_TOOL_EVIDENCE["sha256"]:
        raise RuntimeError("promotion-gate tool changed after hash-bound import")
    artifact = require_regular(
        PROMOTION_GATE_CONTRACT,
        PROMOTION_GATE_CONTRACT_SHA256,
        "frozen full-train promotion-gate contract",
        expected_mode=0o444,
    )
    payload = strict_json(PROMOTION_GATE_CONTRACT, "promotion-gate contract")
    generated = promotion_gate.promotion_contract()
    lineage = payload.get("lineage")
    sole_endpoint = (
        lineage.get("sole_endpoint") if isinstance(lineage, dict) else None
    )
    scope = payload.get("scope")
    arming = payload.get("arming_requirements")
    checkpoint_format = payload.get("checkpoint_format")
    if (
        payload != generated
        or payload.get("schema_version") != promotion_gate.SCHEMA
        or payload.get("status")
        != "contract_only_unarmed_endpoint_placeholders_required"
        or not isinstance(scope, dict)
        or scope.get("split") != "train"
        or scope.get("all_train_rows") is not True
        or scope.get("validation_member_payloads_opened") is not False
        or scope.get("validation_results_read") is not False
        or scope.get("candidate_evaluation_executed") is not False
        or scope.get("training") is not False
        or scope.get("package_upload_submission") is not False
        or not isinstance(arming, dict)
        or arming.get("gate_contract_must_be_frozen_before_endpoint_materialization")
        is not True
        or arming.get("actual_mode_in_this_revision") is not False
        or not isinstance(lineage, dict)
        or lineage.get("shadow_runner", {}).get("sha256")
        != SHADOW_RUNNER_SHA256
        or lineage.get("shadow_preregistration", {}).get("sha256")
        != SHADOW_PREREGISTRATION_SHA256
        or lineage.get("shadow_result", {}).get("sha256")
        != SHADOW_RESULT_SHA256
        or not isinstance(sole_endpoint, dict)
        or sole_endpoint
        != {
            "path": None,
            "file_sha256": None,
            "model_state_sha256": None,
            "generation_manifest_path": None,
            "generation_manifest_sha256": None,
            "all_frozen": False,
            "actual_enabled": False,
        }
        or not isinstance(checkpoint_format, dict)
        or checkpoint_format.get("kind") != "ppo_eval_only"
        or checkpoint_format.get("update") != 468
        or checkpoint_format.get("evaluation_only") is not True
        or checkpoint_format.get("resume_forbidden") is not True
        or checkpoint_format.get("changed_tensor_names_exact") != list(ACTOR_NAMES)
        or checkpoint_format.get("all_six_actor_tensors_must_change") is not True
        or checkpoint_format.get("all_other_tensors_bit_exact_parent") is not True
    ):
        raise RuntimeError("frozen promotion-gate precommit contract mismatch")
    return {
        "tool": tool,
        "frozen_contract_artifact": artifact,
        "contract_payload_canonical_sha256": sha256_bytes(canonical_json(payload)),
        "contract_schema_version": payload["schema_version"],
        "status": payload["status"],
        "frozen_before_endpoint_materialization": True,
        "endpoint_placeholders_unarmed": True,
        "actual_mode_in_contract_revision": False,
        "validation_member_payloads_opened": False,
        "validation_results_read": False,
    }


def verify_fixed_inputs() -> dict[str, Any]:
    chain, _, _ = verify_shadow_chain()
    return {
        "shadow_chain": chain,
        "promotion_gate_precommit": verify_promotion_gate_precommit(),
        "raw_u468": shadow.require_single_link_regular(
            frozen.U468, frozen.U468_SHA256, "raw U468"
        ),
        "general_bc": shadow.require_single_link_regular(
            frozen.GENERAL_BC, frozen.GENERAL_BC_SHA256, "general BC"
        ),
        "profile": shadow.require_single_link_regular(
            frozen.PROFILE, frozen.PROFILE_SHA256, "train profile"
        ),
        "datasets": {
            source: shadow.require_single_link_regular(
                path, frozen.DATA_SHA256[source], f"{source} train archive"
            )
            for source, path in frozen.DATASETS.items()
        },
    }


def output_absence() -> dict[str, bool]:
    return {
        "preregistration_absent": not (
            PREREGISTRATION.exists() or PREREGISTRATION.is_symlink()
        ),
        "attempt_marker_absent": not (
            ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink()
        ),
        "output_root_absent": not (OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink()),
        "endpoint_absent": not (ENDPOINT.exists() or ENDPOINT.is_symlink()),
        "manifest_absent": not (MANIFEST.exists() or MANIFEST.is_symlink()),
        "result_absent": not (RESULT.exists() or RESULT.is_symlink()),
    }


def require_outputs_absent(*, include_preregistration: bool) -> None:
    targets = [ATTEMPT_MARKER, OUTPUT_ROOT]
    if include_preregistration:
        targets.insert(0, PREREGISTRATION)
    for path in targets:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if ROOT.resolve() != ROOT or (ROOT / "artifacts").resolve() != ROOT / "artifacts":
        raise RuntimeError("root/artifacts path identity drift")


def ast_audit() -> dict[str, Any]:
    path = normalize_repo_path(__file__, "materializer runner")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [dotted(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    counts = {
        "shadow.execute_equalblend_step": calls.count(
            "shadow.execute_equalblend_step"
        ),
        "aggregate.evaluate_union": calls.count("aggregate.evaluate_union"),
        "torch.save": calls.count("torch.save"),
        "torch.optim.SGD": calls.count("torch.optim.SGD"),
        "torch.optim.AdamW": calls.count("torch.optim.AdamW"),
        "torch.autograd.grad": calls.count("torch.autograd.grad"),
        "loss.backward": calls.count("loss.backward"),
        "optimizer.step": calls.count("optimizer.step"),
        "publish_o_excl": calls.count("publish_o_excl"),
    }
    expected = {
        "shadow.execute_equalblend_step": 1,
        "aggregate.evaluate_union": 2,
        "torch.save": 1,
        "torch.optim.SGD": 0,
        "torch.optim.AdamW": 0,
        "torch.autograd.grad": 0,
        "loss.backward": 0,
        "optimizer.step": 0,
        "publish_o_excl": 4,
    }
    if counts != expected:
        raise RuntimeError(f"materializer source call-count drift: {counts}")
    if "SHADOW_RUNNER_EVIDENCE = require_regular" not in source:
        raise RuntimeError("shadow runner is not visibly verified before import")
    return {
        "status": "zero_write_static_ast_audit_passed",
        "observed_calls": counts,
        "expected_calls": expected,
        "delegated_update_is_exactly_one_hash_bound_shadow_step": True,
        "checkpoint_serialization_calls_exactly_one": True,
        "evidence_publications_exactly_marker_checkpoint_manifest_result": True,
        "validation_member_payloads_opened": False,
        "writes_performed": False,
    }


def build_cache_audit() -> tuple[
    dict[str, Any],
    list[list[dict[str, Any]]],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    audit, selections, union, masks = shadow.build_cache_and_objective_audit()
    if (
        audit["direct512"]["cache_sha256"] != EXPECTED_DIRECT_CACHE_SHA256
        or audit["direct512"]["batch_sha256"] != EXPECTED_DIRECT_BATCH_SHA256
        or audit["direct512"]["flat_identity_sha256"]
        != EXPECTED_FLAT_IDENTITY_SHA256
        or audit["direct512"]["rows"] != 512
        or audit["partition_audit"]["aggregate_retention_rows"] != 288
        or audit["validation_member_payloads_opened"] is not False
        or audit["writes_performed"] is not False
    ):
        raise RuntimeError("materialization direct512 cache contract drift")
    return audit, selections, union, masks


def cuda_contract() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for materialization contract/actual")
    index = torch.cuda.current_device()
    return {
        "device": "cuda",
        "cuda_device_index": index,
        "cuda_device_name": torch.cuda.get_device_name(index),
        "cuda_device_capability": list(torch.cuda.get_device_capability(index)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "high",
        "native_policy_logits_dtype": "torch.bfloat16",
    }


def expected_contract(
    runner: Mapping[str, Any],
    shadow_evidence: Mapping[str, Any],
    shadow_authorization: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    promotion_gate_precommit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "runner": {"path": runner["path"], "sha256": runner["sha256"]},
        "branch": BRANCH,
        "runtime": cuda_contract(),
        "execution_seed": EXECUTION_SEED,
        "shadow_authorization_chain": {
            "runner": {
                "path": shadow_evidence["runner"]["path"],
                "sha256": SHADOW_RUNNER_SHA256,
            },
            "result": {
                "path": shadow_evidence["result"]["path"],
                "sha256": SHADOW_RESULT_SHA256,
            },
            "preregistration": {
                "path": shadow_evidence["preregistration"]["path"],
                "sha256": SHADOW_PREREGISTRATION_SHA256,
            },
            "attempt_marker": {
                "path": shadow_evidence["attempt_marker"]["path"],
                "sha256": SHADOW_ATTEMPT_MARKER_SHA256,
            },
            "authorization": dict(shadow_authorization),
        },
        "promotion_gate_precommit": dict(promotion_gate_precommit),
        "training_base": {
            "kind": "raw_full_u468",
            "path": str(frozen.U468.relative_to(ROOT)),
            "checkpoint_sha256": frozen.U468_SHA256,
            "runtime_model_state_sha256": frozen.BASE_MODEL_SHA256,
            "update": 468,
        },
        "direct512": {
            "cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
            "batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
            "flat_identity_sha256": EXPECTED_FLAT_IDENTITY_SHA256,
            "rows": 512,
            "single_batch": True,
            "critical_line_sha256": shadow.CRITICAL_LINE_SHA256,
            "partition_audit": cache_audit["partition_audit"],
        },
        "reproduction": {
            "implementation": "one call to the exact SHA-bound shadow step",
            "native_bf16_gradient_forward_calls": 1,
            "complete_ordered_composite_scalar_calls": 4,
            "combined_scalar_backward_calls": 1,
            "gradient_clip_calls": 1,
            "plain_sgd_step_calls": 1,
            "learning_rate": shadow.LEARNING_RATE,
            "max_grad_norm": shadow.MAX_GRAD_NORM,
            "actor_parameter_names": list(ACTOR_NAMES),
            "post_step_retained_gradient_vector_sha256_float64": (
                EXPECTED_EQUALBLEND_GRADIENT_VECTOR_SHA256_FLOAT64
            ),
            "gradient_hash_algorithm": (
                "v1 probe actor6 order; name, NUL float64 NUL, JSON shape, NUL, "
                "contiguous CPU float64 tensor bytes, NUL"
            ),
            "training_record_match": "exact canonical JSON equality to shadow",
            "evaluation_record_match": "exact canonical JSON equality to shadow",
            "displacement_record_match": "exact canonical JSON equality to shadow",
            "endpoint_state_sha_strategy": (
                "shadow omitted endpoint state SHA; derive once after exact full-record "
                "reproduction and require exact pre-serialize/in-memory-reload/"
                "published-file-reload equality"
            ),
        },
        "checkpoint": {
            "count": 1,
            "path": str(ENDPOINT.relative_to(ROOT)),
            "format": "slim PPO evaluation-only checkpoint",
            "required_slim_keys": list(REQUIRED_SLIM_KEYS),
            "evaluation_only": True,
            "resume_forbidden": True,
            "forbidden_resume_keys": sorted(FORBIDDEN_RESUME_KEYS),
            "update": 468,
            "publication": "O_EXCL then descriptor-held exact reload",
        },
        "outputs": {
            "preregistration": str(PREREGISTRATION.relative_to(ROOT)),
            "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
            "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
            "endpoint": str(ENDPOINT.relative_to(ROOT)),
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "result": str(RESULT.relative_to(ROOT)),
            "fixed_contents": sorted(
                [ENDPOINT.name, MANIFEST.name, RESULT.name]
            ),
            "all_files_non_overwriting": True,
        },
        "scope": {
            "train_only_reproduction": True,
            "validation": False,
            "submission": False,
            "network": False,
            "resumable_training": False,
            "checkpoint_writes_exact": 1,
            "manifest_writes_exact": 1,
            "result_writes_exact": 1,
            "attempt_marker_writes_exact": 1,
            "preregistration_must_be_precreated_with_O_EXCL": True,
        },
    }


def validate_materialization_preregistration(
    expected_sha256: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        len(expected_sha256) != 64
        or expected_sha256 != expected_sha256.lower()
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RuntimeError("expected materialization prereg SHA must be lowercase hex")
    evidence = require_regular(
        PREREGISTRATION,
        expected_sha256,
        "materialization preregistration",
        expected_mode=0o444,
    )
    payload = strict_json(PREREGISTRATION, "materialization preregistration")
    if payload != {
        "schema_version": PREREGISTRATION_SCHEMA,
        "status": "locked_before_materialization",
        "materialization_contract": contract,
    }:
        raise RuntimeError("materialization preregistration exact-contract mismatch")
    return evidence


def exact_match(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"deterministic reproduction mismatch: {label}")


def retained_actor6_gradient_evidence(model: torch.nn.Module) -> dict[str, Any]:
    named = dict(model.named_parameters())
    gradient_names = sorted(
        name
        for name, parameter in named.items()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    )
    if gradient_names != sorted(ACTOR_NAMES):
        raise RuntimeError("retained post-step gradient scope is not exact actor6")
    digest = hashlib.sha256()
    for name in ACTOR_NAMES:
        parameter = named.get(name)
        if (
            parameter is None
            or parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all())
        ):
            raise FloatingPointError(f"missing/nonfinite retained gradient: {name}")
        # This is byte-for-byte the frozen V1 probe's gradient_sha256 recipe.
        value = parameter.grad.detach().cpu().contiguous().to(torch.float64)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0float64\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    observed = digest.hexdigest()
    if observed != EXPECTED_EQUALBLEND_GRADIENT_VECTOR_SHA256_FLOAT64:
        raise RuntimeError(f"equalblend retained gradient SHA drift: {observed}")
    return {
        "algorithm": "frozen_v1_probe_gradient_sha256_exact",
        "buffer_point": "after clip and plain-SGD step; before endpoint evaluation",
        "parameter_names": list(ACTOR_NAMES),
        "vector_sha256_float64": observed,
        "expected_vector_sha256_float64": (
            EXPECTED_EQUALBLEND_GRADIENT_VECTOR_SHA256_FLOAT64
        ),
        "exact_match": True,
    }


def reproduce_endpoint(
    reference_candidate: Mapping[str, Any],
    selections: Sequence[Sequence[dict[str, Any]]],
    union: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    model, parent = frozen.load_raw_u468(device)
    base_state = repair.clone_model_state(model)
    raw_evaluation, raw_correct = aggregate.evaluate_union(
        model, union, selections, device
    )
    if raw_correct.get(shadow.CRITICAL_LINE_SHA256) is not True:
        raise RuntimeError("critical line is not raw ordered-correct")
    training = shadow.execute_equalblend_step(model, union, masks, device)
    gradient_evidence = retained_actor6_gradient_evidence(model)
    endpoint_evaluation, endpoint_correct = aggregate.evaluate_union(
        model, union, selections, device
    )
    endpoint_state = repair.clone_model_state(model)
    displacement = frozen.displacement_report(base_state, model)
    frozen_names = sorted(set(base_state) - set(ACTOR_NAMES))
    frozen_unchanged = all(
        torch.equal(endpoint_state[name], base_state[name]) for name in frozen_names
    )
    retention_flips = frozen.correct_to_wrong_flips(
        raw_correct, endpoint_correct, selections
    )
    improvements = shadow.evaluation_improvements(
        raw_evaluation, endpoint_evaluation
    )
    critical = {
        "line_sha256": shadow.CRITICAL_LINE_SHA256,
        "raw_ordered_correct": raw_correct[shadow.CRITICAL_LINE_SHA256],
        "endpoint_ordered_correct": endpoint_correct[shadow.CRITICAL_LINE_SHA256],
    }
    evaluations = {"raw": raw_evaluation, "endpoint": endpoint_evaluation}
    exact_match("training", training, reference_candidate["training"])
    exact_match("evaluations", evaluations, reference_candidate["evaluations"])
    exact_match("displacement", displacement, reference_candidate["displacement_from_raw"])
    exact_match("improvements", improvements, reference_candidate["improvements"])
    exact_match(
        "retention flips",
        retention_flips,
        reference_candidate["retention_flips_from_raw"],
    )
    exact_match("critical line", critical, reference_candidate["critical_line"])
    exact_match(
        "frozen tensors unchanged",
        frozen_unchanged,
        reference_candidate["frozen_tensors_unchanged"],
    )
    if not frozen_unchanged or displacement["changed_parameter_names"] != sorted(
        ACTOR_NAMES
    ):
        raise RuntimeError("reproduced endpoint scope/frozen gate failed")
    endpoint_model_sha256 = frozen.model_state_sha256(endpoint_state)
    if ppo.model_state_sha256(model) != endpoint_model_sha256:
        raise RuntimeError("module/state endpoint SHA mismatch")
    reproduction = {
        "status": "exact_shadow_record_reproduced",
        "training": training,
        "retained_actor6_gradient": gradient_evidence,
        "evaluations": evaluations,
        "displacement_from_raw": displacement,
        "improvements": improvements,
        "retention_flips_from_raw": retention_flips,
        "critical_line": critical,
        "frozen_tensors_unchanged": frozen_unchanged,
        "endpoint_model_state_sha256": endpoint_model_sha256,
        "shadow_endpoint_model_state_sha256_present": False,
        "deterministic_full_record_matches_shadow_exactly": True,
        "validation_member_payloads_opened": False,
    }
    return endpoint_state, parent, reproduction


def build_checkpoint(
    parent: Mapping[str, Any],
    endpoint_state: Mapping[str, torch.Tensor],
    endpoint_model_sha256: str,
    runner_sha256: str,
    materialization_preregistration_sha256: str,
    shadow_reference_hashes: Mapping[str, str],
) -> dict[str, Any]:
    missing = [key for key in REQUIRED_SLIM_KEYS if key not in parent]
    if missing:
        raise RuntimeError(f"raw U468 lacks slim checkpoint keys: {missing}")
    checkpoint = {
        key: copy.deepcopy(parent[key]) for key in REQUIRED_SLIM_KEYS
    }
    checkpoint.update(
        {
            "model_state_dict": dict(endpoint_state),
            "update": 468,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_states_omitted": sorted(FORBIDDEN_RESUME_KEYS),
            "equalblend_sgd512_materialization": {
                "schema_version": SCHEMA,
                "materializer_runner_sha256": runner_sha256,
                "materialization_preregistration_sha256": (
                    materialization_preregistration_sha256
                ),
                "shadow_runner_sha256": SHADOW_RUNNER_SHA256,
                "shadow_result_sha256": SHADOW_RESULT_SHA256,
                "shadow_preregistration_sha256": SHADOW_PREREGISTRATION_SHA256,
                "shadow_attempt_marker_sha256": SHADOW_ATTEMPT_MARKER_SHA256,
                "promotion_gate_tool_sha256": PROMOTION_GATE_TOOL_SHA256,
                "promotion_gate_contract_sha256": (
                    PROMOTION_GATE_CONTRACT_SHA256
                ),
                "base_model_state_sha256": frozen.BASE_MODEL_SHA256,
                "endpoint_model_state_sha256": endpoint_model_sha256,
                "shadow_reference_hashes": dict(shadow_reference_hashes),
                "equalblend_gradient_vector_sha256_float64": (
                    EXPECTED_EQUALBLEND_GRADIENT_VECTOR_SHA256_FLOAT64
                ),
                "direct512_cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
                "direct512_batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
                "actor_parameter_names": list(ACTOR_NAMES),
                "evaluation_only": True,
                "resume_forbidden": True,
                "validation_member_payloads_opened": False,
            },
        }
    )
    if FORBIDDEN_RESUME_KEYS.intersection(checkpoint):
        raise RuntimeError("slim endpoint contains forbidden resume state")
    return checkpoint


def verify_serialized_checkpoint(
    payload: bytes,
    expected_model_sha256: str,
    expected_shadow_reference_hashes: Mapping[str, str],
) -> dict[str, Any]:
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("serialized endpoint root is not a dict")
    observed_model_sha256 = frozen.model_state_sha256(checkpoint["model_state_dict"])
    metadata = checkpoint.get("equalblend_sgd512_materialization")
    if (
        observed_model_sha256 != expected_model_sha256
        or checkpoint.get("evaluation_only") is not True
        or checkpoint.get("resume_forbidden") is not True
        or checkpoint.get("update") != 468
        or FORBIDDEN_RESUME_KEYS.intersection(checkpoint)
        or set(REQUIRED_SLIM_KEYS) - set(checkpoint)
        or not isinstance(metadata, dict)
        or metadata.get("endpoint_model_state_sha256") != expected_model_sha256
        or metadata.get("shadow_result_sha256") != SHADOW_RESULT_SHA256
        or metadata.get("promotion_gate_tool_sha256")
        != PROMOTION_GATE_TOOL_SHA256
        or metadata.get("promotion_gate_contract_sha256")
        != PROMOTION_GATE_CONTRACT_SHA256
        or metadata.get("shadow_reference_hashes")
        != dict(expected_shadow_reference_hashes)
        or metadata.get("equalblend_gradient_vector_sha256_float64")
        != EXPECTED_EQUALBLEND_GRADIENT_VECTOR_SHA256_FLOAT64
        or metadata.get("evaluation_only") is not True
        or metadata.get("resume_forbidden") is not True
    ):
        raise RuntimeError("serialized evaluation-only endpoint verification failed")
    return {
        "model_state_sha256": observed_model_sha256,
        "evaluation_only": True,
        "resume_forbidden": True,
        "optimizer_and_quota_states_absent": True,
        "shadow_reference_hashes_exact": True,
        "update": 468,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "static-audit",
            "cache-audit",
            "materialization-contract",
            "actual",
        ),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--expected-preregistration-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    runner = self_evidence()
    shadow_evidence, shadow_authorization, shadow_result = verify_shadow_chain()
    fixed_inputs = verify_fixed_inputs()
    static = ast_audit()
    if args.mode == "static-audit":
        if args.device != "cpu" or args.expected_preregistration_sha256 is not None:
            raise RuntimeError("static-audit is CPU/default and accepts no SHA arg")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "zero_write_static_audit_passed",
                    "runner": runner,
                    "fixed_inputs": fixed_inputs,
                    "shadow_authorization": shadow_authorization,
                    "ast_audit": static,
                    "output_absence": output_absence(),
                    "actual_executed": False,
                    "validation_member_payloads_opened": False,
                    "writes_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    cache_audit, selections, union, masks = build_cache_audit()
    if args.mode == "cache-audit":
        if args.device != "cpu" or args.expected_preregistration_sha256 is not None:
            raise RuntimeError("cache-audit is CPU/default and accepts no SHA arg")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "zero_write_cache_audit_passed",
                    "runner": runner,
                    "fixed_inputs": fixed_inputs,
                    "shadow_authorization": shadow_authorization,
                    "ast_audit": static,
                    "cache": cache_audit,
                    "output_absence": output_absence(),
                    "actual_executed": False,
                    "validation_member_payloads_opened": False,
                    "writes_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return


    if args.mode == "materialization-contract":
        if args.device != "cuda" or args.expected_preregistration_sha256 is not None:
            raise RuntimeError("materialization-contract requires --device cuda only")
        require_outputs_absent(include_preregistration=True)
        contract = expected_contract(
            runner,
            shadow_evidence,
            shadow_authorization,
            cache_audit,
            fixed_inputs["promotion_gate_precommit"],
        )
        print(
            json.dumps(
                {
                    "schema_version": PREREGISTRATION_SCHEMA,
                    "status": "locked_before_materialization",
                    "materialization_contract": contract,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return


    if args.device != "cuda" or args.expected_preregistration_sha256 is None:
        raise RuntimeError("actual requires CUDA and exact preregistration SHA")
    require_outputs_absent(include_preregistration=False)
    contract = expected_contract(
        runner,
        shadow_evidence,
        shadow_authorization,
        cache_audit,
        fixed_inputs["promotion_gate_precommit"],
    )
    preregistration_evidence = validate_materialization_preregistration(
        args.expected_preregistration_sha256, contract
    )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    random.seed(EXECUTION_SEED)
    torch.manual_seed(EXECUTION_SEED)
    torch.cuda.manual_seed_all(EXECUTION_SEED)

    # Repeat every mutable binding immediately before consuming the marker.
    fixed_inputs_at_lock = verify_fixed_inputs()
    shadow_evidence_at_lock, shadow_authorization_at_lock, _ = verify_shadow_chain()
    preregistration_at_lock = validate_materialization_preregistration(
        args.expected_preregistration_sha256, contract
    )
    if (
        fixed_inputs_at_lock != fixed_inputs
        or shadow_evidence_at_lock != shadow_evidence
        or shadow_authorization_at_lock != shadow_authorization
        or preregistration_at_lock != preregistration_evidence
        or self_evidence()["sha256"] != runner["sha256"]
    ):
        raise RuntimeError("materialization identity changed before lock")
    require_outputs_absent(include_preregistration=False)
    marker_payload = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "materialization_attempt_consumed_before_model_or_optimizer",
        "created_at_utc": utc_now(),
        "branch": BRANCH,
        "runner": runner,
        "shadow_chain": shadow_evidence,
        "shadow_authorization": shadow_authorization,
        "promotion_gate_precommit": fixed_inputs["promotion_gate_precommit"],
        "preregistration": preregistration_evidence,
        "materialization_contract": contract,
        "cuda_verified_before_lock": True,
        "model_instances_created_before_lock": 0,
        "optimizer_instances_created_before_lock": 0,
        "checkpoint_writes_before_lock": 0,
        "validation_member_payloads_opened": False,
    }
    marker = publish_o_excl(ATTEMPT_MARKER, canonical_json(marker_payload), 0o444)
    fsync_directory(ROOT)

    os.mkdir(OUTPUT_ROOT, mode=0o700)
    fsync_directory(ROOT / "artifacts")
    if list(OUTPUT_ROOT.iterdir()):
        raise RuntimeError("fresh materialization output root is not empty")

    reference_candidate = shadow_result["candidate"]
    endpoint_state, parent, reproduction = reproduce_endpoint(
        reference_candidate, selections, union, masks, device
    )
    torch.cuda.synchronize(device)
    endpoint_model_sha256 = reproduction["endpoint_model_state_sha256"]
    checkpoint = build_checkpoint(
        parent,
        endpoint_state,
        endpoint_model_sha256,
        runner["sha256"],
        preregistration_evidence["sha256"],
        shadow_authorization["reference_hashes"],
    )
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint, checkpoint_buffer)
    checkpoint_payload = checkpoint_buffer.getvalue()
    in_memory_reload = verify_serialized_checkpoint(
        checkpoint_payload,
        endpoint_model_sha256,
        shadow_authorization["reference_hashes"],
    )
    endpoint_evidence = publish_o_excl(ENDPOINT, checkpoint_payload, 0o444)
    fsync_directory(OUTPUT_ROOT)
    published_payload = ENDPOINT.read_bytes()
    if sha256_bytes(published_payload) != endpoint_evidence["sha256"]:
        raise RuntimeError("published endpoint file SHA drift")
    published_reload = verify_serialized_checkpoint(
        published_payload,
        endpoint_model_sha256,
        shadow_authorization["reference_hashes"],
    )
    if published_reload != in_memory_reload:
        raise RuntimeError("published endpoint semantic reload drift")

    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "endpoint_materialized_and_verified_before_completion_result",
        "created_at_utc": utc_now(),
        "branch": BRANCH,
        "runner": runner,
        "shadow_chain": shadow_evidence,
        "shadow_authorization": shadow_authorization,
        "promotion_gate_precommit": fixed_inputs["promotion_gate_precommit"],
        "materialization_contract": contract,
        "preregistration": preregistration_evidence,
        "attempt_marker": marker,
        "cache": {
            "cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
            "batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
            "flat_identity_sha256": EXPECTED_FLAT_IDENTITY_SHA256,
        },
        "reproduction": reproduction,
        "endpoint": {
            **endpoint_evidence,
            "model_state_sha256": endpoint_model_sha256,
            "in_memory_reload": in_memory_reload,
            "published_file_reload": published_reload,
        },
        "integrity": {
            "shadow_full_training_record_exact": True,
            "shadow_full_evaluation_record_exact": True,
            "shadow_displacement_record_exact": True,
            "endpoint_model_state_sha_pre_serialize_reload_exact": True,
            "endpoint_checkpoint_count_exact": 1,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_and_quota_states_omitted": True,
            "validation_member_payloads_opened": False,
            "submission_performed": False,
        },
        "promotion_gate_state": {
            "contract_frozen_before_endpoint_materialization": True,
            "precommit_contract_intentionally_unarmed": True,
            "promotion_gate_evaluation_executed": False,
            "future_armed_gate_must_bind_this_manifest_and_endpoint": True,
            "generation_manifest_is_this_artifact": True,
        },
    }
    manifest = publish_o_excl(MANIFEST, canonical_json(manifest_payload), 0o444)
    fsync_directory(OUTPUT_ROOT)
    result_payload = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_one_eval_only_endpoint_materialization",
        "created_at_utc": utc_now(),
        "branch": BRANCH,
        "decision": "MATERIALIZED",
        "endpoint": {
            **endpoint_evidence,
            "model_state_sha256": endpoint_model_sha256,
            "evaluation_only": True,
            "resume_forbidden": True,
        },
        "manifest": manifest,
        "attempt_marker": marker,
        "preregistration": preregistration_evidence,
        "shadow_result": shadow_evidence["result"],
        "promotion_gate_tool": fixed_inputs["promotion_gate_precommit"]["tool"],
        "promotion_gate_contract": fixed_inputs["promotion_gate_precommit"][
            "frozen_contract_artifact"
        ],
        "promotion_gate_state": {
            "precommit_contract_intentionally_unarmed": True,
            "promotion_gate_evaluation_executed": False,
            "future_armed_gate_revision_required": True,
        },
        "checkpoint_writes": 1,
        "manifest_writes": 1,
        "result_writes": 1,
        "validation_member_payloads_opened": False,
        "submission_performed": False,
    }
    result = publish_o_excl(RESULT, canonical_json(result_payload), 0o444)
    fsync_directory(OUTPUT_ROOT)
    expected_names = sorted([ENDPOINT.name, MANIFEST.name, RESULT.name])
    observed_names = sorted(path.name for path in OUTPUT_ROOT.iterdir())
    if observed_names != expected_names:
        raise RuntimeError(f"materialization output contents drift: {observed_names}")
    os.chmod(OUTPUT_ROOT, 0o555)
    fsync_directory(ROOT / "artifacts")
    print(
        json.dumps(
            {
                "status": result_payload["status"],
                "endpoint": result_payload["endpoint"],
                "manifest": manifest,
                "result": result,
                "attempt_marker": marker,
                "validation_member_payloads_opened": False,
                "submission_performed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

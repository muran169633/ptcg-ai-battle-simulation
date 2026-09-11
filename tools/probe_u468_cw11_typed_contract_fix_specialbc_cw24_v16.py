#!/usr/bin/env python3
"""Typed-contract repair wrapper over frozen CW24 v15.

The frozen v15 production function is reused unchanged.  Its authenticated
load_v14_module callback is replaced in memory with a frozen-v14 loader whose
only source change replaces v14's invalid truthiness-based result-contract
validator with typed, exact predicates.  The five intentionally falsy audit
fields are checked as exact zero/False values; all remaining fields have
explicit type, value, range, sequence, or STALL predicates.

No training code, v15 quantization-safe projector, v13 target, 29+32 budget,
gate, payload, or STALL logic is changed.  This is an unfrozen audit-first,
one-shot, train-only draft.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_typed_contract_fix_specialbc_cw24_v16.py"
OUTPUT = ROOT / "artifacts/cw24_cw11_typed_contract_fix_specialbc_trainonly_v16.json"
ATTEMPT_MARKER = ROOT / "artifacts/.ptcg-cw24-cw11-typed-contract-fix-specialbc-v16-attempt.json"
FAILURE = ROOT / "artifacts/cw24_cw11_typed_contract_fix_specialbc_failure_v16.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-typed-contract-fix-specialbc-cw24-v16"
SEED = 202608052
STAGE1_REFERENCE_STEP = 29
STAGE2_MAX_STEPS = 32
TOTAL_MAX_STEPS = 61

V15_SOURCE = TOOLS / "probe_u468_cw11_quant_safe_projected_specialbc_cw24_v15.py"
V15_SOURCE_SHA256 = "95dbbd243c82812c8fd85d8958593f1f143eb229d7599361c9dcaa72d25589f1"
V15_ATTEMPT = ROOT / "artifacts/.ptcg-cw24-cw11-quant-safe-projected-specialbc-v15-attempt.json"
V15_ATTEMPT_SHA256 = "7b58d270b3d324bfddd7af92f01a72f212ab9a9d5434d1b3538f90c793ae2081"
V15_OUTPUT = ROOT / "artifacts/cw24_cw11_quant_safe_projected_specialbc_trainonly_v15.json"
V15_TRANSFORMED_V13_SHA256 = "0733e3af5276c1292ba54e476eb72cfefff87089407408f9fa021d399dab986d"

V14_SOURCE = TOOLS / "probe_u468_cw11_stall_diagnostic_specialbc_cw24_v14.py"
V14_SOURCE_SHA256 = "1580c0dddb9dba15967f1ca19df18ba0aad850d81063114e4b7f4104cb87073b"
V14_ATTEMPT = ROOT / "artifacts/.ptcg-cw24-cw11-stall-diagnostic-specialbc-v14-attempt.json"
V14_ATTEMPT_SHA256 = "b819c19e02c8894938695c29adffa8ecd7b632c3526e42e6731b6b5c90a568ec"
V14_OUTPUT = ROOT / "artifacts/cw24_cw11_stall_diagnostic_specialbc_trainonly_v14.json"
V13_SOURCE = TOOLS / "probe_u468_cw11_two_stage_pf7_specialbc_cw24_v13.py"
V13_SOURCE_SHA256 = "a99762d1a2c98b0521199d845d3f041fff40b3787ff3ceda0fd8207e1c6d0ff7"
V13_ATTEMPT = ROOT / "artifacts/.ptcg-cw24-cw11-two-stage-pf7-specialbc-v13-attempt.json"
V13_ATTEMPT_SHA256 = "924e4741a407351246e25c19e476e74e7cd504886403114c540638f72c00316d"
V13_OUTPUT = ROOT / "artifacts/cw24_cw11_two_stage_pf7_specialbc_trainonly_v13.json"

VALIDATOR_OLD = '''    if not all(
        value
        for key, value in v14_contract.items()
        if key
        not in {
            "pre_cuda_runtime",
            "optimizer_step_calls_attempted",
            "unique_changed_train_shadow_count",
            "standard_projected_SGD_sequence",
            "structured_stall_audit",
        }
    ):
        raise ProtocolError(f"v14 production contract failed: {v14_contract}")
'''
VALIDATOR_NEW = '''    v16_expected_contract_keys = {
        "python_exact",
        "pre_cuda_runtime",
        "post_run_rehash_checks",
        "train_only",
        "new_validation_or_test_rows_opened",
        "checkpoint_writes",
        "model_artifact_writes",
        "optimizer_instances_created",
        "optimizer_step_calls_attempted",
        "unique_changed_train_shadow_count",
        "within_total61_budget",
        "stage1_reference_exact",
        "preclip_radial_projection_absent_every_record",
        "standard_projected_SGD_sequence",
        "candidate_payload_present_iff_GO",
        "structured_stall_audit",
        "frozen_v13_output_absent_before_and_after",
        "official_unique_changed_candidate_count_consumed",
        "submission_performed",
    }
    v16_expected_sequence = [
        "backward",
        "original_clip_grad_norm",
        "plain_SGD_step",
        "v12_project_actor6_endpoint_ball_projection",
    ]
    v16_pre_cuda_check_keys = {
        "repo_root",
        "my_project_env",
        "isolated",
        "dont_write_bytecode",
        "pycache_prefix_dev_null",
        "torch_not_imported_before_claim",
    }
    v16_rehash_keys = {
        "v14_source_byte_identity",
        "all_dependencies_byte_identity",
        "frozen_v13_terminal_state_identity",
        "v13_output_absent_after_training",
    }
    v16_pre_cuda = v14_contract.get("pre_cuda_runtime")
    v16_rehash = v14_contract.get("post_run_rehash_checks")
    v16_optimizer_calls = v14_contract.get("optimizer_step_calls_attempted")
    v16_unique_changed = v14_contract.get("unique_changed_train_shadow_count")
    v16_stall = v14_contract.get("structured_stall_audit")
    v16_typed_contract_checks = {
        "key_set_exact": set(v14_contract) == v16_expected_contract_keys,
        "python_exact_true": v14_contract.get("python_exact") is True,
        "pre_cuda_runtime_exact": isinstance(v16_pre_cuda, Mapping)
        and set(v16_pre_cuda) == {"python", "checks", "pass"}
        and v16_pre_cuda.get("python") == str(EXPECTED_PYTHON.resolve())
        and v16_pre_cuda.get("pass") is True
        and isinstance(v16_pre_cuda.get("checks"), Mapping)
        and set(v16_pre_cuda["checks"]) == v16_pre_cuda_check_keys
        and all(value is True for value in v16_pre_cuda["checks"].values()),
        "post_run_rehash_checks_exact": isinstance(v16_rehash, Mapping)
        and set(v16_rehash) == v16_rehash_keys
        and all(value is True for value in v16_rehash.values()),
        "train_only_true": v14_contract.get("train_only") is True,
        "new_validation_zero": type(
            v14_contract.get("new_validation_or_test_rows_opened")
        )
        is int
        and v14_contract["new_validation_or_test_rows_opened"] == 0,
        "checkpoint_writes_zero": type(v14_contract.get("checkpoint_writes"))
        is int
        and v14_contract["checkpoint_writes"] == 0,
        "model_artifact_writes_zero": type(
            v14_contract.get("model_artifact_writes")
        )
        is int
        and v14_contract["model_artifact_writes"] == 0,
        "one_optimizer": type(v14_contract.get("optimizer_instances_created"))
        is int
        and v14_contract["optimizer_instances_created"] == 1,
        "optimizer_calls_in_budget": type(v16_optimizer_calls) is int
        and STAGE1_REFERENCE_STEP <= v16_optimizer_calls <= TOTAL_MAX_STEPS,
        "unique_changed_in_budget": type(v16_unique_changed) is int
        and STAGE1_REFERENCE_STEP <= v16_unique_changed <= TOTAL_MAX_STEPS
        and v16_unique_changed <= v16_optimizer_calls,
        "within_total61_true": v14_contract.get("within_total61_budget") is True,
        "stage1_reference_exact_true": v14_contract.get(
            "stage1_reference_exact"
        )
        is True,
        "preclip_projection_absent_true": v14_contract.get(
            "preclip_radial_projection_absent_every_record"
        )
        is True,
        "optimizer_sequence_exact": v14_contract.get(
            "standard_projected_SGD_sequence"
        )
        == v16_expected_sequence,
        "payload_equivalence_true": v14_contract.get(
            "candidate_payload_present_iff_GO"
        )
        is True,
        "stall_shape_exact": (
            isinstance(v16_stall, Mapping)
            and v16_stall.get("pass") is True
            if decision == "NO_GO_V14_STALLED"
            else v16_stall is None
        ),
        "v13_output_absent_true": v14_contract.get(
            "frozen_v13_output_absent_before_and_after"
        )
        is True,
        "official_count_zero": type(
            v14_contract.get("official_unique_changed_candidate_count_consumed")
        )
        is int
        and v14_contract["official_unique_changed_candidate_count_consumed"] == 0,
        "submission_false": v14_contract.get("submission_performed") is False,
    }
    if not all(value is True for value in v16_typed_contract_checks.values()):
        raise ProtocolError(
            f"v16 typed v14 production contract failed: {v16_typed_contract_checks}"
        )
'''

DEPENDENCIES = (
    ("frozen_v15_source", V15_SOURCE, V15_SOURCE_SHA256, 0o555),
    ("frozen_v15_consumed_attempt", V15_ATTEMPT, V15_ATTEMPT_SHA256, 0o444),
    ("frozen_v14_source", V14_SOURCE, V14_SOURCE_SHA256, 0o555),
    ("frozen_v14_consumed_attempt", V14_ATTEMPT, V14_ATTEMPT_SHA256, 0o444),
    ("frozen_v13_source", V13_SOURCE, V13_SOURCE_SHA256, 0o555),
    ("frozen_v13_consumed_attempt", V13_ATTEMPT, V13_ATTEMPT_SHA256, 0o444),
    (
        "frozen_v12_source",
        TOOLS / "probe_u468_cw11_projected_nonlinear_specialbc_cw24_v12.py",
        "003e476325a08ff5be400167fccbd43705fac9a4071d806101ddb02969ea2301",
        0o555,
    ),
    (
        "frozen_v12_result",
        ROOT / "artifacts/cw24_cw11_projected_nonlinear_specialbc_trainonly_v12.json",
        "effee98dc03c6d505f7bc2007dae88e28190c0fbba80f22c7902363592b554dc",
        0o444,
    ),
    (
        "frozen_CW24_v1",
        TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py",
        "a3c96eed95a6b558c61303e2190eaa4f353970add4509e8770c3dfcc1621db39",
        0o555,
    ),
    (
        "frozen_CW22",
        TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py",
        "f307f3f26355125f97c857f14b97a73d9a7a4f2ea39ca8b0a578e801055e82b4",
        0o555,
    ),
    (
        "frozen_CW23",
        TOOLS / "probe_u468_cw11_boundary_qp_specialbc_cw23_v1.py",
        "4a2d209af71a7ea5d955bd373f1fbada936688fc5bc597615c41192dc360b4b8",
        0o555,
    ),
    (
        "plain_SGD_reference",
        TOOLS / "run_u468_raw_actor6_equalblend_sgd512_shadow.py",
        "e04f7b7579ef42d0c6f643db833779fb837ae86e3205d948b5564d08f8b30f8e",
        0o555,
    ),
    (
        "dynamic_pair_reference",
        TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v1.py",
        "9c5a7377d82b18e437ca89d064ce8be60554646573e3aca27f4e40f4989ab77c",
        0o555,
    ),
    (
        "frozen_B352_selection",
        ROOT / "artifacts/cw24_top1_b352_selection_v1.json",
        "1f72b26eccd436ca0d341f837ec42949f5823bf7f0be73461aaeab28166cc222",
        0o444,
    ),
)


class ProtocolError(RuntimeError):
    """Fail-closed v16 protocol error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    def check(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ProtocolError("nonfinite JSON value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ProtocolError("non-string JSON key")
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
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


def path_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    return False


def regular_source(
    path: Path, expected_sha: str, expected_mode: int, label: str
) -> tuple[bytes, dict[str, Any]]:
    before = path.lstat()
    raw = path.read_bytes()
    after = path.lstat()
    digest = hashlib.sha256(raw).hexdigest()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
        "sha_exact": digest == expected_sha,
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} drift: {checks}")
    return raw, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(raw),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def binding_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(record["path"]),
        "sha256": str(record["sha256"]),
        "bytes": int(record["bytes"]),
        "mode_octal": str(record["mode_octal"]),
    }


def dependency_evidence() -> dict[str, Any]:
    return {
        label: regular_source(path, digest, mode, label)[1]
        for label, path, digest, mode in DEPENDENCIES
    }


def load_json_no_duplicates(raw: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"{label}: duplicate key {key}")
            result[key] = value
        return result

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ProtocolError(f"{label}: nonfinite constant {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ProtocolError(f"{label}: root is not an object")
    return value


def validate_pre_cuda_runtime() -> dict[str, Any]:
    checks = {
        "repo_root": Path.cwd().resolve() == ROOT,
        "my_project_env": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1
        and sys.dont_write_bytecode is True,
        "pycache_prefix_dev_null": sys.pycache_prefix == "/dev/null",
        "torch_not_imported_before_claim": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        raise ProtocolError(f"pre-CUDA runtime drift: {checks}")
    return {
        "python": str(Path(sys.executable).resolve()),
        "checks": checks,
        "pass": True,
    }


def validate_v15_terminal_state() -> dict[str, Any]:
    raw, evidence = regular_source(
        V15_ATTEMPT, V15_ATTEMPT_SHA256, 0o444, "consumed v15 attempt"
    )
    payload = load_json_no_duplicates(raw, "consumed v15 attempt")
    checks = {
        "schema_exact": payload.get("schema_version")
        == "ptcg-u468-cw11-quant-safe-projected-specialbc-cw24-v15-attempt",
        "status_claimed": payload.get("status")
        == "claimed_before_module_exec_or_CUDA",
        "source_exact": payload.get("source", {}).get("sha256")
        == V15_SOURCE_SHA256,
        "transform_exact": payload.get("in_memory_v15_transform", {}).get(
            "transformed_sha256"
        )
        == V15_TRANSFORMED_V13_SHA256,
        "stage1_29_stage2_32": payload.get("stage1_reference_steps") == 29
        and payload.get("stage2_maximum_steps") == 32,
        "total61": payload.get("total_maximum_changed_train_shadows") == 61,
        "retry_forbidden": payload.get("retry_authorized") is False,
        "v15_output_absent_now": path_absent(V15_OUTPUT),
        "v14_output_absent_now": path_absent(V14_OUTPUT),
        "v13_output_absent_now": path_absent(V13_OUTPUT),
    }
    if not all(checks.values()):
        raise ProtocolError(f"frozen v15 terminal-state drift: {checks}")
    return {"attempt": evidence, "checks": checks, "payload": payload}


def terminal_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "consumed_attempt": binding_summary(state["attempt"]),
        "checks": dict(state["checks"]),
        "v15_output": str(V15_OUTPUT.relative_to(ROOT)),
        "v15_output_absent": bool(state["checks"]["v15_output_absent_now"]),
        "v14_output_absent": bool(state["checks"]["v14_output_absent_now"]),
        "v13_output_absent": bool(state["checks"]["v13_output_absent_now"]),
    }


def transform_v14_source() -> tuple[bytes, dict[str, Any]]:
    raw, evidence = regular_source(
        V14_SOURCE, V14_SOURCE_SHA256, 0o555, "frozen v14 source"
    )
    source = raw.decode("utf-8")
    occurrences = source.count(VALIDATOR_OLD)
    if occurrences != 1:
        raise ProtocolError(
            f"v16 expected one exact v14 validator site, got {occurrences}"
        )
    source = source.replace(VALIDATOR_OLD, VALIDATOR_NEW, 1)
    transformed = source.encode("utf-8")
    compile(transformed, str(V14_SOURCE), "exec", dont_inherit=True)
    checks = {
        "one_exact_validator_replacement": occurrences == 1,
        "old_truthiness_validator_removed": VALIDATOR_OLD.encode()
        not in transformed,
        "typed_key_set_check_present": transformed.count(
            b'"key_set_exact": set(v14_contract) == v16_expected_contract_keys'
        )
        == 1,
        "five_falsy_exact_checks_present": all(
            transformed.count(token) >= 1
            for token in (
                b'"new_validation_zero"',
                b'"checkpoint_writes_zero"',
                b'"model_artifact_writes_zero"',
                b'"official_count_zero"',
                b'"submission_false"',
            )
        ),
        "optimizer_range_check_present": transformed.count(
            b'"optimizer_calls_in_budget"'
        )
        == 1,
        "unique_changed_range_check_present": transformed.count(
            b'"unique_changed_in_budget"'
        )
        == 1,
        "sequence_exact_check_present": transformed.count(
            b'"optimizer_sequence_exact"'
        )
        == 1,
        "stall_shape_check_present": transformed.count(b'"stall_shape_exact"') == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v16 transformed-v14 audit failed: {checks}")
    return transformed, {
        "frozen_v14_source": evidence,
        "transformed_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_bytes": len(transformed),
        "replacement": {
            "label": "v14_truthiness_contract_to_typed_exact_predicates",
            "occurrences": occurrences,
            "old_sha256": hashlib.sha256(VALIDATOR_OLD.encode()).hexdigest(),
            "new_sha256": hashlib.sha256(VALIDATOR_NEW.encode()).hexdigest(),
        },
        "checks": checks,
    }


def load_transformed_v14_module() -> tuple[ModuleType, dict[str, Any]]:
    source, audit = transform_v14_source()
    module_name = "cw24_v16_typed_contract_transformed_v14"
    if module_name in sys.modules:
        raise ProtocolError("v16 transformed v14 module name already occupied")
    module = types.ModuleType(module_name)
    module.__file__ = str(V14_SOURCE)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(V14_SOURCE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, {
        "frozen_source": audit["frozen_v14_source"],
        "in_memory_typed_contract_transform": audit,
    }


def load_v15_module() -> tuple[ModuleType, dict[str, Any]]:
    raw, evidence = regular_source(
        V15_SOURCE, V15_SOURCE_SHA256, 0o555, "frozen v15 source"
    )
    module_name = "cw24_v16_frozen_v15_wrapper"
    if module_name in sys.modules:
        raise ProtocolError("frozen v15 module name already occupied")
    module = types.ModuleType(module_name)
    module.__file__ = str(V15_SOURCE)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(raw, str(V15_SOURCE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, evidence


def no_go_material_audit(value: Any) -> dict[str, Any]:
    violations: list[str] = []
    forbidden = {"actor_bytes", "candidate_actor_float32_le", "actual_delta"}

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                child_path = f"{path}.{key}"
                if key == "candidate_payload" and child is not None:
                    violations.append(f"{child_path}:non_null")
                if key in forbidden:
                    violations.append(f"{child_path}:forbidden_material_key")
                if "delta" in key.lower() and isinstance(child, (list, tuple)):
                    violations.append(f"{child_path}:delta_array")
                walk(child, child_path)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
        elif isinstance(item, (bytes, bytearray, memoryview)):
            violations.append(f"{path}:binary_value")

    walk(value, "$<result>")
    if violations:
        raise ProtocolError(f"v16 NO_GO material rejected: {violations[:8]}")
    return {
        "pass": True,
        "candidate_payloads_all_null": True,
        "actor_bytes_absent": True,
        "actual_delta_arrays_absent": True,
        "binary_values_absent": True,
    }


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    imports = []
    loader_assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "v15"
                and target.attr == "load_v14_module"
                and isinstance(node.value, ast.Name)
            ):
                loader_assignments.append(node.value.id)
    _, transform = transform_v14_source()
    checks = {
        "syntax_valid": True,
        "transform_static_audit": all(transform["checks"].values()),
        "single_exact_source_replacement": transform["replacement"]["occurrences"]
        == 1,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_network_import": not any(
            name.split(".")[0] in {"requests", "urllib", "httpx", "socket"}
            for name in imports
        ),
        "single_loader_override_site": loader_assignments.count(
            "replacement_loader"
        )
        == 1,
        "single_loader_restore_site": loader_assignments.count("original_loader")
        == 1,
        "single_result_publication_site": source.count(
            b"publish_o_excl(" + b"OUTPUT"
        )
        == 1,
        "single_attempt_publication_site": source.count(
            b"publish_o_excl(" + b"ATTEMPT_MARKER"
        )
        == 1,
        "single_failure_publication_site": source.count(
            b"publish_o_excl(" + b"FAILURE"
        )
        == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v16 source audit failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "transform": transform,
    }


def self_evidence(require_frozen: bool) -> dict[str, Any]:
    before = SCRIPT.lstat()
    source = SCRIPT.read_bytes()
    after = SCRIPT.lstat()
    mode = stat.S_IMODE(after.st_mode)
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
        "mode_allowed": mode == 0o555
        if require_frozen
        else mode in {0o644, 0o664, 0o555},
        "source_audit": source_audit()["pass"],
    }
    if not all(checks.values()):
        raise ProtocolError(f"v16 self evidence failed: {checks}")
    return {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "mode_octal": format(mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ProtocolError("short v16 publication write")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_fd = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    observed = path.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and int(observed.st_nlink) == 1,
        "mode_0444": stat.S_IMODE(observed.st_mode) == 0o444,
        "size_exact": int(observed.st_size) == len(payload),
        "sha_exact": sha256_file(path) == hashlib.sha256(payload).hexdigest(),
    }
    if not all(checks.values()):
        raise ProtocolError(f"v16 publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def claim_attempt() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v16 output already exists before attempt claim")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v16 one-shot attempt was already consumed")
    if not path_absent(FAILURE):
        raise ProtocolError("v16 failure target already exists before attempt claim")
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    terminal = validate_v15_terminal_state()
    transform = transform_v14_source()[1]
    payload = {
        "schema_version": f"{SCHEMA}-attempt",
        "status": "claimed_before_module_exec_or_CUDA",
        "source": binding_summary(source),
        "dependencies": {
            name: binding_summary(record) for name, record in dependencies.items()
        },
        "frozen_v15_terminal_state": terminal_summary(terminal),
        "in_memory_v14_contract_transform": {
            "transformed_sha256": transform["transformed_sha256"],
            "transformed_bytes": transform["transformed_bytes"],
            "replacement": transform["replacement"],
            "checks": transform["checks"],
        },
        "runtime": runtime,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent_by_lstat_at_claim": True,
        "failure": str(FAILURE.relative_to(ROOT)),
        "failure_absent_by_lstat_at_claim": True,
        "v15_v14_v13_outputs_absent_by_lstat_at_claim": True,
        "CUDA_preflight_state": "not_imported_pending_production_validation",
        "single_seed": SEED,
        "stage1_reference_steps": STAGE1_REFERENCE_STEP,
        "stage2_maximum_steps": STAGE2_MAX_STEPS,
        "total_maximum_changed_train_shadows": TOTAL_MAX_STEPS,
        "algorithm_change": None,
        "only_wrapper_change": "typed_exact_v14_result_contract_validation",
        "retry_authorized": False,
        "network_package_upload_submission": False,
    }
    publication = publish_o_excl(ATTEMPT_MARKER, canonical_json(payload))
    return {"payload": payload, "publication": publication}


def production_run() -> dict[str, Any]:
    pre_cuda_runtime = validate_pre_cuda_runtime()
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    terminal = validate_v15_terminal_state()
    pre_transform = transform_v14_source()[1]
    v15, v15_evidence = load_v15_module()
    loader_state: dict[str, Any] = {}

    def replacement_loader() -> tuple[ModuleType, dict[str, Any]]:
        if loader_state:
            raise ProtocolError("v16 transformed v14 loader called more than once")
        module, evidence = load_transformed_v14_module()
        loader_state.update({"calls": 1, "module": module, "evidence": evidence})
        return module, evidence

    original_loader = v15.load_v14_module
    v15.load_v14_module = replacement_loader
    try:
        result = v15.production_run()
    finally:
        v15.load_v14_module = original_loader

    post_source = self_evidence(require_frozen=True)
    post_dependencies = dependency_evidence()
    post_terminal = validate_v15_terminal_state()
    post_transform = transform_v14_source()[1]
    rehash_checks = {
        "v16_source_byte_identity": post_source == source,
        "all_dependencies_byte_identity": post_dependencies == dependencies,
        "frozen_v15_terminal_state_identity": post_terminal == terminal,
        "v14_contract_transform_identity": post_transform == pre_transform,
        "frozen_v15_loader_restored": v15.load_v14_module is original_loader,
        "replacement_loader_executed_once": loader_state.get("calls") == 1,
        "v15_output_absent_after_training": path_absent(V15_OUTPUT),
        "v14_output_absent_after_training": path_absent(V14_OUTPUT),
        "v13_output_absent_after_training": path_absent(V13_OUTPUT),
    }
    if not all(rehash_checks.values()):
        raise ProtocolError(f"v16 post-run immutable input drift: {rehash_checks}")
    if not isinstance(result, dict) or not isinstance(result.get("endpoint"), dict):
        raise ProtocolError("frozen v15 returned malformed result")
    endpoint = result["endpoint"]
    decision = str(endpoint.get("decision", ""))
    if decision not in {
        "GO_CW24_TWO_STAGE_PF7_TRAIN_GATE",
        "NO_GO_CW24_TWO_STAGE_PF7_TRAIN_GATE",
        "NO_GO_V15_STALLED",
    }:
        raise ProtocolError(f"unexpected frozen-v15 decision: {decision}")
    payload_present = endpoint.get("candidate_payload") is not None
    if payload_present != decision.startswith("GO_"):
        raise ProtocolError("v16 GO/payload equivalence failed")
    audit = result.get("audit")
    v14_contract = audit.get("CW24_v14_contract") if isinstance(audit, dict) else None
    v15_contract = audit.get("CW24_v15_contract") if isinstance(audit, dict) else None
    contract_checks = {
        "v14_contract_present": isinstance(v14_contract, Mapping),
        "v15_contract_present": isinstance(v15_contract, Mapping),
        "new_validation_zero": isinstance(v14_contract, Mapping)
        and type(v14_contract.get("new_validation_or_test_rows_opened")) is int
        and v14_contract["new_validation_or_test_rows_opened"] == 0,
        "checkpoint_writes_zero": isinstance(v14_contract, Mapping)
        and type(v14_contract.get("checkpoint_writes")) is int
        and v14_contract["checkpoint_writes"] == 0,
        "model_artifact_writes_zero": isinstance(v14_contract, Mapping)
        and type(v14_contract.get("model_artifact_writes")) is int
        and v14_contract["model_artifact_writes"] == 0,
        "official_count_zero": isinstance(v14_contract, Mapping)
        and type(
            v14_contract.get("official_unique_changed_candidate_count_consumed")
        )
        is int
        and v14_contract["official_unique_changed_candidate_count_consumed"] == 0,
        "submission_false": isinstance(v14_contract, Mapping)
        and v14_contract.get("submission_performed") is False,
        "v15_training_transform_unchanged": result.get("inputs", {})
        .get("CW24_v15_in_memory_transform", {})
        .get("transformed_sha256")
        == V15_TRANSFORMED_V13_SHA256,
    }
    if not all(contract_checks.values()):
        raise ProtocolError(f"v16 result contract failed: {contract_checks}")

    result["schema_version"] = SCHEMA
    result["status"] = decision
    result["decision"] = decision
    result["seed"] = SEED
    result.setdefault("base", {})["wrapper_contract"] = (
        "frozen_v15_training_with_typed_exact_v14_result_validation"
    )
    historical = result.get("historical_exact_CW11_replay")
    if not isinstance(historical, dict):
        raise ProtocolError("v16 result lacks historical replay audit")
    historical.pop(
        "new_validation_rows_opened_for_CW24_v15_selection_or_candidate", None
    )
    historical["new_validation_rows_opened_for_CW24_v16_selection_or_candidate"] = 0
    inputs = result.setdefault("inputs", {})
    inputs["CW24_v16_source"] = source
    inputs["CW24_v16_source_post_run"] = post_source
    inputs["CW24_v16_dependencies"] = dependencies
    inputs["CW24_v16_dependencies_post_run"] = post_dependencies
    inputs["CW24_v16_frozen_v15_terminal_state"] = terminal
    inputs["CW24_v16_frozen_v15_terminal_state_post_run"] = post_terminal
    inputs["CW24_v16_frozen_v15_module"] = v15_evidence
    inputs["CW24_v16_in_memory_v14_contract_transform"] = pre_transform
    inputs["CW24_v16_in_memory_v14_contract_transform_post_run"] = post_transform
    inputs["CW24_v16_transformed_v14_loader_evidence"] = loader_state["evidence"]
    audit = result.setdefault("audit", {})
    audit["CW24_v16_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "pre_cuda_runtime": pre_cuda_runtime,
        "post_run_rehash_checks": rehash_checks,
        "typed_falsy_contract_checks": contract_checks,
        "training_algorithm_changed": False,
        "frozen_v15_training_transform_sha256": V15_TRANSFORMED_V13_SHA256,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"][
        "only_v16_change_is_typed_result_contract_validation"
    ] = True
    if decision.startswith("NO_GO"):
        audit["NO_GO_material_audit_v16"] = no_go_material_audit(result)
        no_go_material_audit(result)
    canonical_json(result)
    return result


def failure_record(
    error: BaseException, attempt: Mapping[str, Any]
) -> dict[str, Any]:
    if not path_absent(FAILURE):
        raise ProtocolError("v16 failure target already exists")
    if not path_absent(OUTPUT):
        raise ProtocolError("v16 result exists while constructing failure record")
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    terminal = validate_v15_terminal_state()
    transform = transform_v14_source()[1]
    attempt_post = regular_source(
        ATTEMPT_MARKER,
        attempt["publication"]["sha256"],
        0o444,
        "v16 attempt marker at failure",
    )[1]
    attempt_payload = attempt["payload"]
    compact_transform = {
        "transformed_sha256": transform["transformed_sha256"],
        "transformed_bytes": transform["transformed_bytes"],
        "replacement": transform["replacement"],
        "checks": transform["checks"],
    }
    binding_checks = {
        "source_equals_claim": binding_summary(source)
        == attempt_payload["source"],
        "dependencies_equal_claim": {
            name: binding_summary(record) for name, record in dependencies.items()
        }
        == attempt_payload["dependencies"],
        "terminal_state_equals_claim": terminal_summary(terminal)
        == attempt_payload["frozen_v15_terminal_state"],
        "transform_equals_claim": compact_transform
        == attempt_payload["in_memory_v14_contract_transform"],
        "attempt_marker_unchanged": attempt_post["sha256"]
        == attempt["publication"]["sha256"],
        "v16_output_absent": path_absent(OUTPUT),
        "v15_output_absent": path_absent(V15_OUTPUT),
        "v14_output_absent": path_absent(V14_OUTPUT),
        "v13_output_absent": path_absent(V13_OUTPUT),
    }
    if not all(binding_checks.values()):
        raise ProtocolError(f"v16 failure binding drift: {binding_checks}")
    error_type = f"{type(error).__module__}.{type(error).__qualname__}"
    message = str(error).encode("utf-8", errors="backslashreplace")
    fingerprint = hashlib.sha256(error_type.encode() + b"\0" + message).hexdigest()
    record = {
        "schema_version": f"{SCHEMA}-failure",
        "status": "FAILURE_V16_BEFORE_RESULT_PUBLICATION",
        "error_type": error_type,
        "error_type_and_message_sha256": fingerprint,
        "error_message_bytes": len(message),
        "source": source,
        "dependencies": dependencies,
        "frozen_v15_terminal_state": terminal,
        "in_memory_v14_contract_transform": transform,
        "one_shot_attempt": {
            "claim_publication": attempt["publication"],
            "post_failure_rehash": attempt_post,
        },
        "binding_checks": binding_checks,
        "candidate_payload": None,
        "actor_model_delta_material_present": False,
        "checkpoint_writes": 0,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    no_go_material_audit(record)
    canonical_json(record)
    return record


def audit_only() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v16 output target must be absent during static audit")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v16 attempt marker must be absent during static audit")
    if not path_absent(FAILURE):
        raise ProtocolError("v16 failure target must be absent during static audit")
    source = self_evidence(require_frozen=False)
    dependencies = dependency_evidence()
    terminal = validate_v15_terminal_state()
    transform = transform_v14_source()[1]
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_pass_unfrozen_draft",
        "source": source,
        "source_audit": source_audit(),
        "dependencies": dependencies,
        "frozen_v15_terminal_state": terminal,
        "in_memory_v14_contract_transform": transform,
        "design": {
            "frozen_v15_production_reused": True,
            "only_callback_override": "load_v14_module",
            "only_v14_source_change": (
                "truthiness_contract_validator_to_typed_exact_predicates"
            ),
            "five_intentionally_falsy_fields_checked_exactly": [
                "new_validation_or_test_rows_opened",
                "checkpoint_writes",
                "model_artifact_writes",
                "official_unique_changed_candidate_count_consumed",
                "submission_performed",
            ],
            "training_transform_sha256_unchanged": V15_TRANSFORMED_V13_SHA256,
            "algorithm_change": None,
        },
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent": True,
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
        "attempt_marker_absent": True,
        "failure": str(FAILURE.relative_to(ROOT)),
        "failure_absent": True,
        "v15_v14_v13_outputs_absent": True,
        "pre_cuda_runtime": runtime,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "CUDA_initialized": False,
        "production_modules_executed": False,
        "writes_performed": 0,
        "draft_freeze_required_before_production": source["mode_octal"] != "0555",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        raise ProtocolError("output path is not the exact v16 target")
    if args.audit_only:
        print(canonical_json(audit_only()).decode("utf-8"), end="")
        return
    if not path_absent(OUTPUT):
        raise ProtocolError("v16 O_EXCL output already exists")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v16 one-shot attempt marker already exists")
    if not path_absent(FAILURE):
        raise ProtocolError("v16 O_EXCL failure target already exists")
    attempt = claim_attempt()
    try:
        result = production_run()
    except BaseException as error:
        failure = failure_record(error, attempt)
        publish_o_excl(FAILURE, canonical_json(failure))
        raise
    attempt_post = regular_source(
        ATTEMPT_MARKER,
        attempt["publication"]["sha256"],
        0o444,
        "v16 attempt marker post-run",
    )[1]
    final_source = self_evidence(require_frozen=True)
    final_dependencies = dependency_evidence()
    final_terminal = validate_v15_terminal_state()
    final_transform = transform_v14_source()[1]
    inputs = result["inputs"]
    attempt_source = attempt["payload"]["source"]
    attempt_dependencies = attempt["payload"]["dependencies"]
    attempt_terminal = attempt["payload"]["frozen_v15_terminal_state"]
    attempt_transform = attempt["payload"]["in_memory_v14_contract_transform"]

    def compact_transform(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "transformed_sha256": record["transformed_sha256"],
            "transformed_bytes": record["transformed_bytes"],
            "replacement": record["replacement"],
            "checks": record["checks"],
        }

    binding_checks = {
        "attempt_source_equals_production_pre": attempt_source
        == binding_summary(inputs["CW24_v16_source"]),
        "attempt_source_equals_production_post": attempt_source
        == binding_summary(inputs["CW24_v16_source_post_run"]),
        "attempt_source_equals_final": attempt_source == binding_summary(final_source),
        "attempt_dependencies_equal_production_pre": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in inputs["CW24_v16_dependencies"].items()
        },
        "attempt_dependencies_equal_production_post": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in inputs["CW24_v16_dependencies_post_run"].items()
        },
        "attempt_dependencies_equal_final": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in final_dependencies.items()
        },
        "attempt_terminal_equals_production_pre": attempt_terminal
        == terminal_summary(inputs["CW24_v16_frozen_v15_terminal_state"]),
        "attempt_terminal_equals_production_post": attempt_terminal
        == terminal_summary(
            inputs["CW24_v16_frozen_v15_terminal_state_post_run"]
        ),
        "attempt_terminal_equals_final": attempt_terminal
        == terminal_summary(final_terminal),
        "attempt_transform_equals_production_pre": attempt_transform
        == compact_transform(
            inputs["CW24_v16_in_memory_v14_contract_transform"]
        ),
        "attempt_transform_equals_production_post": attempt_transform
        == compact_transform(
            inputs["CW24_v16_in_memory_v14_contract_transform_post_run"]
        ),
        "attempt_transform_equals_final": attempt_transform
        == compact_transform(final_transform),
        "v15_output_still_absent": path_absent(V15_OUTPUT),
        "v14_output_still_absent": path_absent(V14_OUTPUT),
        "v13_output_still_absent": path_absent(V13_OUTPUT),
        "failure_output_still_absent": path_absent(FAILURE),
    }
    if not all(binding_checks.values()):
        raise ProtocolError(f"v16 attempt-to-publication drift: {binding_checks}")
    inputs["CW24_v16_source_final_before_publication"] = final_source
    inputs["CW24_v16_dependencies_final_before_publication"] = final_dependencies
    inputs["CW24_v16_frozen_v15_terminal_state_final_before_publication"] = (
        final_terminal
    )
    inputs["CW24_v16_in_memory_v14_contract_transform_final"] = final_transform
    result.setdefault("audit", {})["one_shot_attempt"] = {
        "claim": attempt,
        "post_run": attempt_post,
        "unchanged": attempt_post["sha256"] == attempt["publication"]["sha256"],
        "claim_to_publication_binding_checks": binding_checks,
    }
    if str(result["decision"]).startswith("NO_GO"):
        result["audit"]["NO_GO_material_audit_final"] = no_go_material_audit(
            result
        )
        no_go_material_audit(result)
    canonical_json(result)
    if not path_absent(V15_OUTPUT) or not path_absent(V14_OUTPUT) or not path_absent(
        V13_OUTPUT
    ):
        raise ProtocolError("frozen predecessor output appeared before publication")
    if not path_absent(FAILURE):
        raise ProtocolError("v16 failure output appeared before result publication")
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    endpoint = result["endpoint"]
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "candidate_payload_present": endpoint.get("candidate_payload")
                is not None,
                "changed_candidate_train_shadow_count": endpoint.get(
                    "changed_candidate_train_shadow_count"
                ),
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
                "output": publication,
            }
        ).decode("utf-8"),
        end="",
    )


if __name__ == "__main__":
    main()

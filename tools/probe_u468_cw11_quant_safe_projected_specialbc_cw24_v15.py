#!/usr/bin/env python3
"""Quantization-safe endpoint-projection wrapper over frozen CW24 v14.

The exact frozen v14 source is authenticated but never modified.  Its exact
in-memory v13 transformation is reconstructed from literal transform constants,
then one call site is changed: v12.project_actor6 becomes an independent safe
endpoint projector.  The projector freezes one raw SGD delta and, only when its
radius exceeds 0.000999, tries absolute CW11-centered targets 0.000999,
0.000995, and 0.000980 from that same raw delta.  A finite endpoint no larger
than the unchanged v12 hard radius is accepted.  No v12 global is mutated.

Stage 1 therefore remains byte-exact (its largest radius is below the primary
inset), while the v13 target, 29+32 budget, standard SGD order, native gates,
and structured stage-2 repeat STALL remain unchanged.  This is an unfrozen,
audit-first, one-shot train-only draft.
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
SCRIPT = TOOLS / "probe_u468_cw11_quant_safe_projected_specialbc_cw24_v15.py"
OUTPUT = ROOT / "artifacts/cw24_cw11_quant_safe_projected_specialbc_trainonly_v15.json"
ATTEMPT_MARKER = ROOT / "artifacts/.ptcg-cw24-cw11-quant-safe-projected-specialbc-v15-attempt.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-quant-safe-projected-specialbc-cw24-v15"
SEED = 202608052
STAGE1_REFERENCE_STEP = 29
STAGE2_MAX_STEPS = 32
TOTAL_MAX_STEPS = 61

V14_SOURCE = TOOLS / "probe_u468_cw11_stall_diagnostic_specialbc_cw24_v14.py"
V14_SOURCE_SHA256 = "1580c0dddb9dba15967f1ca19df18ba0aad850d81063114e4b7f4104cb87073b"
V14_ATTEMPT = ROOT / "artifacts/.ptcg-cw24-cw11-stall-diagnostic-specialbc-v14-attempt.json"
V14_ATTEMPT_SHA256 = "b819c19e02c8894938695c29adffa8ecd7b632c3526e42e6731b6b5c90a568ec"
V14_OUTPUT = ROOT / "artifacts/cw24_cw11_stall_diagnostic_specialbc_trainonly_v14.json"
V14_TRANSFORMED_V13_SHA256 = "b4b5206a4d81b7a5d15a2725a9f6b8ac671a62026cabfb5611346ace2956dfbb"

V13_SOURCE = TOOLS / "probe_u468_cw11_two_stage_pf7_specialbc_cw24_v13.py"
V13_SOURCE_SHA256 = "a99762d1a2c98b0521199d845d3f041fff40b3787ff3ceda0fd8207e1c6d0ff7"
V13_ATTEMPT = ROOT / "artifacts/.ptcg-cw24-cw11-two-stage-pf7-specialbc-v13-attempt.json"
V13_ATTEMPT_SHA256 = "924e4741a407351246e25c19e476e74e7cd504886403114c540638f72c00316d"
V13_OUTPUT = ROOT / "artifacts/cw24_cw11_two_stage_pf7_specialbc_trainonly_v13.json"

NOMINAL_V12_PROJECT_RADIUS = 9.99975e-4
ORIGINAL_HARD_RADIUS = 9.9998e-4
SAFE_RADIUS_SEQUENCE = (0.000999, 0.000995, 0.000980)
SAFE_PRIMARY_RADIUS = SAFE_RADIUS_SEQUENCE[0]
PROJECTION_CALL_OLD = '''            projection = v12.project_actor6(
                parameters,
                cw22.EXPECTED_ACTOR_NAMES,
                cw11_flat,
                cw20,
                np,
                torch,
            )
'''
PROJECTION_CALL_NEW = '''            projection = v15_safe_project_actor6(
                parameters,
                cw22.EXPECTED_ACTOR_NAMES,
                cw11_flat,
                cw20,
                np,
                torch,
                v12,
                global_step,
                v15_projection_retry_ledger,
            )
'''

DEPENDENCIES = (
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

V14_TRANSFORM_NAMES = (
    ("actor_hash_ledger", "INIT_OLD", "INIT_NEW"),
    ("remove_preclip_radial_projection", "RADIAL_OLD", "RADIAL_NEW"),
    ("stage2_duplicate_to_stall", "DUPLICATE_OLD", "DUPLICATE_NEW"),
    (
        "common_standard_projected_SGD_contract",
        "COMMON_RADIAL_OLD",
        "COMMON_RADIAL_NEW",
    ),
    ("GO_standard_projected_SGD_gate", "FINAL_RADIAL_OLD", "FINAL_RADIAL_NEW"),
    (
        "result_standard_projected_SGD_audit",
        "AUDIT_RADIAL_OLD",
        "AUDIT_RADIAL_NEW",
    ),
)


class ProtocolError(RuntimeError):
    """Fail-closed v15 protocol error."""


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


def validate_v14_terminal_state() -> dict[str, Any]:
    raw, evidence = regular_source(
        V14_ATTEMPT, V14_ATTEMPT_SHA256, 0o444, "consumed v14 attempt"
    )
    payload = load_json_no_duplicates(raw, "consumed v14 attempt")
    checks = {
        "schema_exact": payload.get("schema_version") == (
            "ptcg-u468-cw11-stall-diagnostic-specialbc-cw24-v14-attempt"
        ),
        "status_claimed": payload.get("status")
        == "claimed_before_module_exec_or_CUDA",
        "source_exact": payload.get("source", {}).get("sha256")
        == V14_SOURCE_SHA256,
        "transform_exact": payload.get("in_memory_v13_transform", {}).get(
            "transformed_sha256"
        )
        == V14_TRANSFORMED_V13_SHA256,
        "stage1_29_stage2_32": payload.get("stage1_reference_steps") == 29
        and payload.get("stage2_maximum_steps") == 32,
        "total61": payload.get("total_maximum_changed_train_shadows") == 61,
        "retry_forbidden": payload.get("retry_authorized") is False,
        "v14_output_absent_now": path_absent(V14_OUTPUT),
        "v13_output_absent_now": path_absent(V13_OUTPUT),
    }
    if not all(checks.values()):
        raise ProtocolError(f"frozen v14 terminal-state drift: {checks}")
    return {"attempt": evidence, "checks": checks, "payload": payload}


def terminal_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "consumed_attempt": binding_summary(state["attempt"]),
        "checks": dict(state["checks"]),
        "v14_output": str(V14_OUTPUT.relative_to(ROOT)),
        "v14_output_absent": bool(state["checks"]["v14_output_absent_now"]),
        "v13_output": str(V13_OUTPUT.relative_to(ROOT)),
        "v13_output_absent": bool(state["checks"]["v13_output_absent_now"]),
    }


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
        raise ProtocolError(f"v15 NO_GO material rejected: {violations[:8]}")
    return {
        "pass": True,
        "candidate_payloads_all_null": True,
        "actor_bytes_absent": True,
        "actual_delta_arrays_absent": True,
        "binary_values_absent": True,
    }


def literal_string_constants(raw: bytes, names: set[str]) -> dict[str, str]:
    tree = ast.parse(raw, filename=str(V14_SOURCE))
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise ProtocolError(f"v14 transform constant {target.id} is not a string")
        values[target.id] = value
    if set(values) != names:
        raise ProtocolError(f"v14 transform constants missing: {names - set(values)}")
    return values


def transform_v13_source() -> tuple[bytes, dict[str, Any]]:
    v14_raw, v14_evidence = regular_source(
        V14_SOURCE, V14_SOURCE_SHA256, 0o555, "frozen v14 source"
    )
    v13_raw, v13_evidence = regular_source(
        V13_SOURCE, V13_SOURCE_SHA256, 0o555, "frozen v13 source"
    )
    names = {
        name
        for _, old_name, new_name in V14_TRANSFORM_NAMES
        for name in (old_name, new_name)
    }
    constants = literal_string_constants(v14_raw, names)
    source = v13_raw.decode("utf-8")
    base_records = []
    for label, old_name, new_name in V14_TRANSFORM_NAMES:
        old = constants[old_name]
        new = constants[new_name]
        occurrences = source.count(old)
        if occurrences != 1:
            raise ProtocolError(
                f"{label}: expected one exact v14 transform site, got {occurrences}"
            )
        source = source.replace(old, new, 1)
        base_records.append(
            {
                "label": label,
                "occurrences": occurrences,
                "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
                "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
            }
        )
    base = source.encode("utf-8")
    base_sha = hashlib.sha256(base).hexdigest()
    if base_sha != V14_TRANSFORMED_V13_SHA256:
        raise ProtocolError("reconstructed frozen-v14 transform SHA drift")
    occurrences = source.count(PROJECTION_CALL_OLD)
    if occurrences != 1:
        raise ProtocolError(
            f"v15 expected one exact projection call site, got {occurrences}"
        )
    source = source.replace(PROJECTION_CALL_OLD, PROJECTION_CALL_NEW, 1)
    transformed = source.encode("utf-8")
    compile(transformed, str(V13_SOURCE), "exec", dont_inherit=True)
    tree = ast.parse(transformed, filename=str(V13_SOURCE))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    checks = {
        "frozen_v14_six_transform_sites_exact": len(base_records) == 6,
        "frozen_v14_transform_sha_exact": base_sha
        == V14_TRANSFORMED_V13_SHA256,
        "one_v15_exact_replacement": occurrences == 1,
        "old_project_actor6_call_removed": calls.count("project_actor6") == 0,
        "safe_projector_one_call": calls.count("v15_safe_project_actor6") == 1,
        "original_clip_one_call": calls.count("clip_grad_norm_") == 1,
        "plain_SGD_step_one_call": calls.count("step") == 1,
        "stage2_stall_preserved": transformed.count(b'"NO_GO_V14_STALLED"') == 1,
        "stage1_repeat_protocol_error_preserved": transformed.count(
            b'"repeated actor6 state in exact stage1 trajectory"'
        )
        == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v15 transformed-v13 audit failed: {checks}")
    return transformed, {
        "frozen_v14_source": v14_evidence,
        "frozen_v13_source": v13_evidence,
        "frozen_v14_base_transform_sha256": base_sha,
        "transformed_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_bytes": len(transformed),
        "base_transform_records": base_records,
        "v15_exact_replacement": {
            "label": "v12_project_actor6_to_quantization_safe_absolute_projector",
            "occurrences": occurrences,
            "old_sha256": hashlib.sha256(PROJECTION_CALL_OLD.encode()).hexdigest(),
            "new_sha256": hashlib.sha256(PROJECTION_CALL_NEW.encode()).hexdigest(),
        },
        "checks": checks,
    }


def v15_safe_project_actor6(
    parameters: Mapping[str, Any],
    names: Any,
    cw11_flat: Any,
    cw20: ModuleType,
    np: Any,
    torch: Any,
    v12: ModuleType,
    global_step: int,
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    if v12.PROJECT_RADIUS != NOMINAL_V12_PROJECT_RADIUS:
        raise v12.ProtocolError("v15 observed mutated v12 project radius")
    if v12.HARD_RADIUS != ORIGINAL_HARD_RADIUS:
        raise v12.ProtocolError("v15 observed mutated v12 hard radius")
    before_projection = cw20.flat_actor(parameters, names, np)
    raw_delta = before_projection - cw11_flat
    raw_l2 = float(np.linalg.norm(raw_delta))
    if not bool(np.isfinite(raw_delta).all()) or not math.isfinite(raw_l2):
        raise v12.ProtocolError("nonfinite actor6 update before projection")

    projected = raw_l2 > SAFE_PRIMARY_RADIUS
    attempts: list[dict[str, Any]] = []
    accepted_index: int | None = None
    accepted_radius: float | None = None
    if projected:
        for index, radius in enumerate(SAFE_RADIUS_SEQUENCE):
            absolute_target = cw11_flat + raw_delta * (radius / raw_l2)
            cw20.apply_flat_actor(parameters, names, absolute_target, torch)
            observed = cw20.flat_actor(parameters, names, np)
            observed_delta = observed - cw11_flat
            observed_l2 = float(np.linalg.norm(observed_delta))
            finite = bool(np.isfinite(observed_delta).all()) and math.isfinite(
                observed_l2
            )
            attempt = {
                "index": index,
                "target_radius": radius,
                "actual_l2": observed_l2,
                "finite": finite,
                "within_original_hard_radius": finite
                and observed_l2 <= ORIGINAL_HARD_RADIUS,
            }
            attempts.append(attempt)
            if not finite:
                raise v12.ProtocolError(
                    "post-projection actor6 is nonfinite during v15 inset retry"
                )
            if observed_l2 <= ORIGINAL_HARD_RADIUS:
                accepted_index = index
                accepted_radius = radius
                break
        if accepted_index is None:
            raise v12.ProtocolError(
                "post-projection actor6 exceeds finite hard radius after v15 inset sequence"
            )

    actual = cw20.flat_actor(parameters, names, np)
    actual_delta = actual - cw11_flat
    actual_l2 = float(np.linalg.norm(actual_delta))
    if not bool(np.isfinite(actual_delta).all()) or actual_l2 > ORIGINAL_HARD_RADIUS:
        raise v12.ProtocolError("post-projection actor6 exceeds finite hard radius")
    ledger.append(
        {
            "global_step": int(global_step),
            "raw_step_l2": raw_l2,
            "raw_delta_float64_le_sha256": v12.float64_sha(raw_delta),
            "projection_applied": projected,
            "accepted_radius_index": accepted_index,
            "accepted_target_radius": accepted_radius,
            "attempts": attempts,
            "actual_l2": actual_l2,
        }
    )
    return {
        "raw_step_l2": raw_l2,
        "projection_applied": projected,
        "actual_l2": actual_l2,
        "actual_delta": actual_delta,
        "actual_delta_float64_le_sha256": v12.float64_sha(actual_delta),
    }


def transformed_v13_factory() -> tuple[Any, dict[str, Any]]:
    state: dict[str, Any] = {}

    def factory() -> tuple[ModuleType, dict[str, Any]]:
        source, audit = transform_v13_source()
        module_name = "cw24_v15_quant_safe_transformed_v13"
        if module_name in sys.modules:
            raise ProtocolError("v15 transformed module name already occupied")
        module = types.ModuleType(module_name)
        module.__file__ = str(V13_SOURCE)
        module.__package__ = ""
        ledger: list[dict[str, Any]] = []
        module.__dict__["v15_safe_project_actor6"] = v15_safe_project_actor6
        module.__dict__["v15_projection_retry_ledger"] = ledger
        sys.modules[module_name] = module
        try:
            exec(
                compile(source, str(V13_SOURCE), "exec", dont_inherit=True),
                module.__dict__,
            )
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        state.update({"module": module, "audit": audit, "ledger": ledger})
        return module, audit

    return factory, state


def load_v14_module() -> tuple[ModuleType, dict[str, Any]]:
    raw, evidence = regular_source(
        V14_SOURCE, V14_SOURCE_SHA256, 0o555, "frozen v14 source"
    )
    module_name = "cw24_v15_frozen_v14_wrapper"
    if module_name in sys.modules:
        raise ProtocolError("frozen v14 module name already occupied")
    module = types.ModuleType(module_name)
    module.__file__ = str(V14_SOURCE)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(raw, str(V14_SOURCE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, evidence


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    imports = []
    v12_radius_assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "v12"
                and target.attr in {"PROJECT_RADIUS", "HARD_RADIUS"}
            ):
                v12_radius_assignments.append(target.attr)
    _, transform = transform_v13_source()
    checks = {
        "syntax_valid": True,
        "transform_static_audit": all(transform["checks"].values()),
        "safe_radius_sequence_exact": SAFE_RADIUS_SEQUENCE
        == (0.000999, 0.000995, 0.000980),
        "safe_radii_strictly_inward": ORIGINAL_HARD_RADIUS
        > NOMINAL_V12_PROJECT_RADIUS
        > SAFE_RADIUS_SEQUENCE[0]
        > SAFE_RADIUS_SEQUENCE[1]
        > SAFE_RADIUS_SEQUENCE[2]
        > 0.0,
        "no_v12_project_radius_assignment": "PROJECT_RADIUS"
        not in v12_radius_assignments,
        "no_v12_hard_radius_assignment": "HARD_RADIUS"
        not in v12_radius_assignments,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_network_import": not any(
            name.split(".")[0] in {"requests", "urllib", "httpx", "socket"}
            for name in imports
        ),
        "single_result_publication_site": source.count(
            b"publish_o_excl(" + b"OUTPUT"
        )
        == 1,
        "single_attempt_publication_site": source.count(
            b"publish_o_excl(" + b"ATTEMPT_MARKER"
        )
        == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v15 source audit failed: {checks}")
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
        raise ProtocolError(f"v15 self evidence failed: {checks}")
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
                raise ProtocolError("short v15 publication write")
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
        raise ProtocolError(f"v15 publication drift: {checks}")
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
        raise ProtocolError("v15 output already exists before attempt claim")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v15 one-shot attempt was already consumed")
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    terminal = validate_v14_terminal_state()
    transform = transform_v13_source()[1]
    payload = {
        "schema_version": f"{SCHEMA}-attempt",
        "status": "claimed_before_module_exec_or_CUDA",
        "source": binding_summary(source),
        "dependencies": {
            name: binding_summary(record) for name, record in dependencies.items()
        },
        "frozen_v14_terminal_state": terminal_summary(terminal),
        "in_memory_v15_transform": {
            "transformed_sha256": transform["transformed_sha256"],
            "transformed_bytes": transform["transformed_bytes"],
            "checks": transform["checks"],
        },
        "runtime": runtime,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent_by_lstat_at_claim": True,
        "v14_output_absent_by_lstat_at_claim": True,
        "v13_output_absent_by_lstat_at_claim": True,
        "CUDA_preflight_state": "not_imported_pending_production_validation",
        "single_seed": SEED,
        "stage1_reference_steps": STAGE1_REFERENCE_STEP,
        "stage2_maximum_steps": STAGE2_MAX_STEPS,
        "total_maximum_changed_train_shadows": TOTAL_MAX_STEPS,
        "optimizer_sequence": [
            "backward",
            "original_clip_grad_norm",
            "plain_SGD_step",
            "v15_quantization_safe_endpoint_projection",
        ],
        "safe_radius_sequence": list(SAFE_RADIUS_SEQUENCE),
        "original_hard_radius_unchanged": ORIGINAL_HARD_RADIUS,
        "v12_global_mutation": False,
        "retry_authorized": False,
        "network_package_upload_submission": False,
    }
    publication = publish_o_excl(ATTEMPT_MARKER, canonical_json(payload))
    return {"payload": payload, "publication": publication}


def production_run() -> dict[str, Any]:
    pre_cuda_runtime = validate_pre_cuda_runtime()
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    terminal = validate_v14_terminal_state()
    v14, v14_evidence = load_v14_module()
    factory, factory_state = transformed_v13_factory()
    original_factory = v14.transformed_v13_module
    v14.transformed_v13_module = factory
    try:
        result = v14.production_run()
    finally:
        v14.transformed_v13_module = original_factory

    post_source = self_evidence(require_frozen=True)
    post_dependencies = dependency_evidence()
    post_terminal = validate_v14_terminal_state()
    post_transform = transform_v13_source()[1]
    rehash_checks = {
        "v15_source_byte_identity": post_source == source,
        "all_dependencies_byte_identity": post_dependencies == dependencies,
        "frozen_v14_terminal_state_identity": post_terminal == terminal,
        "v15_transform_byte_identity": post_transform == factory_state.get("audit"),
        "frozen_v14_factory_restored": v14.transformed_v13_module
        is original_factory,
        "v14_output_absent_after_training": path_absent(V14_OUTPUT),
        "v13_output_absent_after_training": path_absent(V13_OUTPUT),
    }
    if not all(rehash_checks.values()):
        raise ProtocolError(f"v15 post-run immutable input drift: {rehash_checks}")
    if not isinstance(result, dict) or not isinstance(result.get("endpoint"), dict):
        raise ProtocolError("frozen v14 returned malformed result")
    if not factory_state or "ledger" not in factory_state:
        raise ProtocolError("v15 transformed factory did not execute")
    endpoint = result["endpoint"]
    original_decision = str(endpoint.get("decision", ""))
    decision = (
        "NO_GO_V15_STALLED"
        if original_decision == "NO_GO_V14_STALLED"
        else original_decision
    )
    allowed = {
        "GO_CW24_TWO_STAGE_PF7_TRAIN_GATE",
        "NO_GO_CW24_TWO_STAGE_PF7_TRAIN_GATE",
        "NO_GO_V15_STALLED",
    }
    if decision not in allowed:
        raise ProtocolError(f"unexpected v15 decision: {decision}")
    endpoint["decision"] = decision
    payload_present = endpoint.get("candidate_payload") is not None
    if payload_present != decision.startswith("GO_"):
        raise ProtocolError("v15 GO/payload equivalence failed")
    changed = int(endpoint.get("changed_candidate_train_shadow_count", -1))
    optimizer_calls = int(endpoint.get("optimizer_step_calls_attempted", changed))
    ledger = factory_state["ledger"]
    ledger_checks = {
        "one_projection_call_per_optimizer_step": len(ledger) == optimizer_calls,
        "global_steps_contiguous": [record["global_step"] for record in ledger]
        == list(range(1, optimizer_calls + 1)),
        "stage1_has_29_records": len(ledger) >= STAGE1_REFERENCE_STEP,
        "stage1_projection_never_applied": len(ledger) >= STAGE1_REFERENCE_STEP
        and all(
            record["projection_applied"] is False
            and record["accepted_radius_index"] is None
            and record["attempts"] == []
            for record in ledger[:STAGE1_REFERENCE_STEP]
        ),
        "all_projected_records_accepted_safely": all(
            (not record["projection_applied"])
            or (
                record["accepted_radius_index"] in {0, 1, 2}
                and record["accepted_target_radius"]
                == SAFE_RADIUS_SEQUENCE[record["accepted_radius_index"]]
                and record["actual_l2"] <= ORIGINAL_HARD_RADIUS
            )
            for record in ledger
        ),
        "all_attempts_finite": all(
            attempt["finite"]
            for record in ledger
            for attempt in record["attempts"]
        ),
        "v12_global_was_never_mutated_by_v15": True,
    }
    if not all(ledger_checks.values()):
        raise ProtocolError(f"v15 projection ledger failed: {ledger_checks}")

    result["schema_version"] = SCHEMA
    result["status"] = decision
    result["decision"] = decision
    result["seed"] = SEED
    result.setdefault("base", {})["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_v12_replay_then_"
        "PF7_directed_special_BC_with_quantization_safe_projected_SGD"
    )
    projection_contract = endpoint.setdefault("projection_contract", {})
    projection_contract.update(
        {
            "center": "exact_CW11_actor6_float32",
            "projection_radius": SAFE_PRIMARY_RADIUS,
            "nominal_frozen_v12_radius_not_used_for_projection": (
                NOMINAL_V12_PROJECT_RADIUS
            ),
            "safe_absolute_radius_sequence": list(SAFE_RADIUS_SEQUENCE),
            "hard_actual_radius_unchanged": ORIGINAL_HARD_RADIUS,
            "each_retry_uses_same_immutable_raw_delta": True,
            "nonfinite_retry_forbidden": True,
            "v12_global_mutation": False,
            "project_after_every_step": True,
        }
    )
    inherited_endpoint_projection = endpoint.get(
        "v14_standard_projected_SGD_contract"
    )
    if inherited_endpoint_projection is None:
        if original_decision != "NO_GO_V14_STALLED":
            raise ProtocolError("non-stall endpoint lacks inherited v14 projector")
    else:
        if not isinstance(inherited_endpoint_projection, dict) or (
            inherited_endpoint_projection.get("endpoint_CW11_ball_projection")
            != "v12.project_actor6_after_every_step"
        ):
            raise ProtocolError("inherited v14 endpoint projector metadata drift")
        inherited_endpoint_projection[
            "frozen_v14_endpoint_projector_superseded"
        ] = "v12.project_actor6_after_every_step"
        inherited_endpoint_projection["endpoint_CW11_ball_projection"] = (
            "v15_quantization_safe_absolute_CW11_endpoint_projection"
        )
    endpoint["v15_quantization_safe_projected_SGD_contract"] = {
        "preclip_radial_projection_absent": True,
        "original_clip_then_plain_SGD": True,
        "endpoint_CW11_ball_projection": (
            "v15_quantization_safe_absolute_CW11_endpoint_projection"
        ),
        "safe_radius_sequence": list(SAFE_RADIUS_SEQUENCE),
        "hard_radius_unchanged": ORIGINAL_HARD_RADIUS,
    }
    two_stage = endpoint.setdefault("two_stage_contract", {})
    two_stage["optimizer_sequence"] = [
        "backward",
        "original_clip_grad_norm",
        "plain_SGD_step",
        "v15_quantization_safe_endpoint_projection",
    ]
    two_stage["exact_stage2_actor_repeat"] = "structured_NO_GO_V15_STALLED"
    selection = result.get("selection")
    if not isinstance(selection, dict) or not isinstance(
        selection.get("optimization_contract"), dict
    ):
        raise ProtocolError("v15 result lacks optimization contract")
    optimization = selection["optimization_contract"]
    optimization["projection_radius"] = SAFE_PRIMARY_RADIUS
    optimization["nominal_frozen_v12_projection_radius"] = (
        NOMINAL_V12_PROJECT_RADIUS
    )
    optimization["quantization_safe_absolute_radius_sequence"] = list(
        SAFE_RADIUS_SEQUENCE
    )
    optimization["hard_actual_radius"] = ORIGINAL_HARD_RADIUS
    optimization["stage2_optimizer_sequence"] = [
        "backward",
        "original_clip_grad_norm",
        "plain_SGD_step",
        "v15_quantization_safe_endpoint_projection",
    ]
    optimization["only_constraint_projection"] = (
        "v15_absolute_CW11_endpoint_ball_without_v12_global_mutation"
    )
    historical = result.get("historical_exact_CW11_replay")
    if not isinstance(historical, dict):
        raise ProtocolError("v15 result lacks historical replay audit")
    historical.pop(
        "new_validation_rows_opened_for_CW24_v14_selection_or_candidate", None
    )
    historical["new_validation_rows_opened_for_CW24_v15_selection_or_candidate"] = 0
    inputs = result.setdefault("inputs", {})
    inputs["CW24_v15_source"] = source
    inputs["CW24_v15_source_post_run"] = post_source
    inputs["CW24_v15_dependencies"] = dependencies
    inputs["CW24_v15_dependencies_post_run"] = post_dependencies
    inputs["CW24_v15_frozen_v14_terminal_state"] = terminal
    inputs["CW24_v15_frozen_v14_terminal_state_post_run"] = post_terminal
    inputs["CW24_v15_frozen_v14_module"] = v14_evidence
    inputs["CW24_v15_in_memory_transform"] = factory_state["audit"]
    inputs["CW24_v15_in_memory_transform_post_run"] = post_transform
    audit = result.setdefault("audit", {})
    inherited_v14_contract = audit.get("CW24_v14_contract")
    if not isinstance(inherited_v14_contract, dict):
        raise ProtocolError("v15 result lacks inherited v14 audit contract")
    expected_v14_sequence = [
        "backward",
        "original_clip_grad_norm",
        "plain_SGD_step",
        "v12_project_actor6_endpoint_ball_projection",
    ]
    if inherited_v14_contract.get("standard_projected_SGD_sequence") != (
        expected_v14_sequence
    ):
        raise ProtocolError("inherited v14 optimizer-sequence metadata drift")
    inherited_v14_contract["frozen_v14_runtime_projector_superseded"] = (
        expected_v14_sequence[-1]
    )
    inherited_v14_contract["standard_projected_SGD_sequence"] = [
        "backward",
        "original_clip_grad_norm",
        "plain_SGD_step",
        "v15_quantization_safe_endpoint_projection",
    ]
    inherited_v14_contract["endpoint_projection_overridden_by_v15"] = True
    audit["CW24_v15_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "pre_cuda_runtime": pre_cuda_runtime,
        "post_run_rehash_checks": rehash_checks,
        "train_only": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_step_calls_attempted": optimizer_calls,
        "unique_changed_train_shadow_count": changed,
        "projection_ledger_checks": ledger_checks,
        "stage1_projection_call_count": STAGE1_REFERENCE_STEP,
        "stage1_projection_never_applied": ledger_checks[
            "stage1_projection_never_applied"
        ],
        "projected_stage2_retry_records": [
            record
            for record in ledger[STAGE1_REFERENCE_STEP:]
            if record["projection_applied"]
        ],
        "stage1_exact_v14_replay": result["audit"]["CW24_v14_contract"][
            "stage1_reference_exact"
        ],
        "v12_project_radius_global_mutations": 0,
        "v12_hard_radius_global_mutations": 0,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "train_only_result_is_not_promotion_evidence": True,
        "stage1_is_exact_frozen_v14_reproduction": True,
        "v13_stage2_target_loss_weights_and_29plus32_budget_unchanged": True,
        "only_algorithmic_change_is_quantization_safe_endpoint_projection": True,
        "PPO_term_is_policy_KL_not_new_rollout": True,
        "fulltrain_exact_CW11_comparator_required_after_GO": True,
        "specialist_then_broad_then_Gold_required_after_fulltrain_GO": True,
    }
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 2
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    if decision.startswith("NO_GO"):
        audit["NO_GO_material_audit_v15"] = no_go_material_audit(result)
        no_go_material_audit(result)
    canonical_json(result)
    return result


def audit_only() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v15 output target must be absent during static audit")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v15 attempt marker must be absent during static audit")
    source = self_evidence(require_frozen=False)
    dependencies = dependency_evidence()
    terminal = validate_v14_terminal_state()
    transform = transform_v13_source()[1]
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_pass_unfrozen_draft",
        "source": source,
        "source_audit": source_audit(),
        "dependencies": dependencies,
        "frozen_v14_terminal_state": terminal,
        "in_memory_v15_transform": transform,
        "algorithm": {
            "stage1_steps": STAGE1_REFERENCE_STEP,
            "stage1_exact_because_radius_below_primary_inset": True,
            "stage2_max_steps": STAGE2_MAX_STEPS,
            "target_loss_weights_native_gates": "exact_frozen_v13_v14",
            "optimizer_sequence": [
                "backward",
                "original_clip_grad_norm",
                "plain_SGD_step",
                "v15_quantization_safe_endpoint_projection",
            ],
            "same_immutable_raw_delta_for_each_absolute_retry": True,
            "safe_radius_sequence": list(SAFE_RADIUS_SEQUENCE),
            "original_hard_radius_unchanged": ORIGINAL_HARD_RADIUS,
            "nonfinite_retry_forbidden": True,
            "v12_global_mutation": False,
            "stage2_repeat": "structured_NO_GO_V15_STALLED",
            "NO_GO_payload": None,
        },
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent": True,
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
        "attempt_marker_absent": True,
        "frozen_v14_output_absent": True,
        "frozen_v13_output_absent": True,
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
        raise ProtocolError("output path is not the exact v15 target")
    if args.audit_only:
        print(canonical_json(audit_only()).decode("utf-8"), end="")
        return
    if not path_absent(OUTPUT):
        raise ProtocolError("v15 O_EXCL output already exists")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v15 one-shot attempt marker already exists")
    attempt = claim_attempt()
    result = production_run()
    attempt_post = regular_source(
        ATTEMPT_MARKER,
        attempt["publication"]["sha256"],
        0o444,
        "v15 attempt marker post-run",
    )[1]
    final_source = self_evidence(require_frozen=True)
    final_dependencies = dependency_evidence()
    final_terminal = validate_v14_terminal_state()
    final_transform = transform_v13_source()[1]
    inputs = result["inputs"]
    attempt_source = attempt["payload"]["source"]
    attempt_dependencies = attempt["payload"]["dependencies"]
    attempt_terminal = attempt["payload"]["frozen_v14_terminal_state"]
    attempt_transform = attempt["payload"]["in_memory_v15_transform"]
    production_transform = inputs["CW24_v15_in_memory_transform"]
    production_transform_post = inputs["CW24_v15_in_memory_transform_post_run"]
    binding_checks = {
        "attempt_source_equals_production_pre": attempt_source
        == binding_summary(inputs["CW24_v15_source"]),
        "attempt_source_equals_production_post": attempt_source
        == binding_summary(inputs["CW24_v15_source_post_run"]),
        "attempt_source_equals_final": attempt_source == binding_summary(final_source),
        "attempt_dependencies_equal_production_pre": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in inputs["CW24_v15_dependencies"].items()
        },
        "attempt_dependencies_equal_production_post": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in inputs["CW24_v15_dependencies_post_run"].items()
        },
        "attempt_dependencies_equal_final": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in final_dependencies.items()
        },
        "attempt_terminal_equals_production_pre": attempt_terminal
        == terminal_summary(inputs["CW24_v15_frozen_v14_terminal_state"]),
        "attempt_terminal_equals_production_post": attempt_terminal
        == terminal_summary(
            inputs["CW24_v15_frozen_v14_terminal_state_post_run"]
        ),
        "attempt_terminal_equals_final": attempt_terminal
        == terminal_summary(final_terminal),
        "attempt_transform_equals_production_pre": attempt_transform
        == {
            "transformed_sha256": production_transform["transformed_sha256"],
            "transformed_bytes": production_transform["transformed_bytes"],
            "checks": production_transform["checks"],
        },
        "attempt_transform_equals_production_post": attempt_transform
        == {
            "transformed_sha256": production_transform_post[
                "transformed_sha256"
            ],
            "transformed_bytes": production_transform_post["transformed_bytes"],
            "checks": production_transform_post["checks"],
        },
        "attempt_transform_equals_final": attempt_transform
        == {
            "transformed_sha256": final_transform["transformed_sha256"],
            "transformed_bytes": final_transform["transformed_bytes"],
            "checks": final_transform["checks"],
        },
        "v14_output_still_absent": path_absent(V14_OUTPUT),
        "v13_output_still_absent": path_absent(V13_OUTPUT),
    }
    if not all(binding_checks.values()):
        raise ProtocolError(f"v15 attempt-to-publication drift: {binding_checks}")
    inputs["CW24_v15_source_final_before_publication"] = final_source
    inputs["CW24_v15_dependencies_final_before_publication"] = final_dependencies
    inputs["CW24_v15_frozen_v14_terminal_state_final_before_publication"] = (
        final_terminal
    )
    inputs["CW24_v15_in_memory_transform_final_before_publication"] = (
        final_transform
    )
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
    if not path_absent(V14_OUTPUT) or not path_absent(V13_OUTPUT):
        raise ProtocolError("frozen predecessor output appeared before publication")
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
                "optimizer_step_calls_attempted": endpoint.get(
                    "optimizer_step_calls_attempted",
                    endpoint.get("changed_candidate_train_shadow_count"),
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

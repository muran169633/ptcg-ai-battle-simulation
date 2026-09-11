#!/usr/bin/env python3
"""Consume exactly one CW14 official-six anchored-total attempt.

CW14 hash-binds the final CW13 one-shot launcher and executes its exact held
bytes.  The adapter replaces only the solver/schema/artifact bindings, expands
the frozen solver manifest from the prior 31 records to those same 31 plus the
three immutable CW13 terminal artifacts, and independently validates that
closed predecessor.  Run mode remains fail-closed until the final CW14 solver
SHA and terminal contract are installed.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
ARTIFACTS = ROOT / "artifacts"
SCRIPT = TOOLS / "run_cw14_consumed_valid_official6_cuttingplane_one_shot_v1.py"
SOLVER = TOOLS / (
    "probe_u468_cw11_consumed_valid_official6_cw14_cuttingplane_v1.py"
)
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_EXECUTABLE_MODE = 0o555

PARENT_LAUNCHER = TOOLS / (
    "run_cw13_consumed_valid_official6_cuttingplane_one_shot_v1.py"
)
PARENT_LAUNCHER_SHA256 = (
    "d170a1a6aefcbdaac86d76bbe5ab64a5cf915ffa20cb2ab84b790ec5124636bf"
)
CW13_SOLVER_SHA256 = (
    "dbbdc12e7c2f571f82de92d30450d8ac794c7215ee86137b9aa97ebc42d28297"
)

CW13_ATTEMPT_MARKER = ARTIFACTS / (
    ".ptcg-cw13_consumed_valid_official6_cuttingplane_20260802_v1-attempt.json"
)
CW13_ATTEMPT_MARKER_SHA256 = (
    "d9e069424ce6ffa49d9617838f085ebdf2723951caeb605e2b9ad6406e3efa34"
)
CW13_CLOSED_STDOUT = ARTIFACTS / (
    "cw13_consumed_valid_official6_cuttingplane_20260802_v1.stdout.json"
)
CW13_CLOSED_STDOUT_SHA256 = (
    "6e92ad787adb2ce9a074c3a4cebaf1ce8e5dd9d9174db26822f524c7e85e0c7b"
)
CW13_STDERR_AUDIT = ARTIFACTS / (
    "cw13_consumed_valid_official6_cuttingplane_20260802_v1.stderr-audit.json"
)
CW13_STDERR_AUDIT_SHA256 = (
    "f72537d134612242865c7fe689f20e740deb542959ac57e68ce426fec7255193"
)
CW13_CHILD_STDERR_SHA256 = (
    "5ceb096b8ac5376a3ee276bf70012f90cb2fd479e3fa2804f2c0a8145aff4512"
)
CW13_CHILD_STDERR_BYTES = 347

SCHEMA = "ptcg-cw14-consumed-valid-official6-one-shot-launcher-v1"
ATTEMPT_SCHEMA = "ptcg-cw14-consumed-valid-official6-one-shot-attempt-v1"
FAILURE_SCHEMA = "ptcg-cw14-consumed-valid-official6-one-shot-terminal-failure-v1"
STDERR_SCHEMA = "ptcg-cw14-consumed-valid-official6-one-shot-stderr-audit-v1"
SOLVER_SCHEMA = (
    "ptcg-cw14-consumed-valid-official-b256-anchored-total-cuttingplane-v1"
)

ARTIFACT_ID = "cw14_consumed_valid_official6_cuttingplane_20260802_v1"
ATTEMPT_MARKER = ARTIFACTS / f".ptcg-{ARTIFACT_ID}-attempt.json"
STDOUT_OUTPUT = ARTIFACTS / f"{ARTIFACT_ID}.stdout.json"
STDERR_AUDIT = ARTIFACTS / f"{ARTIFACT_ID}.stderr-audit.json"

# Exact final independently audited CW14 anchored-total solver lock.
EXPECTED_SOLVER_SHA256 = (
    "febfc16225cc0b915fead337cbf140ff9b4c94d1f5b1cf12921f51b3b340e51f"
)
EXPECTED_STATIC_STATUS = "static_ready_CW14_anchored_total_run_implemented"
EXPECTED_TERMINAL_STATUSES = (
    "consumed_valid_CW14_anchored_total_closure_first_feasible",
    "closed_no_CW14_candidate",
)
EXPECTED_SOLVER_FROZEN_BINDINGS = 34

STATIC_ARGV = (
    str(EXPECTED_PYTHON),
    "-I",
    "-B",
    str(SOLVER),
    "--mode",
    "static",
)
RUN_ARGV = (
    str(EXPECTED_PYTHON),
    "-I",
    "-B",
    str(SOLVER),
    "--mode",
    "run",
)


class ProtocolError(RuntimeError):
    """The inherited launcher or CW14 binding is invalid."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def root_relative(path: Path) -> str:
    resolved = Path(os.path.abspath(os.fspath(path)))
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError as error:
        raise ProtocolError(f"path escapes repository: {resolved}") from error


def read_frozen_parent() -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(PARENT_LAUNCHER, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or stat.S_IMODE(before.st_mode) != FROZEN_EXECUTABLE_MODE
        ):
            raise ProtocolError("parent launcher must be frozen 0555 one-link regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        visible = os.lstat(PARENT_LAUNCHER)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    stable = (
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            stat.S_IMODE(before.st_mode),
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            stat.S_IMODE(after.st_mode),
        )
        == (
            visible.st_dev,
            visible.st_ino,
            visible.st_size,
            visible.st_mtime_ns,
            stat.S_IMODE(visible.st_mode),
        )
        and not stat.S_ISLNK(visible.st_mode)
        and stat.S_ISREG(visible.st_mode)
        and int(visible.st_nlink) == 1
        and len(payload) == int(after.st_size)
        and digest == PARENT_LAUNCHER_SHA256
    )
    if not stable:
        raise ProtocolError("frozen parent launcher identity drift")
    return payload, {
        "path": root_relative(PARENT_LAUNCHER),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": "0555",
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
    }


def import_parent(source: bytes) -> ModuleType:
    """Execute only the exact held-fd parent bytes already matched to its lock."""

    module_name = f"_cw14_parent_one_shot_{PARENT_LAUNCHER_SHA256[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(PARENT_LAUNCHER)
    module.__package__ = ""
    module.__spec__ = None
    code = compile(source, str(PARENT_LAUNCHER), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


PARENT_SOURCE, PARENT_EVIDENCE = read_frozen_parent()
parent_adapter = import_parent(PARENT_SOURCE)
PARENT_SOURCE_AUDIT = parent_adapter.wrapper_source_audit(PARENT_SOURCE)
if PARENT_SOURCE_AUDIT.get("pass") is not True:
    raise ProtocolError("frozen CW13 parent launcher source audit failed")
if not isinstance(getattr(parent_adapter, "parent", None), ModuleType):
    raise ProtocolError("frozen CW13 parent did not expose its inherited engine")
engine = parent_adapter.parent

PARENT_COMPOSED_CHECKS = {
    "cw13_script_exact": engine.SCRIPT == PARENT_LAUNCHER,
    "cw13_solver_path_exact": engine.SOLVER.name
    == "probe_u468_cw11_consumed_valid_official6_cw13_cuttingplane_v1.py",
    "cw13_solver_sha_exact": engine.EXPECTED_SOLVER_SHA256 == CW13_SOLVER_SHA256,
    "cw13_manifest_count_exact_31": engine.EXPECTED_SOLVER_FROZEN_BINDINGS == 31,
    "cw13_static_status_exact": engine.EXPECTED_STATIC_STATUS
    == "static_ready_CW13_run_implemented",
    "cw13_success_status_exact": engine.EXPECTED_TERMINAL_STATUSES[0]
    == "consumed_valid_CW13_optimization_closure_first_feasible",
    "cw13_closed_status_exact": engine.EXPECTED_TERMINAL_STATUSES[1]
    == "closed_no_CW13_candidate",
    "cw13_scope_restoration_exact": engine.RUN_SCOPE.get(
        "CW13_favorable_transition_restoration"
    )
    is True,
}
if not all(PARENT_COMPOSED_CHECKS.values()):
    raise ProtocolError(f"frozen CW13 composed contract drift: {PARENT_COMPOSED_CHECKS}")

# Skip only the CW13-specific 31-record wrapper; the exact frozen CW12 core
# preflight beneath it is manifest-count-parametric and is overridden to 34.
_BASE_PREFLIGHT = parent_adapter._PARENT_PREFLIGHT
_PARENT_STDERR_AUDIT = engine.validated_stderr_audit


def wrapper_source_audit(source: bytes) -> dict[str, Any]:
    text = source.decode("utf-8")
    tree = ast.parse(text, filename=str(SCRIPT))
    imported_roots: set[str] = set()
    calls: set[str] = set()
    output_arguments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value not in {"--mode"}
            ):
                output_arguments.append(node.args[0].value)
    forbidden_imports = {"kaggle", "requests", "socket", "torch", "urllib"}
    checks = {
        "parent_launcher_sha_exact": PARENT_EVIDENCE["sha256"]
        == PARENT_LAUNCHER_SHA256,
        "parent_launcher_mode_exact": PARENT_EVIDENCE["mode_octal"] == "0555",
        "parent_launcher_source_audit_pass": PARENT_SOURCE_AUDIT.get("pass") is True,
        "parent_composed_contract_all_true": all(PARENT_COMPOSED_CHECKS.values()),
        "base_preflight_is_exact_frozen_parent_core": _BASE_PREFLIGHT
        is parent_adapter._PARENT_PREFLIGHT,
        "parent_executed_from_exact_held_bytes": "compile" in calls
        and "exec" in calls
        and "spec_from_file_location" not in calls,
        "no_forbidden_imports": not bool(imported_roots.intersection(forbidden_imports)),
        "no_direct_subprocess_call": "run" not in calls,
        "cli_only_mode": not output_arguments,
        "solver_path_exact": SOLVER.name
        == "probe_u468_cw11_consumed_valid_official6_cw14_cuttingplane_v1.py",
        "solver_sha_lock_exact": EXPECTED_SOLVER_SHA256
        == "febfc16225cc0b915fead337cbf140ff9b4c94d1f5b1cf12921f51b3b340e51f",
        "solver_manifest_count_exact_34": EXPECTED_SOLVER_FROZEN_BINDINGS == 34,
        "cw14_static_validator_installed": engine.validate_static_payload
        is cw14_validate_static_payload,
        "cw14_status_contract_exact": EXPECTED_STATIC_STATUS
        == "static_ready_CW14_anchored_total_run_implemented"
        and EXPECTED_TERMINAL_STATUSES
        == (
            "consumed_valid_CW14_anchored_total_closure_first_feasible",
            "closed_no_CW14_candidate",
        ),
        "cw14_scope_contract_exact": engine.RUN_SCOPE
        == {
            "exact_CW11_reconstruction": True,
            "CW14_anchored_minimum_total_from_CW11": True,
            "favorable_transition_restoration": True,
            "authoritative_full_six_official_B256_oracle_each_proposal": True,
            "candidate_RAM_only": True,
            "optimizer_backward_training": False,
            "model_or_result_writes": 0,
            "network_upload_submission": False,
            "broad_or_gold_access": False,
            "specialist_valid_consumed_dev_only": True,
        },
        "static_argv_exact": STATIC_ARGV
        == (
            str(EXPECTED_PYTHON),
            "-I",
            "-B",
            str(SOLVER),
            "--mode",
            "static",
        ),
        "run_argv_exact": RUN_ARGV
        == (
            str(EXPECTED_PYTHON),
            "-I",
            "-B",
            str(SOLVER),
            "--mode",
            "run",
        ),
        "targets_exact": (
            ATTEMPT_MARKER.name
            == ".ptcg-cw14_consumed_valid_official6_cuttingplane_20260802_v1-attempt.json"
            and STDOUT_OUTPUT.name
            == "cw14_consumed_valid_official6_cuttingplane_20260802_v1.stdout.json"
            and STDERR_AUDIT.name
            == "cw14_consumed_valid_official6_cuttingplane_20260802_v1.stderr-audit.json"
        ),
        "targets_distinct_under_artifacts": len(
            {ATTEMPT_MARKER, STDOUT_OUTPUT, STDERR_AUDIT}
        )
        == 3
        and ATTEMPT_MARKER.parent
        == STDOUT_OUTPUT.parent
        == STDERR_AUDIT.parent
        == ARTIFACTS,
        "cw13_terminal_bindings_literal": CW13_ATTEMPT_MARKER_SHA256 in text
        and CW13_CLOSED_STDOUT_SHA256 in text
        and CW13_STDERR_AUDIT_SHA256 in text,
        "inherited_one_shot_paths_called": "static_result" in calls
        and "run_once" in calls,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "parent_launcher": PARENT_EVIDENCE,
        "parent_source_audit": PARENT_SOURCE_AUDIT,
        "parent_composed_checks": PARENT_COMPOSED_CHECKS,
        "imported_roots": sorted(imported_roots),
        "unexpected_cli_arguments": output_arguments,
    }


def apply_cw14_overrides() -> None:
    cw14_run_scope = {
        "exact_CW11_reconstruction": True,
        "CW14_anchored_minimum_total_from_CW11": True,
        "favorable_transition_restoration": True,
        "authoritative_full_six_official_B256_oracle_each_proposal": True,
        "candidate_RAM_only": True,
        "optimizer_backward_training": False,
        "model_or_result_writes": 0,
        "network_upload_submission": False,
        "broad_or_gold_access": False,
        "specialist_valid_consumed_dev_only": True,
    }
    values = {
        "SCRIPT": SCRIPT,
        "SOLVER": SOLVER,
        "SCHEMA": SCHEMA,
        "ATTEMPT_SCHEMA": ATTEMPT_SCHEMA,
        "FAILURE_SCHEMA": FAILURE_SCHEMA,
        "SOLVER_SCHEMA": SOLVER_SCHEMA,
        "ARTIFACT_ID": ARTIFACT_ID,
        "ATTEMPT_MARKER": ATTEMPT_MARKER,
        "STDOUT_OUTPUT": STDOUT_OUTPUT,
        "STDERR_AUDIT": STDERR_AUDIT,
        "EXPECTED_SOLVER_SHA256": EXPECTED_SOLVER_SHA256,
        "EXPECTED_STATIC_STATUS": EXPECTED_STATIC_STATUS,
        "EXPECTED_TERMINAL_STATUSES": EXPECTED_TERMINAL_STATUSES,
        "EXPECTED_SOLVER_FROZEN_BINDINGS": EXPECTED_SOLVER_FROZEN_BINDINGS,
        "STATIC_ARGV": STATIC_ARGV,
        "RUN_ARGV": RUN_ARGV,
        "RUN_SCOPE": cw14_run_scope,
    }
    for name, value in values.items():
        setattr(engine, name, value)
    engine.static_source_audit = wrapper_source_audit


apply_cw14_overrides()


def cw14_validate_static_payload(payload: bytes) -> dict[str, Any]:
    """Validate the CW14 anchored-total static contract, not CW13 step semantics."""

    document = engine.strict_json_object(payload, "CW14 solver static stdout")
    contract = document.get("contract")
    runtime = document.get("runtime")
    forensic = document.get("forensic_contract")
    cw12_failure = document.get("cw12_consumed_failure_evidence")
    cw13_closure = document.get("cw13_consumed_closure_evidence")
    anchored_math = document.get("anchored_total_geometry_self_test")
    frozen = document.get("frozen_inputs")
    expected_contract_subset = {
        "starting_model_sha256": engine.CW11_MODEL_SHA256,
        "parent_CW12_solver_sha256": (
            "e16eb0aa6d3ea0905309210126abd9f1f4cbb2f093b7e9dfdac49271466377ec"
        ),
        "parent_CW13_solver_sha256": CW13_SOLVER_SHA256,
        "CW14_anchored_minimum_total_from_CW11": True,
        "favorable_transition_restoration": True,
        "fixed_repair_physical_rows": 3,
        "fixed_repair_context_guards": 4,
        "legacy_B33_rows": 33,
        "legacy_active_pairs": 34,
        "authoritative_view_order": [
            "pokemonfan",
            "flg",
            "core5",
            "dominic",
            "luca",
            "szlach",
        ],
        "official_batch_size": 256,
        "official_workers": 8,
        "changed_physical_monitor_rows": 47,
        "raw_to_cw11_transition_rows": 22,
        "transition_panel_occurrences": 26,
        "unique_transition_batch_contexts": 25,
        "max_outer_iterations": engine.MAX_OUTER_ITERATIONS,
        "frozen_solver_trust_region_cap_reinterpreted_as_anchored_total": (
            engine.STEP_L2_CAP
        ),
        "previous_to_proposal_delta_l2_cap": engine.STEP_L2_CAP,
        "additional_total_l2_cap_from_cw11": engine.ADDITIONAL_TOTAL_L2_CAP,
        "direct_uncapped_linear_residual_tolerance": 1e-8,
        "uncapped_over_total_cap_closes_before_proposal": True,
        "clipped_vector_must_never_be_applied": True,
        "nested_active_minimum_total_norm_monotone_gate": True,
        "nested_projection_dot_nonnegative_gate": True,
        "prior_active_full_linearization_record_hash_exact": True,
        "threshold_change_from_CW13": False,
        "l2_cap_absolute_tolerance": engine.L2_CAP_ABS_TOL,
        "actor_parameter_names": list(engine.ACTOR6_NAMES),
        "first_feasible": True,
        "candidate_ordering": "fixed_iteration_order_no_best_of_N",
        "terminal_semantics": "consumed_valid_optimization_closure_only",
    }
    checks = {
        "schema_exact": document.get("schema_version") == SOLVER_SCHEMA,
        "status_exact": document.get("status") == EXPECTED_STATIC_STATUS,
        "classification_exact": document.get("classification")
        == engine.CLASSIFICATION,
        "runtime_checks_all_true": isinstance(runtime, Mapping)
        and engine.all_true_checks(runtime.get("checks")),
        "frozen_inputs_exact_34": isinstance(frozen, Mapping)
        and frozen.get("all_exact") is True
        and frozen.get("binding_count") == 34
        and isinstance(frozen.get("records"), list)
        and len(frozen["records"]) == 34,
        "forensic_contract_all_true": isinstance(forensic, Mapping)
        and engine.all_true_checks(forensic.get("checks")),
        "cw12_failure_evidence_all_true": isinstance(cw12_failure, Mapping)
        and cw12_failure.get("pass") is True
        and engine.all_true_checks(cw12_failure.get("checks")),
        "cw13_closure_evidence_all_true": isinstance(cw13_closure, Mapping)
        and cw13_closure.get("pass") is True
        and engine.all_true_checks(cw13_closure.get("checks")),
        "anchored_total_geometry_self_test_all_true": isinstance(
            anchored_math, Mapping
        )
        and anchored_math.get("pass") is True
        and engine.all_true_checks(anchored_math.get("checks"))
        and anchored_math.get("witness", {}).get("cuts")
        == ["x>=0.0008", "x+y>=0.0013"]
        and anchored_math.get("witness", {}).get("cap") == 0.001
        and anchored_math.get("witness", {}).get("anchored_minimum_total_l2", 1.0)
        < 0.001
        < anchored_math.get("witness", {}).get("cumulative_total_l2", 0.0),
        "contract_subset_exact": isinstance(contract, Mapping)
        and all(
            contract.get(key) == value
            for key, value in expected_contract_subset.items()
        ),
        "source_audit_pass": document.get("source_audit", {}).get("pass") is True
        and engine.all_true_checks(document.get("source_audit", {}).get("checks")),
        "run_executed_false": document.get("run_executed") is False,
        "cuda_accessed_false": document.get("cuda_accessed") is False,
        "writes_performed_false": document.get("writes_performed") is False,
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW14 solver static payload contract failed: {checks}")
    return {
        "checks": checks,
        "stdout_sha256": sha256_bytes(payload),
        "stdout_bytes": len(payload),
        "payload": document,
    }


engine.validate_static_payload = cw14_validate_static_payload


def close_float(value: Any, expected: float) -> bool:
    return isinstance(value, (int, float)) and math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1e-15
    )


def known_cw13_terminal_evidence() -> dict[str, Any]:
    marker_raw, marker_evidence = engine.read_regular_stable(
        CW13_ATTEMPT_MARKER,
        "consumed CW13 attempt marker",
        expected_sha256=CW13_ATTEMPT_MARKER_SHA256,
        expected_mode=0o444,
    )
    closed_raw, closed_evidence = engine.read_regular_stable(
        CW13_CLOSED_STDOUT,
        "immutable CW13 closed stdout",
        expected_sha256=CW13_CLOSED_STDOUT_SHA256,
        expected_mode=0o444,
    )
    stderr_raw, stderr_evidence = engine.read_regular_stable(
        CW13_STDERR_AUDIT,
        "immutable CW13 stderr audit",
        expected_sha256=CW13_STDERR_AUDIT_SHA256,
        expected_mode=0o444,
    )
    marker = engine.strict_json_object(marker_raw, "CW13 attempt marker")
    closed = engine.strict_json_object(closed_raw, "CW13 closed stdout")
    stderr = engine.strict_json_object(stderr_raw, "CW13 stderr audit")
    second = closed.get("second_stage")
    terminal = closed.get("terminal")
    final_integrity = closed.get("final_integrity")
    iterations = second.get("iterations") if isinstance(second, Mapping) else None
    iteration_shape = (
        isinstance(iterations, list)
        and len(iterations) == 3
        and [value.get("iteration") for value in iterations] == [0, 1, 2]
    )
    iteration1 = iterations[1] if iteration_shape else {}
    iteration2 = iterations[2] if iteration_shape else {}
    oracle = iteration1.get("oracle") if isinstance(iteration1, Mapping) else None
    harms = oracle.get("new_harms") if isinstance(oracle, Mapping) else None
    contexts = (
        {
            (
                value.get("view"),
                value.get("context_hash"),
                tuple(value.get("physical_identity", [])),
            )
            for value in harms
            if isinstance(value, Mapping)
        }
        if isinstance(harms, list)
        else set()
    )
    context_metrics: dict[tuple[str, str], set[str]] = {}
    if isinstance(harms, list):
        for value in harms:
            key = (str(value.get("view")), str(value.get("context_hash")))
            context_metrics.setdefault(key, set()).add(str(value.get("metric")))
    all_cuts = (
        oracle.get("deterministic_all_separating_cuts", [])
        if isinstance(oracle, Mapping)
        else []
    )
    cut_profiles = {
        (
            str(value.get("view")),
            str(value.get("context_hash")),
            float(value.get("current_margin")),
            float(value.get("threshold_sources", {}).get("frozen_cw11_same_shape")),
            float(
                value.get("threshold_sources", {}).get(
                    "strict_positive_native_bf16_q"
                )
            ),
            float(value.get("threshold")),
        )
        for value in all_cuts
        if isinstance(value, Mapping)
    }
    post_merge = iteration1.get("post_oracle_cut_merge", {})
    frozen = closed.get("frozen_inputs")
    prior_records = frozen.get("records") if isinstance(frozen, Mapping) else None
    post = stderr.get("post_child_integrity")
    try:
        child_stderr = base64.b64decode(
            str(stderr.get("base64", "")).encode("ascii"), validate=True
        )
        child_stderr_utf8 = child_stderr.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        child_stderr = b""
        child_stderr_utf8 = "Traceback (most recent call last):"
    checks = {
        "marker_schema_status_exact": marker.get("schema_version")
        == "ptcg-cw13-consumed-valid-official6-one-shot-attempt-v1"
        and marker.get("status")
        == "one_shot_attempt_consumed_before_cuda_and_official6",
        "marker_launcher_solver_exact": marker.get("launcher", {}).get("sha256")
        == PARENT_LAUNCHER_SHA256
        and marker.get("solver", {}).get("sha256") == CW13_SOLVER_SHA256
        and marker.get("launcher", {}).get("mode_octal") == "0555"
        and marker.get("solver", {}).get("mode_octal") == "0555",
        "marker_targets_exact": marker.get("stdout_output")
        == root_relative(CW13_CLOSED_STDOUT)
        and marker.get("stderr_audit") == root_relative(CW13_STDERR_AUDIT),
        "marker_one_attempt_no_retry": marker.get("attempts_authorized") == 1
        and marker.get("retry_authorized") is False
        and marker.get("marker_created_before_cuda_and_official6") is True,
        "marker_prior_manifest_exact_31": marker.get(
            "independent_frozen_input_rehash", {}
        ).get("binding_count")
        == 31
        and marker.get("independent_frozen_input_rehash", {}).get("pass") is True,
        "closed_schema_status_exact": closed.get("schema_version")
        == "ptcg-cw13-consumed-valid-official-b256-cuttingplane-v1"
        and closed.get("status") == "closed_no_CW13_candidate"
        and isinstance(second, Mapping)
        and second.get("status") == "closed_no_CW13_candidate"
        and isinstance(terminal, Mapping)
        and terminal.get("status") == "closed_no_CW13_candidate",
        "closed_scope_and_execution_exact": closed.get("scope", {}).get(
            "CW13_favorable_transition_restoration"
        )
        is True
        and closed.get("run_executed") is True
        and closed.get("cuda_accessed") is True
        and closed.get("writes_performed") is False,
        "closed_decision_exact": terminal.get("decision", {}).get("close_reason")
        == "fail_closed_L2_cap_gate"
        and terminal.get("decision", {}).get("terminal_iteration") is None
        and terminal.get("decision", {}).get("terminal_model_state_sha256") is None,
        "closed_final_restore_exact": isinstance(final_integrity, Mapping)
        and final_integrity.get("pass") is True
        and final_integrity.get("model_left_raw_after_outer_finally") is True
        and engine.all_true_checks(final_integrity.get("checks")),
        "closed_prior_manifest_exact_31": isinstance(prior_records, list)
        and len(prior_records) == 31
        and frozen.get("binding_count") == 31
        and frozen.get("all_exact") is True,
        "iteration_shape_exact_0_1_2": iteration_shape,
        "iteration1_exact_12_harms_3_contexts": isinstance(harms, list)
        and len(harms) == 12
        and len(contexts) == 3
        and oracle.get("new_harm_count_vs_cw11") == 12,
        "iteration1_context_metric_quartets_exact": context_metrics
        == {
            (
                "flg",
                "ffd866085ed5f2741e44a98d6c34bb33a2d421c1289c29c9b393a752d9b24a0a",
            ): {"set", "hybrid", "ordered", "top1"},
            (
                "pokemonfan",
                "4afd6b4112ce49bab88c5192c1c5425a395cd629ebf2d8fc81cbbcba6d0d2b0b",
            ): {"set", "hybrid", "ordered", "top1"},
            (
                "pokemonfan",
                "861c6645f1764ac84ecbc9fe9448697ddf74b3e728d65dc28142c6d038ccea66",
            ): {"set", "hybrid", "ordered", "top1"},
        },
        "iteration1_threshold_profiles_exact": cut_profiles
        == {
            (
                "flg",
                "ffd866085ed5f2741e44a98d6c34bb33a2d421c1289c29c9b393a752d9b24a0a",
                -0.0078125,
                0.0078125,
                0.001953125,
                0.0078125,
            ),
            (
                "pokemonfan",
                "4afd6b4112ce49bab88c5192c1c5425a395cd629ebf2d8fc81cbbcba6d0d2b0b",
                -0.00390625,
                0.0,
                0.0009765625,
                0.0009765625,
            ),
            (
                "pokemonfan",
                "861c6645f1764ac84ecbc9fe9448697ddf74b3e728d65dc28142c6d038ccea66",
                -0.001220703125,
                0.0028076171875,
                0.000244140625,
                0.0028076171875,
            ),
        },
        "iteration1_cut_split_exact_4_plus_8": isinstance(oracle, Mapping)
        and len(oracle.get("deterministic_new_harm_cuts", [])) == 4
        and len(oracle.get("deterministic_restoration_cuts", [])) == 8
        and len(oracle.get("deterministic_all_separating_cuts", [])) == 12,
        "iteration1_merge_exact_active50": isinstance(post_merge, Mapping)
        and post_merge.get("active_count") == 50
        and len(post_merge.get("added", [])) == 12
        and post_merge.get("strengthened") == [],
        "iteration1_additional_l2_exact": close_float(
            iteration1.get("additional_l2"), 0.0009424461480998953
        ),
        "iteration2_capped_close_exact": iteration2.get("kind")
        == "fail_closed_before_proposal"
        and iteration2.get("close_reason") == "fail_closed_L2_cap_gate"
        and iteration2.get("qp", {}).get("capped") is True
        and close_float(
            iteration2.get("qp", {}).get("uncapped_l2"),
            0.0011619656695555538,
        )
        and close_float(
            iteration2.get("proposal_additional_l2"),
            0.0013625501502583153,
        )
        and iteration2.get("qp", {}).get("gradient_shape") == [50, 65793]
        and iteration2.get("qp", {}).get("svd_rank") == 40
        and close_float(
            iteration2.get("qp", {}).get("predicted_applied_residual_min"),
            -0.0021779589992305982,
        )
        and iteration2.get("cap_checks", {}).get(
            "additional_total_at_most_cap"
        )
        is False,
        "stderr_schema_status_exact": stderr.get("schema_version")
        == "ptcg-cw13-consumed-valid-official6-one-shot-stderr-audit-v1"
        and stderr.get("status") == "captured_losslessly_not_a_success_veto",
        "stderr_lossless_exact": len(child_stderr) == CW13_CHILD_STDERR_BYTES
        and stderr.get("bytes") == CW13_CHILD_STDERR_BYTES
        and sha256_bytes(child_stderr)
        == stderr.get("sha256")
        == CW13_CHILD_STDERR_SHA256
        and "Traceback (most recent call last):" not in child_stderr_utf8,
        "stderr_post_child_crosslinks_exact": isinstance(post, Mapping)
        and post.get("pass") is True
        and post.get("solver", {}).get("sha256") == CW13_SOLVER_SHA256
        and post.get("attempt_marker", {}).get("sha256")
        == CW13_ATTEMPT_MARKER_SHA256
        and post.get("independent_frozen_input_rehash", {}).get("binding_count")
        == 31
        and post.get("independent_frozen_input_rehash", {}).get("pass") is True,
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW13 immutable terminal evidence gate failed: {checks}")
    return {
        "checks": checks,
        "attempt_marker": marker_evidence,
        "closed_stdout": closed_evidence,
        "stderr_audit": stderr_evidence,
        "prior_frozen_records": prior_records,
        "observed_harm_context_count": len(contexts),
        "pass": True,
    }


def cw14_preflight(*, require_lock: bool) -> dict[str, Any]:
    evidence = _BASE_PREFLIGHT(require_lock=require_lock)
    evidence["inherited_parent_launcher"] = PARENT_EVIDENCE
    evidence["inherited_parent_composed_checks"] = PARENT_COMPOSED_CHECKS
    evidence["known_cw12_failure_evidence"] = (
        parent_adapter.known_cw12_failure_evidence()
    )
    terminal_evidence = known_cw13_terminal_evidence()
    evidence["known_cw13_terminal_evidence"] = terminal_evidence
    if evidence.get("solver_lock_armed"):
        frozen = evidence.get("solver_static", {}).get("payload", {}).get(
            "frozen_inputs"
        )
        records = frozen.get("records") if isinstance(frozen, Mapping) else None
        if not isinstance(records, list):
            raise ProtocolError("armed CW14 solver did not emit its frozen manifest")
        observed = {
            record.get("path"): record
            for record in records
            if isinstance(record, Mapping)
        }
        prior = terminal_evidence["prior_frozen_records"]
        expected = {
            record["path"]: {
                "sha256": record["sha256"],
                "mode": record["mode"],
                "expected_mode": record["mode"],
                "nlink": 1,
                "regular": True,
            }
            for record in prior
        }
        expected.update(
            {
                root_relative(CW13_ATTEMPT_MARKER): {
                    "sha256": CW13_ATTEMPT_MARKER_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
                root_relative(CW13_CLOSED_STDOUT): {
                    "sha256": CW13_CLOSED_STDOUT_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
                root_relative(CW13_STDERR_AUDIT): {
                    "sha256": CW13_STDERR_AUDIT_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
            }
        )
        checks = {
            "solver_manifest_count_exact_34": frozen.get("binding_count") == 34
            and len(records) == 34
            and len(expected) == 34,
            "old31_plus_cw13_three_path_set_exact": set(observed) == set(expected),
            "all_34_record_identities_exact": all(
                path in observed
                and all(observed[path].get(key) == value for key, value in identity.items())
                for path, identity in expected.items()
            ),
        }
        if not all(checks.values()):
            raise ProtocolError(f"CW14 exact 31+3 manifest gate failed: {checks}")
        evidence["cw14_manifest_31_plus_3_bindings"] = {
            "checks": checks,
            "cw13_terminal_records": {
                path: observed[path]
                for path in sorted(
                    {
                        root_relative(CW13_ATTEMPT_MARKER),
                        root_relative(CW13_CLOSED_STDOUT),
                        root_relative(CW13_STDERR_AUDIT),
                    }
                )
            },
            "pass": True,
        }
    return evidence


def cw14_stderr_audit(
    payload: bytes,
    *,
    post_child_integrity: Mapping[str, Any],
) -> bytes:
    inherited = _PARENT_STDERR_AUDIT(
        payload,
        post_child_integrity=post_child_integrity,
    )
    document = engine.strict_json_object(inherited, "inherited CW13 stderr audit")
    if document.get("schema_version") != (
        "ptcg-cw13-consumed-valid-official6-one-shot-stderr-audit-v1"
    ):
        raise ProtocolError("inherited CW13 stderr audit schema drift")
    grandparent = document.get("inherited_parent_launcher")
    document["schema_version"] = STDERR_SCHEMA
    document["inherited_grandparent_launcher"] = grandparent
    document["inherited_parent_launcher"] = PARENT_EVIDENCE
    return engine.canonical_json(document)


engine.preflight = cw14_preflight
engine.validated_stderr_audit = cw14_stderr_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="static")
    args = parser.parse_args()
    result = engine.static_result() if args.mode == "static" else engine.run_once()
    print(engine.canonical_json(result).decode("utf-8"), end="")
    return (
        1
        if result.get("status") == "terminal_failure_recorded_attempt_consumed"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())

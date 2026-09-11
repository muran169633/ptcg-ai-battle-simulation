#!/usr/bin/env python3
"""Consume exactly one CW15 official-six sequential-tangent attempt.

CW15 hash-binds the final CW14 one-shot launcher and executes its exact held
bytes.  This adapter replaces only the solver/schema/artifact bindings,
expands the frozen solver manifest from the prior 34 records to those same 34
plus the three immutable CW14 terminal artifacts, and independently validates
all thirteen CW14 iterations including the exact anchor-tangent fixed point.
Run mode remains a single fail-closed attempt under the installed final CW15
solver SHA and terminal contract.
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
SCRIPT = TOOLS / (
    "run_cw15_consumed_valid_official6_sequential_tangent_one_shot_v1.py"
)
SOLVER = TOOLS / (
    "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py"
)
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_EXECUTABLE_MODE = 0o555

PARENT_LAUNCHER = TOOLS / (
    "run_cw14_consumed_valid_official6_cuttingplane_one_shot_v1.py"
)
PARENT_LAUNCHER_SHA256 = (
    "11d64eb708cbd39e06b8ae2c8028c515c50ad9541ab88848afed0a388ad20af7"
)
CW14_SOLVER_SHA256 = (
    "febfc16225cc0b915fead337cbf140ff9b4c94d1f5b1cf12921f51b3b340e51f"
)

CW14_ATTEMPT_MARKER = ARTIFACTS / (
    ".ptcg-cw14_consumed_valid_official6_cuttingplane_20260802_v1-attempt.json"
)
CW14_ATTEMPT_MARKER_SHA256 = (
    "5a427b29aab2996472f4cad1344370bc4784d5ce40f611bdfd9c2565556c316d"
)
CW14_CLOSED_STDOUT = ARTIFACTS / (
    "cw14_consumed_valid_official6_cuttingplane_20260802_v1.stdout.json"
)
CW14_CLOSED_STDOUT_SHA256 = (
    "3f903ba9320b61c70fdcf5794d1676e3aea44f576498f58e6148b69230576e69"
)
CW14_STDERR_AUDIT = ARTIFACTS / (
    "cw14_consumed_valid_official6_cuttingplane_20260802_v1.stderr-audit.json"
)
CW14_STDERR_AUDIT_SHA256 = (
    "4897afc99ec44b2e532ff24f16e8d993b46d761323072642e03dc95d4b61486d"
)
CW14_CHILD_STDERR_SHA256 = (
    "5ceb096b8ac5376a3ee276bf70012f90cb2fd479e3fa2804f2c0a8145aff4512"
)
CW14_CHILD_STDERR_BYTES = 347

CW14_ITERATION1_ADDITIONAL_L2 = 0.0009424461480998953
CW14_TERMINAL_ADDITIONAL_L2 = 0.0009447829261258565
CW14_TERMINAL_TOTAL_FROM_RAW_L2 = 0.007979438546138178
CW14_ITERATION1_MODEL_SHA256 = (
    "ba5098d4e5c52bb6a02c4e69fa2015428d018f85953edc4fb6bc14330634d769"
)
CW14_STALLED_MODEL_SHA256 = (
    "d3d4d3a573c2153a33de691b37a42a490d7f0d9a327f165dffa6bade5576a50d"
)
CW14_ITERATION1_ADDITIONAL_SHA256 = (
    "e3814600afa6f1c7843ca073d89a86cb1e32cf162d89f8cbfaf8db378ad3183f"
)
CW14_STALLED_ADDITIONAL_SHA256 = (
    "08712b376218bbe223ff7a9de1373413af8f2bf32e6cf797ac1959b12e342721"
)
CW14_ITERATION1_LINEARIZATION_SHA256 = (
    "a4cd72180e54b70bc28bdcb63dd2272dfdc175babe04ea88b82eff1d36dcc159"
)
CW14_STALLED_LINEARIZATION_SHA256 = (
    "5ca3b7ef897fbc76ea25c27a0ab89fd7a233bad90d36f4a26d43dd42b60eecc1"
)
CW14_INITIAL_ACTIVE_IDENTITY_SHA256 = (
    "e70e14b370ae22ec2ab88d4cca9d9337ca005aa68c67af02b7a581e5ae332682"
)
CW14_STALLED_ACTIVE_IDENTITY_SHA256 = (
    "0f2dd346d8b1a1c13eff42c2671a9a78648fc94797399a140c98988d6a26bd6b"
)
CW14_ITERATION1_ADDITION_IDENTITY_SHA256 = (
    "e91917703f6573f90016444c07eb7350a565b69a8a3362b48d1c7ec4710f5dc0"
)
CW14_STALLED_ADDITION_IDENTITY_SHA256 = (
    "a461117a85019c0043834070e27fea0dcf4c657ee413b6b31cb032a32b8bb56e"
)
CW14_STALLED_METADATA_NORMALIZED_SHA256 = (
    "e02a18c9b7874b1f053ba0f1f7fbae69e2fadbaffa4b7eed44bf154ed276eb9f"
)
CW15_CRITICAL_FUNCTION_AST_SHA256 = {
    "cw15_run_scope": (
        "ac230ee52c578c207291bd9d1b1fa0f2e5d2b5c14e2b548a9361b6cf4bc5e882"
    ),
    "cw15_validate_static_payload": (
        "b0ab273b9851376f8fd4e7d075f7181ad09f6659371904c21d7cc2ea9d812d8f"
    ),
    "known_cw14_terminal_evidence": (
        "d8b66c50b3599364d190cdd4b2b4f8c6805461d067463f0f696aee5deddaded4"
    ),
    "cw15_preflight": (
        "80eb9fa4a1b892733e696525909b5f4841b76e819e18dfbf74dd5627f5c3843d"
    ),
    "cw15_stderr_audit": (
        "d3ca266a820790fbf510824ce27549e752b593fdce83516e3e340940eba98aff"
    ),
}

SCHEMA = "ptcg-cw15-consumed-valid-official6-one-shot-launcher-v1"
ATTEMPT_SCHEMA = "ptcg-cw15-consumed-valid-official6-one-shot-attempt-v1"
FAILURE_SCHEMA = "ptcg-cw15-consumed-valid-official6-one-shot-terminal-failure-v1"
STDERR_SCHEMA = "ptcg-cw15-consumed-valid-official6-one-shot-stderr-audit-v1"
SOLVER_SCHEMA = (
    "ptcg-cw15-consumed-valid-official-b256-sequential-affine-tangent-"
    "cuttingplane-v1"
)
EXPECTED_STATIC_TOP_KEYS = (
    "affine_ledger_integrity_self_test",
    "anchored_total_geometry_self_test",
    "classification",
    "contract",
    "cuda_accessed",
    "cw12_consumed_failure_evidence",
    "cw13_consumed_closure_evidence",
    "cw14_consumed_closure_evidence",
    "forensic_contract",
    "frozen_inputs",
    "run_executed",
    "runtime",
    "schema_version",
    "sequential_affine_tangent_self_test",
    "source_audit",
    "status",
    "writes_performed",
)

ARTIFACT_ID = "cw15_consumed_valid_official6_sequential_tangent_20260802_v1"
ATTEMPT_MARKER = ARTIFACTS / f".ptcg-{ARTIFACT_ID}-attempt.json"
STDOUT_OUTPUT = ARTIFACTS / f"{ARTIFACT_ID}.stdout.json"
STDERR_AUDIT = ARTIFACTS / f"{ARTIFACT_ID}.stderr-audit.json"

# Exact final independently audited CW15 sequential-affine solver lock.
EXPECTED_SOLVER_SHA256 = (
    "2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24"
)
EXPECTED_STATIC_STATUS = (
    "static_ready_CW15_sequential_affine_tangent_run_implemented"
)
EXPECTED_TERMINAL_STATUSES = (
    "consumed_valid_CW15_sequential_affine_tangent_closure_first_feasible",
    "closed_no_CW15_candidate",
)
EXPECTED_SOLVER_FROZEN_BINDINGS = 37

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
    """The inherited launcher, CW14 evidence, or CW15 binding is invalid."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    """Execute only exact held-fd parent bytes already matched to their lock."""

    module_name = f"_cw15_parent_one_shot_{PARENT_LAUNCHER_SHA256[:16]}"
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
    raise ProtocolError("frozen CW14 parent launcher source audit failed")
if not isinstance(getattr(parent_adapter, "engine", None), ModuleType):
    raise ProtocolError("frozen CW14 parent did not expose its inherited engine")
engine = parent_adapter.engine

PARENT_COMPOSED_CHECKS = {
    "cw14_script_exact": engine.SCRIPT == PARENT_LAUNCHER,
    "cw14_solver_path_exact": engine.SOLVER.name
    == "probe_u468_cw11_consumed_valid_official6_cw14_cuttingplane_v1.py",
    "cw14_solver_sha_exact": engine.EXPECTED_SOLVER_SHA256 == CW14_SOLVER_SHA256,
    "cw14_manifest_count_exact_34": engine.EXPECTED_SOLVER_FROZEN_BINDINGS == 34,
    "cw14_static_status_exact": engine.EXPECTED_STATIC_STATUS
    == "static_ready_CW14_anchored_total_run_implemented",
    "cw14_success_status_exact": engine.EXPECTED_TERMINAL_STATUSES[0]
    == "consumed_valid_CW14_anchored_total_closure_first_feasible",
    "cw14_closed_status_exact": engine.EXPECTED_TERMINAL_STATUSES[1]
    == "closed_no_CW14_candidate",
    "cw14_scope_anchored_total_exact": engine.RUN_SCOPE.get(
        "CW14_anchored_minimum_total_from_CW11"
    )
    is True,
}
if not all(PARENT_COMPOSED_CHECKS.values()):
    raise ProtocolError(f"frozen CW14 composed contract drift: {PARENT_COMPOSED_CHECKS}")

# Skip only the CW14-specific 34-record wrapper.  Its exact frozen CW12 core
# preflight is manifest-count-parametric and is overridden below to 37.
_BASE_PREFLIGHT = parent_adapter._BASE_PREFLIGHT
_PARENT_STDERR_AUDIT = engine.validated_stderr_audit


def cw15_run_scope() -> dict[str, Any]:
    """Return one fresh exact scope mapping for override and source audit."""

    return {
        "exact_CW11_reconstruction": True,
        "CW15_minimum_total_from_CW11_origin": True,
        "semantic_active_cut_ledger": True,
        "sequential_affine_tangent_optimization_ledger": True,
        "post_merge_all_active_false_rows_tangented": True,
        "favorable_transition_restoration": True,
        "authoritative_full_six_official_B256_oracle_each_new_model_proposal": True,
        "candidate_RAM_only": True,
        "optimizer_backward_training": False,
        "model_or_result_writes": 0,
        "network_upload_submission": False,
        "broad_or_gold_access": False,
        "specialist_valid_consumed_dev_only": True,
    }


def wrapper_source_audit(source: bytes) -> dict[str, Any]:
    text = source.decode("utf-8")
    tree = ast.parse(text, filename=str(SCRIPT))
    top_level_literals: dict[str, Any] = {}
    for node in tree.body:
        names: list[str] = []
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
            value_node = node.value
        if value_node is None:
            continue
        try:
            literal = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            continue
        for name in names:
            top_level_literals[name] = literal
    imported_roots: set[str] = set()
    calls: set[str] = set()
    output_arguments = []
    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function_names = set(function_nodes)
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
        is parent_adapter._BASE_PREFLIGHT,
        "parent_executed_from_exact_held_bytes": "compile" in calls
        and "exec" in calls
        and "spec_from_file_location" not in calls,
        "no_forbidden_imports": not bool(imported_roots.intersection(forbidden_imports)),
        "no_direct_subprocess_call": "run" not in calls,
        "cli_only_mode": not output_arguments,
        "solver_path_exact": SOLVER.name
        == "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py",
        "top_level_parent_launcher_sha_literal_exact": top_level_literals.get(
            "PARENT_LAUNCHER_SHA256"
        )
        == PARENT_LAUNCHER_SHA256,
        "top_level_cw14_solver_sha_literal_exact": top_level_literals.get(
            "CW14_SOLVER_SHA256"
        )
        == CW14_SOLVER_SHA256,
        "top_level_cw14_artifact_sha_literals_exact": all(
            top_level_literals.get(name) == expected
            for name, expected in {
                "CW14_ATTEMPT_MARKER_SHA256": CW14_ATTEMPT_MARKER_SHA256,
                "CW14_CLOSED_STDOUT_SHA256": CW14_CLOSED_STDOUT_SHA256,
                "CW14_STDERR_AUDIT_SHA256": CW14_STDERR_AUDIT_SHA256,
            }.items()
        ),
        "top_level_solver_sha_lock_exact": top_level_literals.get(
            "EXPECTED_SOLVER_SHA256"
        )
        == EXPECTED_SOLVER_SHA256
        == "2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24",
        "top_level_schema_literals_exact": all(
            top_level_literals.get(name) == expected
            for name, expected in {
                "SCHEMA": SCHEMA,
                "ATTEMPT_SCHEMA": ATTEMPT_SCHEMA,
                "FAILURE_SCHEMA": FAILURE_SCHEMA,
                "STDERR_SCHEMA": STDERR_SCHEMA,
                "SOLVER_SCHEMA": SOLVER_SCHEMA,
                "ARTIFACT_ID": ARTIFACT_ID,
                "EXPECTED_STATIC_TOP_KEYS": EXPECTED_STATIC_TOP_KEYS,
            }.items()
        ),
        "top_level_status_literals_exact": top_level_literals.get(
            "EXPECTED_STATIC_STATUS"
        )
        == EXPECTED_STATIC_STATUS
        and top_level_literals.get("EXPECTED_TERMINAL_STATUSES")
        == EXPECTED_TERMINAL_STATUSES,
        "top_level_critical_ast_hash_map_literal_exact": top_level_literals.get(
            "CW15_CRITICAL_FUNCTION_AST_SHA256"
        )
        == CW15_CRITICAL_FUNCTION_AST_SHA256,
        "solver_manifest_count_exact_37": EXPECTED_SOLVER_FROZEN_BINDINGS == 37
        and top_level_literals.get("EXPECTED_SOLVER_FROZEN_BINDINGS") == 37,
        "cw15_static_validator_installed": engine.validate_static_payload
        is cw15_validate_static_payload,
        "cw15_status_contract_exact": EXPECTED_STATIC_STATUS
        == "static_ready_CW15_sequential_affine_tangent_run_implemented"
        and EXPECTED_TERMINAL_STATUSES
        == (
            "consumed_valid_CW15_sequential_affine_tangent_closure_first_feasible",
            "closed_no_CW15_candidate",
        ),
        "cw15_scope_contract_exact": engine.RUN_SCOPE == cw15_run_scope(),
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
            == ".ptcg-cw15_consumed_valid_official6_sequential_tangent_20260802_v1-attempt.json"
            and STDOUT_OUTPUT.name
            == "cw15_consumed_valid_official6_sequential_tangent_20260802_v1.stdout.json"
            and STDERR_AUDIT.name
            == "cw15_consumed_valid_official6_sequential_tangent_20260802_v1.stderr-audit.json"
        ),
        "targets_distinct_under_artifacts": len(
            {ATTEMPT_MARKER, STDOUT_OUTPUT, STDERR_AUDIT}
        )
        == 3
        and ATTEMPT_MARKER.parent
        == STDOUT_OUTPUT.parent
        == STDERR_AUDIT.parent
        == ARTIFACTS,
        "cw14_deep_terminal_validator_declared": "known_cw14_terminal_evidence"
        in function_names,
        "critical_function_ast_hashes_exact": set(
            CW15_CRITICAL_FUNCTION_AST_SHA256
        ).issubset(function_nodes)
        and all(
            sha256_bytes(
                ast.dump(function_nodes[name], include_attributes=False).encode(
                    "utf-8"
                )
            )
            == expected
            for name, expected in CW15_CRITICAL_FUNCTION_AST_SHA256.items()
        ),
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


def apply_cw15_overrides() -> None:
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
        "RUN_SCOPE": cw15_run_scope(),
    }
    for name, value in values.items():
        setattr(engine, name, value)
    engine.static_source_audit = wrapper_source_audit


apply_cw15_overrides()


def cw15_validate_static_payload(payload: bytes) -> dict[str, Any]:
    """Validate the CW15 sequential-affine static contract."""

    document = engine.strict_json_object(payload, "CW15 solver static stdout")
    contract = document.get("contract")
    runtime = document.get("runtime")
    forensic = document.get("forensic_contract")
    cw12_failure = document.get("cw12_consumed_failure_evidence")
    cw13_closure = document.get("cw13_consumed_closure_evidence")
    cw14_closure = document.get("cw14_consumed_closure_evidence")
    anchored_math = document.get("anchored_total_geometry_self_test")
    tangent_math = document.get("sequential_affine_tangent_self_test")
    affine_ledger_math = document.get("affine_ledger_integrity_self_test")
    frozen = document.get("frozen_inputs")
    expected_contract_subset = {
        "starting_model_sha256": engine.CW11_MODEL_SHA256,
        "parent_CW12_solver_sha256": (
            "e16eb0aa6d3ea0905309210126abd9f1f4cbb2f093b7e9dfdac49271466377ec"
        ),
        "parent_CW13_solver_sha256": (
            "dbbdc12e7c2f571f82de92d30450d8ac794c7215ee86137b9aa97ebc42d28297"
        ),
        "parent_CW14_solver_sha256": CW14_SOLVER_SHA256,
        "CW15_minimum_total_from_CW11_origin": True,
        "semantic_active_cut_ledger_separate": True,
        "optimization_affine_tangent_ledger_monotone": True,
        "initial_and_new_semantic_cuts_get_CW11_anchor_tangent": True,
        "post_merge_all_active_false_rows_get_candidate_tangent": True,
        "candidate_tangent_key_fields": [
            "semantic_cut_identity",
            "linearization_point_float64_le_sha256",
            "linearization_model_state_sha256",
            "threshold_float64_le_sha256",
        ],
        "candidate_tangent_rhs": "threshold-margin+gradient_dot_point",
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
        "frozen_solver_trust_region_cap_reinterpreted_as_minimum_total": (
            engine.STEP_L2_CAP
        ),
        "previous_to_proposal_delta_l2_cap": engine.STEP_L2_CAP,
        "additional_total_l2_cap_from_cw11": engine.ADDITIONAL_TOTAL_L2_CAP,
        "direct_uncapped_linear_residual_tolerance": 1e-8,
        "uncapped_over_total_cap_closes_before_proposal": True,
        "clipped_vector_must_never_be_applied": True,
        "nested_affine_minimum_total_norm_monotone_gate": True,
        "nested_projection_dot_nonnegative_gate": True,
        "actual_violation_requires_strict_affine_ledger_growth": True,
        "x_model_and_semantic_state_stagnation_fail_closed": True,
        "single_BF16_output_plateau_with_changed_model_allowed": True,
        "anchor_candidate_and_combined_QP_residuals_separately_gated": True,
        "threshold_change_from_CW14": False,
        "l2_cap_absolute_tolerance": engine.L2_CAP_ABS_TOL,
        "actor_parameter_names": list(engine.ACTOR6_NAMES),
        "first_feasible": True,
        "candidate_ordering": "fixed_iteration_order_no_best_of_N",
        "terminal_semantics": "consumed_valid_optimization_closure_only",
    }
    checks = {
        "top_level_keys_exact": set(document) == set(EXPECTED_STATIC_TOP_KEYS)
        and len(document) == len(EXPECTED_STATIC_TOP_KEYS),
        "schema_exact": document.get("schema_version") == SOLVER_SCHEMA,
        "status_exact": document.get("status") == EXPECTED_STATIC_STATUS,
        "classification_exact": document.get("classification")
        == engine.CLASSIFICATION,
        "runtime_checks_all_true": isinstance(runtime, Mapping)
        and engine.all_true_checks(runtime.get("checks")),
        "frozen_inputs_exact_37": isinstance(frozen, Mapping)
        and frozen.get("all_exact") is True
        and frozen.get("binding_count") == 37
        and isinstance(frozen.get("records"), list)
        and len(frozen["records"]) == 37,
        "forensic_contract_all_true": isinstance(forensic, Mapping)
        and engine.all_true_checks(forensic.get("checks")),
        "cw12_failure_evidence_all_true": isinstance(cw12_failure, Mapping)
        and cw12_failure.get("pass") is True
        and engine.all_true_checks(cw12_failure.get("checks")),
        "cw13_closure_evidence_all_true": isinstance(cw13_closure, Mapping)
        and cw13_closure.get("pass") is True
        and engine.all_true_checks(cw13_closure.get("checks")),
        "cw14_closure_evidence_all_true": isinstance(cw14_closure, Mapping)
        and cw14_closure.get("pass") is True
        and engine.all_true_checks(cw14_closure.get("checks")),
        "anchored_total_geometry_self_test_all_true": isinstance(
            anchored_math, Mapping
        )
        and anchored_math.get("pass") is True
        and engine.all_true_checks(anchored_math.get("checks")),
        "sequential_affine_tangent_self_test_all_true": isinstance(
            tangent_math, Mapping
        )
        and tangent_math.get("pass") is True
        and engine.all_true_checks(tangent_math.get("checks")),
        "affine_ledger_integrity_self_test_all_true": isinstance(
            affine_ledger_math, Mapping
        )
        and affine_ledger_math.get("pass") is True
        and engine.all_true_checks(affine_ledger_math.get("checks")),
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
        raise ProtocolError(f"CW15 solver static payload contract failed: {checks}")
    return {
        "checks": checks,
        "stdout_sha256": sha256_bytes(payload),
        "stdout_bytes": len(payload),
        "payload": document,
    }


engine.validate_static_payload = cw15_validate_static_payload


def close_float(value: Any, expected: float, *, tolerance: float = 1e-18) -> bool:
    return isinstance(value, (int, float)) and math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=tolerance
    )


def strip_iteration_metadata(value: Any) -> Any:
    """Remove only audited iteration-label metadata for fixed-point equality."""

    if isinstance(value, Mapping):
        return {
            key: strip_iteration_metadata(item)
            for key, item in value.items()
            if key not in {"iteration", "created_iteration"}
        }
    if isinstance(value, list):
        return [strip_iteration_metadata(item) for item in value]
    return value


def known_cw14_terminal_evidence() -> dict[str, Any]:
    """Bind the immutable CW14 closure and its exact sequential-tangent stall."""

    marker_raw, marker_evidence = engine.read_regular_stable(
        CW14_ATTEMPT_MARKER,
        "consumed CW14 attempt marker",
        expected_sha256=CW14_ATTEMPT_MARKER_SHA256,
        expected_mode=0o444,
    )
    closed_raw, closed_evidence = engine.read_regular_stable(
        CW14_CLOSED_STDOUT,
        "immutable CW14 closed stdout",
        expected_sha256=CW14_CLOSED_STDOUT_SHA256,
        expected_mode=0o444,
    )
    stderr_raw, stderr_evidence = engine.read_regular_stable(
        CW14_STDERR_AUDIT,
        "immutable CW14 stderr audit",
        expected_sha256=CW14_STDERR_AUDIT_SHA256,
        expected_mode=0o444,
    )
    marker = engine.strict_json_object(marker_raw, "CW14 attempt marker")
    closed = engine.strict_json_object(closed_raw, "CW14 closed stdout")
    stderr = engine.strict_json_object(stderr_raw, "CW14 stderr audit")
    second = closed.get("second_stage")
    terminal = closed.get("terminal")
    final_integrity = closed.get("final_integrity")
    iterations = second.get("iterations") if isinstance(second, Mapping) else None
    iteration_shape = (
        isinstance(iterations, list)
        and len(iterations) == 13
        and [value.get("iteration") for value in iterations] == list(range(13))
    )
    iteration0 = iterations[0] if iteration_shape else {}
    iteration1 = iterations[1] if iteration_shape else {}
    iteration2 = iterations[2] if iteration_shape else {}
    repeated = iterations[3:] if iteration_shape else []
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

    stable_fields = (
        "acceptance",
        "active_cut_gates",
        "additional_float64_le_sha256",
        "additional_l2",
        "anchor_audit",
        "anchored_minimum_total_l2",
        "cap_checks",
        "deterministic_addition_count",
        "deterministic_addition_identity_sha256",
        "favorable_transition_restoration",
        "fixed_repair_gates",
        "geometry_checks",
        "gradient_audit",
        "gradient_reuse_audit",
        "integrity",
        "kind",
        "legacy_gate",
        "model_state_sha256",
        "post_oracle_cut_merge",
        "previous_additional_l2",
        "previous_to_proposal_delta_l2",
        "previous_to_proposal_projection_dot",
        "qp",
        "qp_cap_audit",
        "qp_checks",
        "six_view_snapshot",
        "total_from_raw_float64_le_sha256",
        "total_from_raw_l2",
    )
    normalized_hashes = [
        sha256_bytes(engine.canonical_json(strip_iteration_metadata(value)))
        for value in repeated
    ]
    metadata_exact = all(
        value.get("oracle", {}).get("iteration") == ordinal
        and len(value.get("oracle", {}).get("deterministic_all_separating_cuts", []))
        == 8
        and len(value.get("oracle", {}).get("deterministic_new_harm_cuts", []))
        == 4
        and len(value.get("oracle", {}).get("deterministic_restoration_cuts", []))
        == 4
        and all(
            cut.get("created_iteration") == ordinal
            for key in (
                "deterministic_all_separating_cuts",
                "deterministic_new_harm_cuts",
                "deterministic_restoration_cuts",
            )
            for cut in value.get("oracle", {}).get(key, [])
        )
        for ordinal, value in enumerate(repeated, start=3)
    )

    metrics = {"hybrid", "ordered", "set", "top1"}
    expected_pair_specs = {
        (
            "4afd6b4112ce49bab88c5192c1c5425a395cd629ebf2d8fc81cbbcba6d0d2b0b",
            "ccd32395c70e834cc78a32d66fd182596a5977de86f2c5506c811878975e3f1c",
            1,
            8,
        ): (-0.00390625, 0.0009765625, -0.0048828125),
        (
            "861c6645f1764ac84ecbc9fe9448697ddf74b3e728d65dc28142c6d038ccea66",
            "3b8a466d01e49811024d8d01a187aab370672b50b64804ef889cc41a20ed4145",
            1,
            2,
        ): (0.002685546875, 0.0028076171875, -0.0001220703125),
        (
            "ffd866085ed5f2741e44a98d6c34bb33a2d421c1289c29c9b393a752d9b24a0a",
            "aad8534415cc11e23338dcd82ed3bbf1ec14c375f953fa67ff17d006955463f4",
            3,
            6,
        ): (-0.0078125, 0.0078125, -0.015625),
    }
    expected_violations = sorted(
        (
            context,
            decision,
            metric,
            positive,
            negative,
            margin,
            threshold,
            residual,
        )
        for (context, decision, positive, negative), (
            margin,
            threshold,
            residual,
        ) in expected_pair_specs.items()
        for metric in metrics
    )

    def observed_violations(value: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        return sorted(
            (
                str(record["identity"][0]),
                str(record["identity"][1]),
                str(record["identity"][2]),
                int(record["identity"][3]),
                int(record["identity"][4]),
                float(record["margin"]),
                float(record["threshold"]),
                float(record["residual"]),
            )
            for record in value.get("active_cut_gates", {}).get("records", [])
            if record.get("pass") is False
        )

    stalled = [iteration2, *repeated] if iteration_shape else []
    expected_false_acceptance = {
        "active_cut_violations_zero",
        "forensic_favorable_retention_pass",
        "new_harm_count_zero",
    }
    decision = second.get("decision") if isinstance(second, Mapping) else None
    checks = {
        "marker_schema_status_exact": marker.get("schema_version")
        == "ptcg-cw14-consumed-valid-official6-one-shot-attempt-v1"
        and marker.get("status")
        == "one_shot_attempt_consumed_before_cuda_and_official6",
        "marker_launcher_solver_exact": marker.get("launcher", {}).get("sha256")
        == PARENT_LAUNCHER_SHA256
        and marker.get("solver", {}).get("sha256") == CW14_SOLVER_SHA256
        and marker.get("launcher", {}).get("mode_octal") == "0555"
        and marker.get("solver", {}).get("mode_octal") == "0555",
        "marker_targets_exact": marker.get("stdout_output")
        == root_relative(CW14_CLOSED_STDOUT)
        and marker.get("stderr_audit") == root_relative(CW14_STDERR_AUDIT),
        "marker_one_attempt_no_retry": marker.get("attempts_authorized") == 1
        and marker.get("retry_authorized") is False
        and marker.get("marker_created_before_cuda_and_official6") is True,
        "marker_prior_manifest_exact_34": marker.get(
            "independent_frozen_input_rehash", {}
        ).get("binding_count")
        == 34
        and marker.get("independent_frozen_input_rehash", {}).get("pass") is True,
        "closed_schema_status_exact": closed.get("schema_version")
        == "ptcg-cw14-consumed-valid-official-b256-anchored-total-cuttingplane-v1"
        and closed.get("status") == "closed_no_CW14_candidate"
        and isinstance(second, Mapping)
        and second.get("status") == "closed_no_CW14_candidate"
        and isinstance(terminal, Mapping)
        and terminal.get("status") == "closed_no_CW14_candidate",
        "closed_scope_and_execution_exact": closed.get("scope", {}).get(
            "CW14_anchored_minimum_total_from_CW11"
        )
        is True
        and closed.get("run_executed") is True
        and closed.get("cuda_accessed") is True
        and closed.get("writes_performed") is False,
        "closed_final_restore_exact": isinstance(final_integrity, Mapping)
        and final_integrity.get("pass") is True
        and final_integrity.get("model_left_raw_after_outer_finally") is True
        and engine.all_true_checks(final_integrity.get("checks")),
        "closed_prior_manifest_exact_34": isinstance(prior_records, list)
        and len(prior_records) == 34
        and frozen.get("binding_count") == 34
        and frozen.get("all_exact") is True,
        "closed_decision_exact": isinstance(decision, Mapping)
        and decision.get("close_reason")
        == "maximum_12_iterations_without_first_feasible"
        and decision.get("terminal_iteration") is None
        and decision.get("terminal_model_state_sha256") is None
        and decision.get("candidate_consumer_called") is False
        and decision.get("terminal_active_cut_count") == 50
        and close_float(
            decision.get("terminal_additional_l2"), CW14_TERMINAL_ADDITIONAL_L2
        )
        and close_float(
            decision.get("terminal_total_from_raw_l2"),
            CW14_TERMINAL_TOTAL_FROM_RAW_L2,
        ),
        "closed_terminal_payloads_absent": isinstance(terminal, Mapping)
        and terminal.get("reconstruction_payload") is None
        and terminal.get("reconstruction_audit") is None
        and terminal.get("active_cut_ledger") is None
        and terminal.get("active_cut_ledger_sha256") is None,
        "iteration_shape_exact_0_through_12": iteration_shape,
        "iteration0_reference_exact": iteration0.get("kind")
        == "exact_CW11_reference_before_fixed3_repair"
        and iteration0.get("active_cut_count") == 38
        and iteration0.get("active_cut_identity_sha256")
        == CW14_INITIAL_ACTIVE_IDENTITY_SHA256
        and iteration0.get("model_state_sha256")
        == "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
        and iteration0.get("additional_l2") == 0.0
        and iteration0.get("active_cut_gates", {}).get("violated_count") == 4,
        "iteration1_anchor_solution_and_merge_exact": iteration1.get("kind")
        == "unique_fixed_order_official_context_proposal"
        and iteration1.get("model_state_sha256") == CW14_ITERATION1_MODEL_SHA256
        and iteration1.get("additional_float64_le_sha256")
        == CW14_ITERATION1_ADDITIONAL_SHA256
        and close_float(
            iteration1.get("additional_l2"), CW14_ITERATION1_ADDITIONAL_L2
        )
        and iteration1.get("gradient_audit", {}).get("active_cut_count") == 38
        and iteration1.get("gradient_audit", {}).get(
            "ordered_cut_identity_sha256"
        )
        == CW14_INITIAL_ACTIVE_IDENTITY_SHA256
        and iteration1.get("gradient_audit", {}).get(
            "cut_linearization_ledger_sha256"
        )
        == CW14_ITERATION1_LINEARIZATION_SHA256
        and iteration1.get("qp", {}).get("gradient_shape") == [38, 65793]
        and iteration1.get("qp", {}).get("svd_rank") == 37
        and close_float(
            iteration1.get("qp", {}).get("direct_uncapped_residual_min"),
            -3.0357660829594124e-18,
            tolerance=1e-30,
        )
        and iteration1.get("active_cut_gates", {}).get("violated_count") == 0
        and iteration1.get("oracle", {}).get("new_harm_count_vs_cw11") == 12
        and iteration1.get("oracle", {}).get(
            "favorable_transition_break_count"
        )
        == 8
        and iteration1.get("deterministic_addition_count") == 12
        and iteration1.get("deterministic_addition_identity_sha256")
        == CW14_ITERATION1_ADDITION_IDENTITY_SHA256
        and len(iteration1.get("post_oracle_cut_merge", {}).get("added", []))
        == 12
        and iteration1.get("post_oracle_cut_merge", {}).get("strengthened") == []
        and iteration1.get("post_oracle_cut_merge", {}).get("active_count") == 50,
        "iteration2_exact_new_anchored_total": iteration2.get("kind")
        == "unique_fixed_order_official_context_proposal"
        and iteration2.get("model_state_sha256") == CW14_STALLED_MODEL_SHA256
        and iteration2.get("additional_float64_le_sha256")
        == CW14_STALLED_ADDITIONAL_SHA256
        and close_float(
            iteration2.get("additional_l2"), CW14_TERMINAL_ADDITIONAL_L2
        )
        and close_float(
            iteration2.get("previous_to_proposal_delta_l2"),
            6.640809762833165e-05,
        )
        and close_float(
            iteration2.get("previous_to_proposal_projection_dot"),
            -4.174548916749809e-21,
            tolerance=1e-30,
        )
        and iteration2.get("gradient_audit", {}).get("active_cut_count") == 50
        and iteration2.get("gradient_audit", {}).get(
            "ordered_cut_identity_sha256"
        )
        == CW14_STALLED_ACTIVE_IDENTITY_SHA256
        and iteration2.get("gradient_audit", {}).get(
            "cut_linearization_ledger_sha256"
        )
        == CW14_STALLED_LINEARIZATION_SHA256
        and iteration2.get("qp", {}).get("gradient_shape") == [50, 65793]
        and iteration2.get("qp", {}).get("svd_rank") == 40
        and close_float(
            iteration2.get("qp", {}).get("direct_uncapped_residual_min"),
            -6.071532165918825e-18,
            tolerance=1e-30,
        )
        and iteration2.get("qp", {}).get("solver", {}).get("success") is True
        and iteration2.get("qp", {}).get("capped") is False
        and iteration2.get("active_cut_gates", {}).get("violated_count") == 12
        and iteration2.get("oracle", {}).get("new_harm_count_vs_cw11") == 8
        and iteration2.get("oracle", {}).get(
            "favorable_transition_break_count"
        )
        == 4
        and iteration2.get("deterministic_addition_count") == 8
        and iteration2.get("deterministic_addition_identity_sha256")
        == CW14_STALLED_ADDITION_IDENTITY_SHA256
        and iteration2.get("post_oracle_cut_merge", {}).get("added") == []
        and iteration2.get("post_oracle_cut_merge", {}).get("strengthened") == []
        and iteration2.get("post_oracle_cut_merge", {}).get("active_count") == 50,
        "iterations3_through12_top_level_fixed_point_exact": len(repeated) == 10
        and all(
            all(value.get(key) == repeated[0].get(key) for key in stable_fields)
            for value in repeated[1:]
        ),
        "iterations3_through12_only_iteration_metadata_varies": len(repeated) == 10
        and metadata_exact
        and normalized_hashes
        == [CW14_STALLED_METADATA_NORMALIZED_SHA256] * 10,
        "iterations2_through12_same_candidate_outcome_exact": len(stalled) == 11
        and all(
            value.get("model_state_sha256") == CW14_STALLED_MODEL_SHA256
            and value.get("additional_float64_le_sha256")
            == CW14_STALLED_ADDITIONAL_SHA256
            and close_float(value.get("additional_l2"), CW14_TERMINAL_ADDITIONAL_L2)
            and value.get("gradient_audit", {}).get(
                "cut_linearization_ledger_sha256"
            )
            == CW14_STALLED_LINEARIZATION_SHA256
            and value.get("qp") == iteration2.get("qp")
            and value.get("active_cut_gates") == iteration2.get("active_cut_gates")
            and value.get("acceptance") == iteration2.get("acceptance")
            and value.get("six_view_snapshot")
            == iteration2.get("six_view_snapshot")
            for value in stalled
        ),
        "iterations3_through12_zero_step_reuse_exact": all(
            value.get("previous_additional_l2") == CW14_TERMINAL_ADDITIONAL_L2
            and value.get("previous_to_proposal_delta_l2") == 0.0
            and value.get("previous_to_proposal_projection_dot") == 0.0
            and value.get("gradient_reuse_audit", {}).get("repeated_count") == 50
            and value.get("gradient_reuse_audit", {}).get("new_count") == 0
            for value in repeated
        ),
        "post_merge_all_12_actual_active_violations_exact": len(stalled) == 11
        and len(expected_violations) == 12
        and all(observed_violations(value) == expected_violations for value in stalled),
        "stalled_acceptance_failures_exact": all(
            {
                key
                for key, passed in value.get("acceptance", {}).get("checks", {}).items()
                if passed is False
            }
            == expected_false_acceptance
            for value in stalled
        ),
        "stderr_schema_status_exact": stderr.get("schema_version")
        == "ptcg-cw14-consumed-valid-official6-one-shot-stderr-audit-v1"
        and stderr.get("status") == "captured_losslessly_not_a_success_veto",
        "stderr_lossless_exact": len(child_stderr) == CW14_CHILD_STDERR_BYTES
        and stderr.get("bytes") == CW14_CHILD_STDERR_BYTES
        and sha256_bytes(child_stderr)
        == stderr.get("sha256")
        == CW14_CHILD_STDERR_SHA256
        and "Traceback (most recent call last):" not in child_stderr_utf8,
        "stderr_post_child_crosslinks_exact": isinstance(post, Mapping)
        and post.get("pass") is True
        and post.get("solver", {}).get("sha256") == CW14_SOLVER_SHA256
        and post.get("attempt_marker", {}).get("sha256")
        == CW14_ATTEMPT_MARKER_SHA256
        and post.get("independent_frozen_input_rehash", {}).get("binding_count")
        == 34
        and post.get("independent_frozen_input_rehash", {}).get("pass") is True,
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW14 immutable terminal evidence gate failed: {checks}")
    return {
        "checks": checks,
        "attempt_marker": marker_evidence,
        "closed_stdout": closed_evidence,
        "stderr_audit": stderr_evidence,
        "prior_frozen_records": prior_records,
        "stalled_iterations": list(range(2, 13)),
        "stalled_model_state_sha256": CW14_STALLED_MODEL_SHA256,
        "stalled_additional_float64_le_sha256": CW14_STALLED_ADDITIONAL_SHA256,
        "stalled_linearization_ledger_sha256": CW14_STALLED_LINEARIZATION_SHA256,
        "stalled_native_violation_count": len(expected_violations),
        "pass": True,
    }


def cw15_preflight(*, require_lock: bool) -> dict[str, Any]:
    evidence = _BASE_PREFLIGHT(require_lock=require_lock)
    evidence["inherited_parent_launcher"] = PARENT_EVIDENCE
    evidence["inherited_parent_composed_checks"] = PARENT_COMPOSED_CHECKS
    evidence["known_cw12_failure_evidence"] = (
        parent_adapter.parent_adapter.known_cw12_failure_evidence()
    )
    evidence["known_cw13_terminal_evidence"] = (
        parent_adapter.known_cw13_terminal_evidence()
    )
    terminal_evidence = known_cw14_terminal_evidence()
    evidence["known_cw14_terminal_evidence"] = terminal_evidence
    if evidence.get("solver_lock_armed"):
        frozen = evidence.get("solver_static", {}).get("payload", {}).get(
            "frozen_inputs"
        )
        records = frozen.get("records") if isinstance(frozen, Mapping) else None
        if not isinstance(records, list):
            raise ProtocolError("armed CW15 solver did not emit its frozen manifest")
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
                root_relative(CW14_ATTEMPT_MARKER): {
                    "sha256": CW14_ATTEMPT_MARKER_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
                root_relative(CW14_CLOSED_STDOUT): {
                    "sha256": CW14_CLOSED_STDOUT_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
                root_relative(CW14_STDERR_AUDIT): {
                    "sha256": CW14_STDERR_AUDIT_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
            }
        )
        checks = {
            "solver_manifest_count_exact_37": frozen.get("binding_count") == 37
            and len(records) == 37
            and len(expected) == 37,
            "old34_plus_cw14_three_path_set_exact": set(observed) == set(expected),
            "all_37_record_identities_exact": all(
                path in observed
                and all(
                    observed[path].get(key) == value
                    for key, value in identity.items()
                )
                for path, identity in expected.items()
            ),
        }
        if not all(checks.values()):
            raise ProtocolError(f"CW15 exact 34+3 manifest gate failed: {checks}")
        evidence["cw15_manifest_34_plus_3_bindings"] = {
            "checks": checks,
            "cw14_terminal_records": {
                path: observed[path]
                for path in sorted(
                    {
                        root_relative(CW14_ATTEMPT_MARKER),
                        root_relative(CW14_CLOSED_STDOUT),
                        root_relative(CW14_STDERR_AUDIT),
                    }
                )
            },
            "pass": True,
        }
    return evidence


def cw15_stderr_audit(
    payload: bytes,
    *,
    post_child_integrity: Mapping[str, Any],
) -> bytes:
    inherited = _PARENT_STDERR_AUDIT(
        payload,
        post_child_integrity=post_child_integrity,
    )
    document = engine.strict_json_object(inherited, "inherited CW14 stderr audit")
    if document.get("schema_version") != (
        "ptcg-cw14-consumed-valid-official6-one-shot-stderr-audit-v1"
    ):
        raise ProtocolError("inherited CW14 stderr audit schema drift")
    prior_parent = document.get("inherited_parent_launcher")
    prior_grandparent = document.get("inherited_grandparent_launcher")
    document["schema_version"] = STDERR_SCHEMA
    document["inherited_greatgrandparent_launcher"] = prior_grandparent
    document["inherited_grandparent_launcher"] = prior_parent
    document["inherited_parent_launcher"] = PARENT_EVIDENCE
    return engine.canonical_json(document)


engine.preflight = cw15_preflight
engine.validated_stderr_audit = cw15_stderr_audit


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

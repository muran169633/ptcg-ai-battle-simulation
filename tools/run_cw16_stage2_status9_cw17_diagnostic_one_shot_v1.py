#!/usr/bin/env python3
"""Consume exactly one CW17 diagnostic attempt for the CW16 stage-2 status 9.

This adapter executes the exact held bytes of the frozen CW16 one-shot
launcher and reuses its already-audited O_EXCL publication core.  CW17 is a
diagnostic-only protocol: it may reconstruct the one historical CW16
bootstrap evaluation and inspect the failed fixed-z stage-2 solve, but it may
not evaluate a changed model, call a candidate consumer, create a checkpoint,
materialize a candidate, access broad/Gold data, or use the network.

The solver contract below is frozen, while this launcher intentionally remains
writable until its independent review is complete.  The inherited core still
requires this launcher itself to be frozen 0555 before either static preflight
or run mode can pass.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import math
import os
import stat
import struct
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
ARTIFACTS = ROOT / "artifacts"
SCRIPT = TOOLS / "run_cw16_stage2_status9_cw17_diagnostic_one_shot_v1.py"
SOLVER = TOOLS / "probe_cw16_stage2_status9_cw17_diagnostic_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_EXECUTABLE_MODE = 0o555
FROZEN_EVIDENCE_MODE = 0o444

PARENT_LAUNCHER = TOOLS / (
    "run_cw16_consumed_valid_official6_local_trust_one_shot_v1.py"
)
PARENT_LAUNCHER_SHA256 = (
    "0aa701207f485572df130c9b83ad2bc6de2ffc076bb84d50e79675925ba28cb7"
)
CW16_SOLVER = TOOLS / (
    "probe_u468_cw11_consumed_valid_official6_cw16_local_trust_v1.py"
)
CW16_SOLVER_SHA256 = (
    "a05df3df944451fabefa8e2a037ad541b1f90af7b966be6fc9ecb87bafe2ebbf"
)
CW16_ATTEMPT_MARKER = ARTIFACTS / (
    ".ptcg-cw16_consumed_valid_official6_local_trust_20260802_v1-attempt.json"
)
CW16_ATTEMPT_MARKER_SHA256 = (
    "ecee069b30e7cb0b33b33365d58bbae048569c0e4c5b2f2b96307f3bb40102a1"
)
CW16_ATTEMPT_MARKER_BYTES = 3118
CW16_CLOSED_STDOUT = ARTIFACTS / (
    "cw16_consumed_valid_official6_local_trust_20260802_v1.stdout.json"
)
CW16_CLOSED_STDOUT_SHA256 = (
    "62eaf348a4c125c8da72ce15ed8904e4ddfd32320d1121a2bba1ca2ce34dbb4e"
)
CW16_CLOSED_STDOUT_BYTES = 8127613
CW16_STDERR_AUDIT = ARTIFACTS / (
    "cw16_consumed_valid_official6_local_trust_20260802_v1.stderr-audit.json"
)
CW16_STDERR_AUDIT_SHA256 = (
    "fa10668b9c51a2fc6047962decdfd33e864211e4ee38e78f6b68299a587e89c3"
)
CW16_STDERR_AUDIT_BYTES = 18208
CW16_CHILD_STDERR_SHA256 = (
    "5ceb096b8ac5376a3ee276bf70012f90cb2fd479e3fa2804f2c0a8145aff4512"
)
CW16_CHILD_STDERR_BYTES = 347
CW16_STATUS = "closed_no_CW16_candidate"
CW16_STAGE2_STATUS = 9
CW16_STAGE2_MESSAGE = "Iteration limit reached"
CW16_CLOSE_REASON = (
    "fail_closed_hard_anchor_local_trust_subproblem:RuntimeError:"
    "CW16 fixed-z minimum-norm stage2 failed: 9 Iteration limit reached"
)
CW16_OFFICIAL_COUNT_BEFORE = 1
CW16_OFFICIAL_COUNT_AFTER = 1
CW16_OFFICIAL_COUNT_DELTA = 0
CW16_ADDITIONAL_L2 = 0.0009424461480998953
CW16_ACTIVE_CUT_COUNT = 50
CW16_ANCHOR_LEDGER_SHA256 = (
    "765b74c9a500699e65f43c27edfb7da4b568842dd2065cf55e46efb826d1ef09"
)

SCHEMA = "ptcg-cw17-stage2-status9-diagnostic-one-shot-launcher-v1"
ATTEMPT_SCHEMA = "ptcg-cw17-stage2-status9-diagnostic-one-shot-attempt-v1"
FAILURE_SCHEMA = (
    "ptcg-cw17-stage2-status9-diagnostic-one-shot-terminal-failure-v1"
)
STDERR_SCHEMA = (
    "ptcg-cw17-stage2-status9-diagnostic-one-shot-stderr-audit-v1"
)
SOLVER_SCHEMA = "ptcg-cw17-stage2-status9-diagnostic-v1"

ARTIFACT_ID = "cw17_stage2_status9_diagnostic_20260803_v1"
ATTEMPT_MARKER = ARTIFACTS / f".ptcg-{ARTIFACT_ID}-attempt.json"
STDOUT_OUTPUT = ARTIFACTS / f"{ARTIFACT_ID}.stdout.json"
STDERR_AUDIT = ARTIFACTS / f"{ARTIFACT_ID}.stderr-audit.json"

# This block is one coherent lock over the independently reviewed frozen
# diagnostic solver and its externally bound pure-CPU certificate backend.
FINAL_SOLVER_CONTRACT_PENDING = False
EXPECTED_SOLVER_SHA256 = (
    "9125a1238473c732155c2b107dbae29762cc6dbfb378426335a0fa8aa8962156"
)
EXPECTED_DUAL_MATH_SHA256 = (
    "a14241d500f352e6ae1b5662d5d7ea51ca4da22b61126536997e69e39750a600"
)
EXPECTED_STATIC_STATUS = "static_ready_CW17_stage2_diagnostic_implemented"
EXPECTED_TERMINAL_STATUSES = (
    "diagnostic_complete_CW17_stage2_certificate_reported",
)
EXPECTED_STATIC_TOP_KEYS = (
    "CW16_solver_source",
    "basis_operator_norm_selftest",
    "certificate_failure_recovery_selftest",
    "checks",
    "classification",
    "contract",
    "cuda_accessed",
    "cw16_closure_evidence",
    "dual_math_contract",
    "dual_math_selftest",
    "dual_math_source",
    "frozen_inputs_43",
    "run_executed",
    "runtime",
    "schema_version",
    "self_source",
    "source_audit",
    "status",
    "writes_performed",
)
EXPECTED_TERMINAL_TOP_KEYS = (
    "schema_version",
    "status",
    "classification",
    "contract",
    "input_lock",
    "frozen_inputs_43",
    "cw16_closure_evidence",
    "replay_identity",
    "diagnostic",
    "dual_math_selftest",
    "dual_math_contract",
    "runtime",
    "source_audit",
    "run_executed",
    "cuda_accessed",
    "writes_performed",
)
EXPECTED_SOLVER_FROZEN_BINDINGS = 43
CW17_CRITICAL_FUNCTION_AST_SHA256: dict[str, str] = {
    "_cw17_certificate_checks": "34d78b7a7e3baffb1f4307f52bc90b7c347b26027ea6ceba5ccd118ee3438837",
    "_cw17_original_space_checks": "950c667dcd437a06dcff6d27c32e5174f90e8ecc083d13854fc662558a718c93",
    "_mutation_paths": "1b719f1036fe415391f333b79b15c6667f9056318dbfc91db5cdb0e692afbf12",
    "cw17_basis_operator_norm_checks": "4dd2027df4078bb116b1210f5dc6f9e20da93ed86a57f0ae5bac20d61c639e5f",
    "cw17_candidate_identity_checks": "e3ceb864ea76cc6fa260de9b8fa562552ff172f5cd64a7d1445d12a75b01fde7",
    "cw17_candidate_record_checks": "a6bfb881abe8482f6834a1a08f73e92e1e842e0c2b4c83f64f7ae37ce4f03b43",
    "cw17_diagnostic_contract_checks": "825edd77df5019fa98d154df51f0ef2fc3a8ee37967596c3395a12e946ce4e65",
    "cw17_encoded_array_checks": "4e367434fbc7458376688b8cc3fcb0fc4d8b2ef72c371ee1694eca6a8b81e6b3",
    "cw17_fixture_contract_checks": "07cccc25ef98b6e7bd0cdd298d4a8b23cb17a8eec0f28dfaea782b0b57af420e",
    "cw17_formal_selection_checks": "65eed6ed15f7dd7ce12e3d4c02fabd616baae070b1122f7241542f606211f5b0",
    "cw17_math_canonical_json": "36d5a92bd07d8166fc38100c366fb9b781e26c5ee28f4ce21e65f9f14dbb53b1",
    "cw17_multiplier_capture_checks": "359d92d714361a70e8cf0733f21bd506fd497ad99fb3159367cf02f60ed4c042",
    "cw17_parent_manifest_adapter": "f1008560df161fcc4245978ef05b38e15106577b6a1a73a1620a0dd8d783e769",
    "cw17_preflight": "3c620cc7ad5da457d1a411eb58a7c3473cfb158b25c86166dbda9e0b86c845ee",
    "cw17_run_scope": "c1fb686cd6fb700427adcc9b1463dd828cc04f9ba50656c062f2a9361a69d2cb",
    "cw17_solver_ast_audit": "dc9833a5d058edbf0b7875d1bcf051a4b52c36755b8cd83e32406e3bb0af2aea",
    "cw17_stderr_audit": "f172a70b0e2ab8a04a36d75dfd27f592c6ecc3e0182483b89b80c00b8ec4192c",
    "cw17_terminal_validator_mutation_self_test": "e80722c554f0f6510e901cfffaf0f35e644aa0d3603c7a702bc7e8df913fedcd",
    "cw17_validate_static_payload": "e4e104e13b39701012e73f79e8ed43a7e72509f4d36ec070969f89dcb0019702",
    "cw17_validate_terminal_payload": "57070b2957c81d06779c596c5018902fa61155b76d72dc80a10e544e6e295238",
    "known_cw16_terminal_evidence": "986d7064c409f2acdf8be8b6ba36710d080ef7cf73924be89fb9e33330fb1ff9",
    "nested_mutate": "7c210e03919ae9df490d5ad9f98fa9c63ef20011b88016995b50764087c9ede2",
    "wrapper_source_audit": "ec02bdbd396b4dedbd4d9212fdc2116de5f0caac16fed38917cec824c06f5546",
}
CW17_SOLVER_CRITICAL_FUNCTION_AST_SHA256: dict[str, str] = {
    "load_modules": "bb9dc1b9726abc8ccda2411a9849bc2d7b885b7f582e12eee3e5de016f20c2da",
    "validate_cw16_closure": "91015ceb3dfc90d2d8891b59adccf3cd28e889db34da6157953d3e384e8f1b75",
    "validate_frozen_inputs_43": "da3da6df819797ee8c5ed30a375b00c00536cfafd2e27f67212340659c33c494",
    "validate_dual_math_contract": "2866357304c73d8585919c58bd04b8848cbf25d3d8d4250f3db669e0db1d8ded",
    "conservative_basis_operator_norm": "833fe364094e30551e471258a36eafde468aa110f860a14d48652cf848046eec",
    "_rebuild_fixture_and_certificates": "b8375f631b1282ace95a2be2aee8ff7fe84704f4aeb4a7f16b86eb1bd02df022",
    "certificate_failure_recovery_selftest": "f5f21f8a5cf68a608e6c4f916d90a1fee31f3fd38082b881937d5f1c403be8d6",
    "run_diagnostic": "492f0c50f50df1bd8b4f42186cc842ff584d924184678e9a71788d5c30bd41e9",
    "contract": "56260a4a236fba2ee4025d0ce6e092def53dd9410ee18cfbef7c65f7c28f639e",
    "static_source_audit": "fd9bc4dd902a9be8740e875cb292dd358b494eaa5a72857ce8598a201f044bf4",
    "selftest_result": "5b7badcff1086a0d098cb9ca10c68d359bfcf8908392c0b35089fcae40e2e648",
    "static_result": "6fce5af9f66a4e9c2d1a288e47d5e455f23ece4adbd94b12237b957e68147c45",
    "main": "fc9c41b9973c879dda55b3dc61b5577dfe00fb47a70d2c02d817bbce8380e1c0",
}
EXPECTED_LAUNCHER_CRITICAL_FUNCTIONS = (
    "cw17_run_scope",
    "cw17_solver_ast_audit",
    "wrapper_source_audit",
    "known_cw16_terminal_evidence",
    "cw17_math_canonical_json",
    "cw17_encoded_array_checks",
    "cw17_fixture_contract_checks",
    "cw17_basis_operator_norm_checks",
    "cw17_multiplier_capture_checks",
    "_cw17_original_space_checks",
    "_cw17_certificate_checks",
    "cw17_candidate_record_checks",
    "cw17_candidate_identity_checks",
    "cw17_formal_selection_checks",
    "cw17_diagnostic_contract_checks",
    "_mutation_paths",
    "nested_mutate",
    "cw17_terminal_validator_mutation_self_test",
    "cw17_parent_manifest_adapter",
    "cw17_validate_static_payload",
    "cw17_validate_terminal_payload",
    "cw17_preflight",
    "cw17_stderr_audit",
)
EXPECTED_SOLVER_CRITICAL_FUNCTIONS = (
    "load_modules",
    "validate_cw16_closure",
    "validate_frozen_inputs_43",
    "validate_dual_math_contract",
    "conservative_basis_operator_norm",
    "_rebuild_fixture_and_certificates",
    "certificate_failure_recovery_selftest",
    "run_diagnostic",
    "contract",
    "static_source_audit",
    "selftest_result",
    "static_result",
    "main",
)

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

CLASSIFICATION = {
    "specialist_valid_consumed": True,
    "numerical_diagnostic_only": True,
    "unique_official_model_count_before": 1,
    "unique_official_model_count_after": 1,
    "unique_official_model_count_delta": 0,
    "changed_model_official_evaluation": False,
    "candidate_consumer": False,
    "checkpoint_materialization": False,
    "broad_access": False,
    "gold_access": False,
    "network_upload_submission": False,
}

EXPECTED_SOLVER_CONTRACT = {
    "purpose": "capture_consumed_CW16_stage2_status9_without_new_model_evaluation",
    "frozen_input_count": 43,
    "old_input_count": 40,
    "new_CW16_closure_count": 3,
    "unique_official_model_count_before": 1,
    "unique_official_model_count_after": 1,
    "unique_official_model_count_delta": 0,
    "cumulative_official_budget_max": 12,
    "cumulative_official_budget_remaining": 11,
    "identical_bootstrap_reconstruction_access": True,
    "changed_model_official_evaluation": False,
    "candidate_consumer": False,
    "checkpoint_materialization": False,
    "dual_math_source_sha256": EXPECTED_DUAL_MATH_SHA256,
    "fixture_coordinate_scale": 0.001,
    "formal_dual_gap_scaled_max": 5e-13,
    "formal_raw_optimal_radius_max": 1e-9,
    "formal_stationarity_inf_max": 1e-10,
    "formal_complementarity_inf_max": 1e-10,
    "raw_radius_mapping_operator_norm_method": (
        "Gershgorin_with_gamma_4n_float64_dot_error_enclosure"
    ),
    "formal_raw_radius_requires_mapped_bound": True,
    "total_ball_captured_and_intentionally_omitted_from_dual": True,
    "formal_candidate_requires_exact_total_feasibility": True,
    "optimizer_status_is_not_acceptance": True,
    "exact_float_and_protocol_tolerance_feasibility_separate": True,
    "no_clip": True,
    "network_broad_gold_submission": False,
}


class ProtocolError(RuntimeError):
    """The frozen parent, CW16 evidence, or CW17 diagnostic contract is invalid."""


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


def cw17_math_canonical_json(value: Any) -> bytes:
    """Match the frozen dual-math JSON identity exactly, with no newline."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def root_relative(path: Path) -> str:
    resolved = Path(os.path.abspath(os.fspath(path)))
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError as error:
        raise ProtocolError(f"path escapes repository: {resolved}") from error


def read_frozen_parent() -> tuple[bytes, dict[str, Any]]:
    """Read the frozen CW16 launcher once and execute only those held bytes."""

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
            raise ProtocolError("CW16 parent launcher must be frozen 0555 one-link regular")
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
        raise ProtocolError("frozen CW16 parent launcher identity drift")
    return payload, {
        "path": root_relative(PARENT_LAUNCHER),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": "0555",
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "held_bytes_executed": True,
    }


def import_parent(source: bytes) -> ModuleType:
    module_name = f"_cw17_parent_one_shot_{PARENT_LAUNCHER_SHA256[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(PARENT_LAUNCHER)
    module.__package__ = ""
    module.__spec__ = None
    code = compile(source, str(PARENT_LAUNCHER), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


PARENT_SOURCE, PARENT_EVIDENCE = read_frozen_parent()
parent_adapter = import_parent(PARENT_SOURCE)
if not isinstance(getattr(parent_adapter, "engine", None), ModuleType):
    raise ProtocolError("frozen CW16 parent did not expose its inherited engine")
engine = parent_adapter.engine
PARENT_SOURCE_AUDIT = parent_adapter.wrapper_source_audit(PARENT_SOURCE)
if PARENT_SOURCE_AUDIT.get("pass") is not True:
    raise ProtocolError("frozen CW16 parent launcher source audit failed")
CW12_CORE_SOURCE_AUDIT = PARENT_SOURCE_AUDIT
for _audit_generation in range(4):
    nested_audit = CW12_CORE_SOURCE_AUDIT.get("parent_source_audit")
    if not isinstance(nested_audit, Mapping):
        raise ProtocolError("frozen CW16 source-audit ancestry is incomplete")
    CW12_CORE_SOURCE_AUDIT = nested_audit
CW12_CORE_REQUIRED_CHECKS = {
    "one_subprocess_call_site_exact",
    "shell_true_absent",
    "no_loop_around_execute_child",
    "run_and_static_argv_python_exact",
    "run_and_static_flags_exact",
    "run_and_static_solver_exact",
    "post_child_rehash_declared",
    "run_once_calls_post_child_rehash",
    "post_child_rehash_covers_solver_bindings_marker",
    "success_order_rehash_validate_stderr_stdout",
    "stderr_audit_receives_post_child_integrity",
}
CW12_CORE_CHECKS = CW12_CORE_SOURCE_AUDIT.get("checks")
if (
    CW12_CORE_SOURCE_AUDIT.get("pass") is not True
    or not isinstance(CW12_CORE_CHECKS, Mapping)
    or not CW12_CORE_REQUIRED_CHECKS.issubset(CW12_CORE_CHECKS)
    or not all(CW12_CORE_CHECKS.get(name) is True for name in CW12_CORE_REQUIRED_CHECKS)
):
    raise ProtocolError("frozen inherited one-shot execution core audit failed")

PARENT_COMPOSED_CHECKS = {
    "launcher_path_exact": parent_adapter.SCRIPT == PARENT_LAUNCHER,
    "launcher_sha_exact": PARENT_EVIDENCE["sha256"] == PARENT_LAUNCHER_SHA256,
    "solver_path_exact": parent_adapter.SOLVER == CW16_SOLVER,
    "solver_sha_exact": parent_adapter.EXPECTED_SOLVER_SHA256 == CW16_SOLVER_SHA256,
    "solver_manifest_count_exact_40": (
        parent_adapter.EXPECTED_SOLVER_FROZEN_BINDINGS == 40
    ),
    "static_status_exact": parent_adapter.EXPECTED_STATIC_STATUS
    == "static_ready_CW16_local_trust_run_implemented",
    "terminal_statuses_exact": parent_adapter.EXPECTED_TERMINAL_STATUSES
    == (
        "consumed_valid_CW16_local_trust_first_feasible",
        "closed_no_CW16_candidate",
    ),
    "parent_contract_frozen": parent_adapter.FINAL_SOLVER_CONTRACT_PENDING is False,
    "parent_static_validator_installed": engine.validate_static_payload
    is parent_adapter.cw16_validate_static_payload,
    "parent_terminal_validator_installed": engine.validate_terminal_payload
    is parent_adapter.cw16_validate_terminal_payload,
    "parent_preflight_installed": engine.preflight is parent_adapter.cw16_preflight,
    "base_execution_core_audit_exact": all(
        CW12_CORE_CHECKS.get(name) is True for name in CW12_CORE_REQUIRED_CHECKS
    ),
}
if not all(PARENT_COMPOSED_CHECKS.values()):
    raise ProtocolError(f"frozen CW16 composed contract drift: {PARENT_COMPOSED_CHECKS}")

# The generic CW12 core owns exactly one subprocess call site, O_EXCL target
# publication, post-child rehash, and lossless stdout/stderr publication.  The
# CW16 wrapper exposed that frozen core specifically so descendants can change
# only their schema and manifest validators.
_BASE_PREFLIGHT = parent_adapter._BASE_PREFLIGHT
_PARENT_TERMINAL_VALIDATOR = engine.validate_terminal_payload
_PARENT_STDERR_AUDIT = engine.validated_stderr_audit
_CW16_ENGINE_BINDING_NAMES = (
    "SCRIPT",
    "SOLVER",
    "SCHEMA",
    "ATTEMPT_SCHEMA",
    "FAILURE_SCHEMA",
    "SOLVER_SCHEMA",
    "ARTIFACT_ID",
    "ATTEMPT_MARKER",
    "STDOUT_OUTPUT",
    "STDERR_AUDIT",
    "EXPECTED_SOLVER_SHA256",
    "EXPECTED_STATIC_STATUS",
    "EXPECTED_TERMINAL_STATUSES",
    "EXPECTED_SOLVER_FROZEN_BINDINGS",
    "STATIC_ARGV",
    "RUN_ARGV",
    "CLASSIFICATION",
    "RUN_SCOPE",
)
_CW16_ENGINE_BINDINGS = {
    name: getattr(engine, name) for name in _CW16_ENGINE_BINDING_NAMES
}


def cw17_run_scope() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "source_CW16_status9_failure_only": True,
        "source_official_evaluation_count_before": CW16_OFFICIAL_COUNT_BEFORE,
        "source_official_evaluation_count_after": CW16_OFFICIAL_COUNT_AFTER,
        "official_evaluation_count_delta": CW16_OFFICIAL_COUNT_DELTA,
        "changed_model_evaluation_count": 0,
        "candidate_consumer_calls": 0,
        "checkpoint_writes": 0,
        "candidate_materializations": 0,
        "model_or_result_writes": 0,
        "broad_access": False,
        "gold_access": False,
        "network_access": False,
        "package_upload_submission": False,
        "promotion_evidence": False,
        "single_run_child": True,
        "retry_authorized": False,
    }


def cw17_solver_ast_audit() -> dict[str, Any]:
    """Bind every critical final solver body independently of its file SHA."""

    if FINAL_SOLVER_CONTRACT_PENDING:
        return {
            "checks": {"final_solver_contract_installed": False},
            "functions": {},
            "pass": False,
        }
    source, evidence = engine.read_regular_stable(
        SOLVER,
        "CW17 diagnostic solver for critical AST audit",
        expected_sha256=EXPECTED_SOLVER_SHA256,
        expected_mode=FROZEN_EXECUTABLE_MODE,
    )
    tree = ast.parse(source, filename=str(SOLVER))
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    observed = {
        name: sha256_bytes(
            ast.dump(nodes[name], include_attributes=False).encode("utf-8")
        )
        for name in CW17_SOLVER_CRITICAL_FUNCTION_AST_SHA256
        if name in nodes
    }
    checks = {
        "final_solver_contract_installed": FINAL_SOLVER_CONTRACT_PENDING is False,
        "critical_map_nonempty": bool(CW17_SOLVER_CRITICAL_FUNCTION_AST_SHA256),
        "critical_function_set_exact": set(CW17_SOLVER_CRITICAL_FUNCTION_AST_SHA256)
        == set(EXPECTED_SOLVER_CRITICAL_FUNCTIONS),
        "critical_functions_present": set(observed)
        == set(CW17_SOLVER_CRITICAL_FUNCTION_AST_SHA256),
        "critical_function_hashes_exact": observed
        == CW17_SOLVER_CRITICAL_FUNCTION_AST_SHA256,
        "solver_sha_exact": evidence.get("sha256") == EXPECTED_SOLVER_SHA256,
        "solver_mode_exact": evidence.get("mode_octal") == "0555",
        "solver_one_link": evidence.get("nlink") == 1,
    }
    return {
        "checks": checks,
        "functions": observed,
        "pass": all(checks.values()),
        "solver": evidence,
    }


def apply_cw17_overrides() -> None:
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
        "CLASSIFICATION": CLASSIFICATION,
        "RUN_SCOPE": cw17_run_scope(),
    }
    for name, value in values.items():
        setattr(engine, name, value)


apply_cw17_overrides()


def validate_parent_terminal_in_parent_context(payload: bytes) -> dict[str, Any]:
    """Call the frozen CW16 validator with its exact dynamic engine bindings."""

    current = {name: getattr(engine, name) for name in _CW16_ENGINE_BINDING_NAMES}
    try:
        for name, value in _CW16_ENGINE_BINDINGS.items():
            setattr(engine, name, value)
        return _PARENT_TERMINAL_VALIDATOR(payload)
    finally:
        for name, value in current.items():
            setattr(engine, name, value)


def known_cw16_terminal_evidence() -> dict[str, Any]:
    """Deep-validate the three immutable CW16 records and status-9 closure."""

    marker_raw, marker_evidence = engine.read_regular_stable(
        CW16_ATTEMPT_MARKER,
        "consumed CW16 attempt marker",
        expected_sha256=CW16_ATTEMPT_MARKER_SHA256,
        expected_mode=FROZEN_EVIDENCE_MODE,
    )
    stdout_raw, stdout_evidence = engine.read_regular_stable(
        CW16_CLOSED_STDOUT,
        "immutable CW16 terminal stdout",
        expected_sha256=CW16_CLOSED_STDOUT_SHA256,
        expected_mode=FROZEN_EVIDENCE_MODE,
    )
    stderr_raw, stderr_evidence = engine.read_regular_stable(
        CW16_STDERR_AUDIT,
        "immutable CW16 stderr audit",
        expected_sha256=CW16_STDERR_AUDIT_SHA256,
        expected_mode=FROZEN_EVIDENCE_MODE,
    )
    marker = engine.strict_json_object(marker_raw, "CW16 attempt marker")
    terminal = engine.strict_json_object(stdout_raw, "CW16 terminal stdout")
    stderr = engine.strict_json_object(stderr_raw, "CW16 stderr audit")
    parent_validation = validate_parent_terminal_in_parent_context(stdout_raw)

    second = terminal.get("second_stage")
    decision = second.get("decision") if isinstance(second, Mapping) else None
    iterations = second.get("iterations") if isinstance(second, Mapping) else None
    failure_row = (
        iterations[-1]
        if isinstance(iterations, list) and iterations and isinstance(iterations[-1], Mapping)
        else None
    )
    frozen = terminal.get("frozen_inputs")
    prior_records = frozen.get("records") if isinstance(frozen, Mapping) else None
    post = stderr.get("post_child_integrity")
    try:
        child_stderr = base64.b64decode(
            str(stderr.get("base64", "")).encode("ascii"), validate=True
        )
        child_stderr.decode("utf-8")
        stderr_roundtrip = True
    except (ValueError, UnicodeDecodeError):
        child_stderr = b""
        stderr_roundtrip = False

    checks = {
        "marker_size_exact": len(marker_raw) == CW16_ATTEMPT_MARKER_BYTES,
        "stdout_size_exact": len(stdout_raw) == CW16_CLOSED_STDOUT_BYTES,
        "stderr_audit_size_exact": len(stderr_raw) == CW16_STDERR_AUDIT_BYTES,
        "parent_terminal_validation_all_true": parent_validation.get("payload")
        == terminal
        and engine.all_true_checks(parent_validation.get("checks"))
        and engine.all_true_checks(parent_validation.get("branch_checks")),
        "marker_schema_exact": marker.get("schema_version")
        == "ptcg-cw16-consumed-valid-official6-one-shot-attempt-v1",
        "marker_status_exact": marker.get("status")
        == "one_shot_attempt_consumed_before_cuda_and_official6",
        "marker_attempt_once_no_retry": marker.get("attempts_authorized") == 1
        and marker.get("retry_authorized") is False,
        "marker_before_cuda": marker.get(
            "marker_created_before_cuda_and_official6"
        )
        is True,
        "marker_launcher_exact": isinstance(marker.get("launcher"), Mapping)
        and marker["launcher"].get("path") == root_relative(PARENT_LAUNCHER)
        and marker["launcher"].get("sha256") == PARENT_LAUNCHER_SHA256
        and marker["launcher"].get("mode_octal") == "0555",
        "marker_solver_exact": isinstance(marker.get("solver"), Mapping)
        and marker["solver"].get("path") == root_relative(CW16_SOLVER)
        and marker["solver"].get("sha256") == CW16_SOLVER_SHA256
        and marker["solver"].get("mode_octal") == "0555",
        "terminal_schema_exact": terminal.get("schema_version")
        == "ptcg-cw16-consumed-valid-official-b256-local-trust-v1",
        "terminal_status_exact": terminal.get("status") == CW16_STATUS
        and isinstance(second, Mapping)
        and second.get("status") == CW16_STATUS,
        "old_status9_close_reason_exact": isinstance(decision, Mapping)
        and decision.get("close_reason") == CW16_CLOSE_REASON
        and isinstance(failure_row, Mapping)
        and failure_row.get("close_reason") == CW16_CLOSE_REASON
        and failure_row.get("kind") == "fail_closed_before_official_proposal",
        "official_count_before_after_exact_one": isinstance(decision, Mapping)
        and decision.get("bootstrap_evaluation_count") == CW16_OFFICIAL_COUNT_BEFORE
        and decision.get("official_candidate_evaluation_count")
        == CW16_OFFICIAL_COUNT_AFTER
        and isinstance(failure_row, Mapping)
        and failure_row.get("official_evaluation_count")
        == CW16_OFFICIAL_COUNT_AFTER,
        "no_candidate_consumer": isinstance(decision, Mapping)
        and decision.get("candidate_consumer_called") is False,
        "no_terminal_candidate": isinstance(decision, Mapping)
        and decision.get("terminal_iteration") is None
        and decision.get("terminal_model_state_sha256") is None,
        "anchor_identity_exact": isinstance(decision, Mapping)
        and decision.get("terminal_active_cut_count") == CW16_ACTIVE_CUT_COUNT
        and decision.get("terminal_active_CW11_anchor_count")
        == CW16_ACTIVE_CUT_COUNT
        and decision.get("terminal_active_CW11_anchor_ledger_sha256")
        == CW16_ANCHOR_LEDGER_SHA256,
        "additional_l2_exact": isinstance(decision, Mapping)
        and decision.get("terminal_additional_l2") == CW16_ADDITIONAL_L2,
        "frozen_manifest_exact40": isinstance(frozen, Mapping)
        and frozen.get("all_exact") is True
        and frozen.get("binding_count") == 40
        and isinstance(prior_records, list)
        and len(prior_records) == 40,
        "stderr_schema_exact": stderr.get("schema_version")
        == "ptcg-cw16-consumed-valid-official6-one-shot-stderr-audit-v1",
        "stderr_lossless_roundtrip": stderr_roundtrip
        and len(child_stderr) == CW16_CHILD_STDERR_BYTES
        and sha256_bytes(child_stderr) == CW16_CHILD_STDERR_SHA256
        and stderr.get("bytes") == CW16_CHILD_STDERR_BYTES
        and stderr.get("sha256") == CW16_CHILD_STDERR_SHA256,
        "stderr_checks_all_true": engine.all_true_checks(stderr.get("checks")),
        "post_child_integrity_pass": isinstance(post, Mapping)
        and post.get("pass") is True
        and engine.all_true_checks(post.get("checks")),
    }
    if not all(checks.values()):
        raise ProtocolError(f"immutable CW16 terminal evidence failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "attempt_marker": marker_evidence,
        "terminal_stdout": stdout_evidence,
        "stderr_audit": stderr_evidence,
        "prior_frozen_records": prior_records,
        "old_stage2": {
            "status": CW16_STAGE2_STATUS,
            "message": CW16_STAGE2_MESSAGE,
            "close_reason": CW16_CLOSE_REASON,
        },
        "official_evaluation_count": {
            "before": CW16_OFFICIAL_COUNT_BEFORE,
            "after": CW16_OFFICIAL_COUNT_AFTER,
            "delta": CW16_OFFICIAL_COUNT_DELTA,
        },
    }


def cw17_encoded_array_checks(
    value: Any,
    *,
    expected_shape: Sequence[int] | None = None,
) -> dict[str, bool]:
    """Validate a complete little-endian float64 payload without NumPy."""

    expected_keys = {"dtype", "shape", "bytes", "sha256", "base64"}
    mapping = isinstance(value, Mapping)
    if not mapping:
        return {"mapping": False}
    shape = value.get("shape")
    shape_valid = (
        isinstance(shape, list)
        and 1 <= len(shape) <= 2
        and all(type(item) is int and item > 0 for item in shape)
    )
    item_count = math.prod(shape) if shape_valid else 0
    expected_bytes = item_count * 8
    encoded = value.get("base64")
    length_valid = (
        type(value.get("bytes")) is int
        and value.get("bytes") == expected_bytes
        and expected_bytes <= 800_000_000
        and isinstance(encoded, str)
        and len(encoded) == 4 * ((expected_bytes + 2) // 3)
    )
    raw = b""
    base64_valid = False
    if length_valid:
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
            base64_valid = len(raw) == expected_bytes
        except (ValueError, UnicodeEncodeError):
            base64_valid = False
    finite = False
    if base64_valid:
        finite = all(math.isfinite(item[0]) for item in struct.iter_unpack("<d", raw))
    return {
        "mapping": mapping,
        "keys_exact": set(value) == expected_keys,
        "dtype_exact": value.get("dtype") == "<f8",
        "shape_valid": shape_valid,
        "shape_exact": expected_shape is None or shape == list(expected_shape),
        "encoded_length_exact": length_valid,
        "base64_roundtrip_exact": base64_valid,
        "raw_sha256_exact": base64_valid
        and isinstance(value.get("sha256"), str)
        and len(value["sha256"]) == 64
        and sha256_bytes(raw) == value["sha256"],
        "all_float64_finite": finite,
    }


def cw17_fixture_contract_checks(
    value: Any,
    *,
    reduced_rank: int,
) -> dict[str, bool]:
    """Validate and aggregate-hash the full captured 91-row dual fixture."""

    if not isinstance(value, Mapping):
        return {"mapping": False}
    arrays = value.get("arrays")
    identity = {key: value[key] for key in value if key != "fixture_sha256"}
    K = arrays.get("K") if isinstance(arrays, Mapping) else None
    q = arrays.get("q") if isinstance(arrays, Mapping) else None
    center = arrays.get("center") if isinstance(arrays, Mapping) else None
    K_checks = cw17_encoded_array_checks(K, expected_shape=(91, reduced_rank))
    q_checks = cw17_encoded_array_checks(q, expected_shape=(91,))
    center_checks = cw17_encoded_array_checks(center, expected_shape=(reduced_rank,))
    rho = value.get("rho")
    return {
        "mapping": True,
        "keys_exact": set(value)
        == {
            "schema_version",
            "coordinate_scale",
            "rho",
            "row_count",
            "reduced_dimension",
            "arrays",
            "fixture_sha256",
        },
        "schema_exact": value.get("schema_version")
        == "ptcg-cw17-stage2-dual-math-v1",
        "coordinate_scale_exact": value.get("coordinate_scale") == 0.001,
        "rho_positive_finite": type(rho) is float and math.isfinite(rho) and rho > 0.0,
        "dimensions_exact": value.get("row_count") == 91
        and value.get("reduced_dimension") == reduced_rank,
        "array_names_exact": isinstance(arrays, Mapping)
        and set(arrays) == {"K", "q", "center"},
        "K_full_bytes_valid": all(K_checks.values()),
        "q_full_bytes_valid": all(q_checks.values()),
        "center_full_bytes_valid": all(center_checks.values()),
        "aggregate_fixture_sha256_exact": isinstance(value.get("fixture_sha256"), str)
        and len(value["fixture_sha256"]) == 64
        and value["fixture_sha256"]
        == sha256_bytes(cw17_math_canonical_json(identity)),
    }


def cw17_basis_operator_norm_checks(
    value: Any,
    *,
    reduced_rank: int,
    actor_dimension: int,
) -> dict[str, bool]:
    """Validate the conservative raw-radius mapping certificate."""

    if not isinstance(value, Mapping):
        return {"mapping": False}
    checks = value.get("checks")
    upper = value.get("operator_norm_upper")
    observed = value.get("observed_spectral_norm_audit_only")
    return {
        "mapping": True,
        "pass_true": value.get("pass") is True,
        "checks_all_true": isinstance(checks, Mapping)
        and set(checks)
        == {
            "finite_positive_upper_bound",
            "upper_not_below_observed_spectral_norm",
            "gamma_strictly_between_zero_and_one",
            "gram_shapes_exact",
        }
        and engine.all_true_checks(checks),
        "method_exact": value.get("method")
        == "Gershgorin_with_gamma_4n_float64_dot_error_enclosure",
        "basis_shape_exact": value.get("basis_shape")
        == [reduced_rank, actor_dimension],
        "upper_finite_positive": type(upper) is float
        and math.isfinite(upper)
        and upper > 0.0,
        "upper_not_below_observed": type(observed) is float
        and math.isfinite(observed)
        and type(upper) is float
        and upper >= observed,
    }


def cw17_multiplier_capture_checks(
    value: Any,
    *,
    multiplier_count: int,
    reduced_rank: int,
) -> dict[str, bool]:
    """Validate arbitrary unusable multipliers and the exact usable warm map."""

    if not isinstance(value, Mapping):
        return {"mapping": False}
    count_valid = (
        type(multiplier_count) is int and 0 <= multiplier_count <= 1_000_000
    )
    audit = value.get("multiplier_capture_audit")
    audit_keys = {
        "shape_exact",
        "finite",
        "usable_as_warm_start",
        "fallback_to_zero_if_unusable",
    }
    audit_valid = (
        isinstance(audit, Mapping)
        and set(audit) == audit_keys
        and all(type(audit.get(name)) is bool for name in audit_keys)
    )
    shape_exact = audit.get("shape_exact") if audit_valid else None
    finite = audit.get("finite") if audit_valid else None
    usable = audit.get("usable_as_warm_start") if audit_valid else None
    multipliers = value.get("multipliers")

    multiplier_raw = b""
    multiplier_values: tuple[float, ...] = ()
    multiplier_shape: list[int] | None = None
    multiplier_payload_valid = False
    if isinstance(multipliers, Mapping) and count_valid and multiplier_count > 0:
        shape = multipliers.get("shape")
        shape_valid = (
            isinstance(shape, list)
            and len(shape) <= 64
            and all(type(item) is int and item > 0 for item in shape)
        )
        item_count = math.prod(shape) if shape_valid else 0
        expected_bytes = item_count * 8
        encoded = multipliers.get("base64")
        if (
            set(multipliers) == {"dtype", "shape", "bytes", "sha256", "base64"}
            and multipliers.get("dtype") == "<f8"
            and item_count == multiplier_count
            and expected_bytes <= 8_000_000
            and multipliers.get("bytes") == expected_bytes
            and isinstance(encoded, str)
            and len(encoded) == 4 * ((expected_bytes + 2) // 3)
        ):
            try:
                multiplier_raw = base64.b64decode(
                    encoded.encode("ascii"), validate=True
                )
            except (ValueError, UnicodeEncodeError):
                multiplier_raw = b""
            if (
                len(multiplier_raw) == expected_bytes
                and isinstance(multipliers.get("sha256"), str)
                and multipliers["sha256"] == sha256_bytes(multiplier_raw)
            ):
                multiplier_values = tuple(
                    item[0] for item in struct.iter_unpack("<d", multiplier_raw)
                )
                multiplier_shape = list(shape)
                multiplier_payload_valid = all(
                    math.isfinite(item) for item in multiplier_values
                )

    warm = value.get("warm_dual_without_redundant_total_ball")
    warm_checks = cw17_encoded_array_checks(warm, expected_shape=(92,))
    warm_raw = b""
    warm_values: tuple[float, ...] = ()
    if all(warm_checks.values()) and isinstance(warm, Mapping):
        try:
            warm_raw = base64.b64decode(
                str(warm["base64"]).encode("ascii"), validate=True
            )
            warm_values = tuple(item[0] for item in struct.iter_unpack("<d", warm_raw))
        except (ValueError, UnicodeEncodeError, struct.error):
            warm_raw = b""
            warm_values = ()

    multiplier_none_expected = count_valid and (
        multiplier_count == 0 or finite is False
    )
    encoded_expected = count_valid and multiplier_count > 0 and finite is True
    shape_exact_expected = (
        encoded_expected
        and multiplier_payload_valid
        and multiplier_shape == [93]
    )
    usable_expected = shape_exact is True and finite is True
    usable_warm_expected = (
        usable is True
        and multiplier_payload_valid
        and len(multiplier_values) == 93
        and math.isfinite(2.0 * multiplier_values[-1])
        and warm_values
        == (*multiplier_values[:91], 2.0 * multiplier_values[-1])
    )
    fallback_warm_expected = usable is not True and warm_raw == bytes(92 * 8)
    return {
        "mapping": True,
        "old_stage2_keys_exact": set(value)
        == {
            "success",
            "status",
            "message",
            "iterations",
            "function_evaluations",
            "x_scaled",
            "multipliers",
            "multiplier_capture_audit",
            "warm_dual_without_redundant_total_ball",
        },
        "count_bounded_nonnegative_int": count_valid,
        "stage2_x_full_payload": type(reduced_rank) is int
        and reduced_rank > 0
        and all(
            cw17_encoded_array_checks(
                value.get("x_scaled"), expected_shape=(reduced_rank,)
            ).values()
        ),
        "audit_keys_and_booleans_exact": audit_valid,
        "fallback_declared_true": audit_valid
        and audit.get("fallback_to_zero_if_unusable") is True,
        "usable_equals_shape_and_finite": audit_valid
        and usable is (shape_exact is True and finite is True),
        "multipliers_none_iff_empty_or_nonfinite": (
            multipliers is None
        )
        is multiplier_none_expected,
        "finite_nonempty_payload_complete": not encoded_expected
        or multiplier_payload_valid,
        "encoded_item_product_equals_count": not encoded_expected
        or len(multiplier_values) == multiplier_count,
        "shape_exact_iff_finite_encoded_shape_93": audit_valid
        and (finite is not True or shape_exact is shape_exact_expected),
        "warm_payload_complete_92": all(warm_checks.values())
        and len(warm_values) == 92,
        "usable_warm_exact_concat": usable is not True or usable_warm_expected,
        "unusable_warm_exact_all_zero": usable is True or fallback_warm_expected,
        "usable_boolean_consistent": audit_valid and usable is usable_expected,
    }


def _cw17_original_space_checks(
    value: Any,
    *,
    reduced_rank: int,
    actor_dimension: int,
) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {"mapping": False}
    checks = value.get("checks")
    scaled = value.get("candidate_scaled")
    raw = value.get("candidate_raw")
    scaled_checks = cw17_encoded_array_checks(
        scaled, expected_shape=(reduced_rank,)
    )
    raw_checks = cw17_encoded_array_checks(raw, expected_shape=(actor_dimension,))
    declared_pass = isinstance(checks, Mapping) and all(
        item is True for item in checks.values()
    )
    return {
        "mapping": True,
        "checks_exact": isinstance(checks, Mapping)
        and set(checks)
        == {
            "anchor_existing_hard_gate",
            "local_existing_hard_gate",
            "total_existing_hard_gate",
            "trust_existing_hard_gate",
            "no_clip",
        },
        "pass_matches_checks": value.get("pass") is declared_pass,
        "no_clip_true": isinstance(checks, Mapping)
        and checks.get("no_clip") is True,
        "scaled_array_valid": all(scaled_checks.values()),
        "raw_array_valid": all(raw_checks.values()),
        "raw_sha_repeated_exact": isinstance(raw, Mapping)
        and value.get("candidate_raw_float64_le_sha256") == raw.get("sha256"),
        "metrics_finite": all(
            type(value.get(name)) is float and math.isfinite(value[name])
            for name in (
                "anchor_residual_min",
                "local_prediction_min",
                "z_star",
                "total_l2",
                "trust_l2",
            )
        ),
    }


def _cw17_certificate_checks(
    value: Any,
    *,
    reduced_rank: int,
) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {"mapping": False}
    checks = value.get("checks")
    gates = value.get("formal_gate_names")
    exact = value.get("exact_feasibility")
    dual = value.get("dual")
    selected = dual.get("selected") if isinstance(dual, Mapping) else None
    formal_gate_set = {
        "exact_float_feasible",
        "dual_variables_nonnegative_finite",
        "conservative_gap_nonnegative",
        "dual_gap_at_most_5e_13",
        "raw_optimal_radius_at_most_1e_9",
        "stationarity_inf_at_most_1e_10",
        "complementarity_inf_at_most_1e_10",
        "direct_gap_decomposition_matches",
        "no_clip",
    }
    formal_from_checks = isinstance(checks, Mapping) and all(
        checks.get(name) is True for name in formal_gate_set
    )
    selected_variables = (
        cw17_encoded_array_checks(selected.get("variables"), expected_shape=(92,))
        if isinstance(selected, Mapping)
        else {"mapping": False}
    )
    selected_u = (
        cw17_encoded_array_checks(
            selected.get("u_dual"), expected_shape=(reduced_rank,)
        )
        if isinstance(selected, Mapping)
        else {"mapping": False}
    )
    selected_gradient = (
        cw17_encoded_array_checks(selected.get("gradient"), expected_shape=(92,))
        if isinstance(selected, Mapping)
        else {"mapping": False}
    )
    return {
        "mapping": True,
        "formal_gates_exact": isinstance(gates, list)
        and gates == sorted(formal_gate_set),
        "formal_pass_matches_gates": value.get("formal_certificate_pass")
        is formal_from_checks,
        "optimizer_status_not_acceptance": value.get(
            "optimizer_status_not_used_for_acceptance"
        )
        is True,
        "no_clip": value.get("clipped_vector_applied") is False
        and isinstance(checks, Mapping)
        and checks.get("no_clip") is True,
        "primal_array_valid": all(
            cw17_encoded_array_checks(
                value.get("primal"), expected_shape=(reduced_rank,)
            ).values()
        ),
        "linear_slack_array_valid": all(
            cw17_encoded_array_checks(
                value.get("linear_slack"), expected_shape=(91,)
            ).values()
        ),
        "exact_feasibility_typed": isinstance(exact, Mapping)
        and type(exact.get("linear_all_nonnegative")) is bool
        and all(
            type(exact.get(name)) is int and exact[name] in {-1, 0, 1}
            for name in (
                "linear_min_sign",
                "trust_squared_slack_sign",
                "total_squared_slack_sign",
            )
        ),
        "dual_fixed_starts_exact": isinstance(dual, Mapping)
        and dual.get("pass") is True
        and dual.get("fixed_start_order")
        == ["zero", "slsqp_or_zero", "active_nnls"]
        and dual.get("optimizer_status_is_not_acceptance") is True
        and dual.get("optimizer_point_projection_allowed") is False
        and isinstance(dual.get("records"), list)
        and len(dual["records"]) == 3,
        "selected_dual_checks_all_true": isinstance(selected, Mapping)
        and selected.get("pass") is True
        and engine.all_true_checks(selected.get("checks")),
        "selected_dual_arrays_valid": all(selected_variables.values())
        and all(selected_u.values())
        and all(selected_gradient.values()),
        "formal_exact_feasibility_consistent": value.get(
            "formal_certificate_pass"
        )
        is not True
        or (
            isinstance(exact, Mapping)
            and exact.get("linear_all_nonnegative") is True
            and exact.get("linear_min_sign") >= 0
            and exact.get("trust_squared_slack_sign") >= 0
            and exact.get("total_squared_slack_sign") >= 0
        ),
    }


def cw17_candidate_record_checks(
    value: Any,
    *,
    reduced_rank: int,
    actor_dimension: int,
) -> dict[str, bool]:
    """Validate one candidate while permitting serialized backend failure."""

    if not isinstance(value, Mapping):
        return {"mapping": False}
    candidate_available = value.get("candidate_available")
    certificate_available = value.get("certificate_available")
    original_available = value.get("original_space_available")
    scaled_checks = (
        cw17_encoded_array_checks(
            value.get("candidate_scaled"), expected_shape=(reduced_rank,)
        )
        if candidate_available is True
        else {"absent": value.get("candidate_scaled") is None}
    )
    certificate_checks = (
        _cw17_certificate_checks(
            value.get("certificate"), reduced_rank=reduced_rank
        )
        if certificate_available is True
        else {
            "certificate_absent": value.get("certificate") is None,
            "mapped_absent": value.get("mapped_raw_radius_certificate") is None,
        }
    )
    original_checks = (
        _cw17_original_space_checks(
            value.get("original_space"),
            reduced_rank=reduced_rank,
            actor_dimension=actor_dimension,
        )
        if original_available is True
        else {"absent": value.get("original_space") is None}
    )
    mapped = value.get("mapped_raw_radius_certificate")
    cert = value.get("certificate")
    mapped_consistent = certificate_available is not True or (
        isinstance(mapped, Mapping)
        and isinstance(mapped.get("checks"), Mapping)
        and mapped.get("pass")
        is all(item is True for item in mapped["checks"].values())
        and mapped.get("multiplication_rounding")
        == "float64_product_then_nextafter_up"
        and isinstance(cert, Mapping)
        and mapped.get("reduced_radius_bound")
        == cert.get("raw_optimal_radius_bound")
        and type(mapped.get("mapped_raw_radius_bound")) is float
        and math.isfinite(mapped["mapped_raw_radius_bound"])
        and (
            mapped.get("pass") is not True
            or mapped["mapped_raw_radius_bound"] <= 1e-9
        )
    )
    errors = value.get("errors")
    return {
        "mapping": True,
        "keys_exact": set(value)
        == {
            "candidate_available",
            "candidate_scaled",
            "certificate_available",
            "certificate",
            "mapped_raw_radius_certificate",
            "original_space_available",
            "original_space",
            "errors",
        },
        "availability_flags_boolean": all(
            type(item) is bool
            for item in (candidate_available, certificate_available, original_available)
        ),
        "candidate_array_consistent": all(scaled_checks.values()),
        "certificate_consistent": all(certificate_checks.values()),
        "mapped_radius_consistent": mapped_consistent,
        "original_space_consistent": all(original_checks.values()),
        "errors_serializable": isinstance(errors, list)
        and all(
            isinstance(error, Mapping)
            and set(error) == {"phase", "error_type", "error_message"}
            and all(isinstance(error[name], str) for name in error)
            for error in errors
        ),
        "unavailable_has_error": (
            candidate_available is True
            and certificate_available is True
            and original_available is True
        )
        or bool(errors),
    }


def cw17_candidate_identity_checks(
    candidates: Any,
    *,
    stage1: Any,
    stage2: Any,
    basis_operator_norm_certificate: Any,
) -> dict[str, bool]:
    """Close every producer identity and mapped-radius arithmetic relation."""

    names = (
        "old_slsqp_stage2_last_x",
        "dual_recovered",
        "stage1_feasible_upper_bound",
    )
    if (
        not isinstance(candidates, Mapping)
        or set(candidates) != set(names)
        or not isinstance(stage1, Mapping)
        or not isinstance(stage2, Mapping)
        or not isinstance(basis_operator_norm_certificate, Mapping)
    ):
        return {"mappings_and_candidate_keys_exact": False}
    old = candidates[names[0]]
    dual_recovered = candidates[names[1]]
    stage1_candidate = candidates[names[2]]
    if not all(isinstance(record, Mapping) for record in candidates.values()):
        return {
            "mappings_and_candidate_keys_exact": True,
            "candidate_records_mapping": False,
        }
    upper = basis_operator_norm_certificate.get("operator_norm_upper")

    component_relations: list[bool] = []
    mapped_relations: list[bool] = []
    for record in candidates.values():
        candidate = record.get("candidate_scaled")
        certificate = record.get("certificate")
        original = record.get("original_space")
        candidate_available = record.get("candidate_available") is True
        certificate_available = record.get("certificate_available") is True
        original_available = record.get("original_space_available") is True
        component_relations.append(
            (not certificate_available)
            or (
                candidate_available
                and isinstance(certificate, Mapping)
                and certificate.get("primal") == candidate
            )
        )
        component_relations.append(
            (not original_available)
            or (
                candidate_available
                and isinstance(original, Mapping)
                and original.get("candidate_scaled") == candidate
            )
        )
        mapped = record.get("mapped_raw_radius_certificate")
        if not certificate_available:
            mapped_relations.append(mapped is None)
            continue
        cert_checks = certificate.get("checks") if isinstance(certificate, Mapping) else None
        mapped_checks = mapped.get("checks") if isinstance(mapped, Mapping) else None
        reduced = mapped.get("reduced_radius_bound") if isinstance(mapped, Mapping) else None
        expected_mapped = (
            0.0
            if reduced == 0.0
            else (
                math.nextafter(reduced * upper, math.inf)
                if type(reduced) is float
                and reduced > 0.0
                and type(upper) is float
                and math.isfinite(reduced)
                and math.isfinite(upper)
                and math.isfinite(reduced * upper)
                else None
            )
        )
        expected_basis_check = basis_operator_norm_certificate.get("pass") is True
        expected_reduced_check = isinstance(cert_checks, Mapping) and cert_checks.get(
            "raw_optimal_radius_at_most_1e_9"
        ) is True
        expected_mapped_check = (
            type(expected_mapped) is float and expected_mapped <= 1e-9
        )
        mapped_relations.append(
            isinstance(certificate, Mapping)
            and isinstance(mapped, Mapping)
            and set(mapped)
            == {
                "checks",
                "basis_operator_norm_upper",
                "reduced_radius_bound",
                "mapped_raw_radius_bound",
                "multiplication_rounding",
                "pass",
            }
            and isinstance(mapped_checks, Mapping)
            and set(mapped_checks)
            == {
                "basis_operator_norm_certificate_pass",
                "mapped_raw_optimal_radius_at_most_1e_9",
                "reduced_certificate_raw_radius_gate_pass",
            }
            and type(reduced) is float
            and math.isfinite(reduced)
            and reduced >= 0.0
            and certificate.get("raw_optimal_radius_bound") == reduced
            and mapped.get("basis_operator_norm_upper") == upper
            and mapped.get("mapped_raw_radius_bound") == expected_mapped
            and mapped.get("multiplication_rounding")
            == "float64_product_then_nextafter_up"
            and mapped_checks.get("basis_operator_norm_certificate_pass")
            is expected_basis_check
            and mapped_checks.get("reduced_certificate_raw_radius_gate_pass")
            is expected_reduced_check
            and mapped_checks.get("mapped_raw_optimal_radius_at_most_1e_9")
            is expected_mapped_check
            and mapped.get("pass")
            is (
                expected_basis_check
                and expected_reduced_check
                and expected_mapped_check
            )
        )

    old_certificate = old.get("certificate") if isinstance(old, Mapping) else None
    old_selected = (
        old_certificate.get("dual", {}).get("selected")
        if isinstance(old_certificate, Mapping)
        and isinstance(old_certificate.get("dual"), Mapping)
        else None
    )
    old_u_dual = (
        old_selected.get("u_dual") if isinstance(old_selected, Mapping) else None
    )
    old_certificate_available = old.get("certificate_available") is True
    return {
        "mappings_and_candidate_keys_exact": True,
        "candidate_records_mapping": True,
        "basis_upper_finite_positive": type(upper) is float
        and math.isfinite(upper)
        and upper > 0.0,
        "old_candidate_equals_old_stage2_x": old.get("candidate_available") is True
        and old.get("candidate_scaled") == stage2.get("x_scaled"),
        "stage1_candidate_equals_stage1_x": stage1_candidate.get(
            "candidate_available"
        )
        is True
        and stage1_candidate.get("candidate_scaled") == stage1.get("x_scaled"),
        "dual_recovered_dependency_exact": (
            old_certificate_available
            and dual_recovered.get("candidate_available") is True
            and old_u_dual is not None
            and dual_recovered.get("candidate_scaled") == old_u_dual
        )
        or (
            not old_certificate_available
            and dual_recovered.get("candidate_available") is False
            and dual_recovered.get("candidate_scaled") is None
        ),
        "available_component_vectors_identical": all(component_relations),
        "mapped_radius_formula_and_checks_exact": all(mapped_relations),
    }


def cw17_formal_selection_checks(value: Any) -> dict[str, bool]:
    """Recompute the frozen candidate priority and formal eligibility list."""

    if not isinstance(value, Mapping):
        return {"mapping": False}
    priority = [
        "old_slsqp_stage2_last_x",
        "dual_recovered",
        "stage1_feasible_upper_bound",
    ]
    candidates = value.get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != set(priority):
        return {"mapping": True, "candidate_keys_exact": False}

    def eligible(record: Any) -> bool:
        certificate = record.get("certificate") if isinstance(record, Mapping) else None
        exact = (
            certificate.get("exact_feasibility")
            if isinstance(certificate, Mapping)
            else None
        )
        mapped = (
            record.get("mapped_raw_radius_certificate")
            if isinstance(record, Mapping)
            else None
        )
        original = record.get("original_space") if isinstance(record, Mapping) else None
        return (
            isinstance(record, Mapping)
            and record.get("certificate_available") is True
            and record.get("original_space_available") is True
            and isinstance(certificate, Mapping)
            and certificate.get("formal_certificate_pass") is True
            and isinstance(exact, Mapping)
            and type(exact.get("total_squared_slack_sign")) is int
            and exact["total_squared_slack_sign"] >= 0
            and isinstance(mapped, Mapping)
            and mapped.get("pass") is True
            and isinstance(original, Mapping)
            and original.get("pass") is True
        )

    expected = [name for name in priority if eligible(candidates[name])]
    return {
        "mapping": True,
        "candidate_keys_exact": True,
        "priority_exact": value.get("candidate_priority") == priority,
        "formal_list_exact": value.get("formally_certified_candidates") == expected,
        "first_exact": value.get("first_formally_certified_candidate")
        == (expected[0] if expected else None),
        "formal_build_boolean_exact": value.get("formal_CW17_may_be_built")
        is bool(expected),
        "diagnostic_only_true": value.get(
            "diagnostic_does_not_materialize_or_consume_candidate"
        )
        is True,
        "fixture_survives_backend_errors": value.get(
            "fixture_and_reconstruction_survive_certificate_backend_errors"
        )
        is True,
    }


def cw17_diagnostic_contract_checks(value: Any) -> dict[str, bool]:
    """Validate captured SLSQP identity, full fixture, basis, and candidates."""

    if not isinstance(value, Mapping):
        return {"mapping": False}
    dimensions = value.get("dimensions")
    actor = dimensions.get("actor_original") if isinstance(dimensions, Mapping) else 0
    rank = dimensions.get("reduced_rank") if isinstance(dimensions, Mapping) else 0
    dimension_valid = (
        isinstance(dimensions, Mapping)
        and type(actor) is int
        and actor == 65793
        and type(rank) is int
        and 43 <= rank <= 81
        and dimensions.get("anchor_rows") == 50
        and dimensions.get("local_physical_rows") == 41
        and dimensions.get("span_shape") == [92, actor]
        and dimensions.get("expected_stage2_multiplier_count") == 93
        and type(dimensions.get("stage2_multiplier_count")) is int
        and 0 <= dimensions["stage2_multiplier_count"] <= 1_000_000
    )
    fixture = value.get("fixture")
    reconstruction = value.get("reconstruction_map")
    basis = (
        reconstruction.get("basis_reduced_by_actor")
        if isinstance(reconstruction, Mapping)
        else None
    )
    basis_array_checks = cw17_encoded_array_checks(
        basis,
        expected_shape=(rank, actor) if dimension_valid else None,
    )
    fixture_checks = cw17_fixture_contract_checks(
        fixture,
        reduced_rank=rank if type(rank) is int and rank > 0 else 1,
    )
    operator_checks = cw17_basis_operator_norm_checks(
        value.get("basis_operator_norm_certificate"),
        reduced_rank=rank if type(rank) is int else 0,
        actor_dimension=actor if type(actor) is int else 0,
    )
    capture = value.get("captured_constraint_audit")
    capture_checks = capture.get("checks") if isinstance(capture, Mapping) else None
    scaling = value.get("scaling")
    stage1 = value.get("stage1")
    stage2 = value.get("old_stage2")
    multiplier_checks = cw17_multiplier_capture_checks(
        stage2,
        multiplier_count=(
            dimensions.get("stage2_multiplier_count")
            if isinstance(dimensions, Mapping)
            else -1
        ),
        reduced_rank=rank if type(rank) is int else 0,
    )
    candidates = value.get("candidates")
    candidate_checks = {
        name: cw17_candidate_record_checks(
            record,
            reduced_rank=rank,
            actor_dimension=actor,
        )
        for name, record in candidates.items()
    } if isinstance(candidates, Mapping) and dimension_valid else {}
    candidate_identity_checks = cw17_candidate_identity_checks(
        candidates,
        stage1=stage1,
        stage2=stage2,
        basis_operator_norm_certificate=value.get(
            "basis_operator_norm_certificate"
        ),
    )
    selection_checks = cw17_formal_selection_checks(value)
    expected_backend_errors = (
        [
            {"candidate": name, **dict(error)}
            for name in value.get("candidate_priority", [])
            for error in candidates[name].get("errors", [])
        ]
        if isinstance(candidates, Mapping)
        and value.get("candidate_priority")
        == [
            "old_slsqp_stage2_last_x",
            "dual_recovered",
            "stage1_feasible_upper_bound",
        ]
        and all(name in candidates for name in value["candidate_priority"])
        else None
    )
    fixture_rho = fixture.get("rho") if isinstance(fixture, Mapping) else None
    return {
        "mapping": True,
        "top_level_keys_exact": set(value)
        == {
            "checks",
            "pass",
            "dimensions",
            "singular_values",
            "basis_float64_le_sha256",
            "basis_operator_norm_certificate",
            "reconstruction_map",
            "center_reconstruction_error",
            "captured_constraint_audit",
            "scaling",
            "fixture",
            "stage1",
            "old_stage2",
            "candidate_priority",
            "candidates",
            "certificate_backend_complete",
            "certificate_backend_errors",
            "fixture_and_reconstruction_survive_certificate_backend_errors",
            "formally_certified_candidates",
            "first_formally_certified_candidate",
            "formal_CW17_may_be_built",
            "diagnostic_does_not_materialize_or_consume_candidate",
        },
        "capture_checks_all_true": value.get("pass") is True
        and engine.all_true_checks(value.get("checks")),
        "dimensions_exact": dimension_valid,
        "singular_values_full": dimension_valid
        and all(
            cw17_encoded_array_checks(
                value.get("singular_values"), expected_shape=(92,)
            ).values()
        ),
        "basis_full_bytes_valid": dimension_valid
        and all(basis_array_checks.values()),
        "basis_hash_repeated_exact": isinstance(basis, Mapping)
        and value.get("basis_float64_le_sha256") == basis.get("sha256"),
        "reconstruction_formula_exact": isinstance(reconstruction, Mapping)
        and reconstruction.get("formula")
        == "candidate_raw=basis_reduced_by_actor.T@(coordinate_scale*candidate_scaled)"
        and reconstruction.get("basis_orientation")
        == "reduced_rank_by_actor_original"
        and reconstruction.get("coordinate_scale") == 0.001
        and reconstruction.get("actor_original_dimension") == actor
        and reconstruction.get("reduced_rank") == rank
        and reconstruction.get("externally_bound_by_diagnostic_stdout_sha256")
        is True,
        "basis_operator_norm_certificate_valid": all(operator_checks.values()),
        "fixture_full_external_identity_valid": all(fixture_checks.values()),
        "constraint_capture_all_true": isinstance(capture, Mapping)
        and capture.get("pass") is True
        and isinstance(capture_checks, Mapping)
        and engine.all_true_checks(capture_checks)
        and capture_checks.get("total_ball_intentionally_omitted_from_dual") is True
        and capture_checks.get("trust_ball_captured_into_dual_fixture") is True
        and capture.get("fixture_uses_captured_closure_jacobians_and_fun_at_zero")
        is True
        and capture.get("total_ball_omission_reason")
        == (
            "minimum_norm_relaxation_optimum_is_inside_total_ball_when_accepted_"
            "candidate_is_exact_total_feasible"
        )
        and all(
            cw17_encoded_array_checks(
                capture.get("captured_trust_center"), expected_shape=(rank,)
            ).values()
        )
        and type(fixture_rho) is float
        and capture.get("captured_trust_rho") == fixture_rho
        and capture.get("captured_trust_radius_squared") == fixture_rho**2,
        "scaling_exact": isinstance(scaling, Mapping)
        and scaling.get("coordinate_scale") == 0.001
        and type(scaling.get("trust_radius_raw")) is float
        and scaling["trust_radius_raw"] > 0.0
        and scaling.get("rho_scaled") == fixture_rho
        and scaling["trust_radius_raw"] / 0.001 == scaling["rho_scaled"]
        and type(scaling.get("z_scale")) is float
        and scaling["z_scale"] > 0.0
        and scaling.get("local_global_scale") == scaling.get("z_scale")
        and all(
            cw17_encoded_array_checks(
                scaling.get("anchor_row_scales"), expected_shape=(50,)
            ).values()
        ),
        "stage1_identity_exact": isinstance(stage1, Mapping)
        and stage1.get("success") is True
        and stage1.get("status") == 0
        and all(
            cw17_encoded_array_checks(
                stage1.get("x_scaled"), expected_shape=(rank,)
            ).values()
        )
        and type(stage1.get("z_certification_adjustment")) is float
        and 0.0 <= stage1["z_certification_adjustment"] <= 1e-8,
        "stage2_status9_exact": isinstance(stage2, Mapping)
        and stage2.get("success") is False
        and stage2.get("status") == CW16_STAGE2_STATUS
        and stage2.get("message") == CW16_STAGE2_MESSAGE
        and stage2.get("iterations") == 2000
        and type(stage2.get("function_evaluations")) is int
        and stage2["function_evaluations"] > 0
        and all(
            cw17_encoded_array_checks(
                stage2.get("x_scaled"), expected_shape=(rank,)
            ).values()
        )
        and all(multiplier_checks.values()),
        "all_candidate_records_valid": set(candidate_checks)
        == {
            "old_slsqp_stage2_last_x",
            "dual_recovered",
            "stage1_feasible_upper_bound",
        }
        and all(all(record.values()) for record in candidate_checks.values()),
        "candidate_identity_closed": all(candidate_identity_checks.values()),
        "formal_selection_exact": all(selection_checks.values()),
        "backend_complete_exact": isinstance(candidates, Mapping)
        and value.get("certificate_backend_complete")
        is all(
            candidates[name].get("certificate_available") is True
            for name in candidates
        ),
        "backend_errors_exact": expected_backend_errors is not None
        and value.get("certificate_backend_errors") == expected_backend_errors,
        "fixture_present_even_on_backend_error": isinstance(fixture, Mapping)
        and isinstance(reconstruction, Mapping)
        and value.get("fixture_and_reconstruction_survive_certificate_backend_errors")
        is True,
    }


def _mutation_paths() -> tuple[tuple[str, ...], ...]:
    return (
        ("candidate_priority",),
        ("formally_certified_candidates",),
        ("first_formally_certified_candidate",),
        ("formal_CW17_may_be_built",),
        ("diagnostic_does_not_materialize_or_consume_candidate",),
        ("fixture_and_reconstruction_survive_certificate_backend_errors",),
        ("candidates", "old_slsqp_stage2_last_x", "certificate", "formal_certificate_pass"),
        ("candidates", "old_slsqp_stage2_last_x", "original_space", "pass"),
        ("candidates", "old_slsqp_stage2_last_x", "mapped_raw_radius_certificate", "pass"),
    )


def nested_mutate(document: Mapping[str, Any], path: Sequence[str]) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    cursor: dict[str, Any] = result
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            raise ProtocolError(f"mutation fixture missing mapping at {path}")
        cursor = child
    key = path[-1]
    value = cursor.get(key)
    if isinstance(value, bool):
        cursor[key] = not value
    elif isinstance(value, int):
        cursor[key] = value + 1
    elif isinstance(value, str):
        cursor[key] = value + "__MUTATED"
    else:
        cursor[key] = "__MUTATED"
    return result


def cw17_terminal_validator_mutation_self_test(
) -> dict[str, Any]:
    """Mutation-test selection, multipliers, and math-canonical fixture hashes."""

    def candidate(formal: bool) -> dict[str, Any]:
        return {
            "certificate_available": True,
            "original_space_available": True,
            "certificate": {
                "formal_certificate_pass": formal,
                "exact_feasibility": {"total_squared_slack_sign": 1},
            },
            "mapped_raw_radius_certificate": {"pass": formal},
            "original_space": {"pass": formal},
        }

    def encoded(values: Sequence[float], shape: Sequence[int]) -> dict[str, Any]:
        raw = struct.pack(f"<{len(values)}d", *values)
        return {
            "dtype": "<f8",
            "shape": list(shape),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "base64": base64.b64encode(raw).decode("ascii"),
        }

    def stage2_record(
        *,
        multipliers: Mapping[str, Any] | None,
        warm_values: Sequence[float],
        shape_exact: bool,
        finite: bool,
        usable: bool,
    ) -> dict[str, Any]:
        return {
            "success": False,
            "status": 9,
            "message": "Iteration limit reached",
            "iterations": 2000,
            "function_evaluations": 2001,
            "x_scaled": encoded([0.0], [1]),
            "multipliers": multipliers,
            "multiplier_capture_audit": {
                "shape_exact": shape_exact,
                "finite": finite,
                "usable_as_warm_start": usable,
                "fallback_to_zero_if_unusable": True,
            },
            "warm_dual_without_redundant_total_ball": encoded(warm_values, [92]),
        }

    fixture = {
        "candidate_priority": [
            "old_slsqp_stage2_last_x",
            "dual_recovered",
            "stage1_feasible_upper_bound",
        ],
        "candidates": {
            "old_slsqp_stage2_last_x": candidate(True),
            "dual_recovered": candidate(False),
            "stage1_feasible_upper_bound": candidate(False),
        },
        "formally_certified_candidates": ["old_slsqp_stage2_last_x"],
        "first_formally_certified_candidate": "old_slsqp_stage2_last_x",
        "formal_CW17_may_be_built": True,
        "diagnostic_does_not_materialize_or_consume_candidate": True,
        "fixture_and_reconstruction_survive_certificate_backend_errors": True,
    }
    baseline_checks = cw17_formal_selection_checks(fixture)
    records: list[dict[str, Any]] = []
    for path in _mutation_paths():
        mutated = nested_mutate(fixture, path)
        observed = cw17_formal_selection_checks(mutated)
        records.append(
            {
                "path": list(path),
                "rejected": not all(observed.values()),
                "false_checks": sorted(
                    key for key, passed in observed.items() if passed is not True
                ),
            }
        )

    zero_warm = [0.0] * 92
    usable_multipliers = [float(index) / 100.0 for index in range(93)]
    multiplier_cases = {
        "empty": (
            stage2_record(
                multipliers=None,
                warm_values=zero_warm,
                shape_exact=False,
                finite=True,
                usable=False,
            ),
            0,
        ),
        "finite_wrong_shape": (
            stage2_record(
                multipliers=encoded([1.0, 2.0], [1, 2]),
                warm_values=zero_warm,
                shape_exact=False,
                finite=True,
                usable=False,
            ),
            2,
        ),
        "finite_scalar_shape": (
            stage2_record(
                multipliers=encoded([7.0], []),
                warm_values=zero_warm,
                shape_exact=False,
                finite=True,
                usable=False,
            ),
            1,
        ),
        "finite_3d_shape": (
            stage2_record(
                multipliers=encoded(
                    [float(index) for index in range(8)], [2, 2, 2]
                ),
                warm_values=zero_warm,
                shape_exact=False,
                finite=True,
                usable=False,
            ),
            8,
        ),
        "nonfinite_unserialized": (
            stage2_record(
                multipliers=None,
                warm_values=zero_warm,
                shape_exact=False,
                finite=False,
                usable=False,
            ),
            17,
        ),
        "nonfinite_shape93_unserialized": (
            stage2_record(
                multipliers=None,
                warm_values=zero_warm,
                shape_exact=True,
                finite=False,
                usable=False,
            ),
            93,
        ),
        "usable_93": (
            stage2_record(
                multipliers=encoded(usable_multipliers, [93]),
                warm_values=[
                    *usable_multipliers[:91],
                    2.0 * usable_multipliers[-1],
                ],
                shape_exact=True,
                finite=True,
                usable=True,
            ),
            93,
        ),
    }
    multiplier_case_checks = {
        name: cw17_multiplier_capture_checks(
            record,
            multiplier_count=count,
            reduced_rank=1,
        )
        for name, (record, count) in multiplier_cases.items()
    }
    multiplier_mutations: list[dict[str, Any]] = []
    for name, record, count in (
        (
            "usable_flag",
            nested_mutate(
                multiplier_cases["usable_93"][0],
                ("multiplier_capture_audit", "usable_as_warm_start"),
            ),
            93,
        ),
        (
            "fallback_flag",
            nested_mutate(
                multiplier_cases["finite_wrong_shape"][0],
                ("multiplier_capture_audit", "fallback_to_zero_if_unusable"),
            ),
            2,
        ),
        (
            "shape_flag",
            nested_mutate(
                multiplier_cases["finite_wrong_shape"][0],
                ("multiplier_capture_audit", "shape_exact"),
            ),
            2,
        ),
        (
            "finite_flag",
            nested_mutate(
                multiplier_cases["nonfinite_unserialized"][0],
                ("multiplier_capture_audit", "finite"),
            ),
            17,
        ),
        (
            "usable_payload_missing",
            nested_mutate(
                multiplier_cases["usable_93"][0],
                ("multipliers",),
            ),
            93,
        ),
        (
            "count_mismatch",
            multiplier_cases["usable_93"][0],
            94,
        ),
    ):
        observed = cw17_multiplier_capture_checks(
            record,
            multiplier_count=count,
            reduced_rank=1,
        )
        multiplier_mutations.append(
            {
                "name": name,
                "rejected": not all(observed.values()),
                "false_checks": sorted(
                    key for key, passed in observed.items() if passed is not True
                ),
            }
        )

    identity_point = encoded([0.0], [1])

    def identity_candidate() -> dict[str, Any]:
        return {
            "candidate_available": True,
            "candidate_scaled": copy.deepcopy(identity_point),
            "certificate_available": True,
            "certificate": {
                "primal": copy.deepcopy(identity_point),
                "raw_optimal_radius_bound": 0.0,
                "checks": {"raw_optimal_radius_at_most_1e_9": True},
                "dual": {
                    "selected": {"u_dual": copy.deepcopy(identity_point)}
                },
            },
            "mapped_raw_radius_certificate": {
                "checks": {
                    "basis_operator_norm_certificate_pass": True,
                    "mapped_raw_optimal_radius_at_most_1e_9": True,
                    "reduced_certificate_raw_radius_gate_pass": True,
                },
                "basis_operator_norm_upper": 1.0,
                "reduced_radius_bound": 0.0,
                "mapped_raw_radius_bound": 0.0,
                "multiplication_rounding": "float64_product_then_nextafter_up",
                "pass": True,
            },
            "original_space_available": True,
            "original_space": {
                "candidate_scaled": copy.deepcopy(identity_point),
                "pass": True,
            },
        }

    identity_fixture = {
        "candidates": {
            "old_slsqp_stage2_last_x": identity_candidate(),
            "dual_recovered": identity_candidate(),
            "stage1_feasible_upper_bound": identity_candidate(),
        },
        "stage1": {"x_scaled": copy.deepcopy(identity_point)},
        "stage2": {"x_scaled": copy.deepcopy(identity_point)},
        "basis": {"operator_norm_upper": 1.0, "pass": True},
    }
    identity_baseline = cw17_candidate_identity_checks(
        identity_fixture["candidates"],
        stage1=identity_fixture["stage1"],
        stage2=identity_fixture["stage2"],
        basis_operator_norm_certificate=identity_fixture["basis"],
    )
    identity_mutations: list[dict[str, Any]] = []
    for name, path in (
        (
            "old_vs_stage2",
            ("candidates", "old_slsqp_stage2_last_x", "candidate_scaled"),
        ),
        (
            "stage1_vs_certificate",
            (
                "candidates",
                "stage1_feasible_upper_bound",
                "certificate",
                "primal",
            ),
        ),
        (
            "dual_recovered_vs_selected",
            ("candidates", "dual_recovered", "candidate_scaled"),
        ),
        (
            "mapped_basis_upper",
            (
                "candidates",
                "old_slsqp_stage2_last_x",
                "mapped_raw_radius_certificate",
                "basis_operator_norm_upper",
            ),
        ),
        (
            "mapped_bound",
            (
                "candidates",
                "old_slsqp_stage2_last_x",
                "mapped_raw_radius_certificate",
                "mapped_raw_radius_bound",
            ),
        ),
        (
            "mapped_check",
            (
                "candidates",
                "old_slsqp_stage2_last_x",
                "mapped_raw_radius_certificate",
                "checks",
                "mapped_raw_optimal_radius_at_most_1e_9",
            ),
        ),
    ):
        mutated = nested_mutate(identity_fixture, path)
        observed = cw17_candidate_identity_checks(
            mutated["candidates"],
            stage1=mutated["stage1"],
            stage2=mutated["stage2"],
            basis_operator_norm_certificate=mutated["basis"],
        )
        identity_mutations.append(
            {
                "name": name,
                "rejected": not all(observed.values()),
                "false_checks": sorted(
                    key for key, passed in observed.items() if passed is not True
                ),
            }
        )

    fixture_identity = {
        "schema_version": "ptcg-cw17-stage2-dual-math-v1",
        "coordinate_scale": 0.001,
        "rho": 0.125,
        "row_count": 91,
        "reduced_dimension": 1,
        "arrays": {
            "K": encoded([0.0] * 91, [91, 1]),
            "q": encoded([0.0] * 91, [91]),
            "center": encoded([0.0], [1]),
        },
    }
    fixture_sha = sha256_bytes(cw17_math_canonical_json(fixture_identity))
    math_fixture = {**fixture_identity, "fixture_sha256": fixture_sha}
    fixture_checks = cw17_fixture_contract_checks(math_fixture, reduced_rank=1)
    newline_fixture = copy.deepcopy(math_fixture)
    newline_fixture["fixture_sha256"] = sha256_bytes(
        cw17_math_canonical_json(fixture_identity) + b"\n"
    )
    newline_checks = cw17_fixture_contract_checks(
        newline_fixture, reduced_rank=1
    )
    checks = {
        "baseline_all_true": all(baseline_checks.values()),
        "mutation_count_exact": len(records) == len(_mutation_paths()),
        "mutation_paths_unique": len({tuple(row["path"]) for row in records})
        == len(records),
        "every_mutation_rejected": all(row["rejected"] for row in records),
        "seven_multiplier_fallback_cases_pass": set(multiplier_case_checks)
        == {
            "empty",
            "finite_wrong_shape",
            "finite_scalar_shape",
            "finite_3d_shape",
            "nonfinite_unserialized",
            "nonfinite_shape93_unserialized",
            "usable_93",
        }
        and all(
            all(case.values()) for case in multiplier_case_checks.values()
        ),
        "all_multiplier_mutations_rejected": len(multiplier_mutations) == 6
        and all(row["rejected"] for row in multiplier_mutations),
        "candidate_identity_baseline_pass": all(identity_baseline.values()),
        "all_candidate_identity_mutations_rejected": len(identity_mutations) == 6
        and all(row["rejected"] for row in identity_mutations),
        "math_fixture_roundtrip_exact": all(fixture_checks.values()),
        "newline_hash_negative_rejected": fixture_sha
        != newline_fixture["fixture_sha256"]
        and newline_checks.get("aggregate_fixture_sha256_exact") is False,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "records": records,
        "multiplier_case_checks": multiplier_case_checks,
        "multiplier_mutations": multiplier_mutations,
        "candidate_identity_baseline": identity_baseline,
        "candidate_identity_mutations": identity_mutations,
        "fixture_checks": fixture_checks,
        "newline_fixture_false_checks": sorted(
            key for key, passed in newline_checks.items() if passed is not True
        ),
    }


def cw17_parent_manifest_adapter(value: Any) -> dict[str, Any]:
    """Expose the heterogeneous old40+closure3 records to the frozen CW12 core."""

    if not isinstance(value, Mapping) or not isinstance(value.get("records"), list):
        raise ProtocolError("CW17 manifest cannot be adapted for inherited rehash")
    records: list[dict[str, Any]] = []
    for ordinal, record in enumerate(value["records"]):
        if not isinstance(record, Mapping):
            raise ProtocolError(f"CW17 manifest record {ordinal} is not a mapping")
        mode = record.get("mode", record.get("mode_octal"))
        regular = record.get("regular")
        if regular is None and isinstance(record.get("checks"), Mapping):
            regular = record["checks"].get("regular")
        normalized = {
            "path": record.get("path"),
            "sha256": record.get("sha256"),
            "mode": mode,
            "expected_mode": mode,
            "nlink": record.get("nlink"),
            "regular": regular,
        }
        if (
            not isinstance(normalized["path"], str)
            or not isinstance(normalized["sha256"], str)
            or len(normalized["sha256"]) != 64
            or not isinstance(mode, str)
            or len(mode) != 4
            or normalized["nlink"] != 1
            or normalized["regular"] is not True
        ):
            raise ProtocolError(f"CW17 manifest record {ordinal} identity drift")
        records.append(normalized)
    if len(records) != 43 or len({record["path"] for record in records}) != 43:
        raise ProtocolError("CW17 adapted manifest is not exactly 43 unique paths")
    return {
        "all_exact": value.get("all_exact"),
        "binding_count": value.get("binding_count"),
        "records": records,
    }


def cw17_validate_static_payload(payload: bytes) -> dict[str, Any]:
    """Validate the final zero-CUDA 43-lock CW17 static payload."""

    if FINAL_SOLVER_CONTRACT_PENDING:
        raise ProtocolError("CW17 static validator is pending final solver contract")
    document = engine.strict_json_object(payload, "CW17 solver static stdout")
    runtime = document.get("runtime")
    self_source = document.get("self_source")
    parent_source = document.get("CW16_solver_source")
    math_source = document.get("dual_math_source")
    frozen = document.get("frozen_inputs_43")
    closure = document.get("cw16_closure_evidence")
    math_selftest = document.get("dual_math_selftest")
    math_contract = document.get("dual_math_contract")
    basis_selftest = document.get("basis_operator_norm_selftest")
    recovery_selftest = document.get("certificate_failure_recovery_selftest")
    mutation_result = cw17_terminal_validator_mutation_self_test()
    source_audit = document.get("source_audit")
    checks = {
        "top_level_keys_exact": set(document) == set(EXPECTED_STATIC_TOP_KEYS)
        and len(document) == len(EXPECTED_STATIC_TOP_KEYS),
        "schema_exact": document.get("schema_version") == SOLVER_SCHEMA,
        "status_exact": document.get("status") == EXPECTED_STATIC_STATUS,
        "classification_exact": document.get("classification") == CLASSIFICATION,
        "contract_exact": document.get("contract") == EXPECTED_SOLVER_CONTRACT,
        "solver_checks_all_true": engine.all_true_checks(document.get("checks")),
        "runtime_all_true_no_cuda": isinstance(runtime, Mapping)
        and engine.all_true_checks(runtime.get("checks"))
        and runtime.get("cuda_required") is False,
        "self_source_exact_frozen": isinstance(self_source, Mapping)
        and self_source.get("path") == root_relative(SOLVER)
        and self_source.get("sha256") == EXPECTED_SOLVER_SHA256
        and self_source.get("mode_octal") == "0555"
        and self_source.get("nlink") == 1,
        "parent_source_separate_exact": isinstance(parent_source, Mapping)
        and parent_source.get("path") == root_relative(CW16_SOLVER)
        and parent_source.get("sha256") == CW16_SOLVER_SHA256
        and parent_source.get("mode_octal") == "0555"
        and parent_source.get("nlink") == 1,
        "dual_math_source_exact": isinstance(math_source, Mapping)
        and math_source.get("path") == "tools/cw17_stage2_dual_math_v1.py"
        and math_source.get("sha256") == EXPECTED_DUAL_MATH_SHA256
        and math_source.get("mode_octal") == "0555"
        and math_source.get("nlink") == 1,
        "frozen_inputs_exact43": isinstance(frozen, Mapping)
        and frozen.get("all_exact") is True
        and frozen.get("binding_count") == EXPECTED_SOLVER_FROZEN_BINDINGS
        and isinstance(frozen.get("records"), list)
        and len(frozen["records"]) == EXPECTED_SOLVER_FROZEN_BINDINGS,
        "closure_evidence_all_true": isinstance(closure, Mapping)
        and closure.get("pass") is True
        and engine.all_true_checks(closure.get("checks")),
        "dual_math_selftest_all_true": isinstance(math_selftest, Mapping)
        and math_selftest.get("pass") is True
        and math_selftest.get("writes_performed") is False
        and math_selftest.get("cuda_accessed") is False
        and engine.all_true_checks(math_selftest.get("checks")),
        "dual_math_contract_all_true": isinstance(math_contract, Mapping)
        and math_contract.get("pass") is True
        and engine.all_true_checks(math_contract.get("checks")),
        "basis_selftests_all_true": isinstance(basis_selftest, Mapping)
        and set(basis_selftest) == {"identity", "nonorthogonal"}
        and all(
            isinstance(record, Mapping)
            and record.get("pass") is True
            and engine.all_true_checks(record.get("checks"))
            for record in basis_selftest.values()
        ),
        "certificate_failure_recovery_all_true": isinstance(
            recovery_selftest, Mapping
        )
        and recovery_selftest.get("pass") is True
        and recovery_selftest.get("error_count") == 3
        and engine.all_true_checks(recovery_selftest.get("checks")),
        "synthetic_mutations_all_rejected": mutation_result.get("pass") is True,
        "source_audit_all_true": isinstance(source_audit, Mapping)
        and source_audit.get("pass") is True
        and source_audit.get("source_sha256") == EXPECTED_SOLVER_SHA256
        and engine.all_true_checks(source_audit.get("checks")),
        "run_executed_false": document.get("run_executed") is False,
        "cuda_accessed_false": document.get("cuda_accessed") is False,
        "writes_performed_false": document.get("writes_performed") is False,
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW17 static payload contract failed: {checks}")
    parent_payload = copy.deepcopy(document)
    parent_payload["frozen_inputs"] = cw17_parent_manifest_adapter(frozen)
    return {
        "checks": checks,
        "mutation_self_test": mutation_result,
        "stdout_sha256": sha256_bytes(payload),
        "stdout_bytes": len(payload),
        "payload": parent_payload,
        "lossless_payload": document,
    }


def cw17_validate_terminal_payload(payload: bytes) -> dict[str, Any]:
    """Accept only diagnostic complete/closed payloads with zero new model evals."""

    if FINAL_SOLVER_CONTRACT_PENDING:
        raise ProtocolError("CW17 terminal validator is pending final solver contract")
    document = engine.strict_json_object(payload, "CW17 diagnostic terminal stdout")
    frozen = document.get("frozen_inputs_43")
    diagnostic = document.get("diagnostic")
    input_lock = document.get("input_lock")
    closure = document.get("cw16_closure_evidence")
    replay = document.get("replay_identity")
    runtime = document.get("runtime")
    math_selftest = document.get("dual_math_selftest")
    math_contract = document.get("dual_math_contract")
    source_audit = document.get("source_audit")
    diagnostic_checks = cw17_diagnostic_contract_checks(diagnostic)
    self_lock = input_lock.get("self") if isinstance(input_lock, Mapping) else None
    cw16_lock = (
        input_lock.get("CW16_solver_source") if isinstance(input_lock, Mapping) else None
    )
    math_lock = (
        input_lock.get("dual_math_source") if isinstance(input_lock, Mapping) else None
    )
    closure_lock = (
        input_lock.get("CW16_closure") if isinstance(input_lock, Mapping) else None
    )
    common_checks = {
        "top_level_keys_exact": set(document) == set(EXPECTED_TERMINAL_TOP_KEYS)
        and len(document) == len(EXPECTED_TERMINAL_TOP_KEYS),
        "schema_exact": document.get("schema_version") == SOLVER_SCHEMA,
        "status_diagnostic_complete_or_closed": document.get("status")
        in EXPECTED_TERMINAL_STATUSES,
        "classification_exact": document.get("classification") == CLASSIFICATION,
        "contract_exact": document.get("contract") == EXPECTED_SOLVER_CONTRACT,
        "input_lock_keys_exact": isinstance(input_lock, Mapping)
        and set(input_lock)
        == {"self", "CW16_solver_source", "dual_math_source", "CW16_closure"},
        "input_lock_sources_exact": isinstance(self_lock, Mapping)
        and self_lock.get("path") == root_relative(SOLVER)
        and self_lock.get("sha256") == EXPECTED_SOLVER_SHA256
        and self_lock.get("mode_octal") == "0555"
        and self_lock.get("nlink") == 1
        and isinstance(cw16_lock, Mapping)
        and cw16_lock.get("path") == root_relative(CW16_SOLVER)
        and cw16_lock.get("sha256") == CW16_SOLVER_SHA256
        and cw16_lock.get("mode_octal") == "0555"
        and isinstance(math_lock, Mapping)
        and math_lock.get("path") == "tools/cw17_stage2_dual_math_v1.py"
        and math_lock.get("sha256") == EXPECTED_DUAL_MATH_SHA256
        and math_lock.get("mode_octal") == "0555",
        "closure_lock_exact_three": isinstance(closure_lock, Mapping)
        and set(closure_lock) == {"attempt", "stdout", "stderr_audit"}
        and closure_lock.get("attempt", {}).get("sha256")
        == CW16_ATTEMPT_MARKER_SHA256
        and closure_lock.get("stdout", {}).get("sha256")
        == CW16_CLOSED_STDOUT_SHA256
        and closure_lock.get("stderr_audit", {}).get("sha256")
        == CW16_STDERR_AUDIT_SHA256,
        "frozen_inputs_exact43": isinstance(frozen, Mapping)
        and frozen.get("all_exact") is True
        and frozen.get("binding_count") == EXPECTED_SOLVER_FROZEN_BINDINGS
        and isinstance(frozen.get("records"), list)
        and len(frozen["records"]) == EXPECTED_SOLVER_FROZEN_BINDINGS,
        "diagnostic_contract_all_true": bool(diagnostic_checks)
        and all(diagnostic_checks.values()),
        "closure_evidence_all_true": isinstance(closure, Mapping)
        and closure.get("pass") is True
        and engine.all_true_checks(closure.get("checks")),
        "replay_identity_exact": isinstance(replay, Mapping)
        and replay.get("pass") is True
        and engine.all_true_checks(replay.get("checks"))
        and replay.get("canonical_stdout_sha256") == CW16_CLOSED_STDOUT_SHA256
        and replay.get("canonical_stdout_bytes") == CW16_CLOSED_STDOUT_BYTES,
        "dual_math_selftest_all_true": isinstance(math_selftest, Mapping)
        and math_selftest.get("pass") is True
        and math_selftest.get("writes_performed") is False
        and math_selftest.get("cuda_accessed") is False
        and engine.all_true_checks(math_selftest.get("checks")),
        "dual_math_contract_all_true": isinstance(math_contract, Mapping)
        and math_contract.get("pass") is True
        and engine.all_true_checks(math_contract.get("checks")),
        "runtime_run_exact": isinstance(runtime, Mapping)
        and runtime.get("cuda_required") is True
        and engine.all_true_checks(runtime.get("checks")),
        "source_audit_all_true": isinstance(source_audit, Mapping)
        and source_audit.get("pass") is True
        and source_audit.get("source_sha256") == EXPECTED_SOLVER_SHA256
        and engine.all_true_checks(source_audit.get("checks")),
        "run_executed_true": document.get("run_executed") is True,
        "cuda_accessed_true": document.get("cuda_accessed") is True,
        "writes_performed_false": document.get("writes_performed") is False,
    }
    if not all(common_checks.values()):
        raise ProtocolError(f"CW17 terminal diagnostic contract failed: {common_checks}")
    reloaded = engine.strict_json_object(payload, "CW17 terminal lossless reparse")
    if reloaded != document:
        raise ProtocolError("CW17 terminal payload changed during validation")
    return {
        "checks": common_checks,
        "diagnostic_checks": diagnostic_checks,
        "fixture_sha256": diagnostic.get("fixture", {}).get("fixture_sha256"),
        "stdout_sha256": sha256_bytes(payload),
        "stdout_bytes": len(payload),
        "payload": document,
    }


def wrapper_source_audit(source: bytes) -> dict[str, Any]:
    """Audit locks, AST bodies, one-shot inheritance, and forbidden surfaces."""

    text = source.decode("utf-8")
    tree = ast.parse(text, filename=str(SCRIPT))
    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    top_level_literals: dict[str, Any] = {}
    imported_roots: set[str] = set()
    output_arguments: list[str] = []
    calls: set[str] = set()
    for node in tree.body:
        names: list[str] = []
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
            value_node = node.value
        if value_node is not None:
            try:
                literal = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                literal = None
            else:
                for name in names:
                    top_level_literals[name] = literal
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
                and node.args[0].value != "--mode"
            ):
                output_arguments.append(node.args[0].value)
    observed_launcher_hashes = {
        name: sha256_bytes(
            ast.dump(function_nodes[name], include_attributes=False).encode("utf-8")
        )
        for name in CW17_CRITICAL_FUNCTION_AST_SHA256
        if name in function_nodes
    }
    solver_ast = cw17_solver_ast_audit()
    forbidden_imports = {"kaggle", "requests", "socket", "torch", "urllib"}
    checks = {
        "final_solver_contract_installed": FINAL_SOLVER_CONTRACT_PENDING is False
        and top_level_literals.get("FINAL_SOLVER_CONTRACT_PENDING") is False,
        "parent_held_bytes_exact": PARENT_EVIDENCE.get("held_bytes_executed") is True
        and PARENT_EVIDENCE.get("sha256") == PARENT_LAUNCHER_SHA256,
        "parent_source_audit_pass": PARENT_SOURCE_AUDIT.get("pass") is True,
        "parent_composed_checks_all_true": all(PARENT_COMPOSED_CHECKS.values()),
        "parent_import_from_held_bytes": "compile" in calls
        and "exec" in calls
        and "spec_from_file_location" not in calls,
        "base_preflight_exact": _BASE_PREFLIGHT is parent_adapter._BASE_PREFLIGHT,
        "no_forbidden_imports": not imported_roots.intersection(forbidden_imports),
        "no_direct_subprocess_call": "run" not in calls,
        "cli_only_mode": not output_arguments,
        "paths_exact": SCRIPT.name
        == "run_cw16_stage2_status9_cw17_diagnostic_one_shot_v1.py"
        and SOLVER.name == "probe_cw16_stage2_status9_cw17_diagnostic_v1.py",
        "artifact_id_exact": ARTIFACT_ID
        == "cw17_stage2_status9_diagnostic_20260803_v1",
        "parent_lock_literals_exact": all(
            top_level_literals.get(name) == expected
            for name, expected in {
                "PARENT_LAUNCHER_SHA256": PARENT_LAUNCHER_SHA256,
                "CW16_SOLVER_SHA256": CW16_SOLVER_SHA256,
                "CW16_ATTEMPT_MARKER_SHA256": CW16_ATTEMPT_MARKER_SHA256,
                "CW16_CLOSED_STDOUT_SHA256": CW16_CLOSED_STDOUT_SHA256,
                "CW16_STDERR_AUDIT_SHA256": CW16_STDERR_AUDIT_SHA256,
            }.items()
        ),
        "schema_literals_exact": all(
            top_level_literals.get(name) == expected
            for name, expected in {
                "SCHEMA": SCHEMA,
                "ATTEMPT_SCHEMA": ATTEMPT_SCHEMA,
                "FAILURE_SCHEMA": FAILURE_SCHEMA,
                "STDERR_SCHEMA": STDERR_SCHEMA,
                "SOLVER_SCHEMA": SOLVER_SCHEMA,
                "ARTIFACT_ID": ARTIFACT_ID,
                "EXPECTED_STATIC_STATUS": EXPECTED_STATIC_STATUS,
                "EXPECTED_TERMINAL_STATUSES": EXPECTED_TERMINAL_STATUSES,
                "EXPECTED_STATIC_TOP_KEYS": EXPECTED_STATIC_TOP_KEYS,
                "EXPECTED_TERMINAL_TOP_KEYS": EXPECTED_TERMINAL_TOP_KEYS,
                "EXPECTED_SOLVER_CRITICAL_FUNCTIONS": (
                    EXPECTED_SOLVER_CRITICAL_FUNCTIONS
                ),
                "EXPECTED_LAUNCHER_CRITICAL_FUNCTIONS": (
                    EXPECTED_LAUNCHER_CRITICAL_FUNCTIONS
                ),
                "CLASSIFICATION": CLASSIFICATION,
            }.items()
        ),
        "terminal_statuses_diagnostic_only": bool(EXPECTED_TERMINAL_STATUSES)
        and all(
            isinstance(status, str)
            and ("diagnostic" in status.lower())
            and "candidate" not in status.lower()
            and "promotion" not in status.lower()
            for status in EXPECTED_TERMINAL_STATUSES
        ),
        "targets_exact": ATTEMPT_MARKER.name
        == ".ptcg-cw17_stage2_status9_diagnostic_20260803_v1-attempt.json"
        and STDOUT_OUTPUT.name
        == "cw17_stage2_status9_diagnostic_20260803_v1.stdout.json"
        and STDERR_AUDIT.name
        == "cw17_stage2_status9_diagnostic_20260803_v1.stderr-audit.json",
        "targets_distinct_under_artifacts": len(
            {ATTEMPT_MARKER, STDOUT_OUTPUT, STDERR_AUDIT}
        )
        == 3
        and ATTEMPT_MARKER.parent
        == STDOUT_OUTPUT.parent
        == STDERR_AUDIT.parent
        == ARTIFACTS,
        "solver_sha_literal_exact": top_level_literals.get(
            "EXPECTED_SOLVER_SHA256"
        )
        == EXPECTED_SOLVER_SHA256
        and isinstance(EXPECTED_SOLVER_SHA256, str)
        and len(EXPECTED_SOLVER_SHA256) == 64,
        "dual_math_sha_literal_exact": top_level_literals.get(
            "EXPECTED_DUAL_MATH_SHA256"
        )
        == EXPECTED_DUAL_MATH_SHA256
        and isinstance(EXPECTED_DUAL_MATH_SHA256, str)
        and len(EXPECTED_DUAL_MATH_SHA256) == 64,
        "solver_contract_runtime_exact": EXPECTED_SOLVER_CONTRACT[
            "dual_math_source_sha256"
        ]
        == EXPECTED_DUAL_MATH_SHA256
        and EXPECTED_SOLVER_CONTRACT["frozen_input_count"] == 43
        and EXPECTED_SOLVER_CONTRACT["unique_official_model_count_delta"] == 0
        and EXPECTED_SOLVER_CONTRACT["changed_model_official_evaluation"] is False
        and EXPECTED_SOLVER_CONTRACT["candidate_consumer"] is False
        and EXPECTED_SOLVER_CONTRACT["checkpoint_materialization"] is False
        and EXPECTED_SOLVER_CONTRACT["network_broad_gold_submission"] is False,
        "manifest_count_literal_exact43": top_level_literals.get(
            "EXPECTED_SOLVER_FROZEN_BINDINGS"
        )
        == EXPECTED_SOLVER_FROZEN_BINDINGS
        == 43,
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
        "launcher_critical_set_exact": set(CW17_CRITICAL_FUNCTION_AST_SHA256)
        == set(EXPECTED_LAUNCHER_CRITICAL_FUNCTIONS),
        "launcher_critical_hashes_exact": observed_launcher_hashes
        == CW17_CRITICAL_FUNCTION_AST_SHA256,
        "solver_critical_set_exact": set(CW17_SOLVER_CRITICAL_FUNCTION_AST_SHA256)
        == set(EXPECTED_SOLVER_CRITICAL_FUNCTIONS),
        "solver_ast_audit_pass": solver_ast.get("pass") is True
        and engine.all_true_checks(solver_ast.get("checks")),
        "validators_installed": engine.validate_static_payload
        is cw17_validate_static_payload
        and engine.validate_terminal_payload is cw17_validate_terminal_payload,
        "preflight_installed": engine.preflight is cw17_preflight,
        "stderr_audit_installed": engine.validated_stderr_audit
        is cw17_stderr_audit,
        "mutation_test_declared": "cw17_terminal_validator_mutation_self_test"
        in function_nodes,
        "inherited_one_shot_paths_called": "static_result" in calls
        and "run_once" in calls,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "parent_launcher": PARENT_EVIDENCE,
        "parent_source_audit": PARENT_SOURCE_AUDIT,
        "parent_composed_checks": PARENT_COMPOSED_CHECKS,
        "solver_critical_ast_audit": solver_ast,
        "imported_roots": sorted(imported_roots),
        "unexpected_cli_arguments": output_arguments,
    }


engine.static_source_audit = wrapper_source_audit
engine.validate_static_payload = cw17_validate_static_payload
engine.validate_terminal_payload = cw17_validate_terminal_payload


def cw17_preflight(*, require_lock: bool) -> dict[str, Any]:
    """Run CPU source/static/43-lock gates before the inherited marker write."""

    evidence = _BASE_PREFLIGHT(require_lock=require_lock)
    evidence["inherited_parent_launcher"] = PARENT_EVIDENCE
    evidence["inherited_parent_composed_checks"] = PARENT_COMPOSED_CHECKS
    terminal_evidence = known_cw16_terminal_evidence()
    evidence["known_cw16_terminal_evidence"] = terminal_evidence
    if evidence.get("solver_lock_armed"):
        frozen = evidence.get("solver_static", {}).get("payload", {}).get(
            "frozen_inputs_43"
        )
        raw_records = frozen.get("records") if isinstance(frozen, Mapping) else None
        if not isinstance(raw_records, list):
            raise ProtocolError("armed CW17 solver did not emit its frozen manifest")
        adapted = cw17_parent_manifest_adapter(frozen)
        records = adapted["records"]
        observed = {
            record.get("path"): record
            for record in records
            if isinstance(record, Mapping)
        }
        prior_records = terminal_evidence["prior_frozen_records"]
        expected = {
            record["path"]: {
                "sha256": record["sha256"],
                "mode": record["mode"],
                "expected_mode": record["mode"],
                "nlink": 1,
                "regular": True,
            }
            for record in prior_records
        }
        expected.update(
            {
                root_relative(CW16_ATTEMPT_MARKER): {
                    "sha256": CW16_ATTEMPT_MARKER_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
                root_relative(CW16_CLOSED_STDOUT): {
                    "sha256": CW16_CLOSED_STDOUT_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
                root_relative(CW16_STDERR_AUDIT): {
                    "sha256": CW16_STDERR_AUDIT_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
            }
        )
        checks = {
            "solver_manifest_count_exact43": frozen.get("binding_count") == 43
            and len(records) == 43
            and len(expected) == 43,
            "independent_rehash_exact43": evidence.get(
                "independent_frozen_input_rehash", {}
            ).get("pass")
            is True
            and evidence.get("independent_frozen_input_rehash", {}).get(
                "binding_count"
            )
            == 43,
            "old40_plus_cw16_three_path_set_exact": set(observed) == set(expected),
            "all_43_record_identities_exact": all(
                path in observed
                and all(
                    observed[path].get(key) == value
                    for key, value in identity.items()
                )
                for path, identity in expected.items()
            ),
            "source_identity_not_in_data_manifest": root_relative(SOLVER)
            not in observed
            and root_relative(PARENT_LAUNCHER) not in observed,
        }
        if not all(checks.values()):
            raise ProtocolError(f"CW17 exact old40+CW16-three manifest gate failed: {checks}")
        evidence["cw17_manifest_old40_plus_cw16_three"] = {
            "checks": checks,
            "pass": True,
            "CW16_records": {
                path: observed[path]
                for path in sorted(
                    {
                        root_relative(CW16_ATTEMPT_MARKER),
                        root_relative(CW16_CLOSED_STDOUT),
                        root_relative(CW16_STDERR_AUDIT),
                    }
                )
            },
            "source_identity_separate": {
                "solver": evidence.get("solver"),
                "parent_launcher": PARENT_EVIDENCE,
            },
        }
        _, solver_after_evidence = engine.read_regular_stable(
            SOLVER,
            "CW17 solver immediately before attempt-marker handoff",
            expected_sha256=EXPECTED_SOLVER_SHA256,
            expected_mode=FROZEN_EXECUTABLE_MODE,
        )
        if solver_after_evidence != evidence.get("solver"):
            raise ProtocolError("CW17 solver identity changed during CPU preflight")
        final_absence = {
            "attempt_marker": engine.absent_by_lstat(ATTEMPT_MARKER),
            "stdout_output": engine.absent_by_lstat(STDOUT_OUTPUT),
            "stderr_audit": engine.absent_by_lstat(STDERR_AUDIT),
        }
        if require_lock and not all(final_absence.values()):
            raise FileExistsError(
                "CW17 one-shot target appeared before marker handoff: "
                f"{final_absence}"
            )
        evidence["solver_after_all_CPU_preflight"] = solver_after_evidence
        evidence["targets_absent_after_all_CPU_preflight_by_lstat"] = final_absence
    return evidence


def cw17_stderr_audit(
    payload: bytes,
    *,
    post_child_integrity: Mapping[str, Any],
) -> bytes:
    """Retain exact child stderr while rebasing only the wrapper schema chain."""

    inherited = _PARENT_STDERR_AUDIT(
        payload,
        post_child_integrity=post_child_integrity,
    )
    document = engine.strict_json_object(inherited, "inherited CW16 stderr audit")
    if document.get("schema_version") != (
        "ptcg-cw16-consumed-valid-official6-one-shot-stderr-audit-v1"
    ):
        raise ProtocolError("inherited CW16 stderr audit schema drift")
    prior_parent = document.get("inherited_parent_launcher")
    prior_grandparent = document.get("inherited_grandparent_launcher")
    prior_greatgrandparent = document.get("inherited_greatgrandparent_launcher")
    try:
        decoded = base64.b64decode(
            str(document.get("base64", "")).encode("ascii"), validate=True
        )
    except ValueError as error:
        raise ProtocolError("inherited stderr base64 is not canonical") from error
    checks = {
        "payload_roundtrip_exact": decoded == payload,
        "bytes_exact": document.get("bytes") == len(payload),
        "sha256_exact": document.get("sha256") == sha256_bytes(payload),
        "stderr_checks_all_true": engine.all_true_checks(document.get("checks")),
        "post_child_integrity_exact": document.get("post_child_integrity")
        == dict(post_child_integrity),
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW17 inherited stderr lossless gate failed: {checks}")
    document["schema_version"] = STDERR_SCHEMA
    document["CW17_lossless_checks"] = checks
    document["inherited_greatgreatgrandparent_launcher"] = prior_greatgrandparent
    document["inherited_greatgrandparent_launcher"] = prior_grandparent
    document["inherited_grandparent_launcher"] = prior_parent
    document["inherited_parent_launcher"] = PARENT_EVIDENCE
    return engine.canonical_json(document)


engine.preflight = cw17_preflight
engine.validated_stderr_audit = cw17_stderr_audit


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

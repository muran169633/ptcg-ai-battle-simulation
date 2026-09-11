#!/usr/bin/env python3
"""Consume exactly one CW16 official-six local-trust attempt.

CW16 hash-binds the final CW15 one-shot launcher and executes its exact held
bytes.  This adapter expands the frozen solver manifest from the prior 37
records to those same 37 plus the three immutable CW15 closure artifacts,
independently validates that closure, and installs CW16-specific static and
terminal gates for the unique ineligible bootstrap, current-local physical
rows, strict expanded-ledger Phi merit, four shrinking trust buckets, and the
exact imported ten-check terminal acceptance.

The file remains a writable, non-runnable placeholder until the independently
audited final solver SHA and critical function hashes are installed and this
launcher itself is frozen mode 0555.
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
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
ARTIFACTS = ROOT / "artifacts"
SCRIPT = TOOLS / (
    "run_cw16_consumed_valid_official6_local_trust_one_shot_v1.py"
)
SOLVER = TOOLS / (
    "probe_u468_cw11_consumed_valid_official6_cw16_local_trust_v1.py"
)
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_EXECUTABLE_MODE = 0o555

PARENT_LAUNCHER = TOOLS / (
    "run_cw15_consumed_valid_official6_sequential_tangent_one_shot_v1.py"
)
PARENT_LAUNCHER_SHA256 = (
    "88f813644c132ffd3fdb3109d962d51826a8835705ac579c713021e4c0c508e8"
)
CW15_SOLVER_SHA256 = (
    "2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24"
)

CW15_ATTEMPT_MARKER = ARTIFACTS / (
    ".ptcg-cw15_consumed_valid_official6_sequential_tangent_20260802_v1-"
    "attempt.json"
)
CW15_ATTEMPT_MARKER_SHA256 = (
    "e395acf51486cceb68e7f9524b716e1172a644623d097e32053a68ea999d1967"
)
CW15_CLOSED_STDOUT = ARTIFACTS / (
    "cw15_consumed_valid_official6_sequential_tangent_20260802_v1.stdout.json"
)
CW15_CLOSED_STDOUT_SHA256 = (
    "6e7f7bd58e0cf4465f9be3dfba4222d328ee8520d1050d99a008c4c543117b54"
)
CW15_STDERR_AUDIT = ARTIFACTS / (
    "cw15_consumed_valid_official6_sequential_tangent_20260802_v1."
    "stderr-audit.json"
)
CW15_STDERR_AUDIT_SHA256 = (
    "b05165d9ba1476afc6be133045029eb18b508ed3afea29ffb3716d3ca37b7864"
)
CW15_CHILD_STDERR_SHA256 = (
    "5ceb096b8ac5376a3ee276bf70012f90cb2fd479e3fa2804f2c0a8145aff4512"
)
CW15_CHILD_STDERR_BYTES = 347

CW15_ITERATION1_ADDITIONAL_L2 = 0.0009424461480998953
CW15_TERMINAL_ADDITIONAL_L2 = 0.0009424461480998953
CW15_TERMINAL_TOTAL_FROM_RAW_L2 = 0.007980250928340182
CW15_ITERATION1_MODEL_SHA256 = (
    "ba5098d4e5c52bb6a02c4e69fa2015428d018f85953edc4fb6bc14330634d769"
)
CW15_ITERATION1_ADDITIONAL_SHA256 = (
    "e3814600afa6f1c7843ca073d89a86cb1e32cf162d89f8cbfaf8db378ad3183f"
)
CW15_INITIAL_ACTIVE_IDENTITY_SHA256 = (
    "e70e14b370ae22ec2ab88d4cca9d9337ca005aa68c67af02b7a581e5ae332682"
)
CW15_ITERATION1_ADDITION_IDENTITY_SHA256 = (
    "e91917703f6573f90016444c07eb7350a565b69a8a3362b48d1c7ec4710f5dc0"
)
CW15_ITERATION1_OUTPUT_SHA256 = (
    "8d5fd5ced2091c2ecb3dcc8de4f18f317959742da443dd5c6812921bd56be9e6"
)
CW15_ITERATION1_CANDIDATE_LINEARIZATION_SHA256 = (
    "f7d58b5a6d379f396fbf2a693babc824818a08ca77be83127456c1071c2b365b"
)
CW15_AFFINE_CLOSURE_LEDGER_SHA256 = (
    "392d211b303078127363ab65c34493f5c5d90ac6fb7b7fc3cffaed928b83fec3"
)
CW15_ITERATION2_UNCAPPED_L2 = 0.001501167697632579
CW14_ANCHOR50_ONLY_L2 = 0.0009447829261258565
CW14_ANCHOR50_ONLY_POINT_SHA256 = (
    "08712b376218bbe223ff7a9de1373413af8f2bf32e6cf797ac1959b12e342721"
)
CW15_COMBINED_ANCHOR50_HISTORICAL12_ROW_COUNT = 62
CW11_PHI_ON_EXPANDED_50 = -0.0078125
CW15_ITERATION1_PHI_ON_EXPANDED_50 = -0.015625
EXPECTED_THRESHOLD_FUNCTION_SHAS = {
    "dynamic_cut_for_harm": (
        "db6098d3d017c35db1f1a2a70b1116c4f38b32fd5181b8537220944cad44ea2c"
    ),
    "dynamic_restoration_cut_for_favorable_transition": (
        "5155a7a2ed1d2863d0da612191bf3d75205170d8493855a70d8cf26dc8f401e6"
    ),
    "fixed_repair_gates_and_cuts": (
        "ae72debe74b2409200cadc6b21e81542201c334cd0aa941c75425ec8e5501081"
    ),
    "initial_official_repair_cuts": (
        "9aead99e06041cb1a30a17544d0a59a3b9b4163872534b5ed3f1baafa08a239e"
    ),
    "legacy_dynamic_cuts": (
        "c52f9967d7e84ce46b78e2482e9cdfea213f3bf17af803acb34be09f182c9187"
    ),
    "legacy_pair_cuts": (
        "123fda651f92498669f5a32a99a4cc5ecfe66f2e4b96a7a6c2700af64ed255b9"
    ),
    "merge_active_cuts": (
        "1e45787ba545d5b0090d3758d0b9e2aa3483ee7853173807cb827776ab990b35"
    ),
    "positive_bf16_q": (
        "c39f13a4cc48ebf04296506059d41ed4e14bfedffac7df89be4f2a87a1286242"
    ),
}

# Replaced with exact hashes only after the final solver contract is installed.
FINAL_SOLVER_CONTRACT_PENDING = False
CW16_CRITICAL_FUNCTION_AST_SHA256: dict[str, str] = {
    "cw16_run_scope": (
        "36d87ad96e75f70797c4718f656bba6ad1ebb45176896d4d989b9e8fc3da2eca"
    ),
    "cw16_solver_ast_audit": (
        "14ea08a549b36787e27e0d137a819e951f780676de7e6de35b5a5ea4a9052942"
    ),
    "wrapper_source_audit": (
        "05fa4b95e9acc5ed96b59ce68a0645c828c2c03bf1b855c6942ff608b87e3371"
    ),
    "cw16_validate_static_payload": (
        "bddf1f7f766295a186b2aec0d7186f7441d5540e4f5f7f3d0f42b5715ba58168"
    ),
    "known_cw15_terminal_evidence": (
        "bc3f516e37b2c57629f0342bf9c396e976a1b0619a40fa5430964ec714dfd85d"
    ),
    "cw16_bootstrap_checks": (
        "927697bf3aafb4c11f86fcc722c29926175df2a247c5cd913b78e3277d7c38a2"
    ),
    "cw16_residual_gate_ledger_checks": (
        "29180c63e6b00cacbd1632d7ba7776daf6c3de9a6daf4f7764deae338cc6adc3"
    ),
    "cw16_proposal_checks": (
        "8835abb70552475ab8147fb8f71746da30a20cb540039537d6a9e6eb81731a5d"
    ),
    "cw16_proposal_sequence_checks": (
        "debac269be444add9ce1e0351e265b725401af6550a1050e62e888ab8b578949"
    ),
    "cw16_terminal_semantics_self_test": (
        "c33e241f78e4832eab9273e4ae6fd8ca70f26d22de3c3d059b0837f022e55e50"
    ),
    "cw16_validate_terminal_payload": (
        "ec83ac9e2910b8beca34b26cbc00949e16fa6009f5a7f1f92fbba975b8d4c828"
    ),
    "cw16_preflight": (
        "b7aa2527c4c3374f3a45798687eebc4b32636db2aab58ba615dc0a70115355bb"
    ),
    "cw16_stderr_audit": (
        "d2ad42493a942d9435fe120aec67824b0071493e789f3a70ae849103f491ddb0"
    ),
}
CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256: dict[str, str] = {
    "cached_active_cut_gates": (
        "b2f20e2e3f411eef0ffb012fafefe3bc46656898e520f887911224c2a3d97dcc"
    ),
    "canonical_json": (
        "a5232f2143b85fd9f67d069cac341fcf3e0cc01174a376587b8c9bb7fe90e2e8"
    ),
    "decide_transition": (
        "20c9127042db943a735c88eb915fb83ab4b3dcda81a4db138e3736b8ba51be03"
    ),
    "deduplicate_active_physical": (
        "d804ca3163651c1cafe9decc6bc1084353ded2e30290e22d2f08870998c52d28"
    ),
    "ensure_current_anchor_rows": (
        "bde923c2174f4b8f55260dba5aa06e4586314aa033e363b9ed8870dfe914929f"
    ),
    "evaluate_candidate_once": (
        "ac95d167f9c0d7d63610b415a8c8e69bb0269e168f00db83f7f70b68bc2b6ba6"
    ),
    "linearize_current_physical_rows": (
        "147807505c4e0009e7c01e222f6ece3eb4dd499f7dfb2e08aa242382b47e47d5"
    ),
    "main": (
        "483204cc097c5ef6cac489decf54750770099246ce2e2c925322ca45057e9eaf"
    ),
    "physical_dedup_self_test": (
        "495604d051f80386419201f4d285d43c365d39f0d3bc610a59dc304fe238fe96"
    ),
    "regular_file_evidence": (
        "3378286cba74b866179249a820e7677b15f467472c3bb0ac7f4792d088c3d985"
    ),
    "run_outer_local_trust": (
        "e616d0e80a4cccca6ecb90663bd3cdafebec57a4178c3a8b76efd8c38177b6b7"
    ),
    "run_probe": (
        "12e27f50a1f4720ff3900d3741d2d41b80578dc26c62132a464d4e3621c496b8"
    ),
    "select_current_hard_anchor_rows": (
        "af1adeb7a168b6fe00d0e96cd773734429fa59d056c9aa79a1eaf3315c2530b2"
    ),
    "selftest_result": (
        "e7b475213c22ea4969a6a3ff6c0ddd926efe8cbce17361b4d73316254013313f"
    ),
    "solve_local_trust_maximin": (
        "77cecff5a6ba4f6d606b45a60a52b6805cdefdb6f533a4624cc2677ae92a6c00"
    ),
    "static_result": (
        "6bbdd3c606d08d04b96a89ef35a34234c2465767f86425d5ae1eb2e755ff3c18"
    ),
    "static_source_audit": (
        "222775704244df589769a55a3d57534fad17aa39e8aa6260fb154789a5f040af"
    ),
    "terminal_integrity": (
        "5647c922bc3f5d1a1d730c65a7448d9a5db4a1090e41c9c84963c48cf76c987b"
    ),
    "validate_cw15_closure_evidence": (
        "c12a3f598b7d5acf73fe8c16d3ef9596c4f03792d58c567dc533f5faecf036ff"
    ),
    "validate_frozen_inputs_40": (
        "afd1a6e424fc7c2f283fc0b327e1b78cbcb3a2ae1691e1dbd53291560ad4319f"
    ),
}
EXPECTED_LAUNCHER_CRITICAL_FUNCTIONS = (
    "cw16_run_scope",
    "cw16_solver_ast_audit",
    "wrapper_source_audit",
    "cw16_validate_static_payload",
    "known_cw15_terminal_evidence",
    "cw16_bootstrap_checks",
    "cw16_residual_gate_ledger_checks",
    "cw16_proposal_checks",
    "cw16_proposal_sequence_checks",
    "cw16_terminal_semantics_self_test",
    "cw16_validate_terminal_payload",
    "cw16_preflight",
    "cw16_stderr_audit",
)
EXPECTED_SOLVER_CRITICAL_FUNCTIONS = (
    "canonical_json",
    "regular_file_evidence",
    "validate_frozen_inputs_40",
    "validate_cw15_closure_evidence",
    "deduplicate_active_physical",
    "physical_dedup_self_test",
    "solve_local_trust_maximin",
    "decide_transition",
    "static_source_audit",
    "cached_active_cut_gates",
    "ensure_current_anchor_rows",
    "select_current_hard_anchor_rows",
    "linearize_current_physical_rows",
    "evaluate_candidate_once",
    "terminal_integrity",
    "run_outer_local_trust",
    "run_probe",
    "selftest_result",
    "static_result",
    "main",
)

SCHEMA = "ptcg-cw16-consumed-valid-official6-one-shot-launcher-v1"
ATTEMPT_SCHEMA = "ptcg-cw16-consumed-valid-official6-one-shot-attempt-v1"
FAILURE_SCHEMA = "ptcg-cw16-consumed-valid-official6-one-shot-terminal-failure-v1"
STDERR_SCHEMA = "ptcg-cw16-consumed-valid-official6-one-shot-stderr-audit-v1"
SOLVER_SCHEMA = "ptcg-cw16-consumed-valid-official-b256-local-trust-v1"
EXPECTED_STATIC_TOP_KEYS = (
    "classification",
    "contract",
    "cuda_accessed",
    "curvature_and_negative_self_test",
    "cw15_consumed_closure_evidence",
    "frozen_inputs",
    "inherited_pure_math_self_tests",
    "local_trust_math_self_test",
    "parent_source",
    "physical_dedup_self_test",
    "run_executed",
    "runtime",
    "schema_version",
    "self_source",
    "source_audit",
    "status",
    "writes_performed",
)
EXPECTED_TERMINAL_TOP_KEYS = (
    "classification",
    "cuda_accessed",
    "curvature_and_negative_self_test",
    "cw11_reconstruction_summary",
    "cw15_consumed_closure_evidence",
    "final_integrity",
    "frozen_inputs",
    "input_lock",
    "local_trust_math_self_test",
    "physical_dedup_self_test",
    "run_executed",
    "runtime",
    "schema_version",
    "scope",
    "second_stage",
    "source_audit",
    "status",
    "terminal",
    "terminal_reconstruction_audit",
    "threshold_function_shas",
    "writes_performed",
)

ARTIFACT_ID = "cw16_consumed_valid_official6_local_trust_20260802_v1"
ATTEMPT_MARKER = ARTIFACTS / f".ptcg-{ARTIFACT_ID}-attempt.json"
STDOUT_OUTPUT = ARTIFACTS / f"{ARTIFACT_ID}.stdout.json"
STDERR_AUDIT = ARTIFACTS / f"{ARTIFACT_ID}.stderr-audit.json"

EXPECTED_SOLVER_SHA256 = (
    "a05df3df944451fabefa8e2a037ad541b1f90af7b966be6fc9ecb87bafe2ebbf"
)
EXPECTED_STATIC_STATUS = "static_ready_CW16_local_trust_run_implemented"
EXPECTED_TERMINAL_STATUSES = (
    "consumed_valid_CW16_local_trust_first_feasible",
    "closed_no_CW16_candidate",
)
EXPECTED_SOLVER_FROZEN_BINDINGS = 40

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
    """The inherited launcher, CW15 evidence, or CW16 binding is invalid."""


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

    module_name = f"_cw16_parent_one_shot_{PARENT_LAUNCHER_SHA256[:16]}"
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
    raise ProtocolError("frozen CW15 parent launcher source audit failed")
if not isinstance(getattr(parent_adapter, "engine", None), ModuleType):
    raise ProtocolError("frozen CW15 parent did not expose its inherited engine")
engine = parent_adapter.engine

PARENT_COMPOSED_CHECKS = {
    "cw15_script_exact": engine.SCRIPT == PARENT_LAUNCHER,
    "cw15_solver_path_exact": engine.SOLVER.name
    == "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py",
    "cw15_solver_sha_exact": engine.EXPECTED_SOLVER_SHA256 == CW15_SOLVER_SHA256,
    "cw15_manifest_count_exact_37": engine.EXPECTED_SOLVER_FROZEN_BINDINGS == 37,
    "cw15_static_status_exact": engine.EXPECTED_STATIC_STATUS
    == "static_ready_CW15_sequential_affine_tangent_run_implemented",
    "cw15_success_status_exact": engine.EXPECTED_TERMINAL_STATUSES[0]
    == "consumed_valid_CW15_sequential_affine_tangent_closure_first_feasible",
    "cw15_closed_status_exact": engine.EXPECTED_TERMINAL_STATUSES[1]
    == "closed_no_CW15_candidate",
    "cw15_scope_sequential_affine_exact": engine.RUN_SCOPE.get(
        "CW15_minimum_total_from_CW11_origin"
    ) is True
    and engine.RUN_SCOPE.get("sequential_affine_tangent_optimization_ledger")
    is True
    and engine.RUN_SCOPE.get("post_merge_all_active_false_rows_tangented")
    is True,
}
if not all(PARENT_COMPOSED_CHECKS.values()):
    raise ProtocolError(f"frozen CW15 composed contract drift: {PARENT_COMPOSED_CHECKS}")

# Skip only the CW15-specific 37-record wrapper.  Its exact frozen CW12 core
# preflight is manifest-count-parametric and is overridden below to 40.
_BASE_PREFLIGHT = parent_adapter._BASE_PREFLIGHT
_PARENT_STDERR_AUDIT = engine.validated_stderr_audit
_PARENT_TERMINAL_VALIDATOR = engine.validate_terminal_payload


def cw16_run_scope() -> dict[str, Any]:
    """Return one fresh exact scope mapping for override and source audit."""

    return {
        "exact_CW11_reconstruction": True,
        "exact_CW15_iteration1_bootstrap_replay": True,
        "bootstrap_seed_nonacceptance_exempt_once": True,
        "bootstrap_eligible": False,
        "semantic_active_cut_ledger_monotone": True,
        "CW11_anchor_hard_active": True,
        "CW11_anchor50_only_reference_l2": CW14_ANCHOR50_ONLY_L2,
        "CW15_combined_anchor50_historical12_rows_excluded": True,
        "historical_candidate_tangent_archive_only": True,
        "historical_candidate_tangent_active_count": 0,
        "current_point_all_active_physical_relinearized": True,
        "exact_metric_four_to_one_optimization_dedup": True,
        "physical_dedup_complete_four_metrics_only": True,
        "physical_dedup_threshold_margin_gradient_all_identical": True,
        "physical_partial_or_nonidentical_preserved_rowwise": True,
        "physical_mapping_bijective": True,
        "max_z_nonpositive_then_fixed_z_minimum_norm": True,
        "additional_total_L2_cap": 0.001,
        "trust_radius_divisors": [8, 16, 32, 64],
        "trust_radius_expansion": False,
        "strict_exact_post_merge_Phi_improvement": True,
        "single_changed_model_same_BF16_plateau": True,
        "max_official_candidate_evaluations_including_bootstrap": 12,
        "authoritative_full_six_official_B256_oracle_each_new_model_proposal": True,
        "terminal_gate_count": 10,
        "candidate_RAM_only": True,
        "optimizer_backward_training": False,
        "model_or_result_writes": 0,
        "network_upload_submission": False,
        "broad_or_gold_access": False,
        "specialist_valid_consumed_dev_only": True,
    }


def cw16_solver_ast_audit() -> dict[str, Any]:
    """Bind critical final-solver function bodies independently of file SHA."""

    if FINAL_SOLVER_CONTRACT_PENDING:
        return {
            "checks": {"final_solver_contract_installed": False},
            "pass": False,
            "functions": {},
        }
    source, evidence = engine.read_regular_stable(
        SOLVER,
        "CW16 solver for critical AST audit",
        expected_sha256=EXPECTED_SOLVER_SHA256,
        expected_mode=FROZEN_EXECUTABLE_MODE,
    )
    tree = ast.parse(source, filename=str(SOLVER))
    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    observed = {
        name: sha256_bytes(
            ast.dump(function_nodes[name], include_attributes=False).encode("utf-8")
        )
        for name in CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256
        if name in function_nodes
    }
    checks = {
        "final_solver_contract_installed": FINAL_SOLVER_CONTRACT_PENDING is False,
        "critical_map_nonempty": bool(CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256),
        "critical_map_function_set_exact": set(
            CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256
        )
        == set(EXPECTED_SOLVER_CRITICAL_FUNCTIONS),
        "critical_functions_exact": set(observed)
        == set(CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256),
        "critical_function_hashes_exact": observed
        == CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256,
        "solver_sha_exact": evidence.get("sha256") == EXPECTED_SOLVER_SHA256,
        "solver_mode_exact": evidence.get("mode_octal") == "0555",
        "solver_one_link": evidence.get("nlink") == 1,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "functions": observed,
        "solver": evidence,
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
    solver_ast_audit = cw16_solver_ast_audit()
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
        == "probe_u468_cw11_consumed_valid_official6_cw16_local_trust_v1.py",
        "final_solver_contract_installed": FINAL_SOLVER_CONTRACT_PENDING is False
        and top_level_literals.get("FINAL_SOLVER_CONTRACT_PENDING") is False,
        "top_level_parent_launcher_sha_literal_exact": top_level_literals.get(
            "PARENT_LAUNCHER_SHA256"
        )
        == PARENT_LAUNCHER_SHA256,
        "top_level_cw15_solver_sha_literal_exact": top_level_literals.get(
            "CW15_SOLVER_SHA256"
        )
        == CW15_SOLVER_SHA256,
        "top_level_cw15_artifact_sha_literals_exact": all(
            top_level_literals.get(name) == expected
            for name, expected in {
                "CW15_ATTEMPT_MARKER_SHA256": CW15_ATTEMPT_MARKER_SHA256,
                "CW15_CLOSED_STDOUT_SHA256": CW15_CLOSED_STDOUT_SHA256,
                "CW15_STDERR_AUDIT_SHA256": CW15_STDERR_AUDIT_SHA256,
            }.items()
        ),
        "top_level_solver_sha_lock_exact": top_level_literals.get(
            "EXPECTED_SOLVER_SHA256"
        )
        == EXPECTED_SOLVER_SHA256
        and len(EXPECTED_SOLVER_SHA256) == 64,
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
                "EXPECTED_TERMINAL_TOP_KEYS": EXPECTED_TERMINAL_TOP_KEYS,
                "EXPECTED_LAUNCHER_CRITICAL_FUNCTIONS": (
                    EXPECTED_LAUNCHER_CRITICAL_FUNCTIONS
                ),
                "EXPECTED_SOLVER_CRITICAL_FUNCTIONS": (
                    EXPECTED_SOLVER_CRITICAL_FUNCTIONS
                ),
                "EXPECTED_THRESHOLD_FUNCTION_SHAS": (
                    EXPECTED_THRESHOLD_FUNCTION_SHAS
                ),
            }.items()
        ),
        "top_level_status_literals_exact": top_level_literals.get(
            "EXPECTED_STATIC_STATUS"
        )
        == EXPECTED_STATIC_STATUS
        and top_level_literals.get("EXPECTED_TERMINAL_STATUSES")
        == EXPECTED_TERMINAL_STATUSES,
        "top_level_critical_ast_hash_map_literal_exact": top_level_literals.get(
            "CW16_CRITICAL_FUNCTION_AST_SHA256"
        )
        == CW16_CRITICAL_FUNCTION_AST_SHA256,
        "top_level_solver_critical_ast_hash_map_literal_exact": (
            top_level_literals.get("CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256")
            == CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256
        ),
        "critical_ast_maps_nonempty": bool(CW16_CRITICAL_FUNCTION_AST_SHA256)
        and bool(CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256),
        "critical_ast_function_sets_exact": set(
            CW16_CRITICAL_FUNCTION_AST_SHA256
        )
        == set(EXPECTED_LAUNCHER_CRITICAL_FUNCTIONS)
        and set(CW16_SOLVER_CRITICAL_FUNCTION_AST_SHA256)
        == set(EXPECTED_SOLVER_CRITICAL_FUNCTIONS),
        "solver_critical_ast_audit_pass": solver_ast_audit.get("pass") is True
        and engine.all_true_checks(solver_ast_audit.get("checks")),
        "solver_manifest_count_exact_40": EXPECTED_SOLVER_FROZEN_BINDINGS == 40
        and top_level_literals.get("EXPECTED_SOLVER_FROZEN_BINDINGS") == 40,
        "cw16_static_validator_installed": engine.validate_static_payload
        is cw16_validate_static_payload,
        "cw16_terminal_validator_installed": engine.validate_terminal_payload
        is cw16_validate_terminal_payload,
        "cw16_status_contract_exact": EXPECTED_STATIC_STATUS
        == "static_ready_CW16_local_trust_run_implemented"
        and EXPECTED_TERMINAL_STATUSES
        == (
            "consumed_valid_CW16_local_trust_first_feasible",
            "closed_no_CW16_candidate",
        ),
        "cw16_scope_contract_exact": engine.RUN_SCOPE == cw16_run_scope(),
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
            == ".ptcg-cw16_consumed_valid_official6_local_trust_20260802_v1-attempt.json"
            and STDOUT_OUTPUT.name
            == "cw16_consumed_valid_official6_local_trust_20260802_v1.stdout.json"
            and STDERR_AUDIT.name
            == "cw16_consumed_valid_official6_local_trust_20260802_v1.stderr-audit.json"
        ),
        "targets_distinct_under_artifacts": len(
            {ATTEMPT_MARKER, STDOUT_OUTPUT, STDERR_AUDIT}
        )
        == 3
        and ATTEMPT_MARKER.parent
        == STDOUT_OUTPUT.parent
        == STDERR_AUDIT.parent
        == ARTIFACTS,
        "cw15_deep_terminal_validator_declared": "known_cw15_terminal_evidence"
        in function_names,
        "cw16_solver_ast_audit_declared": "cw16_solver_ast_audit"
        in function_names,
        "cw16_terminal_validator_declared": "cw16_validate_terminal_payload"
        in function_names,
        "critical_function_ast_hashes_exact": set(
            CW16_CRITICAL_FUNCTION_AST_SHA256
        ).issubset(function_nodes)
        and all(
            sha256_bytes(
                ast.dump(function_nodes[name], include_attributes=False).encode(
                    "utf-8"
                )
            )
            == expected
            for name, expected in CW16_CRITICAL_FUNCTION_AST_SHA256.items()
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
        "solver_critical_ast_audit": solver_ast_audit,
        "imported_roots": sorted(imported_roots),
        "unexpected_cli_arguments": output_arguments,
    }


def apply_cw16_overrides() -> None:
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
        "RUN_SCOPE": cw16_run_scope(),
    }
    for name, value in values.items():
        setattr(engine, name, value)
    engine.static_source_audit = wrapper_source_audit


apply_cw16_overrides()


def cw16_validate_static_payload(payload: bytes) -> dict[str, Any]:
    """Validate the exact frozen CW16 local-trust static contract."""

    document = engine.strict_json_object(payload, "CW16 solver static stdout")
    contract = document.get("contract")
    runtime = document.get("runtime")
    self_source = document.get("self_source")
    parent_source = document.get("parent_source")
    frozen = document.get("frozen_inputs")
    cw15_closure = document.get("cw15_consumed_closure_evidence")
    local_math = document.get("local_trust_math_self_test")
    negative_math = document.get("curvature_and_negative_self_test")
    dedup_math = document.get("physical_dedup_self_test")
    inherited_math = document.get("inherited_pure_math_self_tests")
    source_audit = document.get("source_audit")
    expected_contract_subset = {
        "starting_model_sha256": engine.CW11_MODEL_SHA256,
        "CW15_parent_solver_sha256": CW15_SOLVER_SHA256,
        "CW15_attempt_sha256": CW15_ATTEMPT_MARKER_SHA256,
        "CW15_stdout_sha256": CW15_CLOSED_STDOUT_SHA256,
        "CW15_stderr_sha256": CW15_STDERR_AUDIT_SHA256,
        "frozen_input_count": 40,
        "old_input_count": 37,
        "new_CW15_input_count": 3,
        "bootstrap_point_sha256": CW15_ITERATION1_ADDITIONAL_SHA256,
        "bootstrap_model_sha256": CW15_ITERATION1_MODEL_SHA256,
        "bootstrap_output_sha256": CW15_ITERATION1_OUTPUT_SHA256,
        "bootstrap_additions_sha256": CW15_ITERATION1_ADDITION_IDENTITY_SHA256,
        "bootstrap_seed_nonacceptance_exempt_once": True,
        "bootstrap_eligible": False,
        "bootstrap_counts_official_budget": True,
        "semantic_threshold_change_from_CW15": False,
        "threshold_function_shas": EXPECTED_THRESHOLD_FUNCTION_SHAS,
        "terminal_function": "exact imported CW15 terminal_acceptance 10 checks",
        "total_additional_L2_cap": 0.001,
        "max_official_candidate_evaluations": 12,
        "trust_radius_divisors": [8, 16, 32, 64],
        "trust_radius_values": [0.000125, 0.0000625, 0.00003125, 0.000015625],
        "trust_radius_expansion": False,
        "semantic_ledger_monotone": True,
        "CW11_anchor_hard": True,
        "CW11_anchor50_only_reference_l2": CW14_ANCHOR50_ONLY_L2,
        "CW11_anchor50_only_reference_point_sha256": (
            CW14_ANCHOR50_ONLY_POINT_SHA256
        ),
        "CW15_combined_anchor50_historical12_row_count": (
            CW15_COMBINED_ANCHOR50_HISTORICAL12_ROW_COUNT
        ),
        "CW15_combined62_uncapped_l2_archive_only": CW15_ITERATION2_UNCAPPED_L2,
        "historical_candidate_tangent_active_count": 0,
        "current_local_physical_exact_four_to_one": True,
        "physical_dedup_complete_four_metrics_only": True,
        "physical_dedup_threshold_margin_gradient_all_identical": True,
        "physical_partial_or_nonidentical_preserved_rowwise": True,
        "physical_mapping_bijective": True,
        "local_objective": "max z<=0 then fixed-z minimum total anchored L2",
        "exact_merit": "Phi=min post-merge semantic residual",
        "strict_improvement_without_tolerance": True,
        "single_plateau_exception": True,
        "no_clip": True,
        "one_full_six_per_proposal": True,
        "first_feasible": True,
        "candidate_ordering": "fixed sequential no best-of-N",
        "attempt_semantics": (
            "stdout-only probe has no marker; an audited one-shot launcher owns "
            "unique absent output and attempt-marker enforcement"
        ),
    }
    cw15_bootstrap = (
        cw15_closure.get("bootstrap") if isinstance(cw15_closure, Mapping) else None
    )
    cw15_archive = (
        cw15_closure.get("archive") if isinstance(cw15_closure, Mapping) else None
    )
    inherited_names = {
        "anchored_total_geometry",
        "sequential_affine_tangent",
        "affine_ledger_integrity",
    }
    required_dedup_checks = {
        "partial1_2_3_preserved_rowwise",
        "four_nonidentical_threshold_preserved_rowwise",
        "four_nonidentical_margin_preserved_rowwise",
        "four_nonidentical_gradient_preserved_rowwise",
        "semantic_mapping_bijective",
        "duplicate_identity_fails_closed",
        "missing_identity_fails_closed",
        "dedup_feasible_region_equivalent_to_unreduced",
    }
    dedup_checks = dedup_math.get("checks") if isinstance(dedup_math, Mapping) else None
    dedup_audit = dedup_math.get("audit") if isinstance(dedup_math, Mapping) else None
    partial_audits = (
        dedup_math.get("partial_audits") if isinstance(dedup_math, Mapping) else None
    )
    nonidentical_audits = (
        dedup_math.get("nonidentical_four_audits")
        if isinstance(dedup_math, Mapping)
        else None
    )
    checks = {
        "top_level_keys_exact": set(document) == set(EXPECTED_STATIC_TOP_KEYS)
        and len(document) == len(EXPECTED_STATIC_TOP_KEYS),
        "schema_exact": document.get("schema_version") == SOLVER_SCHEMA,
        "status_exact": document.get("status") == EXPECTED_STATIC_STATUS,
        "classification_exact": document.get("classification")
        == engine.CLASSIFICATION,
        "runtime_checks_all_true_no_cuda": isinstance(runtime, Mapping)
        and engine.all_true_checks(runtime.get("checks"))
        and runtime.get("cuda_required") is False,
        "self_source_exact_frozen": isinstance(self_source, Mapping)
        and self_source.get("path") == root_relative(SOLVER)
        and self_source.get("sha256") == EXPECTED_SOLVER_SHA256
        and self_source.get("mode") == "0555"
        and self_source.get("nlink") == 1
        and self_source.get("frozen_required") is True,
        "parent_source_exact_frozen": isinstance(parent_source, Mapping)
        and parent_source.get("path")
        == "tools/probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py"
        and parent_source.get("sha256") == CW15_SOLVER_SHA256
        and parent_source.get("mode") == "0555"
        and parent_source.get("expected_mode") == "0555"
        and parent_source.get("nlink") == 1
        and parent_source.get("regular") is True
        and engine.all_true_checks(parent_source.get("checks")),
        "frozen_inputs_exact_40": isinstance(frozen, Mapping)
        and frozen.get("all_exact") is True
        and frozen.get("binding_count") == 40
        and isinstance(frozen.get("records"), list)
        and len(frozen["records"]) == 40,
        "cw15_closure_evidence_all_true": isinstance(cw15_closure, Mapping)
        and cw15_closure.get("pass") is True
        and engine.all_true_checks(cw15_closure.get("checks")),
        "cw15_bootstrap_exact": isinstance(cw15_bootstrap, Mapping)
        and close_float(
            cw15_bootstrap.get("additional_l2"), CW15_ITERATION1_ADDITIONAL_L2
        )
        and cw15_bootstrap.get("point_sha256")
        == CW15_ITERATION1_ADDITIONAL_SHA256
        and cw15_bootstrap.get("model_sha256") == CW15_ITERATION1_MODEL_SHA256
        and cw15_bootstrap.get("output_sha256") == CW15_ITERATION1_OUTPUT_SHA256
        and cw15_bootstrap.get("additions_sha256")
        == CW15_ITERATION1_ADDITION_IDENTITY_SHA256
        and cw15_bootstrap.get("semantic_count_after_merge") == 50
        and cw15_bootstrap.get("CW11_phi_expanded50") == CW11_PHI_ON_EXPANDED_50
        and cw15_bootstrap.get("bootstrap_phi_expanded50")
        == CW15_ITERATION1_PHI_ON_EXPANDED_50,
        "cw15_archive_anchor50_history12": isinstance(cw15_archive, Mapping)
        and cw15_archive.get("CW11_anchor_count") == 50
        and close_float(
            cw15_archive.get("CW11_anchor_only_reference_l2"),
            CW14_ANCHOR50_ONLY_L2,
        )
        and cw15_archive.get("CW11_anchor_only_reference_point_sha256")
        == CW14_ANCHOR50_ONLY_POINT_SHA256
        and cw15_archive.get("historical_candidate_count") == 12
        and isinstance(cw15_archive.get("historical_candidate_records"), list)
        and len(cw15_archive["historical_candidate_records"]) == 12,
        "local_trust_math_all_true": isinstance(local_math, Mapping)
        and local_math.get("pass") is True
        and engine.all_true_checks(local_math.get("checks"))
        and local_math.get("audit", {}).get("pass") is True
        and engine.all_true_checks(local_math.get("audit", {}).get("checks"))
        and local_math.get("audit", {}).get("scaling", {}).get("pass") is True
        and engine.all_true_checks(
            local_math.get("audit", {}).get("scaling", {}).get("checks")
        )
        and engine.all_true_checks(
            local_math.get("audit", {}).get("stage1", {}).get(
                "raw_certification_checks"
            )
        )
        and local_math.get("audit", {}).get("stage2", {}).get("success") is True,
        "curvature_negative_math_all_true": isinstance(negative_math, Mapping)
        and negative_math.get("pass") is True
        and engine.all_true_checks(negative_math.get("checks")),
        "physical_dedup_math_all_true": isinstance(dedup_math, Mapping)
        and dedup_math.get("pass") is True
        and engine.all_true_checks(dedup_math.get("checks")),
        "physical_dedup_behavior_exact": isinstance(dedup_checks, Mapping)
        and required_dedup_checks.issubset(dedup_checks)
        and all(dedup_checks.get(name) is True for name in required_dedup_checks)
        and isinstance(dedup_audit, Mapping)
        and dedup_audit.get("pass") is True
        and engine.all_true_checks(dedup_audit.get("checks"))
        and dedup_audit.get("semantic_count") == 5
        and dedup_audit.get("optimization_row_count") == 2
        and dedup_audit.get("exact_four_to_one_group_count") == 1
        and isinstance(partial_audits, list)
        and [value.get("semantic_count") for value in partial_audits] == [1, 2, 3]
        and all(
            value.get("optimization_row_count") == value.get("semantic_count")
            and value.get("audit", {}).get("pass") is True
            and value.get("audit", {}).get("exact_four_to_one_group_count") == 0
            for value in partial_audits
        )
        and isinstance(nonidentical_audits, Mapping)
        and set(nonidentical_audits) == {"threshold", "margin", "gradient"}
        and all(
            value.get("pass") is True
            and value.get("semantic_count") == 4
            and value.get("optimization_row_count") == 4
            and value.get("exact_four_to_one_group_count") == 0
            and value.get("checks", {}).get("semantic_mapping_bijective") is True
            for value in nonidentical_audits.values()
        ),
        "inherited_math_exact_all_true": isinstance(inherited_math, Mapping)
        and set(inherited_math) == inherited_names
        and all(
            isinstance(inherited_math.get(name), Mapping)
            and inherited_math[name].get("pass") is True
            and engine.all_true_checks(inherited_math[name].get("checks"))
            for name in inherited_names
        ),
        "contract_subset_exact": isinstance(contract, Mapping)
        and all(
            contract.get(key) == value
            for key, value in expected_contract_subset.items()
        ),
        "source_audit_pass": isinstance(source_audit, Mapping)
        and source_audit.get("pass") is True
        and engine.all_true_checks(source_audit.get("checks"))
        and source_audit.get("threshold_function_shas")
        == EXPECTED_THRESHOLD_FUNCTION_SHAS,
        "run_executed_false": document.get("run_executed") is False,
        "cuda_accessed_false": document.get("cuda_accessed") is False,
        "writes_performed_false": document.get("writes_performed") is False,
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW16 solver static payload contract failed: {checks}")
    return {
        "checks": checks,
        "stdout_sha256": sha256_bytes(payload),
        "stdout_bytes": len(payload),
        "payload": document,
    }


engine.validate_static_payload = cw16_validate_static_payload


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


def known_cw15_terminal_evidence() -> dict[str, Any]:
    """Bind the immutable CW15 affine-cap closure and its three physical rows."""

    marker_raw, marker_evidence = engine.read_regular_stable(
        CW15_ATTEMPT_MARKER,
        "consumed CW15 attempt marker",
        expected_sha256=CW15_ATTEMPT_MARKER_SHA256,
        expected_mode=0o444,
    )
    closed_raw, closed_evidence = engine.read_regular_stable(
        CW15_CLOSED_STDOUT,
        "immutable CW15 closed stdout",
        expected_sha256=CW15_CLOSED_STDOUT_SHA256,
        expected_mode=0o444,
    )
    stderr_raw, stderr_evidence = engine.read_regular_stable(
        CW15_STDERR_AUDIT,
        "immutable CW15 stderr audit",
        expected_sha256=CW15_STDERR_AUDIT_SHA256,
        expected_mode=0o444,
    )
    marker = engine.strict_json_object(marker_raw, "CW15 attempt marker")
    closed = engine.strict_json_object(closed_raw, "CW15 closed stdout")
    stderr = engine.strict_json_object(stderr_raw, "CW15 stderr audit")
    second = closed.get("second_stage")
    terminal = closed.get("terminal")
    final_integrity = closed.get("final_integrity")
    iterations = second.get("iterations") if isinstance(second, Mapping) else None
    iteration_shape = (
        isinstance(iterations, list)
        and len(iterations) == 3
        and [value.get("iteration") for value in iterations] == [0, 1, 2]
    )
    iteration0 = iterations[0] if iteration_shape else {}
    iteration1 = iterations[1] if iteration_shape else {}
    iteration2 = iterations[2] if iteration_shape else {}
    frozen = closed.get("frozen_inputs")
    prior_records = frozen.get("records") if isinstance(frozen, Mapping) else None
    decision = second.get("decision") if isinstance(second, Mapping) else None
    post = stderr.get("post_child_integrity")
    try:
        child_stderr = base64.b64decode(
            str(stderr.get("base64", "")).encode("ascii"), validate=True
        )
        child_stderr_utf8 = child_stderr.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        child_stderr = b""
        child_stderr_utf8 = "Traceback (most recent call last):"

    metrics = {"hybrid", "ordered", "set", "top1"}
    expected_pair_specs = {
        (
            "4afd6b4112ce49bab88c5192c1c5425a395cd629ebf2d8fc81cbbcba6d0d2b0b",
            "ccd32395c70e834cc78a32d66fd182596a5977de86f2c5506c811878975e3f1c",
            1,
            8,
        ): (
            -0.00390625,
            0.0009765625,
            -0.0048828125,
            "048e74d24e08efb74d2ef979d72cbd7e1dbfdbe12c2257e847889987e9f2add2",
        ),
        (
            "861c6645f1764ac84ecbc9fe9448697ddf74b3e728d65dc28142c6d038ccea66",
            "3b8a466d01e49811024d8d01a187aab370672b50b64804ef889cc41a20ed4145",
            1,
            2,
        ): (
            -0.001220703125,
            0.0028076171875,
            -0.0040283203125,
            "ac92898de00c81b06f59319ff0328b350ea6948ac095459c241bf37111d85f9e",
        ),
        (
            "ffd866085ed5f2741e44a98d6c34bb33a2d421c1289c29c9b393a752d9b24a0a",
            "aad8534415cc11e23338dcd82ed3bbf1ec14c375f953fa67ff17d006955463f4",
            3,
            6,
        ): (
            -0.0078125,
            0.0078125,
            -0.015625,
            "cd7706328247a32c50be7f38ccbd95ae6ce975462594cb56626006087845b51a",
        ),
    }
    expected_violations = sorted(
        (
            context,
            decision_sha,
            metric,
            positive,
            negative,
            margin,
            threshold,
            residual,
        )
        for (context, decision_sha, positive, negative), (
            margin,
            threshold,
            residual,
            _,
        ) in expected_pair_specs.items()
        for metric in metrics
    )
    observed_violations = sorted(
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
        for record in iteration1.get("active_cut_gates", {}).get("records", [])
        if record.get("pass") is False
    )
    candidate_records = iteration1.get("candidate_linearization_audit", {}).get(
        "records", []
    )
    candidate_gradient_groups: dict[tuple[Any, ...], set[str]] = {}
    for record in candidate_records if isinstance(candidate_records, list) else []:
        identity = record.get("semantic_identity", [])
        if not isinstance(identity, list) or len(identity) != 5:
            continue
        key = (str(identity[0]), str(identity[1]), int(identity[3]), int(identity[4]))
        candidate_gradient_groups.setdefault(key, set()).add(
            str(record.get("gradient_float64_le_sha256"))
        )
    expected_gradient_groups = {
        key: {values[3]} for key, values in expected_pair_specs.items()
    }
    false_acceptance = {
        key
        for key, passed in iteration1.get("acceptance", {}).get("checks", {}).items()
        if passed is False
    }
    affine_closure = (
        second.get("optimization_affine_closure_ledger", [])
        if isinstance(second, Mapping)
        else []
    )
    affine_kinds = {
        kind: sum(
            int(record.get("linearization_kind") == kind)
            for record in affine_closure
            if isinstance(record, Mapping)
        )
        for kind in ("CW11_anchor", "post_merge_actual_active_violation")
    }

    checks = {
        "marker_schema_status_exact": marker.get("schema_version")
        == "ptcg-cw15-consumed-valid-official6-one-shot-attempt-v1"
        and marker.get("status")
        == "one_shot_attempt_consumed_before_cuda_and_official6",
        "marker_launcher_solver_exact": marker.get("launcher", {}).get("sha256")
        == PARENT_LAUNCHER_SHA256
        and marker.get("solver", {}).get("sha256") == CW15_SOLVER_SHA256
        and marker.get("launcher", {}).get("mode_octal") == "0555"
        and marker.get("solver", {}).get("mode_octal") == "0555"
        and marker.get("launcher", {}).get("nlink") == 1
        and marker.get("solver", {}).get("nlink") == 1,
        "marker_targets_exact": marker.get("stdout_output")
        == root_relative(CW15_CLOSED_STDOUT)
        and marker.get("stderr_audit") == root_relative(CW15_STDERR_AUDIT),
        "marker_one_attempt_no_retry": marker.get("attempts_authorized") == 1
        and marker.get("retry_authorized") is False
        and marker.get("marker_created_before_cuda_and_official6") is True,
        "marker_prior_manifest_exact_37": marker.get(
            "independent_frozen_input_rehash", {}
        ).get("binding_count")
        == 37
        and marker.get("independent_frozen_input_rehash", {}).get("pass") is True,
        "closed_schema_status_exact": closed.get("schema_version")
        == (
            "ptcg-cw15-consumed-valid-official-b256-sequential-affine-"
            "tangent-cuttingplane-v1"
        )
        and closed.get("status") == "closed_no_CW15_candidate"
        and isinstance(second, Mapping)
        and second.get("status") == "closed_no_CW15_candidate"
        and isinstance(terminal, Mapping)
        and terminal.get("status") == "closed_no_CW15_candidate",
        "closed_scope_and_execution_exact": closed.get("scope", {}).get(
            "CW15_minimum_total_from_CW11_origin"
        )
        is True
        and closed.get("scope", {}).get(
            "sequential_affine_tangent_optimization_ledger"
        )
        is True
        and closed.get("scope", {}).get(
            "post_merge_all_active_false_rows_tangented"
        )
        is True
        and closed.get("run_executed") is True
        and closed.get("cuda_accessed") is True
        and closed.get("writes_performed") is False,
        "closed_final_restore_exact": isinstance(final_integrity, Mapping)
        and final_integrity.get("pass") is True
        and final_integrity.get("model_left_raw_after_outer_finally") is True
        and engine.all_true_checks(final_integrity.get("checks")),
        "closed_prior_manifest_exact_37": isinstance(prior_records, list)
        and len(prior_records) == 37
        and frozen.get("binding_count") == 37
        and frozen.get("all_exact") is True,
        "closed_prior_evidence_chain_exact": closed.get(
            "cw12_consumed_failure_evidence", {}
        ).get("pass")
        is True
        and closed.get("cw13_consumed_closure_evidence", {}).get("pass") is True
        and closed.get("cw14_consumed_closure_evidence", {}).get("pass") is True,
        "cw14_anchor50_only_reference_exact_under_cap": closed.get(
            "source_audit", {}
        ).get("top_level_literals", {}).get("CW14_TERMINAL_ADDITIONAL_L2")
        == CW14_ANCHOR50_ONLY_L2
        and closed.get("source_audit", {}).get("top_level_literals", {}).get(
            "CW14_STALLED_ADDITIONAL_SHA256"
        )
        == CW14_ANCHOR50_ONLY_POINT_SHA256
        and closed.get("cw14_consumed_closure_evidence", {}).get(
            "fixed_point_diagnosis", {}
        ).get("stalled_additional_float64_le_sha256")
        == CW14_ANCHOR50_ONLY_POINT_SHA256
        and CW14_ANCHOR50_ONLY_L2 < 0.001,
        "closed_decision_exact": isinstance(decision, Mapping)
        and decision.get("close_reason")
        == "fail_closed_anchored_minimum_total_exceeds_cap"
        and decision.get("terminal_iteration") is None
        and decision.get("terminal_model_state_sha256") is None
        and decision.get("candidate_consumer_called") is False
        and decision.get("terminal_active_cut_count") == 50
        and decision.get("optimization_affine_tangent_count") == 62
        and decision.get("optimization_affine_tangent_ledger_sha256")
        == CW15_AFFINE_CLOSURE_LEDGER_SHA256
        and close_float(
            decision.get("terminal_additional_l2"), CW15_TERMINAL_ADDITIONAL_L2
        )
        and close_float(
            decision.get("terminal_total_from_raw_l2"),
            CW15_TERMINAL_TOTAL_FROM_RAW_L2,
        ),
        "closed_terminal_payloads_absent": isinstance(terminal, Mapping)
        and terminal.get("reconstruction_payload") is None
        and terminal.get("reconstruction_audit") is None
        and terminal.get("active_cut_ledger") is None
        and terminal.get("active_cut_ledger_sha256") is None
        and terminal.get("optimization_affine_tangent_ledger") is None
        and terminal.get("optimization_affine_tangent_ledger_sha256") is None,
        "iteration_shape_exact_0_1_2": iteration_shape,
        "iteration0_reference_exact": iteration0.get("kind")
        == "exact_CW11_reference_before_fixed3_repair"
        and iteration0.get("model_state_sha256")
        == "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
        and iteration0.get("active_cut_count") == 38
        and iteration0.get("active_cut_identity_sha256")
        == CW15_INITIAL_ACTIVE_IDENTITY_SHA256
        and iteration0.get("optimization_affine_tangent_count") == 0
        and iteration0.get("active_cut_gates", {}).get("violated_count") == 4,
        "iteration1_exact_replay_and_merge": iteration1.get("kind")
        == "unique_fixed_order_official_context_proposal"
        and iteration1.get("model_state_sha256") == CW15_ITERATION1_MODEL_SHA256
        and iteration1.get("additional_float64_le_sha256")
        == CW15_ITERATION1_ADDITIONAL_SHA256
        and close_float(
            iteration1.get("additional_l2"), CW15_ITERATION1_ADDITIONAL_L2
        )
        and iteration1.get("output_state_fingerprint", {}).get("sha256")
        == CW15_ITERATION1_OUTPUT_SHA256
        and iteration1.get("qp", {}).get("gradient_shape") == [38, 65793]
        and iteration1.get("qp", {}).get("svd_rank") == 37
        and iteration1.get("qp", {}).get("capped") is False
        and iteration1.get("anchor_affine_merge", {}).get("added_count") == 38
        and iteration1.get("post_oracle_cut_merge", {}).get("active_count") == 50
        and iteration1.get("deterministic_addition_count") == 12
        and iteration1.get("deterministic_addition_identity_sha256")
        == CW15_ITERATION1_ADDITION_IDENTITY_SHA256
        and iteration1.get("oracle", {}).get("new_harm_count_vs_cw11") == 12
        and iteration1.get("oracle", {}).get(
            "favorable_transition_break_count"
        )
        == 8,
        "iteration1_all_12_false_rows_exact": len(expected_violations) == 12
        and observed_violations == expected_violations,
        "iteration1_three_physical_gradients_exact":
        candidate_gradient_groups == expected_gradient_groups
        and len(candidate_records) == 12,
        "iteration1_candidate_tangent_bijection_exact": iteration1.get(
            "candidate_linearization_audit", {}
        ).get("pass")
        is True
        and iteration1.get("candidate_linearization_audit", {}).get(
            "semantic_cut_count"
        )
        == 12
        and iteration1.get("candidate_linearization_audit", {}).get(
            "affine_record_count"
        )
        == 12
        and iteration1.get("candidate_linearization_audit", {}).get(
            "semantic_identity_sha256"
        )
        == CW15_ITERATION1_ADDITION_IDENTITY_SHA256
        and iteration1.get("candidate_linearization_audit", {}).get(
            "sanitized_affine_record_sha256"
        )
        == CW15_ITERATION1_CANDIDATE_LINEARIZATION_SHA256
        and engine.all_true_checks(iteration1.get("candidate_tangent_checks")),
        "iteration1_affine_growth_38_50_62_exact": iteration1.get(
            "post_merge_candidate_affine_tangent_merge", {}
        ).get("added_count")
        == 12
        and iteration1.get(
            "post_merge_candidate_affine_tangent_merge", {}
        ).get("affine_count")
        == 50
        and iteration1.get(
            "post_merge_new_semantic_anchor_affine_merge", {}
        ).get("added_count")
        == 12
        and iteration1.get(
            "post_merge_new_semantic_anchor_affine_merge", {}
        ).get("affine_count")
        == 62
        and iteration1.get(
            "optimization_affine_ledger_sha256_after_candidate_and_new_anchor"
        )
        == CW15_AFFINE_CLOSURE_LEDGER_SHA256,
        "iteration1_integrity_and_acceptance_exact": engine.all_true_checks(
            iteration1.get("integrity")
        )
        and iteration1.get("acceptance", {}).get("pass") is False
        and false_acceptance
        == {
            "active_cut_violations_zero",
            "forensic_favorable_retention_pass",
            "new_harm_count_zero",
        },
        "iteration2_exact_affine_cap_closure": iteration2.get("kind")
        == "fail_closed_before_proposal"
        and iteration2.get("close_reason")
        == "fail_closed_anchored_minimum_total_exceeds_cap"
        and iteration2.get("active_cut_count") == 50
        and iteration2.get("optimization_affine_tangent_count") == 62
        and close_float(
            iteration2.get("anchored_minimum_total_l2"),
            CW15_ITERATION2_UNCAPPED_L2,
        )
        and iteration2.get("clipped_vector_applied") is False
        and iteration2.get("qp", {}).get("affine_tangent_count") == 62
        and iteration2.get("qp", {}).get("gradient_shape") == [62, 65793]
        and iteration2.get("qp", {}).get("svd_rank") == 43
        and iteration2.get("qp", {}).get("solver", {}).get("success") is True
        and iteration2.get("qp", {}).get("solver", {}).get("status") == 0
        and iteration2.get("qp", {}).get("capped") is True
        and close_float(
            iteration2.get("qp", {}).get("uncapped_l2"),
            CW15_ITERATION2_UNCAPPED_L2,
        )
        and close_float(
            iteration2.get("qp", {}).get("applied_l2"),
            0.0010000000000000002,
        )
        and iteration2.get("qp", {}).get("affine_ledger_sha256")
        == CW15_AFFINE_CLOSURE_LEDGER_SHA256,
        "affine_closure_ledger_exact": isinstance(affine_closure, list)
        and len(affine_closure) == 62
        and affine_kinds
        == {
            "CW11_anchor": 50,
            "post_merge_actual_active_violation": 12,
        }
        and second.get("optimization_affine_closure_ledger_sha256")
        == CW15_AFFINE_CLOSURE_LEDGER_SHA256,
        "stderr_schema_status_exact": stderr.get("schema_version")
        == "ptcg-cw15-consumed-valid-official6-one-shot-stderr-audit-v1"
        and stderr.get("status") == "captured_losslessly_not_a_success_veto",
        "stderr_lossless_exact": len(child_stderr) == CW15_CHILD_STDERR_BYTES
        and stderr.get("bytes") == CW15_CHILD_STDERR_BYTES
        and sha256_bytes(child_stderr)
        == stderr.get("sha256")
        == CW15_CHILD_STDERR_SHA256
        and "Traceback (most recent call last):" not in child_stderr_utf8,
        "stderr_post_child_crosslinks_exact": isinstance(post, Mapping)
        and post.get("pass") is True
        and post.get("solver", {}).get("sha256") == CW15_SOLVER_SHA256
        and post.get("attempt_marker", {}).get("sha256")
        == CW15_ATTEMPT_MARKER_SHA256
        and post.get("independent_frozen_input_rehash", {}).get("binding_count")
        == 37
        and post.get("independent_frozen_input_rehash", {}).get("pass") is True,
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW15 immutable closure evidence gate failed: {checks}")
    return {
        "checks": checks,
        "attempt_marker": marker_evidence,
        "closed_stdout": closed_evidence,
        "stderr_audit": stderr_evidence,
        "prior_frozen_records": prior_records,
        "iteration1_model_state_sha256": CW15_ITERATION1_MODEL_SHA256,
        "iteration1_additional_float64_le_sha256":
        CW15_ITERATION1_ADDITIONAL_SHA256,
        "iteration1_output_state_sha256": CW15_ITERATION1_OUTPUT_SHA256,
        "physical_violation_count": len(expected_pair_specs),
        "semantic_violation_count": len(expected_violations),
        "affine_closure_ledger_sha256": CW15_AFFINE_CLOSURE_LEDGER_SHA256,
        "hard_active_CW11_anchor_count": 50,
        "hard_active_CW11_anchor_only_reference_l2": CW14_ANCHOR50_ONLY_L2,
        "hard_active_CW11_anchor_only_reference_point_sha256": (
            CW14_ANCHOR50_ONLY_POINT_SHA256
        ),
        "historical_candidate_archive_only_count": 12,
        "combined_anchor50_historical12_archive_only_uncapped_l2": (
            CW15_ITERATION2_UNCAPPED_L2
        ),
        "affine_uncapped_minimum_total_l2": CW15_ITERATION2_UNCAPPED_L2,
        "unchanged_total_cap": 0.001,
        "pass": True,
    }


def cw16_bootstrap_checks(row: Mapping[str, Any]) -> dict[str, bool]:
    """Gate the unique counted but formally ineligible CW15 bootstrap replay."""

    output = row.get("output_state_fingerprint")
    pre = row.get("replay_checks_pre_oracle")
    post = row.get("replay_checks_post_oracle")
    merge = row.get("post_oracle_cut_merge")
    gates = row.get("active_cut_gates")
    acceptance = row.get("acceptance")
    anchor50_only = row.get("hard_anchor50_only_audit")
    return {
        "identity_exact": row.get("iteration") == 1
        and row.get("kind") == "exact_CW15_iteration1_bootstrap_seed"
        and row.get("bootstrap_seed_nonacceptance_exempt_once") is True,
        "ineligible_nonacceptance_exact": row.get("eligible") is False
        and row.get("is_acceptance") is False,
        "budget_count_exact1": row.get("counts_toward_official_budget") is True
        and row.get("official_evaluation_count_after") == 1,
        "point_model_output_exact": row.get("additional_float64_le_sha256")
        == CW15_ITERATION1_ADDITIONAL_SHA256
        and row.get("model_state_sha256") == CW15_ITERATION1_MODEL_SHA256
        and isinstance(output, Mapping)
        and output.get("sha256") == CW15_ITERATION1_OUTPUT_SHA256,
        "bootstrap_total_hash_hex64": engine.is_sha256(
            row.get("total_from_raw_float64_le_sha256")
        ),
        "additional_l2_exact_within_cap": close_float(
            row.get("additional_l2"), CW15_ITERATION1_ADDITIONAL_L2
        )
        and float(row.get("additional_l2", math.inf)) <= 0.001 + 1e-12,
        "replay_checks_all_true": engine.all_true_checks(pre)
        and engine.all_true_checks(post),
        "addition_merge_exact12_to50": isinstance(merge, Mapping)
        and merge.get("active_count") == 50
        and isinstance(merge.get("added"), list)
        and len(merge["added"]) == 12
        and not merge.get("strengthened"),
        "active50_exact": isinstance(gates, Mapping)
        and gates.get("count") == 50
        and isinstance(gates.get("records"), list)
        and len(gates["records"]) == 50,
        "expanded_phi_exact_and_not_improved": row.get(
            "CW11_phi_on_expanded_ledger"
        )
        == CW11_PHI_ON_EXPANDED_50
        and row.get("bootstrap_phi_on_expanded_ledger")
        == CW15_ITERATION1_PHI_ON_EXPANDED_50
        and CW15_ITERATION1_PHI_ON_EXPANDED_50 <= CW11_PHI_ON_EXPANDED_50,
        "history_archive12_active0": row.get(
            "historical_candidate_tangent_archive_count"
        )
        == 12
        and row.get("historical_candidate_tangent_active_count") == 0,
        "hard_anchor50_restored": row.get("post_bootstrap_anchor_growth", {}).get(
            "pass"
        )
        is True
        and engine.all_true_checks(
            row.get("post_bootstrap_anchor_growth", {}).get("checks")
        )
        and row.get("hard_anchor50_selection", {}).get("pass") is True
        and engine.all_true_checks(
            row.get("hard_anchor50_selection", {}).get("checks")
        ),
        "anchor50_only_hard_under_cap_history12_excluded": isinstance(
            anchor50_only, Mapping
        )
        and anchor50_only.get("pass") is True
        and engine.all_true_checks(anchor50_only.get("checks"))
        and anchor50_only.get("selected_anchor_row_count") == 50
        and anchor50_only.get("selected_anchor_kind") == "CW11_anchor"
        and anchor50_only.get("historical_candidate_active_count") == 0
        and anchor50_only.get("historical_candidate_archive_count") == 12
        and close_float(anchor50_only.get("anchor_only_l2"), CW14_ANCHOR50_ONLY_L2)
        and anchor50_only.get("anchor_only_point_sha256")
        == CW14_ANCHOR50_ONLY_POINT_SHA256
        and close_float(
            anchor50_only.get("consumed_combined62_uncapped_l2"),
            CW15_ITERATION2_UNCAPPED_L2,
        )
        and anchor50_only.get("anchor_array_audit", {}).get("pass") is True
        and engine.all_true_checks(
            anchor50_only.get("anchor_array_audit", {}).get("checks")
        ),
        "integrity_all_true": engine.all_true_checks(row.get("integrity")),
        "imported_acceptance_exact10_not_terminal": isinstance(acceptance, Mapping)
        and isinstance(acceptance.get("checks"), Mapping)
        and len(acceptance["checks"]) == 10
        and acceptance.get("pass") is False
        and row.get("acceptance_false_keys")
        == [
            "active_cut_violations_zero",
            "forensic_favorable_retention_pass",
            "new_harm_count_zero",
        ],
    }


def cw16_residual_gate_ledger_checks(value: Any) -> dict[str, bool]:
    """Recompute one exact residual minimum from its complete gate records."""

    records = value.get("records") if isinstance(value, Mapping) else None
    residuals = (
        [record.get("residual") for record in records]
        if isinstance(records, list)
        and all(isinstance(record, Mapping) for record in records)
        else []
    )
    record_values_exact = bool(residuals) and all(
        isinstance(residual, (int, float))
        and math.isfinite(float(residual))
        and isinstance(record.get("pass"), bool)
        for residual, record in zip(residuals, records)
    )
    return {
        "mapping_and_count_exact": isinstance(value, Mapping)
        and isinstance(records, list)
        and bool(records)
        and value.get("count") == len(records),
        "record_residuals_finite_and_pass_bits_boolean": record_values_exact,
        "residual_min_recomputed_exact": record_values_exact
        and value.get("residual_min")
        == min(float(residual) for residual in residuals),
        "violated_count_recomputed_exact": record_values_exact
        and value.get("violated_count")
        == sum(int(record.get("pass") is False) for record in records),
        "aggregate_pass_recomputed_exact": record_values_exact
        and value.get("pass")
        is all(record.get("pass") is True for record in records),
    }


def cw16_proposal_checks(row: Mapping[str, Any]) -> dict[str, bool]:
    """Gate one and only one official-six evaluation per CW16 proposal row."""

    iteration = row.get("iteration")
    trust_index = row.get("trust_radius_index")
    expected_divisors = (8, 16, 32, 64)
    expected_radii = (0.000125, 0.0000625, 0.00003125, 0.000015625)
    output = row.get("output_state_fingerprint")
    local_audit = row.get("current_local_audit")
    dedup = (
        local_audit.get("deduplication")
        if isinstance(local_audit, Mapping)
        else None
    )
    mappings = dedup.get("mappings") if isinstance(dedup, Mapping) else None
    subproblem = row.get("subproblem")
    merit = row.get("exact_merit")
    transition = row.get("transition")
    acceptance = row.get("acceptance")
    oracle = row.get("oracle")
    fixed = row.get("fixed_repair_gates")
    legacy_gate = row.get("legacy_gate")
    semantic_merge = row.get("post_oracle_cut_merge")
    candidate_gates = row.get("active_cut_gates")
    current_expanded_gates = row.get(
        "cached_current_active_cut_gates_on_expanded_ledger"
    )
    quiescence = row.get("terminal_semantic_quiescence")
    proposal_anchor = row.get("proposal_subproblem_anchor_ledger")
    post_anchor_selection = row.get("post_merge_current_anchor_selection")
    post_anchor = row.get("post_merge_current_anchor_ledger")
    official_gates = (
        oracle.get("authoritative_60_gates") if isinstance(oracle, Mapping) else None
    )
    phi_current = (
        merit.get("current_on_expanded_ledger")
        if isinstance(merit, Mapping)
        else None
    )
    phi_candidate = (
        merit.get("candidate_on_expanded_ledger")
        if isinstance(merit, Mapping)
        else None
    )
    strict_phi = (
        isinstance(phi_current, (int, float))
        and isinstance(phi_candidate, (int, float))
        and math.isfinite(float(phi_current))
        and math.isfinite(float(phi_candidate))
        and float(phi_candidate) > float(phi_current)
    )
    predicted_phi = (
        subproblem.get("predicted_phi")
        if isinstance(subproblem, Mapping)
        else None
    )
    changed_model = row.get("model_state_sha256") != row.get(
        "center_model_state_sha256"
    )
    same_bf16_output = (
        isinstance(output, Mapping)
        and output.get("sha256") == row.get("center_output_state_sha256")
    )
    semantic_changed = bool(
        isinstance(semantic_merge, Mapping)
        and (semantic_merge.get("added") or semantic_merge.get("strengthened"))
    )
    strict_predicted = bool(
        isinstance(predicted_phi, (int, float))
        and math.isfinite(float(predicted_phi))
        and isinstance(phi_current, (int, float))
        and math.isfinite(float(phi_current))
        and float(predicted_phi) > float(phi_current)
    )
    plateau_already_used = (
        transition.get("plateau_already_used")
        if isinstance(transition, Mapping)
        else None
    )
    plateau_allowed = bool(
        not strict_phi
        and strict_predicted
        and changed_model
        and same_bf16_output
        and not semantic_changed
        and plateau_already_used is False
    )
    expected_action = (
        "accept_strict_phi_improvement"
        if strict_phi
        else (
            "move_once_same_BF16_plateau_and_shrink"
            if plateau_allowed
            else "reject_and_shrink"
        )
    )
    acceptance_pass = isinstance(acceptance, Mapping) and acceptance.get("pass") is True
    terminal_component_expected = bool(
        isinstance(oracle, Mapping)
        and oracle.get("authoritative_60_gates", {}).get("pass") is True
        and isinstance(fixed, Mapping)
        and fixed.get("pass") is True
        and isinstance(legacy_gate, Mapping)
        and legacy_gate.get("pass") is True
        and oracle.get("new_harm_count_vs_cw11") == 0
        and oracle.get("transition_state_checks", {}).get(
            "favorable_transitions_retained"
        )
        is True
    )
    quiescence_exact = (
        isinstance(quiescence, Mapping)
        and isinstance(semantic_merge, Mapping)
        and isinstance(transition, Mapping)
        and quiescence.get("terminal_component_gates_pass")
        is terminal_component_expected
        and quiescence.get("deterministic_addition_count_zero")
        is (row.get("deterministic_addition_count") == 0)
        and quiescence.get("semantic_merge_added_zero")
        is (not bool(semantic_merge.get("added")))
        and quiescence.get("semantic_merge_strengthened_zero")
        is (not bool(semantic_merge.get("strengthened")))
        and quiescence.get("semantic_changed_false")
        is (transition.get("semantic_changed") is False)
        and isinstance(proposal_anchor, Mapping)
        and isinstance(post_anchor, Mapping)
        and quiescence.get("proposal_and_post_merge_anchor_count_equal")
        is (proposal_anchor.get("row_count") == post_anchor.get("row_count"))
        and quiescence.get("proposal_and_post_merge_anchor_ledger_sha_equal")
        is (
            proposal_anchor.get("sanitized_ledger_sha256")
            == post_anchor.get("sanitized_ledger_sha256")
        )
    )
    dedup_mapping_exact = isinstance(mappings, list) and all(
        isinstance(value, Mapping)
        and isinstance(value.get("semantic_count"), int)
        and value["semantic_count"] >= 1
        and isinstance(value.get("retained_optimization_count"), int)
        and (
            (
                value.get("exact_four_to_one") is True
                and value["semantic_count"] == 4
                and value["retained_optimization_count"] == 1
                and engine.all_true_checks(value.get("exact_four_checks"))
            )
            or (
                value.get("exact_four_to_one") is False
                and value["retained_optimization_count"]
                == value["semantic_count"]
            )
        )
        for value in mappings
    )
    return {
        "proposal_identity_and_budget_exact": isinstance(iteration, int)
        and 2 <= iteration <= 12
        and row.get("kind") == "CW16_unique_local_trust_official_proposal"
        and row.get("bootstrap_seed_nonacceptance_exempt_once") is False
        and row.get("official_evaluation_count_after") == iteration,
        "point_model_output_total_hashes_hex64": all(
            engine.is_sha256(value)
            for value in (
                row.get("additional_float64_le_sha256"),
                row.get("model_state_sha256"),
                row.get("total_from_raw_float64_le_sha256"),
                output.get("sha256") if isinstance(output, Mapping) else None,
            )
        ),
        "additional_total_cap": isinstance(row.get("additional_l2"), (int, float))
        and 0.0 < float(row["additional_l2"]) <= 0.001 + 1e-12,
        "trust_bucket_exact": isinstance(trust_index, int)
        and 0 <= trust_index < 4
        and row.get("trust_radius_divisor") == expected_divisors[trust_index]
        and close_float(
            row.get("trust_radius"), expected_radii[trust_index], tolerance=1e-20
        ),
        "single_evaluation_checks_all_true": engine.all_true_checks(
            row.get("single_evaluation_checks")
        )
        and engine.all_true_checks(row.get("pre_oracle_repeat_checks")),
        "hard_anchor_selection_arrays_pass": row.get(
            "hard_anchor_selection", {}
        ).get("pass")
        is True
        and engine.all_true_checks(row.get("hard_anchor_selection", {}).get("checks"))
        and row.get("hard_anchor_array_audit", {}).get("pass") is True
        and engine.all_true_checks(
            row.get("hard_anchor_array_audit", {}).get("checks")
        ),
        "proposal_QP_anchor_ledger_exact": isinstance(proposal_anchor, Mapping)
        and proposal_anchor.get("row_count")
        == row.get("hard_anchor_selection", {}).get("active_anchor_count")
        and proposal_anchor.get("sanitized_ledger_sha256")
        == row.get("hard_anchor_selection", {}).get(
            "active_anchor_ledger_sha256"
        )
        and proposal_anchor.get("linearization_kind") == "CW11_anchor"
        and proposal_anchor.get("used_in_this_proposal_subproblem") is True,
        "local_relinearization_and_dedup_pass": isinstance(local_audit, Mapping)
        and local_audit.get("pass") is True
        and engine.all_true_checks(local_audit.get("checks"))
        and isinstance(dedup, Mapping)
        and dedup.get("pass") is True
        and engine.all_true_checks(dedup.get("checks"))
        and dedup.get("checks", {}).get("semantic_mapping_bijective") is True
        and dedup.get("checks", {}).get(
            "only_exact_four_groups_are_deduplicated"
        )
        is True
        and dedup_mapping_exact,
        "subproblem_two_stage_all_true_no_clip": isinstance(subproblem, Mapping)
        and subproblem.get("pass") is True
        and engine.all_true_checks(subproblem.get("checks"))
        and subproblem.get("scaling", {}).get("pass") is True
        and engine.all_true_checks(subproblem.get("scaling", {}).get("checks"))
        and subproblem.get("scaling", {}).get(
            "raw_problem_direct_gates_after_inverse_scaling"
        )
        is True
        and subproblem.get("stage1", {}).get("success") is True
        and engine.all_true_checks(
            subproblem.get("stage1", {}).get("raw_certification_checks")
        )
        and isinstance(
            subproblem.get("stage1", {}).get("z_certification_adjustment"),
            (int, float),
        )
        and 0.0
        <= float(subproblem["stage1"]["z_certification_adjustment"])
        <= 1e-8
        and subproblem.get("stage2", {}).get("success") is True
        and subproblem.get("clipped_vector_applied") is False
        and subproblem.get("historical_candidate_active_count") == 0
        and subproblem.get("trust_radius") == row.get("trust_radius"),
        "history_archive12_active0": row.get(
            "historical_candidate_tangent_archive_count"
        )
        == 12
        and row.get("historical_candidate_tangent_active_count") == 0,
        "semantic_merge_monotone": engine.all_true_checks(
            row.get("semantic_monotone_checks")
        ),
        "exact_merit_strict_bit_exact": isinstance(merit, Mapping)
        and merit.get("definition") == "Phi=min post-merge semantic residual"
        and merit.get("strict_comparison_without_tolerance") is strict_phi,
        "exact_merit_recomputed_from_post_merge_gate_ledgers": isinstance(
            merit, Mapping
        )
        and engine.all_true_checks(
            cw16_residual_gate_ledger_checks(current_expanded_gates)
        )
        and engine.all_true_checks(
            cw16_residual_gate_ledger_checks(candidate_gates)
        )
        and merit.get("current_on_expanded_ledger")
        == current_expanded_gates.get("residual_min")
        and merit.get("candidate_on_expanded_ledger")
        == candidate_gates.get("residual_min")
        and merit.get("predicted_phi") == predicted_phi,
        "transition_strict_bit_exact": isinstance(transition, Mapping)
        and transition.get("strict_actual_phi_improvement") is strict_phi
        and transition.get("strict_predicted_phi_improvement")
        is strict_predicted
        and transition.get("changed_model") is changed_model
        and transition.get("same_BF16_output") is same_bf16_output
        and transition.get("semantic_changed") is semantic_changed
        and isinstance(plateau_already_used, bool)
        and transition.get("plateau_exception_allowed") is plateau_allowed
        and transition.get("action") == expected_action
        and row.get("second_same_BF16_plateau")
        is (
            same_bf16_output
            and changed_model
            and not strict_phi
            and plateau_already_used
        ),
        "eligibility_exact_acceptance_and_strict_phi": row.get("eligible")
        is (acceptance_pass and strict_phi),
        "imported_acceptance_exact10": isinstance(acceptance, Mapping)
        and isinstance(acceptance.get("checks"), Mapping)
        and len(acceptance["checks"]) == 10
        and (
            acceptance.get("pass") is False
            or engine.all_true_checks(acceptance.get("checks"))
        ),
        "post_merge_anchor_growth_pass": row.get(
            "post_merge_hard_anchor_growth", {}
        ).get("pass")
        is True
        and engine.all_true_checks(
            row.get("post_merge_hard_anchor_growth", {}).get("checks")
        ),
        "post_merge_anchor_ledger_distinct_semantics_exact": isinstance(
            post_anchor_selection, Mapping
        )
        and post_anchor_selection.get("pass") is True
        and engine.all_true_checks(post_anchor_selection.get("checks"))
        and isinstance(post_anchor, Mapping)
        and post_anchor.get("row_count")
        == post_anchor_selection.get("active_anchor_count")
        and post_anchor.get("sanitized_ledger_sha256")
        == post_anchor_selection.get("active_anchor_ledger_sha256")
        and post_anchor.get("linearization_kind") == "CW11_anchor"
        and post_anchor.get("used_in_already_evaluated_proposal_subproblem")
        is False,
        "terminal_quiescence_bits_exact": quiescence_exact
        and (not acceptance_pass or engine.all_true_checks(quiescence)),
        "integrity_all_true": engine.all_true_checks(row.get("integrity")),
        "official_oracle_shape_and_hard_checks": isinstance(oracle, Mapping)
        and engine.all_true_checks(oracle.get("hard_checks"))
        and isinstance(official_gates, Mapping)
        and official_gates.get("gate_count") == 60
        and isinstance(official_gates.get("records"), list)
        and len(official_gates["records"]) == 60,
        "local_tangent_archived_inactive": row.get(
            "local_tangent_archived_after_subproblem", {}
        ).get("active_in_future_subproblem")
        is False
        and engine.is_sha256(
            row.get("local_tangent_archived_after_subproblem", {}).get(
                "sanitized_ledger_sha256"
            )
        ),
    }


def cw16_proposal_sequence_checks(
    iterations: list[Any],
    bootstrap_rows: list[Mapping[str, Any]],
    proposal_rows: list[Mapping[str, Any]],
    fail_before_rows: list[Mapping[str, Any]],
) -> dict[str, bool]:
    """Gate the complete iteration partition, trust schedule, and plateau use."""

    partition_exact = (
        len(bootstrap_rows) == 1
        and len(iterations)
        == 1 + len(bootstrap_rows) + len(proposal_rows) + len(fail_before_rows)
    )
    proposals_contiguous = (
        proposal_rows == iterations[2 : 2 + len(proposal_rows)]
        and [row.get("iteration") for row in proposal_rows]
        == list(range(2, 2 + len(proposal_rows)))
    )
    fail_position_exact = not fail_before_rows or (
        len(fail_before_rows) == 1
        and fail_before_rows[0] is iterations[-1]
        and fail_before_rows[0].get("iteration") == 2 + len(proposal_rows)
    )
    expected_plateau_used = False
    plateau_sequence_exact = True
    plateau_use_count = 0
    trust_sequence_exact = True
    expected_radii = (0.000125, 0.0000625, 0.00003125, 0.000015625)
    for ordinal, row in enumerate(proposal_rows):
        transition = row.get("transition")
        if not isinstance(transition, Mapping):
            plateau_sequence_exact = False
            trust_sequence_exact = False
            continue
        if transition.get("plateau_already_used") is not expected_plateau_used:
            plateau_sequence_exact = False
        if transition.get("plateau_exception_allowed") is True:
            plateau_use_count += 1
            if expected_plateau_used:
                plateau_sequence_exact = False
            expected_plateau_used = True
        trust_index = row.get("trust_radius_index")
        has_next = ordinal + 1 < len(proposal_rows)
        next_index = row.get("next_trust_radius_index")
        next_radius = row.get("next_trust_radius")
        if has_next:
            observed_following = proposal_rows[ordinal + 1].get(
                "trust_radius_index"
            )
            trust_sequence_exact = trust_sequence_exact and (
                isinstance(trust_index, int)
                and isinstance(next_index, int)
                and next_index == observed_following
                and trust_index <= next_index <= trust_index + 1
                and 0 <= next_index < len(expected_radii)
                and close_float(
                    next_radius, expected_radii[next_index], tolerance=1e-20
                )
            )
            shrink_action = transition.get("action") in {
                "move_once_same_BF16_plateau_and_shrink",
                "reject_and_shrink",
            }
            trust_sequence_exact = trust_sequence_exact and (
                next_index == trust_index + int(shrink_action)
            )
        elif next_index is not None or next_radius is not None:
            trust_sequence_exact = trust_sequence_exact and (
                isinstance(trust_index, int)
                and isinstance(next_index, int)
                and trust_index <= next_index <= trust_index + 1
                and 0 <= next_index < len(expected_radii)
                and close_float(
                    next_radius, expected_radii[next_index], tolerance=1e-20
                )
            )
    return {
        "iteration_kind_partition_exact_no_unknown_rows": partition_exact,
        "proposal_rows_contiguous_after_bootstrap": proposals_contiguous,
        "fail_before_row_position_exact": fail_position_exact,
        "plateau_state_monotone_and_single_use": plateau_sequence_exact
        and plateau_use_count <= 1,
        "trust_radius_nonexpanding_cross_row_sequence": trust_sequence_exact,
    }


def cw16_terminal_semantics_self_test() -> dict[str, Any]:
    """Exercise positive witnesses and fail-closed mutations without CUDA."""

    digest = "a" * 64
    bootstrap_acceptance = {
        **{f"bootstrap_true_{index}": True for index in range(7)},
        "active_cut_violations_zero": False,
        "forensic_favorable_retention_pass": False,
        "new_harm_count_zero": False,
    }
    bootstrap = {
        "iteration": 1,
        "kind": "exact_CW15_iteration1_bootstrap_seed",
        "bootstrap_seed_nonacceptance_exempt_once": True,
        "eligible": False,
        "is_acceptance": False,
        "counts_toward_official_budget": True,
        "official_evaluation_count_after": 1,
        "additional_float64_le_sha256": CW15_ITERATION1_ADDITIONAL_SHA256,
        "model_state_sha256": CW15_ITERATION1_MODEL_SHA256,
        "output_state_fingerprint": {"sha256": CW15_ITERATION1_OUTPUT_SHA256},
        "total_from_raw_float64_le_sha256": "b" * 64,
        "additional_l2": CW15_ITERATION1_ADDITIONAL_L2,
        "replay_checks_pre_oracle": {"pass": True},
        "replay_checks_post_oracle": {"pass": True},
        "post_oracle_cut_merge": {
            "active_count": 50,
            "added": [{} for _ in range(12)],
            "strengthened": [],
        },
        "active_cut_gates": {"count": 50, "records": [{} for _ in range(50)]},
        "CW11_phi_on_expanded_ledger": CW11_PHI_ON_EXPANDED_50,
        "bootstrap_phi_on_expanded_ledger": CW15_ITERATION1_PHI_ON_EXPANDED_50,
        "historical_candidate_tangent_archive_count": 12,
        "historical_candidate_tangent_active_count": 0,
        "post_bootstrap_anchor_growth": {
            "pass": True,
            "checks": {"pass": True},
        },
        "hard_anchor50_selection": {"pass": True, "checks": {"pass": True}},
        "hard_anchor50_only_audit": {
            "pass": True,
            "checks": {"pass": True},
            "selected_anchor_row_count": 50,
            "selected_anchor_kind": "CW11_anchor",
            "historical_candidate_active_count": 0,
            "historical_candidate_archive_count": 12,
            "anchor_only_l2": CW14_ANCHOR50_ONLY_L2,
            "anchor_only_point_sha256": CW14_ANCHOR50_ONLY_POINT_SHA256,
            "consumed_combined62_uncapped_l2": CW15_ITERATION2_UNCAPPED_L2,
            "anchor_array_audit": {
                "pass": True,
                "checks": {"pass": True},
            },
        },
        "integrity": {"pass": True},
        "acceptance": {"pass": False, "checks": bootstrap_acceptance},
        "acceptance_false_keys": [
            "active_cut_violations_zero",
            "forensic_favorable_retention_pass",
            "new_harm_count_zero",
        ],
    }

    def gate_ledger(residual: float) -> dict[str, Any]:
        return {
            "count": 1,
            "violated_count": 0,
            "residual_min": residual,
            "pass": True,
            "records": [{"residual": residual, "pass": True}],
        }

    quiescence_keys = (
        "terminal_component_gates_pass",
        "deterministic_addition_count_zero",
        "semantic_merge_added_zero",
        "semantic_merge_strengthened_zero",
        "semantic_changed_false",
        "proposal_and_post_merge_anchor_count_equal",
        "proposal_and_post_merge_anchor_ledger_sha_equal",
    )
    mappings = [
        {
            "case": f"partial{count}",
            "semantic_count": count,
            "retained_optimization_count": count,
            "exact_four_to_one": False,
        }
        for count in (1, 2, 3)
    ]
    mappings.extend(
        [
            {
                "case": "exact4",
                "semantic_count": 4,
                "retained_optimization_count": 1,
                "exact_four_to_one": True,
                "exact_four_checks": {
                    "threshold": True,
                    "margin": True,
                    "gradient": True,
                },
            },
            {
                "case": "nonidentical4",
                "semantic_count": 4,
                "retained_optimization_count": 4,
                "exact_four_to_one": False,
            },
        ]
    )
    proposal = {
        "iteration": 2,
        "kind": "CW16_unique_local_trust_official_proposal",
        "bootstrap_seed_nonacceptance_exempt_once": False,
        "official_evaluation_count_after": 2,
        "additional_float64_le_sha256": digest,
        "model_state_sha256": "b" * 64,
        "center_model_state_sha256": "0" * 64,
        "total_from_raw_float64_le_sha256": "c" * 64,
        "output_state_fingerprint": {"sha256": "d" * 64},
        "center_output_state_sha256": "0" * 64,
        "additional_l2": 0.00095,
        "trust_radius_index": 0,
        "trust_radius_divisor": 8,
        "trust_radius": 0.000125,
        "single_evaluation_checks": {"pass": True},
        "pre_oracle_repeat_checks": {"pass": True},
        "hard_anchor_selection": {
            "pass": True,
            "checks": {"pass": True},
            "active_anchor_count": 50,
            "active_anchor_ledger_sha256": digest,
        },
        "hard_anchor_array_audit": {"pass": True, "checks": {"pass": True}},
        "proposal_subproblem_anchor_ledger": {
            "row_count": 50,
            "sanitized_ledger_sha256": digest,
            "linearization_kind": "CW11_anchor",
            "used_in_this_proposal_subproblem": True,
        },
        "current_local_audit": {
            "pass": True,
            "checks": {"pass": True},
            "deduplication": {
                "pass": True,
                "checks": {
                    "semantic_mapping_bijective": True,
                    "only_exact_four_groups_are_deduplicated": True,
                },
                "mappings": mappings,
            },
        },
        "subproblem": {
            "pass": True,
            "checks": {"pass": True},
            "predicted_phi": 0.5,
            "scaling": {
                "pass": True,
                "checks": {"pass": True},
                "raw_problem_direct_gates_after_inverse_scaling": True,
            },
            "stage1": {
                "success": True,
                "raw_certification_checks": {"pass": True},
                "z_certification_adjustment": 1e-9,
            },
            "stage2": {"success": True},
            "clipped_vector_applied": False,
            "historical_candidate_active_count": 0,
            "trust_radius": 0.000125,
        },
        "historical_candidate_tangent_archive_count": 12,
        "historical_candidate_tangent_active_count": 0,
        "semantic_monotone_checks": {"pass": True},
        "active_cut_gates": gate_ledger(1.0),
        "cached_current_active_cut_gates_on_expanded_ledger": gate_ledger(0.0),
        "exact_merit": {
            "definition": "Phi=min post-merge semantic residual",
            "current_on_expanded_ledger": 0.0,
            "candidate_on_expanded_ledger": 1.0,
            "predicted_phi": 0.5,
            "strict_comparison_without_tolerance": True,
        },
        "transition": {
            "action": "accept_strict_phi_improvement",
            "strict_actual_phi_improvement": True,
            "strict_predicted_phi_improvement": True,
            "changed_model": True,
            "same_BF16_output": False,
            "semantic_changed": False,
            "plateau_already_used": False,
            "plateau_exception_allowed": False,
        },
        "second_same_BF16_plateau": False,
        "eligible": True,
        "acceptance": {
            "pass": True,
            "checks": {f"terminal_gate_{index}": True for index in range(10)},
        },
        "post_merge_hard_anchor_growth": {
            "pass": True,
            "checks": {"pass": True},
        },
        "post_merge_current_anchor_selection": {
            "pass": True,
            "checks": {"pass": True},
            "active_anchor_count": 50,
            "active_anchor_ledger_sha256": digest,
        },
        "post_merge_current_anchor_ledger": {
            "row_count": 50,
            "sanitized_ledger_sha256": digest,
            "linearization_kind": "CW11_anchor",
            "used_in_already_evaluated_proposal_subproblem": False,
        },
        "terminal_semantic_quiescence": {
            key: True for key in quiescence_keys
        },
        "integrity": {"pass": True},
        "oracle": {
            "hard_checks": {"pass": True},
            "authoritative_60_gates": {
                "pass": True,
                "gate_count": 60,
                "records": [{} for _ in range(60)],
            },
            "new_harm_count_vs_cw11": 0,
            "transition_state_checks": {"favorable_transitions_retained": True},
        },
        "fixed_repair_gates": {"pass": True},
        "legacy_gate": {"pass": True},
        "post_oracle_cut_merge": {"added": [], "strengthened": []},
        "deterministic_addition_count": 0,
        "local_tangent_archived_after_subproblem": {
            "active_in_future_subproblem": False,
            "sanitized_ledger_sha256": "e" * 64,
        },
    }

    bootstrap_positive = cw16_bootstrap_checks(bootstrap)
    proposal_positive = cw16_proposal_checks(proposal)
    sequence_positive = cw16_proposal_sequence_checks(
        [{"iteration": 0}, bootstrap, proposal],
        [bootstrap],
        [proposal],
        [],
    )

    def mutation_rejected(
        source: Mapping[str, Any], path: tuple[Any, ...], replacement: Any
    ) -> bool:
        mutated = copy.deepcopy(source)
        cursor: Any = mutated
        for component in path[:-1]:
            cursor = cursor[component]
        cursor[path[-1]] = replacement
        validator = (
            cw16_bootstrap_checks
            if source is bootstrap
            else cw16_proposal_checks
        )
        return not all(validator(mutated).values())

    mutation_rejections = {
        "bootstrap_eligible_true": mutation_rejected(
            bootstrap, ("eligible",), True
        ),
        "bootstrap_budget_zero": mutation_rejected(
            bootstrap, ("official_evaluation_count_after",), 0
        ),
        "bootstrap_anchor50_replaced_by_combined62_l2": mutation_rejected(
            bootstrap,
            ("hard_anchor50_only_audit", "anchor_only_l2"),
            CW15_ITERATION2_UNCAPPED_L2,
        ),
        "proposal_budget_mismatch": mutation_rejected(
            proposal, ("official_evaluation_count_after",), 3
        ),
        "strict_Phi_detached_from_candidate_ledger": mutation_rejected(
            proposal, ("exact_merit", "candidate_on_expanded_ledger"), 2.0
        ),
        "candidate_residual_min_not_recomputed": mutation_rejected(
            proposal, ("active_cut_gates", "residual_min"), 2.0
        ),
        "eligibility_bit_false": mutation_rejected(
            proposal, ("eligible",), False
        ),
        "QP_anchor_claim_false": mutation_rejected(
            proposal,
            ("proposal_subproblem_anchor_ledger", "used_in_this_proposal_subproblem"),
            False,
        ),
        "postmerge_anchor_claimed_as_used_in_QP": mutation_rejected(
            proposal,
            (
                "post_merge_current_anchor_ledger",
                "used_in_already_evaluated_proposal_subproblem",
            ),
            True,
        ),
        "postmerge_anchor_SHA_drift": mutation_rejected(
            proposal,
            ("post_merge_current_anchor_ledger", "sanitized_ledger_sha256"),
            "f" * 64,
        ),
        "dedup_partial2_collapsed": mutation_rejected(
            proposal,
            (
                "current_local_audit",
                "deduplication",
                "mappings",
                1,
                "retained_optimization_count",
            ),
            1,
        ),
        "dedup_nonidentical4_collapsed": mutation_rejected(
            proposal,
            (
                "current_local_audit",
                "deduplication",
                "mappings",
                4,
                "retained_optimization_count",
            ),
            1,
        ),
    }
    for key in quiescence_keys:
        mutation_rejections[f"quiescence_{key}_false"] = mutation_rejected(
            proposal, ("terminal_semantic_quiescence", key), False
        )

    unknown = {"iteration": 3, "kind": "unknown_unvalidated_row"}
    unknown_checks = cw16_proposal_sequence_checks(
        [{"iteration": 0}, bootstrap, proposal, unknown],
        [bootstrap],
        [proposal],
        [],
    )
    trust_first = copy.deepcopy(proposal)
    trust_second = copy.deepcopy(proposal)
    trust_first["trust_radius_index"] = 1
    trust_first["next_trust_radius_index"] = 0
    trust_first["next_trust_radius"] = 0.000125
    trust_second["iteration"] = 3
    trust_second["trust_radius_index"] = 0
    trust_checks = cw16_proposal_sequence_checks(
        [{"iteration": 0}, bootstrap, trust_first, trust_second],
        [bootstrap],
        [trust_first, trust_second],
        [],
    )
    plateau_first = copy.deepcopy(proposal)
    plateau_second = copy.deepcopy(proposal)
    plateau_first["next_trust_radius_index"] = 1
    plateau_first["next_trust_radius"] = 0.0000625
    plateau_first["transition"].update(
        {
            "action": "move_once_same_BF16_plateau_and_shrink",
            "plateau_already_used": False,
            "plateau_exception_allowed": True,
        }
    )
    plateau_second["iteration"] = 3
    plateau_second["trust_radius_index"] = 1
    plateau_second["transition"].update(
        {
            "action": "move_once_same_BF16_plateau_and_shrink",
            "plateau_already_used": False,
            "plateau_exception_allowed": True,
        }
    )
    plateau_checks = cw16_proposal_sequence_checks(
        [{"iteration": 0}, bootstrap, plateau_first, plateau_second],
        [bootstrap],
        [plateau_first, plateau_second],
        [],
    )
    sequence_mutation_rejections = {
        "unknown_iteration_kind": unknown_checks.get(
            "iteration_kind_partition_exact_no_unknown_rows"
        )
        is False,
        "trust_radius_expansion": trust_checks.get(
            "trust_radius_nonexpanding_cross_row_sequence"
        )
        is False,
        "plateau_exception_twice": plateau_checks.get(
            "plateau_state_monotone_and_single_use"
        )
        is False,
    }
    checks = {
        "bootstrap_positive_all_true": engine.all_true_checks(bootstrap_positive),
        "proposal_positive_all_true": engine.all_true_checks(proposal_positive),
        "sequence_positive_all_true": engine.all_true_checks(sequence_positive),
        "all_field_mutations_rejected": all(mutation_rejections.values()),
        "all_sequence_mutations_rejected": all(
            sequence_mutation_rejections.values()
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(
            "CW16 terminal semantics self-test failed: "
            f"checks={checks}; field={mutation_rejections}; "
            f"sequence={sequence_mutation_rejections}"
        )
    return {
        "checks": checks,
        "bootstrap_positive": bootstrap_positive,
        "proposal_positive": proposal_positive,
        "sequence_positive": sequence_positive,
        "mutation_rejections": mutation_rejections,
        "sequence_mutation_rejections": sequence_mutation_rejections,
        "pass": True,
    }


def cw16_validate_terminal_payload(payload: bytes) -> dict[str, Any]:
    """Deep-gate a lossless CW16 terminal stdout before exclusive publication."""

    document = engine.strict_json_object(payload, "CW16 solver terminal stdout")
    terminal = document.get("terminal")
    second = document.get("second_stage")
    final_integrity = document.get("final_integrity")
    frozen = document.get("frozen_inputs")
    input_lock = document.get("input_lock")
    runtime = document.get("runtime")
    source_audit = document.get("source_audit")
    consumed = document.get("cw15_consumed_closure_evidence")
    if not isinstance(terminal, Mapping) or not isinstance(second, Mapping):
        raise ProtocolError("CW16 terminal/second-stage mapping absent")
    decision = terminal.get("decision")
    iterations = second.get("iterations")
    if not isinstance(decision, Mapping) or not isinstance(iterations, list):
        raise ProtocolError("CW16 decision/iteration ledger absent")
    compat = copy.deepcopy(document)
    if document.get("status") == EXPECTED_TERMINAL_STATUSES[0]:
        compat["terminal"]["decision"]["close_reason"] = (
            "first_feasible_fixed_iteration_candidate"
        )
        compat["second_stage"]["decision"]["close_reason"] = (
            "first_feasible_fixed_iteration_candidate"
        )
    parent_validation = _PARENT_TERMINAL_VALIDATOR(engine.canonical_json(compat))
    bootstrap_rows = [
        value
        for value in iterations
        if isinstance(value, Mapping)
        and value.get("bootstrap_seed_nonacceptance_exempt_once") is True
    ]
    proposal_rows = [
        value
        for value in iterations
        if isinstance(value, Mapping)
        and value.get("kind") == "CW16_unique_local_trust_official_proposal"
    ]
    fail_before_rows = [
        value
        for value in iterations
        if isinstance(value, Mapping)
        and value.get("kind") == "fail_closed_before_official_proposal"
    ]
    sequence_checks = cw16_proposal_sequence_checks(
        iterations,
        bootstrap_rows,
        proposal_rows,
        fail_before_rows,
    )
    self_lock = input_lock.get("self") if isinstance(input_lock, Mapping) else None
    closure_lock = (
        input_lock.get("consumed_CW15_closure")
        if isinstance(input_lock, Mapping)
        else None
    )
    parent_lock = (
        input_lock.get("CW15_parent_source")
        if isinstance(input_lock, Mapping)
        else None
    )
    archive = second.get("optimization_affine_closure_ledger")
    archive_sha = second.get("optimization_affine_closure_ledger_sha256")
    archive_current_selection = (
        archive.get("current_active_CW11_anchor_selection")
        if isinstance(archive, Mapping)
        else None
    )
    common_checks = {
        "top_level_keys_exact": set(document) == set(EXPECTED_TERMINAL_TOP_KEYS)
        and len(document) == len(EXPECTED_TERMINAL_TOP_KEYS),
        "parent_deep_validator_passed_original_except_reason_adapter": all(
            engine.all_true_checks(parent_validation.get(name))
            for name in ("checks", "cross_link_checks", "terminal_checks")
        ),
        "runtime_cuda_all_true": isinstance(runtime, Mapping)
        and runtime.get("cuda_required") is True
        and engine.all_true_checks(runtime.get("checks"))
        and isinstance(runtime.get("cuda"), Mapping)
        and engine.all_true_checks(runtime.get("cuda", {}).get("checks")),
        "source_audit_all_true_threshold_exact": isinstance(source_audit, Mapping)
        and source_audit.get("pass") is True
        and engine.all_true_checks(source_audit.get("checks"))
        and source_audit.get("threshold_function_shas")
        == EXPECTED_THRESHOLD_FUNCTION_SHAS,
        "frozen_inputs_exact40": isinstance(frozen, Mapping)
        and frozen.get("all_exact") is True
        and frozen.get("binding_count") == 40
        and isinstance(frozen.get("records"), list)
        and len(frozen["records"]) == 40,
        "consumed_CW15_closure_all_true": isinstance(consumed, Mapping)
        and consumed.get("pass") is True
        and engine.all_true_checks(consumed.get("checks")),
        "math_selftests_all_true": all(
            isinstance(document.get(name), Mapping)
            and document[name].get("pass") is True
            and engine.all_true_checks(document[name].get("checks"))
            for name in (
                "local_trust_math_self_test",
                "curvature_and_negative_self_test",
                "physical_dedup_self_test",
            )
        ),
        "input_lock_self_exact_CW16": isinstance(self_lock, Mapping)
        and self_lock.get("path") == root_relative(SOLVER)
        and self_lock.get("sha256") == EXPECTED_SOLVER_SHA256
        and self_lock.get("mode_octal") == "0555"
        and self_lock.get("single_link_regular_held_fd_identity_exact") is True
        and isinstance(self_lock.get("bytes"), int)
        and self_lock["bytes"] > 0
        and engine.all_true_checks(
            input_lock.get("CW16_exact_self_binding_checks")
        ),
        "input_lock_parent_CW15_exact": isinstance(parent_lock, Mapping)
        and parent_lock.get("path")
        == "tools/probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py"
        and parent_lock.get("sha256") == CW15_SOLVER_SHA256
        and parent_lock.get("mode") == "0555"
        and parent_lock.get("nlink") == 1
        and engine.all_true_checks(parent_lock.get("checks")),
        "input_lock_consumed_closure_exact": isinstance(closure_lock, Mapping)
        and closure_lock.get("attempt_marker_sha256")
        == CW15_ATTEMPT_MARKER_SHA256
        and closure_lock.get("closed_stdout_sha256")
        == CW15_CLOSED_STDOUT_SHA256
        and closure_lock.get("stderr_audit_sha256") == CW15_STDERR_AUDIT_SHA256
        and closure_lock.get("parent_solver_sha256") == CW15_SOLVER_SHA256,
        "iteration_ordinals_sequential": len(iterations) >= 2
        and [value.get("iteration") for value in iterations]
        == list(range(len(iterations))),
        "iteration0_exact_CW11_reference": iterations[0].get("kind")
        == "exact_CW11_reference_before_bootstrap"
        and iterations[0].get("model_state_sha256") == engine.CW11_MODEL_SHA256
        and iterations[0].get("additional_l2") == 0.0
        and iterations[0].get("additional_float64_le_sha256")
        == "87b4dec5267ce5343f75acfae5807f1f91192a49192703bc2b3f70677b42b31b"
        and iterations[0].get("total_from_raw_float64_le_sha256")
        == "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
        and iterations[0].get("active_cut_count") == 38
        and iterations[0].get("active_cut_identity_sha256")
        == CW15_INITIAL_ACTIVE_IDENTITY_SHA256,
        "bootstrap_unique_exact": len(bootstrap_rows) == 1
        and bootstrap_rows[0] is iterations[1]
        and all(cw16_bootstrap_checks(bootstrap_rows[0]).values()),
        "proposal_rows_all_exact": all(
            all(cw16_proposal_checks(value).values()) for value in proposal_rows
        ),
        "iteration_partition_trust_and_plateau_sequence_exact": (
            engine.all_true_checks(sequence_checks)
        ),
        "fail_before_at_most_one_and_last": len(fail_before_rows) <= 1
        and (not fail_before_rows or fail_before_rows[0] is iterations[-1]),
        "official_budget_crosslink_exact": decision.get(
            "official_candidate_evaluation_count"
        )
        == 1 + len(proposal_rows)
        and decision.get("max_official_candidate_evaluations") == 12
        and 1 <= decision.get("official_candidate_evaluation_count", 0) <= 12
        and decision.get("bootstrap_evaluation_count") == 1
        and decision.get("bootstrap_seed_nonacceptance_exempt_once") is True
        and decision.get("bootstrap_eligible") is False,
        "decision_scope_exact": decision.get("semantic_ledger_monotone") is True
        and decision.get("CW11_anchor_always_hard") is True
        and decision.get("terminal_active_CW11_anchor_count")
        == decision.get("terminal_active_cut_count")
        and engine.is_sha256(
            decision.get("terminal_active_CW11_anchor_ledger_sha256")
        )
        and isinstance(decision.get("full_CW11_anchor_archive_count"), int)
        and decision.get("full_CW11_anchor_archive_count")
        >= decision.get("terminal_active_CW11_anchor_count", 0)
        and decision.get("historical_candidate_tangent_active_count") == 0
        and decision.get("no_clip") is True
        and decision.get("candidate_consumer_called") is False
        and decision.get("eligible_as_promotion_evidence") is False,
        "final_integrity_all_true": isinstance(final_integrity, Mapping)
        and final_integrity.get("pass") is True
        and engine.all_true_checks(final_integrity.get("checks"))
        and final_integrity.get("model_left_raw_after_outer_finally") is True,
        "archive_hash_and_history_separation_exact": isinstance(archive, Mapping)
        and archive_sha == sha256_bytes(engine.canonical_json(archive))
        and isinstance(archive.get("CW11_anchor_archive"), list)
        and len(archive["CW11_anchor_archive"]) >= 50
        and isinstance(
            archive.get("historical_CW15_candidate_tangent_archive"), list
        )
        and len(archive["historical_CW15_candidate_tangent_archive"]) == 12
        and isinstance(archive_current_selection, Mapping)
        and archive_current_selection.get("row_count")
        == decision.get("terminal_active_CW11_anchor_count")
        and archive_current_selection.get("sanitized_ledger_sha256")
        == decision.get("terminal_active_CW11_anchor_ledger_sha256")
        and archive_current_selection.get("archive_row_count")
        == decision.get("full_CW11_anchor_archive_count")
        == len(archive["CW11_anchor_archive"])
        and archive_current_selection.get("selection_audit", {}).get("pass")
        is True
        and engine.all_true_checks(
            archive_current_selection.get("selection_audit", {}).get("checks")
        )
        and archive.get("historical_candidate_active_count") == 0
        and isinstance(archive.get("expired_current_local_tangent_archives"), list)
        and all(
            value.get("active_in_future_subproblem") is False
            and engine.is_sha256(value.get("sanitized_ledger_sha256"))
            for value in archive["expired_current_local_tangent_archives"]
        ),
        "threshold_shas_top_exact": document.get("threshold_function_shas")
        == EXPECTED_THRESHOLD_FUNCTION_SHAS,
        "cw11_reconstruction_exact": document.get(
            "cw11_reconstruction_summary", {}
        ).get("model_state_sha256")
        == engine.CW11_MODEL_SHA256,
        "terminal_optimization_semantics_crosslink": terminal.get(
            "optimization_affine_tangent_ledger_semantics"
        )
        == second.get("terminal_optimization_affine_ledger_semantics"),
    }
    success = document.get("status") == EXPECTED_TERMINAL_STATUSES[0]
    eligible_rows = [value for value in proposal_rows if value.get("eligible") is True]
    terminal_optimization = terminal.get("optimization_affine_tangent_ledger")
    terminal_optimization_sha = terminal.get(
        "optimization_affine_tangent_ledger_sha256"
    )
    terminal_semantics = terminal.get(
        "optimization_affine_tangent_ledger_semantics"
    )
    if success:
        selected = [
            value
            for value in proposal_rows
            if value.get("iteration") == decision.get("terminal_iteration")
        ]
        selected_row = selected[0] if len(selected) == 1 else {}
        terminal_anchor_rows = (
            [
                value
                for value in terminal_optimization
                if isinstance(value, Mapping)
                and value.get("linearization_kind") == "CW11_anchor"
            ]
            if isinstance(terminal_optimization, list)
            else []
        )
        terminal_local_rows = (
            [
                value
                for value in terminal_optimization
                if isinstance(value, Mapping)
                and value.get("linearization_kind")
                == "CW16_current_local_physical"
            ]
            if isinstance(terminal_optimization, list)
            else []
        )
        branch_checks = {
            "new_success_reason_exact": decision.get("close_reason")
            == "first_feasible_strict_post_merge_Phi_candidate",
            "first_feasible_unique_eligible": len(eligible_rows) == 1
            and len(selected) == 1
            and eligible_rows[0] is selected[0]
            and selected[0] is proposal_rows[-1],
            "terminal_acceptance_and_strict_merit": selected_row.get(
                "acceptance", {}
            ).get("pass")
            is True
            and engine.all_true_checks(
                selected_row.get("acceptance", {}).get("checks")
            )
            and selected_row.get("exact_merit", {}).get(
                "strict_comparison_without_tolerance"
            )
            is True,
            "terminal_semantic_quiescence_exact": isinstance(
                terminal_semantics, Mapping
            )
            and terminal_semantics.get("terminal_semantic_quiescence")
            == selected_row.get("terminal_semantic_quiescence")
            and engine.all_true_checks(
                terminal_semantics.get("terminal_semantic_quiescence")
            )
            and selected_row.get("deterministic_addition_count") == 0
            and not selected_row.get("post_oracle_cut_merge", {}).get("added")
            and not selected_row.get("post_oracle_cut_merge", {}).get(
                "strengthened"
            )
            and selected_row.get("transition", {}).get("semantic_changed")
            is False,
            "terminal_anchor_semantics_distinguish_QP_and_post_merge": isinstance(
                terminal_semantics, Mapping
            )
            and terminal_semantics.get("proposal_subproblem_anchor_ledger")
            == selected_row.get("proposal_subproblem_anchor_ledger")
            and terminal_semantics.get("post_merge_current_anchor_ledger")
            == selected_row.get("post_merge_current_anchor_ledger")
            and terminal_semantics.get(
                "post_merge_new_anchors_are_not_claimed_as_used_in_QP"
            )
            is True
            and terminal_semantics.get("eligible_implies_semantic_changed_false")
            is True,
            "formal_eligibility_true": decision.get(
                "eligible_only_for_formal_fulltrain_revalidation"
            )
            is True,
            "terminal_optimization_ledger_exact": isinstance(
                terminal_optimization, list
            )
            and terminal_optimization
            == second.get("terminal_optimization_affine_ledger")
            and engine.is_sha256(terminal_optimization_sha)
            and terminal_optimization_sha
            == second.get("terminal_optimization_affine_ledger_sha256")
            == sha256_bytes(engine.canonical_json(terminal_optimization)),
            "terminal_optimization_partition_exact": isinstance(
                terminal_semantics, Mapping
            )
            and isinstance(terminal_optimization, list)
            and len(terminal_anchor_rows) + len(terminal_local_rows)
            == len(terminal_optimization)
            and len(terminal_anchor_rows)
            == terminal_semantics.get("proposal_subproblem_anchor_ledger", {}).get(
                "row_count"
            )
            and sha256_bytes(engine.canonical_json(terminal_anchor_rows))
            == terminal_semantics.get("proposal_subproblem_anchor_ledger", {}).get(
                "sanitized_ledger_sha256"
            )
            and len(terminal_local_rows)
            == terminal_semantics.get("proposal_subproblem_local_row_count")
            and sha256_bytes(engine.canonical_json(terminal_local_rows))
            == terminal_semantics.get(
                "proposal_subproblem_local_sanitized_ledger_sha256"
            )
            and terminal_semantics.get(
                "combined_proposal_subproblem_optimization_row_count"
            )
            == len(terminal_optimization)
            and terminal_semantics.get(
                "combined_proposal_subproblem_optimization_ledger_sha256"
            )
            == terminal_optimization_sha,
            "terminal_current_anchor_selection_matches_quiescent_proposal": (
                isinstance(archive_current_selection, Mapping)
                and archive_current_selection.get("row_count")
                == selected_row.get("post_merge_current_anchor_ledger", {}).get(
                    "row_count"
                )
                == selected_row.get("proposal_subproblem_anchor_ledger", {}).get(
                    "row_count"
                )
                and archive_current_selection.get("sanitized_ledger_sha256")
                == selected_row.get("post_merge_current_anchor_ledger", {}).get(
                    "sanitized_ledger_sha256"
                )
                == selected_row.get("proposal_subproblem_anchor_ledger", {}).get(
                    "sanitized_ledger_sha256"
                )
            ),
        }
    else:
        branch_checks = {
            "no_eligible_rows": not eligible_rows,
            "formal_eligibility_false": decision.get(
                "eligible_only_for_formal_fulltrain_revalidation"
            )
            is False,
            "success_reason_absent": decision.get("close_reason")
            != "first_feasible_strict_post_merge_Phi_candidate",
            "terminal_optimization_ledger_absent": terminal_optimization is None
            and terminal_optimization_sha is None
            and terminal_semantics is None
            and second.get("terminal_optimization_affine_ledger") is None
            and second.get("terminal_optimization_affine_ledger_sha256") is None,
        }
    if not all(common_checks.values()) or not all(branch_checks.values()):
        raise ProtocolError(
            f"CW16 solver terminal contract failed: common={common_checks}; "
            f"branch={branch_checks}"
        )
    reloaded = engine.strict_json_object(payload, "CW16 terminal lossless reparse")
    if reloaded != document:
        raise ProtocolError("CW16 terminal payload changed during validation")
    return {
        "checks": common_checks,
        "branch_checks": branch_checks,
        "parent_validation": {
            key: parent_validation[key]
            for key in (
                "checks",
                "cross_link_checks",
                "terminal_checks",
                "reconstruction_audit",
            )
        },
        "bootstrap_checks": cw16_bootstrap_checks(bootstrap_rows[0]),
        "proposal_checks": [cw16_proposal_checks(value) for value in proposal_rows],
        "proposal_sequence_checks": sequence_checks,
        "stdout_sha256": sha256_bytes(payload),
        "stdout_bytes": len(payload),
        "payload": document,
    }


engine.validate_terminal_payload = cw16_validate_terminal_payload


def cw16_preflight(*, require_lock: bool) -> dict[str, Any]:
    evidence = _BASE_PREFLIGHT(require_lock=require_lock)
    evidence["cw16_terminal_semantics_self_test"] = (
        cw16_terminal_semantics_self_test()
    )
    evidence["inherited_parent_launcher"] = PARENT_EVIDENCE
    evidence["inherited_parent_composed_checks"] = PARENT_COMPOSED_CHECKS
    evidence["known_cw12_failure_evidence"] = (
        parent_adapter.parent_adapter.parent_adapter.known_cw12_failure_evidence()
    )
    evidence["known_cw13_terminal_evidence"] = (
        parent_adapter.parent_adapter.known_cw13_terminal_evidence()
    )
    evidence["known_cw14_terminal_evidence"] = (
        parent_adapter.known_cw14_terminal_evidence()
    )
    terminal_evidence = known_cw15_terminal_evidence()
    evidence["known_cw15_terminal_evidence"] = terminal_evidence
    if evidence.get("solver_lock_armed"):
        frozen = evidence.get("solver_static", {}).get("payload", {}).get(
            "frozen_inputs"
        )
        records = frozen.get("records") if isinstance(frozen, Mapping) else None
        if not isinstance(records, list):
            raise ProtocolError("armed CW16 solver did not emit its frozen manifest")
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
                root_relative(CW15_ATTEMPT_MARKER): {
                    "sha256": CW15_ATTEMPT_MARKER_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
                root_relative(CW15_CLOSED_STDOUT): {
                    "sha256": CW15_CLOSED_STDOUT_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
                root_relative(CW15_STDERR_AUDIT): {
                    "sha256": CW15_STDERR_AUDIT_SHA256,
                    "mode": "0444",
                    "expected_mode": "0444",
                    "nlink": 1,
                    "regular": True,
                },
            }
        )
        checks = {
            "solver_manifest_count_exact_40": frozen.get("binding_count") == 40
            and len(records) == 40
            and len(expected) == 40,
            "old37_plus_cw15_three_path_set_exact": set(observed) == set(expected),
            "all_40_record_identities_exact": all(
                path in observed
                and all(
                    observed[path].get(key) == value
                    for key, value in identity.items()
                )
                for path, identity in expected.items()
            ),
        }
        if not all(checks.values()):
            raise ProtocolError(f"CW16 exact 37+3 manifest gate failed: {checks}")
        evidence["cw16_manifest_37_plus_3_bindings"] = {
            "checks": checks,
            "cw15_terminal_records": {
                path: observed[path]
                for path in sorted(
                    {
                        root_relative(CW15_ATTEMPT_MARKER),
                        root_relative(CW15_CLOSED_STDOUT),
                        root_relative(CW15_STDERR_AUDIT),
                    }
                )
            },
            "pass": True,
        }
    return evidence


def cw16_stderr_audit(
    payload: bytes,
    *,
    post_child_integrity: Mapping[str, Any],
) -> bytes:
    inherited = _PARENT_STDERR_AUDIT(
        payload,
        post_child_integrity=post_child_integrity,
    )
    document = engine.strict_json_object(inherited, "inherited CW15 stderr audit")
    if document.get("schema_version") != (
        "ptcg-cw15-consumed-valid-official6-one-shot-stderr-audit-v1"
    ):
        raise ProtocolError("inherited CW15 stderr audit schema drift")
    prior_parent = document.get("inherited_parent_launcher")
    prior_grandparent = document.get("inherited_grandparent_launcher")
    document["schema_version"] = STDERR_SCHEMA
    document["inherited_greatgrandparent_launcher"] = prior_grandparent
    document["inherited_grandparent_launcher"] = prior_parent
    document["inherited_parent_launcher"] = PARENT_EVIDENCE
    return engine.canonical_json(document)


engine.preflight = cw16_preflight
engine.validated_stderr_audit = cw16_stderr_audit


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

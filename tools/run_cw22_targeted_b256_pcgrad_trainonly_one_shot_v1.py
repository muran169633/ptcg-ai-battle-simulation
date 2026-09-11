#!/usr/bin/env python3
"""One-shot launcher for the frozen CW22 targeted B256 train-only probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "run_cw22_targeted_b256_pcgrad_trainonly_one_shot_v1.py"
SOLVER = TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py"
BASE = TOOLS / "run_cw20_raw_fixed_pcgrad_trainonly_one_shot_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-cw22-targeted-b256-pcgrad-one-shot-launcher-v1"
CHILD_SCHEMA = "ptcg-u468-cw11-targeted-b256-pcgrad-specialbc-cw22-v1"

ATTEMPT = ROOT / "artifacts/.ptcg-cw22_targeted_b256_pcgrad_trainonly_20260803_v1-attempt.json"
CHILD_STDOUT = ROOT / "artifacts/cw22_targeted_b256_pcgrad_trainonly_20260803_v1.stdout.json"
CHILD_STDERR = ROOT / "artifacts/cw22_targeted_b256_pcgrad_trainonly_20260803_v1.stderr.txt"
EXECUTION = ROOT / "artifacts/cw22_targeted_b256_pcgrad_trainonly_20260803_v1.execution.json"
OUTPUT = ROOT / "artifacts/cw22_cw11_targeted_b256_pcgrad_specialbc_trainonly_20260803_v1.json"

FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444
SOLVER_SHA256 = "f307f3f26355125f97c857f14b97a73d9a7a4f2ea39ca8b0a578e801055e82b4"
BASE_SHA256 = "ab05db655b9c375929f4aa6219cc693ac6b2e11b89fbec173370e2974dd004d6"
SELECTION_PAYLOAD_SHA256 = "c4a9b711d8f5636daea8a428b9fd5e506bffcfdaeb614c7faedf7c75e4165a21"
EXPECTED_CACHE_SHA256 = "a936e106c6ef0e02895046ab70fa9480a9426f2b371d11c6d25ed1447771519b"
CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
RAW_NONACTOR_SHA256 = "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"

DIRECT_INPUTS: dict[Path, tuple[str, int]] = {
    SOLVER: (SOLVER_SHA256, 0o555),
    BASE: (BASE_SHA256, 0o555),
    TOOLS / "select_cw22_targeted_b256_from_cw11_profile_v2.py": (
        "fe3ade2f38ea9e2972976654ffd584e85cc17183a7c1787c444261e515dd6c30",
        0o555,
    ),
    ROOT / "artifacts/cw22_targeted_b256_selection_v2_20260803.json": (
        "035f071902c3f43acd5cec328400d6172a990bd19864d2d794cb2cdff5f7bee7",
        0o444,
    ),
    ROOT / "artifacts/u468_cw11_full_train_margin_profile_v1_20260803.json": (
        "da684858d2c459416234cdc3cae1ffc32a5ec23a1ec376f66a0cf440776a846e",
        0o444,
    ),
    TOOLS / "probe_u468_cw11_fixed_pcgrad_specialbc_cw21_v1.py": (
        "9c18db25e16fe6b9feb1ca72d0682a1baa31ad6dc19e1a6283fd7db2da0f9cf4",
        0o555,
    ),
    TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py": (
        "8f5b64c8a04b3de9a6d2890aac582967a50f4d708f3a11a64cbd395423d907d2",
        0o555,
    ),
    TOOLS / "probe_u468_cw11_bootstrap_pcgrad_specialbc_cw19_v1.py": (
        "65f16009641481cd13538714520828640d00aed42aae920e24d9b70910423ae8",
        0o555,
    ),
    TOOLS / "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py": (
        "2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24",
        0o555,
    ),
    TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_cw4_cuttingplane_v1.py": (
        "40715927549457b161fc907757279e4d85466ded1d1e6bfed62f67c13b529e7c",
        0o555,
    ),
    ROOT
    / "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt": (
        "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f",
        0o664,
    ),
    ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip": (
        "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
        0o664,
    ),
    ROOT / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip": (
        "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
        0o664,
    ),
    ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip": (
        "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
        0o600,
    ),
}

ALLOWED_DECISIONS = {
    "NO_GO_CW22_TARGETED_B256_BASELINE",
    "NO_GO_CW22_TARGETED_B256_DIRECTION",
    "NO_GO_CW22_TARGETED_B256_PREFLIGHT",
    "GO_CW22_TARGETED_B256_PREFLIGHT",
}


class LauncherError(RuntimeError):
    """Fail-closed CW22 launcher error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_engine() -> tuple[ModuleType, dict[str, Any]]:
    before = BASE.lstat()
    source = BASE.read_bytes()
    after = BASE.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == FROZEN_MODE,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": hashlib.sha256(source).hexdigest() == BASE_SHA256,
    }
    if not all(checks.values()):
        raise LauncherError(f"frozen launcher engine drift: {checks}")
    name = "cw22_frozen_memfd_launcher_engine"
    if name in sys.modules:
        raise LauncherError("launcher engine module name already occupied")
    module = types.ModuleType(name)
    module.__file__ = str(BASE)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, str(BASE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module.SOLVER = SOLVER
    module.SOLVER_SHA256 = SOLVER_SHA256
    module.DIRECT_INPUTS = DIRECT_INPUTS
    return module, {
        "path": str(BASE.relative_to(ROOT)),
        "sha256": BASE_SHA256,
        "bytes": len(source),
        "mode_octal": "0555",
        "checks": checks,
    }


def input_snapshot(engine: ModuleType) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(ROOT)): engine.regular_evidence(path, digest, mode)
        for path, (digest, mode) in DIRECT_INPUTS.items()
    }


def strict_json(path: Path, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LauncherError(f"{label}: duplicate key {key}")
            result[key] = value
        return result

    def reject(value: str) -> Any:
        raise LauncherError(f"{label}: nonfinite constant {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise LauncherError(f"{label}: root is not an object")
    return value


def all_true(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        item is True for item in value.values()
    )


def validate_child(summary: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, Any]:
    decision = document.get("decision")
    endpoint = document.get("endpoint")
    selection = document.get("selection")
    historical = document.get("historical_exact_CW11_replay")
    audit = document.get("audit")
    base = document.get("base")
    if not all(
        isinstance(value, Mapping)
        for value in (endpoint, selection, historical, audit, base)
    ):
        raise LauncherError("child artifact structural root drift")
    cache = endpoint.get("cache")
    source = audit.get("source")
    runtime = audit.get("runtime")
    outer_checks = endpoint.get("outer_restore_checks")
    selection_checks = selection.get("checks")
    historical_checks = historical.get("checks")
    context_checks = endpoint.get("context_checks")
    dependency_checks = endpoint.get("dependency_checks")
    output_record = summary.get("output")
    checks = {
        "schemas_exact": document.get("schema_version") == CHILD_SCHEMA
        and summary.get("schema_version") == CHILD_SCHEMA,
        "decision_allowed_consistent": decision in ALLOWED_DECISIONS
        and document.get("status") == decision
        and summary.get("status") == decision
        and endpoint.get("decision") == decision,
        "summary_reason_exact": summary.get("reason") == endpoint.get("reason"),
        "summary_output_exact": isinstance(output_record, Mapping)
        and output_record.get("path") == str(OUTPUT.relative_to(ROOT))
        and output_record.get("sha256") == sha256_file(OUTPUT)
        and output_record.get("mode_octal") == "0444"
        and all_true(output_record.get("checks")),
        "summary_budget_zero": summary.get(
            "official_unique_changed_candidate_count_consumed"
        )
        == 0
        and summary.get("new_validation_rows_opened") == 0
        and summary.get("submission_performed") is False,
        "source_exact": isinstance(source, Mapping)
        and source.get("sha256") == SOLVER_SHA256
        and source.get("pass") is True
        and all_true(source.get("checks")),
        "runtime_exact": isinstance(runtime, Mapping)
        and runtime.get("pass") is True
        and runtime.get("python") == str(EXPECTED_PYTHON.resolve())
        and all_true(runtime.get("checks")),
        "base_exact": base.get("model_state_sha256") == CW11_MODEL_SHA256
        and base.get("raw_nonactor_sha256") == RAW_NONACTOR_SHA256
        and base.get("materialized_eval_only_CW11_opened") is False
        and base.get("resume_forbidden_respected") is True,
        "selection_exact": selection.get("sha256") == SELECTION_PAYLOAD_SHA256
        and selection.get("rows") == 256
        and all_true(selection_checks)
        and selection.get("scope", {}).get("validation_or_test_rows_opened") == 0,
        "historical_replay_disclosed": historical.get("pass") is True
        and all_true(historical_checks)
        and historical.get("historical_specialist_valid_rows_replayed") == 4
        and historical.get("validation_replayed_for_historical_reconstruction")
        is True
        and historical.get("new_validation_rows_opened_for_CW22_selection_or_candidate")
        == 0
        and historical.get("promotion_evidence") is False,
        "outer_restore_exact": endpoint.get("outer_restore_pass") is True
        and all_true(outer_checks),
        "cache_exact": isinstance(cache, Mapping)
        and cache.get("cache_sha256") == EXPECTED_CACHE_SHA256
        and cache.get("non_train_members_opened") is False
        and all_true(cache.get("checks")),
        "context_and_dependencies_exact": all_true(context_checks)
        and all_true(dependency_checks),
        "official_and_external_zero": document.get(
            "official_unique_changed_candidate_count_consumed"
        )
        == 0
        and document.get("cumulative_official_unique_changed_candidate_count") == 1
        and document.get("submission_performed") is False
        and document.get("package_upload_performed") is False,
        "json_safety_audited": audit.get("finite_nested_and_json_serializable")
        is True,
        "shadow_count_bounded": type(
            endpoint.get("changed_candidate_train_shadow_count")
        )
        is int
        and 0 <= endpoint.get("changed_candidate_train_shadow_count") <= 4,
    }
    candidate_payload = endpoint.get("candidate_payload")
    if decision == "GO_CW22_TARGETED_B256_PREFLIGHT":
        selected = endpoint.get("selected_trial")
        reconstruction = endpoint.get("pure_payload_reconstruction")
        checks["GO_semantics"] = (
            endpoint.get("reason")
            == "SMALLEST_PREREGISTERED_RADIUS_PASSED_ALL_TARGETED_TRAIN_GATES"
            and isinstance(candidate_payload, Mapping)
            and isinstance(selected, Mapping)
            and selected.get("pass") is True
            and selected.get("train_gate", {}).get("pass") is True
            and all_true(selected.get("train_gate", {}).get("checks"))
            and all_true(selected.get("integrity"))
            and all_true(endpoint.get("baseline_checks"))
            and all_true(endpoint.get("final_checks"))
            and isinstance(reconstruction, Mapping)
            and reconstruction.get("pass") is True
            and all_true(reconstruction.get("checks"))
            and candidate_payload.get("anchor", {}).get("raw_model_state_sha256")
            == RAW_MODEL_SHA256
            and candidate_payload.get("anchor", {}).get(
                "terminal_model_state_sha256"
            )
            == selected.get("candidate_model_state_sha256")
            and candidate_payload.get("selection_sha256")
            == SELECTION_PAYLOAD_SHA256
            and candidate_payload.get("cache_sha256") == EXPECTED_CACHE_SHA256
        )
    elif decision == "NO_GO_CW22_TARGETED_B256_BASELINE":
        checks["NO_GO_baseline_semantics"] = (
            endpoint.get("reason") == "FINAL_B256_BASELINE_CERTIFICATION_FAILED"
            and candidate_payload is None
            and endpoint.get("changed_candidate_train_shadow_count") == 0
            and isinstance(endpoint.get("baseline_checks"), Mapping)
            and not all_true(endpoint.get("baseline_checks"))
        )
    elif decision == "NO_GO_CW22_TARGETED_B256_DIRECTION":
        checks["NO_GO_direction_semantics"] = (
            endpoint.get("reason") == "TARGETED_PCGRAD_DIRECTION_GATE_FAILED"
            and candidate_payload is None
            and endpoint.get("changed_candidate_train_shadow_count") == 0
            and all_true(endpoint.get("baseline_checks"))
            and endpoint.get("pcgrad", {}).get(
                "all_tasks_robust_first_order_descent"
            )
            is False
        )
    else:
        trials = endpoint.get("trials")
        reason = endpoint.get("reason")
        common = (
            decision == "NO_GO_CW22_TARGETED_B256_PREFLIGHT"
            and candidate_payload is None
            and isinstance(trials, list)
            and bool(trials)
            and len(trials) == endpoint.get("changed_candidate_train_shadow_count")
        )
        if reason == "NO_PREREGISTERED_RADIUS_PASSED_TARGETED_TRAIN_GATE":
            checks["NO_GO_no_radius_semantics"] = (
                common
                and len(trials) == 4
                and endpoint.get("radii_contract")
                == [0.000125, 0.00025, 0.0005, 0.00075]
                and all(
                    isinstance(trial, Mapping) and trial.get("pass") is False
                    for trial in trials
                )
                and "selected_trial" not in endpoint
                and "final_checks" not in endpoint
                and "pure_payload_reconstruction" not in endpoint
            )
        elif reason == "ABSOLUTE_PAYLOAD_RECONSTRUCTION_FAILED":
            selected = endpoint.get("selected_trial")
            final_checks = endpoint.get("final_checks")
            reconstruction = endpoint.get("pure_payload_reconstruction")
            passing_indices = [
                index
                for index, trial in enumerate(trials)
                if isinstance(trial, Mapping) and trial.get("pass") is True
            ]
            checks["NO_GO_reconstruction_semantics"] = (
                common
                and isinstance(selected, Mapping)
                and selected.get("pass") is True
                and selected.get("train_gate", {}).get("pass") is True
                and all_true(selected.get("train_gate", {}).get("checks"))
                and all_true(selected.get("integrity"))
                and selected == trials[-1]
                and passing_indices == [len(trials) - 1]
                and isinstance(final_checks, Mapping)
                and final_checks.get("absolute_payload_reconstruction") is False
                and all(
                    value is True
                    for key, value in final_checks.items()
                    if key != "absolute_payload_reconstruction"
                )
                and isinstance(reconstruction, Mapping)
                and reconstruction.get("pass") is False
                and isinstance(reconstruction.get("checks"), Mapping)
                and bool(reconstruction.get("checks"))
                and not all_true(reconstruction.get("checks"))
            )
        else:
            checks["NO_GO_preflight_reason_known"] = False
    if not all(checks.values()):
        raise LauncherError(f"child semantic validation failed: {checks}")
    return {"decision": decision, "checks": checks, "pass": True}


def target_paths() -> tuple[Path, ...]:
    return (ATTEMPT, CHILD_STDOUT, CHILD_STDERR, EXECUTION, OUTPUT)


def static_audit(engine: ModuleType, engine_record: Mapping[str, Any]) -> dict[str, Any]:
    mode = stat.S_IMODE(SCRIPT.lstat().st_mode)
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "launcher_path_exact": Path(__file__).resolve() == SCRIPT.resolve(),
        "launcher_regular_single_link": stat.S_ISREG(SCRIPT.lstat().st_mode)
        and not stat.S_ISLNK(SCRIPT.lstat().st_mode)
        and int(SCRIPT.lstat().st_nlink) == 1,
        "launcher_mode_ready_or_editable": mode in {0o555, 0o664, 0o644},
        "targets_absent": not any(
            path.exists() or path.is_symlink() for path in target_paths()
        ),
    }
    if not all(checks.values()):
        raise LauncherError(f"static checks failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_passed",
        "checks": checks,
        "pass": True,
        "launcher": engine.regular_evidence(SCRIPT, None, mode),
        "engine": dict(engine_record),
        "direct_inputs": input_snapshot(engine),
        "command": [
            str(EXPECTED_PYTHON),
            "-I",
            "-B",
            "<sealed_solver_memfd>",
            "--mode",
            "run",
            "--device",
            "cuda",
            "--output",
            str(OUTPUT),
        ],
        "targets": [str(path.relative_to(ROOT)) for path in target_paths()],
        "writes_performed": 0,
        "submission_performed": False,
    }


def safe_evidence(engine: ModuleType, path: Path) -> dict[str, Any]:
    if not path.exists() or path.is_symlink():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    return engine.output_evidence(path)


def run_once() -> int:
    if stat.S_IMODE(SCRIPT.lstat().st_mode) != FROZEN_MODE:
        raise LauncherError("launcher must be frozen mode 0555")
    engine, engine_record = load_engine()
    audit = static_audit(engine, engine_record)
    launcher = engine.regular_evidence(SCRIPT, None, FROZEN_MODE)
    pre_inputs = audit["direct_inputs"]
    solver_key = str(SOLVER.relative_to(ROOT))
    open_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(SOLVER, open_flags)
    solver_descriptor: int | None = None
    try:
        source_solver = engine.sealed_solver_evidence(
            source_descriptor, pre_inputs[solver_key]
        )
        solver_descriptor, sealed_solver = engine.immutable_solver_memfd(
            source_descriptor, source_solver
        )
    finally:
        os.close(source_descriptor)
    command = [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        sealed_solver["execution_path"],
        "--mode",
        "run",
        "--device",
        "cuda",
        "--output",
        str(OUTPUT),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": "0",
            "OPENBLAS_NUM_THREADS": "32",
            "MKL_NUM_THREADS": "32",
            "OMP_NUM_THREADS": "32",
            "NUMEXPR_NUM_THREADS": "32",
        }
    )
    marker = {
        "schema_version": SCHEMA,
        "status": "attempt_committed_before_child_creation",
        "launcher": launcher,
        "engine": dict(engine_record),
        "direct_inputs_pre": pre_inputs,
        "sealed_solver": sealed_solver,
        "command": command,
        "cwd": str(ROOT),
        "environment_contract": {
            key: environment[key]
            for key in (
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "official_budget_consumed_before": 0,
        "submission_performed": False,
    }
    engine.exclusive_json(ATTEMPT, marker)
    return_code: int | None = None
    validation: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    error_text: str | None = None
    stdout_descriptor: int | None = None
    stderr_descriptor: int | None = None
    stdout_descriptor_record: dict[str, Any] | None = None
    stderr_descriptor_record: dict[str, Any] | None = None
    try:
        stdout_descriptor = engine.exclusive_rw_descriptor(CHILD_STDOUT)
        stderr_descriptor = engine.exclusive_rw_descriptor(CHILD_STDERR)
        with os.fdopen(
            stdout_descriptor, "w+b", closefd=False
        ) as stdout_handle, os.fdopen(
            stderr_descriptor, "w+b", closefd=False
        ) as stderr_handle:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                pass_fds=(solver_descriptor,),
                check=False,
            )
            stdout_handle.flush()
            os.fsync(stdout_handle.fileno())
            stderr_handle.flush()
            os.fsync(stderr_handle.fileno())
            return_code = int(process.returncode)
        if return_code == 0:
            try:
                summary = strict_json(CHILD_STDOUT, "child stdout")
                document = strict_json(OUTPUT, "child artifact")
                validation = validate_child(summary, document)
            except BaseException as error:
                error_text = f"{type(error).__name__}: {error}"
        else:
            error_text = f"child_return_code_{return_code}"
    finally:
        if solver_descriptor is not None:
            os.close(solver_descriptor)
        for descriptor, label in (
            (stdout_descriptor, "stdout"),
            (stderr_descriptor, "stderr"),
        ):
            if descriptor is None:
                continue
            os.fchmod(descriptor, EVIDENCE_MODE)
            record = engine.descriptor_evidence(descriptor)
            if label == "stdout":
                stdout_descriptor_record = record
            else:
                stderr_descriptor_record = record
            os.close(descriptor)
        if ATTEMPT.exists():
            ATTEMPT.chmod(EVIDENCE_MODE)

    post_inputs = input_snapshot(engine)
    stable = engine.assert_snapshot_stable(pre_inputs, post_inputs)
    launcher_post = engine.regular_evidence(SCRIPT, launcher["sha256"], FROZEN_MODE)
    stdout_path_record = safe_evidence(engine, CHILD_STDOUT)
    stderr_path_record = safe_evidence(engine, CHILD_STDERR)
    descriptor_paths_match = (
        stdout_descriptor_record is not None
        and stderr_descriptor_record is not None
        and engine.assert_path_matches_descriptor(
            stdout_path_record, stdout_descriptor_record
        )
        and engine.assert_path_matches_descriptor(
            stderr_path_record, stderr_descriptor_record
        )
    )
    output_record = safe_evidence(engine, OUTPUT)
    output_path_valid = (
        output_record.get("exists", True) is not False
        and output_record.get("mode_octal") == "0444"
        and validation is not None
        and isinstance(summary, Mapping)
        and output_record.get("sha256")
        == summary.get("output", {}).get("sha256")
    )
    execution = {
        "schema_version": SCHEMA,
        "status": (
            "one_shot_completed_validated"
            if return_code == 0
            and validation is not None
            and descriptor_paths_match
            and output_path_valid
            else "one_shot_failed_preserved"
        ),
        "return_code": return_code,
        "child_validation": validation,
        "child_error": error_text,
        "descriptor_paths_match": descriptor_paths_match,
        "output_path_valid": output_path_valid,
        "launcher": launcher,
        "launcher_post": launcher_post,
        "engine": dict(engine_record),
        "sealed_solver": sealed_solver,
        "direct_inputs_pre": pre_inputs,
        "direct_inputs_post": post_inputs,
        "direct_inputs_stable": stable,
        "attempt": safe_evidence(engine, ATTEMPT),
        "stdout": stdout_path_record,
        "stdout_O_EXCL_descriptor": stdout_descriptor_record,
        "stderr": stderr_path_record,
        "stderr_O_EXCL_descriptor": stderr_descriptor_record,
        "output": output_record,
        "official_budget_consumed": 0,
        "submission_performed": False,
        "rerun_permitted": False,
    }
    engine.exclusive_json(EXECUTION, execution)
    EXECUTION.chmod(EVIDENCE_MODE)
    if execution["status"] != "one_shot_completed_validated":
        raise LauncherError(
            f"one-shot child failed; evidence preserved: rc={return_code}, {error_text}"
        )
    print(
        json.dumps(
            {
                "status": execution["status"],
                "decision": validation["decision"],
                "execution": str(EXECUTION.relative_to(ROOT)),
                "output": str(OUTPUT.relative_to(ROOT)),
                "official_budget_consumed": 0,
                "submission_performed": False,
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise LauncherError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise LauncherError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise LauncherError("requires Python -I -B")
    if args.mode == "static":
        engine, engine_record = load_engine()
        print(
            json.dumps(
                static_audit(engine, engine_record),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    return run_once()


if __name__ == "__main__":
    raise SystemExit(main())

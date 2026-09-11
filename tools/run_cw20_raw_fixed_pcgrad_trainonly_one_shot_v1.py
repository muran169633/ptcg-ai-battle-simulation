#!/usr/bin/env python3
"""One-shot launcher for the frozen CW20 cache512-only PCGrad preflight."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "run_cw20_raw_fixed_pcgrad_trainonly_one_shot_v1.py"
SOLVER = TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-cw20-raw-fixed-pcgrad-cache512-one-shot-launcher-v1"

ATTEMPT = ROOT / "artifacts/.ptcg-cw20_raw_fixed_pcgrad_trainonly_20260803_v1-attempt.json"
CHILD_STDOUT = ROOT / "artifacts/cw20_raw_fixed_pcgrad_trainonly_20260803_v1.stdout.json"
CHILD_STDERR = ROOT / "artifacts/cw20_raw_fixed_pcgrad_trainonly_20260803_v1.stderr.txt"
EXECUTION = ROOT / "artifacts/cw20_raw_fixed_pcgrad_trainonly_20260803_v1.execution.json"

FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444
MFD_CLOEXEC = 0x0001
MFD_ALLOW_SEALING = 0x0002
F_ADD_SEALS = 1033
F_GET_SEALS = 1034
F_SEAL_SEAL = 0x0001
F_SEAL_SHRINK = 0x0002
F_SEAL_GROW = 0x0004
F_SEAL_WRITE = 0x0008
REQUIRED_MEMFD_SEALS = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE
SOLVER_SHA256 = "8f5b64c8a04b3de9a6d2890aac582967a50f4d708f3a11a64cbd395423d907d2"
DIRECT_INPUTS: dict[Path, tuple[str, int]] = {
    SOLVER: (SOLVER_SHA256, 0o555),
    TOOLS / "probe_u468_cw11_bootstrap_pcgrad_specialbc_cw19_v1.py": (
        "65f16009641481cd13538714520828640d00aed42aae920e24d9b70910423ae8",
        0o555,
    ),
    TOOLS / "probe_u468_raw_aggregate512_actor6_gradients.py": (
        "ed85b59bd69558ab6f027ba21df9cb56c744baf9793bbb6b550d46bffbc3389a",
        0o555,
    ),
    TOOLS / "run_u468_raw_trainhard_actor6_balanced_mix_sweep.py": (
        "419feeeb5644adf34a8d56142cfcd87b3a4b79446d95b9b566bf9fcabdb721b1",
        0o555,
    ),
}


class LauncherError(RuntimeError):
    """Fail-closed one-shot launcher error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sealed_solver_evidence(
    descriptor: int, expected: dict[str, Any]
) -> dict[str, Any]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise LauncherError("sealed solver descriptor is not a single-link regular file")
    digest = hashlib.sha256()
    offset = 0
    while offset < int(before.st_size):
        chunk = os.pread(descriptor, min(1024 * 1024, int(before.st_size) - offset), offset)
        if not chunk:
            raise LauncherError("short read from sealed solver descriptor")
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    checks = {
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "device_exact": int(after.st_dev) == int(expected["device"]),
        "inode_exact": int(after.st_ino) == int(expected["inode"]),
        "bytes_exact": int(after.st_size) == int(expected["bytes"]),
        "mode_exact": stat.S_IMODE(after.st_mode) == FROZEN_MODE,
        "sha_exact": digest.hexdigest() == SOLVER_SHA256 == expected["sha256"],
    }
    if not all(checks.values()):
        raise LauncherError(f"sealed solver identity mismatch: {checks}")
    return {
        "fd": descriptor,
        "execution_path": f"/proc/self/fd/{descriptor}",
        "sha256": digest.hexdigest(),
        "bytes": int(after.st_size),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "checks": checks,
    }


def immutable_solver_memfd(
    source_descriptor: int, source_record: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    libc = ctypes.CDLL(None, use_errno=True)
    memfd_create = getattr(libc, "memfd_create", None)
    if memfd_create is None:
        raise LauncherError("libc memfd_create unavailable")
    memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    memfd_create.restype = ctypes.c_int
    descriptor = int(
        memfd_create(
            b"ptcg-cw20-immutable-solver",
            MFD_CLOEXEC | MFD_ALLOW_SEALING,
        )
    )
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise LauncherError(f"memfd_create failed: errno={error_number}")
    try:
        offset = 0
        while offset < int(source_record["bytes"]):
            chunk = os.pread(
                source_descriptor,
                min(1024 * 1024, int(source_record["bytes"]) - offset),
                offset,
            )
            if not chunk:
                raise LauncherError("short source read while constructing solver memfd")
            view = memoryview(chunk)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise LauncherError("short write while constructing solver memfd")
                written += count
            offset += len(chunk)
        os.fchmod(descriptor, FROZEN_MODE)
        before_seal = descriptor_evidence(descriptor)
        if (
            before_seal["sha256"] != SOLVER_SHA256
            or before_seal["bytes"] != source_record["bytes"]
            or before_seal["mode_octal"] != "0555"
        ):
            raise LauncherError("solver memfd copy identity mismatch")
        fcntl.fcntl(descriptor, F_ADD_SEALS, REQUIRED_MEMFD_SEALS)
        observed_seals = int(fcntl.fcntl(descriptor, F_GET_SEALS))
        if observed_seals != REQUIRED_MEMFD_SEALS:
            raise LauncherError(f"solver memfd seal mismatch: {observed_seals}")
        after_seal = descriptor_evidence(descriptor)
        if after_seal != before_seal:
            raise LauncherError("solver memfd changed while sealing")
        return descriptor, {
            **after_seal,
            "execution_path": f"/proc/self/fd/{descriptor}",
            "seals": observed_seals,
            "required_seals": REQUIRED_MEMFD_SEALS,
            "immutable_write_grow_shrink_and_seal": True,
            "source": source_record,
        }
    except BaseException:
        os.close(descriptor)
        raise


def regular_evidence(
    path: Path, expected_sha256: str | None, expected_mode: int
) -> dict[str, Any]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise LauncherError(f"identity/mode drift: {path}")
    observed_sha = sha256_file(path)
    after = path.lstat()
    checks = {
        "regular": stat.S_ISREG(after.st_mode) and not stat.S_ISLNK(after.st_mode),
        "single_link": int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": expected_sha256 is None or observed_sha == expected_sha256,
    }
    if not all(checks.values()):
        raise LauncherError(f"file drift: {path}: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed_sha,
        "bytes": int(after.st_size),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def input_snapshot() -> dict[str, Any]:
    return {
        str(path.relative_to(ROOT)): regular_evidence(path, digest, mode)
        for path, (digest, mode) in DIRECT_INPUTS.items()
    }


def assert_snapshot_stable(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, bool]:
    if set(before) != set(after):
        raise LauncherError("direct input key set changed across child")
    checks = {
        key: all(
            before[key][field] == after[key][field]
            for field in ("sha256", "bytes", "mode_octal", "device", "inode")
        )
        for key in before
    }
    if not all(checks.values()):
        raise LauncherError(f"direct input changed across child: {checks}")
    return checks


def exclusive_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def exclusive_rw_descriptor(path: Path, mode: int = 0o600) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, mode)


def descriptor_evidence(descriptor: int) -> dict[str, Any]:
    observed = os.fstat(descriptor)
    digest = hashlib.sha256()
    offset = 0
    while offset < int(observed.st_size):
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, int(observed.st_size) - offset),
            offset,
        )
        if not chunk:
            raise LauncherError("short read from evidence descriptor")
        digest.update(chunk)
        offset += len(chunk)
    return {
        "sha256": digest.hexdigest(),
        "bytes": int(observed.st_size),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
    }


def assert_path_matches_descriptor(
    path_evidence: dict[str, Any], descriptor_record: dict[str, Any]
) -> bool:
    return all(
        path_evidence[key] == descriptor_record[key]
        for key in ("sha256", "bytes", "device", "inode", "mode_octal")
    )


def exclusive_json(path: Path, value: Any, mode: int = 0o600) -> dict[str, Any]:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    exclusive_bytes(path, payload, mode)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
    }


def output_evidence(path: Path) -> dict[str, Any]:
    return regular_evidence(path, None, EVIDENCE_MODE)


def collect_child_input_records(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = document.get("integrity", {}).get("frozen_inputs")
    if not isinstance(root, dict):
        raise LauncherError("child frozen-input evidence missing")
    records: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if (
                isinstance(value.get("path"), str)
                and isinstance(value.get("sha256"), str)
                and isinstance(value.get("bytes"), int)
            ):
                key = str(value["path"])
                prior = records.get(key)
                if prior is not None and (
                    prior["sha256"] != value["sha256"]
                    or prior["bytes"] != value["bytes"]
                ):
                    raise LauncherError(f"conflicting child evidence: {key}")
                records[key] = value
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(root)
    return records


def validate_child_reported_inputs(document: dict[str, Any]) -> dict[str, Any]:
    records = collect_child_input_records(document)
    if len(records) < 12:
        raise LauncherError(f"too few child input identities: {len(records)}")
    audits: dict[str, Any] = {}
    for key, record in sorted(records.items()):
        declared = Path(key)
        path = declared if declared.is_absolute() else ROOT / declared
        resolved = path.resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError as error:
            raise LauncherError(f"child input outside workspace: {key}") from error
        observed = resolved.lstat()
        mode_record = record.get("mode_octal")
        checks = {
            "regular": stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode),
            "single_link": int(observed.st_nlink) == 1,
            "sha_exact": sha256_file(resolved) == str(record["sha256"]),
            "bytes_exact": int(observed.st_size) == int(record["bytes"]),
            "device_exact_if_reported": "device" not in record
            or int(observed.st_dev) == int(record["device"]),
            "inode_exact_if_reported": "inode" not in record
            or int(observed.st_ino) == int(record["inode"]),
            "mode_exact_if_reported": mode_record is None
            or stat.S_IMODE(observed.st_mode) == int(str(mode_record), 8),
        }
        if not all(checks.values()):
            raise LauncherError(f"child input changed: {key}: {checks}")
        audits[key] = checks
    required_suffixes = {
        "artifacts/u468_raw_full_train_margin_profile_v3_20260802.json",
        "checkpoints/update-0468.pt",
        "best.pt",
        "rank01_flg.zip",
        "bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
        "bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
        "tools/train_ppo.py",
        "tools/run_ppo_bc_repair.py",
    }
    suffix_checks = {
        suffix: any(key.endswith(suffix) for key in records)
        for suffix in required_suffixes
    }
    if not all(suffix_checks.values()):
        raise LauncherError(f"required child inputs absent: {suffix_checks}")
    return {
        "pass": True,
        "reported_identity_count": len(records),
        "required_suffixes": suffix_checks,
        "post_run_audits": audits,
    }


def validate_child(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise LauncherError("child stdout is not one JSON object")
    decision = document.get("decision")
    allowed = {
        "GO_CW20_CACHE512_PREFLIGHT",
        "NO_GO_CW20_CACHE512_PREFLIGHT",
    }
    contract = document.get("contract", {})
    endpoint = document.get("endpoint", {})
    final_checks = endpoint.get("final_checks")
    expected_go_keys = {"integrity", "train_gate", "pure_payload_reconstruction"}
    go_final_exact = (
        isinstance(final_checks, dict)
        and set(final_checks) == expected_go_keys
        and all(final_checks[key] is True for key in expected_go_keys)
    )
    cuda = document.get("integrity", {}).get("runtime", {}).get("cuda", {})
    runtime_checks = (
        document.get("integrity", {}).get("runtime", {}).get("checks", {})
    )
    expected_cuda_keys = {
        "available",
        "native_bf16",
        "device_count_exact_1",
        "current_device_exact_0",
        "device_name_exact",
        "compute_capability_exact_12_0",
        "torch_version_exact",
        "cuda_runtime_exact",
    }
    expected_runtime_check_keys = {
        "cwd_exact",
        "python_exact_my_project_env",
        "isolated",
        "dont_write_bytecode",
        "cublas_workspace_exact",
    }
    checks = {
        "schema_exact": document.get("schema_version")
        == "ptcg-u468-raw-fixed-pcgrad-specialbc-cw20-v1",
        "status_decision_exact": document.get("status") == decision,
        "endpoint_decision_exact": endpoint.get("decision") == decision,
        "decision_allowed": decision in allowed,
        "cache512_scope_exact": contract.get("promotion_scope")
        == "cache512_train_only_preflight_not_fulltrain_or_specialist",
        "cw11_no_harm_not_claimed": contract.get("cw11_no_harm_proven") is False,
        "official_evaluation_zero": contract.get(
            "changed_candidate_official_or_validation_evaluation_count"
        )
        == 0,
        "official_budget_zero": contract.get("official_candidate_budget_consumed") == 0,
        "no_writes_inside_child": contract.get("checkpoint_writes") == 0
        and contract.get("model_writes") == 0
        and contract.get("result_artifact_writes") == 0,
        "no_optimizer_or_parameter_projection": contract.get(
            "optimizer_instances"
        )
        == 0
        and contract.get("optimizer_steps") == 0
        and contract.get("backward_calls") == 0
        and contract.get("parameter_space_projections") == 0,
        "pcgrad_projection_disclosed": contract.get(
            "pcgrad_gradient_conflict_projection"
        )
        is True,
        "stdout_only": contract.get("stdout_only") is True,
        "train_only": document.get("train_cache", {}).get(
            "non_train_members_opened"
        )
        is False,
        "runtime_gpu_contract": isinstance(cuda, dict)
        and set(cuda) == expected_cuda_keys
        and all(value is True for value in cuda.values()),
        "runtime_process_contract": isinstance(runtime_checks, dict)
        and set(runtime_checks) == expected_runtime_check_keys
        and all(value is True for value in runtime_checks.values()),
        "runtime_pass_true": document.get("integrity", {})
        .get("runtime", {})
        .get("pass")
        is True,
        "source_pass_true": document.get("integrity", {})
        .get("source", {})
        .get("pass")
        is True,
        "child_source_exact": document.get("integrity", {})
        .get("source", {})
        .get("source_sha256")
        == SOLVER_SHA256,
        "submission_false": document.get("submission_performed") is False,
        "writes_zero": document.get("writes_performed") == 0,
        "endpoint_payload_consistent": (
            decision == "GO_CW20_CACHE512_PREFLIGHT"
            and endpoint.get("candidate_payload") is not None
            and endpoint.get("changed_candidate_train_endpoint_count") == 1
            and go_final_exact
        )
        or (
            decision == "NO_GO_CW20_CACHE512_PREFLIGHT"
            and endpoint.get("candidate_payload") is None
            and endpoint.get("changed_candidate_train_endpoint_count") in {0, 1}
        ),
    }
    if decision == "NO_GO_CW20_CACHE512_PREFLIGHT":
        changed_count = endpoint.get("changed_candidate_train_endpoint_count")
        reason = endpoint.get("reason")
        direction_gate = endpoint.get("pcgrad", {}).get("direction_gate")
        if changed_count == 0:
            no_go_coherent = (
                reason == "PCGRAD_DIRECTION_GATE_FAILED"
                and final_checks is None
                and isinstance(direction_gate, dict)
                and bool(direction_gate)
                and not all(value is True for value in direction_gate.values())
            )
        else:
            no_go_coherent = isinstance(final_checks, dict) and not all(
                value is True for value in final_checks.values()
            )
            if reason == "integrity_gate_failed":
                no_go_coherent = no_go_coherent and final_checks == {
                    "integrity": False,
                    "train_gate": final_checks.get("train_gate"),
                }
            elif reason == "cache512_train_gate_failed":
                no_go_coherent = no_go_coherent and final_checks == {
                    "integrity": True,
                    "train_gate": False,
                }
            elif reason == "payload_reconstruction_gate_failed":
                no_go_coherent = no_go_coherent and final_checks == {
                    "integrity": True,
                    "train_gate": True,
                    "pure_payload_reconstruction": False,
                }
            else:
                no_go_coherent = False
        checks["no_go_reason_and_gate_coherent"] = no_go_coherent
    else:
        checks["go_reason_coherent"] = (
            endpoint.get("reason") == "all_cache512_train_only_gates_passed"
        )
    if not all(checks.values()):
        raise LauncherError(f"child semantic validation failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "decision": decision,
        "expected_GO_final_check_keys": sorted(expected_go_keys),
        "child_reported_inputs_post": validate_child_reported_inputs(document),
    }


def static_audit() -> dict[str, Any]:
    targets = (ATTEMPT, CHILD_STDOUT, CHILD_STDERR, EXECUTION)
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "launcher_path_exact": Path(__file__).resolve() == SCRIPT.resolve(),
        "targets_absent": not any(path.exists() or path.is_symlink() for path in targets),
    }
    if not all(checks.values()):
        raise LauncherError(f"static checks failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_passed",
        "checks": checks,
        "pass": True,
        "launcher": regular_evidence(
            SCRIPT, None, stat.S_IMODE(SCRIPT.lstat().st_mode)
        ),
        "direct_inputs": input_snapshot(),
        "command": [
            str(EXPECTED_PYTHON),
            "-I",
            "-B",
            str(SOLVER),
            "--mode",
            "run",
            "--device",
            "cuda",
        ],
        "targets": [str(path.relative_to(ROOT)) for path in targets],
        "writes_performed": 0,
        "submission_performed": False,
    }


def run_once() -> int:
    if stat.S_IMODE(SCRIPT.lstat().st_mode) != FROZEN_MODE:
        raise LauncherError("launcher must be frozen mode 0555")
    audit = static_audit()
    launcher = regular_evidence(SCRIPT, None, FROZEN_MODE)
    pre_inputs = audit["direct_inputs"]
    solver_key = str(SOLVER.relative_to(ROOT))
    open_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(SOLVER, open_flags)
    try:
        source_solver = sealed_solver_evidence(
            source_descriptor, pre_inputs[solver_key]
        )
        solver_descriptor, sealed_solver = immutable_solver_memfd(
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
    exclusive_json(ATTEMPT, marker)
    return_code: int | None = None
    validation: dict[str, Any] | None = None
    child_error: str | None = None
    stdout_descriptor: int | None = None
    stderr_descriptor: int | None = None
    stdout_descriptor_record: dict[str, Any] | None = None
    stderr_descriptor_record: dict[str, Any] | None = None
    try:
        stdout_descriptor = exclusive_rw_descriptor(CHILD_STDOUT)
        stderr_descriptor = exclusive_rw_descriptor(CHILD_STDERR)
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
                    stdout_handle.seek(0)
                    validation = validate_child(
                        json.loads(stdout_handle.read().decode("utf-8"))
                    )
                except BaseException as error:
                    child_error = f"{type(error).__name__}: {error}"
            else:
                child_error = f"child_return_code_{return_code}"
    finally:
        os.close(solver_descriptor)
        for descriptor, label in (
            (stdout_descriptor, "stdout"),
            (stderr_descriptor, "stderr"),
        ):
            if descriptor is None:
                continue
            os.fchmod(descriptor, EVIDENCE_MODE)
            record = descriptor_evidence(descriptor)
            if label == "stdout":
                stdout_descriptor_record = record
            else:
                stderr_descriptor_record = record
            os.close(descriptor)
        if ATTEMPT.exists():
            ATTEMPT.chmod(EVIDENCE_MODE)

    post_inputs = input_snapshot()
    stable = assert_snapshot_stable(pre_inputs, post_inputs)
    launcher_post = regular_evidence(SCRIPT, launcher["sha256"], FROZEN_MODE)
    stdout_path_record = output_evidence(CHILD_STDOUT)
    stderr_path_record = output_evidence(CHILD_STDERR)
    if (
        stdout_descriptor_record is None
        or stderr_descriptor_record is None
        or not assert_path_matches_descriptor(
            stdout_path_record, stdout_descriptor_record
        )
        or not assert_path_matches_descriptor(
            stderr_path_record, stderr_descriptor_record
        )
    ):
        raise LauncherError("stdout/stderr O_EXCL descriptor identity drift")
    execution = {
        "schema_version": SCHEMA,
        "status": (
            "one_shot_completed_validated"
            if return_code == 0 and validation is not None
            else "one_shot_failed_preserved"
        ),
        "return_code": return_code,
        "child_validation": validation,
        "child_error": child_error,
        "launcher": launcher,
        "launcher_post": launcher_post,
        "sealed_solver": sealed_solver,
        "direct_inputs_pre": pre_inputs,
        "direct_inputs_post": post_inputs,
        "direct_inputs_stable": stable,
        "attempt": output_evidence(ATTEMPT),
        "stdout": stdout_path_record,
        "stdout_O_EXCL_descriptor": stdout_descriptor_record,
        "stderr": stderr_path_record,
        "stderr_O_EXCL_descriptor": stderr_descriptor_record,
        "official_budget_consumed": 0,
        "submission_performed": False,
        "rerun_permitted": False,
    }
    exclusive_json(EXECUTION, execution)
    EXECUTION.chmod(EVIDENCE_MODE)
    if return_code != 0 or validation is None:
        raise LauncherError(
            f"one-shot child failed; evidence preserved: rc={return_code}, {child_error}"
        )
    print(
        json.dumps(
            {
                "status": execution["status"],
                "decision": validation["decision"],
                "execution": str(EXECUTION.relative_to(ROOT)),
                "stdout": str(CHILD_STDOUT.relative_to(ROOT)),
                "official_budget_consumed": 0,
                "submission_performed": False,
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


def preserve_failure(error: BaseException) -> None:
    if not ATTEMPT.exists():
        return
    for path in (ATTEMPT, CHILD_STDOUT, CHILD_STDERR, EXECUTION):
        if path.exists():
            try:
                path.chmod(EVIDENCE_MODE)
            except OSError:
                pass
    if EXECUTION.exists():
        return
    files: dict[str, Any] = {}
    for path in (ATTEMPT, CHILD_STDOUT, CHILD_STDERR):
        if not path.exists():
            files[str(path.relative_to(ROOT))] = {"exists": False}
            continue
        try:
            files[str(path.relative_to(ROOT))] = output_evidence(path)
        except BaseException as evidence_error:
            files[str(path.relative_to(ROOT))] = {
                "exists": True,
                "evidence_error": f"{type(evidence_error).__name__}: {evidence_error}",
            }
    failure = {
        "schema_version": SCHEMA,
        "status": "one_shot_unclassified_failure_preserved",
        "error_type": type(error).__name__,
        "error": str(error),
        "files": files,
        "official_budget_consumed": 0,
        "submission_performed": False,
        "rerun_permitted": False,
    }
    try:
        exclusive_json(EXECUTION, failure)
        EXECUTION.chmod(EVIDENCE_MODE)
    except BaseException:
        pass


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
        print(
            json.dumps(
                static_audit(), sort_keys=True, ensure_ascii=False, allow_nan=False
            )
        )
        return 0
    try:
        return run_once()
    except BaseException as error:
        preserve_failure(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

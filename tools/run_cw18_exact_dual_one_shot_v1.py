#!/usr/bin/env python3
"""Run the frozen CW18 CPU solver once and preserve immutable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "run_cw18_exact_dual_one_shot_v1.py"
SOLVER = TOOLS / "solve_cw18_exact_dual_v1.py"
FIXTURE = ROOT / "artifacts/cw18_direct_fixture_20260803_v1.stdout.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

SOLVER_SHA256 = (
    "1f92b7b58aa6faa095f7f40fef62aee35ed3ee906c3faa4703198d31cda44310"
)
FIXTURE_SHA256 = (
    "a0d3ecc5a1315160b8b06ed212af9c6b6179c13695488c44cf5e445e043cd929"
)
FIXTURE_INTERNAL_SHA256 = (
    "ef6c7bfa1017ea9ceb77b24543a2a9507bdd48873c7b3a21865691684987c8ec"
)
EXPECTED_BASIS_SHA256 = (
    "ecd77ab625b571cd04037d4eeba113a9b646fc66503fe35c1713159770ede604"
)
EXPECTED_CENTER_SHA256 = (
    "e3814600afa6f1c7843ca073d89a86cb1e32cf162d89f8cbfaf8db378ad3183f"
)

ATTEMPT = ROOT / "artifacts/.ptcg-cw18_exact_dual_20260803_v1-attempt.json"
CHILD_STDOUT = ROOT / "artifacts/cw18_exact_dual_20260803_v1.stdout.json"
CHILD_STDERR = ROOT / "artifacts/cw18_exact_dual_20260803_v1.stderr.txt"
EXECUTION = ROOT / "artifacts/cw18_exact_dual_20260803_v1.execution.json"

SCHEMA = "ptcg-cw18-exact-dual-one-shot-launcher-v1"
SOLVER_SCHEMA = "ptcg-cw18-exact-dual-solver-v1"
SOLVED_STATUS = "solved_formally_certified_CW18_exact_dual_candidate"
NO_CANDIDATE_STATUS = "complete_no_formally_certified_CW18_candidate"
FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444
MAX_STDOUT_BYTES = 256 * 1024 * 1024


class LauncherError(RuntimeError):
    """Fail-closed launcher contract error."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_evidence(
    path: Path, *, expected_sha256: str | None = None, expected_mode: int | None = None
) -> dict[str, Any]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise LauncherError(f"not one regular single-link file: {path}")
    observed_sha = sha256_file(path)
    after = path.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or int(after.st_nlink) != 1
    ):
        raise LauncherError(f"file changed during hash: {path}")
    observed_mode = stat.S_IMODE(after.st_mode)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise LauncherError(f"SHA256 drift: {path}")
    if expected_mode is not None and observed_mode != expected_mode:
        raise LauncherError(f"mode drift: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed_sha,
        "bytes": int(after.st_size),
        "mode_octal": format(observed_mode, "04o"),
        "nlink": int(after.st_nlink),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
    }


def create_exclusive(path: Path, payload: bytes, final_mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, final_mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def open_exclusive_capture(path: Path) -> Any:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "wb", buffering=0)


def seal_capture(handle: Any | None, label: str) -> str | None:
    if handle is None:
        return None
    error: str | None = None
    try:
        handle.flush()
        os.fsync(handle.fileno())
        os.fchmod(handle.fileno(), EVIDENCE_MODE)
        os.fsync(handle.fileno())
    except BaseException as exc:
        error = f"{label}:{type(exc).__name__}:{exc}"
    finally:
        try:
            handle.close()
        except BaseException as exc:
            suffix = f"{label}_close:{type(exc).__name__}:{exc}"
            error = suffix if error is None else error + ";" + suffix
    return error


def strict_json_object(payload: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise LauncherError("solver stdout is not one JSON object")
    return value


def all_true_mapping(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(item is True for item in value.values())
    )


def validate_solver_result(value: dict[str, Any], returncode: int) -> dict[str, Any]:
    status = value.get("status")
    solved = status == SOLVED_STATUS
    no_candidate = status == NO_CANDIDATE_STATUS
    selected = value.get("selected_candidate")
    reconstruction = value.get("reconstruction")
    fixture = value.get("fixture")
    input_evidence = value.get("input")
    classification = value.get("classification")
    common_checks = {
        "schema_exact": value.get("schema_version") == SOLVER_SCHEMA,
        "status_and_returncode_consistent": (solved and returncode == 0)
        or (no_candidate and returncode == 2),
        "top_checks_all_true": all_true_mapping(value.get("checks")),
        "run_executed_true": value.get("run_executed") is True,
        "cuda_false": value.get("cuda_accessed") is False,
        "writes_false": value.get("writes_performed") is False,
        "classification_safe": isinstance(classification, Mapping)
        and classification.get("cpu_only") is True
        and classification.get("fixture_read_only") is True
        and classification.get("stdout_only") is True
        and classification.get("filesystem_writes") is False
        and classification.get("cuda_access") is False
        and classification.get("model_access") is False
        and classification.get("checkpoint_writes") is False
        and classification.get("network_access") is False
        and classification.get("broad_gold_access") is False
        and classification.get("package_upload_submission") is False,
        "input_sha_bound": isinstance(input_evidence, Mapping)
        and input_evidence.get("sha256") == FIXTURE_SHA256
        and input_evidence.get("mode_octal") == "0444"
        and input_evidence.get("path")
        == str(FIXTURE.relative_to(ROOT)),
        "fixture_identity_bound": isinstance(fixture, Mapping)
        and fixture.get("fixture_sha256") == FIXTURE_INTERNAL_SHA256
        and fixture.get("row_count") == 91
        and fixture.get("reduced_dimension") == 80
        and fixture.get("rho") == 0.125
        and fixture.get("coordinate_scale") == 0.001,
        "reconstruction_identity_bound": isinstance(reconstruction, Mapping)
        and reconstruction.get("basis_float64_le_sha256")
        == EXPECTED_BASIS_SHA256
        and reconstruction.get("center_raw_float64_le_sha256")
        == EXPECTED_CENTER_SHA256
        and reconstruction.get("expected_bootstrap_point_sha256")
        == EXPECTED_CENTER_SHA256
        and reconstruction.get("basis_shape") == [80, 65793]
        and all_true_mapping(reconstruction.get("center_binding_checks")),
        "acceptance_contract_all_true": all_true_mapping(value.get("acceptance")),
    }
    outcome_checks: dict[str, bool]
    if solved:
        certificate = selected.get("certificate") if isinstance(selected, Mapping) else None
        original = selected.get("original_space") if isinstance(selected, Mapping) else None
        mapped = (
            selected.get("mapped_raw_radius_certificate")
            if isinstance(selected, Mapping)
            else None
        )
        outcome_checks = {
            "solved_pass_true": value.get("pass") is True,
            "formal_count_positive": type(
                value.get("formally_certified_candidate_count")
            ) is int
            and value["formally_certified_candidate_count"] >= 1,
            "selected_object_formal": isinstance(selected, Mapping)
            and selected.get("formal_pass") is True
            and all_true_mapping(selected.get("identity_checks")),
            "exact_certificate_pass": isinstance(certificate, Mapping)
            and certificate.get("formal_certificate_pass") is True
            and all_true_mapping(certificate.get("checks")),
            "mapped_radius_pass": isinstance(mapped, Mapping)
            and mapped.get("pass") is True
            and all_true_mapping(mapped.get("checks")),
            "original_raw_pass": isinstance(original, Mapping)
            and original.get("pass") is True
            and all_true_mapping(original.get("checks"))
            and isinstance(original.get("raw_gate"), Mapping)
            and original["raw_gate"].get("pass") is True
            and all_true_mapping(original["raw_gate"].get("checks")),
            "candidate_raw_hash_present": isinstance(original, Mapping)
            and isinstance(original.get("candidate_raw_float64_le_sha256"), str)
            and len(original["candidate_raw_float64_le_sha256"]) == 64,
        }
        outcome = "formally_certified_candidate"
    else:
        outcome_checks = {
            "no_candidate_pass_false": no_candidate and value.get("pass") is False,
            "formal_count_zero": value.get("formally_certified_candidate_count") == 0,
            "selected_absent": selected is None
            and value.get("selected_candidate_index") is None,
        }
        outcome = "complete_no_formally_certified_candidate"
    checks = {**common_checks, **outcome_checks}
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "outcome": outcome,
        "solver_status": status,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def validate_hex64(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise LauncherError(f"{label} must be lowercase SHA256 hex")


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    for value, label in (
        (SOLVER_SHA256, "SOLVER_SHA256"),
        (FIXTURE_SHA256, "FIXTURE_SHA256"),
        (FIXTURE_INTERNAL_SHA256, "FIXTURE_INTERNAL_SHA256"),
        (EXPECTED_BASIS_SHA256, "EXPECTED_BASIS_SHA256"),
        (EXPECTED_CENTER_SHA256, "EXPECTED_CENTER_SHA256"),
    ):
        validate_hex64(value, label)
    if Path(__file__).resolve() != SCRIPT.resolve():
        raise LauncherError("launcher must execute from its frozen SCRIPT path")
    if Path.cwd().resolve() != ROOT:
        raise LauncherError("launcher cwd drift")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise LauncherError("launcher is not using my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise LauncherError("launcher requires Python -I -B")
    if stat.S_IMODE(SCRIPT.lstat().st_mode) != FROZEN_MODE:
        raise LauncherError("launcher source must be frozen mode 0555")
    targets = (ATTEMPT, CHILD_STDOUT, CHILD_STDERR, EXECUTION)
    existing = [str(path.relative_to(ROOT)) for path in targets if path.exists()]
    if existing:
        raise LauncherError(f"one-shot target already exists: {existing}")

    launcher_evidence = regular_evidence(SCRIPT, expected_mode=FROZEN_MODE)
    solver_evidence = regular_evidence(
        SOLVER, expected_sha256=SOLVER_SHA256, expected_mode=FROZEN_MODE
    )
    fixture_evidence = regular_evidence(
        FIXTURE, expected_sha256=FIXTURE_SHA256, expected_mode=EVIDENCE_MODE
    )
    command = [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        str(SOLVER),
        "--mode",
        "solve",
        "--fixture-sha256",
        FIXTURE_SHA256,
    ]
    marker = {
        "schema_version": SCHEMA,
        "status": "attempt_committed_before_child_creation",
        "launcher": launcher_evidence,
        "solver": solver_evidence,
        "fixture": fixture_evidence,
        "command": command,
        "cwd": str(ROOT),
        "outputs": {
            "stdout": str(CHILD_STDOUT.relative_to(ROOT)),
            "stderr": str(CHILD_STDERR.relative_to(ROOT)),
            "execution": str(EXECUTION.relative_to(ROOT)),
        },
        "single_attempt_no_retry": True,
        "cpu_only_no_model_no_official_eval": True,
        "network_upload_submission": False,
    }
    create_exclusive(ATTEMPT, canonical_json(marker), EVIDENCE_MODE)

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    stdout_handle: Any | None = None
    stderr_handle: Any | None = None
    completed_returncode: int | None = None
    launch_error: str | None = None
    seal_errors: list[str] = []
    try:
        stdout_handle = open_exclusive_capture(CHILD_STDOUT)
        stderr_handle = open_exclusive_capture(CHILD_STDERR)
        regular_evidence(
            SOLVER, expected_sha256=SOLVER_SHA256, expected_mode=FROZEN_MODE
        )
        regular_evidence(
            FIXTURE, expected_sha256=FIXTURE_SHA256, expected_mode=EVIDENCE_MODE
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
        completed_returncode = int(completed.returncode)
    except BaseException as exc:
        launch_error = f"{type(exc).__name__}:{exc}"
    finally:
        for error in (
            seal_capture(stdout_handle, "stdout"),
            seal_capture(stderr_handle, "stderr"),
        ):
            if error is not None:
                seal_errors.append(error)

    stdout_evidence: dict[str, Any] | None = None
    stderr_evidence: dict[str, Any] | None = None
    capture_errors: list[str] = []
    for label, path in (("stdout", CHILD_STDOUT), ("stderr", CHILD_STDERR)):
        if not path.exists():
            capture_errors.append(f"{label}:missing_after_capture")
            continue
        if any(value.startswith(label) for value in seal_errors):
            continue
        try:
            evidence = regular_evidence(path, expected_mode=EVIDENCE_MODE)
            if label == "stdout":
                stdout_evidence = evidence
            else:
                stderr_evidence = evidence
        except BaseException as exc:
            capture_errors.append(f"{label}:{type(exc).__name__}:{exc}")

    validation: dict[str, Any] = {
        "pass": False,
        "checks": {},
        "error": "child did not produce a recognized completed outcome",
    }
    if (
        completed_returncode in (0, 2)
        and launch_error is None
        and not seal_errors
        and not capture_errors
        and stdout_evidence is not None
    ):
        try:
            if stdout_evidence["bytes"] > MAX_STDOUT_BYTES:
                raise LauncherError("solver stdout exceeds 256 MiB")
            stdout_payload = CHILD_STDOUT.read_bytes()
            if (
                len(stdout_payload) != stdout_evidence["bytes"]
                or sha256_bytes(stdout_payload) != stdout_evidence["sha256"]
            ):
                raise LauncherError("solver stdout changed after evidence hash")
            validation = validate_solver_result(
                strict_json_object(stdout_payload), completed_returncode
            )
            if validation["pass"] is not True:
                validation["error"] = "solver semantic result failed"
        except BaseException as exc:
            validation = {
                "pass": False,
                "checks": {},
                "error": f"{type(exc).__name__}:{exc}",
            }

    solver_post: dict[str, Any] | None = None
    fixture_post: dict[str, Any] | None = None
    post_errors: list[str] = []
    try:
        solver_post = regular_evidence(
            SOLVER, expected_sha256=SOLVER_SHA256, expected_mode=FROZEN_MODE
        )
    except BaseException as exc:
        post_errors.append(f"solver:{type(exc).__name__}:{exc}")
    try:
        fixture_post = regular_evidence(
            FIXTURE, expected_sha256=FIXTURE_SHA256, expected_mode=EVIDENCE_MODE
        )
    except BaseException as exc:
        post_errors.append(f"fixture:{type(exc).__name__}:{exc}")

    validated = (
        validation.get("pass") is True
        and launch_error is None
        and not seal_errors
        and not capture_errors
        and not post_errors
        and solver_post is not None
        and fixture_post is not None
        and stdout_evidence is not None
        and stderr_evidence is not None
    )
    outcome = validation.get("outcome") if validated else "failure"
    execution = {
        "schema_version": SCHEMA,
        "status": "validated_" + str(outcome) if validated else "child_failed",
        "child_returncode": completed_returncode,
        "launch_error": launch_error,
        "seal_errors": seal_errors,
        "capture_evidence_errors": capture_errors,
        "post_evidence_errors": post_errors,
        "result_validation": validation,
        "attempt": regular_evidence(ATTEMPT, expected_mode=EVIDENCE_MODE),
        "launcher": launcher_evidence,
        "solver": solver_evidence,
        "solver_post_child": solver_post,
        "fixture": fixture_evidence,
        "fixture_post_child": fixture_post,
        "stdout": stdout_evidence,
        "stderr": stderr_evidence,
        "command": command,
        "cwd": str(ROOT),
        "single_attempt_no_retry": True,
    }
    create_exclusive(EXECUTION, canonical_json(execution), EVIDENCE_MODE)
    final = {
        **execution,
        "execution": regular_evidence(EXECUTION, expected_mode=EVIDENCE_MODE),
    }
    print(canonical_json(final).decode("utf-8"), end="")
    if not validated:
        return 1
    return 0 if outcome == "formally_certified_candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())

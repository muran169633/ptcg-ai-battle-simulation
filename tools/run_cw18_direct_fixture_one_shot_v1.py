#!/usr/bin/env python3
"""Run the frozen CW18 fixture builder exactly once and preserve all evidence."""

from __future__ import annotations

import argparse
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
SCRIPT = TOOLS / "run_cw18_direct_fixture_one_shot_v1.py"
DEFAULT_BUILDER = TOOLS / "build_cw18_direct_fixture_from_cw16_stdout_v1.py"
BUILDER_SHA256 = (
    "b2b86a6ce117bb5063eaf36c3a86603af1694f6dc9be05e40221e59ee886f2aa"
)
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

ATTEMPT = ROOT / "artifacts/.ptcg-cw18_direct_fixture_20260803_v1-attempt.json"
CHILD_STDOUT = ROOT / "artifacts/cw18_direct_fixture_20260803_v1.stdout.json"
CHILD_STDERR = ROOT / "artifacts/cw18_direct_fixture_20260803_v1.stderr.txt"
EXECUTION = ROOT / "artifacts/cw18_direct_fixture_20260803_v1.execution.json"

SCHEMA = "ptcg-cw18-direct-fixture-one-shot-launcher-v1"
FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def validate_hex64(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise LauncherError(f"{label} must be lowercase SHA256 hex")


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
        raise LauncherError("builder stdout is not one JSON object")
    return value


def validate_builder_result(value: dict[str, Any]) -> dict[str, Any]:
    fixture = value.get("fixture")
    reconstruction = value.get("reconstruction_map")
    stage1 = value.get("stage1")
    source_audit = value.get("source_audit")
    raw_space = value.get("raw_space_contract")
    captured = value.get("captured_constraint_audit")
    precision = value.get("precision_scope")
    graph = value.get("graph_fingerprint_audit")
    reconstruction_audit = value.get("reconstruction")
    checks = {
        "schema_exact": value.get("schema_version")
        == "ptcg-cw18-direct-fixture-from-cw16-stdout-v1",
        "status_exact": value.get("status") == "fixture_built",
        "pass_true": value.get("pass") is True,
        "fixture_object": isinstance(fixture, dict)
        and isinstance(fixture.get("fixture_sha256"), str),
        "reconstruction_map_object": isinstance(reconstruction, dict)
        and reconstruction.get("basis_orientation")
        == "reduced_rank_by_actor_original"
        and isinstance(reconstruction.get("basis_reduced_by_actor"), dict)
        and isinstance(reconstruction.get("center_raw"), dict),
        "basis_sha_bound": isinstance(value.get("basis_float64_le_sha256"), str)
        and value.get("basis_float64_le_sha256")
        == (reconstruction or {}).get("basis_float64_le_sha256"),
        "bootstrap_sha_bound": value.get("expected_bootstrap_point_sha256")
        == "e3814600afa6f1c7843ca073d89a86cb1e32cf162d89f8cbfaf8db378ad3183f"
        == (reconstruction or {}).get("center_raw_float64_le_sha256"),
        "stage1_seed_present": isinstance(stage1, dict)
        and isinstance(stage1.get("x_scaled"), dict)
        and isinstance(stage1.get("raw_certification"), dict)
        and stage1["raw_certification"].get("pass") is True,
        "captured_constraint_pass": isinstance(captured, dict)
        and captured.get("pass") is True,
        "raw_space_contract_pass": isinstance(raw_space, dict)
        and raw_space.get("schema_version") == "ptcg-cw18-raw-space-contract-v1"
        and raw_space.get("pass") is True
        and raw_space.get("actor_original_dimension") == 65793
        and raw_space.get("anchor_row_count") == 50
        and raw_space.get("local_physical_row_count") == 41,
        "precision_scope_pass": isinstance(precision, dict)
        and precision.get("pass") is True,
        "graph_fingerprint_pass": isinstance(graph, dict)
        and graph.get("pass") is True,
        "reconstruction_audit_pass": isinstance(reconstruction_audit, dict)
        and reconstruction_audit.get("pass") is True,
        "source_sha_exact": isinstance(source_audit, dict)
        and source_audit.get("source_sha256") == BUILDER_SHA256,
        "zero_changed_model_official_eval": value.get(
            "changed_model_official_evaluation_count"
        )
        == 0,
        "no_changed_candidate_consumer": value.get(
            "changed_model_candidate_consumer_called"
        )
        is False,
        "no_writes": value.get("writes_performed") == 0,
        "no_submission": value.get("submission_performed") is False,
    }
    return {"checks": checks, "pass": all(checks.values())}


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


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    validate_hex64(BUILDER_SHA256, "frozen BUILDER_SHA256")
    builder = DEFAULT_BUILDER.resolve()
    if builder != DEFAULT_BUILDER or builder.parent != TOOLS.resolve():
        raise LauncherError("frozen builder path identity drift")
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
    builder_evidence = regular_evidence(
        builder, expected_sha256=BUILDER_SHA256, expected_mode=FROZEN_MODE
    )
    command = [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        str(builder),
        "--mode",
        "run",
    ]
    marker = {
        "schema_version": SCHEMA,
        "status": "attempt_committed_before_child_creation",
        "launcher": launcher_evidence,
        "builder": builder_evidence,
        "command": command,
        "cwd": str(ROOT),
        "outputs": {
            "stdout": str(CHILD_STDOUT.relative_to(ROOT)),
            "stderr": str(CHILD_STDERR.relative_to(ROOT)),
            "execution": str(EXECUTION.relative_to(ROOT)),
        },
        "single_attempt_no_retry": True,
        "network_upload_submission": False,
        "changed_model_official_evaluation_count_expected": 0,
    }
    create_exclusive(ATTEMPT, canonical_json(marker), EVIDENCE_MODE)

    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    stdout_handle: Any | None = None
    stderr_handle: Any | None = None
    completed_returncode: int | None = None
    launch_error: str | None = None
    seal_errors: list[str] = []
    try:
        stdout_handle = open_exclusive_capture(CHILD_STDOUT)
        stderr_handle = open_exclusive_capture(CHILD_STDERR)
        # Recheck the pathname immediately before exec; builder stdout also binds
        # its own source SHA, which is validated below.
        regular_evidence(
            builder, expected_sha256=BUILDER_SHA256, expected_mode=FROZEN_MODE
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
    capture_evidence_errors: list[str] = []
    for label, path in (("stdout", CHILD_STDOUT), ("stderr", CHILD_STDERR)):
        if not path.exists():
            capture_evidence_errors.append(f"{label}:missing_after_capture")
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
            capture_evidence_errors.append(
                f"{label}:{type(exc).__name__}:{exc}"
            )
    result_validation: dict[str, Any] = {
        "pass": False,
        "checks": {},
        "error": "child did not complete successfully",
    }
    if (
        completed_returncode == 0
        and launch_error is None
        and not seal_errors
        and not capture_evidence_errors
        and stdout_evidence is not None
    ):
        try:
            stdout_payload = CHILD_STDOUT.read_bytes()
            if len(stdout_payload) > 256 * 1024 * 1024:
                raise LauncherError("builder stdout exceeds 256 MiB")
            if (
                len(stdout_payload) != stdout_evidence["bytes"]
                or sha256_bytes(stdout_payload) != stdout_evidence["sha256"]
            ):
                raise LauncherError("builder stdout changed after evidence hash")
            result_validation = validate_builder_result(
                strict_json_object(stdout_payload)
            )
            if result_validation["pass"] is not True:
                result_validation["error"] = "builder semantic result failed"
        except BaseException as exc:
            result_validation = {
                "pass": False,
                "checks": {},
                "error": f"{type(exc).__name__}:{exc}",
            }
    builder_post_evidence: dict[str, Any] | None = None
    try:
        builder_post_evidence = regular_evidence(
            builder, expected_sha256=BUILDER_SHA256, expected_mode=FROZEN_MODE
        )
    except BaseException as exc:
        result_validation = {
            "pass": False,
            "checks": result_validation.get("checks", {}),
            "error": f"post-child builder drift:{type(exc).__name__}:{exc}",
        }
    validated_success = (
        completed_returncode == 0
        and launch_error is None
        and not seal_errors
        and not capture_evidence_errors
        and result_validation.get("pass") is True
        and builder_post_evidence is not None
        and stdout_evidence is not None
        and stderr_evidence is not None
    )
    execution = {
        "schema_version": SCHEMA,
        "status": "validated_fixture_built" if validated_success else "child_failed",
        "child_returncode": completed_returncode,
        "launch_error": launch_error,
        "seal_errors": seal_errors,
        "capture_evidence_errors": capture_evidence_errors,
        "result_validation": result_validation,
        "attempt": regular_evidence(ATTEMPT, expected_mode=EVIDENCE_MODE),
        "launcher": launcher_evidence,
        "builder": builder_evidence,
        "builder_post_child": builder_post_evidence,
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
    return 0 if validated_success else 1


if __name__ == "__main__":
    raise SystemExit(main())

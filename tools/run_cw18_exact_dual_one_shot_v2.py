#!/usr/bin/env python3
"""Run frozen CW18 exact-dual solving with the fixture's BLAS contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "run_cw18_exact_dual_one_shot_v2.py"
V1_SOURCE = TOOLS / "run_cw18_exact_dual_one_shot_v1.py"
V1_SHA256 = "e0bcfc7eff037c5eaaaa6f1bc9bf5b3632807f91ae1d0ffa7f0a73148dbbff42"

ATTEMPT = ROOT / "artifacts/.ptcg-cw18_exact_dual_20260803_v2-attempt.json"
CHILD_STDOUT = ROOT / "artifacts/cw18_exact_dual_20260803_v2.stdout.json"
CHILD_STDERR = ROOT / "artifacts/cw18_exact_dual_20260803_v2.stderr.txt"
EXECUTION = ROOT / "artifacts/cw18_exact_dual_20260803_v2.execution.json"

SCHEMA = "ptcg-cw18-exact-dual-one-shot-launcher-v2"
FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444
BLAS_THREADS = "32"
EXPECTED_K_SHA256 = "eddaf72274061f63250861fc30e03c9ad610cb4f2296f62abd833290d971e9a6"
EXPECTED_Q_SHA256 = "6e7a8ef84d9b0a3ff8c1ff865a4d4da10a9bb7b549bfa10332e68ed0104afb1e"


class LauncherError(RuntimeError):
    """Fail-closed v2 launcher contract error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_source_evidence(path: Path, expected_sha256: str) -> dict[str, Any]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != FROZEN_MODE
    ):
        raise LauncherError(f"source identity or mode drift: {path}")
    observed_sha = sha256_file(path)
    after = path.lstat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or int(after.st_nlink) != 1
        or stat.S_IMODE(after.st_mode) != FROZEN_MODE
        or observed_sha != expected_sha256
    ):
        raise LauncherError(f"source changed or SHA drifted: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed_sha,
        "bytes": int(after.st_size),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "nlink": int(after.st_nlink),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
    }


def import_v1() -> tuple[ModuleType, dict[str, Any]]:
    evidence = frozen_source_evidence(V1_SOURCE, V1_SHA256)
    spec = importlib.util.spec_from_file_location("cw18_solver_launcher_v1_frozen", V1_SOURCE)
    if spec is None or spec.loader is None:
        raise LauncherError("cannot create frozen v1 launcher module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity_checks = {
        "root_exact": module.ROOT == ROOT,
        "python_exact": module.EXPECTED_PYTHON
        == Path("/home/xxc/miniconda3/envs/my_project_env/bin/python"),
        "solver_sha_exact": module.SOLVER_SHA256
        == "1f92b7b58aa6faa095f7f40fef62aee35ed3ee906c3faa4703198d31cda44310",
        "fixture_sha_exact": module.FIXTURE_SHA256
        == "a0d3ecc5a1315160b8b06ed212af9c6b6179c13695488c44cf5e445e043cd929",
        "fixture_internal_sha_exact": module.FIXTURE_INTERNAL_SHA256
        == "ef6c7bfa1017ea9ceb77b24543a2a9507bdd48873c7b3a21865691684987c8ec",
        "solved_status_exact": module.SOLVED_STATUS
        == "solved_formally_certified_CW18_exact_dual_candidate",
        "no_candidate_status_exact": module.NO_CANDIDATE_STATUS
        == "complete_no_formally_certified_CW18_candidate",
        "evidence_mode_exact": module.EVIDENCE_MODE == EVIDENCE_MODE,
    }
    if not all(identity_checks.values()):
        raise LauncherError(f"frozen v1 semantic identity drift: {identity_checks}")
    return module, {**evidence, "identity_checks": identity_checks}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    if Path(__file__).resolve() != SCRIPT.resolve():
        raise LauncherError("launcher must execute from its frozen SCRIPT path")
    if Path.cwd().resolve() != ROOT:
        raise LauncherError("launcher cwd drift")
    expected_python = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise LauncherError("launcher is not using my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise LauncherError("launcher requires Python -I -B")
    if stat.S_IMODE(SCRIPT.lstat().st_mode) != FROZEN_MODE:
        raise LauncherError("launcher source must be frozen mode 0555")
    targets = (ATTEMPT, CHILD_STDOUT, CHILD_STDERR, EXECUTION)
    existing = [str(path.relative_to(ROOT)) for path in targets if path.exists()]
    if existing:
        raise LauncherError(f"one-shot v2 target already exists: {existing}")

    v1, v1_evidence = import_v1()
    launcher_evidence = v1.regular_evidence(SCRIPT, expected_mode=FROZEN_MODE)
    solver_evidence = v1.regular_evidence(
        v1.SOLVER,
        expected_sha256=v1.SOLVER_SHA256,
        expected_mode=FROZEN_MODE,
    )
    fixture_evidence = v1.regular_evidence(
        v1.FIXTURE,
        expected_sha256=v1.FIXTURE_SHA256,
        expected_mode=EVIDENCE_MODE,
    )
    command = [
        str(expected_python),
        "-I",
        "-B",
        str(v1.SOLVER),
        "--mode",
        "solve",
        "--fixture-sha256",
        v1.FIXTURE_SHA256,
    ]
    numerical_contract = {
        "blas_threads": int(BLAS_THREADS),
        "reason": "match_the_frozen_fixture_builder_default_OpenBLAS_environment",
        "expected_rebuilt_K_float64_le_sha256": EXPECTED_K_SHA256,
        "expected_rebuilt_q_float64_le_sha256": EXPECTED_Q_SHA256,
        "solver_enforces_bit_exact_rebuild_from_raw_contract": True,
        "v1_single_thread_failure_preserved": {
            "execution": "artifacts/cw18_exact_dual_20260803_v1.execution.json",
            "stdout_sha256": "2689b0bf45d502c783dc64366fa05e370b1e3921baa66afa1e826f814171aa16",
            "failure_before_optimization": True,
        },
    }
    marker = {
        "schema_version": SCHEMA,
        "status": "attempt_committed_before_child_creation",
        "launcher": launcher_evidence,
        "frozen_v1_validation_library": v1_evidence,
        "solver": solver_evidence,
        "fixture": fixture_evidence,
        "command": command,
        "cwd": str(ROOT),
        "outputs": {
            "stdout": str(CHILD_STDOUT.relative_to(ROOT)),
            "stderr": str(CHILD_STDERR.relative_to(ROOT)),
            "execution": str(EXECUTION.relative_to(ROOT)),
        },
        "numerical_contract": numerical_contract,
        "single_attempt_no_retry": True,
        "cpu_only_no_model_no_official_eval": True,
        "network_upload_submission": False,
    }
    v1.create_exclusive(ATTEMPT, v1.canonical_json(marker), EVIDENCE_MODE)

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": BLAS_THREADS,
            "OPENBLAS_NUM_THREADS": BLAS_THREADS,
            "MKL_NUM_THREADS": BLAS_THREADS,
            "NUMEXPR_NUM_THREADS": BLAS_THREADS,
        }
    )
    stdout_handle: Any | None = None
    stderr_handle: Any | None = None
    completed_returncode: int | None = None
    launch_error: str | None = None
    seal_errors: list[str] = []
    try:
        stdout_handle = v1.open_exclusive_capture(CHILD_STDOUT)
        stderr_handle = v1.open_exclusive_capture(CHILD_STDERR)
        v1.regular_evidence(
            v1.SOLVER,
            expected_sha256=v1.SOLVER_SHA256,
            expected_mode=FROZEN_MODE,
        )
        v1.regular_evidence(
            v1.FIXTURE,
            expected_sha256=v1.FIXTURE_SHA256,
            expected_mode=EVIDENCE_MODE,
        )
        frozen_source_evidence(V1_SOURCE, V1_SHA256)
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
            v1.seal_capture(stdout_handle, "stdout"),
            v1.seal_capture(stderr_handle, "stderr"),
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
            evidence = v1.regular_evidence(path, expected_mode=EVIDENCE_MODE)
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
            if stdout_evidence["bytes"] > v1.MAX_STDOUT_BYTES:
                raise LauncherError("solver stdout exceeds frozen 256 MiB limit")
            stdout_payload = CHILD_STDOUT.read_bytes()
            if (
                len(stdout_payload) != stdout_evidence["bytes"]
                or v1.sha256_bytes(stdout_payload) != stdout_evidence["sha256"]
            ):
                raise LauncherError("solver stdout changed after evidence hash")
            validation = v1.validate_solver_result(
                v1.strict_json_object(stdout_payload), completed_returncode
            )
            if validation["pass"] is not True:
                validation["error"] = "solver semantic result failed"
        except BaseException as exc:
            validation = {
                "pass": False,
                "checks": {},
                "error": f"{type(exc).__name__}:{exc}",
            }

    post_errors: list[str] = []
    solver_post: dict[str, Any] | None = None
    fixture_post: dict[str, Any] | None = None
    v1_post: dict[str, Any] | None = None
    try:
        solver_post = v1.regular_evidence(
            v1.SOLVER,
            expected_sha256=v1.SOLVER_SHA256,
            expected_mode=FROZEN_MODE,
        )
    except BaseException as exc:
        post_errors.append(f"solver:{type(exc).__name__}:{exc}")
    try:
        fixture_post = v1.regular_evidence(
            v1.FIXTURE,
            expected_sha256=v1.FIXTURE_SHA256,
            expected_mode=EVIDENCE_MODE,
        )
    except BaseException as exc:
        post_errors.append(f"fixture:{type(exc).__name__}:{exc}")
    try:
        v1_post = frozen_source_evidence(V1_SOURCE, V1_SHA256)
    except BaseException as exc:
        post_errors.append(f"v1_validation:{type(exc).__name__}:{exc}")

    validated = (
        validation.get("pass") is True
        and launch_error is None
        and not seal_errors
        and not capture_errors
        and not post_errors
        and solver_post is not None
        and fixture_post is not None
        and v1_post is not None
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
        "attempt": v1.regular_evidence(ATTEMPT, expected_mode=EVIDENCE_MODE),
        "launcher": launcher_evidence,
        "frozen_v1_validation_library": v1_evidence,
        "frozen_v1_validation_library_post_child": v1_post,
        "solver": solver_evidence,
        "solver_post_child": solver_post,
        "fixture": fixture_evidence,
        "fixture_post_child": fixture_post,
        "stdout": stdout_evidence,
        "stderr": stderr_evidence,
        "command": command,
        "cwd": str(ROOT),
        "numerical_contract": numerical_contract,
        "single_attempt_no_retry": True,
    }
    v1.create_exclusive(EXECUTION, v1.canonical_json(execution), EVIDENCE_MODE)
    final = {
        **execution,
        "execution": v1.regular_evidence(EXECUTION, expected_mode=EVIDENCE_MODE),
    }
    print(v1.canonical_json(final).decode("utf-8"), end="")
    if not validated:
        return 1
    return 0 if outcome == "formally_certified_candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())

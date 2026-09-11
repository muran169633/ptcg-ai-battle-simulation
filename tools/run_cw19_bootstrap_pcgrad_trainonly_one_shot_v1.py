#!/usr/bin/env python3
"""One-shot launcher for the frozen CW19 train-only PCGrad preflight."""

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
SCRIPT = TOOLS / "run_cw19_bootstrap_pcgrad_trainonly_one_shot_v1.py"
SOLVER = TOOLS / "probe_u468_cw11_bootstrap_pcgrad_specialbc_cw19_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-cw19-bootstrap-pcgrad-trainonly-one-shot-launcher-v1"

ATTEMPT = ROOT / "artifacts/.ptcg-cw19_bootstrap_pcgrad_trainonly_20260803_v1-attempt.json"
CHILD_STDOUT = ROOT / "artifacts/cw19_bootstrap_pcgrad_trainonly_20260803_v1.stdout.json"
CHILD_STDERR = ROOT / "artifacts/cw19_bootstrap_pcgrad_trainonly_20260803_v1.stderr.txt"
EXECUTION = ROOT / "artifacts/cw19_bootstrap_pcgrad_trainonly_20260803_v1.execution.json"

FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444
SOLVER_SHA256 = "65f16009641481cd13538714520828640d00aed42aae920e24d9b70910423ae8"
DIRECT_INPUTS: dict[Path, tuple[str, int]] = {
    SOLVER: (SOLVER_SHA256, 0o555),
    TOOLS / "probe_u468_raw_aggregate512_actor6_gradients.py": (
        "ed85b59bd69558ab6f027ba21df9cb56c744baf9793bbb6b550d46bffbc3389a",
        0o555,
    ),
    TOOLS / "run_u468_raw_trainhard_actor6_balanced_mix_sweep.py": (
        "419feeeb5644adf34a8d56142cfcd87b3a4b79446d95b9b566bf9fcabdb721b1",
        0o555,
    ),
    TOOLS / "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py": (
        "2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24",
        0o555,
    ),
    TOOLS / "cw17_stage2_dual_math_v1.py": (
        "a14241d500f352e6ae1b5662d5d7ea51ca4da22b61126536997e69e39750a600",
        0o555,
    ),
    ROOT / "artifacts/cw18_direct_fixture_20260803_v1.stdout.json": (
        "a0d3ecc5a1315160b8b06ed212af9c6b6179c13695488c44cf5e445e043cd929",
        0o444,
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
        str(path.relative_to(ROOT)): regular_evidence(path, sha, mode)
        for path, (sha, mode) in DIRECT_INPUTS.items()
    }


def assert_snapshot_stable(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, bool]:
    checks = {
        key: before[key]["sha256"] == after[key]["sha256"]
        and before[key]["bytes"] == after[key]["bytes"]
        and before[key]["mode_octal"] == after[key]["mode_octal"]
        and before[key]["device"] == after[key]["device"]
        and before[key]["inode"] == after[key]["inode"]
        for key in before
    }
    if set(before) != set(after) or not all(checks.values()):
        raise LauncherError(f"direct input changed across child: {checks}")
    return checks


def exclusive_json(path: Path, value: Any, mode: int = 0o600) -> dict[str, Any]:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
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
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
    }


def output_evidence(path: Path) -> dict[str, Any]:
    return regular_evidence(path, None, EVIDENCE_MODE)


def validate_child_reported_inputs_post(document: dict[str, Any]) -> dict[str, Any]:
    root = document.get("integrity", {}).get("frozen_inputs")
    if not isinstance(root, dict):
        raise LauncherError("child frozen-input evidence is missing")
    records: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if {
                "path",
                "sha256",
                "bytes",
            }.issubset(value) and isinstance(value.get("path"), str):
                key = str(value["path"])
                prior = records.get(key)
                if prior is not None and (
                    str(prior["sha256"]) != str(value["sha256"])
                    or int(prior["bytes"]) != int(value["bytes"])
                ):
                    raise LauncherError(f"conflicting child evidence for {key}")
                records[key] = value
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(root)
    if len(records) < 17:
        raise LauncherError(
            f"child reported too few consumed input identities: {len(records)}"
        )
    audits: dict[str, Any] = {}
    for key, record in sorted(records.items()):
        declared = Path(key)
        path = declared if declared.is_absolute() else ROOT / declared
        resolved = path.resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError as error:
            raise LauncherError(f"reported input is outside workspace: {key}") from error
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
            "nlink_exact_if_reported": "nlink" not in record
            or int(observed.st_nlink) == int(record["nlink"]),
            "mode_exact_if_reported": mode_record is None
            or stat.S_IMODE(observed.st_mode) == int(str(mode_record), 8),
        }
        if not all(checks.values()):
            raise LauncherError(f"child-reported input changed: {key}: {checks}")
        audits[key] = {
            "sha256": str(record["sha256"]),
            "bytes": int(record["bytes"]),
            "checks": checks,
        }
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
        raise LauncherError(
            f"child consumed-input evidence lacks required cache inputs: {suffix_checks}"
        )
    return {
        "pass": True,
        "reported_identity_count": len(records),
        "required_cache_input_suffixes": suffix_checks,
        "post_run_audits": audits,
    }


def validate_child(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise LauncherError("child stdout is not a JSON object")
    decision = document.get("decision")
    allowed = {
        "GO_CW19_SPECIALIST_PREFLIGHT",
        "NO_GO_CW19_PREFLIGHT",
    }
    contract = document.get("contract", {})
    endpoint = document.get("endpoint", {})
    historical = document.get("historical_exact_CW11_replay", {})
    expected_final_check_keys = {
        "planned_contract",
        "actual_FP32_contract",
        "train_gate",
        "nonactor_still_exact",
        "pure_payload_reconstruction",
    }
    final_checks = endpoint.get("final_checks")
    go_final_checks_exact = (
        isinstance(final_checks, dict)
        and set(final_checks) == expected_final_check_keys
        and all(final_checks[key] is True for key in expected_final_check_keys)
    )
    checks = {
        "schema_exact": document.get("schema_version")
        == "ptcg-u468-cw11-bootstrap-pcgrad-specialbc-cw19-v1",
        "status_decision_exact": document.get("status") == decision,
        "decision_allowed": decision in allowed,
        "historical_replay_pass": historical.get("pass") is True,
        "new_candidate_official_count_zero": contract.get(
            "changed_candidate_official_or_validation_evaluation_count"
        )
        == 0,
        "official_budget_zero": contract.get("official_candidate_budget_consumed") == 0,
        "no_checkpoint_or_model_write": contract.get("checkpoint_writes") == 0
        and contract.get("model_writes") == 0,
        "no_result_write_inside_child": contract.get("result_artifact_writes") == 0,
        "no_optimizer_or_step": contract.get("optimizer_instances_created") == 0
        and contract.get("optimizer_step_calls") == 0
        and contract.get("backward_calls") == 0,
        "stdout_only": contract.get("stdout_only") is True,
        "submission_false": document.get("submission_performed") is False,
        "writes_zero": document.get("writes_performed") == 0,
        "endpoint_payload_consistent": (
            decision == "GO_CW19_SPECIALIST_PREFLIGHT"
            and endpoint.get("candidate_payload") is not None
            and endpoint.get("changed_candidate_train_endpoint_count") == 1
            and go_final_checks_exact
        )
        or (
            decision == "NO_GO_CW19_PREFLIGHT"
            and endpoint.get("candidate_payload") is None
            and endpoint.get("changed_candidate_train_endpoint_count") in {0, 1}
        ),
    }
    if not all(checks.values()):
        raise LauncherError(f"child semantic validation failed: {checks}")
    reported_inputs_post = validate_child_reported_inputs_post(document)
    return {
        "checks": checks,
        "pass": True,
        "decision": decision,
        "expected_GO_final_check_keys": sorted(expected_final_check_keys),
        "child_reported_consumed_inputs_post": reported_inputs_post,
    }


def static_audit() -> dict[str, Any]:
    targets = (ATTEMPT, CHILD_STDOUT, CHILD_STDERR, EXECUTION)
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "launcher_at_expected_path": Path(__file__).resolve() == SCRIPT.resolve(),
        "targets_absent": not any(path.exists() for path in targets),
    }
    if not all(checks.values()):
        raise LauncherError(f"launcher static checks failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_passed",
        "checks": checks,
        "pass": True,
        "launcher": regular_evidence(SCRIPT, None, stat.S_IMODE(SCRIPT.lstat().st_mode)),
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
        "one_shot_targets": [str(path.relative_to(ROOT)) for path in targets],
        "writes_performed": 0,
        "submission_performed": False,
    }


def run_once() -> int:
    if stat.S_IMODE(SCRIPT.lstat().st_mode) != FROZEN_MODE:
        raise LauncherError("launcher must be frozen mode 0555")
    audit = static_audit()
    launcher_evidence = regular_evidence(SCRIPT, None, FROZEN_MODE)
    pre_inputs = audit["direct_inputs"]
    command = audit["command"]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OPENBLAS_NUM_THREADS": "32",
            "MKL_NUM_THREADS": "32",
            "OMP_NUM_THREADS": "32",
            "NUMEXPR_NUM_THREADS": "32",
        }
    )
    marker = {
        "schema_version": SCHEMA,
        "status": "attempt_committed_before_child_creation",
        "launcher": launcher_evidence,
        "direct_inputs_pre": pre_inputs,
        "command": command,
        "cwd": str(ROOT),
        "environment_contract": {
            key: environment[key]
            for key in (
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "new_candidate_official_budget_consumed_before": 0,
        "submission_performed": False,
    }
    exclusive_json(ATTEMPT, marker)
    return_code: int | None = None
    validation: dict[str, Any] | None = None
    parse_error: str | None = None
    try:
        with CHILD_STDOUT.open("xb") as stdout_handle, CHILD_STDERR.open(
            "xb"
        ) as stderr_handle:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
            stdout_handle.flush()
            os.fsync(stdout_handle.fileno())
            stderr_handle.flush()
            os.fsync(stderr_handle.fileno())
        return_code = int(process.returncode)
        if return_code == 0:
            try:
                document = json.loads(CHILD_STDOUT.read_text(encoding="utf-8"))
                validation = validate_child(document)
            except BaseException as error:
                parse_error = f"{type(error).__name__}: {error}"
        else:
            parse_error = f"child_return_code_{return_code}"
    finally:
        for path in (ATTEMPT, CHILD_STDOUT, CHILD_STDERR):
            if path.exists():
                path.chmod(EVIDENCE_MODE)

    post_inputs = input_snapshot()
    stable = assert_snapshot_stable(pre_inputs, post_inputs)
    launcher_post = regular_evidence(
        SCRIPT, launcher_evidence["sha256"], FROZEN_MODE
    )
    execution = {
        "schema_version": SCHEMA,
        "status": (
            "one_shot_completed_validated"
            if return_code == 0 and validation is not None
            else "one_shot_failed_preserved"
        ),
        "return_code": return_code,
        "child_validation": validation,
        "child_error": parse_error,
        "launcher": launcher_evidence,
        "launcher_post": launcher_post,
        "direct_inputs_pre": pre_inputs,
        "direct_inputs_post": post_inputs,
        "direct_inputs_stable": stable,
        "attempt": output_evidence(ATTEMPT),
        "stdout": output_evidence(CHILD_STDOUT),
        "stderr": output_evidence(CHILD_STDERR),
        "new_candidate_official_budget_consumed": 0,
        "submission_performed": False,
        "rerun_permitted": False,
    }
    exclusive_json(EXECUTION, execution)
    EXECUTION.chmod(EVIDENCE_MODE)
    if return_code != 0 or validation is None:
        raise LauncherError(
            f"one-shot child failed; preserved evidence: rc={return_code}, {parse_error}"
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


def preserve_unclassified_failure(error: BaseException) -> None:
    """Best-effort immutable evidence for failures outside the normal path."""
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
                "evidence_error": (
                    f"{type(evidence_error).__name__}: {evidence_error}"
                ),
            }
    try:
        direct_post: Any = input_snapshot()
    except BaseException as snapshot_error:
        direct_post = {
            "snapshot_error": f"{type(snapshot_error).__name__}: {snapshot_error}"
        }
    failure = {
        "schema_version": SCHEMA,
        "status": "one_shot_unclassified_failure_preserved",
        "error_type": type(error).__name__,
        "error": str(error),
        "files": files,
        "direct_inputs_post_or_error": direct_post,
        "new_candidate_official_budget_consumed": 0,
        "submission_performed": False,
        "rerun_permitted": False,
    }
    try:
        exclusive_json(EXECUTION, failure)
        EXECUTION.chmod(EVIDENCE_MODE)
    except BaseException:
        # The O_EXCL attempt marker and any child logs remain as the lower-level
        # evidence even if the filesystem rejects the final classification.
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
        print(json.dumps(static_audit(), sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    try:
        return run_once()
    except BaseException as error:
        preserve_unclassified_failure(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

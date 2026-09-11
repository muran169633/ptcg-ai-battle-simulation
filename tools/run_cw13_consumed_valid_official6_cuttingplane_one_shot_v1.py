#!/usr/bin/env python3
"""Consume exactly one CW13 official-six cutting-plane attempt.

CW13 inherits the frozen, independently audited one-shot implementation from
the final CW12 launcher.  This adapter hash-binds that implementation, replaces
only the solver/schema/artifact bindings, requires the CW13 solver's exact
31-input static manifest, and independently binds the consumed CW12 marker and
failure stdout.  Run mode remains fail-closed until the final CW13 solver SHA
is installed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
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
SCRIPT = TOOLS / "run_cw13_consumed_valid_official6_cuttingplane_one_shot_v1.py"
SOLVER = TOOLS / "probe_u468_cw11_consumed_valid_official6_cw13_cuttingplane_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_EXECUTABLE_MODE = 0o555

PARENT_LAUNCHER = TOOLS / (
    "run_cw12_consumed_valid_official6_cuttingplane_one_shot_v1.py"
)
PARENT_LAUNCHER_SHA256 = (
    "169e29e4e47e23e46c421264d0b7b0c5a6cf8a911506235d864ccbca25994050"
)

CW12_ATTEMPT_MARKER = ARTIFACTS / (
    ".ptcg-cw12_consumed_valid_official6_cuttingplane_20260802_v1-attempt.json"
)
CW12_ATTEMPT_MARKER_SHA256 = (
    "e8285105d1c50c6608e020433bd82425c7c6debeb357cfc82993db2f90a4ab3a"
)
CW12_FAILURE_STDOUT = ARTIFACTS / (
    "cw12_consumed_valid_official6_cuttingplane_20260802_v1.stdout.json"
)
CW12_FAILURE_STDOUT_SHA256 = (
    "dd7367f10112b70ae3264c005e52f839ec83f440dea66da827f9ec3306004b43"
)

SCHEMA = "ptcg-cw13-consumed-valid-official6-one-shot-launcher-v1"
ATTEMPT_SCHEMA = "ptcg-cw13-consumed-valid-official6-one-shot-attempt-v1"
FAILURE_SCHEMA = "ptcg-cw13-consumed-valid-official6-one-shot-terminal-failure-v1"
STDERR_SCHEMA = "ptcg-cw13-consumed-valid-official6-one-shot-stderr-audit-v1"
SOLVER_SCHEMA = "ptcg-cw13-consumed-valid-official-b256-cuttingplane-v1"

ARTIFACT_ID = "cw13_consumed_valid_official6_cuttingplane_20260802_v1"
ATTEMPT_MARKER = ARTIFACTS / f".ptcg-{ARTIFACT_ID}-attempt.json"
STDOUT_OUTPUT = ARTIFACTS / f"{ARTIFACT_ID}.stdout.json"
STDERR_AUDIT = ARTIFACTS / f"{ARTIFACT_ID}.stderr-audit.json"

# Exact final independently audited CW13 solver lock.
EXPECTED_SOLVER_SHA256 = (
    "dbbdc12e7c2f571f82de92d30450d8ac794c7215ee86137b9aa97ebc42d28297"
)
EXPECTED_STATIC_STATUS = "static_ready_CW13_run_implemented"
EXPECTED_TERMINAL_STATUSES = (
    "consumed_valid_CW13_optimization_closure_first_feasible",
    "closed_no_CW13_candidate",
)
EXPECTED_SOLVER_FROZEN_BINDINGS = 31

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
    """The inherited launcher or CW13 binding is invalid."""


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
    """Execute the exact held-fd bytes already matched to the parent lock."""

    module_name = f"_cw13_parent_one_shot_{PARENT_LAUNCHER_SHA256[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(PARENT_LAUNCHER)
    module.__package__ = ""
    module.__spec__ = None
    code = compile(source, str(PARENT_LAUNCHER), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


PARENT_SOURCE, PARENT_EVIDENCE = read_frozen_parent()
parent = import_parent(PARENT_SOURCE)
PARENT_SOURCE_AUDIT = parent.static_source_audit(PARENT_SOURCE)
if PARENT_SOURCE_AUDIT.get("pass") is not True:
    raise ProtocolError("frozen parent launcher source audit failed")

_PARENT_PREFLIGHT = parent.preflight
_PARENT_STDERR_AUDIT = parent.validated_stderr_audit


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
        "parent_executed_from_exact_held_bytes": "compile" in calls
        and "exec" in calls
        and "spec_from_file_location" not in calls,
        "no_forbidden_imports": not bool(imported_roots.intersection(forbidden_imports)),
        "no_direct_subprocess_call": "run" not in calls,
        "cli_only_mode": not output_arguments,
        "solver_path_exact": SOLVER.name
        == "probe_u468_cw11_consumed_valid_official6_cw13_cuttingplane_v1.py",
        "solver_manifest_count_exact_31": EXPECTED_SOLVER_FROZEN_BINDINGS == 31,
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
            == ".ptcg-cw13_consumed_valid_official6_cuttingplane_20260802_v1-attempt.json"
            and STDOUT_OUTPUT.name
            == "cw13_consumed_valid_official6_cuttingplane_20260802_v1.stdout.json"
            and STDERR_AUDIT.name
            == "cw13_consumed_valid_official6_cuttingplane_20260802_v1.stderr-audit.json"
        ),
        "targets_distinct_under_artifacts": len(
            {ATTEMPT_MARKER, STDOUT_OUTPUT, STDERR_AUDIT}
        )
        == 3
        and ATTEMPT_MARKER.parent
        == STDOUT_OUTPUT.parent
        == STDERR_AUDIT.parent
        == ARTIFACTS,
        "cw12_failure_bindings_literal": CW12_ATTEMPT_MARKER_SHA256 in text
        and CW12_FAILURE_STDOUT_SHA256 in text,
        "inherited_one_shot_paths_called": "static_result" in calls
        and "run_once" in calls,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "parent_launcher": PARENT_EVIDENCE,
        "parent_source_audit": PARENT_SOURCE_AUDIT,
        "imported_roots": sorted(imported_roots),
        "unexpected_cli_arguments": output_arguments,
    }


def apply_cw13_overrides() -> None:
    cw13_run_scope = dict(parent.RUN_SCOPE)
    cw13_run_scope["CW13_favorable_transition_restoration"] = True
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
        "RUN_SCOPE": cw13_run_scope,
    }
    for name, value in values.items():
        setattr(parent, name, value)
    parent.static_source_audit = wrapper_source_audit


apply_cw13_overrides()


def known_cw12_failure_evidence() -> dict[str, Any]:
    marker = parent.rehash_regular_stable(
        CW12_ATTEMPT_MARKER,
        "consumed CW12 attempt marker",
        expected_sha256=CW12_ATTEMPT_MARKER_SHA256,
        expected_mode=0o444,
    )
    failure = parent.rehash_regular_stable(
        CW12_FAILURE_STDOUT,
        "immutable CW12 failure stdout",
        expected_sha256=CW12_FAILURE_STDOUT_SHA256,
        expected_mode=0o444,
    )
    return {
        "cw12_attempt_marker": marker,
        "cw12_failure_stdout": failure,
        "pass": True,
    }


def cw13_preflight(*, require_lock: bool) -> dict[str, Any]:
    evidence = _PARENT_PREFLIGHT(require_lock=require_lock)
    evidence["inherited_parent_launcher"] = PARENT_EVIDENCE
    evidence["known_cw12_failure_evidence"] = known_cw12_failure_evidence()
    if evidence.get("solver_lock_armed"):
        frozen = evidence.get("solver_static", {}).get("payload", {}).get(
            "frozen_inputs"
        )
        records = frozen.get("records") if isinstance(frozen, Mapping) else None
        if not isinstance(records, list):
            raise ProtocolError("armed CW13 solver did not emit its frozen manifest")
        by_path = {
            record.get("path"): record
            for record in records
            if isinstance(record, Mapping)
        }
        expected = {
            root_relative(CW12_ATTEMPT_MARKER): CW12_ATTEMPT_MARKER_SHA256,
            root_relative(CW12_FAILURE_STDOUT): CW12_FAILURE_STDOUT_SHA256,
        }
        checks = {
            "solver_manifest_count_exact_31": frozen.get("binding_count") == 31
            and len(records) == 31,
            "cw12_failure_records_exact": all(
                path in by_path
                and by_path[path].get("sha256") == digest
                and by_path[path].get("mode") == "0444"
                and by_path[path].get("nlink") == 1
                and by_path[path].get("regular") is True
                for path, digest in expected.items()
            ),
        }
        if not all(checks.values()):
            raise ProtocolError(f"CW13 failure-evidence manifest gate failed: {checks}")
        evidence["cw13_manifest_failure_bindings"] = {
            "checks": checks,
            "records": {path: by_path[path] for path in sorted(expected)},
            "pass": True,
        }
    return evidence


def cw13_stderr_audit(
    payload: bytes,
    *,
    post_child_integrity: Mapping[str, Any],
) -> bytes:
    inherited = _PARENT_STDERR_AUDIT(
        payload,
        post_child_integrity=post_child_integrity,
    )
    document = parent.strict_json_object(inherited, "inherited stderr audit")
    if document.get("schema_version") != (
        "ptcg-cw12-consumed-valid-official6-one-shot-stderr-audit-v1"
    ):
        raise ProtocolError("inherited stderr audit schema drift")
    document["schema_version"] = STDERR_SCHEMA
    document["inherited_parent_launcher"] = PARENT_EVIDENCE
    return parent.canonical_json(document)


parent.preflight = cw13_preflight
parent.validated_stderr_audit = cw13_stderr_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="static")
    args = parser.parse_args()
    result = parent.static_result() if args.mode == "static" else parent.run_once()
    print(parent.canonical_json(result).decode("utf-8"), end="")
    return (
        1
        if result.get("status") == "terminal_failure_recorded_attempt_consumed"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())

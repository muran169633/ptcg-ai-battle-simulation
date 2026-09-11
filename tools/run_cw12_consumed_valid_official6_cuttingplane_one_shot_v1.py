#!/usr/bin/env python3
"""Consume exactly one CW12 official-six cutting-plane attempt.

This launcher is the sole owner of the CW12 attempt marker and captured stdout
artifact.  It validates the frozen solver and every input through the solver's
zero-CUDA static mode before consuming the attempt marker.  Only then may it
invoke the solver's CUDA ``--mode run`` command once.  It never retries, never
changes the solver, and never accesses broad, Gold, package, upload, submission,
or network surfaces.

Until ``EXPECTED_SOLVER_SHA256`` and the terminal schema/status constants are
replaced with the independently audited final values, run mode fails closed
without creating either artifact.
"""

from __future__ import annotations

import argparse
import array
import ast
import base64
import hashlib
import json
import lzma
import math
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
ARTIFACTS = ROOT / "artifacts"
SCRIPT = TOOLS / "run_cw12_consumed_valid_official6_cuttingplane_one_shot_v1.py"
SOLVER = TOOLS / "probe_u468_cw11_consumed_valid_official6_cw12_cuttingplane_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_EXECUTABLE_MODE = 0o555
FROZEN_EVIDENCE_MODE = 0o444

SCHEMA = "ptcg-cw12-consumed-valid-official6-one-shot-launcher-v1"
ATTEMPT_SCHEMA = "ptcg-cw12-consumed-valid-official6-one-shot-attempt-v1"
FAILURE_SCHEMA = "ptcg-cw12-consumed-valid-official6-one-shot-terminal-failure-v1"
SOLVER_SCHEMA = "ptcg-cw12-consumed-valid-official-b256-cuttingplane-v1"

ARTIFACT_ID = "cw12_consumed_valid_official6_cuttingplane_20260802_v1"
ATTEMPT_MARKER = ARTIFACTS / f".ptcg-{ARTIFACT_ID}-attempt.json"
STDOUT_OUTPUT = ARTIFACTS / f"{ARTIFACT_ID}.stdout.json"
STDERR_AUDIT = ARTIFACTS / f"{ARTIFACT_ID}.stderr-audit.json"

# Fail-closed placeholders.  Replace only after the solver is complete, frozen,
# and independently audited.  A non-lowercase-hex value can never arm run mode.
EXPECTED_SOLVER_SHA256 = "e16eb0aa6d3ea0905309210126abd9f1f4cbb2f093b7e9dfdac49271466377ec"
EXPECTED_STATIC_STATUS = "static_ready_run_implemented"
EXPECTED_TERMINAL_STATUSES = (
    "consumed_valid_optimization_closure_first_feasible",
    "closed_no_CW12_candidate",
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

# Deliberately do not inherit the launcher's environment.  These are the exact
# variables passed to both child commands.  Absolute argv paths make PATH a
# compatibility aid only, not executable selection state.
CHILD_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "HOME": "/home/xxc",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": (
        "/home/xxc/miniconda3/envs/my_project_env/bin:"
        "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
}

CLASSIFICATION = {
    "specialist_valid_consumed": True,
    "dev_tuning_only": True,
    "promotion_evidence": False,
    "broad_access": False,
    "gold_access": False,
    "package_upload_submission": False,
}
RUN_SCOPE = {
    "exact_CW11_reconstruction": True,
    "authoritative_full_six_official_B256_oracle_each_proposal": True,
    "candidate_RAM_only": True,
    "optimizer_backward_training": False,
    "model_or_result_writes": 0,
    "network_upload_submission": False,
    "broad_or_gold_access": False,
    "specialist_valid_consumed_dev_only": True,
}
RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
RAW_NONACTOR_SHA256 = "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
CW11_VECTOR_SHA256 = "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
ACTOR6_LAYOUT_SHA256 = "b86476b9ccbbeeac7b46754f7f15e349e398aa627fc3dc6060d6c3d6e3bd26cb"
ACTOR6_FLAT_LENGTH = 65793
ACTOR6_FLOAT64_BYTES = 526344
ACTOR6_FLOAT32_BYTES = 263172
ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
MAX_OUTER_ITERATIONS = 12
STEP_L2_CAP = 0.001
ADDITIONAL_TOTAL_L2_CAP = 0.001
L2_CAP_ABS_TOL = 1e-12
EXPECTED_SOLVER_FROZEN_BINDINGS = 29


class ProtocolError(RuntimeError):
    """The one-shot protocol or a frozen identity is invalid."""


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def solver_lock_armed() -> bool:
    return (
        is_sha256(EXPECTED_SOLVER_SHA256)
        and not EXPECTED_STATIC_STATUS.startswith("PENDING_")
        and len(EXPECTED_TERMINAL_STATUSES) >= 1
        and all(
            isinstance(value, str) and value and not value.startswith("PENDING_")
            for value in EXPECTED_TERMINAL_STATUSES
        )
    )


def root_relative(path: Path) -> str:
    resolved = Path(os.path.abspath(os.fspath(path)))
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError as error:
        raise ProtocolError(f"path escapes repository: {resolved}") from error


def read_regular_stable(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ProtocolError(f"{label} must be a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        visible = os.lstat(path)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    stable = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        and not stat.S_ISLNK(visible.st_mode)
        and stat.S_ISREG(visible.st_mode)
        and int(visible.st_nlink) == 1
        and (
            visible.st_dev,
            visible.st_ino,
            visible.st_size,
            visible.st_mtime_ns,
            stat.S_IMODE(visible.st_mode),
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            stat.S_IMODE(after.st_mode),
        )
        and len(payload) == int(after.st_size)
    )
    if not stable:
        raise ProtocolError(f"{label} changed while being read")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProtocolError(
            f"{label} SHA-256 mismatch: observed {digest}, expected {expected_sha256}"
        )
    mode = stat.S_IMODE(after.st_mode)
    if expected_mode is not None and mode != expected_mode:
        raise ProtocolError(
            f"{label} mode mismatch: observed {mode:#o}, expected {expected_mode:#o}"
        )
    return payload, {
        "path": root_relative(path),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
    }


def rehash_regular_stable(
    path: Path,
    label: str,
    *,
    expected_sha256: str,
    expected_mode: int,
) -> dict[str, Any]:
    """Stream-rehash a potentially large frozen input without retaining it."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ProtocolError(f"{label} must be a single-link regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        visible = os.lstat(path)
    finally:
        os.close(descriptor)
    observed_sha = digest.hexdigest()
    checks = {
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
        "size_exact": total == int(after.st_size),
        "sha_exact": observed_sha == expected_sha256,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "one_link_regular": stat.S_ISREG(after.st_mode) and int(after.st_nlink) == 1,
        "visible_path_same_inode": not stat.S_ISLNK(visible.st_mode)
        and stat.S_ISREG(visible.st_mode)
        and int(visible.st_nlink) == 1
        and (
            visible.st_dev,
            visible.st_ino,
            visible.st_size,
            visible.st_mtime_ns,
            stat.S_IMODE(visible.st_mode),
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            stat.S_IMODE(after.st_mode),
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} independent rehash failed: {checks}")
    return {
        "path": root_relative(path),
        "sha256": observed_sha,
        "bytes": total,
        "mode_octal": format(expected_mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def absent_by_lstat(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    return False


def validate_target_parent(path: Path, label: str) -> None:
    parent = path.parent
    observed = os.lstat(parent)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise ProtocolError(f"{label} parent must be a real directory")
    if parent.resolve() != ARTIFACTS.resolve():
        raise ProtocolError(f"{label} parent must be the artifacts directory")


def validate_runtime() -> dict[str, bool]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated_exact": sys.flags.isolated == 1,
        "dont_write_bytecode_exact": sys.flags.dont_write_bytecode == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"launcher runtime contract failed: {checks}")
    return checks


def reject_constant(value: str) -> None:
    raise ProtocolError(f"non-standard JSON constant is forbidden: {value}")


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError(f"{label} is not UTF-8") from error
    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise ProtocolError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be one JSON object")
    return value


def environment_binding() -> dict[str, Any]:
    ordered = {key: CHILD_ENV[key] for key in sorted(CHILD_ENV)}
    return {
        "variables": ordered,
        "canonical_json_sha256": sha256_bytes(canonical_json(ordered)),
        "inherit_parent_environment": False,
    }


def argv_binding(argv: Sequence[str]) -> dict[str, Any]:
    command = list(argv)
    return {
        "argv": command,
        "canonical_json_sha256": sha256_bytes(canonical_json(command)),
        "cwd": str(ROOT),
        "stdin": "DEVNULL",
        "stdout": "PIPE",
        "stderr": "PIPE",
        "shell": False,
        "check": False,
        "retry_count": 0,
    }


def execute_child(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        cwd=str(ROOT),
        env=dict(CHILD_ENV),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        close_fds=True,
    )


def validate_static_payload(payload: bytes) -> dict[str, Any]:
    document = strict_json_object(payload, "solver static stdout")
    contract = document.get("contract")
    runtime = document.get("runtime")
    forensic = document.get("forensic_contract")
    expected_contract_subset = {
        "starting_model_sha256": CW11_MODEL_SHA256,
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
        "max_outer_iterations": MAX_OUTER_ITERATIONS,
        "per_step_l2_cap": STEP_L2_CAP,
        "additional_total_l2_cap_from_cw11": ADDITIONAL_TOTAL_L2_CAP,
        "l2_cap_absolute_tolerance": L2_CAP_ABS_TOL,
        "actor_parameter_names": list(ACTOR6_NAMES),
        "first_feasible": True,
        "candidate_ordering": "fixed_iteration_order_no_best_of_N",
        "terminal_semantics": "consumed_valid_optimization_closure_only",
    }
    checks = {
        "schema_exact": document.get("schema_version") == SOLVER_SCHEMA,
        "status_exact": document.get("status") == EXPECTED_STATIC_STATUS,
        "classification_exact": document.get("classification") == CLASSIFICATION,
        "runtime_checks_all_true": isinstance(runtime, Mapping)
        and all_true_checks(runtime.get("checks")),
        "frozen_inputs_present": isinstance(document.get("frozen_inputs"), (list, dict)),
        "forensic_contract_all_true": isinstance(forensic, Mapping)
        and all_true_checks(forensic.get("checks")),
        "contract_subset_exact": isinstance(contract, Mapping)
        and all(contract.get(key) == value for key, value in expected_contract_subset.items()),
        "source_audit_pass": document.get("source_audit", {}).get("pass") is True,
        "run_executed_false": document.get("run_executed") is False,
        "cuda_accessed_false": document.get("cuda_accessed") is False,
        "writes_performed_false": document.get("writes_performed") is False,
    }
    if not all(checks.values()):
        raise ProtocolError(f"solver static payload contract failed: {checks}")
    return {
        "checks": checks,
        "stdout_sha256": sha256_bytes(payload),
        "stdout_bytes": len(payload),
        "payload": document,
    }


def independently_rehash_solver_bindings(
    static_document: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = static_document.get("frozen_inputs")
    if not isinstance(frozen, Mapping):
        raise ProtocolError("solver frozen-input manifest is not an object")
    records = frozen.get("records")
    header_checks = {
        "all_exact_true": frozen.get("all_exact") is True,
        "binding_count_exact": frozen.get("binding_count")
        == EXPECTED_SOLVER_FROZEN_BINDINGS,
        "records_exact_count": isinstance(records, list)
        and len(records) == EXPECTED_SOLVER_FROZEN_BINDINGS,
    }
    if not all(header_checks.values()):
        raise ProtocolError(f"solver frozen-input manifest drift: {header_checks}")
    assert isinstance(records, list)
    seen: set[str] = set()
    evidence = []
    for ordinal, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ProtocolError(f"solver frozen binding {ordinal} is not an object")
        path_text = record.get("path")
        digest = record.get("sha256")
        mode_text = record.get("mode")
        if (
            not isinstance(path_text, str)
            or not path_text
            or Path(path_text).is_absolute()
            or path_text in seen
            or not is_sha256(digest)
            or not isinstance(mode_text, str)
            or len(mode_text) != 4
            or record.get("nlink") != 1
            or record.get("regular") is not True
        ):
            raise ProtocolError(f"solver frozen binding {ordinal} identity drift")
        try:
            mode = int(mode_text, 8)
        except ValueError as error:
            raise ProtocolError(
                f"solver frozen binding {ordinal} mode is invalid"
            ) from error
        path = Path(os.path.abspath(os.fspath(ROOT / path_text)))
        if root_relative(path) != path_text:
            raise ProtocolError(f"solver frozen binding {ordinal} path is not canonical")
        if path in {ATTEMPT_MARKER, STDOUT_OUTPUT, STDERR_AUDIT}:
            raise ProtocolError("solver frozen binding aliases a mutable output")
        seen.add(path_text)
        evidence.append(
            rehash_regular_stable(
                path,
                f"solver frozen input {ordinal}",
                expected_sha256=str(digest),
                expected_mode=mode,
            )
        )
    return {
        "header_checks": header_checks,
        "binding_count": len(evidence),
        "ordered_identity_sha256": sha256_bytes(
            canonical_json(
                [
                    {
                        "path": value["path"],
                        "sha256": value["sha256"],
                        "mode_octal": value["mode_octal"],
                    }
                    for value in evidence
                ]
            )
        ),
        "records": evidence,
        "pass": True,
    }


def all_true_checks(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(item is True for item in value.values())
    )


def decode_xz_base64_payload(
    value: Any,
    label: str,
    *,
    expected_dtype: str,
    expected_raw_bytes: int,
) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    chunks = value.get("base64_chunks_76")
    if (
        not isinstance(chunks, list)
        or not chunks
        or not all(isinstance(chunk, str) and chunk for chunk in chunks)
    ):
        raise ProtocolError(f"{label} base64 chunks are invalid")
    encoded = "".join(chunks)
    canonical_chunks = [
        encoded[index : index + 76] for index in range(0, len(encoded), 76)
    ]
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ProtocolError(f"{label} base64 is invalid") from error
    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    try:
        raw = decompressor.decompress(compressed)
    except lzma.LZMAError as error:
        raise ProtocolError(f"{label} XZ stream is invalid") from error
    checks = {
        "dtype_exact": value.get("payload_dtype") == expected_dtype,
        "raw_bytes_exact": value.get("raw_bytes") == expected_raw_bytes == len(raw),
        "raw_sha_exact": value.get("raw_sha256") == sha256_bytes(raw),
        "compression_exact": value.get("compression")
        == "XZ_preset9_extreme_CRC64",
        "compressed_bytes_exact": value.get("compressed_bytes") == len(compressed),
        "compressed_sha_exact": value.get("compressed_sha256")
        == sha256_bytes(compressed),
        "base64_variant_exact": value.get("base64_variant") == "standard_RFC4648",
        "base64_chunking_exact": chunks == canonical_chunks,
        "xz_reached_eof": decompressor.eof,
        "xz_has_no_trailing_stream": decompressor.unused_data == b"",
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} compressed payload contract failed: {checks}")
    return raw, checks


def float64_l2(raw: bytes, label: str) -> float:
    values = array.array("d")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != ACTOR6_FLAT_LENGTH or not all(
        math.isfinite(value) for value in values
    ):
        raise ProtocolError(f"{label} float64 vector shape/finite gate failed")
    return math.sqrt(math.fsum(value * value for value in values))


def validate_reconstruction_payload(
    reconstruction: Any,
    decision: Mapping[str, Any],
    terminal_iteration: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(reconstruction, Mapping):
        raise ProtocolError("successful terminal reconstruction payload is absent")
    anchor = reconstruction.get("anchor")
    layout = reconstruction.get("actor_layout")
    if not isinstance(anchor, Mapping) or not isinstance(layout, list):
        raise ProtocolError("terminal reconstruction anchor/layout is invalid")
    layout_names = []
    next_start = 0
    layout_rows_valid = len(layout) == len(ACTOR6_NAMES)
    for row in layout:
        if not isinstance(row, Mapping):
            layout_rows_valid = False
            continue
        name = row.get("name")
        shape = row.get("shape")
        numel = row.get("numel")
        start = row.get("start")
        stop = row.get("stop")
        layout_names.append(name)
        row_valid = (
            isinstance(shape, list)
            and shape
            and all(isinstance(item, int) and item > 0 for item in shape)
            and isinstance(numel, int)
            and numel > 0
            and math.prod(shape) == numel
            and start == next_start
            and stop == next_start + numel
        )
        layout_rows_valid = layout_rows_valid and row_valid
        if row_valid:
            next_start = int(stop)
    additional_raw, additional_checks = decode_xz_base64_payload(
        reconstruction.get("additional_from_CW11_float64_le"),
        "terminal additional vector",
        expected_dtype="<f8",
        expected_raw_bytes=ACTOR6_FLOAT64_BYTES,
    )
    total_raw, total_checks = decode_xz_base64_payload(
        reconstruction.get("terminal_total_from_raw_float64_le"),
        "terminal total vector",
        expected_dtype="<f8",
        expected_raw_bytes=ACTOR6_FLOAT64_BYTES,
    )
    xor_raw, xor_checks = decode_xz_base64_payload(
        reconstruction.get("cw11_to_terminal_actor_float32_xor_backup"),
        "terminal actor XOR backup",
        expected_dtype="uint8_xor",
        expected_raw_bytes=ACTOR6_FLOAT32_BYTES,
    )
    additional_l2 = float64_l2(additional_raw, "terminal additional")
    total_l2 = float64_l2(total_raw, "terminal total")
    terminal_model = decision.get("terminal_model_state_sha256")
    checks = {
        "raw_anchor_exact": anchor.get("raw_model_state_sha256") == RAW_MODEL_SHA256,
        "cw11_anchor_exact": anchor.get("cw11_model_state_sha256") == CW11_MODEL_SHA256,
        "cw11_vector_anchor_exact": anchor.get("cw11_total_float64_le_sha256")
        == CW11_VECTOR_SHA256,
        "terminal_model_anchor_exact": is_sha256(terminal_model)
        and anchor.get("terminal_model_state_sha256") == terminal_model,
        "terminal_iteration_model_exact": terminal_iteration.get("model_state_sha256")
        == terminal_model,
        "layout_rows_exact": layout_rows_valid
        and tuple(layout_names) == ACTOR6_NAMES
        and next_start == ACTOR6_FLAT_LENGTH,
        "layout_sha_exact": reconstruction.get("actor_layout_sha256")
        == ACTOR6_LAYOUT_SHA256
        == sha256_bytes(canonical_json(layout)),
        "actor_dtype_exact": reconstruction.get("actor_parameter_dtype")
        == "torch.float32",
        "flat_length_exact": reconstruction.get("flat_length")
        == ACTOR6_FLAT_LENGTH,
        "additional_sha_matches_iteration": sha256_bytes(additional_raw)
        == terminal_iteration.get("additional_float64_le_sha256"),
        "total_sha_matches_iteration": sha256_bytes(total_raw)
        == terminal_iteration.get("total_from_raw_float64_le_sha256"),
        "additional_l2_matches_decision": math.isclose(
            additional_l2,
            float(decision.get("terminal_additional_l2", math.nan)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "total_l2_matches_decision": math.isclose(
            total_l2,
            float(decision.get("terminal_total_from_raw_l2", math.nan)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "additional_within_cap": 0.0 < additional_l2
        <= ADDITIONAL_TOTAL_L2_CAP + L2_CAP_ABS_TOL,
        "actor_hashes_hex64": is_sha256(
            reconstruction.get("cw11_actor_float32_le_sha256")
        )
        and is_sha256(reconstruction.get("terminal_actor_float32_le_sha256")),
        "xor_policy_exact": reconstruction.get("xor_backup_policy")
        == "audit_only_never_used_to_correct_mathematical_reconstruction",
        "raw_template_exact": reconstruction.get(
            "raw_checkpoint_template_is_original_U468"
        )
        is True,
        "application_dependency_shas_hex64": isinstance(
            reconstruction.get("application_dependency_shas"), Mapping
        )
        and set(reconstruction["application_dependency_shas"])
        == {"cutting", "ram", "geometry"}
        and all(is_sha256(value) for value in reconstruction["application_dependency_shas"].values()),
    }
    if not all(checks.values()):
        raise ProtocolError(f"terminal reconstruction deep gate failed: {checks}")
    return {
        "checks": checks,
        "additional_payload_checks": additional_checks,
        "total_payload_checks": total_checks,
        "xor_payload_checks": xor_checks,
        "additional_l2": additional_l2,
        "total_from_raw_l2": total_l2,
        "additional_raw_sha256": sha256_bytes(additional_raw),
        "total_raw_sha256": sha256_bytes(total_raw),
        "xor_raw_sha256": sha256_bytes(xor_raw),
    }


def validate_terminal_payload(payload: bytes) -> dict[str, Any]:
    document = strict_json_object(payload, "solver terminal stdout")
    terminal = document.get("terminal")
    second_stage = document.get("second_stage")
    final_integrity = document.get("final_integrity")
    common_checks = {
        "schema_exact": document.get("schema_version") == SOLVER_SCHEMA,
        "status_terminal_exact": document.get("status") in EXPECTED_TERMINAL_STATUSES,
        "classification_exact": document.get("classification") == CLASSIFICATION,
        "scope_exact": document.get("scope") == RUN_SCOPE,
        "run_executed_true": document.get("run_executed") is True,
        "cuda_accessed_true": document.get("cuda_accessed") is True,
        "writes_performed_false": document.get("writes_performed") is False,
        "terminal_mapping_present": isinstance(terminal, Mapping),
        "second_stage_mapping_present": isinstance(second_stage, Mapping),
        "final_integrity_pass": isinstance(final_integrity, Mapping)
        and final_integrity.get("pass") is True
        and all_true_checks(final_integrity.get("checks"))
        and final_integrity.get("model_left_raw_after_outer_finally") is True,
    }
    if not all(common_checks.values()):
        raise ProtocolError(f"solver terminal common contract failed: {common_checks}")
    assert isinstance(terminal, Mapping)
    assert isinstance(second_stage, Mapping)
    decision = terminal.get("decision")
    iterations = second_stage.get("iterations")
    common_deep = {
        "terminal_status_self_consistent": terminal.get("status")
        == second_stage.get("status")
        == document.get("status"),
        "terminal_decision_exact_second_stage": isinstance(decision, Mapping)
        and decision == second_stage.get("decision"),
        "terminal_reconstruction_exact_second_stage": terminal.get(
            "reconstruction_payload"
        )
        == second_stage.get("terminal_reconstruction_payload"),
        "terminal_reconstruction_audit_crosslinks": terminal.get(
            "reconstruction_audit"
        )
        == second_stage.get("terminal_reconstruction_audit")
        == document.get("terminal_reconstruction_audit"),
        "terminal_active_ledger_crosslinks": terminal.get("active_cut_ledger")
        == second_stage.get("terminal_active_cut_ledger"),
        "terminal_active_ledger_sha_crosslinks": terminal.get(
            "active_cut_ledger_sha256"
        )
        == second_stage.get("terminal_active_cut_ledger_sha256"),
        "consumer_only_mirrors_absent_from_second_stage": all(
            key not in second_stage
            for key in (
                "terminal_consumed_valid_oracle",
                "legacy_B33_gate",
                "terminal_active_cut_gates",
            )
        ),
        "terminal_downstream_exact_second_stage": terminal.get("downstream_contract")
        == second_stage.get("downstream_contract"),
        "iterations_bounded": isinstance(iterations, list)
        and 2 <= len(iterations) <= MAX_OUTER_ITERATIONS + 1,
        "iteration_zero_exact": isinstance(iterations, list)
        and iterations[0].get("iteration") == 0,
        "iteration_ordinals_unique_bounded": isinstance(iterations, list)
        and len({row.get("iteration") for row in iterations}) == len(iterations)
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("iteration"), int)
            and 0 <= row["iteration"] <= MAX_OUTER_ITERATIONS
            for row in iterations
        ),
        "candidate_consumer_not_called": isinstance(decision, Mapping)
        and decision.get("candidate_consumer_called") is False,
        "first_feasible_required": isinstance(decision, Mapping)
        and decision.get("first_feasible_required") is True,
        "promotion_evidence_false": isinstance(decision, Mapping)
        and decision.get("eligible_as_promotion_evidence") is False,
        "local_finally_restored": second_stage.get("local_finally", {}).get(
            "restored_exact_CW11_before_outer_finally"
        )
        is True,
    }
    if not all(common_deep.values()):
        raise ProtocolError(f"solver terminal cross-link contract failed: {common_deep}")
    assert isinstance(decision, Mapping)
    assert isinstance(iterations, list)
    success = document["status"] == EXPECTED_TERMINAL_STATUSES[0]
    reconstruction_audit: dict[str, Any] | None = None
    if success:
        ordinal = decision.get("terminal_iteration")
        selected = [row for row in iterations if row.get("iteration") == ordinal]
        if len(selected) != 1:
            raise ProtocolError("terminal iteration is not unique in iteration ledger")
        terminal_iteration = selected[0]
        oracle = terminal_iteration.get("oracle")
        gates = oracle.get("authoritative_60_gates", {}) if isinstance(oracle, Mapping) else {}
        active_ledger = terminal.get("active_cut_ledger")
        active_ledger_sha = terminal.get("active_cut_ledger_sha256")
        success_checks = {
            "terminal_iteration_in_range": isinstance(ordinal, int)
            and 1 <= ordinal <= MAX_OUTER_ITERATIONS,
            "close_reason_exact": decision.get("close_reason")
            == "first_feasible_fixed_iteration_candidate",
            "formal_eligibility_true": decision.get(
                "eligible_only_for_formal_fulltrain_revalidation"
            )
            is True,
            "terminal_model_hex64": is_sha256(
                decision.get("terminal_model_state_sha256")
            ),
            "terminal_additional_l2_cap": isinstance(
                decision.get("terminal_additional_l2"), (int, float)
            )
            and 0.0 < float(decision["terminal_additional_l2"])
            <= ADDITIONAL_TOTAL_L2_CAP + L2_CAP_ABS_TOL,
            "terminal_total_l2_finite": isinstance(
                decision.get("terminal_total_from_raw_l2"), (int, float)
            )
            and math.isfinite(float(decision["terminal_total_from_raw_l2"])),
            "terminal_active_cut_count_minimum": isinstance(
                decision.get("terminal_active_cut_count"), int
            )
            and decision["terminal_active_cut_count"] >= 38,
            "terminal_active_ledger_exact": isinstance(active_ledger, list)
            and len(active_ledger) == decision.get("terminal_active_cut_count")
            and all(isinstance(row, Mapping) for row in active_ledger)
            and len({sha256_bytes(canonical_json(row)) for row in active_ledger})
            == len(active_ledger),
            "terminal_active_ledger_sha_exact": is_sha256(active_ledger_sha)
            and isinstance(active_ledger, list)
            and active_ledger_sha == sha256_bytes(canonical_json(active_ledger)),
            "acceptance_all_true": terminal_iteration.get("acceptance", {}).get(
                "pass"
            )
            is True
            and all_true_checks(
                terminal_iteration.get("acceptance", {}).get("checks")
            ),
            "integrity_all_true_including_nonactor": all_true_checks(
                terminal_iteration.get("integrity")
            )
            and terminal_iteration.get("integrity", {}).get(
                "nonactor_74_exact_raw"
            )
            is True,
            "fixed_repairs_pass": terminal_iteration.get(
                "fixed_repair_gates", {}
            ).get("pass")
            is True,
            "legacy_B33_pass": terminal_iteration.get("legacy_gate", {}).get("pass")
            is True,
            "active_cut_gates_pass": terminal_iteration.get(
                "active_cut_gates", {}
            ).get("pass")
            is True,
            "official_60_of_60": gates.get("gate_count") == 60
            and gates.get("passed_gate_count") == 60
            and gates.get("pass") is True
            and isinstance(gates.get("records"), list)
            and len(gates["records"]) == 60
            and all(row.get("pass") is True for row in gates["records"]),
            "new_harm_zero": isinstance(oracle, Mapping)
            and oracle.get("new_harm_count_vs_cw11") == 0
            and oracle.get("new_harms") == [],
            "official_oracle_hard_checks_all_true": isinstance(oracle, Mapping)
            and all_true_checks(oracle.get("hard_checks")),
            "terminal_active_gate_count_matches_ledger": isinstance(
                active_ledger, list
            )
            and terminal_iteration.get("active_cut_gates", {}).get("count")
            == len(active_ledger)
            and terminal_iteration.get("active_cut_gates", {}).get(
                "violated_count"
            )
            == 0
            and isinstance(
                terminal_iteration.get("active_cut_gates", {}).get("records"),
                list,
            )
            and len(terminal_iteration["active_cut_gates"]["records"])
            == len(active_ledger)
            and all(
                row.get("pass") is True
                for row in terminal_iteration["active_cut_gates"]["records"]
            ),
            "legacy_gate_deep_pass": terminal_iteration.get(
                "legacy_gate", {}
            ).get("gate", {}).get("pass")
            is True
            and terminal_iteration.get("legacy_gate", {}).get(
                "patch_audit", {}
            ).get("global_restored_every_call")
            is True,
            "solver_reconstruction_audit_all_true": isinstance(
                terminal.get("reconstruction_audit"), Mapping
            )
            and all_true_checks(terminal["reconstruction_audit"].get("checks"))
            and terminal["reconstruction_audit"].get("model_state_sha256")
            == decision.get("terminal_model_state_sha256")
            and terminal["reconstruction_audit"].get(
                "total_float64_le_sha256"
            )
            == terminal_iteration.get("total_from_raw_float64_le_sha256"),
        }
        if not all(success_checks.values()):
            raise ProtocolError(f"solver success terminal gate failed: {success_checks}")
        reconstruction_audit = validate_reconstruction_payload(
            terminal.get("reconstruction_payload"), decision, terminal_iteration
        )
        terminal_checks = success_checks
    else:
        closed_checks = {
            "terminal_iteration_none": decision.get("terminal_iteration") is None,
            "terminal_model_none": decision.get("terminal_model_state_sha256") is None,
            "reconstruction_none": terminal.get("reconstruction_payload") is None,
            "reconstruction_audit_none": terminal.get("reconstruction_audit") is None,
            "active_ledger_none": terminal.get("active_cut_ledger") is None,
            "active_ledger_sha_none": terminal.get("active_cut_ledger_sha256") is None,
            "formal_eligibility_false": decision.get(
                "eligible_only_for_formal_fulltrain_revalidation"
            )
            is False,
            "close_reason_not_success": decision.get("close_reason")
            != "first_feasible_fixed_iteration_candidate",
            "terminal_additional_l2_bounded": isinstance(
                decision.get("terminal_additional_l2"), (int, float)
            )
            and 0.0 <= float(decision["terminal_additional_l2"])
            <= ADDITIONAL_TOTAL_L2_CAP + L2_CAP_ABS_TOL,
            "terminal_total_l2_finite": isinstance(
                decision.get("terminal_total_from_raw_l2"), (int, float)
            )
            and math.isfinite(float(decision["terminal_total_from_raw_l2"])),
        }
        if not all(closed_checks.values()):
            raise ProtocolError(f"solver closed terminal gate failed: {closed_checks}")
        terminal_checks = closed_checks
    # The caller publishes these exact bytes.  Parsing a second time guards
    # against an accidental parse/re-serialize output path.
    reloaded = strict_json_object(payload, "solver terminal stdout lossless check")
    if reloaded != document:
        raise ProtocolError("solver terminal payload changed during validation")
    return {
        "checks": common_checks,
        "cross_link_checks": common_deep,
        "terminal_checks": terminal_checks,
        "reconstruction_audit": reconstruction_audit,
        "stdout_sha256": sha256_bytes(payload),
        "stdout_bytes": len(payload),
        "payload": document,
    }


def post_child_success_rehash(
    preflight_evidence: Mapping[str, Any],
    marker_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-establish every immutable identity after the exit-zero child."""
    _, solver_after = read_regular_stable(
        SOLVER,
        "CW12 solver after run child",
        expected_sha256=EXPECTED_SOLVER_SHA256,
        expected_mode=FROZEN_EXECUTABLE_MODE,
    )
    static_document = preflight_evidence.get("solver_static", {}).get("payload")
    if not isinstance(static_document, Mapping):
        raise ProtocolError("preflight static payload is unavailable after child")
    bindings_after = independently_rehash_solver_bindings(static_document)
    marker_payload, marker_after = read_regular_stable(
        ATTEMPT_MARKER,
        "attempt marker after run child",
        expected_sha256=str(marker_evidence.get("sha256")),
        expected_mode=FROZEN_EVIDENCE_MODE,
    )
    del marker_payload
    targets = {
        "stdout_output_absent": absent_by_lstat(STDOUT_OUTPUT),
        "stderr_audit_absent": absent_by_lstat(STDERR_AUDIT),
    }
    checks = {
        "solver_identity_exact_pre_post": solver_after
        == preflight_evidence.get("solver"),
        "all_29_binding_records_exact_pre_post": bindings_after.get("records")
        == preflight_evidence.get("independent_frozen_input_rehash", {}).get(
            "records"
        ),
        "binding_summary_exact_pre_post": {
            key: bindings_after.get(key)
            for key in ("binding_count", "ordered_identity_sha256", "pass")
        }
        == {
            key: preflight_evidence.get("independent_frozen_input_rehash", {}).get(
                key
            )
            for key in ("binding_count", "ordered_identity_sha256", "pass")
        },
        "attempt_marker_identity_exact": marker_after == marker_evidence,
        "success_outputs_still_absent": all(targets.values()),
    }
    if not all(checks.values()):
        raise ProtocolError(f"post-child frozen identity gate failed: {checks}")
    return {
        "checks": checks,
        "solver": solver_after,
        "independent_frozen_input_rehash": bindings_after,
        "attempt_marker": marker_after,
        "success_targets_before_publish": targets,
        "pass": True,
    }


def validated_stderr_audit(
    payload: bytes,
    *,
    post_child_integrity: Mapping[str, Any],
) -> bytes:
    """Preserve benign warning output losslessly without vetoing exit-zero.

    Torch/pynvml can emit a FutureWarning on an otherwise valid run.  The
    solver's strict terminal stdout contract remains authoritative.  Stderr
    must nevertheless be UTF-8 and must not contain a Python traceback; its
    exact bytes are retained as base64 in a separate immutable audit record.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError("solver stderr is not UTF-8") from error
    checks = {
        "utf8_exact": True,
        "python_traceback_absent": "Traceback (most recent call last):" not in text,
    }
    if not all(checks.values()):
        raise ProtocolError(f"solver stderr hard gate failed: {checks}")
    return canonical_json(
        {
            "schema_version": (
                "ptcg-cw12-consumed-valid-official6-one-shot-stderr-audit-v1"
            ),
            "status": "captured_losslessly_not_a_success_veto",
            "created_at_utc": utc_now(),
            "checks": checks,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "base64": base64.b64encode(payload).decode("ascii"),
            "post_child_integrity": dict(post_child_integrity),
        }
    )


def static_source_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    source_text = source.decode("utf-8")
    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def calls_in(function_name: str) -> set[str]:
        node = function_nodes.get(function_name)
        if node is None:
            return set()
        return {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }

    run_node = function_nodes.get("run_once")
    run_segment = (
        "" if run_node is None else ast.get_source_segment(source_text, run_node) or ""
    )
    ordered_success_tokens = (
        "post_child_integrity = post_child_success_rehash(",
        "terminal_validation = validate_terminal_payload(",
        "stderr_payload = validated_stderr_audit(",
        "stderr_evidence = publish_exclusive(",
        "output_evidence = publish_exclusive(",
    )
    token_offsets = [
        *[run_segment.find(token) for token in ordered_success_tokens[:-1]],
        run_segment.rfind(ordered_success_tokens[-1]),
    ]
    imported_roots: set[str] = set()
    subprocess_calls = 0
    shell_true = False
    retry_syntax = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.func.attr == "run"
            ):
                subprocess_calls += 1
                shell_true = shell_true or any(
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                )
        elif isinstance(node, (ast.While, ast.For)):
            retry_syntax = retry_syntax or any(
                isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Name)
                and descendant.func.id == "execute_child"
                for descendant in ast.walk(node)
            )
    forbidden_imports = {
        "kaggle",
        "requests",
        "socket",
        "torch",
        "urllib",
    }
    checks = {
        "no_forbidden_imports": not bool(imported_roots.intersection(forbidden_imports)),
        "one_subprocess_call_site_exact": subprocess_calls == 1,
        "shell_true_absent": not shell_true,
        "no_loop_around_execute_child": not retry_syntax,
        "run_and_static_argv_python_exact": STATIC_ARGV[0] == RUN_ARGV[0] == str(EXPECTED_PYTHON),
        "run_and_static_flags_exact": STATIC_ARGV[1:3] == RUN_ARGV[1:3] == ("-I", "-B"),
        "run_and_static_solver_exact": STATIC_ARGV[3] == RUN_ARGV[3] == str(SOLVER),
        "run_mode_exact": RUN_ARGV[4:] == ("--mode", "run"),
        "static_mode_exact": STATIC_ARGV[4:] == ("--mode", "static"),
        "marker_and_output_distinct": ATTEMPT_MARKER != STDOUT_OUTPUT,
        "all_targets_distinct": len({ATTEMPT_MARKER, STDOUT_OUTPUT, STDERR_AUDIT}) == 3,
        "all_targets_under_artifacts": (
            ATTEMPT_MARKER.parent
            == STDOUT_OUTPUT.parent
            == STDERR_AUDIT.parent
            == ARTIFACTS
        ),
        "post_child_rehash_declared": "post_child_success_rehash" in function_nodes,
        "run_once_calls_post_child_rehash": "post_child_success_rehash"
        in calls_in("run_once"),
        "post_child_rehash_covers_solver_bindings_marker": {
            "read_regular_stable",
            "independently_rehash_solver_bindings",
            "absent_by_lstat",
        }.issubset(calls_in("post_child_success_rehash")),
        "success_order_rehash_validate_stderr_stdout": all(
            offset >= 0 for offset in token_offsets
        )
        and token_offsets == sorted(token_offsets),
        "stderr_audit_receives_post_child_integrity": (
            "post_child_integrity=post_child_integrity"
            in "".join(run_segment.split())
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "imported_roots": sorted(imported_roots),
        "subprocess_call_sites": subprocess_calls,
    }


def preflight(*, require_lock: bool) -> dict[str, Any]:
    runtime = validate_runtime()
    self_payload, self_evidence = read_regular_stable(
        SCRIPT,
        "one-shot launcher",
        expected_mode=FROZEN_EXECUTABLE_MODE,
    )
    source_audit = static_source_audit(self_payload)
    if not source_audit["pass"]:
        raise ProtocolError(f"launcher source audit failed: {source_audit}")
    validate_target_parent(ATTEMPT_MARKER, "attempt marker")
    validate_target_parent(STDOUT_OUTPUT, "stdout output")
    validate_target_parent(STDERR_AUDIT, "stderr audit")

    armed = solver_lock_armed()
    if require_lock and not armed:
        raise ProtocolError("CW12 launcher is fail-closed pending final solver lock")

    if armed:
        _, solver_evidence = read_regular_stable(
            SOLVER,
            "CW12 solver",
            expected_sha256=EXPECTED_SOLVER_SHA256,
            expected_mode=FROZEN_EXECUTABLE_MODE,
        )
    else:
        _, solver_evidence = read_regular_stable(
            SOLVER,
            "pending CW12 solver",
            # The builder may legitimately hold the not-yet-audited solver in
            # 0755 while editing.  Armed mode below requires the final 0555.
            expected_mode=None,
        )

    absence = {
        "attempt_marker": absent_by_lstat(ATTEMPT_MARKER),
        "stdout_output": absent_by_lstat(STDOUT_OUTPUT),
        "stderr_audit": absent_by_lstat(STDERR_AUDIT),
    }
    if require_lock and not all(absence.values()):
        raise FileExistsError(f"one-shot target already exists: {absence}")

    result: dict[str, Any] = {
        "runtime": runtime,
        "self": self_evidence,
        "source_audit": source_audit,
        "solver": solver_evidence,
        "solver_lock_armed": armed,
        "targets_absent_by_lstat": absence,
        "static_command": argv_binding(STATIC_ARGV),
        "run_command": argv_binding(RUN_ARGV),
        "child_environment": environment_binding(),
    }
    if armed:
        static_completed = execute_child(STATIC_ARGV)
        if static_completed.returncode != 0:
            raise ProtocolError(
                "solver static subprocess failed before attempt consumption: "
                f"returncode={static_completed.returncode}; "
                f"stderr_sha256={sha256_bytes(static_completed.stderr)}"
            )
        if static_completed.stderr:
            raise ProtocolError(
                "solver static subprocess produced unexpected stderr: "
                f"sha256={sha256_bytes(static_completed.stderr)}"
            )
        result["solver_static"] = validate_static_payload(static_completed.stdout)
        result["independent_frozen_input_rehash"] = (
            independently_rehash_solver_bindings(
                result["solver_static"]["payload"]
            )
        )
        _, after_static = read_regular_stable(
            SOLVER,
            "CW12 solver after static preflight",
            expected_sha256=EXPECTED_SOLVER_SHA256,
            expected_mode=FROZEN_EXECUTABLE_MODE,
        )
        if after_static != solver_evidence:
            raise ProtocolError("solver identity changed during static preflight")
        after_absence = {
            "attempt_marker": absent_by_lstat(ATTEMPT_MARKER),
            "stdout_output": absent_by_lstat(STDOUT_OUTPUT),
            "stderr_audit": absent_by_lstat(STDERR_AUDIT),
        }
        if require_lock and not all(after_absence.values()):
            raise FileExistsError(
                f"one-shot target appeared during static preflight: {after_absence}"
            )
        result["targets_absent_after_static_by_lstat"] = after_absence
    return result


def write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise RuntimeError("short exclusive artifact write")
        offset += written


def publish_exclusive(path: Path, payload: bytes, *, mode: int) -> dict[str, Any]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        visible = os.lstat(path)
        if (
            not stat.S_ISREG(observed.st_mode)
            or int(observed.st_nlink) != 1
            or stat.S_ISLNK(visible.st_mode)
            or (observed.st_dev, observed.st_ino, observed.st_size)
            != (visible.st_dev, visible.st_ino, visible.st_size)
            or int(observed.st_size) != len(payload)
            or stat.S_IMODE(observed.st_mode) != mode
        ):
            raise RuntimeError("unsafe O_EXCL artifact publication")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        reloaded = b"".join(chunks)
        if reloaded != payload:
            raise RuntimeError("exclusive artifact payload changed after write")
        return {
            "path": root_relative(path),
            "sha256": sha256_bytes(reloaded),
            "bytes": len(reloaded),
            "mode_octal": format(mode, "04o"),
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
            "nlink": int(observed.st_nlink),
        }
    finally:
        os.close(descriptor)


def failure_payload(
    *,
    stage: str,
    marker: Mapping[str, Any],
    error: BaseException | None = None,
    completed: subprocess.CompletedProcess[bytes] | None = None,
) -> bytes:
    record: dict[str, Any] = {
        "schema_version": FAILURE_SCHEMA,
        "status": "terminal_failure_attempt_consumed_no_retry",
        "created_at_utc": utc_now(),
        "stage": stage,
        "attempt_marker": dict(marker),
        "solver": {
            "path": root_relative(SOLVER),
            "sha256": EXPECTED_SOLVER_SHA256,
        },
        "run_command": argv_binding(RUN_ARGV),
        "child_environment": environment_binding(),
        "retry_authorized": False,
        "attempt_remains_consumed": True,
        "broad_gold_package_upload_submission_authorized": False,
    }
    if error is not None:
        record["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    if completed is not None:
        record["child"] = {
            "returncode": int(completed.returncode),
            "stdout_bytes": len(completed.stdout),
            "stdout_sha256": sha256_bytes(completed.stdout),
            "stdout_base64": base64.b64encode(completed.stdout).decode("ascii"),
            "stderr_bytes": len(completed.stderr),
            "stderr_sha256": sha256_bytes(completed.stderr),
            "stderr_base64": base64.b64encode(completed.stderr).decode("ascii"),
        }
    return canonical_json(record)


def run_once() -> dict[str, Any]:
    preflight_evidence = preflight(require_lock=True)
    # Rehash every frozen solver-owned input a second time immediately before
    # consumption.  The zero-CUDA static child is the authoritative input
    # verifier; preserving its exact payload/hash in the marker binds that pass.
    marker_payload = canonical_json(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "one_shot_attempt_consumed_before_cuda_and_official6",
            "created_at_utc": utc_now(),
            "pid": os.getpid(),
            "launcher": preflight_evidence["self"],
            "solver": preflight_evidence["solver"],
            "solver_static": {
                key: preflight_evidence["solver_static"][key]
                for key in ("checks", "stdout_sha256", "stdout_bytes")
            },
            "independent_frozen_input_rehash": {
                key: preflight_evidence["independent_frozen_input_rehash"][key]
                for key in ("binding_count", "ordered_identity_sha256", "pass")
            },
            "run_command": preflight_evidence["run_command"],
            "child_environment": preflight_evidence["child_environment"],
            "classification": CLASSIFICATION,
            "stdout_output": root_relative(STDOUT_OUTPUT),
            "stderr_audit": root_relative(STDERR_AUDIT),
            "attempts_authorized": 1,
            "retry_authorized": False,
            "marker_created_before_cuda_and_official6": True,
        }
    )
    marker_evidence = publish_exclusive(
        ATTEMPT_MARKER,
        marker_payload,
        mode=FROZEN_EVIDENCE_MODE,
    )

    completed: subprocess.CompletedProcess[bytes] | None = None
    try:
        completed = execute_child(RUN_ARGV)
        if completed.returncode != 0:
            terminal = failure_payload(
                stage="solver_subprocess_nonzero",
                marker=marker_evidence,
                completed=completed,
            )
            output_evidence = publish_exclusive(
                STDOUT_OUTPUT,
                terminal,
                mode=FROZEN_EVIDENCE_MODE,
            )
            return {
                "schema_version": SCHEMA,
                "status": "terminal_failure_recorded_attempt_consumed",
                "attempt_marker": marker_evidence,
                "stdout_output": output_evidence,
                "child_returncode": completed.returncode,
            }
        post_child_integrity = post_child_success_rehash(
            preflight_evidence,
            marker_evidence,
        )
        terminal_validation = validate_terminal_payload(completed.stdout)
        stderr_payload = validated_stderr_audit(
            completed.stderr,
            post_child_integrity=post_child_integrity,
        )
        stderr_evidence = publish_exclusive(
            STDERR_AUDIT,
            stderr_payload,
            mode=FROZEN_EVIDENCE_MODE,
        )
        # Publish the child's stdout byte-for-byte.  This is intentionally not a
        # wrapper or a JSON reserialization; the captured terminal payload is
        # therefore lossless.
        output_evidence = publish_exclusive(
            STDOUT_OUTPUT,
            completed.stdout,
            mode=FROZEN_EVIDENCE_MODE,
        )
        if (
            output_evidence["sha256"] != terminal_validation["stdout_sha256"]
            or output_evidence["bytes"] != terminal_validation["stdout_bytes"]
        ):
            raise RuntimeError("published stdout differs from validated terminal payload")
        return {
            "schema_version": SCHEMA,
            "status": "terminal_solver_payload_published_losslessly",
            "attempt_marker": marker_evidence,
            "stdout_output": output_evidence,
            "stderr_audit": stderr_evidence,
            "post_child_integrity": post_child_integrity,
            "solver_status": terminal_validation["payload"]["status"],
            "child_returncode": completed.returncode,
            "retry_authorized": False,
        }
    except BaseException as error:
        if absent_by_lstat(STDOUT_OUTPUT):
            terminal = failure_payload(
                stage="launcher_post_marker_exception",
                marker=marker_evidence,
                error=error,
                completed=completed,
            )
            output_evidence = publish_exclusive(
                STDOUT_OUTPUT,
                terminal,
                mode=FROZEN_EVIDENCE_MODE,
            )
            return {
                "schema_version": SCHEMA,
                "status": "terminal_failure_recorded_attempt_consumed",
                "attempt_marker": marker_evidence,
                "stdout_output": output_evidence,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        raise


def static_result() -> dict[str, Any]:
    evidence = preflight(require_lock=False)
    return {
        "schema_version": SCHEMA,
        "status": (
            "static_ready_solver_lock_armed"
            if evidence["solver_lock_armed"]
            else "static_fail_closed_pending_final_solver_lock"
        ),
        "classification": CLASSIFICATION,
        "evidence": evidence,
        "run_executed": False,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="static")
    args = parser.parse_args()
    result = static_result() if args.mode == "static" else run_once()
    print(canonical_json(result).decode("utf-8"), end="")
    return 1 if result.get("status") == "terminal_failure_recorded_attempt_consumed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

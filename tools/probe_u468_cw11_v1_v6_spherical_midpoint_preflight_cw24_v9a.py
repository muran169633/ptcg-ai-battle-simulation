#!/usr/bin/env python3
"""CW24 v9a frozen-v1/v6 spherical-midpoint linear preflight.

This train-only diagnostic replays two already frozen failed recipes solely to
capture their actual float32 actor deltas in memory: the v1 terminal failure
and the rejected second v6 shadow.  It forms the fixed-radius equal-weight
minor-arc midpoint of their directions, then applies the frozen-v4 B352 CW11
baseline tangent matrix as a linear preflight.  The midpoint is never copied
to a model and never receives a forward pass.  No new candidate shadow or
official candidate is created, and no payload/checkpoint/model is written.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import math
import os
import stat
import subprocess
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_v1_v6_spherical_midpoint_preflight_cw24_v9a.py"
OUTPUT = ROOT / "artifacts/cw24_cw11_v1_v6_spherical_midpoint_preflight_cw24_v9a.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-v1-v6-spherical-midpoint-linear-preflight-cw24-v9a"
SEED = 202608049
R_SAFE = 0.000999975
V6_FULL_RADIUS = 0.00099998

V1_SOURCE = TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py"
V1_SOURCE_SHA256 = "a3c96eed95a6b558c61303e2190eaa4f353970add4509e8770c3dfcc1621db39"
V1_SOURCE_MODE = 0o555
V1_RESULT = ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v1.json"
V1_RESULT_SHA256 = "9e3be911bfba95338538f315d47d3f7ce9f598e830ace607332d46697cb0cda5"
V1_RESULT_MODE = 0o444

V4_SOURCE = TOOLS / "probe_u468_cw11_native_cuttingplane_specialbc_cw24_v4.py"
V4_SOURCE_SHA256 = "134eebec7b1e54b68bd47256fde5c8b80dea3cd8d8ea4399428ec4c1de7d1454"
V4_SOURCE_MODE = 0o555
V4_RESULT = ROOT / "artifacts/cw24_cw11_native_cuttingplane_specialbc_trainonly_v4.json"
V4_RESULT_SHA256 = "db0c2c356fac5779e14698e3ff3ee9b36967f2aa6ddecee55531bb37e8b3377c"
V4_RESULT_MODE = 0o444

V6_SOURCE = TOOLS / "probe_u468_cw11_postaccept_fullstep_specialbc_cw24_v6.py"
V6_SOURCE_SHA256 = "98cb3e3fb0e62ba1f8faf7185712cfcbc11639ed396433782ec8b02242d57883"
V6_SOURCE_MODE = 0o555
V6_RESULT = ROOT / "artifacts/cw24_cw11_postaccept_fullstep_specialbc_trainonly_v6.json"
V6_RESULT_SHA256 = "d0a33f55d11df06ce5496a84bd07f1624a56d8fed726efea50b96a8c1e5bd616"
V6_RESULT_MODE = 0o444

V8_SOURCE = TOOLS / "probe_u468_cw11_absolute_residual_specialbc_cw24_v8.py"
V8_SOURCE_SHA256 = "bb22306ed4e1379d49357910485e9c1efc3fe40565a6adce67e1e5462f6177da"
V8_SOURCE_MODE = 0o555
V8_RESULT = ROOT / "artifacts/cw24_cw11_absolute_residual_specialbc_trainonly_v8.json"
V8_RESULT_SHA256 = "5341aa76c33276901bff1acb5ba5fa9810620ee4d055f78a016494b040c5e908"
V8_RESULT_MODE = 0o444

V1_ACTUAL_DELTA_SHA256 = "48497e7b9857199bab0d653fc75ea0e0861ed634c64fca0483a3a0520687ec86"
V1_ACTUAL_L2 = 0.0010000012855891469
V1_CANDIDATE_MODEL_SHA256 = "7caa8ab0ad95cc7ffeb19c5719cbd65dd4740ce195da029cb0525be621392a0e"
V6_S2_PLANNED_L2 = 0.0009909412795123538
V6_S2_ACTUAL_L2 = 0.000990940375725956
V6_S2_ACTUAL_DELTA_SHA256 = "3c8ae987fda33444d6698b44f3164b41a4ff0de65eee1f8438cf5382e1278d41"
V6_S2_TRIAL_MERIT = 25123.250002966826
V6_S2_SOLVER_STEP_L2 = 0.000753542521139092
V6_S2_MATRIX_SHA256 = "9954fe13b421ab0eed6a355904d3cf27b5ef6dbc5eba8a9403969336ec9d668e"
V6_S2_RHS_SHA256 = "e58ec7fa28887e984a323c8018943244dc2e73125e35b299a4c863d050d0af76"
V4_BASELINE_MATRIX_SHA256 = "b39d640e398ea5a1d89a958b549afda04be460c6e4eb192daafc643f31984d74"
V4_BASELINE_RHS_SHA256 = "d206d5bd394982a1bd993395ea090d0675ac5fa32184244895961970fb8ab458"
TOP1_FDE6_SHA256 = "fde6fab074495c238d4c4ef42a2f12ec595da78e35836578f7a3828002037b88"
RETENTION_243F_SHA256 = "243f4a21d6c0b178a6306bdb63ac738aeb5bb13f6be0ccfaa68eebe02eac2d99"

PROTECTED_PAIR_NAMES = (
    "zero_margin_guard_0",
    "zero_margin_guard_1",
    "zero_margin_guard_2",
    "zero_margin_guard_3",
    "zero_margin_guard_4",
    "zero_margin_guard_5",
    "historical_retention_0142a2a3d5bd",
    "historical_retention_243f4a21d6c0",
    "historical_top1_fde6fab07449",
)


class ProtocolError(RuntimeError):
    """Fail-closed CW24 v9a preflight error."""


def canonical_json(value: Any) -> bytes:
    def check(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ProtocolError("nonfinite JSON value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ProtocolError("non-string JSON key")
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: Any) -> str:
    contiguous = value.astype("<f8", copy=False)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def regular_source(
    path: Path, expected_sha: str, expected_mode: int, label: str
) -> tuple[bytes, dict[str, Any]]:
    before = path.lstat()
    source = path.read_bytes()
    after = path.lstat()
    digest = hashlib.sha256(source).hexdigest()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "identity_stable": (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": digest == expected_sha,
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} drift: {checks}")
    return source, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(source),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def self_evidence(require_frozen: bool) -> dict[str, Any]:
    before = SCRIPT.lstat()
    source = SCRIPT.read_bytes()
    after = SCRIPT.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "identity_stable": (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "mode_0555": stat.S_IMODE(after.st_mode) == 0o555,
        "all_source_locks_present": all(
            digest.encode() in source
            for digest in (
                V1_SOURCE_SHA256,
                V4_SOURCE_SHA256,
                V6_SOURCE_SHA256,
                V8_SOURCE_SHA256,
            )
        ),
        "all_result_locks_present": all(
            digest.encode() in source
            for digest in (
                V1_RESULT_SHA256,
                V4_RESULT_SHA256,
                V6_RESULT_SHA256,
                V8_RESULT_SHA256,
            )
        ),
        "safe_radius_single_constant": source.count(b"\nR_SAFE =") == 1
        and R_SAFE == 0.000999975,
        "no_direct_model_forward": (b"ppo." + b"model_forward(") not in source,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "no_network_calls": all(
            token not in source
            for token in (b"requests" + b".", b"urllib" + b".", b"socket" + b".")
        ),
        "no_candidate_payload_construction": (b"candidate_actor_" + b"float32_le")
        not in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v9a source audit failed: {checks}")
    return {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ProtocolError("short v9a result write")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    observed = path.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and int(observed.st_nlink) == 1,
        "mode_0444": stat.S_IMODE(observed.st_mode) == 0o444,
        "size_exact": int(observed.st_size) == len(payload),
        "sha_exact": sha256_file(path) == hashlib.sha256(payload).hexdigest(),
    }
    if not all(checks.values()):
        raise ProtocolError(f"v9a result publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def splice_once(
    source: str,
    old: str,
    new: str,
    label: str,
    audits: list[dict[str, Any]],
) -> str:
    occurrences = source.count(old)
    if occurrences != 1:
        raise ProtocolError(f"{label}: expected one splice, observed {occurrences}")
    audits.append(
        {
            "label": label,
            "occurrences": occurrences,
            "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
            "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
        }
    )
    return source.replace(old, new, 1)


def transformed_v1_module(
    capture: Callable[[Any, Mapping[str, Any]], None]
) -> tuple[ModuleType, dict[str, Any]]:
    raw, source_evidence = regular_source(
        V1_SOURCE, V1_SOURCE_SHA256, V1_SOURCE_MODE, "frozen CW24 v1 source"
    )
    source = raw.decode()
    old = '''        if not candidate_pass:
            return {
'''
    new = '''        if not candidate_pass:
            V9A_CAPTURE_V1_FAILED_ACTUAL_DELTA(actual_delta.copy(), trial)
            return {
'''
    audits: list[dict[str, Any]] = []
    source = splice_once(source, old, new, "capture_frozen_v1_failed_actual_delta", audits)
    transformed = source.encode()
    name = "cw24_v9a_transformed_frozen_v1"
    if name in sys.modules:
        raise ProtocolError("v9a transformed v1 module name occupied")
    module = types.ModuleType(name)
    module.__file__ = str(V1_SOURCE)
    module.__package__ = ""
    module.__dict__["V9A_CAPTURE_V1_FAILED_ACTUAL_DELTA"] = capture
    sys.modules[name] = module
    try:
        exec(compile(transformed, str(V1_SOURCE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    checks = {
        "source_seed_exact": int(module.SEED) == 202608041,
        "source_output_is_v1": Path(module.OUTPUT).resolve() == V1_RESULT.resolve(),
        "capture_splice_once": len(audits) == 1 and audits[0]["occurrences"] == 1,
        "callback_identity_exact": module.V9A_CAPTURE_V1_FAILED_ACTUAL_DELTA is capture,
    }
    if not all(checks.values()):
        raise ProtocolError(f"transformed v1 audit failed: {checks}")
    return module, {
        "frozen_source": source_evidence,
        "transformed_source_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_source_bytes": len(transformed),
        "splices": audits,
        "checks": checks,
    }


def transformed_v4_v6_capture_module(
    capture_baseline: Callable[[Any, Any, Any, Any, Any], None],
    capture_s2: Callable[[Any, Mapping[str, Any]], None],
) -> tuple[ModuleType, dict[str, Any]]:
    raw, source_evidence = regular_source(
        V4_SOURCE, V4_SOURCE_SHA256, V4_SOURCE_MODE, "frozen CW24 v4 source"
    )
    source = raw.decode()
    audits: list[dict[str, Any]] = []
    reject_old = '''            else:
                rejection_count += 1
                trust_radius *= 0.5
                cw20.apply_flat_actor(
'''
    reject_new = '''            else:
                rejection_count += 1
                if new_pair_names:
                    trust_radius = TRUST_RADIUS_MAX
                else:
                    trust_radius *= 0.5
                cw20.apply_flat_actor(
'''
    accepted_old = '''                if gate_pass:
                    terminal_reason = "ALL_TARGETED_TRAIN_GATES_PASSED"
                    break
                reduction_fraction = merit_reduction / max(
                    float(log_entry["current_merit"]), MERIT_TOL
                )
                if reduction_fraction >= 0.25 and not new_pair_names:
                    trust_radius = min(TRUST_RADIUS_MAX, trust_radius * 1.5)
'''
    accepted_new = '''                if gate_pass:
                    terminal_reason = "ALL_TARGETED_TRAIN_GATES_PASSED"
                    break
                trust_radius = TRUST_RADIUS_MAX
'''
    source = splice_once(source, reject_old, reject_new, "v6_dynamic_cut_trust_policy", audits)
    source = splice_once(
        source,
        accepted_old,
        accepted_new,
        "v6_postaccept_full_trust_policy",
        audits,
    )
    baseline_old = "            return matrix, rhs, details, audit\n"
    baseline_new = '''            if trial_evaluations == 0 and accepted_iterates == 0:
                V9A_CAPTURE_V4_BASELINE_LINEARIZATION(
                    matrix.copy(),
                    rhs.copy(),
                    [dict(item) for item in details],
                    [dict(item) for item in pair_specs.values()],
                    dict(audit),
                )
            return matrix, rhs, details, audit
'''
    source = splice_once(
        source,
        baseline_old,
        baseline_new,
        "capture_v4_B352_CW11_baseline_linearization",
        audits,
    )
    s2_old = "            iteration_log.append(log_entry)\n            if accept:\n"
    s2_new = '''            iteration_log.append(log_entry)
            if trial_evaluations == 2 and accepted_iterates == 1 and not accept:
                V9A_CAPTURE_V6_REJECTED_S2_ACTUAL_DELTA(
                    actual_trial_x.copy(),
                    {
                        "actual_trial_l2": actual_trial_l2,
                        "planned_trial_l2": log_entry["planned_trial_l2"],
                        "alpha": log_entry["alpha"],
                        "trial_merit": log_entry["trial_merit"],
                        "PF_repairs": dict(log_entry["PF_repairs"]),
                        "top1_flips": list(trial_state["top1_flips"]),
                        "retention_flips": list(trial_state["retention_flips"]),
                        "new_pair_names": list(new_pair_names),
                        "gate_pass": gate_pass,
                        "accept": accept,
                        "solver": dict(solver_audit),
                        "linearization": dict(linearization_audit),
                        "pair_margins": dict(trial_state["pair_margins"]),
                    },
                )
            if accept:
'''
    source = splice_once(
        source, s2_old, s2_new, "capture_v6_rejected_second_shadow_actual_delta", audits
    )
    transformed = source.encode()
    name = "cw24_v9a_transformed_frozen_v4_v6"
    if name in sys.modules:
        raise ProtocolError("v9a transformed v4/v6 module name occupied")
    module = types.ModuleType(name)
    module.__file__ = str(V4_SOURCE)
    module.__package__ = ""
    module.__dict__["V9A_CAPTURE_V4_BASELINE_LINEARIZATION"] = capture_baseline
    module.__dict__["V9A_CAPTURE_V6_REJECTED_S2_ACTUAL_DELTA"] = capture_s2
    sys.modules[name] = module
    try:
        exec(compile(transformed, str(V4_SOURCE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    checks = {
        "source_total_radius_exact": float(module.TOTAL_RADIUS_CAP) == V6_FULL_RADIUS,
        "source_initial_radius_exact": float(module.TRUST_RADIUS_INITIAL) == 2.5e-4,
        "source_max_radius_was_3e4": float(module.TRUST_RADIUS_MAX) == 3.0e-4,
        "source_seed_exact": int(module.SEED) == 202608044,
        "four_splices_exact": len(audits) == 4
        and all(item["occurrences"] == 1 for item in audits),
        "v6_new_fragment_hashes_exact": audits[0]["new_sha256"]
        == "a1d2af6c1086bbc3f3beb228f2c8efbe8a0469e90e5d1989e9e9aae123544d88"
        and audits[1]["new_sha256"]
        == "a240388b4d3049c0aec74f49c5af0c231accf7c508b205f91120e82061f8a6d9",
        "callbacks_identity_exact": module.V9A_CAPTURE_V4_BASELINE_LINEARIZATION
        is capture_baseline
        and module.V9A_CAPTURE_V6_REJECTED_S2_ACTUAL_DELTA is capture_s2,
    }
    if not all(checks.values()):
        raise ProtocolError(f"transformed v4/v6 audit failed: {checks}")
    return module, {
        "frozen_source": source_evidence,
        "transformed_source_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_source_bytes": len(transformed),
        "splices": audits,
        "checks": checks,
    }


def frozen_inputs() -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for key, path, digest, mode, label in (
        ("v1_source", V1_SOURCE, V1_SOURCE_SHA256, V1_SOURCE_MODE, "v1 source"),
        ("v1_result", V1_RESULT, V1_RESULT_SHA256, V1_RESULT_MODE, "v1 result"),
        ("v4_source", V4_SOURCE, V4_SOURCE_SHA256, V4_SOURCE_MODE, "v4 source"),
        ("v4_result", V4_RESULT, V4_RESULT_SHA256, V4_RESULT_MODE, "v4 result"),
        ("v6_source", V6_SOURCE, V6_SOURCE_SHA256, V6_SOURCE_MODE, "v6 source"),
        ("v6_result", V6_RESULT, V6_RESULT_SHA256, V6_RESULT_MODE, "v6 result"),
        ("v8_source", V8_SOURCE, V8_SOURCE_SHA256, V8_SOURCE_MODE, "v8 source"),
        ("v8_result", V8_RESULT, V8_RESULT_SHA256, V8_RESULT_MODE, "v8 result"),
    ):
        _, item = regular_source(path, digest, mode, f"frozen CW24 {label}")
        evidence[key] = item
    return evidence


def cleanup_phase_modules(before: set[str], label: str) -> dict[str, Any]:
    added = sorted(set(sys.modules) - before)
    identities = []
    for name in added:
        module = sys.modules.get(name)
        identities.append(
            {
                "name": name,
                "file": str(getattr(module, "__file__", "")),
            }
        )
    for name in reversed(added):
        sys.modules.pop(name, None)
    remaining = sorted(set(added).intersection(sys.modules))
    checks = {
        "all_phase_added_module_names_removed": not remaining,
        "no_preexisting_module_name_removed": before.issubset(sys.modules),
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} sys.modules cleanup failed: {checks}")
    return {
        "label": label,
        "added_count": len(added),
        "added_name_manifest_sha256": hashlib.sha256(
            canonical_json(identities)
        ).hexdigest(),
        "added_modules": identities,
        "remaining_added_names": remaining,
        "checks": checks,
    }


def encode_array_xz_base64(value: Any, np: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    raw = array.tobytes(order="C")
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "raw_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "xz_bytes": len(compressed),
        "xz_sha256": hashlib.sha256(compressed).hexdigest(),
        "xz_base64": base64.b64encode(compressed).decode("ascii"),
    }


def decode_array_xz_base64(payload: Mapping[str, Any], np: Any) -> Any:
    if payload.get("dtype") != "<f8":
        raise ProtocolError("isolated-child array dtype contract drift")
    shape = tuple(int(value) for value in payload.get("shape", []))
    compressed = base64.b64decode(str(payload.get("xz_base64", "")), validate=True)
    if len(compressed) != int(payload.get("xz_bytes", -1)):
        raise ProtocolError("isolated-child xz byte count drift")
    if hashlib.sha256(compressed).hexdigest() != payload.get("xz_sha256"):
        raise ProtocolError("isolated-child xz hash drift")
    raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    if len(raw) != int(payload.get("raw_bytes", -1)):
        raise ProtocolError("isolated-child raw byte count drift")
    if hashlib.sha256(raw).hexdigest() != payload.get("raw_sha256"):
        raise ProtocolError("isolated-child raw hash drift")
    expected_values = math.prod(shape)
    if len(raw) != expected_values * 8:
        raise ProtocolError("isolated-child array shape/byte contract drift")
    result = np.frombuffer(raw, dtype=np.dtype("<f8")).reshape(shape).copy()
    result.setflags(write=False)
    return result


def child_capture_v1() -> dict[str, Any]:
    import numpy as np

    captured: list[tuple[Any, dict[str, Any]]] = []

    def callback(delta: Any, trial: Mapping[str, Any]) -> None:
        copied = np.array(delta, dtype=np.float64, order="C", copy=True)
        checks = {
            "callback_once": not captured,
            "shape_65793": copied.shape == (65793,),
            "dtype_float64": copied.dtype == np.float64,
            "finite": bool(np.isfinite(copied).all()),
            "sha_exact": array_sha(copied) == V1_ACTUAL_DELTA_SHA256,
            "l2_exact": math.isclose(
                float(np.linalg.norm(copied)), V1_ACTUAL_L2,
                rel_tol=0.0, abs_tol=5.0e-13,
            ),
            "trial_hash_exact": trial.get("actual_delta_float64_le_sha256")
            == V1_ACTUAL_DELTA_SHA256,
            "candidate_hash_exact": trial.get("candidate_model_state_sha256")
            == V1_CANDIDATE_MODEL_SHA256,
            "trial_failed": trial.get("pass") is False,
        }
        if not all(checks.values()):
            raise ProtocolError(f"frozen v1 child capture drift: {checks}")
        copied.setflags(write=False)
        captured.append((copied, checks))

    v1, transform = transformed_v1_module(callback)
    result = v1.production_run()
    replay_checks = {
        "capture_once": len(captured) == 1,
        "result_canonical_sha_exact": hashlib.sha256(canonical_json(result)).hexdigest()
        == V1_RESULT_SHA256,
        "decision_NO_GO": result.get("status") == "NO_GO_CW24_TWO_STAGE_TRAIN_GATE",
        "payload_absent": result.get("endpoint", {}).get("candidate_payload") is None,
        "outer_restore": result.get("outer_restore_pass") is True,
    }
    if not all(replay_checks.values()):
        raise ProtocolError(f"frozen v1 child replay drift: {replay_checks}")
    return {
        "schema_version": SCHEMA + "-isolated-v1-capture",
        "kind": "frozen_v1_failed_actual_delta",
        "vector": encode_array_xz_base64(captured[0][0], np),
        "capture_checks": captured[0][1],
        "replay_checks": replay_checks,
        "transform": transform,
        "process_contract": {
            "temporary_files_written": 0,
            "stdout_is_only_transport": True,
            "candidate_payload_present": False,
        },
    }


def child_capture_v6() -> dict[str, Any]:
    import numpy as np

    baseline_capture: list[dict[str, Any]] = []
    s2_capture: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []

    def capture_baseline(
        matrix: Any, rhs: Any, details: Any, pair_specs: Any, audit: Any
    ) -> None:
        matrix_copy = np.array(matrix, dtype=np.float64, order="C", copy=True)
        rhs_copy = np.array(rhs, dtype=np.float64, order="C", copy=True)
        detail_copy = json.loads(json.dumps(details, sort_keys=True))
        pair_copy = json.loads(json.dumps(pair_specs, sort_keys=True))
        names = [str(item["name"]) for item in detail_copy]
        checks = {
            "callback_once": not baseline_capture,
            "matrix_shape": matrix_copy.shape == (20, 65793),
            "rhs_shape": rhs_copy.shape == (20,),
            "finite": bool(np.isfinite(matrix_copy).all())
            and bool(np.isfinite(rhs_copy).all()),
            "matrix_sha_exact": array_sha(matrix_copy) == V4_BASELINE_MATRIX_SHA256,
            "rhs_sha_exact": array_sha(rhs_copy) == V4_BASELINE_RHS_SHA256,
            "details_names_unique20": len(names) == len(set(names)) == 20,
            "pair_specs_exact13": len(pair_copy) == 13,
            "audit_hashes_repeat": audit.get("matrix_float64_le_sha256")
            == V4_BASELINE_MATRIX_SHA256
            and audit.get("rhs_float64_le_sha256") == V4_BASELINE_RHS_SHA256,
        }
        if not all(checks.values()):
            raise ProtocolError(f"v4 child baseline capture drift: {checks}")
        matrix_copy.setflags(write=False)
        rhs_copy.setflags(write=False)
        baseline_capture.append(
            {
                "matrix": matrix_copy,
                "rhs": rhs_copy,
                "details": detail_copy,
                "pair_specs": pair_copy,
                "checks": checks,
            }
        )

    def capture_s2(delta: Any, audit: Mapping[str, Any]) -> None:
        copied = np.array(delta, dtype=np.float64, order="C", copy=True)
        linearization = audit.get("linearization", {})
        solver = audit.get("solver", {})
        checks = {
            "callback_once": not s2_capture,
            "shape_65793": copied.shape == (65793,),
            "dtype_float64": copied.dtype == np.float64,
            "finite": bool(np.isfinite(copied).all()),
            "sha_exact": array_sha(copied) == V6_S2_ACTUAL_DELTA_SHA256,
            "planned_l2_exact": math.isclose(
                float(audit.get("planned_trial_l2", -1.0)), V6_S2_PLANNED_L2,
                rel_tol=0.0, abs_tol=5.0e-13,
            ),
            "actual_l2_exact": math.isclose(
                float(np.linalg.norm(copied)), V6_S2_ACTUAL_L2,
                rel_tol=0.0, abs_tol=5.0e-13,
            ) and math.isclose(
                float(audit.get("actual_trial_l2", -1.0)), V6_S2_ACTUAL_L2,
                rel_tol=0.0, abs_tol=5.0e-13,
            ),
            "alpha_full": float(audit.get("alpha", -1.0)) == 1.0,
            "trial_merit_exact": math.isclose(
                float(audit.get("trial_merit", -1.0)), V6_S2_TRIAL_MERIT,
                rel_tol=0.0, abs_tol=5.0e-9,
            ),
            "PF_repairs_exact": audit.get("PF_repairs") == {
                "pf0_boundary_a": True,
                "pf0_boundary_b": True,
                "pf7_boundary": False,
            },
            "top1_flip_identity_exact": audit.get("top1_flips") == [TOP1_FDE6_SHA256],
            "retention_flip_identity_exact": audit.get("retention_flips")
            == [RETENTION_243F_SHA256],
            "no_new_pairs": audit.get("new_pair_names") == [],
            "rejected_failed_gate": audit.get("accept") is False
            and audit.get("gate_pass") is False,
            "solver_certified_step_exact": solver.get("certified") is True
            and math.isclose(
                float(solver.get("step_l2", -1.0)), V6_S2_SOLVER_STEP_L2,
                rel_tol=0.0, abs_tol=5.0e-15,
            ),
            "linearization_hashes_exact": linearization.get("matrix_float64_le_sha256")
            == V6_S2_MATRIX_SHA256
            and linearization.get("rhs_float64_le_sha256") == V6_S2_RHS_SHA256,
        }
        if not all(checks.values()):
            raise ProtocolError(f"frozen v6 child S2 capture drift: {checks}")
        copied.setflags(write=False)
        s2_capture.append((copied, dict(audit), checks))

    v4, transform = transformed_v4_v6_capture_module(capture_baseline, capture_s2)
    originals = {
        "TRUST_RADIUS_MAX": v4.TRUST_RADIUS_MAX,
        "MAX_TRIAL_EVALUATIONS": v4.MAX_TRIAL_EVALUATIONS,
        "MAX_ACCEPTED_ITERATES": v4.MAX_ACCEPTED_ITERATES,
        "SEED": v4.SEED,
    }
    v4.TRUST_RADIUS_MAX = V6_FULL_RADIUS
    v4.MAX_TRIAL_EVALUATIONS = 2
    v4.MAX_ACCEPTED_ITERATES = 2
    v4.SEED = 202608046
    try:
        replay = v4.production_run()
    finally:
        for name, value in originals.items():
            setattr(v4, name, value)
    replay_checks = {
        "baseline_capture_once": len(baseline_capture) == 1,
        "S2_capture_once": len(s2_capture) == 1,
        "trial_evaluations_two": replay.get("endpoint", {})
        .get("cuttingplane", {})
        .get("trial_evaluations")
        == 2,
        "accepted_iterates_one": replay.get("endpoint", {})
        .get("cuttingplane", {})
        .get("accepted_iterates")
        == 1,
        "decision_NO_GO": replay.get("endpoint", {}).get("decision")
        == "NO_GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE",
        "payload_absent": replay.get("endpoint", {}).get("candidate_payload") is None,
        "outer_restore": replay.get("outer_restore_pass") is True,
    }
    if not all(replay_checks.values()):
        raise ProtocolError(f"frozen v6 child replay drift: {replay_checks}")
    baseline = baseline_capture[0]
    return {
        "schema_version": SCHEMA + "-isolated-v6-capture",
        "kind": "frozen_v4_B352_baseline_and_v6_rejected_S2",
        "S2_vector": encode_array_xz_base64(s2_capture[0][0], np),
        "baseline_matrix": encode_array_xz_base64(baseline["matrix"], np),
        "baseline_rhs": encode_array_xz_base64(baseline["rhs"], np),
        "baseline_details": baseline["details"],
        "baseline_pair_specs": baseline["pair_specs"],
        "baseline_checks": baseline["checks"],
        "S2_capture_checks": s2_capture[0][2],
        "replay_checks": replay_checks,
        "transform": transform,
        "process_contract": {
            "temporary_files_written": 0,
            "stdout_is_only_transport": True,
            "candidate_payload_present": False,
        },
    }


def isolated_child_entry(kind: str) -> None:
    if kind == "v1":
        result = child_capture_v1()
    elif kind == "v6":
        result = child_capture_v6()
    else:
        raise ProtocolError(f"unknown isolated child kind: {kind}")
    sys.stdout.buffer.write(canonical_json(result))
    sys.stdout.buffer.flush()


def run_isolated_child(kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    child_code = (
        "import runpy\n"
        f"ns=runpy.run_path({str(SCRIPT)!r},run_name='cw24_v9a_isolated_{kind}')\n"
        f"ns['isolated_child_entry']({kind!r})\n"
    )
    command = [str(sys.executable), "-I", "-B", "-c", child_code]
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    process_audit = {
        "kind": kind,
        "returncode": int(process.returncode),
        "executable": command[0],
        "flags": command[1:3],
        "transport": "stdout canonical JSON with xz/base64 arrays",
        "temporary_files_written": 0,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "child_process_exited": process.returncode is not None,
    }
    if process.returncode != 0:
        raise ProtocolError(f"isolated {kind} capture child failed: {process_audit}")
    try:
        payload = json.loads(stdout)
    except Exception as exc:
        raise ProtocolError(
            f"isolated {kind} child stdout is not canonical JSON: {process_audit}"
        ) from exc
    expected_schema = SCHEMA + f"-isolated-{kind}-capture"
    checks = {
        "schema_exact": payload.get("schema_version") == expected_schema,
        "stdout_canonical_json_bit_exact": canonical_json(payload) == stdout,
        "process_exited_zero": process.returncode == 0,
        "stdout_nonempty": bool(stdout),
        "temporary_files_zero": payload.get("process_contract", {}).get(
            "temporary_files_written"
        )
        == 0,
    }
    if not all(checks.values()):
        raise ProtocolError(f"isolated {kind} child contract drift: {checks}")
    process_audit["checks"] = checks
    return payload, process_audit


def production_run() -> dict[str, Any]:
    import numpy as np

    source = self_evidence(require_frozen=True)
    inputs = frozen_inputs()
    runtime_checks = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "R_safe_exact": R_SAFE == 0.000999975,
        "parent_torch_absent_before_children": "torch" not in sys.modules,
    }
    if not all(runtime_checks.values()):
        raise ProtocolError(f"v9a runtime drift: {runtime_checks}")

    v1_child, v1_process = run_isolated_child("v1")
    if "torch" in sys.modules:
        raise ProtocolError("v1 child contaminated parent sys.modules with torch")
    v6_child, v6_process = run_isolated_child("v6")
    if "torch" in sys.modules:
        raise ProtocolError("v6 child contaminated parent sys.modules with torch")

    v1_delta = decode_array_xz_base64(v1_child["vector"], np)
    v6_delta = decode_array_xz_base64(v6_child["S2_vector"], np)
    baseline_matrix = decode_array_xz_base64(v6_child["baseline_matrix"], np)
    baseline_rhs = decode_array_xz_base64(v6_child["baseline_rhs"], np)
    transport_checks = {
        "v1_vector_sha_exact": array_sha(v1_delta) == V1_ACTUAL_DELTA_SHA256,
        "v6_vector_shape_finite_sha_exact": v6_delta.shape == (65793,)
        and bool(np.isfinite(v6_delta).all())
        and array_sha(v6_delta) == V6_S2_ACTUAL_DELTA_SHA256,
        "baseline_shapes_exact": baseline_matrix.shape == (20, 65793)
        and baseline_rhs.shape == (20,),
        "baseline_matrix_sha_exact": array_sha(baseline_matrix)
        == V4_BASELINE_MATRIX_SHA256,
        "baseline_rhs_sha_exact": array_sha(baseline_rhs) == V4_BASELINE_RHS_SHA256,
        "two_independent_child_contracts": v1_process["kind"] == "v1"
        and v6_process["kind"] == "v6",
        "both_children_exited_zero": v1_process["returncode"] == 0
        and v6_process["returncode"] == 0,
        "parent_torch_still_absent": "torch" not in sys.modules,
        "temporary_files_zero": v1_process["temporary_files_written"] == 0
        and v6_process["temporary_files_written"] == 0,
    }
    if not all(transport_checks.values()):
        raise ProtocolError(f"isolated child transport drift: {transport_checks}")

    v1_capture = [(v1_delta, dict(v1_child["capture_checks"]))]
    s2_capture = [
        (
            v6_delta,
            {},
            dict(v6_child["S2_capture_checks"]),
        )
    ]
    baseline_capture = [
        {
            "matrix": baseline_matrix,
            "rhs": baseline_rhs,
            "details": list(v6_child["baseline_details"]),
            "pair_specs": list(v6_child["baseline_pair_specs"]),
            "checks": dict(v6_child["baseline_checks"]),
        }
    ]
    v1_transform = dict(v1_child["transform"])
    v4_transform = dict(v6_child["transform"])
    v1_result_checks = dict(v1_child["replay_checks"])
    v6_replay_checks = dict(v6_child["replay_checks"])
    v1_cleanup = {
        "policy": "fresh isolated subprocess; process exit discards all child sys.modules",
        "process": v1_process,
        "parent_torch_absent_after_exit": "torch" not in sys.modules,
        "checks": {
            "child_exited_zero": v1_process["returncode"] == 0,
            "no_parent_module_contamination": "torch" not in sys.modules,
            "no_temporary_files": v1_process["temporary_files_written"] == 0,
        },
    }
    v6_cleanup = {
        "policy": "fresh isolated subprocess; process exit discards all child sys.modules",
        "process": v6_process,
        "parent_torch_absent_after_exit": "torch" not in sys.modules,
        "checks": {
            "child_exited_zero": v6_process["returncode"] == 0,
            "no_parent_module_contamination": "torch" not in sys.modules,
            "no_temporary_files": v6_process["temporary_files_written"] == 0,
        },
    }
    v1_delta = v1_capture[0][0]
    v6_delta = s2_capture[0][0]
    v1_norm = float(np.linalg.norm(v1_delta))
    v6_norm = float(np.linalg.norm(v6_delta))
    unit_v1 = np.asarray(v1_delta / v1_norm, dtype=np.float64)
    unit_v6 = np.asarray(v6_delta / v6_norm, dtype=np.float64)
    cosine = float(np.clip(unit_v1 @ unit_v6, -1.0, 1.0))
    angle = math.acos(cosine)
    if not 0.0 < angle < math.pi:
        raise ProtocolError("v1/v6 directions do not define a unique minor arc")
    bisector = np.asarray(unit_v1 + unit_v6, dtype=np.float64)
    bisector_norm = float(np.linalg.norm(bisector))
    if not math.isfinite(bisector_norm) or bisector_norm <= 0.0:
        raise ProtocolError("v1/v6 minor-arc bisector is degenerate")
    midpoint_unit = np.asarray(bisector / bisector_norm, dtype=np.float64)
    midpoint = np.asarray(R_SAFE * midpoint_unit, dtype=np.float64)
    swapped_unit = np.asarray(
        (unit_v6 + unit_v1) / np.linalg.norm(unit_v6 + unit_v1), dtype=np.float64
    )
    swapped_midpoint = np.asarray(R_SAFE * swapped_unit, dtype=np.float64)
    angle_v1_mid = math.acos(float(np.clip(unit_v1 @ midpoint_unit, -1.0, 1.0)))
    angle_v6_mid = math.acos(float(np.clip(unit_v6 @ midpoint_unit, -1.0, 1.0)))
    slerp_weight = math.sin(0.5 * angle) / math.sin(angle)
    slerp_midpoint_unit = np.asarray(
        slerp_weight * unit_v1 + slerp_weight * unit_v6, dtype=np.float64
    )
    slerp_midpoint_unit /= np.linalg.norm(slerp_midpoint_unit)
    geometry_checks = {
        "input_vectors_finite": bool(np.isfinite(v1_delta).all())
        and bool(np.isfinite(v6_delta).all()),
        "unit_norms": math.isclose(float(np.linalg.norm(unit_v1)), 1.0, abs_tol=5e-15)
        and math.isclose(float(np.linalg.norm(unit_v6)), 1.0, abs_tol=5e-15),
        "unique_minor_arc": 0.0 < angle < math.pi,
        "midpoint_radius_exact": math.isclose(
            float(np.linalg.norm(midpoint)), R_SAFE, rel_tol=0.0, abs_tol=5e-15
        ),
        "half_angles_equal": math.isclose(
            angle_v1_mid, 0.5 * angle, rel_tol=0.0, abs_tol=5e-13
        ) and math.isclose(
            angle_v6_mid, 0.5 * angle, rel_tol=0.0, abs_tol=5e-13
        ),
        "exchange_symmetric_bit_exact": bool(np.array_equal(midpoint, swapped_midpoint)),
        "equal_weight_slerp_matches_bisector": bool(
            np.allclose(midpoint_unit, slerp_midpoint_unit, rtol=0.0, atol=5e-15)
        ),
        "safe_below_v4_v6_cap": R_SAFE < V6_FULL_RADIUS,
    }
    if not all(geometry_checks.values()):
        raise ProtocolError(f"spherical midpoint geometry failed: {geometry_checks}")

    baseline = baseline_capture[0]
    matrix = baseline["matrix"]
    rhs = baseline["rhs"]
    details = baseline["details"]
    detail_by_name = {str(item["name"]): item for item in details}
    row_by_name = {str(item["name"]): index for index, item in enumerate(details)}
    normalized_change = matrix @ midpoint
    normalized_slack = normalized_change - rhs

    def pair_prediction(name: str) -> dict[str, Any]:
        index = row_by_name[name]
        detail = detail_by_name[name]
        raw_change = float(normalized_change[index] * float(detail["gradient_l2"]))
        current = float(detail["current"])
        target = float(detail["target"])
        predicted = current + raw_change
        return {
            "name": name,
            "row": index,
            "current_native_margin": current,
            "gradient_l2": float(detail["gradient_l2"]),
            "predicted_linear_change": raw_change,
            "predicted_linear_margin": predicted,
            "original_gate_target": target,
            "normalized_gate_slack": float(normalized_slack[index]),
            "predicted_protected": predicted >= target,
        }

    pf_predictions = {
        name: pair_prediction(name)
        for name in ("pf0_boundary_a", "pf0_boundary_b", "pf7_boundary")
    }
    protected_predictions = {
        name: pair_prediction(name) for name in PROTECTED_PAIR_NAMES
    }
    linear_checks = {
        "PF_A_predicted_change_positive": pf_predictions["pf0_boundary_a"][
            "predicted_linear_change"
        ]
        > 0.0,
        "PF7_predicted_change_positive": pf_predictions["pf7_boundary"][
            "predicted_linear_change"
        ]
        > 0.0,
        "PF_B_predicted_change_nonnegative": pf_predictions["pf0_boundary_b"][
            "predicted_linear_change"
        ]
        >= 0.0,
        "six_zero_guards_predicted_protected": all(
            protected_predictions[f"zero_margin_guard_{index}"]["predicted_protected"]
            for index in range(6)
        ),
        "retention_0142_predicted_protected": protected_predictions[
            "historical_retention_0142a2a3d5bd"
        ]["predicted_protected"],
        "retention_243f_predicted_protected": protected_predictions[
            "historical_retention_243f4a21d6c0"
        ]["predicted_protected"],
        "top1_fde6_predicted_protected": protected_predictions[
            "historical_top1_fde6fab07449"
        ]["predicted_protected"],
    }
    preflight_pass = all(linear_checks.values())
    decision = (
        "GO_CW24_V9A_SPHERICAL_MIDPOINT_LINEAR_PREFLIGHT"
        if preflight_pass
        else "NO_GO_CW24_V9A_SPHERICAL_MIDPOINT_LINEAR_PREFLIGHT"
    )
    result = {
        "schema_version": SCHEMA,
        "status": decision,
        "decision": decision,
        "seed": SEED,
        "purpose": "train-only geometry and frozen-B352 linear preflight; not promotion evidence",
        "inputs": inputs,
        "capture": {
            "v1": {
                "actual_delta_float64_le_sha256": array_sha(v1_delta),
                "actual_l2": v1_norm,
                "checks": v1_capture[0][1],
                "transform": v1_transform,
                "replay_checks": v1_result_checks,
                "sys_modules_cleanup": v1_cleanup,
            },
            "v6_rejected_S2": {
                "actual_delta_float64_le_sha256": array_sha(v6_delta),
                "actual_l2": v6_norm,
                "checks": s2_capture[0][2],
                "transform": v4_transform,
                "replay_checks": v6_replay_checks,
                "sys_modules_cleanup": v6_cleanup,
            },
            "v4_B352_CW11_baseline": {
                "matrix_float64_le_sha256": array_sha(matrix),
                "rhs_float64_le_sha256": array_sha(rhs),
                "matrix_shape": list(matrix.shape),
                "checks": baseline["checks"],
            },
        },
        "spherical_midpoint": {
            "radius": R_SAFE,
            "equal_weights": [0.5, 0.5],
            "construction": "normalize actual deltas then equal-weight minor-arc midpoint",
            "cosine": cosine,
            "angle_radians": angle,
            "half_angle_radians": 0.5 * angle,
            "angle_v1_to_midpoint_radians": angle_v1_mid,
            "angle_v6_to_midpoint_radians": angle_v6_mid,
            "slerp_equal_weight": slerp_weight,
            "unit_v1_float64_le_sha256": array_sha(unit_v1),
            "unit_v6_float64_le_sha256": array_sha(unit_v6),
            "bisector_float64_le_sha256": array_sha(bisector),
            "midpoint_unit_float64_le_sha256": array_sha(midpoint_unit),
            "midpoint_float64_le_sha256": array_sha(midpoint),
            "swapped_midpoint_float64_le_sha256": array_sha(swapped_midpoint),
            "checks": geometry_checks,
        },
        "linear_preflight": {
            "coordinate": "absolute actor delta from exact CW11",
            "midpoint_forward_performed": False,
            "PF_predictions": pf_predictions,
            "protected_pair_predictions": protected_predictions,
            "checks": linear_checks,
            "pass": preflight_pass,
        },
        "audit": {
            "runtime_checks": runtime_checks,
            "source": source,
            "train_only_B352": True,
            "new_validation_or_test_rows_opened": 0,
            "midpoint_model_forward_calls": 0,
            "new_changed_candidate_train_shadow_count": 0,
            "historical_frozen_train_shadow_replays": {
                "v1_terminal_failed_shadow": 1,
                "v6_S1_and_rejected_S2": 2,
            },
            "checkpoint_writes": 0,
            "model_artifact_writes": 0,
            "optimizer_instances_created": 0,
            "optimizer_step_calls": 0,
            "network_calls": 0,
            "candidate_payload_present": False,
            "official_unique_changed_candidate_count_consumed": 0,
            "submission_performed": False,
        },
        "official_unique_changed_candidate_count_consumed": 0,
        "cumulative_official_unique_changed_candidate_count": 2,
        "submission_performed": False,
        "package_upload_performed": False,
    }
    canonical_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        inputs = frozen_inputs()
        noop_v1 = lambda _delta, _trial: None
        noop_baseline = lambda _matrix, _rhs, _details, _pairs, _audit: None
        noop_s2 = lambda _delta, _audit: None
        before_v1 = set(sys.modules)
        try:
            _, v1_transform = transformed_v1_module(noop_v1)
        finally:
            v1_cleanup = cleanup_phase_modules(before_v1, "audit_v1_transform")
        before_v4 = set(sys.modules)
        try:
            _, v4_transform = transformed_v4_v6_capture_module(
                noop_baseline, noop_s2
            )
        finally:
            v4_cleanup = cleanup_phase_modules(before_v4, "audit_v4_v6_transform")
        checks = {
            "v1_transform_checks": all(v1_transform["checks"].values()),
            "v4_v6_transform_checks": all(v4_transform["checks"].values()),
            "transform_cleanup_checks": all(v1_cleanup["checks"].values())
            and all(v4_cleanup["checks"].values()),
            "R_safe_exact": R_SAFE == 0.000999975,
            "output_absent": not OUTPUT.exists(),
        }
        if not all(checks.values()):
            raise ProtocolError(f"v9a static audit failed: {checks}")
        print(
            canonical_json(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_pass",
                    "source": self_evidence(require_frozen=True),
                    "inputs": inputs,
                    "v1_transform": v1_transform,
                    "v4_v6_transform": v4_transform,
                    "module_cleanup": [v1_cleanup, v4_cleanup],
                    "variant": {
                        "seed": SEED,
                        "R_safe": R_SAFE,
                        "equal_weights": [0.5, 0.5],
                        "protected_pair_names": list(PROTECTED_PAIR_NAMES),
                        "midpoint_forward_performed": False,
                        "new_candidate_shadow_count": 0,
                    },
                    "checks": checks,
                    "python_exact": Path(sys.executable).resolve()
                    == EXPECTED_PYTHON.resolve(),
                    "CUDA_initialized": False,
                    "validation_or_test_rows_opened": 0,
                    "writes_performed": 0,
                }
            ).decode(),
            end="",
        )
        return
    if OUTPUT.exists():
        raise ProtocolError("frozen CW24 v9a output already exists")
    result = production_run()
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "midpoint_float64_le_sha256": result["spherical_midpoint"][
                    "midpoint_float64_le_sha256"
                ],
                "linear_preflight_pass": result["linear_preflight"]["pass"],
                "midpoint_forward_performed": False,
                "new_changed_candidate_train_shadow_count": 0,
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
                "output": publication,
            }
        ).decode(),
        end="",
    )


if __name__ == "__main__":
    main()

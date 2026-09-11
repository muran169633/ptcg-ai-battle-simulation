#!/usr/bin/env python3
"""CW24 v10a frozen-v1/v6 minor-arc analytic feasibility preflight.

This train-only diagnostic replays two already frozen failed recipes solely to
capture their actual float32 actor deltas in two isolated child processes: the
v1 terminal failure and the rejected second v6 shadow.  In the parent it forms
the fixed-radius minor arc ``u(phi)=cos(phi)u1+sin(phi)e2`` and analytically
enumerates every sinusoidal constraint root on ``phi in [0, theta]``.  It then
constructs the exact feasible connected components and, if any exist, chooses
the midpoint of the widest component (ties: smallest phi).  There is no grid,
random, radius, or weight scan.  The selected vector is never copied to a model
and never receives a forward pass.  No new candidate shadow, payload,
checkpoint, model artifact, or official candidate is created.
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
SCRIPT = TOOLS / "probe_u468_cw11_v1_v6_minor_arc_feasible_intervals_cw24_v10a.py"
OUTPUT = ROOT / "artifacts/cw24_cw11_v1_v6_minor_arc_feasible_intervals_cw24_v10a.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-v1-v6-minor-arc-feasible-intervals-cw24-v10a"
SEED = 202608049
R_SAFE = 0.000999975
V6_FULL_RADIUS = 0.00099998
PF_STRICT_RAW_CHANGE_BUFFER = 1.0e-6
ROOT_ENUM_TOLERANCE_RADIANS = 5.0e-14
INTERVAL_MERGE_TOLERANCE_RADIANS = 5.0e-14
ROOT_RESIDUAL_TOLERANCE_NORMALIZED = 5.0e-15
WIDTH_AMBIGUITY_TOLERANCE_RADIANS = 5.0e-14
CHEBYSHEV_CERT_TOLERANCE_RADIANS = 5.0e-15

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

V9B_SOURCE = TOOLS / "probe_u468_cw11_v1_v6_spherical_midpoint_preflight_cw24_v9b.py"
V9B_SOURCE_SHA256 = "1d27612293d4807b70af9bdba01922076bf205ed5aee906e4a00850b6eb2c4b5"
V9B_SOURCE_MODE = 0o555
V9B_RESULT = ROOT / "artifacts/cw24_cw11_v1_v6_spherical_midpoint_preflight_cw24_v9b.json"
V9B_RESULT_SHA256 = "666aa26467056c2229e72430b88a52dc0af1677e3cb8ff273f9268f3c242837d"
V9B_RESULT_MODE = 0o444

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
    """Fail-closed CW24 v10a preflight error."""


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
                V9B_SOURCE_SHA256,
            )
        ),
        "all_result_locks_present": all(
            digest.encode() in source
            for digest in (
                V1_RESULT_SHA256,
                V4_RESULT_SHA256,
                V6_RESULT_SHA256,
                V8_RESULT_SHA256,
                V9B_RESULT_SHA256,
            )
        ),
        "safe_radius_single_constant": source.count(b"\nR_SAFE =") == 1
        and R_SAFE == 0.000999975,
        "fixed_positive_buffer_single_constant": source.count(
            b"\nPF_STRICT_RAW_CHANGE_BUFFER ="
        ) == 1
        and PF_STRICT_RAW_CHANGE_BUFFER == 1.0e-6,
        "root_tolerance_single_constant": source.count(
            b"\nROOT_ENUM_TOLERANCE_RADIANS ="
        ) == 1
        and ROOT_ENUM_TOLERANCE_RADIANS == 5.0e-14,
        "interval_merge_tolerance_single_constant": source.count(
            b"\nINTERVAL_MERGE_TOLERANCE_RADIANS ="
        ) == 1
        and INTERVAL_MERGE_TOLERANCE_RADIANS == 5.0e-14,
        "root_residual_tolerance_single_constant": source.count(
            b"\nROOT_RESIDUAL_TOLERANCE_NORMALIZED ="
        ) == 1
        and ROOT_RESIDUAL_TOLERANCE_NORMALIZED == 5.0e-15,
        "width_ambiguity_tolerance_single_constant": source.count(
            b"\nWIDTH_AMBIGUITY_TOLERANCE_RADIANS ="
        ) == 1
        and WIDTH_AMBIGUITY_TOLERANCE_RADIANS == 5.0e-14,
        "Chebyshev_cert_tolerance_single_constant": source.count(
            b"\nCHEBYSHEV_CERT_TOLERANCE_RADIANS ="
        ) == 1
        and CHEBYSHEV_CERT_TOLERANCE_RADIANS == 5.0e-15,
        "no_grid_or_random_search": all(
            token not in source
            for token in (
                b"np." + b"linspace(",
                b"np." + b"arange(",
                b"import " + b"random",
                b"np." + b"random",
            )
        ),
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
        raise ProtocolError(f"v10a source audit failed: {checks}")
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
                raise ProtocolError("short v10a result write")
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
        raise ProtocolError(f"v10a result publication drift: {checks}")
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
            V10A_CAPTURE_V1_FAILED_ACTUAL_DELTA(actual_delta.copy(), trial)
            return {
'''
    audits: list[dict[str, Any]] = []
    source = splice_once(source, old, new, "capture_frozen_v1_failed_actual_delta", audits)
    transformed = source.encode()
    name = "cw24_v10a_transformed_frozen_v1"
    if name in sys.modules:
        raise ProtocolError("v10a transformed v1 module name occupied")
    module = types.ModuleType(name)
    module.__file__ = str(V1_SOURCE)
    module.__package__ = ""
    module.__dict__["V10A_CAPTURE_V1_FAILED_ACTUAL_DELTA"] = capture
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
        "callback_identity_exact": module.V10A_CAPTURE_V1_FAILED_ACTUAL_DELTA is capture,
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
                V10A_CAPTURE_V4_BASELINE_LINEARIZATION(
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
                V10A_CAPTURE_V6_REJECTED_S2_ACTUAL_DELTA(
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
    name = "cw24_v10a_transformed_frozen_v4_v6"
    if name in sys.modules:
        raise ProtocolError("v10a transformed v4/v6 module name occupied")
    module = types.ModuleType(name)
    module.__file__ = str(V4_SOURCE)
    module.__package__ = ""
    module.__dict__["V10A_CAPTURE_V4_BASELINE_LINEARIZATION"] = capture_baseline
    module.__dict__["V10A_CAPTURE_V6_REJECTED_S2_ACTUAL_DELTA"] = capture_s2
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
        "callbacks_identity_exact": module.V10A_CAPTURE_V4_BASELINE_LINEARIZATION
        is capture_baseline
        and module.V10A_CAPTURE_V6_REJECTED_S2_ACTUAL_DELTA is capture_s2,
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
        ("v9b_source", V9B_SOURCE, V9B_SOURCE_SHA256, V9B_SOURCE_MODE, "v9b source"),
        ("v9b_result", V9B_RESULT, V9B_RESULT_SHA256, V9B_RESULT_MODE, "v9b result"),
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


def enumerate_sinusoid_roots(
    constraint: Mapping[str, Any], theta: float
) -> dict[str, Any]:
    """Enumerate all roots of a*cos(phi)+b*sin(phi)+c on [0, theta]."""
    a_cos = float(constraint["a_cos"])
    b_sin = float(constraint["b_sin"])
    c_offset = float(constraint["c_offset"])
    amplitude = math.hypot(a_cos, b_sin)
    phase = math.atan2(b_sin, a_cos) if amplitude > 0.0 else 0.0
    roots: list[float] = []
    equation_target: float | None = None
    classification = "nonconstant"
    if amplitude == 0.0:
        classification = "constant_no_roots"
    else:
        equation_target = -c_offset / amplitude
        if equation_target < -1.0 or equation_target > 1.0:
            classification = "amplitude_cannot_reach_zero"
        else:
            alpha = math.acos(float(min(1.0, max(-1.0, equation_target))))
            period = 2.0 * math.pi
            for base in (phase - alpha, phase + alpha):
                k_min = math.ceil(
                    (0.0 - base - ROOT_ENUM_TOLERANCE_RADIANS) / period
                )
                k_max = math.floor(
                    (theta - base + ROOT_ENUM_TOLERANCE_RADIANS) / period
                )
                for period_index in range(k_min, k_max + 1):
                    root = base + period * period_index
                    if not (
                        -ROOT_ENUM_TOLERANCE_RADIANS
                        <= root
                        <= theta + ROOT_ENUM_TOLERANCE_RADIANS
                    ):
                        continue
                    if abs(root) <= ROOT_ENUM_TOLERANCE_RADIANS:
                        root = 0.0
                    elif abs(root - theta) <= ROOT_ENUM_TOLERANCE_RADIANS:
                        root = theta
                    roots.append(float(root))
    roots.sort()
    deduplicated: list[float] = []
    for root in roots:
        if (
            not deduplicated
            or root - deduplicated[-1] > ROOT_ENUM_TOLERANCE_RADIANS
        ):
            deduplicated.append(root)
    residuals = [
        a_cos * math.cos(root) + b_sin * math.sin(root) + c_offset
        for root in deduplicated
    ]
    if any(abs(residual) > ROOT_RESIDUAL_TOLERANCE_NORMALIZED for residual in residuals):
        raise ProtocolError(
            f"analytic root residual exceeded fixed tolerance for {constraint['name']}"
        )
    return {
        "name": str(constraint["name"]),
        "relation": str(constraint["relation"]),
        "a_cos": a_cos,
        "b_sin": b_sin,
        "c_offset": c_offset,
        "amplitude": amplitude,
        "phase_radians": phase,
        "root_equation_cos_target": equation_target,
        "root_classification": classification,
        "roots_radians": deduplicated,
        "root_equation_residuals_normalized": residuals,
        "max_abs_root_equation_residual_normalized": max(
            (abs(residual) for residual in residuals), default=0.0
        ),
        "root_count": len(deduplicated),
    }


def merge_root_boundaries(
    constraints: list[dict[str, Any]],
    root_analyses: list[dict[str, Any]],
    theta: float,
) -> list[dict[str, Any]]:
    constraint_by_name = {str(item["name"]): item for item in constraints}
    candidates: list[dict[str, Any]] = [
        {"phi": 0.0, "root_names": set(), "left_endpoint": True},
        {"phi": theta, "root_names": set(), "right_endpoint": True},
    ]
    for analysis in root_analyses:
        for root in analysis["roots_radians"]:
            candidates.append(
                {
                    "phi": float(root),
                    "root_names": {str(analysis["name"])},
                }
            )
    candidates.sort(key=lambda item: float(item["phi"]))
    groups: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        if (
            not groups
            or float(candidate["phi"]) - float(groups[-1][-1]["phi"])
            > INTERVAL_MERGE_TOLERANCE_RADIANS
        ):
            groups.append([candidate])
        else:
            groups[-1].append(candidate)

    boundaries: list[dict[str, Any]] = []
    for group in groups:
        has_left = any(bool(item.get("left_endpoint")) for item in group)
        has_right = any(bool(item.get("right_endpoint")) for item in group)
        if has_left and has_right:
            raise ProtocolError("minor arc collapsed inside interval merge tolerance")
        if has_left:
            phi = 0.0
        elif has_right:
            phi = theta
        else:
            phi = min(float(item["phi"]) for item in group)
        root_names = sorted(
            {
                name
                for item in group
                for name in item.get("root_names", set())
            }
        )
        representative_residuals = {}
        for name in root_names:
            constraint = constraint_by_name[name]
            residual = (
                float(constraint["a_cos"]) * math.cos(phi)
                + float(constraint["b_sin"]) * math.sin(phi)
                + float(constraint["c_offset"])
            )
            representative_residuals[name] = residual
            if abs(residual) > ROOT_RESIDUAL_TOLERANCE_NORMALIZED:
                raise ProtocolError(
                    f"merged-root representative residual exceeded fixed tolerance for {name}"
                )
        boundaries.append(
            {
                "phi_radians": phi,
                "root_names": root_names,
                "representative_root_residuals_normalized": representative_residuals,
                "merged_candidate_count": len(group),
                "left_endpoint": has_left,
                "right_endpoint": has_right,
            }
        )
    if boundaries[0]["phi_radians"] != 0.0 or boundaries[-1]["phi_radians"] != theta:
        raise ProtocolError("analytic boundary construction lost an endpoint")
    return boundaries


def evaluate_constraints_at_phi(
    constraints: list[dict[str, Any]],
    phi: float,
    equality_root_names: set[str] | None = None,
) -> dict[str, Any]:
    equality_root_names = equality_root_names or set()
    values: dict[str, float] = {}
    passes: dict[str, bool] = {}
    for constraint in constraints:
        name = str(constraint["name"])
        value = (
            float(constraint["a_cos"]) * math.cos(phi)
            + float(constraint["b_sin"]) * math.sin(phi)
            + float(constraint["c_offset"])
        )
        values[name] = value
        if name in equality_root_names:
            if abs(value) > ROOT_RESIDUAL_TOLERANCE_NORMALIZED:
                raise ProtocolError(
                    f"root-point residual exceeded fixed tolerance for {name}"
                )
            passes[name] = (
                constraint["relation"] == "nonnegative"
                and value >= -ROOT_RESIDUAL_TOLERANCE_NORMALIZED
            )
        elif constraint["relation"] == "strict_positive":
            passes[name] = value > 0.0
        elif constraint["relation"] == "nonnegative":
            passes[name] = value >= 0.0
        else:
            raise ProtocolError(f"unknown sinusoid relation for {name}")
    return {
        "phi_radians": phi,
        "constraint_values": values,
        "constraint_passes": passes,
        "equality_root_names": sorted(equality_root_names),
        "failing_names": sorted(name for name, passed in passes.items() if not passed),
        "pass": all(passes.values()),
    }


def construct_feasible_components(
    constraints: list[dict[str, Any]], boundaries: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    point_cells = [
        evaluate_constraints_at_phi(
            constraints,
            float(boundary["phi_radians"]),
            set(boundary["root_names"]),
        )
        for boundary in boundaries
    ]
    open_cells: list[dict[str, Any]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        lower = float(left["phi_radians"])
        upper = float(right["phi_radians"])
        if upper <= lower:
            raise ProtocolError("non-increasing analytic root boundaries")
        cell = evaluate_constraints_at_phi(constraints, 0.5 * (lower + upper))
        cell.update(
            {
                "lower_phi_radians": lower,
                "upper_phi_radians": upper,
                "lower_inclusive": False,
                "upper_inclusive": False,
            }
        )
        open_cells.append(cell)

    components: list[dict[str, Any]] = []
    segment_index = 0
    while segment_index < len(open_cells):
        if not open_cells[segment_index]["pass"]:
            segment_index += 1
            continue
        first_segment = segment_index
        last_segment = segment_index
        while (
            last_segment + 1 < len(open_cells)
            and point_cells[last_segment + 1]["pass"]
            and open_cells[last_segment + 1]["pass"]
        ):
            last_segment += 1
        lower = float(boundaries[first_segment]["phi_radians"])
        upper = float(boundaries[last_segment + 1]["phi_radians"])
        components.append(
            {
                "lower_phi_radians": lower,
                "upper_phi_radians": upper,
                "angular_width_radians": upper - lower,
                "lower_inclusive": bool(point_cells[first_segment]["pass"]),
                "upper_inclusive": bool(point_cells[last_segment + 1]["pass"]),
                "first_open_cell_index": first_segment,
                "last_open_cell_index": last_segment,
                "isolated_point": False,
            }
        )
        segment_index = last_segment + 1

    for point_index, point in enumerate(point_cells):
        left_feasible = point_index > 0 and bool(open_cells[point_index - 1]["pass"])
        right_feasible = point_index < len(open_cells) and bool(
            open_cells[point_index]["pass"]
        )
        if point["pass"] and not left_feasible and not right_feasible:
            phi = float(boundaries[point_index]["phi_radians"])
            components.append(
                {
                    "lower_phi_radians": phi,
                    "upper_phi_radians": phi,
                    "angular_width_radians": 0.0,
                    "lower_inclusive": True,
                    "upper_inclusive": True,
                    "first_open_cell_index": None,
                    "last_open_cell_index": None,
                    "isolated_point": True,
                }
            )
    components.sort(
        key=lambda item: (
            float(item["lower_phi_radians"]),
            float(item["upper_phi_radians"]),
        )
    )
    for index, component in enumerate(components):
        component["component_index"] = index
    return components, point_cells, open_cells


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
        "outer_restore": result.get("endpoint", {}).get("outer_restore_pass") is True,
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
        "outer_restore": replay.get("endpoint", {}).get("outer_restore_pass") is True,
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
        f"ns=runpy.run_path({str(SCRIPT)!r},run_name='cw24_v10a_isolated_{kind}')\n"
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
        raise ProtocolError(f"v10a runtime drift: {runtime_checks}")

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
    theta = math.acos(cosine)
    if not 0.0 < theta < math.pi:
        raise ProtocolError("v1/v6 directions do not define a unique minor arc")
    orthogonal_residual = np.asarray(unit_v6 - cosine * unit_v1, dtype=np.float64)
    orthogonal_norm = float(np.linalg.norm(orthogonal_residual))
    e2 = np.asarray(orthogonal_residual / orthogonal_norm, dtype=np.float64)
    reconstructed_v6 = np.asarray(
        math.cos(theta) * unit_v1 + math.sin(theta) * e2,
        dtype=np.float64,
    )
    geometry_checks = {
        "input_vectors_finite": bool(np.isfinite(v1_delta).all())
        and bool(np.isfinite(v6_delta).all()),
        "unit_norms": math.isclose(float(np.linalg.norm(unit_v1)), 1.0, abs_tol=5e-15)
        and math.isclose(float(np.linalg.norm(unit_v6)), 1.0, abs_tol=5e-15),
        "unique_minor_arc": 0.0 < theta < math.pi,
        "orthogonal_residual_nonzero": math.isfinite(orthogonal_norm)
        and orthogonal_norm > 0.0,
        "e2_unit": math.isclose(
            float(np.linalg.norm(e2)), 1.0, rel_tol=0.0, abs_tol=5e-15
        ),
        "basis_orthogonal": math.isclose(
            float(unit_v1 @ e2), 0.0, rel_tol=0.0, abs_tol=5e-15
        ),
        "theta_endpoint_reconstructs_v6": bool(
            np.allclose(reconstructed_v6, unit_v6, rtol=0.0, atol=5e-15)
        ),
        "sin_theta_matches_residual_norm": math.isclose(
            math.sin(theta), orthogonal_norm, rel_tol=0.0, abs_tol=5e-15
        ),
        "safe_below_v4_v6_cap": R_SAFE < V6_FULL_RADIUS,
    }
    if not all(geometry_checks.values()):
        raise ProtocolError(f"minor-arc basis geometry failed: {geometry_checks}")

    baseline = baseline_capture[0]
    matrix = baseline["matrix"]
    rhs = baseline["rhs"]
    details = baseline["details"]
    detail_by_name = {str(item["name"]): item for item in details}
    row_by_name = {str(item["name"]): index for index, item in enumerate(details)}
    required_names = {
        "pf0_boundary_a",
        "pf0_boundary_b",
        "pf7_boundary",
        *PROTECTED_PAIR_NAMES,
    }
    detail_checks = {
        "details_unique20": len(details) == len(detail_by_name) == 20,
        "all_required_rows_present": required_names.issubset(row_by_name),
        "all_required_gradient_norms_positive": all(
            float(detail_by_name[name]["gradient_l2"]) > 0.0
            for name in required_names
        ),
    }
    if not all(detail_checks.values()):
        raise ProtocolError(f"frozen B352 row metadata drift: {detail_checks}")

    constraints: list[dict[str, Any]] = []
    pf_definitions = (
        ("pf0_boundary_a", "PF_A", "strict_positive", PF_STRICT_RAW_CHANGE_BUFFER),
        ("pf0_boundary_b", "PF_B", "nonnegative", 0.0),
        ("pf7_boundary", "PF7", "strict_positive", PF_STRICT_RAW_CHANGE_BUFFER),
    )
    for name, role, relation, raw_buffer in pf_definitions:
        row = row_by_name[name]
        gradient_l2 = float(detail_by_name[name]["gradient_l2"])
        normalized_threshold = raw_buffer / gradient_l2
        constraints.append(
            {
                "name": name,
                "role": role,
                "row": row,
                "relation": relation,
                "a_cos": float(R_SAFE * (matrix[row] @ unit_v1)),
                "b_sin": float(R_SAFE * (matrix[row] @ e2)),
                "c_offset": -normalized_threshold,
                "raw_change_buffer": raw_buffer,
                "normalized_change_threshold": normalized_threshold,
                "gradient_l2": gradient_l2,
                "criterion": "predicted linear native-margin change",
            }
        )
    for name in PROTECTED_PAIR_NAMES:
        row = row_by_name[name]
        detail = detail_by_name[name]
        constraints.append(
            {
                "name": name,
                "role": "protected_frozen_target",
                "row": row,
                "relation": "nonnegative",
                "a_cos": float(R_SAFE * (matrix[row] @ unit_v1)),
                "b_sin": float(R_SAFE * (matrix[row] @ e2)),
                "c_offset": -float(rhs[row]),
                "raw_change_buffer": None,
                "normalized_change_threshold": float(rhs[row]),
                "gradient_l2": float(detail["gradient_l2"]),
                "current_native_margin": float(detail["current"]),
                "frozen_target_native_margin": float(detail["target"]),
                "criterion": "predicted frozen-target normalized gate slack",
            }
        )
    if len(constraints) != 12 or len({item["name"] for item in constraints}) != 12:
        raise ProtocolError("analytic constraint roster is not exactly 3 PF plus 9 protected")

    root_analyses = [
        enumerate_sinusoid_roots(constraint, theta) for constraint in constraints
    ]
    boundaries = merge_root_boundaries(constraints, root_analyses, theta)
    feasible_components, point_cells, open_cells = construct_feasible_components(
        constraints, boundaries
    )
    positive_width_components = [
        component
        for component in feasible_components
        if not component["isolated_point"]
        and float(component["angular_width_radians"]) > 0.0
    ]
    selected_component: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    Chebyshev_certificate: dict[str, Any] | None = None
    selected_constraint_slacks: dict[str, Any] | None = None
    all_baseline_row_slacks: dict[str, Any] | None = None
    global_width_ordering = sorted(
        [
            {
                "component_index": int(component["component_index"]),
                "angular_width_radians": float(component["angular_width_radians"]),
                "lower_phi_radians": float(component["lower_phi_radians"]),
                "upper_phi_radians": float(component["upper_phi_radians"]),
                "isolated_point": bool(component["isolated_point"]),
            }
            for component in feasible_components
        ],
        key=lambda item: (
            -float(item["angular_width_radians"]),
            float(item["lower_phi_radians"]),
            int(item["component_index"]),
        ),
    )
    if positive_width_components:
        widest = max(
            float(component["angular_width_radians"])
            for component in positive_width_components
        )
        ambiguous_near_ties = [
            component
            for component in positive_width_components
            if 0.0
            < widest - float(component["angular_width_radians"])
            <= WIDTH_AMBIGUITY_TOLERANCE_RADIANS
        ]
        if ambiguous_near_ties:
            raise ProtocolError(
                "non-bit-exact near tie inside fixed width ambiguity tolerance"
            )
        widest_ties = [
            component
            for component in positive_width_components
            if float(component["angular_width_radians"]) == widest
        ]
        selected_component = min(
            widest_ties,
            key=lambda component: (
                float(component["lower_phi_radians"]),
                int(component["component_index"]),
            ),
        )
        selected_lower = float(selected_component["lower_phi_radians"])
        selected_upper = float(selected_component["upper_phi_radians"])
        selected_phi = 0.5 * (selected_lower + selected_upper)
        Chebyshev_radius = 0.5 * (selected_upper - selected_lower)
        lambda_lower = 0.5
        lambda_upper = 0.5
        lower_primal_slack = selected_phi - Chebyshev_radius - selected_lower
        upper_primal_slack = selected_upper - selected_phi - Chebyshev_radius
        stationarity_phi_residual = lambda_lower - lambda_upper
        stationarity_radius_residual = 1.0 - lambda_lower - lambda_upper
        lower_complementarity_residual = lambda_lower * lower_primal_slack
        upper_complementarity_residual = lambda_upper * upper_primal_slack
        dual_radius_bound = (
            lambda_upper * selected_upper - lambda_lower * selected_lower
        )
        primal_dual_gap = dual_radius_bound - Chebyshev_radius
        Chebyshev_checks = {
            "positive_radius": Chebyshev_radius > 0.0,
            "primal_lower_feasible": lower_primal_slack
            >= -CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            "primal_upper_feasible": upper_primal_slack
            >= -CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            "dual_multipliers_nonnegative": lambda_lower >= 0.0
            and lambda_upper >= 0.0,
            "dual_sum_one": abs(lambda_lower + lambda_upper - 1.0)
            <= CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            "stationarity_phi": abs(stationarity_phi_residual)
            <= CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            "stationarity_radius": abs(stationarity_radius_residual)
            <= CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            "lower_complementarity": abs(lower_complementarity_residual)
            <= CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            "upper_complementarity": abs(upper_complementarity_residual)
            <= CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            "zero_primal_dual_gap": abs(primal_dual_gap)
            <= CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            "selected_width_is_global_max": all(
                selected_upper - selected_lower
                >= float(component["angular_width_radians"])
                for component in feasible_components
            ),
            "global_width_ordering_complete": len(global_width_ordering)
            == len(feasible_components),
            "global_width_ordering_nonincreasing": all(
                float(left["angular_width_radians"])
                >= float(right["angular_width_radians"])
                for left, right in zip(
                    global_width_ordering, global_width_ordering[1:]
                )
            ),
            "exact_width_tie_uses_smallest_lower_phi": all(
                selected_lower <= float(component["lower_phi_radians"])
                for component in widest_ties
            ),
            "nonexact_near_tie_count_zero": not ambiguous_near_ties,
        }
        if not all(Chebyshev_checks.values()):
            raise ProtocolError(
                f"1D Chebyshev-center certificate failed: {Chebyshev_checks}"
            )
        Chebyshev_certificate = {
            "problem": "maximize r subject to r<=phi-L and r<=U-phi",
            "lower_phi_radians": selected_lower,
            "upper_phi_radians": selected_upper,
            "center_phi_radians": selected_phi,
            "primal_radius_radians": Chebyshev_radius,
            "lambda_lower": lambda_lower,
            "lambda_upper": lambda_upper,
            "lower_primal_slack_radians": lower_primal_slack,
            "upper_primal_slack_radians": upper_primal_slack,
            "stationarity_phi_residual": stationarity_phi_residual,
            "stationarity_radius_residual": stationarity_radius_residual,
            "lower_complementarity_residual": lower_complementarity_residual,
            "upper_complementarity_residual": upper_complementarity_residual,
            "dual_radius_bound_radians": dual_radius_bound,
            "primal_dual_gap_radians": primal_dual_gap,
            "fixed_certificate_tolerance_radians": CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            "global_width_ordering": global_width_ordering,
            "checks": Chebyshev_checks,
        }
        selected_t = selected_phi / theta
        selected_unit = np.asarray(
            math.cos(selected_phi) * unit_v1 + math.sin(selected_phi) * e2,
            dtype=np.float64,
        )
        selected_vector = np.asarray(R_SAFE * selected_unit, dtype=np.float64)
        selected_eval = evaluate_constraints_at_phi(constraints, selected_phi)
        if not selected_eval["pass"]:
            raise ProtocolError("widest-component midpoint failed its analytic constraints")
        selected_normalized_change = matrix @ selected_vector
        selected_normalized_gate_slack = selected_normalized_change - rhs
        selected_constraint_slacks = {}
        constraint_by_name = {item["name"]: item for item in constraints}
        for name, constraint in constraint_by_name.items():
            row = int(constraint["row"])
            detail = detail_by_name[name]
            gradient_l2 = float(detail["gradient_l2"])
            raw_change = float(selected_normalized_change[row] * gradient_l2)
            current = float(detail["current"])
            target = float(detail["target"])
            if constraint["role"] in ("PF_A", "PF7"):
                raw_criterion_slack = raw_change - PF_STRICT_RAW_CHANGE_BUFFER
            elif constraint["role"] == "PF_B":
                raw_criterion_slack = raw_change
            else:
                raw_criterion_slack = (current + raw_change) - target
            selected_constraint_slacks[name] = {
                "role": constraint["role"],
                "row": row,
                "relation": constraint["relation"],
                "sinusoid_criterion_slack_normalized": float(
                    selected_eval["constraint_values"][name]
                ),
                "normalized_linear_change": float(selected_normalized_change[row]),
                "normalized_change_threshold": constraint[
                    "normalized_change_threshold"
                ],
                "normalized_frozen_gate_slack": float(
                    selected_normalized_gate_slack[row]
                ),
                "raw_native_margin_change": raw_change,
                "raw_change_buffer": constraint["raw_change_buffer"],
                "raw_native_margin_criterion_slack": raw_criterion_slack,
                "current_native_margin": current,
                "predicted_linear_native_margin": current + raw_change,
                "frozen_target_native_margin": target,
                "pass": bool(selected_eval["constraint_passes"][name]),
            }
        all_baseline_row_slacks = {}
        for row, detail in enumerate(details):
            name = str(detail["name"])
            gradient_l2 = float(detail["gradient_l2"])
            raw_change = float(selected_normalized_change[row] * gradient_l2)
            current = float(detail["current"])
            target = float(detail["target"])
            all_baseline_row_slacks[name] = {
                "row": row,
                "normalized_linear_change": float(selected_normalized_change[row]),
                "normalized_rhs": float(rhs[row]),
                "normalized_frozen_gate_slack": float(
                    selected_normalized_gate_slack[row]
                ),
                "gradient_l2": gradient_l2,
                "raw_native_margin_change": raw_change,
                "current_native_margin": current,
                "predicted_linear_native_margin": current + raw_change,
                "frozen_target_native_margin": target,
                "raw_frozen_gate_slack": (current + raw_change) - target,
            }
        selection = {
            "rule": "midpoint of widest feasible connected component; width ties choose smallest phi",
            "width_tie_comparison": "exact float64 angular width equality",
            "nonexact_near_tie_policy": "fail closed within fixed width ambiguity tolerance",
            "width_ambiguity_tolerance_radians": WIDTH_AMBIGUITY_TOLERANCE_RADIANS,
            "widest_tie_count": len(widest_ties),
            "component_index": int(selected_component["component_index"]),
            "phi_radians": selected_phi,
            "t": selected_t,
            "radius": R_SAFE,
            "unit_float64_le_sha256": array_sha(selected_unit),
            "vector_float64_le_sha256": array_sha(selected_vector),
            "vector_l2": float(np.linalg.norm(selected_vector)),
            "vector_shape": list(selected_vector.shape),
            "Chebyshev_center_certificate": Chebyshev_certificate,
            "constraint_midpoint_evaluation": selected_eval,
            "checks": {
                "phi_in_selected_component": float(
                    selected_component["lower_phi_radians"]
                )
                <= selected_phi
                <= float(selected_component["upper_phi_radians"]),
                "t_in_closed_unit_interval": 0.0 <= selected_t <= 1.0,
                "unit_norm": math.isclose(
                    float(np.linalg.norm(selected_unit)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=5e-15,
                ),
                "fixed_radius": math.isclose(
                    float(np.linalg.norm(selected_vector)),
                    R_SAFE,
                    rel_tol=0.0,
                    abs_tol=5e-15,
                ),
                "all_12_constraints_pass": bool(selected_eval["pass"]),
            },
        }
        if not all(selection["checks"].values()):
            raise ProtocolError(f"selected analytic vector failed checks: {selection['checks']}")

    preflight_pass = selection is not None
    decision = (
        "GO_CW24_V10A_MINOR_ARC_ANALYTIC_PREFLIGHT"
        if preflight_pass
        else "NO_GO_CW24_V10A_MINOR_ARC_ANALYTIC_PREFLIGHT"
    )
    result = {
        "schema_version": SCHEMA,
        "status": decision,
        "decision": decision,
        "seed": SEED,
        "purpose": "train-only analytic minor-arc feasibility over frozen B352; not promotion evidence",
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
        "minor_arc": {
            "radius": R_SAFE,
            "parameterization": "u(phi)=cos(phi)*u1+sin(phi)*e2; delta(phi)=R_SAFE*u(phi)",
            "phi_domain_radians": [0.0, theta],
            "t_definition": "t=phi/theta",
            "cosine": cosine,
            "theta_radians": theta,
            "unit_v1_float64_le_sha256": array_sha(unit_v1),
            "unit_v6_float64_le_sha256": array_sha(unit_v6),
            "e2_float64_le_sha256": array_sha(e2),
            "reconstructed_v6_float64_le_sha256": array_sha(reconstructed_v6),
            "checks": geometry_checks,
        },
        "analytic_feasibility": {
            "coordinate": "absolute actor delta from exact CW11",
            "method": "closed-form sinusoid root enumeration and sign-constant cells",
            "search_performed": False,
            "grid_points_evaluated": 0,
            "random_draws": 0,
            "radius_candidates": [R_SAFE],
            "weight_candidates": [],
            "fixed_constants": {
                "PF_A_and_PF7_raw_native_margin_change_buffer": PF_STRICT_RAW_CHANGE_BUFFER,
                "root_enumeration_tolerance_radians": ROOT_ENUM_TOLERANCE_RADIANS,
                "interval_merge_tolerance_radians": INTERVAL_MERGE_TOLERANCE_RADIANS,
                "root_residual_tolerance_normalized": ROOT_RESIDUAL_TOLERANCE_NORMALIZED,
                "width_ambiguity_tolerance_radians": WIDTH_AMBIGUITY_TOLERANCE_RADIANS,
                "Chebyshev_certificate_tolerance_radians": CHEBYSHEV_CERT_TOLERANCE_RADIANS,
            },
            "detail_checks": detail_checks,
            "constraints": constraints,
            "constraints_canonical_sha256": hashlib.sha256(
                canonical_json(constraints)
            ).hexdigest(),
            "root_analyses": root_analyses,
            "merged_boundaries": boundaries,
            "boundary_point_cells": point_cells,
            "open_interval_cells": open_cells,
            "feasible_connected_components": feasible_components,
            "global_component_width_ordering": global_width_ordering,
            "feasible_component_count": len(feasible_components),
            "positive_width_feasible_component_count": len(
                positive_width_components
            ),
            "zero_width_only_is_NO_GO": True,
            "selected_component": selected_component,
            "selection": selection,
            "selected_constraint_slacks": selected_constraint_slacks,
            "all_B352_row_slacks": all_baseline_row_slacks,
            "selected_vector_forward_performed": False,
            "pass": preflight_pass,
        },
        "audit": {
            "runtime_checks": runtime_checks,
            "source": source,
            "train_only_B352": True,
            "new_validation_or_test_rows_opened": 0,
            "selected_vector_model_copy_calls": 0,
            "selected_vector_model_forward_calls": 0,
            "analytic_root_enumeration_only": True,
            "grid_or_random_or_radius_or_weight_scan_performed": False,
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
            "PF_strict_raw_buffer_exact": PF_STRICT_RAW_CHANGE_BUFFER == 1.0e-6,
            "root_enum_tolerance_exact": ROOT_ENUM_TOLERANCE_RADIANS == 5.0e-14,
            "interval_merge_tolerance_exact": INTERVAL_MERGE_TOLERANCE_RADIANS
            == 5.0e-14,
            "root_residual_tolerance_exact": ROOT_RESIDUAL_TOLERANCE_NORMALIZED
            == 5.0e-15,
            "width_ambiguity_tolerance_exact": WIDTH_AMBIGUITY_TOLERANCE_RADIANS
            == 5.0e-14,
            "Chebyshev_cert_tolerance_exact": CHEBYSHEV_CERT_TOLERANCE_RADIANS
            == 5.0e-15,
            "output_absent": not OUTPUT.exists(),
        }
        if not all(checks.values()):
            raise ProtocolError(f"v10a static audit failed: {checks}")
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
                        "parameterization": "u(phi)=cos(phi)*u1+sin(phi)*e2",
                        "PF_A_and_PF7_raw_change_buffer": PF_STRICT_RAW_CHANGE_BUFFER,
                        "root_enumeration_tolerance_radians": ROOT_ENUM_TOLERANCE_RADIANS,
                        "interval_merge_tolerance_radians": INTERVAL_MERGE_TOLERANCE_RADIANS,
                        "root_residual_tolerance_normalized": ROOT_RESIDUAL_TOLERANCE_NORMALIZED,
                        "width_ambiguity_tolerance_radians": WIDTH_AMBIGUITY_TOLERANCE_RADIANS,
                        "Chebyshev_certificate_tolerance_radians": CHEBYSHEV_CERT_TOLERANCE_RADIANS,
                        "protected_pair_names": list(PROTECTED_PAIR_NAMES),
                        "grid_or_random_or_radius_or_weight_scan_performed": False,
                        "selected_vector_forward_performed": False,
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
        raise ProtocolError("frozen CW24 v10a output already exists")
    result = production_run()
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "selected_phi_radians": (
                    result["analytic_feasibility"]["selection"]["phi_radians"]
                    if result["analytic_feasibility"]["selection"] is not None
                    else None
                ),
                "selected_t": (
                    result["analytic_feasibility"]["selection"]["t"]
                    if result["analytic_feasibility"]["selection"] is not None
                    else None
                ),
                "selected_vector_float64_le_sha256": (
                    result["analytic_feasibility"]["selection"][
                        "vector_float64_le_sha256"
                    ]
                    if result["analytic_feasibility"]["selection"] is not None
                    else None
                ),
                "analytic_preflight_pass": result["analytic_feasibility"]["pass"],
                "selected_vector_forward_performed": False,
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

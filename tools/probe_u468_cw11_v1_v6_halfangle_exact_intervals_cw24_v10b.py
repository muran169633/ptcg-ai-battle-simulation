#!/usr/bin/env python3
"""CW24 v10b exact-rational half-angle root-free-cell preflight.

This train-only diagnostic replays two already frozen failed recipes solely to
capture their actual float32 actor deltas in two isolated child processes: the
v1 terminal failure and the rejected second v6 shadow.  In the parent it forms
the fixed-radius minor arc and substitutes ``t=tan(phi/2)``.  Every float64
sinusoid coefficient is converted with ``as_integer_ratio`` to an exact SymPy
Rational polynomial.  Certified rational root enclosures are conservatively
unioned into unselectable barriers; only exact-sign, root-free complement cells
can be selected.  The rule chooses the widest certified feasible cell in t,
never claims the widest true connected component or widest angle, and performs
no grid, random, radius, or weight scan.  The selected vector is never copied
to a model and never receives a forward pass.  No new candidate shadow,
payload, checkpoint, model artifact, or official candidate is created.
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
SCRIPT = TOOLS / "probe_u468_cw11_v1_v6_halfangle_exact_intervals_cw24_v10b.py"
OUTPUT = ROOT / "artifacts/cw24_cw11_v1_v6_halfangle_exact_intervals_cw24_v10b.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-v1-v6-halfangle-exact-intervals-cw24-v10b"
SEED = 202608049
R_SAFE = 0.000999975
V6_FULL_RADIUS = 0.00099998
PF_STRICT_RAW_CHANGE_BUFFER = 1.0e-6
ROOT_ISOLATION_EPS_DENOMINATOR = 10**40
ENDPOINT_REFINE_EPS_DENOMINATOR = 10**60
MIN_CERTIFIED_INNER_T_WIDTH_NUMERATOR = 4
MIN_CERTIFIED_INNER_T_WIDTH_DENOMINATOR = 10**40
SCALAR_DIRECT_ABS_TOLERANCE = 5.0e-15
SYMPY_VERSION = "1.13.3"
SYMPY_INIT = Path(
    "/home/xxc/miniconda3/envs/my_project_env/lib/python3.11/site-packages/sympy/__init__.py"
)
SYMPY_INIT_SHA256 = "af7d1716a5ccdf21e077be548befd986e9fcf0a15b7914c0afef3c104bec2b96"
SYMPY_INIT_MODE = 0o664

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

V10A_SOURCE = TOOLS / "probe_u468_cw11_v1_v6_minor_arc_feasible_intervals_cw24_v10a.py"
V10A_SOURCE_SHA256 = "69556abdcba12e2b5b1adf4936dc0d04fd4e126548a25e467e2c48c4ca17aab8"
V10A_SOURCE_MODE = 0o664

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
    """Fail-closed CW24 v10b preflight error."""


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
    try:
        reported_path = str(path.relative_to(ROOT))
    except ValueError:
        reported_path = str(path)
    return source, {
        "path": reported_path,
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
                V10A_SOURCE_SHA256,
                SYMPY_INIT_SHA256,
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
        "root_isolation_eps_single_constant": source.count(
            b"\nROOT_ISOLATION_EPS_DENOMINATOR ="
        ) == 1
        and ROOT_ISOLATION_EPS_DENOMINATOR == 10**40,
        "endpoint_refine_eps_single_constant": source.count(
            b"\nENDPOINT_REFINE_EPS_DENOMINATOR ="
        ) == 1
        and ENDPOINT_REFINE_EPS_DENOMINATOR == 10**60,
        "minimum_inner_width_constants_exact": source.count(
            b"\nMIN_CERTIFIED_INNER_T_WIDTH_NUMERATOR ="
        ) == 1
        and source.count(b"\nMIN_CERTIFIED_INNER_T_WIDTH_DENOMINATOR =") == 1
        and MIN_CERTIFIED_INNER_T_WIDTH_NUMERATOR == 4
        and MIN_CERTIFIED_INNER_T_WIDTH_DENOMINATOR == 10**40,
        "scalar_direct_tolerance_single_constant": source.count(
            b"\nSCALAR_DIRECT_ABS_TOLERANCE ="
        ) == 1
        and SCALAR_DIRECT_ABS_TOLERANCE == 5.0e-15,
        "sympy_version_lock_literal": SYMPY_VERSION == "1.13.3"
        and source.count(b'\nSYMPY_VERSION = "1.13.3"') == 1,
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
        raise ProtocolError(f"v10b source audit failed: {checks}")
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
                raise ProtocolError("short v10b result write")
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
        raise ProtocolError(f"v10b result publication drift: {checks}")
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
            V10B_CAPTURE_V1_FAILED_ACTUAL_DELTA(actual_delta.copy(), trial)
            return {
'''
    audits: list[dict[str, Any]] = []
    source = splice_once(source, old, new, "capture_frozen_v1_failed_actual_delta", audits)
    transformed = source.encode()
    name = "cw24_v10b_transformed_frozen_v1"
    if name in sys.modules:
        raise ProtocolError("v10b transformed v1 module name occupied")
    module = types.ModuleType(name)
    module.__file__ = str(V1_SOURCE)
    module.__package__ = ""
    module.__dict__["V10B_CAPTURE_V1_FAILED_ACTUAL_DELTA"] = capture
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
        "callback_identity_exact": module.V10B_CAPTURE_V1_FAILED_ACTUAL_DELTA is capture,
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
                V10B_CAPTURE_V4_BASELINE_LINEARIZATION(
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
                V10B_CAPTURE_V6_REJECTED_S2_ACTUAL_DELTA(
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
    name = "cw24_v10b_transformed_frozen_v4_v6"
    if name in sys.modules:
        raise ProtocolError("v10b transformed v4/v6 module name occupied")
    module = types.ModuleType(name)
    module.__file__ = str(V4_SOURCE)
    module.__package__ = ""
    module.__dict__["V10B_CAPTURE_V4_BASELINE_LINEARIZATION"] = capture_baseline
    module.__dict__["V10B_CAPTURE_V6_REJECTED_S2_ACTUAL_DELTA"] = capture_s2
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
        "callbacks_identity_exact": module.V10B_CAPTURE_V4_BASELINE_LINEARIZATION
        is capture_baseline
        and module.V10B_CAPTURE_V6_REJECTED_S2_ACTUAL_DELTA is capture_s2,
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
        ("v10a_source", V10A_SOURCE, V10A_SOURCE_SHA256, V10A_SOURCE_MODE, "v10a source"),
        ("sympy_init", SYMPY_INIT, SYMPY_INIT_SHA256, SYMPY_INIT_MODE, "SymPy init"),
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


def rational_from_float(value: float, Rational: Any) -> Any:
    value = float(value)
    if not math.isfinite(value):
        raise ProtocolError("nonfinite float64 cannot be rationalized")
    numerator, denominator = value.as_integer_ratio()
    return Rational(numerator, denominator)


def rational_payload(value: Any) -> dict[str, Any]:
    return {
        "numerator": str(int(value.p)),
        "denominator": str(int(value.q)),
        "text": str(value),
    }


def exact_sign(value: Any) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def isolate_exact_polynomial_roots(
    bundle: Mapping[str, Any],
    zero: Any,
    endpoint_T: Any,
    isolation_eps: Any,
    endpoint_refine_eps: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    poly = bundle["poly"]
    if poly.is_zero:
        return [], {
            "constraint_name": str(bundle["name"]),
            "polynomial_zero": True,
            "degree": str(poly.degree()),
            "real_root_interval_count": 0,
            "real_root_multiplicity_sum": 0,
            "domain_root_interval_count": 0,
            "exact_domain_distinct_root_count": None,
            "exact_domain_root_count_matches_intervals": "not_applicable_zero_polynomial",
            "endpoint_refinement_count": 0,
        }

    raw_intervals = list(poly.intervals(eps=isolation_eps, sqf=False))
    records: list[dict[str, Any]] = []
    endpoint_refinement_count = 0
    for interval_index, (interval, multiplicity) in enumerate(raw_intervals):
        lower, upper = interval
        initial_lower, initial_upper = lower, upper
        refined_for_endpoint = False
        for endpoint_name, endpoint in (("zero", zero), ("T", endpoint_T)):
            if lower <= endpoint <= upper:
                endpoint_value = poly.eval(endpoint)
                if endpoint_value == 0:
                    lower = endpoint
                    upper = endpoint
                elif lower < upper:
                    lower, upper = poly.sqf_part().refine_root(
                        lower,
                        upper,
                        eps=endpoint_refine_eps,
                        check_sqf=True,
                    )
                    endpoint_refinement_count += 1
                    refined_for_endpoint = True
                    if lower <= endpoint <= upper:
                        raise ProtocolError(
                            f"{bundle['name']} root enclosure remains ambiguous at {endpoint_name}"
                        )
                else:
                    raise ProtocolError(
                        f"{bundle['name']} exact root interval touches non-root {endpoint_name}"
                    )
        width_limit = endpoint_refine_eps if refined_for_endpoint else isolation_eps
        if upper - lower > width_limit:
            raise ProtocolError(
                f"{bundle['name']} root enclosure exceeds its exact rational eps"
            )
        if upper < zero or lower > endpoint_T:
            domain_classification = "outside_closed_domain"
        elif zero <= lower and upper <= endpoint_T:
            domain_classification = "inside_closed_domain"
        else:
            raise ProtocolError(
                f"{bundle['name']} root enclosure is not separated from [0,T]"
            )
        records.append(
            {
                "constraint_name": str(bundle["name"]),
                "relation": str(bundle["relation"]),
                "interval_index": interval_index,
                "multiplicity": int(multiplicity),
                "lower": lower,
                "upper": upper,
                "initial_lower": initial_lower,
                "initial_upper": initial_upper,
                "refined_for_endpoint": refined_for_endpoint,
                "domain_classification": domain_classification,
            }
        )
    exact_domain_distinct_root_count = int(poly.count_roots(zero, endpoint_T))
    observed_domain_interval_count = sum(
        record["domain_classification"] == "inside_closed_domain"
        for record in records
    )
    if exact_domain_distinct_root_count != observed_domain_interval_count:
        raise ProtocolError(
            f"{bundle['name']} exact domain root count disagrees with enclosures"
        )
    audit = {
        "constraint_name": str(bundle["name"]),
        "polynomial_zero": False,
        "degree": int(poly.degree()),
        "real_root_interval_count": len(raw_intervals),
        "real_root_multiplicity_sum": sum(
            int(multiplicity) for _, multiplicity in raw_intervals
        ),
        "domain_root_interval_count": observed_domain_interval_count,
        "exact_domain_distinct_root_count": exact_domain_distinct_root_count,
        "exact_domain_root_count_matches_intervals": True,
        "endpoint_refinement_count": endpoint_refinement_count,
    }
    return records, audit


def conservative_root_barriers(
    root_records: list[dict[str, Any]], zero: Any, endpoint_T: Any
) -> list[dict[str, Any]]:
    in_domain = [
        record
        for record in root_records
        if record["domain_classification"] == "inside_closed_domain"
    ]
    in_domain.sort(
        key=lambda record: (
            record["lower"],
            record["upper"],
            str(record["constraint_name"]),
            int(record["interval_index"]),
        )
    )
    barriers: list[dict[str, Any]] = []
    for record in in_domain:
        lower = max(zero, record["lower"])
        upper = min(endpoint_T, record["upper"])
        member = {
            "root_record_id": int(record["root_record_id"]),
            "constraint_name": str(record["constraint_name"]),
            "interval_index": int(record["interval_index"]),
            "multiplicity": int(record["multiplicity"]),
            "lower": record["lower"],
            "upper": record["upper"],
        }
        if not barriers or lower > barriers[-1]["upper"]:
            barriers.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "members": [member],
                }
            )
        else:
            barriers[-1]["upper"] = max(barriers[-1]["upper"], upper)
            barriers[-1]["members"].append(member)
    for barrier_index, barrier in enumerate(barriers):
        barrier["barrier_index"] = barrier_index
    return barriers


def certified_root_free_cells(
    barriers: list[dict[str, Any]], zero: Any, endpoint_T: Any
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    cursor = zero
    lower_source = "domain_zero"
    for barrier in barriers:
        if cursor < barrier["lower"]:
            cells.append(
                {
                    "lower": cursor,
                    "upper": barrier["lower"],
                    "width": barrier["lower"] - cursor,
                    "lower_source": lower_source,
                    "upper_source": f"barrier_{barrier['barrier_index']}_lower",
                }
            )
        if barrier["upper"] > cursor:
            cursor = barrier["upper"]
            lower_source = f"barrier_{barrier['barrier_index']}_upper"
    if cursor < endpoint_T:
        cells.append(
            {
                "lower": cursor,
                "upper": endpoint_T,
                "width": endpoint_T - cursor,
                "lower_source": lower_source,
                "upper_source": "domain_T",
            }
        )
    for cell_index, cell in enumerate(cells):
        if not cell["lower"] < cell["upper"]:
            raise ProtocolError("certified root-free cell is not strictly positive width")
        cell["cell_index"] = cell_index
        cell["witness"] = (cell["lower"] + cell["upper"]) / 2
        if not cell["lower"] < cell["witness"] < cell["upper"]:
            raise ProtocolError("exact rational cell witness is not strictly interior")
    return cells


def exact_cell_signs(
    cell: Mapping[str, Any], bundles: list[dict[str, Any]]
) -> dict[str, Any]:
    witness = cell["witness"]
    values: dict[str, Any] = {}
    signs: dict[str, int] = {}
    passes: dict[str, bool] = {}
    for bundle in bundles:
        name = str(bundle["name"])
        value = bundle["poly"].eval(witness)
        sign = exact_sign(value)
        if bundle["relation"] == "strict_positive":
            passed = sign > 0
        elif bundle["relation"] == "nonnegative":
            passed = sign >= 0
        else:
            raise ProtocolError(f"unknown exact relation for {name}")
        values[name] = value
        signs[name] = sign
        passes[name] = passed
    return {
        "cell_index": int(cell["cell_index"]),
        "exact_values": values,
        "exact_signs": signs,
        "passes": passes,
        "failing_names": sorted(name for name, passed in passes.items() if not passed),
        "pass": all(passes.values()),
    }


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
        f"ns=runpy.run_path({str(SCRIPT)!r},run_name='cw24_v10b_isolated_{kind}')\n"
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
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError("v10b production requires exact locked my_project_env Python")
    import numpy as np
    import sympy

    source = self_evidence(require_frozen=True)
    inputs = frozen_inputs()
    sympy_checks = {
        "version_exact": sympy.__version__ == SYMPY_VERSION,
        "init_path_exact": Path(sympy.__file__).resolve() == SYMPY_INIT.resolve(),
        "init_sha_exact": sha256_file(SYMPY_INIT) == SYMPY_INIT_SHA256,
        "parent_torch_absent_after_sympy_import": "torch" not in sys.modules,
    }
    if not all(sympy_checks.values()):
        raise ProtocolError(f"SymPy runtime drift: {sympy_checks}")
    runtime_checks = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "R_safe_exact": R_SAFE == 0.000999975,
        "sympy_exact": all(sympy_checks.values()),
        "parent_torch_absent_before_children": "torch" not in sys.modules,
    }
    if not all(runtime_checks.values()):
        raise ProtocolError(f"v10b runtime drift: {runtime_checks}")

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
    endpoint_T_float64 = float(math.tan(0.5 * theta))
    Rational = sympy.Rational
    endpoint_T = rational_from_float(endpoint_T_float64, Rational)
    zero = Rational(0)
    isolation_eps = Rational(1, ROOT_ISOLATION_EPS_DENOMINATOR)
    endpoint_refine_eps = Rational(1, ENDPOINT_REFINE_EPS_DENOMINATOR)
    minimum_inner_width = Rational(
        MIN_CERTIFIED_INNER_T_WIDTH_NUMERATOR,
        MIN_CERTIFIED_INNER_T_WIDTH_DENOMINATOR,
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
        "endpoint_T_float64_finite_positive": math.isfinite(endpoint_T_float64)
        and endpoint_T_float64 > 0.0,
        "endpoint_T_exact_rational_positive": bool(endpoint_T > zero),
        "safe_below_v4_v6_cap": R_SAFE < V6_FULL_RADIUS,
    }
    if not all(geometry_checks.values()):
        raise ProtocolError(f"half-angle minor-arc geometry failed: {geometry_checks}")

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
        "all_required_gradient_norms_positive": required_names.issubset(row_by_name)
        and all(
            float(detail_by_name[name]["gradient_l2"]) > 0.0
            for name in required_names
        ),
    }
    if not all(detail_checks.values()):
        raise ProtocolError(f"frozen B352 row metadata drift: {detail_checks}")

    float_constraints: list[dict[str, Any]] = []
    pf_definitions = (
        ("pf0_boundary_a", "PF_A", "strict_positive", PF_STRICT_RAW_CHANGE_BUFFER),
        ("pf0_boundary_b", "PF_B", "nonnegative", 0.0),
        ("pf7_boundary", "PF7", "strict_positive", PF_STRICT_RAW_CHANGE_BUFFER),
    )
    for name, role, relation, raw_buffer in pf_definitions:
        row = row_by_name[name]
        gradient_l2 = float(detail_by_name[name]["gradient_l2"])
        normalized_threshold = float(raw_buffer / gradient_l2)
        float_constraints.append(
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
        float_constraints.append(
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
    if (
        len(float_constraints) != 12
        or len({item["name"] for item in float_constraints}) != 12
    ):
        raise ProtocolError("exact constraint roster is not exactly 3 PF plus 9 protected")

    t_symbol = sympy.Symbol("t", real=True)
    exact_bundles: list[dict[str, Any]] = []
    exact_constraint_output: list[dict[str, Any]] = []
    for constraint in float_constraints:
        a_exact = rational_from_float(constraint["a_cos"], Rational)
        b_exact = rational_from_float(constraint["b_sin"], Rational)
        c_exact = rational_from_float(constraint["c_offset"], Rational)
        q2 = c_exact - a_exact
        q1 = 2 * b_exact
        q0 = a_exact + c_exact
        poly = sympy.Poly(q2 * t_symbol**2 + q1 * t_symbol + q0, t_symbol, domain="QQ")
        bundle = {
            **constraint,
            "a_exact": a_exact,
            "b_exact": b_exact,
            "c_exact": c_exact,
            "q2": q2,
            "q1": q1,
            "q0": q0,
            "poly": poly,
        }
        exact_bundles.append(bundle)
        exact_constraint_output.append(
            {
                **constraint,
                "float64_as_integer_ratio": {
                    "a_cos": rational_payload(a_exact),
                    "b_sin": rational_payload(b_exact),
                    "c_offset": rational_payload(c_exact),
                },
                "halfangle_polynomial": {
                    "formula": "Q(t)=(c-a)t^2+2bt+(a+c)",
                    "q2": rational_payload(q2),
                    "q1": rational_payload(q1),
                    "q0": rational_payload(q0),
                    "degree": str(poly.degree()),
                    "is_zero": bool(poly.is_zero),
                    "canonical_srepr_sha256": hashlib.sha256(
                        sympy.srepr(poly.as_expr()).encode()
                    ).hexdigest(),
                },
            }
        )

    root_records: list[dict[str, Any]] = []
    root_audits: list[dict[str, Any]] = []
    for bundle in exact_bundles:
        records, root_audit = isolate_exact_polynomial_roots(
            bundle,
            zero,
            endpoint_T,
            isolation_eps,
            endpoint_refine_eps,
        )
        root_records.extend(records)
        root_audits.append(root_audit)
    for root_record_id, record in enumerate(root_records):
        record["root_record_id"] = root_record_id

    barriers = conservative_root_barriers(root_records, zero, endpoint_T)
    root_free_cells = certified_root_free_cells(barriers, zero, endpoint_T)
    exact_cell_evaluations = [
        exact_cell_signs(cell, exact_bundles) for cell in root_free_cells
    ]
    exact_eval_by_index = {
        int(item["cell_index"]): item for item in exact_cell_evaluations
    }
    domain_root_ids = {
        int(record["root_record_id"])
        for record in root_records
        if record["domain_classification"] == "inside_closed_domain"
    }
    barrier_member_ids = {
        int(member["root_record_id"])
        for barrier in barriers
        for member in barrier["members"]
    }
    exact_analysis_checks = {
        "all_root_enclosures_within_fixed_eps": all(
            record["upper"] - record["lower"]
            <= (
                endpoint_refine_eps
                if record["refined_for_endpoint"]
                else isolation_eps
            )
            for record in root_records
        ),
        "all_domain_roots_covered_by_barriers_exactly_once": domain_root_ids
        == barrier_member_ids
        and sum(len(barrier["members"]) for barrier in barriers)
        == len(domain_root_ids),
        "barriers_strictly_ordered_and_disjoint": all(
            left["upper"] < right["lower"]
            for left, right in zip(barriers, barriers[1:])
        ),
        "barriers_inside_closed_domain": all(
            zero <= barrier["lower"] <= barrier["upper"] <= endpoint_T
            for barrier in barriers
        ),
        "root_free_cells_strictly_positive": all(
            cell["lower"] < cell["witness"] < cell["upper"]
            and cell["width"] == cell["upper"] - cell["lower"]
            for cell in root_free_cells
        ),
        "root_free_cells_do_not_enter_any_barrier": all(
            cell["upper"] <= barrier["lower"]
            or barrier["upper"] <= cell["lower"]
            for cell in root_free_cells
            for barrier in barriers
        ),
        "all_cell_evaluations_cover_exactly_12_constraints": all(
            len(item["exact_values"])
            == len(item["exact_signs"])
            == len(item["passes"])
            == 12
            for item in exact_cell_evaluations
        ),
    }
    if not all(exact_analysis_checks.values()):
        raise ProtocolError(
            f"exact root-free-cell construction failed: {exact_analysis_checks}"
        )
    feasible_cells = [
        cell
        for cell in root_free_cells
        if exact_eval_by_index[int(cell["cell_index"])]["pass"]
    ]
    selectable_cells = [
        cell for cell in feasible_cells if cell["width"] > minimum_inner_width
    ]

    selected_cell: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    exact_float_direct_gates: dict[str, Any] | None = None
    selected_constraint_slacks: dict[str, Any] | None = None
    all_baseline_row_slacks: dict[str, Any] | None = None
    if selectable_cells:
        widest_exact_width = max(cell["width"] for cell in selectable_cells)
        exact_width_ties = [
            cell for cell in selectable_cells if cell["width"] == widest_exact_width
        ]
        selected_cell = min(
            exact_width_ties,
            key=lambda cell: (cell["lower"], int(cell["cell_index"])),
        )
        selection_rule_checks = {
            "selected_cell_exact_feasible": bool(
                exact_eval_by_index[int(selected_cell["cell_index"])]["pass"]
            ),
            "selected_width_above_fixed_minimum": bool(
                selected_cell["width"] > minimum_inner_width
            ),
            "selected_width_is_exact_global_max_among_selectable_cells": all(
                selected_cell["width"] >= cell["width"]
                for cell in selectable_cells
            ),
            "exact_width_tie_uses_smallest_lower_t": all(
                selected_cell["lower"] <= cell["lower"]
                for cell in exact_width_ties
            ),
            "selected_exact_witness_strictly_in_inner": bool(
                selected_cell["lower"]
                < selected_cell["witness"]
                < selected_cell["upper"]
            ),
        }
        if not all(selection_rule_checks.values()):
            raise ProtocolError(
                f"exact certified-cell selection failed: {selection_rule_checks}"
            )
        selected_t_exact = selected_cell["witness"]
        selected_t_float64 = float(selected_t_exact)
        selected_t_float_exact = rational_from_float(selected_t_float64, Rational)
        selected_t_prev_float64 = math.nextafter(selected_t_float64, -math.inf)
        selected_t_next_float64 = math.nextafter(selected_t_float64, math.inf)
        selected_t_prev_exact = rational_from_float(selected_t_prev_float64, Rational)
        selected_t_next_exact = rational_from_float(selected_t_next_float64, Rational)
        round_to_nearest_lower_midpoint = (
            selected_t_prev_exact + selected_t_float_exact
        ) / 2
        round_to_nearest_upper_midpoint = (
            selected_t_float_exact + selected_t_next_exact
        ) / 2
        float_copy_checks = {
            "selected_t_float64_finite": math.isfinite(selected_t_float64),
            "float_copy_strictly_inside_exact_inner": bool(
                selected_cell["lower"]
                < selected_t_float_exact
                < selected_cell["upper"]
            ),
            "previous_float64_strictly_inside_exact_inner": bool(
                selected_cell["lower"]
                < selected_t_prev_exact
                < selected_t_float_exact
            ),
            "next_float64_strictly_inside_exact_inner": bool(
                selected_t_float_exact
                < selected_t_next_exact
                < selected_cell["upper"]
            ),
            "nextafter_neighbors_distinct": selected_t_prev_float64
            < selected_t_float64
            < selected_t_next_float64,
            "unique_round_to_nearest_even_preimage": bool(
                round_to_nearest_lower_midpoint
                < selected_t_exact
                < round_to_nearest_upper_midpoint
            ),
            "rounding_tie_absent": bool(
                selected_t_exact
                not in (
                    round_to_nearest_lower_midpoint,
                    round_to_nearest_upper_midpoint,
                )
            ),
        }

        selected_phi_float64 = float(2.0 * math.atan(selected_t_float64))
        t_squared = selected_t_float64 * selected_t_float64
        halfangle_denominator = 1.0 + t_squared
        selected_cos_float64 = (1.0 - t_squared) / halfangle_denominator
        selected_sin_float64 = (2.0 * selected_t_float64) / halfangle_denominator
        selected_unit = np.asarray(
            selected_cos_float64 * unit_v1 + selected_sin_float64 * e2,
            dtype=np.float64,
        )
        selected_vector = np.asarray(R_SAFE * selected_unit, dtype=np.float64)
        selected_direct_change = matrix @ selected_vector
        selected_direct_gate_slack = selected_direct_change - rhs
        exact_selected_eval = exact_eval_by_index[int(selected_cell["cell_index"])]
        selected_constraint_slacks = {}
        gate_rows: dict[str, Any] = {}
        all_scalar_direct_differences: list[float] = []
        for bundle in exact_bundles:
            name = str(bundle["name"])
            row = int(bundle["row"])
            detail = detail_by_name[name]
            gradient_l2 = float(detail["gradient_l2"])
            scalar_value = float(
                bundle["a_cos"] * selected_cos_float64
                + bundle["b_sin"] * selected_sin_float64
                + bundle["c_offset"]
            )
            direct_normalized_change = float(selected_direct_change[row])
            if bundle["role"] in ("PF_A", "PF7"):
                direct_value = (
                    direct_normalized_change
                    - float(bundle["normalized_change_threshold"])
                )
            elif bundle["role"] == "PF_B":
                direct_value = direct_normalized_change
            else:
                direct_value = direct_normalized_change - float(rhs[row])
            scalar_direct_difference = scalar_value - direct_value
            all_scalar_direct_differences.append(abs(scalar_direct_difference))
            exact_value = exact_selected_eval["exact_values"][name]
            exact_value_sign = exact_sign(exact_value)
            exact_float_copy_value = bundle["poly"].eval(selected_t_float_exact)
            exact_float_copy_sign = exact_sign(exact_float_copy_value)
            exact_float_copy_relation_pass = (
                exact_float_copy_sign > 0
                if bundle["relation"] == "strict_positive"
                else exact_float_copy_sign >= 0
            )
            scalar_value_sign = (
                1 if scalar_value > 0.0 else -1 if scalar_value < 0.0 else 0
            )
            direct_value_sign = (
                1 if direct_value > 0.0 else -1 if direct_value < 0.0 else 0
            )
            scalar_relation_pass = (
                scalar_value > 0.0
                if bundle["relation"] == "strict_positive"
                else scalar_value >= 0.0
            )
            direct_relation_pass = (
                direct_value > 0.0
                if bundle["relation"] == "strict_positive"
                else direct_value >= 0.0
            )
            raw_change = direct_normalized_change * gradient_l2
            if bundle["role"] in ("PF_A", "PF7"):
                direct_native_gate_pass = raw_change > PF_STRICT_RAW_CHANGE_BUFFER
                raw_criterion_slack = raw_change - PF_STRICT_RAW_CHANGE_BUFFER
            elif bundle["role"] == "PF_B":
                direct_native_gate_pass = raw_change >= 0.0
                raw_criterion_slack = raw_change
            else:
                direct_native_gate_pass = direct_value >= 0.0
                raw_criterion_slack = (
                    float(detail["current"]) + raw_change - float(detail["target"])
                )
            row_checks = {
                "exact_polynomial_relation_pass": bool(
                    exact_selected_eval["passes"][name]
                ),
                "exact_polynomial_float_copy_relation_pass": bool(
                    exact_float_copy_relation_pass
                ),
                "exact_witness_and_float_copy_sign_identical": exact_value_sign
                == exact_float_copy_sign,
                "float_scalar_relation_pass": bool(scalar_relation_pass),
                "float_direct_relation_pass": bool(direct_relation_pass),
                "float_direct_native_gate_pass": bool(direct_native_gate_pass),
                "finite": all(
                    math.isfinite(value)
                    for value in (
                        scalar_value,
                        direct_value,
                        scalar_direct_difference,
                        direct_normalized_change,
                        raw_change,
                    )
                ),
                "scalar_direct_abs_difference_within_fixed_tolerance": abs(
                    scalar_direct_difference
                )
                <= SCALAR_DIRECT_ABS_TOLERANCE,
                "exact_scalar_direct_signs_identical": exact_value_sign
                == exact_float_copy_sign
                == scalar_value_sign
                == direct_value_sign,
            }
            gate_rows[name] = {
                "role": bundle["role"],
                "row": row,
                "relation": bundle["relation"],
                "exact_Q_at_selected_t": rational_payload(exact_value),
                "exact_Q_at_float_copy_t": rational_payload(
                    exact_float_copy_value
                ),
                "exact_sign": exact_value_sign,
                "exact_float_copy_sign": exact_float_copy_sign,
                "float_scalar_normalized_criterion_slack": scalar_value,
                "float_direct_normalized_criterion_slack": direct_value,
                "scalar_direct_difference": scalar_direct_difference,
                "scalar_sign": scalar_value_sign,
                "direct_sign": direct_value_sign,
                "checks": row_checks,
                "pass": all(row_checks.values()),
            }
            selected_constraint_slacks[name] = {
                "role": bundle["role"],
                "row": row,
                "relation": bundle["relation"],
                "exact_Q_at_selected_t": rational_payload(exact_value),
                "exact_Q_at_float_copy_t": rational_payload(
                    exact_float_copy_value
                ),
                "float_scalar_normalized_criterion_slack": scalar_value,
                "float_direct_normalized_criterion_slack": direct_value,
                "normalized_linear_change": direct_normalized_change,
                "normalized_change_threshold": bundle[
                    "normalized_change_threshold"
                ],
                "normalized_frozen_gate_slack": float(
                    selected_direct_gate_slack[row]
                ),
                "raw_native_margin_change": raw_change,
                "raw_change_buffer": bundle["raw_change_buffer"],
                "raw_native_margin_criterion_slack": raw_criterion_slack,
                "current_native_margin": float(detail["current"]),
                "predicted_linear_native_margin": float(detail["current"])
                + raw_change,
                "frozen_target_native_margin": float(detail["target"]),
                "pass": all(row_checks.values()),
            }

        all_baseline_row_slacks = {}
        for row, detail in enumerate(details):
            name = str(detail["name"])
            gradient_l2 = float(detail["gradient_l2"])
            raw_change = float(selected_direct_change[row] * gradient_l2)
            current = float(detail["current"])
            target = float(detail["target"])
            all_baseline_row_slacks[name] = {
                "row": row,
                "normalized_linear_change": float(selected_direct_change[row]),
                "normalized_rhs": float(rhs[row]),
                "normalized_frozen_gate_slack": float(
                    selected_direct_gate_slack[row]
                ),
                "gradient_l2": gradient_l2,
                "raw_native_margin_change": raw_change,
                "current_native_margin": current,
                "predicted_linear_native_margin": current + raw_change,
                "frozen_target_native_margin": target,
                "raw_frozen_gate_slack": (current + raw_change) - target,
            }

        vector_checks = {
            "float_copy_checks": all(float_copy_checks.values()),
            "selected_phi_finite_in_domain": math.isfinite(selected_phi_float64)
            and 0.0 < selected_phi_float64 < theta,
            "halfangle_cos_sin_finite": math.isfinite(selected_cos_float64)
            and math.isfinite(selected_sin_float64),
            "halfangle_unit_identity": math.isclose(
                selected_cos_float64 * selected_cos_float64
                + selected_sin_float64 * selected_sin_float64,
                1.0,
                rel_tol=0.0,
                abs_tol=5.0e-15,
            ),
            "selected_unit_norm": math.isclose(
                float(np.linalg.norm(selected_unit)),
                1.0,
                rel_tol=0.0,
                abs_tol=5.0e-15,
            ),
            "fixed_radius": math.isclose(
                float(np.linalg.norm(selected_vector)),
                R_SAFE,
                rel_tol=0.0,
                abs_tol=5.0e-15,
            ),
            "all_12_exact_float_direct_rows_pass": all(
                row["pass"] for row in gate_rows.values()
            )
            and len(gate_rows) == 12,
            "maximum_scalar_direct_difference_within_fixed_tolerance": max(
                all_scalar_direct_differences, default=math.inf
            )
            <= SCALAR_DIRECT_ABS_TOLERANCE,
        }
        exact_float_direct_gates = {
            "fixed_scalar_direct_abs_tolerance": SCALAR_DIRECT_ABS_TOLERANCE,
            "float_copy_checks": float_copy_checks,
            "constraint_rows": gate_rows,
            "maximum_scalar_direct_abs_difference": max(
                all_scalar_direct_differences, default=math.inf
            ),
            "vector_checks": vector_checks,
            "pass": all(vector_checks.values()),
        }
        selection = {
            "rule": "widest certified root-free feasible cell by exact Rational inner t-width; exact-width ties choose smallest lower t",
            "does_not_claim_true_component_or_angular_width_optimality": True,
            "cell_index": int(selected_cell["cell_index"]),
            "exact_width_tie_count": len(exact_width_ties),
            "selection_rule_checks": selection_rule_checks,
            "certified_inner": {
                "lower_t": rational_payload(selected_cell["lower"]),
                "upper_t": rational_payload(selected_cell["upper"]),
                "width_t": rational_payload(selected_cell["width"]),
                "minimum_required_width_t": rational_payload(minimum_inner_width),
            },
            "selected_t_exact": rational_payload(selected_t_exact),
            "selected_t_float64": selected_t_float64,
            "selected_t_float64_exact_ratio": rational_payload(
                selected_t_float_exact
            ),
            "selected_t_previous_float64": selected_t_prev_float64,
            "selected_t_next_float64": selected_t_next_float64,
            "selected_t_previous_float64_exact_ratio": rational_payload(
                selected_t_prev_exact
            ),
            "selected_t_next_float64_exact_ratio": rational_payload(
                selected_t_next_exact
            ),
            "round_to_nearest_lower_midpoint_exact": rational_payload(
                round_to_nearest_lower_midpoint
            ),
            "round_to_nearest_upper_midpoint_exact": rational_payload(
                round_to_nearest_upper_midpoint
            ),
            "selected_phi_float64_radians": selected_phi_float64,
            "selected_cos_float64": selected_cos_float64,
            "selected_sin_float64": selected_sin_float64,
            "unit_float64_le_sha256": array_sha(selected_unit),
            "vector_float64_le_sha256": array_sha(selected_vector),
            "vector_l2": float(np.linalg.norm(selected_vector)),
            "vector_shape": list(selected_vector.shape),
            "exact_float_direct_gates": exact_float_direct_gates,
        }

    preflight_pass = bool(
        selection is not None
        and exact_float_direct_gates is not None
        and exact_float_direct_gates["pass"]
    )
    decision = (
        "GO_CW24_V10B_HALFANGLE_EXACT_ROOT_FREE_CELL_PREFLIGHT"
        if preflight_pass
        else "NO_GO_CW24_V10B_HALFANGLE_EXACT_ROOT_FREE_CELL_PREFLIGHT"
    )

    serialized_root_records = [
        {
            "root_record_id": int(record["root_record_id"]),
            "constraint_name": str(record["constraint_name"]),
            "relation": str(record["relation"]),
            "interval_index": int(record["interval_index"]),
            "multiplicity": int(record["multiplicity"]),
            "initial_lower": rational_payload(record["initial_lower"]),
            "initial_upper": rational_payload(record["initial_upper"]),
            "certified_lower": rational_payload(record["lower"]),
            "certified_upper": rational_payload(record["upper"]),
            "certified_width": rational_payload(record["upper"] - record["lower"]),
            "refined_for_endpoint": bool(record["refined_for_endpoint"]),
            "domain_classification": str(record["domain_classification"]),
        }
        for record in root_records
    ]
    serialized_barriers = [
        {
            "barrier_index": int(barrier["barrier_index"]),
            "lower_t": rational_payload(barrier["lower"]),
            "upper_t": rational_payload(barrier["upper"]),
            "width_t": rational_payload(barrier["upper"] - barrier["lower"]),
            "members": [
                {
                    "constraint_name": str(member["constraint_name"]),
                    "root_record_id": int(member["root_record_id"]),
                    "interval_index": int(member["interval_index"]),
                    "multiplicity": int(member["multiplicity"]),
                    "lower_t": rational_payload(member["lower"]),
                    "upper_t": rational_payload(member["upper"]),
                }
                for member in barrier["members"]
            ],
        }
        for barrier in barriers
    ]
    serialized_cells = []
    for cell in root_free_cells:
        exact_evaluation = exact_eval_by_index[int(cell["cell_index"])]
        serialized_cells.append(
            {
                "cell_index": int(cell["cell_index"]),
                "lower_t": rational_payload(cell["lower"]),
                "upper_t": rational_payload(cell["upper"]),
                "certified_inner_width_t": rational_payload(cell["width"]),
                "lower_source": str(cell["lower_source"]),
                "upper_source": str(cell["upper_source"]),
                "exact_rational_witness_t": rational_payload(cell["witness"]),
                "exact_Q_values": {
                    name: rational_payload(value)
                    for name, value in exact_evaluation["exact_values"].items()
                },
                "exact_signs": exact_evaluation["exact_signs"],
                "constraint_passes": exact_evaluation["passes"],
                "failing_names": exact_evaluation["failing_names"],
                "feasible": bool(exact_evaluation["pass"]),
                "above_minimum_selectable_width": bool(
                    cell["width"] > minimum_inner_width
                ),
            }
        )

    result = {
        "schema_version": SCHEMA,
        "status": decision,
        "decision": decision,
        "seed": SEED,
        "purpose": "train-only exact-rational half-angle certified root-free-cell preflight; not promotion evidence",
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
        "halfangle_domain": {
            "parameterization": "t=tan(phi/2); cos(phi)=(1-t^2)/(1+t^2); sin(phi)=2t/(1+t^2)",
            "theta_float64_radians": theta,
            "cosine_float64": cosine,
            "endpoint_T_contract": "compute tan(theta_float64/2) once in float64, then exact as_integer_ratio Rational",
            "endpoint_T_float64": endpoint_T_float64,
            "endpoint_T_float64_le_sha256": array_sha(
                np.asarray([endpoint_T_float64], dtype=np.float64)
            ),
            "endpoint_T_exact_rational": rational_payload(endpoint_T),
            "unit_v1_float64_le_sha256": array_sha(unit_v1),
            "unit_v6_float64_le_sha256": array_sha(unit_v6),
            "e2_float64_le_sha256": array_sha(e2),
            "reconstructed_v6_float64_le_sha256": array_sha(reconstructed_v6),
            "checks": geometry_checks,
        },
        "exact_root_free_cell_analysis": {
            "coordinate": "absolute actor delta from exact CW11",
            "sympy": {
                "version": sympy.__version__,
                "init_path": str(Path(sympy.__file__).resolve()),
                "init_sha256": sha256_file(SYMPY_INIT),
                "checks": sympy_checks,
            },
            "fixed_constants": {
                "PF_A_and_PF7_raw_native_margin_change_buffer": PF_STRICT_RAW_CHANGE_BUFFER,
                "root_isolation_eps": rational_payload(isolation_eps),
                "endpoint_refine_eps": rational_payload(endpoint_refine_eps),
                "minimum_certified_inner_t_width": rational_payload(
                    minimum_inner_width
                ),
                "scalar_direct_abs_tolerance": SCALAR_DIRECT_ABS_TOLERANCE,
            },
            "method": "exact Rational half-angle polynomials; conservative union of all domain root enclosures into unselectable barriers",
            "does_not_group_or_connect_through_root_enclosures": True,
            "does_not_claim_true_connected_components": True,
            "does_not_claim_widest_angle": True,
            "grid_points_evaluated": 0,
            "random_draws": 0,
            "radius_candidates": [R_SAFE],
            "weight_candidates": [],
            "detail_checks": detail_checks,
            "exact_construction_checks": exact_analysis_checks,
            "constraints": exact_constraint_output,
            "constraints_canonical_sha256": hashlib.sha256(
                canonical_json(exact_constraint_output)
            ).hexdigest(),
            "root_isolation_audits": root_audits,
            "root_records": serialized_root_records,
            "conservative_root_barriers": serialized_barriers,
            "certified_root_free_cells": serialized_cells,
            "certified_root_free_cell_count": len(root_free_cells),
            "exact_feasible_cell_count": len(feasible_cells),
            "selectable_feasible_cell_count": len(selectable_cells),
            "selection": selection,
            "selected_constraint_slacks": selected_constraint_slacks,
            "all_B352_row_slacks": all_baseline_row_slacks,
            "selected_vector_model_copy_performed": False,
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
            "exact_rational_root_isolation_only": True,
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
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError("v10b requires the exact locked my_project_env Python")
    if args.audit_only:
        import sympy

        audit_sympy_checks = {
            "version_exact": sympy.__version__ == SYMPY_VERSION,
            "init_path_exact": Path(sympy.__file__).resolve()
            == SYMPY_INIT.resolve(),
            "init_sha_exact": sha256_file(SYMPY_INIT) == SYMPY_INIT_SHA256,
        }
        if not all(audit_sympy_checks.values()):
            raise ProtocolError(f"v10b audit SymPy drift: {audit_sympy_checks}")
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
            "root_isolation_eps_exact": ROOT_ISOLATION_EPS_DENOMINATOR == 10**40,
            "endpoint_refine_eps_exact": ENDPOINT_REFINE_EPS_DENOMINATOR
            == 10**60,
            "minimum_inner_width_exact": MIN_CERTIFIED_INNER_T_WIDTH_NUMERATOR
            == 4
            and MIN_CERTIFIED_INNER_T_WIDTH_DENOMINATOR == 10**40,
            "scalar_direct_tolerance_exact": SCALAR_DIRECT_ABS_TOLERANCE
            == 5.0e-15,
            "python_exact": Path(sys.executable).resolve()
            == EXPECTED_PYTHON.resolve(),
            "sympy_version_exact": audit_sympy_checks["version_exact"],
            "sympy_init_path_exact": audit_sympy_checks["init_path_exact"],
            "sympy_init_sha_exact": audit_sympy_checks["init_sha_exact"],
            "output_absent": not OUTPUT.exists(),
        }
        if not all(checks.values()):
            raise ProtocolError(f"v10b static audit failed: {checks}")
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
                        "parameterization": "t=tan(phi/2); exact Rational Q(t)",
                        "PF_A_and_PF7_raw_change_buffer": PF_STRICT_RAW_CHANGE_BUFFER,
                        "root_isolation_eps": "1/10^40",
                        "endpoint_refine_eps": "1/10^60",
                        "minimum_certified_inner_t_width": "4/10^40",
                        "scalar_direct_abs_tolerance": SCALAR_DIRECT_ABS_TOLERANCE,
                        "protected_pair_names": list(PROTECTED_PAIR_NAMES),
                        "grid_or_random_or_radius_or_weight_scan_performed": False,
                        "selected_vector_forward_performed": False,
                        "new_candidate_shadow_count": 0,
                    },
                    "checks": checks,
                    "python_exact": checks["python_exact"],
                    "sympy_exact": checks["sympy_version_exact"]
                    and checks["sympy_init_path_exact"]
                    and checks["sympy_init_sha_exact"],
                    "CUDA_initialized": False,
                    "validation_or_test_rows_opened": 0,
                    "writes_performed": 0,
                }
            ).decode(),
            end="",
        )
        return
    if OUTPUT.exists():
        raise ProtocolError("frozen CW24 v10b output already exists")
    result = production_run()
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "selected_phi_float64_radians": (
                    result["exact_root_free_cell_analysis"]["selection"][
                        "selected_phi_float64_radians"
                    ]
                    if result["exact_root_free_cell_analysis"]["selection"]
                    is not None
                    else None
                ),
                "selected_t_exact": (
                    result["exact_root_free_cell_analysis"]["selection"][
                        "selected_t_exact"
                    ]
                    if result["exact_root_free_cell_analysis"]["selection"]
                    is not None
                    else None
                ),
                "selected_vector_float64_le_sha256": (
                    result["exact_root_free_cell_analysis"]["selection"][
                        "vector_float64_le_sha256"
                    ]
                    if result["exact_root_free_cell_analysis"]["selection"]
                    is not None
                    else None
                ),
                "exact_root_free_cell_preflight_pass": result[
                    "exact_root_free_cell_analysis"
                ]["pass"],
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

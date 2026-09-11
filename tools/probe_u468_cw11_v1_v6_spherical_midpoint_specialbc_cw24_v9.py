#!/usr/bin/env python3
"""CW24 v9 frozen-v1/v6 spherical-midpoint special-BC probe.

Two isolated historical replay subprocesses capture, through stdout only, the
frozen CW24-v1 failed actor delta and frozen CW24-v6 rejected second-shadow
actor delta.  Those replays are calibration evidence and are not new v9 train
shadows.  The sole fixed seed is the equal spherical midpoint at radius
0.000999975 from exact CW11.  It is evaluated only after deterministic
train-only B352 linear certificates pass.  If that midpoint fails, one and
only one same-point tangent-QP correction may be evaluated after KKT, tangent,
exponential-map, planned-slack, float32-copy-slack, and radius certification.

There is no interpolation-weight scan, radius scan, BF16 buffer, optimizer,
checkpoint write, network access, validation/test access, or official use.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import types
import zlib
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_v1_v6_spherical_midpoint_specialbc_cw24_v9.py"

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

OUTPUT = ROOT / "artifacts/cw24_cw11_v1_v6_spherical_midpoint_specialbc_trainonly_v9.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-v1-v6-spherical-midpoint-specialbc-cw24-v9"
SEED = 202608049
TOTAL_RADIUS_CAP = 9.9998e-4
R_SAFE = 0.000999975
R_REAL_TOL = 5.0e-9
TANGENT_STEP_CAP = 2.5e-4
MAX_NEW_TRAIN_SHADOWS = 2
LINEAR_TOL = 5.0e-9

V1_DELTA_SHA256 = "48497e7b9857199bab0d653fc75ea0e0861ed634c64fca0483a3a0520687ec86"
V1_ACTUAL_L2 = 0.0010000012855891469
V1_MODEL_SHA256 = "7caa8ab0ad95cc7ffeb19c5719cbd65dd4740ce195da029cb0525be621392a0e"
V6_S2_DELTA_SHA256 = "3c8ae987fda33444d6698b44f3164b41a4ff0de65eee1f8438cf5382e1278d41"
V6_S2_PLANNED_L2 = 0.0009909412795123538
V6_S2_ACTUAL_L2 = 0.000990940375725956
V6_S2_STEP_L2 = 0.000753542521139092
V6_S2_TRIAL_MERIT = 25123.250002966826
V6_S2_MATRIX_SHA256 = "9954fe13b421ab0eed6a355904d3cf27b5ef6dbc5eba8a9403969336ec9d668e"
V6_S2_RHS_SHA256 = "e58ec7fa28887e984a323c8018943244dc2e73125e35b299a4c863d050d0af76"
TOP1_FLIP_SHA256 = "fde6fab074495c238d4c4ef42a2f12ec595da78e35836578f7a3828002037b88"
RETENTION_FLIP_SHA256 = "243f4a21d6c0b178a6306bdb63ac738aeb5bb13f6be0ccfaa68eebe02eac2d99"


class ProtocolError(RuntimeError):
    """Fail-closed CW24-v9 protocol error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(
        np.asarray(value, dtype="<f8").tobytes(order="C")
    ).hexdigest()


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


def locked_inputs() -> tuple[dict[str, bytes], dict[str, Any]]:
    specs = (
        ("v1_source", V1_SOURCE, V1_SOURCE_SHA256, V1_SOURCE_MODE),
        ("v1_result", V1_RESULT, V1_RESULT_SHA256, V1_RESULT_MODE),
        ("v4_source", V4_SOURCE, V4_SOURCE_SHA256, V4_SOURCE_MODE),
        ("v4_result", V4_RESULT, V4_RESULT_SHA256, V4_RESULT_MODE),
        ("v6_source", V6_SOURCE, V6_SOURCE_SHA256, V6_SOURCE_MODE),
        ("v6_result", V6_RESULT, V6_RESULT_SHA256, V6_RESULT_MODE),
        ("v8_source", V8_SOURCE, V8_SOURCE_SHA256, V8_SOURCE_MODE),
        ("v8_result", V8_RESULT, V8_RESULT_SHA256, V8_RESULT_MODE),
    )
    blobs: dict[str, bytes] = {}
    evidence: dict[str, Any] = {}
    for name, path, digest, mode in specs:
        blobs[name], evidence[name] = regular_source(path, digest, mode, f"frozen {name}")
    return blobs, evidence


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
        "eight_frozen_locks_present": all(
            digest.encode() in source
            for digest in (
                V1_SOURCE_SHA256, V1_RESULT_SHA256,
                V4_SOURCE_SHA256, V4_RESULT_SHA256,
                V6_SOURCE_SHA256, V6_RESULT_SHA256,
                V8_SOURCE_SHA256, V8_RESULT_SHA256,
            )
        ),
        "fixed_radius": source.count(b"R_SAFE = " + b"0.000999975") == 1,
        "fixed_equal_midpoint": (b"midpoint_sum = unit_v1 " + b"+ unit_v6") in source,
        "two_new_shadow_cap": source.count(b"MAX_NEW_TRAIN_SHADOWS = " + b"2") == 1,
        "no_half_BF16_buffer": (b"BUFFER_" + b"FRACTION") not in source,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "no_network_calls": all(
            token not in source
            for token in (b"requests" + b".", b"urllib" + b".", b"socket" + b".")
        ),
        "subprocess_pipe_capture": b"capture_output=True" in source
        and b"--internal-capture-v1" in source
        and b"--internal-capture-v6" in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v9 source audit failed: {checks}")
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
                raise ProtocolError("short v9 result write")
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
        raise ProtocolError(f"v9 publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def splice_exact_once(
    source: str, old: str, new: str, label: str, audit: list[dict[str, Any]]
) -> str:
    count = source.count(old)
    if count != 1:
        raise ProtocolError(f"{label}: expected one splice, observed {count}")
    audit.append({
        "label": label,
        "occurrences": count,
        "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
        "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
    })
    return source.replace(old, new, 1)


def splice_region_once(
    source: str, start: str, end: str, replacement: str, label: str,
    audit: list[dict[str, Any]],
) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise ProtocolError(f"{label}: non-unique region marker")
    begin = source.index(start)
    finish = source.index(end, begin)
    old = source[begin:finish]
    audit.append({
        "label": label,
        "occurrences": 1,
        "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
        "new_sha256": hashlib.sha256(replacement.encode()).hexdigest(),
    })
    return source[:begin] + replacement + source[finish:]


def encode_vector(value: Any) -> str:
    import numpy as np

    raw = np.asarray(value, dtype="<f8").tobytes(order="C")
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def decode_vector(payload: str, expected_sha: str) -> Any:
    import numpy as np

    raw = zlib.decompress(base64.b64decode(payload.encode("ascii"), validate=True))
    vector = np.frombuffer(raw, dtype="<f8").copy()
    checks = {
        "shape65793": vector.shape == (65793,),
        "finite": bool(np.isfinite(vector).all()),
        "sha_exact": array_sha(vector) == expected_sha,
    }
    if not all(checks.values()):
        raise ProtocolError(f"captured vector decode drift: {checks}")
    vector.setflags(write=False)
    return vector


def v1_capture_transform(raw: bytes) -> tuple[bytes, dict[str, Any]]:
    source = raw.decode()
    audit: list[dict[str, Any]] = []
    old = '''        if not candidate_pass:
            return {
'''
    new = '''        if not candidate_pass:
            V9_CAPTURE_V1_FAILED(
                actual_delta.copy(),
                {
                    "actual_l2": actual_l2,
                    "candidate_model_state_sha256": candidate_hash,
                    "candidate_pass": candidate_pass,
                    "trial": trial,
                    "geometry": geometry,
                },
            )
            return {
'''
    source = splice_exact_once(
        source, old, new, "capture_frozen_v1_failed_actual_delta", audit
    )
    transformed = source.encode()
    return transformed, {
        "frozen_source_sha256": hashlib.sha256(raw).hexdigest(),
        "transformed_source_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_source_bytes": len(transformed),
        "transformations": audit,
    }


def v6_capture_transform(raw_v4: bytes) -> tuple[bytes, dict[str, Any]]:
    source = raw_v4.decode()
    audit: list[dict[str, Any]] = []
    rejection_old = '''            else:
                rejection_count += 1
                trust_radius *= 0.5
                cw20.apply_flat_actor(
'''
    rejection_new = '''            else:
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
    source = splice_exact_once(
        source, rejection_old, rejection_new, "v6_dynamic_cut_resets_full_trust", audit
    )
    source = splice_exact_once(
        source, accepted_old, accepted_new, "v6_accepted_nonterminal_full_trust", audit
    )
    baseline = source.encode()
    baseline_checks = {
        "sha_exact": hashlib.sha256(baseline).hexdigest()
        == "ac665ebf6723889178a56f3df32666b69990668ed5f4ad1ca3dd20cfa54506f2",
        "bytes_exact": len(baseline) == 72731,
    }
    if not all(baseline_checks.values()):
        raise ProtocolError(f"frozen v6 two-splice baseline drift: {baseline_checks}")
    capture_old = '''            iteration_log.append(log_entry)
            if accept:
'''
    capture_new = '''            if trial_evaluations == 2 and accepted_iterates == 1 and not accept:
                V9_CAPTURE_V6_S2(
                    actual_trial_x.copy(),
                    {
                        "planned_l2": float(np.linalg.norm(planned_trial_x)),
                        "actual_l2": actual_trial_l2,
                        "trial_state": trial_state,
                        "log_entry": log_entry,
                        "linearization": linearization_audit,
                        "solver": solver_audit,
                    },
                )
            iteration_log.append(log_entry)
            if accept:
'''
    source = splice_exact_once(
        source, capture_old, capture_new, "capture_frozen_v6_rejected_S2", audit
    )
    transformed = source.encode()
    return transformed, {
        "frozen_source_sha256": hashlib.sha256(raw_v4).hexdigest(),
        "v6_two_splice_baseline": {
            "sha256": hashlib.sha256(baseline).hexdigest(),
            "bytes": len(baseline),
            "checks": baseline_checks,
        },
        "transformed_source_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_source_bytes": len(transformed),
        "transformations": audit,
    }


def load_module_from_bytes(
    name: str,
    filename: Path,
    source: bytes,
    injected: Mapping[str, Any] | None = None,
) -> ModuleType:
    if name in sys.modules:
        raise ProtocolError(f"module name already occupied: {name}")
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    module.__package__ = ""
    if injected:
        module.__dict__.update(injected)
    sys.modules[name] = module
    try:
        exec(compile(source, str(filename), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def internal_capture_v1() -> dict[str, Any]:
    import numpy as np

    blobs, evidence = locked_inputs()
    transformed, transform_evidence = v1_capture_transform(blobs["v1_source"])
    box: dict[str, Any] = {}

    def capture(vector: Any, observed: Mapping[str, Any]) -> None:
        if box:
            raise ProtocolError("v1 capture invoked more than once")
        value = np.asarray(vector, dtype=np.float64).copy()
        checks = {
            "shape65793": value.shape == (65793,),
            "dtype_float64": value.dtype == np.float64,
            "finite": bool(np.isfinite(value).all()),
            "c_contiguous": bool(value.flags.c_contiguous),
            "sha_exact": array_sha(value) == V1_DELTA_SHA256,
            "l2_exact": math.isclose(
                float(np.linalg.norm(value)), V1_ACTUAL_L2, rel_tol=0.0, abs_tol=5.0e-15
            ),
            "reported_l2_exact": math.isclose(
                float(observed["actual_l2"]), V1_ACTUAL_L2, rel_tol=0.0, abs_tol=5.0e-15
            ),
            "model_exact": observed["candidate_model_state_sha256"] == V1_MODEL_SHA256,
            "candidate_failed": observed["candidate_pass"] is False,
            "trial_failed": observed["trial"]["pass"] is False,
        }
        if not all(checks.values()):
            raise ProtocolError(f"v1 capture fingerprint drift: {checks}")
        value.setflags(write=False)
        box.update({"vector": value, "checks": checks})

    module = load_module_from_bytes(
        "cw24_v9_isolated_v1_capture",
        V1_SOURCE,
        transformed,
        {"V9_CAPTURE_V1_FAILED": capture},
    )
    result = module.production_run()
    replay_checks = {
        "capture_once": bool(box),
        "full_result_canonical_exact": canonical_json(result) == blobs["v1_result"],
        "decision_NO_GO": result["decision"] == "NO_GO_CW24_TWO_STAGE_TRAIN_GATE",
        "candidate_payload_absent": result["endpoint"].get("candidate_payload") is None,
        "candidate_model_exact": result["endpoint"]["trial"][
            "candidate_model_state_sha256"
        ] == V1_MODEL_SHA256,
        "historical_shadow_count_exact_one": result["endpoint"][
            "changed_candidate_train_shadow_count"
        ]
        == 1,
    }
    if not all(replay_checks.values()):
        raise ProtocolError(f"frozen v1 full replay drift: {replay_checks}")
    return {
        "schema_version": SCHEMA,
        "capture_kind": "historical_frozen_v1_failed_actual_delta",
        "vector_float64_le_sha256": V1_DELTA_SHA256,
        "vector_l2": V1_ACTUAL_L2,
        "vector_zlib_base64": encode_vector(box["vector"]),
        "capture_checks": box["checks"],
        "replay_checks": replay_checks,
        "transform": transform_evidence,
        "locked_inputs": evidence,
        "historical_replay_changed_candidate_train_shadow_count": int(
            result["endpoint"]["changed_candidate_train_shadow_count"]
        ),
        "new_v9_train_shadows": 0,
        "writes_performed": 0,
    }


def internal_capture_v6() -> dict[str, Any]:
    import numpy as np

    blobs, evidence = locked_inputs()
    transformed, transform_evidence = v6_capture_transform(blobs["v4_source"])
    frozen_v6 = json.loads(blobs["v6_result"])
    frozen_v8 = json.loads(blobs["v8_result"])
    box: dict[str, Any] = {}

    def capture(vector: Any, observed: Mapping[str, Any]) -> None:
        if box:
            raise ProtocolError("v6 S2 capture invoked more than once")
        value = np.asarray(vector, dtype=np.float64).copy()
        state = observed["trial_state"]
        log = observed["log_entry"]
        linearization = observed["linearization"]
        solver = observed["solver"]
        repairs = {
            name: bool(state["target_observed"][name]["ordered_correct"])
            for name in ("pf0_boundary_a", "pf0_boundary_b", "pf7_boundary")
        }
        checks = {
            "shape65793": value.shape == (65793,),
            "dtype_float64": value.dtype == np.float64,
            "finite": bool(np.isfinite(value).all()),
            "c_contiguous": bool(value.flags.c_contiguous),
            "sha_exact": array_sha(value) == V6_S2_DELTA_SHA256,
            "planned_l2_exact": math.isclose(
                float(observed["planned_l2"]), V6_S2_PLANNED_L2,
                rel_tol=0.0, abs_tol=5.0e-13,
            ),
            "actual_l2_exact": math.isclose(
                float(observed["actual_l2"]), V6_S2_ACTUAL_L2,
                rel_tol=0.0, abs_tol=5.0e-13,
            ),
            "matrix_exact": linearization["matrix_float64_le_sha256"]
            == V6_S2_MATRIX_SHA256,
            "rhs_exact": linearization["rhs_float64_le_sha256"] == V6_S2_RHS_SHA256,
            "solver_certified": solver["certified"] is True,
            "solver_step_exact": math.isclose(
                float(solver["step_l2"]), V6_S2_STEP_L2, rel_tol=0.0, abs_tol=5.0e-15
            ),
            "alpha_full": float(log["alpha"]) == 1.0,
            "trial_merit_exact": math.isclose(
                float(state["merit"]), V6_S2_TRIAL_MERIT, rel_tol=0.0, abs_tol=5.0e-9
            ),
            "PF_repairs_exact": repairs == {
                "pf0_boundary_a": True,
                "pf0_boundary_b": True,
                "pf7_boundary": False,
            },
            "top1_flip_identity_exact": state["top1_flips"] == [TOP1_FLIP_SHA256],
            "retention_flip_identity_exact": state["retention_flips"]
            == [RETENTION_FLIP_SHA256],
            "new_pair_names_empty": not log["new_pair_names"],
            "gate_failed": log["gate_pass"] is False,
            "rejected": log["accepted"] is False,
        }
        if not all(checks.values()):
            raise ProtocolError(f"v6 rejected-S2 capture drift: {checks}")
        value.setflags(write=False)
        box.update({"vector": value, "checks": checks})

    module = load_module_from_bytes(
        "cw24_v9_isolated_v6_capture_v4",
        V4_SOURCE,
        transformed,
        {"V9_CAPTURE_V6_S2": capture},
    )
    originals = {
        "TRUST_RADIUS_MAX": module.TRUST_RADIUS_MAX,
        "MAX_TRIAL_EVALUATIONS": module.MAX_TRIAL_EVALUATIONS,
        "MAX_ACCEPTED_ITERATES": module.MAX_ACCEPTED_ITERATES,
        "SEED": module.SEED,
    }
    module.TRUST_RADIUS_MAX = TOTAL_RADIUS_CAP
    module.MAX_TRIAL_EVALUATIONS = 2
    module.MAX_ACCEPTED_ITERATES = 2
    module.SEED = 202608046
    try:
        result = module.production_run()
    finally:
        for name, value in originals.items():
            setattr(module, name, value)

    replay_endpoint = dict(result["endpoint"])
    frozen_endpoint = dict(frozen_v6["endpoint"])
    replay_endpoint.pop("decision", None)
    replay_endpoint.pop("reason", None)
    frozen_endpoint.pop("decision", None)
    frozen_endpoint.pop("reason", None)
    v8_trial = frozen_v8["endpoint"]["trial"]
    replay_checks = {
        "capture_once": bool(box),
        "endpoint_fields_exact_excluding_wrapper_decision_reason": canonical_json(replay_endpoint)
        == canonical_json(frozen_endpoint),
        "v8_same_actual_l2": math.isclose(
            float(v8_trial["actual_additional_from_CW11_l2"]),
            V6_S2_ACTUAL_L2,
            rel_tol=0.0,
            abs_tol=5.0e-13,
        ),
        "v8_top1_identity_exact": v8_trial[
            "top1_guard_correct_to_wrong_line_sha256"
        ] == [TOP1_FLIP_SHA256],
        "v8_retention_identity_exact": v8_trial[
            "retention160_correct_to_wrong_line_sha256"
        ] == [RETENTION_FLIP_SHA256],
        "candidate_payload_absent": result["endpoint"].get("candidate_payload") is None,
        "historical_shadow_count_exact_two": result["endpoint"][
            "changed_candidate_train_shadow_count"
        ]
        == 2,
    }
    if not all(replay_checks.values()):
        raise ProtocolError(f"frozen v6/v8 replay drift: {replay_checks}")
    return {
        "schema_version": SCHEMA,
        "capture_kind": "historical_frozen_v6_rejected_second_shadow_actual_delta",
        "vector_float64_le_sha256": V6_S2_DELTA_SHA256,
        "vector_l2": V6_S2_ACTUAL_L2,
        "vector_zlib_base64": encode_vector(box["vector"]),
        "capture_checks": box["checks"],
        "replay_checks": replay_checks,
        "transform": transform_evidence,
        "locked_inputs": evidence,
        "historical_replay_changed_candidate_train_shadow_count": int(
            result["endpoint"]["changed_candidate_train_shadow_count"]
        ),
        "new_v9_train_shadows": 0,
        "writes_performed": 0,
    }


def capture_subprocess(mode: str, expected_sha: str) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError("historical capture parent is outside my_project_env")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/dev/null"
    launcher = (
        "import runpy,sys;"
        f"sys.argv=[{str(SCRIPT)!r},{mode!r}];"
        f"runpy.run_path({str(SCRIPT)!r},run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", launcher],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        check=False,
        timeout=900,
    )
    checks = {
        "returncode_zero": completed.returncode == 0,
        "stderr_empty": completed.stderr == b"",
        "stdout_nonempty": bool(completed.stdout),
    }
    if not all(checks.values()):
        raise ProtocolError(
            "isolated historical replay failed: "
            + json.dumps(
                {
                    "mode": mode,
                    "returncode": completed.returncode,
                    "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                    "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                    "checks": checks,
                },
                sort_keys=True,
            )
        )
    payload = json.loads(completed.stdout)
    stdout_checks = {
        "canonical_payload_exactly_stdout": canonical_json(payload) == completed.stdout,
        "reported_vector_sha_exact": payload.get("vector_float64_le_sha256")
        == expected_sha,
        "reported_shape_contract": payload.get("capture_checks", {}).get("shape65793")
        is True,
    }
    if not all(stdout_checks.values()):
        raise ProtocolError(f"isolated capture stdout drift: {stdout_checks}")
    vector = decode_vector(payload.pop("vector_zlib_base64"), expected_sha)
    parent_vector_checks = {
        "shape65793": tuple(vector.shape) == (65793,),
        "finite": bool(np.isfinite(vector).all()),
        "decoded_sha_exact": array_sha(vector) == expected_sha,
    }
    if not all(parent_vector_checks.values()):
        raise ProtocolError(f"parent decoded-vector lock failed: {parent_vector_checks}")
    payload["subprocess"] = {
        "mode": mode,
        "python": str(Path(sys.executable).resolve()),
        "isolated": True,
        "dash_c_launcher": True,
        "bytecode_disabled": True,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "checks": checks,
        "stdout_checks": stdout_checks,
        "parent_vector_checks": parent_vector_checks,
    }
    return vector, payload


def v9_loop_source() -> str:
    return '''        anchor_v1 = np.asarray(V9_V1_X, dtype=np.float64).copy()
        anchor_v6 = np.asarray(V9_V6_X, dtype=np.float64).copy()
        anchor_v1_l2 = float(np.linalg.norm(anchor_v1))
        anchor_v6_l2 = float(np.linalg.norm(anchor_v6))
        unit_v1 = anchor_v1 / anchor_v1_l2
        unit_v6 = anchor_v6 / anchor_v6_l2
        anchor_cosine = float(unit_v1 @ unit_v6)
        midpoint_sum = unit_v1 + unit_v6
        midpoint_sum_l2 = float(np.linalg.norm(midpoint_sum))
        planned_midpoint = V9_R_SAFE * midpoint_sum / midpoint_sum_l2

        zero_x = np.zeros_like(cw11_flat, dtype=np.float64)
        current_x = zero_x.copy()
        current_state = evaluate_outputs(baseline_outputs, current_x)
        accepted_iterates = 0
        trial_evaluations = 0
        rejection_count = 0
        cutting_plane_additions = 0
        iteration_log: list[dict[str, Any]] = []
        terminal_solver_audit: dict[str, Any] | None = None
        terminal_reason = "V9_PRE_CANDIDATE_CERTIFICATE_NOT_RUN"

        matrix0, rhs0, details0, linearization0 = build_linearized_constraints()
        detail_index0 = {item["name"]: index for index, item in enumerate(details0)}
        required_names = {
            "pf0_boundary_a",
            "pf0_boundary_b",
            "pf7_boundary",
            "dominic_closest_margin",
            "historical_retention_0142a2a3d5bd",
            "historical_retention_243f4a21d6c0",
            "historical_top1_fde6fab07449",
            *{f"zero_margin_guard_{offset}" for offset in range(6)},
        }
        missing_names = sorted(required_names - set(detail_index0))
        planned_effect_normalized = matrix0 @ planned_midpoint
        planned_slack0 = planned_effect_normalized - rhs0
        raw_predicted_changes = {
            item["name"]: float(
                planned_effect_normalized[index] * float(item["gradient_l2"])
            )
            for index, item in enumerate(details0)
        }
        predicted_slacks = {
            item["name"]: float(planned_slack0[index])
            for index, item in enumerate(details0)
        }
        protection_names = [f"zero_margin_guard_{offset}" for offset in range(6)] + [
            "historical_retention_0142a2a3d5bd",
            "historical_retention_243f4a21d6c0",
            "historical_top1_fde6fab07449",
        ]
        protection_min_slack = min(
            predicted_slacks.get(name, float("-inf")) for name in protection_names
        )
        pair_targets_exact = all(
            float(item["target"]) == float(pair_specs[item["name"]]["target"])
            for item in details0
            if item["kind"] == "native_pair_margin"
        )
        pre_candidate_checks = {
            "anchor_v1_shape65793": anchor_v1.shape == (65793,),
            "anchor_v6_shape65793": anchor_v6.shape == (65793,),
            "anchors_finite": bool(np.isfinite(anchor_v1).all())
            and bool(np.isfinite(anchor_v6).all()),
            "anchor_v1_sha_exact": engine.array_sha(anchor_v1) == V9_V1_SHA256,
            "anchor_v6_sha_exact": engine.array_sha(anchor_v6) == V9_V6_SHA256,
            "anchor_directions_distinct": abs(anchor_cosine) < 1.0 - 1.0e-10,
            "midpoint_direction_nondegenerate": math.isfinite(midpoint_sum_l2)
            and midpoint_sum_l2 > 1.0e-8,
            "fixed_equal_midpoint_radius": math.isclose(
                float(np.linalg.norm(planned_midpoint)),
                V9_R_SAFE,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ),
            "all_required_constraint_names": not missing_names,
            "PF_A_predicted_improves": raw_predicted_changes.get(
                "pf0_boundary_a", float("-inf")
            ) > 0.0,
            "PF7_predicted_improves": raw_predicted_changes.get(
                "pf7_boundary", float("-inf")
            ) > 0.0,
            "PF_B_predicted_nondegrade": raw_predicted_changes.get(
                "pf0_boundary_b", float("-inf")
            ) >= 0.0,
            "six_zero_two_retention_top1_linear_protected": protection_min_slack
            >= -V9_LINEAR_TOL,
            "pair_targets_are_unbuffered_frozen_v4_gates": pair_targets_exact,
            "no_half_BF16_buffer": True,
            "planned_midpoint_finite": bool(np.isfinite(planned_midpoint).all()),
            "planned_midpoint_inside_total_cap": float(np.linalg.norm(planned_midpoint))
            < TOTAL_RADIUS_CAP,
        }
        pre_candidate_certificate = {
            "checks": pre_candidate_checks,
            "pass": all(pre_candidate_checks.values()),
            "anchor_v1_l2": anchor_v1_l2,
            "anchor_v6_l2": anchor_v6_l2,
            "anchor_cosine": anchor_cosine,
            "midpoint_sum_l2": midpoint_sum_l2,
            "R_safe": V9_R_SAFE,
            "planned_midpoint_l2": float(np.linalg.norm(planned_midpoint)),
            "planned_midpoint_float64_le_sha256": engine.array_sha(planned_midpoint),
            "raw_predicted_changes": raw_predicted_changes,
            "predicted_normalized_slacks": predicted_slacks,
            "protection_minimum_normalized_slack": protection_min_slack,
            "full_constraint_minimum_normalized_slack": float(np.min(planned_slack0)),
            "missing_constraint_names": missing_names,
            "linearization": linearization0,
        }

        if not pre_candidate_certificate["pass"]:
            terminal_reason = "V9_PRE_CANDIDATE_LINEAR_CERTIFICATE_FAILED"
            iteration_log.append(
                {
                    "trial_one_based": 1,
                    "candidate_evaluated": False,
                    "shadow_role": "equal_spherical_midpoint",
                    "reason": terminal_reason,
                    "pre_candidate_certificate": pre_candidate_certificate,
                }
            )
        else:
            cw20.apply_flat_actor(
                parameters,
                cw22.EXPECTED_ACTOR_NAMES,
                cw11_flat + planned_midpoint,
                torch,
            )
            actual_midpoint = (
                cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np) - cw11_flat
            )
            R_real = float(np.linalg.norm(actual_midpoint))
            actual_midpoint_slack0 = matrix0 @ actual_midpoint - rhs0
            actual_protection_min_slack = min(
                float(actual_midpoint_slack0[detail_index0[name]])
                for name in protection_names
            )
            actual_center_checks = {
                "actual_midpoint_finite": bool(np.isfinite(actual_midpoint).all()),
                "actual_midpoint_changed": engine.array_sha(actual_midpoint)
                not in {V9_V1_SHA256, V9_V6_SHA256},
                "R_real_close_to_R_safe": abs(R_real - V9_R_SAFE) <= V9_R_REAL_TOL,
                "R_real_strictly_below_total_cap": R_real < TOTAL_RADIUS_CAP,
                "actual_linear_protection_replay": actual_protection_min_slack
                >= -V9_LINEAR_TOL,
            }
            if not all(actual_center_checks.values()):
                terminal_reason = "V9_FLOAT32_MIDPOINT_PRE_FORWARD_CERTIFICATE_FAILED"
                cw20.apply_flat_actor(
                    parameters,
                    cw22.EXPECTED_ACTOR_NAMES,
                    cw11_flat,
                    torch,
                )
                restored_zero = (
                    cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np) - cw11_flat
                )
                restore_checks = {
                    "CW11_actor_restored_before_forward": bool(
                        np.array_equal(restored_zero, zero_x)
                    )
                }
                if not all(restore_checks.values()):
                    raise ProtocolError(
                        f"v9 midpoint pre-forward restore drift: {restore_checks}"
                    )
                iteration_log.append(
                    {
                        "trial_one_based": 1,
                        "candidate_evaluated": False,
                        "shadow_role": "equal_spherical_midpoint",
                        "reason": terminal_reason,
                        "pre_candidate_certificate": pre_candidate_certificate,
                        "actual_center": {
                            "R_real": R_real,
                            "actual_midpoint_float64_le_sha256": engine.array_sha(
                                actual_midpoint
                            ),
                            "actual_protection_minimum_normalized_slack": actual_protection_min_slack,
                            "checks": actual_center_checks,
                        },
                        "restore_checks": restore_checks,
                    }
                )
            else:
                with torch.no_grad():
                    midpoint_outputs = ppo.model_forward(model, batch, device)
                midpoint_state = evaluate_outputs(midpoint_outputs, actual_midpoint)
                trial_evaluations = 1
                midpoint_new_pairs = []
                for item in discover_failure_pairs(midpoint_state):
                    if add_pair(**item):
                        midpoint_new_pairs.append(item["name"])
                cutting_plane_additions += len(midpoint_new_pairs)
                midpoint_gate_pass = all(
                    midpoint_state["train_checks"].values()
                ) and not midpoint_new_pairs
                iteration_log.append(
                    {
                        "trial_one_based": 1,
                        "candidate_evaluated": True,
                        "shadow_role": "equal_spherical_midpoint",
                        "alpha": 1.0,
                        "interpolation_t": 0.5,
                        "radius_scans": 0,
                        "weight_scans": 0,
                        "planned_trial_l2": float(np.linalg.norm(planned_midpoint)),
                        "actual_trial_l2": R_real,
                        "actual_trial_x_float64_le_sha256": engine.array_sha(
                            actual_midpoint
                        ),
                        "trial_merit": float(midpoint_state["merit"]),
                        "PF_repairs": {
                            name: bool(
                                midpoint_state["target_observed"][name]["ordered_correct"]
                            )
                            for name in PF_FINAL_NATIVE_MARGIN
                        },
                        "top1_flip_identities": midpoint_state["top1_flips"],
                        "retention_flip_identities": midpoint_state["retention_flips"],
                        "new_pair_names": midpoint_new_pairs,
                        "accepted": midpoint_gate_pass,
                        "gate_pass": midpoint_gate_pass,
                        "pre_candidate_certificate": pre_candidate_certificate,
                        "actual_center": {
                            "R_real": R_real,
                            "actual_midpoint_float64_le_sha256": engine.array_sha(
                                actual_midpoint
                            ),
                            "actual_protection_minimum_normalized_slack": actual_protection_min_slack,
                            "checks": actual_center_checks,
                        },
                    }
                )
                current_x = actual_midpoint
                current_state = midpoint_state
                if midpoint_gate_pass:
                    accepted_iterates = 1
                    terminal_reason = "ALL_TARGETED_TRAIN_GATES_PASSED"
                else:
                    rejection_count = 1
                    terminal_reason = "V9_MIDPOINT_FAILED_BEFORE_TANGENT"

                    matrix1, rhs1, details1, linearization1 = build_linearized_constraints()
                    center_l2 = float(np.linalg.norm(current_x))
                    center_radius_matches_R_real = center_l2 == R_real
                    center_unit = current_x / R_real
                    radial_dots = matrix1 @ center_unit
                    projected_rows = matrix1 - np.outer(radial_dots, center_unit)
                    projected_norms = np.linalg.norm(projected_rows, axis=1)
                    tangent_rows_valid = bool(np.isfinite(projected_rows).all()) and bool(
                        np.all(projected_norms > 1.0e-12)
                    )
                    tangent_solver = None
                    tangent_z = None
                    tangent_certificate: dict[str, Any] = {
                        "center_uses_actual_float32_midpoint": True,
                        "center_l2": center_l2,
                        "R_real": R_real,
                        "center_radius_matches_R_real": center_radius_matches_R_real,
                        "center_x_float64_le_sha256": engine.array_sha(current_x),
                        "projected_row_min_l2": float(np.min(projected_norms)),
                        "projected_rows_valid": tangent_rows_valid,
                        "tangent_step_cap": V9_TANGENT_STEP_CAP,
                        "no_half_BF16_buffer": True,
                    }
                    if tangent_rows_valid:
                        normalized_tangent = projected_rows / projected_norms[:, None]
                        tangent_rhs = rhs1 / projected_norms
                        tangent_z, tangent_solver = dual_hildreth_minimum_norm(
                            normalized_tangent, tangent_rhs, np
                        )
                        tangent_z_l2 = float(np.linalg.norm(tangent_z))
                        tangent_qp_slack = matrix1 @ tangent_z - rhs1
                        tangent_dot = float(center_unit @ tangent_z)
                        tangent_checks = {
                            "center_radius_exactly_R_real": center_radius_matches_R_real,
                            "solver_certified": tangent_solver["certified"] is True,
                            "z_finite": bool(np.isfinite(tangent_z).all()),
                            "z_nonzero": tangent_z_l2 > 1.0e-15,
                            "z_within_fixed_cap": tangent_z_l2 <= V9_TANGENT_STEP_CAP,
                            "z_tangent_to_actual_center": abs(tangent_dot) <= 1.0e-12,
                            "original_same_point_constraints_satisfied_by_z": float(
                                np.min(tangent_qp_slack)
                            ) >= -V9_LINEAR_TOL,
                        }
                        tangent_certificate.update(
                            {
                                "checks": tangent_checks,
                                "pass": all(tangent_checks.values()),
                                "z_l2": tangent_z_l2,
                                "z_float64_le_sha256": engine.array_sha(tangent_z),
                                "z_dot_center_unit": tangent_dot,
                                "z_minimum_same_point_normalized_slack": float(
                                    np.min(tangent_qp_slack)
                                ),
                                "solver": tangent_solver,
                                "linearization": linearization1,
                            }
                        )
                    else:
                        tangent_certificate.update(
                            {
                                "checks": {"projected_rows_valid": False},
                                "pass": False,
                                "linearization": linearization1,
                            }
                        )

                    if not tangent_certificate["pass"]:
                        terminal_reason = "V9_TANGENT_QP_CERTIFICATE_FAILED"
                        terminal_solver_audit = tangent_solver
                        iteration_log.append(
                            {
                                "trial_one_based": 2,
                                "candidate_evaluated": False,
                                "shadow_role": "same_point_spherical_tangent_correction",
                                "reason": terminal_reason,
                                "tangent_certificate": tangent_certificate,
                            }
                        )
                    else:
                        tangent_z_l2 = float(np.linalg.norm(tangent_z))
                        theta = tangent_z_l2 / R_real
                        planned_endpoint = R_real * (
                            math.cos(theta) * center_unit
                            + math.sin(theta) * tangent_z / tangent_z_l2
                        )
                        planned_displacement = planned_endpoint - current_x
                        planned_exp_slack = matrix1 @ planned_displacement - rhs1
                        planned_exp_checks = {
                            "exp_map_uses_same_actual_center_radius": math.isclose(
                                float(np.linalg.norm(planned_endpoint)),
                                R_real,
                                rel_tol=0.0,
                                abs_tol=1.0e-15,
                            ),
                            "planned_endpoint_finite": bool(
                                np.isfinite(planned_endpoint).all()
                            ),
                            "planned_endpoint_inside_total_cap": float(
                                np.linalg.norm(planned_endpoint)
                            ) < TOTAL_RADIUS_CAP,
                            "planned_exp_replays_same_point_constraints": float(
                                np.min(planned_exp_slack)
                            ) >= -V9_LINEAR_TOL,
                        }
                        exp_map_certificate = {
                            "theta": theta,
                            "R_real": R_real,
                            "planned_endpoint_l2": float(
                                np.linalg.norm(planned_endpoint)
                            ),
                            "planned_endpoint_float64_le_sha256": engine.array_sha(
                                planned_endpoint
                            ),
                            "planned_minimum_same_point_normalized_slack": float(
                                np.min(planned_exp_slack)
                            ),
                            "checks": planned_exp_checks,
                            "pass": all(planned_exp_checks.values()),
                        }
                        if not exp_map_certificate["pass"]:
                            terminal_reason = "V9_PLANNED_EXP_MAP_LINEAR_REPLAY_FAILED"
                            iteration_log.append(
                                {
                                    "trial_one_based": 2,
                                    "candidate_evaluated": False,
                                    "shadow_role": "same_point_spherical_tangent_correction",
                                    "reason": terminal_reason,
                                    "tangent_certificate": tangent_certificate,
                                    "exp_map_certificate": exp_map_certificate,
                                }
                            )
                        else:
                            cw20.apply_flat_actor(
                                parameters,
                                cw22.EXPECTED_ACTOR_NAMES,
                                cw11_flat + planned_endpoint,
                                torch,
                            )
                            actual_endpoint = (
                                cw20.flat_actor(
                                    parameters, cw22.EXPECTED_ACTOR_NAMES, np
                                )
                                - cw11_flat
                            )
                            actual_endpoint_l2 = float(np.linalg.norm(actual_endpoint))
                            actual_displacement = actual_endpoint - current_x
                            actual_exp_slack = matrix1 @ actual_displacement - rhs1
                            actual_exp_checks = {
                                "actual_endpoint_finite": bool(
                                    np.isfinite(actual_endpoint).all()
                                ),
                                "actual_endpoint_radius_close_to_R_real": abs(
                                    actual_endpoint_l2 - R_real
                                ) <= V9_R_REAL_TOL,
                                "actual_endpoint_inside_total_cap": actual_endpoint_l2
                                < TOTAL_RADIUS_CAP,
                                "actual_float32_exp_replays_same_point_constraints": float(
                                    np.min(actual_exp_slack)
                                ) >= -V9_LINEAR_TOL,
                                "actual_endpoint_new": engine.array_sha(actual_endpoint)
                                != engine.array_sha(current_x),
                            }
                            actual_exp_certificate = {
                                "actual_endpoint_l2": actual_endpoint_l2,
                                "actual_endpoint_float64_le_sha256": engine.array_sha(
                                    actual_endpoint
                                ),
                                "actual_minimum_same_point_normalized_slack": float(
                                    np.min(actual_exp_slack)
                                ),
                                "checks": actual_exp_checks,
                                "pass": all(actual_exp_checks.values()),
                            }
                            if not actual_exp_certificate["pass"]:
                                terminal_reason = (
                                    "V9_FLOAT32_EXP_MAP_PRE_FORWARD_CERTIFICATE_FAILED"
                                )
                                cw20.apply_flat_actor(
                                    parameters,
                                    cw22.EXPECTED_ACTOR_NAMES,
                                    cw11_flat + current_x,
                                    torch,
                                )
                                restored_center = (
                                    cw20.flat_actor(
                                        parameters, cw22.EXPECTED_ACTOR_NAMES, np
                                    )
                                    - cw11_flat
                                )
                                restore_checks = {
                                    "actual_midpoint_restored_before_forward": bool(
                                        np.array_equal(restored_center, current_x)
                                    )
                                }
                                if not all(restore_checks.values()):
                                    raise ProtocolError(
                                        f"v9 tangent pre-forward restore drift: {restore_checks}"
                                    )
                                iteration_log.append(
                                    {
                                        "trial_one_based": 2,
                                        "candidate_evaluated": False,
                                        "shadow_role": "same_point_spherical_tangent_correction",
                                        "reason": terminal_reason,
                                        "tangent_certificate": tangent_certificate,
                                        "exp_map_certificate": exp_map_certificate,
                                        "actual_exp_certificate": actual_exp_certificate,
                                        "restore_checks": restore_checks,
                                    }
                                )
                            else:
                                with torch.no_grad():
                                    endpoint_outputs = ppo.model_forward(
                                        model, batch, device
                                    )
                                endpoint_state = evaluate_outputs(
                                    endpoint_outputs, actual_endpoint
                                )
                                trial_evaluations = 2
                                endpoint_new_pairs = []
                                for item in discover_failure_pairs(endpoint_state):
                                    if add_pair(**item):
                                        endpoint_new_pairs.append(item["name"])
                                cutting_plane_additions += len(endpoint_new_pairs)
                                endpoint_gate_pass = all(
                                    endpoint_state["train_checks"].values()
                                ) and not endpoint_new_pairs
                                iteration_log.append(
                                    {
                                        "trial_one_based": 2,
                                        "candidate_evaluated": True,
                                        "shadow_role": "same_point_spherical_tangent_correction",
                                        "alpha": 1.0,
                                        "radius_scans": 0,
                                        "weight_scans": 0,
                                        "planned_trial_l2": float(
                                            np.linalg.norm(planned_endpoint)
                                        ),
                                        "actual_trial_l2": actual_endpoint_l2,
                                        "actual_trial_x_float64_le_sha256": engine.array_sha(
                                            actual_endpoint
                                        ),
                                        "linearization_point_l2": center_l2,
                                        "linearization_to_candidate_l2": float(
                                            np.linalg.norm(actual_endpoint - current_x)
                                        ),
                                        "trial_merit": float(endpoint_state["merit"]),
                                        "merit_reduction": float(
                                            current_state["merit"]
                                            - endpoint_state["merit"]
                                        ),
                                        "PF_repairs": {
                                            name: bool(
                                                endpoint_state["target_observed"][name][
                                                    "ordered_correct"
                                                ]
                                            )
                                            for name in PF_FINAL_NATIVE_MARGIN
                                        },
                                        "top1_flip_identities": endpoint_state[
                                            "top1_flips"
                                        ],
                                        "retention_flip_identities": endpoint_state[
                                            "retention_flips"
                                        ],
                                        "new_pair_names": endpoint_new_pairs,
                                        "accepted": endpoint_gate_pass,
                                        "gate_pass": endpoint_gate_pass,
                                        "tangent_certificate": tangent_certificate,
                                        "exp_map_certificate": exp_map_certificate,
                                        "actual_exp_certificate": actual_exp_certificate,
                                    }
                                )
                                current_x = actual_endpoint
                                current_state = endpoint_state
                                if endpoint_gate_pass:
                                    accepted_iterates = 1
                                    terminal_reason = (
                                        "ALL_TARGETED_TRAIN_GATES_PASSED"
                                    )
                                else:
                                    rejection_count += 1
                                    terminal_reason = (
                                        "V9_SECOND_AND_FINAL_TRAIN_SHADOW_FAILED"
                                    )

        if trial_evaluations > V9_MAX_NEW_TRAIN_SHADOWS:
            raise ProtocolError("v9 new train-shadow cap exceeded")
        v9_terminal_pass = (
            terminal_reason == "ALL_TARGETED_TRAIN_GATES_PASSED"
            and trial_evaluations in (1, 2)
            and all(current_state["train_checks"].values())
        )
'''


def v9_v4_transform(raw_v4: bytes) -> tuple[bytes, dict[str, Any]]:
    source = raw_v4.decode()
    audit: list[dict[str, Any]] = []
    loop_start = "        current_x = np.zeros_like(cw11_flat, dtype=np.float64)\n"
    loop_end = '        passed = all(current_state["train_checks"].values())\n'
    replacement = v9_loop_source()
    source = splice_region_once(
        source,
        loop_start,
        loop_end,
        replacement,
        "replace_v4_iterative_loop_with_fixed_v9_midpoint_and_tangent",
        audit,
    )
    source = splice_exact_once(
        source,
        '        passed = all(current_state["train_checks"].values())\n',
        "        passed = v9_terminal_pass\n",
        "bind_v4_terminal_payload_gate_to_v9_terminal_certificate",
        audit,
    )
    transformed = source.encode()
    checks = {
        "two_exact_transformations": len(audit) == 2,
        "fixed_equal_midpoint": source.count("midpoint_sum = unit_v1 + unit_v6") == 1,
        "fixed_R_safe_reference": "V9_R_SAFE" in source,
        "tangent_projection_present": "projected_rows = matrix1 - np.outer" in source,
        "planned_exp_slack_replay_present": "planned_exp_slack = matrix1 @ planned_displacement - rhs1" in source,
        "actual_exp_slack_replay_present": "actual_exp_slack = matrix1 @ actual_displacement - rhs1" in source,
        "no_half_BF16_buffer": ("BUFFER_" + "FRACTION") not in source,
        "no_original_while_loop": "while trial_evaluations < MAX_TRIAL_EVALUATIONS" not in source,
        "terminal_payload_gate_bound_to_v9": source.count(
            "        passed = v9_terminal_pass\n"
        ) == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v9 transformed-v4 static drift: {checks}")
    return transformed, {
        "frozen_source_sha256": hashlib.sha256(raw_v4).hexdigest(),
        "transformed_source_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_source_bytes": len(transformed),
        "transformations": audit,
        "checks": checks,
    }


def verify_locked_inputs_stable(
    before_blobs: Mapping[str, bytes],
    before_evidence: Mapping[str, Any],
    after_blobs: Mapping[str, bytes],
    after_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "names_exact": set(before_blobs) == set(after_blobs),
        "bytes_exact": set(before_blobs) == set(after_blobs)
        and all(before_blobs[name] == after_blobs[name] for name in before_blobs),
        "device_inode_exact": set(before_evidence) == set(after_evidence)
        and all(
            (
                before_evidence[name]["device"],
                before_evidence[name]["inode"],
            )
            == (
                after_evidence[name]["device"],
                after_evidence[name]["inode"],
            )
            for name in before_evidence
        ),
        "sha_mode_checks_repeat": all(
            all(item["checks"].values()) for item in after_evidence.values()
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"frozen v1/v4/v6/v8 inputs changed across capture: {checks}")
    return {"checks": checks, "pass": True}


def production_run() -> dict[str, Any]:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError("CW24 v9 must run in my_project_env")
    if OUTPUT.exists():
        raise ProtocolError("frozen CW24 v9 output already exists")

    source = self_evidence(require_frozen=True)
    before_blobs, before_evidence = locked_inputs()

    # These are intentionally two process-lifetime-isolated historical replays.
    # Neither replay is a v9 candidate shadow and neither writes a file.
    v1_x, v1_capture = capture_subprocess("--internal-capture-v1", V1_DELTA_SHA256)
    v6_x, v6_capture = capture_subprocess("--internal-capture-v6", V6_S2_DELTA_SHA256)

    after_blobs, after_evidence = locked_inputs()
    stability = verify_locked_inputs_stable(
        before_blobs, before_evidence, after_blobs, after_evidence
    )
    transformed_v4, transform_evidence = v9_v4_transform(after_blobs["v4_source"])
    v4 = load_module_from_bytes(
        "cw24_v9_single_parent_v4_candidate_flow",
        V4_SOURCE,
        transformed_v4,
        {
            "V9_V1_X": v1_x,
            "V9_V6_X": v6_x,
            "V9_V1_SHA256": V1_DELTA_SHA256,
            "V9_V6_SHA256": V6_S2_DELTA_SHA256,
            "V9_R_SAFE": R_SAFE,
            "V9_R_REAL_TOL": R_REAL_TOL,
            "V9_TANGENT_STEP_CAP": TANGENT_STEP_CAP,
            "V9_LINEAR_TOL": LINEAR_TOL,
            "V9_MAX_NEW_TRAIN_SHADOWS": MAX_NEW_TRAIN_SHADOWS,
        },
    )
    v4.SEED = SEED
    result = v4.production_run()

    endpoint = result["endpoint"]
    scaffold_decision = str(endpoint.get("decision"))
    scaffold_reason = str(endpoint.get("reason"))
    train_shadow_count = int(endpoint.get("changed_candidate_train_shadow_count", 0))
    candidate_payload = endpoint.get("candidate_payload")
    trial_pass = endpoint.get("trial", {}).get("pass") is True
    terminal_reason = endpoint.get("cuttingplane", {}).get("terminal_reason")
    v9_pass = (
        scaffold_decision == "GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE"
        and trial_pass
        and terminal_reason == "ALL_TARGETED_TRAIN_GATES_PASSED"
        and endpoint.get("promotable_terminal_endpoint_count") == 1
        and candidate_payload is not None
        and 1 <= train_shadow_count <= MAX_NEW_TRAIN_SHADOWS
    )
    payload_gate = (candidate_payload is not None) == v9_pass
    contract_checks = {
        "historical_v1_replay_isolated": v1_capture["subprocess"]["isolated"] is True,
        "historical_v6_replay_isolated": v6_capture["subprocess"]["isolated"] is True,
        "v1_child_and_parent_vector_SHA_locks": v1_capture["capture_checks"][
            "sha_exact"
        ]
        is True
        and v1_capture["subprocess"]["parent_vector_checks"]["decoded_sha_exact"]
        is True,
        "v6_S2_child_and_parent_vector_SHA_locks": v6_capture["capture_checks"][
            "sha_exact"
        ]
        is True
        and v6_capture["vector_float64_le_sha256"] == V6_S2_DELTA_SHA256
        and v6_capture["subprocess"]["parent_vector_checks"]["decoded_sha_exact"]
        is True,
        "child_stdout_canonical_byte_exact": v1_capture["subprocess"][
            "stdout_checks"
        ]["canonical_payload_exactly_stdout"]
        is True
        and v6_capture["subprocess"]["stdout_checks"][
            "canonical_payload_exactly_stdout"
        ]
        is True,
        "historical_replays_zero_new_v9_shadows": int(
            v1_capture["new_v9_train_shadows"]
        )
        == 0
        and int(v6_capture["new_v9_train_shadows"]) == 0,
        "historical_replays_zero_writes": int(v1_capture["writes_performed"]) == 0
        and int(v6_capture["writes_performed"]) == 0,
        "historical_replay_counts_disclosed_as_one_plus_two": int(
            v1_capture["historical_replay_changed_candidate_train_shadow_count"]
        )
        == 1
        and int(v6_capture["historical_replay_changed_candidate_train_shadow_count"])
        == 2,
        "new_train_shadow_count_within_cap": 0
        <= train_shadow_count
        <= MAX_NEW_TRAIN_SHADOWS,
        "terminal_only_payload_gate": payload_gate,
        "GO_requires_one_or_two_new_train_shadows": (not v9_pass)
        or 1 <= train_shadow_count <= MAX_NEW_TRAIN_SHADOWS,
        "frozen_total_radius_cap_unchanged": float(v4.TOTAL_RADIUS_CAP)
        == TOTAL_RADIUS_CAP,
        "fixed_R_safe_strictly_inside_cap": R_SAFE < TOTAL_RADIUS_CAP,
        "tangent_step_cap_exact": TANGENT_STEP_CAP == 0.00025,
        "parent_single_v4_candidate_production_flow": True,
    }
    if not all(contract_checks.values()):
        raise ProtocolError(f"v9 terminal contract drift: {contract_checks}")

    decision = (
        "GO_CW24_V9_V1_V6_SPHERICAL_MIDPOINT_TRAIN_GATE"
        if v9_pass
        else "NO_GO_CW24_V9_V1_V6_SPHERICAL_MIDPOINT_TRAIN_GATE"
    )
    reason = (
        "SOLE_V9_TERMINAL_ENDPOINT_PASSED_UNCHANGED_V4_TRAIN_GATES"
        if v9_pass
        else (terminal_reason or scaffold_reason)
    )
    endpoint["v4_scaffold_decision"] = scaffold_decision
    endpoint["v4_scaffold_reason"] = scaffold_reason
    endpoint["decision"] = decision
    endpoint["reason"] = reason

    if candidate_payload is not None:
        candidate_payload["v9_selection_formula"] = (
            "fixed_equal_spherical_midpoint_of_frozen_v1_failed_actual_delta_and_"
            "frozen_v6_rejected_S2_actual_delta_at_R_safe; optional_single_same_point_"
            "tangent_QP_exp_map_correction_at_float32_realized_radius"
        )
        candidate_payload["v9_historical_anchor_float64_le_sha256"] = {
            "frozen_v1_failed_actual_delta": V1_DELTA_SHA256,
            "frozen_v6_rejected_S2_actual_delta": V6_S2_DELTA_SHA256,
        }
        candidate_payload["v9_R_safe"] = R_SAFE
        candidate_payload["v9_no_interpolation_or_radius_scan"] = True

    result["schema_version"] = SCHEMA
    result["seed"] = SEED
    result["status"] = decision
    result["decision"] = decision
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_fixed_v1_v6_spherical_"
        "midpoint_optional_tangent_special_BC"
    )
    result["selection"]["optimization_contract"] = {
        "frozen_v4_B352_train_gates_unchanged": True,
        "historical_calibration_replays_are_not_new_train_shadows": True,
        "fixed_equal_spherical_midpoint_only": True,
        "interpolation_weight_scans": 0,
        "radius_scans": 0,
        "R_safe": R_SAFE,
        "R_real_tolerance": R_REAL_TOL,
        "total_additional_from_CW11_cap": TOTAL_RADIUS_CAP,
        "pre_candidate_linear_certificate_required": True,
        "pre_candidate_failure_means_zero_new_train_shadows": True,
        "midpoint_is_only_first_new_train_shadow": True,
        "optional_same_point_tangent_QP_correction_is_only_second_shadow": True,
        "tangent_step_l2_cap": TANGENT_STEP_CAP,
        "tangent_QP_KKT_certificate_required": True,
        "planned_and_float32_exp_endpoint_constraint_replay_required": True,
        "maximum_new_train_shadows": MAX_NEW_TRAIN_SHADOWS,
        "third_candidate_forbidden": True,
        "half_BF16_buffer_forbidden": True,
        "single_promotable_terminal_endpoint": True,
        "payload_only_after_all_terminal_train_gates": True,
    }
    result["historical_exact_CW11_replay"].pop(
        "new_validation_rows_opened_for_CW24_v4_selection_or_candidate", None
    )
    result["historical_exact_CW11_replay"][
        "new_validation_rows_opened_for_CW24_v9_selection_or_candidate"
    ] = 0
    result["historical_calibration_replays"] = {
        "frozen_v1_failed_actual_delta": v1_capture,
        "frozen_v6_rejected_second_shadow_actual_delta": v6_capture,
        "combined_new_v9_train_shadow_count": 0,
        "transport": "two_separate_sys_executable_dash_I_dash_B_dash_c_stdout_zlib_base64",
        "parent_process_reused_historical_models": False,
    }
    result["inputs"]["CW24_v9_source"] = source
    result["inputs"]["CW24_v9_frozen_v1_v4_v6_v8_sources_and_results"] = after_evidence
    result["inputs"]["CW24_v9_input_stability_across_isolated_captures"] = stability
    result["inputs"]["CW24_v9_transformed_frozen_v4_scaffold"] = transform_evidence
    result["audit"]["CW24_v9_contract"] = {
        "checks": contract_checks,
        "pass": all(contract_checks.values()),
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "train_only_B352": True,
        "new_validation_or_test_rows_opened": 0,
        "historical_replay_new_train_shadows": 0,
        "changed_candidate_train_shadow_count": train_shadow_count,
        "maximum_new_train_shadows": MAX_NEW_TRAIN_SHADOWS,
        "checkpoint_reads_beyond_frozen_v4_scaffold": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "network_calls": 0,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "v9_recipe_fixed_before_new_candidate_forward": True,
        "frozen_v1_and_v6_replays_are_historical_calibration_only": True,
        "historical_replays_are_process_lifetime_isolated": True,
        "midpoint_and_optional_tangent_are_train_only_exploration": True,
        "exploratory_train_results_are_not_promotion_evidence": True,
        "fulltrain_exact_CW11_comparator_required_after_GO": True,
        "specialist_then_broad_then_Gold_required_after_fulltrain_GO": True,
    }
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 2
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    canonical_json(result)
    return result


def static_audit() -> dict[str, Any]:
    blobs, evidence = locked_inputs()
    v1_transformed, v1_transform = v1_capture_transform(blobs["v1_source"])
    v6_transformed, v6_transform = v6_capture_transform(blobs["v4_source"])
    v9_transformed, v9_transform = v9_v4_transform(blobs["v4_source"])
    compile(v1_transformed, str(V1_SOURCE), "exec", dont_inherit=True)
    compile(v6_transformed, str(V4_SOURCE), "exec", dont_inherit=True)
    compile(v9_transformed, str(V4_SOURCE), "exec", dont_inherit=True)
    checks = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "source_self_frozen": all(self_evidence(require_frozen=True)["checks"].values()),
        "all_eight_frozen_inputs_locked": len(evidence) == 8
        and all(all(item["checks"].values()) for item in evidence.values()),
        "three_transformed_sources_compile": True,
        "output_absent": not OUTPUT.exists(),
        "CUDA_initialized": False,
        "validation_or_test_rows_opened": 0,
        "writes_performed": 0,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v9 static audit failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_pass",
        "source": self_evidence(require_frozen=True),
        "frozen_inputs": evidence,
        "transforms": {
            "v1_historical_capture": v1_transform,
            "v6_historical_capture": v6_transform,
            "v9_frozen_v4_scaffold": v9_transform,
        },
        "variant": {
            "R_safe": R_SAFE,
            "R_real_tolerance": R_REAL_TOL,
            "total_radius_cap": TOTAL_RADIUS_CAP,
            "tangent_step_cap": TANGENT_STEP_CAP,
            "maximum_new_train_shadows": MAX_NEW_TRAIN_SHADOWS,
            "interpolation_weight_scans": 0,
            "radius_scans": 0,
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--internal-capture-v1", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--internal-capture-v6", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.internal_capture_v1 or args.internal_capture_v6:
        if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
            raise ProtocolError("isolated replay is outside my_project_env")
        self_evidence(require_frozen=True)
        payload = internal_capture_v1() if args.internal_capture_v1 else internal_capture_v6()
        print(canonical_json(payload).decode(), end="")
        return

    if args.audit_only:
        print(canonical_json(static_audit()).decode(), end="")
        return

    if OUTPUT.exists():
        raise ProtocolError("frozen CW24 v9 output already exists")
    result = production_run()
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "candidate_model_state_sha256": result["endpoint"].get(
                    "trial", {}
                ).get("candidate_model_state_sha256"),
                "candidate_payload_present": result["endpoint"].get(
                    "candidate_payload"
                )
                is not None,
                "promotable_terminal_endpoint_count": result["endpoint"].get(
                    "promotable_terminal_endpoint_count"
                ),
                "new_train_shadow_count": result["endpoint"].get(
                    "changed_candidate_train_shadow_count"
                ),
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
                "output": publication,
            }
        ).decode(),
        end="",
    )


if __name__ == "__main__":
    main()

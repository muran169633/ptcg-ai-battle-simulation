#!/usr/bin/env python3
"""CW24 v8 absolute-residual special-BC probe over frozen v4.

V8 preserves the frozen v4 0.00025 first shadow and the frozen-v6 unbuffered
full second shadow.  A failed second shadow is retained only as a train-only
calibration point.  Its same-point native-BF16 residual tangents are rewritten
as constraints on the absolute actor delta from exact CW11, and the existing
certified Hildreth solver finds the minimum-radius absolute delta.  No endpoint
is radius-scaled.  One further residual correction is allowed only after strict
native-signature/merit/preservation progress, for at most four train shadows.
All frozen v4 actual gates, replay checks, and terminal-only payload rules stay
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_absolute_residual_specialbc_cw24_v8.py"

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

V7_SOURCE = TOOLS / "probe_u468_cw11_bf16buffer_postaccept_specialbc_cw24_v7.py"
V7_SOURCE_SHA256 = "df89b97e843aa7ec31261f3d43eb7c2065f1e9bce38a6cca902230cb6f6d0976"
V7_SOURCE_MODE = 0o555
V7_RESULT = ROOT / "artifacts/cw24_cw11_bf16buffer_postaccept_specialbc_trainonly_v7.json"
V7_RESULT_SHA256 = "3e74b75d4570884c96b54504ab8cd444b6dcdd5eeb6ff0332923ee8fa8e03ccd"
V7_RESULT_MODE = 0o444

OUTPUT = ROOT / "artifacts/cw24_cw11_absolute_residual_specialbc_trainonly_v8.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-absolute-residual-specialbc-cw24-v8"
SEED = 202608048
INITIAL_TRUST_RADIUS = 2.5e-4
TOTAL_RADIUS_CAP = 9.9998e-4
MAX_TRAIN_SHADOWS = 4
MAX_RESIDUAL_CORRECTIONS = 2


class ProtocolError(RuntimeError):
    """Fail-closed CW24 v8 wrapper error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "three_frozen_source_locks": all(
            digest.encode() in source
            for digest in (V4_SOURCE_SHA256, V6_SOURCE_SHA256, V7_SOURCE_SHA256)
        ),
        "three_frozen_result_locks": all(
            digest.encode() in source
            for digest in (V4_RESULT_SHA256, V6_RESULT_SHA256, V7_RESULT_SHA256)
        ),
        "shadow_cap_exact": MAX_TRAIN_SHADOWS == 4
        and b"MAX_TRAIN_SHADOWS = 4" in source,
        "residual_correction_cap_exact": MAX_RESIDUAL_CORRECTIONS == 2
        and b"MAX_RESIDUAL_CORRECTIONS = 2" in source,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "no_network_calls": all(
            token not in source
            for token in (b"requests" + b".", b"urllib" + b".", b"socket" + b".")
        ),
        "no_payload_construction": (b"candidate_actor_" + b"float32_le") not in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v8 source audit failed: {checks}")
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
                raise ProtocolError("short v8 result write")
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
        raise ProtocolError(f"v8 publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def splice_exact_once(
    source: str, old: str, new: str, label: str, audits: list[dict[str, Any]]
) -> str:
    occurrences = source.count(old)
    if occurrences != 1:
        raise ProtocolError(f"{label}: expected one frozen-v4 splice, observed {occurrences}")
    audits.append(
        {
            "label": label,
            "kind": "exact_string_once",
            "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
            "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
            "occurrences": occurrences,
        }
    )
    return source.replace(old, new, 1)


def splice_region_once(
    source: str,
    start: str,
    end: str,
    replacement: str,
    label: str,
    audits: list[dict[str, Any]],
) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise ProtocolError(f"{label}: frozen-v4 region markers are not unique")
    begin = source.index(start)
    finish = source.index(end, begin)
    old = source[begin:finish]
    audits.append(
        {
            "label": label,
            "kind": "unique_marker_region",
            "start_sha256": hashlib.sha256(start.encode()).hexdigest(),
            "end_sha256": hashlib.sha256(end.encode()).hexdigest(),
            "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
            "new_sha256": hashlib.sha256(replacement.encode()).hexdigest(),
            "occurrences": 1,
        }
    )
    return source[:begin] + replacement + source[finish:]


def transformed_v4_module() -> tuple[ModuleType, dict[str, Any]]:
    raw, source_evidence = regular_source(
        V4_SOURCE, V4_SOURCE_SHA256, V4_SOURCE_MODE, "frozen CW24 v4 source"
    )
    source = raw.decode()
    audits: list[dict[str, Any]] = []

    signature_old = (
        "        def build_linearized_constraints() -> "
        "tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:\n"
    )
    signature_new = (
        "        def build_linearized_constraints(absolute_current_x: Any | None = None) -> "
        "tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:\n"
    )
    source = splice_exact_once(
        source, signature_old, signature_new, "absolute_constraint_signature", audits
    )

    rhs_old = '''            matrix = np.stack(rows_normalized, axis=0)
            rhs = np.asarray(rhs_normalized, dtype=np.float64)
            audit = {
                "constraint_count": len(details),
                "pair_constraint_count": len(pair_specs),
                "aggregate_constraint_count": len(LOSS_TARGETS),
                "constraint_semantics": (
                    "persistent active identities, current-point tangent only; "
                    "no global anchor-tangent retention claim"
                ),
                "graph_no_grad_same_point_checks": graph_snapshot_checks,
                "matrix_shape": list(matrix.shape),
                "matrix_float64_le_sha256": engine.array_sha(matrix),
                "rhs_float64_le_sha256": engine.array_sha(rhs),
                "constraints": details,
            }
            return matrix, rhs, details, audit
'''
    rhs_new = '''            matrix = np.stack(rows_normalized, axis=0)
            rhs_delta = np.asarray(rhs_normalized, dtype=np.float64)
            if absolute_current_x is None:
                rhs = rhs_delta
                constraint_coordinate = "incremental_step"
                absolute_audit = None
            else:
                absolute_array = np.asarray(absolute_current_x, dtype=np.float64)
                if absolute_array.shape != (matrix.shape[1],):
                    raise ProtocolError("absolute residual current-x shape drift")
                if not bool(np.isfinite(absolute_array).all()):
                    raise ProtocolError("absolute residual current-x nonfinite")
                rhs = rhs_delta + matrix @ absolute_array
                constraint_coordinate = "absolute_CW11_delta"
                absolute_audit = {
                    "current_x_l2": float(np.linalg.norm(absolute_array)),
                    "current_x_float64_le_sha256": engine.array_sha(absolute_array),
                    "rhs_formula": "rhs_delta_plus_matrix_matmul_current_x",
                    "rhs_delta_float64_le_sha256": engine.array_sha(rhs_delta),
                }
            audit = {
                "constraint_count": len(details),
                "pair_constraint_count": len(pair_specs),
                "aggregate_constraint_count": len(LOSS_TARGETS),
                "constraint_semantics": (
                    "persistent active identities, current-point tangent only; "
                    "no global anchor-tangent retention claim"
                ),
                "constraint_coordinate": constraint_coordinate,
                "absolute_residual": absolute_audit,
                "graph_no_grad_same_point_checks": graph_snapshot_checks,
                "matrix_shape": list(matrix.shape),
                "matrix_float64_le_sha256": engine.array_sha(matrix),
                "rhs_float64_le_sha256": engine.array_sha(rhs),
                "constraints": details,
            }
            return matrix, rhs, details, audit
'''
    source = splice_exact_once(
        source, rhs_old, rhs_new, "absolute_rhs_from_same_point_residual", audits
    )

    loop_start = "        current_x = np.zeros_like(cw11_flat, dtype=np.float64)\n"
    loop_end = '        passed = all(current_state["train_checks"].values())\n'
    loop_new = '''        def native_residual_signature(state: Mapping[str, Any]) -> str:
            value = {
                "pair_margins": state["pair_margins"],
                "target_observed": state["target_observed"],
                "zero_guard_observed": state["zero_guard_observed"],
                "top1_flips": state["top1_flips"],
                "retention_flips": state["retention_flips"],
                "hard_repairs": state["hard_repairs"],
                "train_checks": state["train_checks"],
            }
            return hashlib.sha256(canonical_json(value)).hexdigest()

        def register_discovered(state: Mapping[str, Any]) -> list[str]:
            nonlocal cutting_plane_additions
            names = []
            for item in discover_failure_pairs(state):
                if add_pair(**item):
                    names.append(item["name"])
            cutting_plane_additions += len(names)
            return names

        def apply_candidate_actor(candidate_x: Any) -> Any:
            cw20.apply_flat_actor(
                parameters,
                cw22.EXPECTED_ACTOR_NAMES,
                cw11_flat + candidate_x,
                torch,
            )
            return cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np) - cw11_flat

        def restore_current_actor(expected_x: Any) -> None:
            restored_x = apply_candidate_actor(expected_x)
            if not bool(np.array_equal(restored_x, expected_x)):
                raise ProtocolError("v8 pre-forward fail-close actor restore drift")

        def prepare_candidate_before_forward(
            candidate_x: Any,
            restore_x: Any,
            previous_sha256: set[str],
        ) -> tuple[Any, str, dict[str, Any] | None]:
            actual_x = apply_candidate_actor(candidate_x)
            actual_sha = engine.array_sha(actual_x)
            actual_l2 = float(np.linalg.norm(actual_x))
            reason = None
            if actual_l2 > TOTAL_RADIUS_CAP:
                reason = "FLOAT32_ACTOR_COPY_EXCEEDS_UNCHANGED_CW11_RADIUS"
            elif actual_sha in previous_sha256:
                reason = "FLOAT32_ACTOR_COPY_REPEATS_EVALUATED_SHADOW"
            if reason is not None:
                restore_current_actor(restore_x)
                return actual_x, actual_sha, {
                    "reason": reason,
                    "actual_l2": actual_l2,
                    "actual_x_float64_le_sha256": actual_sha,
                    "model_restored_to_current_x": True,
                    "forward_performed": False,
                }
            return actual_x, actual_sha, None

        def evaluate_applied_actor(actual_x: Any) -> dict[str, Any]:
            with torch.no_grad():
                outputs = ppo.model_forward(model, batch, device)
            return evaluate_outputs(outputs, actual_x)

        current_x = np.zeros_like(cw11_flat, dtype=np.float64)
        with torch.no_grad():
            current_outputs = ppo.model_forward(model, batch, device)
        current_state = evaluate_outputs(current_outputs, current_x)
        trust_radius = TRUST_RADIUS_INITIAL
        accepted_iterates = 0
        trial_evaluations = 0
        rejection_count = 0
        cutting_plane_additions = 0
        iteration_log: list[dict[str, Any]] = []
        terminal_solver_audit: dict[str, Any] | None = None
        terminal_reason = "V8_SHADOW_PROTOCOL_NOT_STARTED"
        evaluated_x_sha256: set[str] = set()

        # S1 is the frozen-v4 0.00025 stepping stone, including its original
        # scaled trust trial and original acceptance predicate.
        matrix, rhs, _, linearization_audit = build_linearized_constraints()
        step, solver_audit = dual_hildreth_minimum_norm(matrix, rhs, np)
        if not solver_audit["certified"]:
            terminal_solver_audit = solver_audit
            terminal_reason = "S1_CURRENT_TANGENT_SOLVER_UNCERTIFIED"
            iteration_log.append(
                {
                    "trial_one_based": 1,
                    "candidate_evaluated": False,
                    "shadow_role": "frozen_v4_stepping_stone",
                    "linearization": linearization_audit,
                    "solver": solver_audit,
                }
            )
        else:
            step_l2 = float(np.linalg.norm(step))
            alpha_trust = min(1.0, trust_radius / step_l2)
            alpha_global = min(
                1.0, maximum_radius_alpha(current_x, step, TOTAL_RADIUS_CAP, np)
            )
            alpha = min(alpha_trust, alpha_global)
            if alpha <= 1.0e-12:
                terminal_reason = "S1_ZERO_ALPHA"
            else:
                planned_x = current_x + alpha * step
                actual_x, actual_sha, preforward_failure = prepare_candidate_before_forward(
                    planned_x, current_x, evaluated_x_sha256
                )
                if preforward_failure is not None:
                    terminal_reason = "S1_" + str(preforward_failure["reason"])
                    iteration_log.append(
                        {
                            "trial_one_based": 1,
                            "candidate_evaluated": False,
                            "shadow_role": "frozen_v4_stepping_stone",
                            "reason": terminal_reason,
                            "preforward_failure": preforward_failure,
                            "linearization": linearization_audit,
                            "solver": solver_audit,
                        }
                    )
                else:
                    trial_state = evaluate_applied_actor(actual_x)
                    trial_evaluations += 1
                    evaluated_x_sha256.add(actual_sha)
                    new_pair_names = register_discovered(trial_state)
                    pf_repairs = {
                        name: bool(trial_state["target_observed"][name]["ordered_correct"])
                        for name in PF_FINAL_NATIVE_MARGIN
                    }
                    preservation_failed = bool(
                        trial_state["top1_flips"] or trial_state["retention_flips"]
                    )
                    immutable_failed = not all(trial_state["immutable_native"].values())
                    merit_reduction = float(current_state["merit"] - trial_state["merit"])
                    gate_pass = all(trial_state["train_checks"].values())
                    accept = (
                        not preservation_failed
                        and not immutable_failed
                        and not new_pair_names
                        and (gate_pass or merit_reduction > MERIT_TOL)
                    )
                    planned_l2 = float(np.linalg.norm(planned_x))
                    actual_l2 = float(np.linalg.norm(actual_x))
                    s1_fingerprint_checks = {
                        "matrix_sha256": linearization_audit[
                            "matrix_float64_le_sha256"
                        ] == "b39d640e398ea5a1d89a958b549afda04be460c6e4eb192daafc643f31984d74",
                        "rhs_sha256": linearization_audit[
                            "rhs_float64_le_sha256"
                        ] == "d206d5bd394982a1bd993395ea090d0675ac5fa32184244895961970fb8ab458",
                        "solver_step_l2": math.isclose(
                            float(solver_audit["step_l2"]), 0.0009457176289065079,
                            rel_tol=0.0, abs_tol=5.0e-15,
                        ),
                        "planned_l2": math.isclose(
                            planned_l2, 0.00024999999999999995,
                            rel_tol=0.0, abs_tol=5.0e-13,
                        ),
                        "actual_l2": math.isclose(
                            actual_l2, 0.00025000002682303664,
                            rel_tol=0.0, abs_tol=5.0e-13,
                        ),
                        "current_merit": math.isclose(
                            float(current_state["merit"]), 22.0002,
                            rel_tol=0.0, abs_tol=5.0e-10,
                        ),
                        "trial_merit": math.isclose(
                            float(trial_state["merit"]), 12.393708709452767,
                            rel_tol=0.0, abs_tol=5.0e-9,
                        ),
                        "merit_reduction": math.isclose(
                            merit_reduction, 9.606491290547233,
                            rel_tol=0.0, abs_tol=5.0e-9,
                        ),
                        "zero_top1_flips": len(trial_state["top1_flips"]) == 0,
                        "zero_retention_flips": len(trial_state["retention_flips"]) == 0,
                        "PF_repairs_exact": pf_repairs == {
                            "pf0_boundary_a": False,
                            "pf0_boundary_b": False,
                            "pf7_boundary": False,
                        },
                        "no_new_pair_names": not new_pair_names,
                        "accepted_nonterminal": accept and not gate_pass,
                    }
                    if not all(s1_fingerprint_checks.values()):
                        raise ProtocolError(
                            f"S1 frozen v6/v7 fingerprint drift: {s1_fingerprint_checks}"
                        )
                    iteration_log.append(
                        {
                            "trial_one_based": 1,
                            "candidate_evaluated": True,
                            "shadow_role": "frozen_v4_stepping_stone",
                            "accepted_before_trial": 0,
                            "trust_radius": trust_radius,
                            "alpha": alpha,
                            "alpha_trust": alpha_trust,
                            "alpha_global": alpha_global,
                            "planned_trial_l2": planned_l2,
                            "actual_trial_l2": actual_l2,
                            "current_merit": float(current_state["merit"]),
                            "trial_merit": float(trial_state["merit"]),
                            "merit_reduction": merit_reduction,
                            "PF_repairs": pf_repairs,
                            "top1_flip_count": len(trial_state["top1_flips"]),
                            "retention_flip_count": len(trial_state["retention_flips"]),
                            "new_pair_names": new_pair_names,
                            "accepted": accept,
                            "gate_pass": gate_pass,
                            "frozen_v6_v7_fingerprint_checks": s1_fingerprint_checks,
                            "native_residual_signature": native_residual_signature(trial_state),
                            "linearization": linearization_audit,
                            "solver": solver_audit,
                        }
                    )
                    current_x = actual_x
                    current_state = trial_state
                    accepted_iterates = 1
                    terminal_reason = "S1_ACCEPTED_NONTERMINAL"

        # S2 is exactly the unbuffered full endpoint policy disclosed by v6.
        if accepted_iterates == 1 and not all(current_state["train_checks"].values()):
            trust_radius = TRUST_RADIUS_MAX
            matrix, rhs, _, linearization_audit = build_linearized_constraints()
            step, solver_audit = dual_hildreth_minimum_norm(matrix, rhs, np)
            if not solver_audit["certified"]:
                terminal_solver_audit = solver_audit
                terminal_reason = "S2_CURRENT_TANGENT_SOLVER_UNCERTIFIED"
                iteration_log.append(
                    {
                        "trial_one_based": 2,
                        "candidate_evaluated": False,
                        "shadow_role": "v6_unbuffered_full_calibration",
                        "linearization": linearization_audit,
                        "solver": solver_audit,
                    }
                )
            else:
                step_l2 = float(np.linalg.norm(step))
                planned_x = current_x + step
                planned_l2 = float(np.linalg.norm(planned_x))
                full_fits = step_l2 <= trust_radius and planned_l2 <= TOTAL_RADIUS_CAP
                if not full_fits:
                    terminal_reason = "S2_FULL_ENDPOINT_EXCEEDS_UNCHANGED_CW11_RADIUS"
                    iteration_log.append(
                        {
                            "trial_one_based": 2,
                            "candidate_evaluated": False,
                            "shadow_role": "v6_unbuffered_full_calibration",
                            "full_unscaled_step_l2": step_l2,
                            "full_unscaled_endpoint_l2": planned_l2,
                            "total_radius_cap": TOTAL_RADIUS_CAP,
                            "linearization": linearization_audit,
                            "solver": solver_audit,
                        }
                    )
                else:
                    actual_x, actual_sha, preforward_failure = prepare_candidate_before_forward(
                        planned_x, current_x, evaluated_x_sha256
                    )
                    if preforward_failure is not None:
                        terminal_reason = "S2_" + str(preforward_failure["reason"])
                        iteration_log.append(
                            {
                                "trial_one_based": 2,
                                "candidate_evaluated": False,
                                "shadow_role": "v6_unbuffered_full_calibration",
                                "reason": terminal_reason,
                                "preforward_failure": preforward_failure,
                                "linearization": linearization_audit,
                                "solver": solver_audit,
                            }
                        )
                    else:
                        trial_state = evaluate_applied_actor(actual_x)
                        trial_evaluations += 1
                        evaluated_x_sha256.add(actual_sha)
                        new_pair_names = register_discovered(trial_state)
                        gate_pass = all(trial_state["train_checks"].values())
                        pf_repairs = {
                            name: bool(
                                trial_state["target_observed"][name]["ordered_correct"]
                            )
                            for name in PF_FINAL_NATIVE_MARGIN
                        }
                        pf_set_repairs = {
                            name: bool(
                                trial_state["target_observed"][name]["set_correct"]
                            )
                            for name in PF_FINAL_NATIVE_MARGIN
                        }
                        pf_native_margins = {
                            name: float(
                                trial_state["target_observed"][name][
                                    "candidate_fixed_threat_margin"
                                ]
                            )
                            for name in PF_FINAL_NATIVE_MARGIN
                        }
                        actual_l2 = float(np.linalg.norm(actual_x))
                        s2_fingerprint_checks = {
                            "matrix_sha256": linearization_audit[
                                "matrix_float64_le_sha256"
                            ] == "9954fe13b421ab0eed6a355904d3cf27b5ef6dbc5eba8a9403969336ec9d668e",
                            "rhs_sha256": linearization_audit[
                                "rhs_float64_le_sha256"
                            ] == "e58ec7fa28887e984a323c8018943244dc2e73125e35b299a4c863d050d0af76",
                            "solver_step_l2": math.isclose(
                                float(solver_audit["step_l2"]), 0.000753542521139092,
                                rel_tol=0.0, abs_tol=5.0e-15,
                            ),
                            "planned_l2": math.isclose(
                                planned_l2, 0.0009909412795123538,
                                rel_tol=0.0, abs_tol=5.0e-13,
                            ),
                            "actual_l2": math.isclose(
                                actual_l2, 0.000990940375725956,
                                rel_tol=0.0, abs_tol=5.0e-13,
                            ),
                            "trial_merit": math.isclose(
                                float(trial_state["merit"]), 25123.250002966826,
                                rel_tol=0.0, abs_tol=5.0e-9,
                            ),
                            "PF_repairs_exact": pf_repairs == {
                                "pf0_boundary_a": True,
                                "pf0_boundary_b": True,
                                "pf7_boundary": False,
                            },
                            "PF_set_repairs_exact": pf_set_repairs == {
                                "pf0_boundary_a": True,
                                "pf0_boundary_b": True,
                                "pf7_boundary": False,
                            },
                            "PF_native_margins_exact": pf_native_margins == {
                                "pf0_boundary_a": 0.00390625,
                                "pf0_boundary_b": 0.001953125,
                                "pf7_boundary": -0.00390625,
                            },
                            "top1_flip_identity_exact": trial_state["top1_flips"] == [
                                "fde6fab074495c238d4c4ef42a2f12ec595da78e35836578f7a3828002037b88"
                            ],
                            "retention_flip_identity_exact": trial_state[
                                "retention_flips"
                            ] == [
                                "243f4a21d6c0b178a6306bdb63ac738aeb5bb13f6be0ccfaa68eebe02eac2d99"
                            ],
                            "no_new_pair_names": not new_pair_names,
                            "failed_train_gate": not gate_pass,
                        }
                        if not all(s2_fingerprint_checks.values()):
                            raise ProtocolError(
                                f"S2 frozen v6 fingerprint drift: {s2_fingerprint_checks}"
                            )
                        iteration_log.append(
                            {
                                "trial_one_based": 2,
                                "candidate_evaluated": True,
                                "shadow_role": "v6_unbuffered_full_calibration",
                                "accepted_before_trial": accepted_iterates,
                                "trust_radius": trust_radius,
                                "alpha": 1.0,
                                "alpha_trust": 1.0,
                                "alpha_global": 1.0,
                                "planned_trial_l2": planned_l2,
                                "actual_trial_l2": actual_l2,
                                "current_merit": float(current_state["merit"]),
                                "trial_merit": float(trial_state["merit"]),
                                "merit_reduction": float(
                                    current_state["merit"] - trial_state["merit"]
                                ),
                                "PF_repairs": pf_repairs,
                                "PF_set_repairs": pf_set_repairs,
                                "PF_native_margins": pf_native_margins,
                                "top1_flip_count": len(trial_state["top1_flips"]),
                                "retention_flip_count": len(trial_state["retention_flips"]),
                                "new_pair_names": new_pair_names,
                                "accepted": False,
                                "gate_pass": False,
                                "frozen_v6_fingerprint_checks": s2_fingerprint_checks,
                                "native_residual_signature": native_residual_signature(trial_state),
                                "linearization": linearization_audit,
                                "solver": solver_audit,
                            }
                        )
                        current_x = actual_x
                        current_state = trial_state
                        rejection_count += 1
                        terminal_reason = "S2_RETAINED_AS_CALIBRATION_POINT"

        # S3 and, under strict progress, S4 use same-point actual native
        # residuals rewritten onto the absolute CW11 actor-delta coordinate.
        for correction_one_based in (1, 2):
            if all(current_state["train_checks"].values()):
                break
            if trial_evaluations < 2:
                break
            if trial_evaluations >= MAX_TRIAL_EVALUATIONS:
                terminal_reason = "V8_MAX_TRAIN_SHADOWS_EXHAUSTED"
                break
            previous_x = current_x
            previous_state = current_state
            previous_signature = native_residual_signature(previous_state)
            previous_top1 = set(previous_state["top1_flips"])
            previous_retention = set(previous_state["retention_flips"])
            observed_linearization_x = (
                cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np) - cw11_flat
            )
            linearization_actor_exact = bool(
                np.array_equal(observed_linearization_x, previous_x)
            )
            if not linearization_actor_exact:
                raise ProtocolError(
                    "absolute residual linearization actor/current-x drift"
                )
            matrix, rhs_absolute, _, linearization_audit = build_linearized_constraints(
                previous_x
            )
            linearization_audit["prelinearization_actor"] = {
                "bit_exact_previous_x": linearization_actor_exact,
                "observed_x_l2": float(np.linalg.norm(observed_linearization_x)),
                "observed_x_float64_le_sha256": engine.array_sha(
                    observed_linearization_x
                ),
                "previous_x_float64_le_sha256": engine.array_sha(previous_x),
            }
            absolute_x, solver_audit = dual_hildreth_minimum_norm(
                matrix, rhs_absolute, np
            )
            trial_number = trial_evaluations + 1
            shadow_role = f"absolute_native_residual_correction_{correction_one_based}"
            if not solver_audit["certified"]:
                terminal_solver_audit = solver_audit
                terminal_reason = "ABSOLUTE_RESIDUAL_SOLVER_UNCERTIFIED"
                iteration_log.append(
                    {
                        "trial_one_based": trial_number,
                        "candidate_evaluated": False,
                        "shadow_role": shadow_role,
                        "reason": terminal_reason,
                        "linearization": linearization_audit,
                        "solver": solver_audit,
                    }
                )
                break
            planned_l2 = float(np.linalg.norm(absolute_x))
            candidate_sha = engine.array_sha(absolute_x)
            if planned_l2 > TOTAL_RADIUS_CAP:
                terminal_reason = "ABSOLUTE_MINIMUM_RADIUS_EXCEEDS_UNCHANGED_CW11_CAP"
                terminal_solver_audit = solver_audit
                iteration_log.append(
                    {
                        "trial_one_based": trial_number,
                        "candidate_evaluated": False,
                        "shadow_role": shadow_role,
                        "reason": terminal_reason,
                        "absolute_minimum_radius_l2": planned_l2,
                        "total_radius_cap": TOTAL_RADIUS_CAP,
                        "alpha_or_scaling_permitted": False,
                        "linearization": linearization_audit,
                        "solver": solver_audit,
                    }
                )
                break
            if candidate_sha in evaluated_x_sha256:
                terminal_reason = "ABSOLUTE_RESIDUAL_CANDIDATE_REPEATED"
                iteration_log.append(
                    {
                        "trial_one_based": trial_number,
                        "candidate_evaluated": False,
                        "shadow_role": shadow_role,
                        "reason": terminal_reason,
                        "absolute_minimum_radius_l2": planned_l2,
                        "candidate_x_float64_le_sha256": candidate_sha,
                        "linearization": linearization_audit,
                        "solver": solver_audit,
                    }
                )
                break
            actual_x, actual_sha, preforward_failure = prepare_candidate_before_forward(
                absolute_x, previous_x, evaluated_x_sha256
            )
            if preforward_failure is not None:
                terminal_reason = "ABSOLUTE_RESIDUAL_" + str(
                    preforward_failure["reason"]
                )
                iteration_log.append(
                    {
                        "trial_one_based": trial_number,
                        "candidate_evaluated": False,
                        "shadow_role": shadow_role,
                        "reason": terminal_reason,
                        "planned_absolute_l2": planned_l2,
                        "actual_absolute_l2": float(np.linalg.norm(actual_x)),
                        "preforward_failure": preforward_failure,
                        "linearization": linearization_audit,
                        "solver": solver_audit,
                    }
                )
                break
            trial_state = evaluate_applied_actor(actual_x)
            trial_evaluations += 1
            evaluated_x_sha256.add(actual_sha)
            new_pair_names = register_discovered(trial_state)
            gate_pass = all(trial_state["train_checks"].values())
            signature = native_residual_signature(trial_state)
            merit_reduction = float(previous_state["merit"] - trial_state["merit"])
            no_new_preservation_identity = (
                set(trial_state["top1_flips"]).issubset(previous_top1)
                and set(trial_state["retention_flips"]).issubset(previous_retention)
            )
            signature_changed = signature != previous_signature
            strict_progress = merit_reduction > MERIT_TOL
            eligible_for_s4 = (
                correction_one_based == 1
                and signature_changed
                and strict_progress
                and no_new_preservation_identity
            )
            iteration_log.append(
                {
                    "trial_one_based": trial_evaluations,
                    "candidate_evaluated": True,
                    "shadow_role": shadow_role,
                    "accepted_before_trial": accepted_iterates,
                    "constraint_coordinate": "absolute_CW11_delta",
                    "alpha": 1.0,
                    "alpha_or_scaling_permitted": False,
                    "planned_trial_l2": planned_l2,
                    "actual_trial_l2": float(np.linalg.norm(actual_x)),
                    "linearization_point_l2": float(np.linalg.norm(previous_x)),
                    "linearization_to_candidate_l2": float(
                        np.linalg.norm(actual_x - previous_x)
                    ),
                    "current_merit": float(previous_state["merit"]),
                    "trial_merit": float(trial_state["merit"]),
                    "merit_reduction": merit_reduction,
                    "top1_flip_count": len(trial_state["top1_flips"]),
                    "retention_flip_count": len(trial_state["retention_flips"]),
                    "new_pair_names": new_pair_names,
                    "accepted": gate_pass,
                    "gate_pass": gate_pass,
                    "native_residual_signature_before": previous_signature,
                    "native_residual_signature": signature,
                    "signature_changed": signature_changed,
                    "strict_merit_progress": strict_progress,
                    "no_new_preservation_flip_identity": no_new_preservation_identity,
                    "eligible_for_next_residual_correction": eligible_for_s4,
                    "linearization": linearization_audit,
                    "solver": solver_audit,
                }
            )
            current_x = actual_x
            current_state = trial_state
            if gate_pass:
                accepted_iterates += 1
                terminal_reason = "ALL_TARGETED_TRAIN_GATES_PASSED"
                break
            rejection_count += 1
            if correction_one_based == 1 and not eligible_for_s4:
                if not signature_changed:
                    terminal_reason = "S3_NATIVE_RESIDUAL_SIGNATURE_DID_NOT_CHANGE"
                elif not strict_progress:
                    terminal_reason = "S3_MERIT_DID_NOT_STRICTLY_DECREASE"
                else:
                    terminal_reason = "S3_ADDED_NEW_PRESERVATION_FLIP_IDENTITY"
                break
            terminal_reason = (
                "S3_STRICT_PROGRESS_PERMITS_ONE_FINAL_CORRECTION"
                if correction_one_based == 1
                else "S4_TERMINAL_TRAIN_GATE_FAILED"
            )

'''
    source = splice_region_once(
        source,
        loop_start,
        loop_end,
        loop_new,
        "four_shadow_absolute_residual_protocol",
        audits,
    )

    transformed = source.encode()
    module_name = "cw24_v8_transformed_frozen_v4"
    if module_name in sys.modules:
        raise ProtocolError("v8 transformed v4 module name occupied")
    module = types.ModuleType(module_name)
    module.__file__ = str(V4_SOURCE)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(transformed, str(V4_SOURCE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    checks = {
        "source_total_radius_exact": float(module.TOTAL_RADIUS_CAP) == TOTAL_RADIUS_CAP,
        "source_initial_radius_exact": float(module.TRUST_RADIUS_INITIAL)
        == INITIAL_TRUST_RADIUS,
        "source_max_radius_was_3e4": float(module.TRUST_RADIUS_MAX) == 3.0e-4,
        "source_v4_trial_cap_was_48": int(module.MAX_TRIAL_EVALUATIONS) == 48,
        "source_v4_seed_exact": int(module.SEED) == 202608044,
        "three_unique_splices": len(audits) == 3
        and all(item["occurrences"] == 1 for item in audits),
        "absolute_rhs_formula_present": transformed.count(
            b"rhs_delta + matrix @ absolute_array"
        )
        == 1,
        "no_half_BF16_buffer": b"BUFFER_FRACTION" not in transformed,
        "no_radius_scaled_absolute_endpoint": transformed.count(
            b'"alpha_or_scaling_permitted": False'
        )
        == 2,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v8 transformed-v4 audit failed: {checks}")
    return module, {
        "frozen_v4_source": source_evidence,
        "transformed_source_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_source_bytes": len(transformed),
        "splices": audits,
        "splice_count": len(audits),
        "checks": checks,
    }


def frozen_dependencies() -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for key, path, digest, mode, label in (
        ("v4_source", V4_SOURCE, V4_SOURCE_SHA256, V4_SOURCE_MODE, "frozen v4 source"),
        ("v4_result", V4_RESULT, V4_RESULT_SHA256, V4_RESULT_MODE, "frozen v4 result"),
        ("v6_source", V6_SOURCE, V6_SOURCE_SHA256, V6_SOURCE_MODE, "frozen v6 source"),
        ("v6_result", V6_RESULT, V6_RESULT_SHA256, V6_RESULT_MODE, "frozen v6 result"),
        ("v7_source", V7_SOURCE, V7_SOURCE_SHA256, V7_SOURCE_MODE, "frozen v7 source"),
        ("v7_result", V7_RESULT, V7_RESULT_SHA256, V7_RESULT_MODE, "frozen v7 result"),
    ):
        _, item = regular_source(path, digest, mode, label)
        evidence[key] = item
    return evidence


def production_run() -> dict[str, Any]:
    source = self_evidence(require_frozen=True)
    dependencies = frozen_dependencies()
    v4, transform_evidence = transformed_v4_module()
    originals = {
        "TRUST_RADIUS_MAX": v4.TRUST_RADIUS_MAX,
        "MAX_TRIAL_EVALUATIONS": v4.MAX_TRIAL_EVALUATIONS,
        "MAX_ACCEPTED_ITERATES": v4.MAX_ACCEPTED_ITERATES,
        "SEED": v4.SEED,
    }
    constant_checks = {
        "total_radius_exact": float(v4.TOTAL_RADIUS_CAP) == TOTAL_RADIUS_CAP,
        "initial_radius_exact": float(v4.TRUST_RADIUS_INITIAL) == INITIAL_TRUST_RADIUS,
        "original_max_radius_3e4": float(v4.TRUST_RADIUS_MAX) == 3.0e-4,
        "original_trial_cap_48": int(v4.MAX_TRIAL_EVALUATIONS) == 48,
        "original_accepted_cap_16": int(v4.MAX_ACCEPTED_ITERATES) == 16,
    }
    if not all(constant_checks.values()):
        raise ProtocolError(f"v8 frozen-v4 constant drift: {constant_checks}")
    v4.TRUST_RADIUS_MAX = TOTAL_RADIUS_CAP
    v4.MAX_TRIAL_EVALUATIONS = MAX_TRAIN_SHADOWS
    v4.MAX_ACCEPTED_ITERATES = MAX_TRAIN_SHADOWS
    v4.SEED = SEED
    try:
        result = v4.production_run()
    finally:
        for name, value in originals.items():
            setattr(v4, name, value)

    endpoint = result["endpoint"]
    cuttingplane = endpoint.get("cuttingplane", {})
    iteration_log = cuttingplane.get("iteration_log", [])
    evaluated = [row for row in iteration_log if bool(row.get("candidate_evaluated", True))]
    absolute_rows = [
        row for row in iteration_log if row.get("constraint_coordinate") == "absolute_CW11_delta"
    ]
    shadow_checks = {
        "at_most_four_evaluated_shadows": 0 <= len(evaluated) <= MAX_TRAIN_SHADOWS,
        "counter_matches_evaluated": int(cuttingplane.get("trial_evaluations", -1))
        == len(evaluated),
        "S1_exact_trust_if_evaluated": not evaluated
        or (
            evaluated[0].get("shadow_role") == "frozen_v4_stepping_stone"
            and float(evaluated[0].get("trust_radius", -1.0))
            == INITIAL_TRUST_RADIUS
        ),
        "S2_unbuffered_full_if_present": len(evaluated) < 2
        or (
            evaluated[1].get("shadow_role") == "v6_unbuffered_full_calibration"
            and float(evaluated[1].get("alpha", -1.0)) == 1.0
        ),
        "absolute_rows_are_unscaled": all(
            row.get("alpha_or_scaling_permitted") is False
            and float(row.get("alpha", -1.0)) == 1.0
            for row in absolute_rows
        ),
        "at_most_two_absolute_corrections": len(absolute_rows)
        <= MAX_RESIDUAL_CORRECTIONS,
        "S4_only_after_S3_eligibility": len(absolute_rows) < 2
        or absolute_rows[0].get("eligible_for_next_residual_correction") is True,
    }
    if not all(shadow_checks.values()):
        raise ProtocolError(f"v8 runtime shadow contract drift: {shadow_checks}")

    old_decision = str(endpoint["decision"])
    old_reason = str(endpoint.get("reason", ""))
    if old_decision == "GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE":
        decision = "GO_CW24_V8_ABSOLUTE_RESIDUAL_TRAIN_GATE"
    elif old_decision.startswith("NO_GO_CW24_V4"):
        decision = "NO_GO_CW24_V8_ABSOLUTE_RESIDUAL_TRAIN_GATE"
    else:
        raise ProtocolError(f"unexpected inherited v4 decision: {old_decision}")
    endpoint["decision"] = decision
    endpoint["reason"] = (
        "V8_ABSOLUTE_RESIDUAL_" + old_reason
        if old_reason
        else "V8_ABSOLUTE_RESIDUAL_INHERITED_REASON_MISSING"
    )
    payload_present = endpoint.get("candidate_payload") is not None
    expected_payload = (
        decision == "GO_CW24_V8_ABSOLUTE_RESIDUAL_TRAIN_GATE"
        and endpoint.get("trial", {}).get("pass") is True
        and endpoint.get("promotable_terminal_endpoint_count") == 1
    )
    terminal_payload_gate = payload_present == expected_payload
    if not terminal_payload_gate:
        raise ProtocolError("v8 inherited terminal-only payload gate drift")

    result["schema_version"] = SCHEMA
    result["seed"] = SEED
    result["status"] = decision
    result["decision"] = decision
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_absolute_native_residual_special_BC"
    )
    result["selection"]["optimization_contract"].update(
        {
            "single_promotable_terminal_endpoint": True,
            "first_shadow_trust_radius": INITIAL_TRUST_RADIUS,
            "second_shadow_exact_unbuffered_v6_full_policy": True,
            "failed_second_shadow_is_calibration_only": True,
            "absolute_residual_formula": "rhs_abs=rhs_delta+A@current_x",
            "absolute_objective": "minimum_L2_delta_from_exact_CW11",
            "absolute_candidates_never_scaled": True,
            "maximum_train_shadows": MAX_TRAIN_SHADOWS,
            "maximum_absolute_residual_corrections": MAX_RESIDUAL_CORRECTIONS,
            "S4_requires_signature_merit_and_preservation_progress": True,
            "total_additional_from_CW11_cap": TOTAL_RADIUS_CAP,
            "terminal_only_payload_gate": terminal_payload_gate,
        }
    )
    result["inputs"]["CW24_v8_source"] = source
    result["inputs"]["CW24_v8_transformed_frozen_v4"] = transform_evidence
    result["inputs"]["CW24_v8_frozen_dependencies"] = dependencies
    result["audit"]["CW24_v8_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "train_only_B352": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "network_calls": 0,
        "runtime_constant_changes": {
            "TRUST_RADIUS_MAX": TOTAL_RADIUS_CAP,
            "MAX_TRIAL_EVALUATIONS": MAX_TRAIN_SHADOWS,
            "MAX_ACCEPTED_ITERATES": MAX_TRAIN_SHADOWS,
            "SEED": SEED,
        },
        "frozen_v4_constant_checks": constant_checks,
        "shadow_checks": shadow_checks,
        "terminal_only_payload_gate": terminal_payload_gate,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "v8_selected_only_from_frozen_v4_v6_v7_NO_GO_evidence": True,
        "no_half_BF16_quantum_buffer": True,
        "no_candidate_grid_or_radius_scan": True,
        "calibration_and_correction_shadows_are_not_promotable": True,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        dependencies = frozen_dependencies()
        v4, transform_evidence = transformed_v4_module()
        checks = {
            "total_radius_exact": float(v4.TOTAL_RADIUS_CAP) == TOTAL_RADIUS_CAP,
            "initial_radius_exact": float(v4.TRUST_RADIUS_INITIAL)
            == INITIAL_TRUST_RADIUS,
            "original_max_radius_exact": float(v4.TRUST_RADIUS_MAX) == 3.0e-4,
            "three_unique_splices": transform_evidence["splice_count"] == 3,
            "four_shadow_cap": MAX_TRAIN_SHADOWS == 4,
            "two_residual_corrections": MAX_RESIDUAL_CORRECTIONS == 2,
        }
        if not all(checks.values()):
            raise ProtocolError(f"v8 static variant audit failed: {checks}")
        print(
            canonical_json(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_pass",
                    "source": self_evidence(require_frozen=True),
                    "frozen_dependencies": dependencies,
                    "transformed_v4": transform_evidence,
                    "variant": {
                        "seed": SEED,
                        "initial_trust_radius": INITIAL_TRUST_RADIUS,
                        "total_radius_cap": TOTAL_RADIUS_CAP,
                        "maximum_train_shadows": MAX_TRAIN_SHADOWS,
                        "maximum_absolute_residual_corrections": MAX_RESIDUAL_CORRECTIONS,
                        "absolute_rhs_formula": "rhs_abs=rhs_delta+A@current_x",
                        "absolute_candidates_never_scaled": True,
                        "checks": checks,
                    },
                    "output_absent": not OUTPUT.exists(),
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
        raise ProtocolError("frozen CW24 v8 output already exists")
    result = production_run()
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "candidate_model_state_sha256": result["endpoint"].get("trial", {}).get(
                    "candidate_model_state_sha256"
                ),
                "candidate_payload_present": result["endpoint"].get("candidate_payload")
                is not None,
                "train_shadow_count": result["endpoint"].get("cuttingplane", {}).get(
                    "trial_evaluations"
                ),
                "promotable_terminal_endpoint_count": result["endpoint"].get(
                    "promotable_terminal_endpoint_count"
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

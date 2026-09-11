#!/usr/bin/env python3
"""CW24 v3 inequality-QP train-only wrapper over the frozen v1 engine."""

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
SCRIPT = TOOLS / "probe_u468_cw11_inequality_top1_specialbc_cw24_v3.py"
ENGINE = TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py"
ENGINE_SHA256 = "a3c96eed95a6b558c61303e2190eaa4f353970add4509e8770c3dfcc1621db39"
ENGINE_MODE = 0o555
V1_RESULT = ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v1.json"
V1_RESULT_SHA256 = "9e3be911bfba95338538f315d47d3f7ce9f598e830ace607332d46697cb0cda5"
V2_RESULT = ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v2.json"
V2_RESULT_SHA256 = "36cae87fbaf4947f1caf1d6229a9d8ade75044eb25bc33398b731fd429d9e465"
OUTPUT = ROOT / "artifacts/cw24_cw11_inequality_top1_specialbc_trainonly_v3.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-inequality-top1-specialbc-cw24-v3"
SEED = 202608043
PF_RADIUS_CAP = 9.6e-4
TOTAL_RADIUS_CAP = 9.9999e-4
FRAGILE_RETENTION_SHA256 = "0142a2a3d5bdf5a9a7edc19761db708549701625055a07a4e90013884c752f43"
AUX_WEIGHTS = {
    "dominic32_nll_descent": 0.30,
    "dominic_closest_margin_ascent": 0.40,
    "special9_nll_descent": 0.15,
    "szlach_top1_ce_descent": 0.08,
    "core_other_top1_ce_descent": 0.07,
}


class ProtocolError(RuntimeError):
    """Fail-closed v3 wrapper error."""


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


def regular_source(path: Path, expected_sha: str, expected_mode: int, label: str) -> tuple[bytes, dict[str, Any]]:
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
        "one_variant": source.count(b"\nPF_RADIUS" + b"_CAP =") == 1
        and source.count(b"\nTOTAL_RADIUS" + b"_CAP =") == 1
        and source.count(b"\nAUX_" + b"WEIGHTS =") == 1,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v3 source audit failed: {checks}")
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
                raise ProtocolError("short v3 result write")
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
        raise ProtocolError(f"v3 publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def replace_once(source: str, old: str, new: str, label: str, audit: list[dict[str, Any]]) -> str:
    count = source.count(old)
    if count != 1:
        raise ProtocolError(f"{label}: expected one source match, observed {count}")
    audit.append({
        "label": label,
        "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
        "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
    })
    return source.replace(old, new, 1)


def transformed_engine() -> tuple[ModuleType, dict[str, Any]]:
    raw, evidence = regular_source(ENGINE, ENGINE_SHA256, ENGINE_MODE, "frozen CW24 v1 engine")
    source = raw.decode()
    audit: list[dict[str, Any]] = []
    source = replace_once(
        source,
        '            if float(objective.detach().cpu()) != 0.0:\n                raise ProtocolError(f"{name}: baseline margin drift")',
        '            expected_guard_margin = (1.0 / 128.0 if sha == "' + FRAGILE_RETENTION_SHA256 + '" else 0.0)\n'
        '            if float(objective.detach().cpu()) != expected_guard_margin:\n'
        '                raise ProtocolError(f"{name}: baseline margin drift")',
        "fragile_retention_guard_margin",
        audit,
    )
    source = replace_once(
        source,
        '            "matrix_shape_9x65793": matrix.shape == (9, 65793),\n            "rank9": gram_rank == 9,',
        '            "matrix_shape_10x65793": matrix.shape == (10, 65793),\n            "rank10": gram_rank == 10,',
        "ten_constraint_shape",
        audit,
    )
    start_marker = "        gram_pinv = np.linalg.pinv(gram, rcond=PINV_RCOND)\n"
    end_marker = "\n        per_row_nll = cw22.ordered_nll_per_row(outputs, batch)"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise ProtocolError("inequality solver splice markers missing")
    old_solver = source[start:end]
    new_solver = '''        requested = np.array(
            [2.0 / 512.0, 2.0 / 512.0, 3.0 / 512.0]
            + [0.0] * len(ZERO_MARGIN_SHA256),
            dtype=np.float64,
        )
        full_gram_pinv = np.linalg.pinv(gram, rcond=PINV_RCOND)
        best = None
        constraint_count = len(requested)
        for active_mask in range(1, 1 << constraint_count):
            indices = [
                index for index in range(constraint_count)
                if active_mask & (1 << index)
            ]
            subgram = gram[np.ix_(indices, indices)]
            if int(np.linalg.matrix_rank(subgram)) != len(indices):
                continue
            sublambda = np.linalg.pinv(subgram, rcond=PINV_RCOND) @ requested[indices]
            if float(np.min(sublambda)) < -1.0e-12:
                continue
            multipliers = np.zeros(constraint_count, dtype=np.float64)
            multipliers[indices] = np.maximum(sublambda, 0.0)
            effects = gram @ multipliers
            if float(np.min(effects - requested)) < -1.0e-10:
                continue
            delta = matrix.T @ multipliers
            norm = float(np.linalg.norm(delta))
            key = (norm, active_mask)
            if best is None or key < best[0]:
                best = (key, delta, multipliers, effects, indices)
        if best is None:
            raise ProtocolError("PF inequality active-set solve is infeasible")
        _, nominal_pf, inequality_multipliers, inequality_effects, active_indices = best
        gram_pinv = full_gram_pinv
        nominal_pf_l2 = float(np.linalg.norm(nominal_pf))
        if not math.isfinite(nominal_pf_l2) or nominal_pf_l2 <= 0.0:
            raise ProtocolError("invalid PF inequality minimum-norm solution")
        pf_scale = min(1.0, PF_RADIUS_CAP / nominal_pf_l2)
        delta_pf = nominal_pf * pf_scale
        delta_pf_l2 = float(np.linalg.norm(delta_pf))
'''
    audit.append({
        "label": "active_set_inequality_solver",
        "old_sha256": hashlib.sha256(old_solver.encode()).hexdigest(),
        "new_sha256": hashlib.sha256(new_solver.encode()).hexdigest(),
    })
    source = source[:start] + new_solver + source[end:]

    aux_start_marker = "        projected: dict[str, Any] = {}\n"
    aux_end_marker = "        direction_l2 = float(np.linalg.norm(direction))\n"
    aux_start = source.find(aux_start_marker)
    aux_end = source.find(aux_end_marker, aux_start)
    if aux_start < 0 or aux_end < 0:
        raise ProtocolError("auxiliary projection splice markers missing")
    old_aux = source[aux_start:aux_end]
    new_aux = '''        projected: dict[str, Any] = {}
        direction = np.zeros_like(delta_pf)
        for name, weight in AUX_WEIGHTS.items():
            value = project_null(aux_gradients[name])
            norm = float(np.linalg.norm(value))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ProtocolError(f"{name}: zero null-space projection")
            projected[name] = value
            sign = 1.0 if name.endswith("margin_ascent") else -1.0
            direction += sign * float(weight) * value / norm
        guard_projected = {
            "retention160_nll_guard": project_null(aux_gradients["retention160_nll_guard"]),
            "szlach_top1_ce_guard": project_null(aux_gradients["szlach_top1_ce_descent"]),
            "core_other_top1_ce_guard": project_null(aux_gradients["core_other_top1_ce_descent"]),
        }
        for name, value in guard_projected.items():
            if float(value @ value) <= 0.0:
                raise ProtocolError(f"{name}: zero halfspace guard")
        guard_dots_before = {
            name: float(value @ direction) for name, value in guard_projected.items()
        }
        projection_names = []
        for projection_pass in range(64):
            for name, value in guard_projected.items():
                dot = float(value @ direction)
                if dot > 1.0e-15:
                    direction = direction - (dot / float(value @ value)) * value
                    projection_names.append(name)
        guard_dots_after = {
            name: float(value @ direction) for name, value in guard_projected.items()
        }
        retention_projected = guard_projected["retention160_nll_guard"]
        retention_dot_before = guard_dots_before["retention160_nll_guard"]
        retention_dot_after = guard_dots_after["retention160_nll_guard"]
        retention_projection_applied = "retention160_nll_guard" in projection_names
'''
    audit.append({
        "label": "cyclic_aux_halfspace_projection",
        "old_sha256": hashlib.sha256(old_aux.encode()).hexdigest(),
        "new_sha256": hashlib.sha256(new_aux.encode()).hexdigest(),
    })
    source = source[:aux_start] + new_aux + source[aux_end:]
    source = replace_once(
        source,
        '            "PF_nominal_residual_small": float(np.max(np.abs(matrix @ nominal_pf - requested))) <= 1e-9,',
        '            "PF_nominal_inequalities_satisfied": float(np.min(matrix @ delta_pf - requested)) >= -1e-10,',
        "inequality_geometry_gate",
        audit,
    )
    source = replace_once(
        source,
        '            "retention_first_order_nondegrade": retention_dot_after <= 1e-12,',
        '            "retention_first_order_nondegrade": retention_dot_after <= 1e-12,\n'
        '            "top1_halfspaces_first_order_nondegrade": max(guard_dots_after.values()) <= 1e-12,',
        "top1_geometry_gate",
        audit,
    )
    source = replace_once(
        source,
        '            "PF_objective_order": pf_names,',
        '            "PF_objective_order": pf_names,\n'
        '            "PF_inequality_active_indices": active_indices,\n'
        '            "PF_inequality_multipliers": inequality_multipliers.tolist(),\n'
        '            "PF_inequality_effects": inequality_effects.tolist(),',
        "inequality_geometry_report",
        audit,
    )
    source = replace_once(
        source,
        '            "retention_conflict_projection_applied": retention_projection_applied,',
        '            "retention_conflict_projection_applied": retention_projection_applied,\n'
        '            "halfspace_projection_names": projection_names,\n'
        '            "halfspace_guard_dots_before": guard_dots_before,\n'
        '            "halfspace_guard_dots_after": guard_dots_after,',
        "halfspace_geometry_report",
        audit,
    )
    transformed_bytes = source.encode()
    name = "cw24_v3_transformed_v1_engine"
    if name in sys.modules:
        raise ProtocolError("transformed engine module name occupied")
    module = types.ModuleType(name)
    module.__file__ = str(ENGINE)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(transformed_bytes, str(ENGINE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, {
        "frozen_engine": evidence,
        "transformed_source_sha256": hashlib.sha256(transformed_bytes).hexdigest(),
        "transformed_source_bytes": len(transformed_bytes),
        "transformations": audit,
        "transformation_count": len(audit),
    }


def production_run() -> dict[str, Any]:
    source = self_evidence(require_frozen=True)
    _, v1_evidence = regular_source(V1_RESULT, V1_RESULT_SHA256, 0o444, "frozen v1 result")
    _, v2_evidence = regular_source(V2_RESULT, V2_RESULT_SHA256, 0o444, "frozen v2 result")
    engine, transform_evidence = transformed_engine()
    originals = {
        "PF_RADIUS_CAP": engine.PF_RADIUS_CAP,
        "TOTAL_RADIUS_CAP": engine.TOTAL_RADIUS_CAP,
        "AUX_WEIGHTS": engine.AUX_WEIGHTS,
        "ZERO_MARGIN_SHA256": engine.ZERO_MARGIN_SHA256,
        "SEED": engine.SEED,
    }
    engine.PF_RADIUS_CAP = PF_RADIUS_CAP
    engine.TOTAL_RADIUS_CAP = TOTAL_RADIUS_CAP
    engine.AUX_WEIGHTS = dict(AUX_WEIGHTS)
    engine.ZERO_MARGIN_SHA256 = tuple(engine.ZERO_MARGIN_SHA256) + (FRAGILE_RETENTION_SHA256,)
    engine.SEED = SEED
    try:
        result = engine.production_run()
    finally:
        for key, value in originals.items():
            setattr(engine, key, value)
    geometry = result["endpoint"].get("two_stage_geometry", {})
    variant_checks = {
        "ten_constraints": len(geometry.get("PF_objective_order", [])) == 10,
        "inequality_active_set_present": bool(geometry.get("PF_inequality_active_indices")),
        "PF_radius_within_cap": float(geometry.get("PF_planned_l2", 1.0)) <= PF_RADIUS_CAP + 1e-15,
        "total_radius_exact": math.isclose(
            float(geometry.get("planned_total_l2", -1.0)),
            TOTAL_RADIUS_CAP,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "weights_exact": geometry.get("aux_weights") == AUX_WEIGHTS,
        "single_candidate": geometry.get("single_candidate_only") is True,
        "transformation_count8": transform_evidence["transformation_count"] == 8,
    }
    if not all(variant_checks.values()):
        raise ProtocolError(f"v3 variant application drift: {variant_checks}")
    result["schema_version"] = SCHEMA
    result["seed"] = SEED
    result["inputs"]["CW24_v3_wrapper"] = source
    result["inputs"]["CW24_v3_transformed_engine"] = transform_evidence
    result["inputs"]["CW24_v1_NO_GO_result"] = v1_evidence
    result["inputs"]["CW24_v2_NO_GO_result"] = v2_evidence
    result["audit"]["CW24_v3_variant"] = {
        "solver": "deterministic exhaustive active-set inequality minimum-norm QP",
        "PF_requested_changes": [2.0 / 512.0, 2.0 / 512.0, 3.0 / 512.0],
        "zero_or_fragile_guard_requested_changes": [0.0] * 7,
        "fragile_retention_guard": FRAGILE_RETENTION_SHA256,
        "PF_radius_cap": PF_RADIUS_CAP,
        "total_radius_cap": TOTAL_RADIUS_CAP,
        "aux_weights": AUX_WEIGHTS,
        "aux_guard_projection": "64 deterministic cyclic halfspace passes",
        "checks": variant_checks,
        "single_changed_train_shadow": True,
        "official_unique_changed_candidate_count_consumed": 0,
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
        _, engine_evidence = regular_source(ENGINE, ENGINE_SHA256, ENGINE_MODE, "frozen v1 engine")
        _, v1_evidence = regular_source(V1_RESULT, V1_RESULT_SHA256, 0o444, "frozen v1 result")
        _, v2_evidence = regular_source(V2_RESULT, V2_RESULT_SHA256, 0o444, "frozen v2 result")
        _, transform_evidence = transformed_engine()
        print(canonical_json({
            "schema_version": SCHEMA,
            "status": "static_audit_pass",
            "source": self_evidence(require_frozen=True),
            "engine": engine_evidence,
            "v1_result": v1_evidence,
            "v2_result": v2_evidence,
            "transform": transform_evidence,
            "variant": {
                "PF_radius_cap": PF_RADIUS_CAP,
                "total_radius_cap": TOTAL_RADIUS_CAP,
                "fragile_retention_guard": FRAGILE_RETENTION_SHA256,
                "aux_weights": AUX_WEIGHTS,
            },
            "output_absent": not OUTPUT.exists(),
            "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
            "CUDA_initialized": False,
            "writes_performed": 0,
        }).decode(), end="")
        return
    if OUTPUT.exists():
        raise ProtocolError("frozen CW24 v3 output already exists")
    result = production_run()
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    print(canonical_json({
        "schema_version": SCHEMA,
        "status": result["status"],
        "decision": result["decision"],
        "candidate_model_state_sha256": result["endpoint"].get("trial", {}).get(
            "candidate_model_state_sha256"
        ),
        "candidate_payload_present": result["endpoint"].get("candidate_payload") is not None,
        "changed_candidate_train_shadow_count": result["endpoint"].get(
            "changed_candidate_train_shadow_count"
        ),
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
        "output": publication,
    }).decode(), end="")


if __name__ == "__main__":
    main()

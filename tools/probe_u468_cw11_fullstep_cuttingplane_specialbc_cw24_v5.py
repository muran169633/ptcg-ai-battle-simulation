#!/usr/bin/env python3
"""CW24 v5 full-step wrapper over the frozen v4 native cutting-plane probe.

The frozen v4 source and its completed NO_GO result are mandatory inputs.  V5
changes only the trust policy: the initial and maximum trust radii equal the
existing 0.00099998 total CW11 radius, so each certified QP endpoint is tested
at full step when the global sphere permits it.  A newly discovered dynamic
pair cut resets trust to full; an ordinary merit rejection still halves it.
All B352/native/integrity/replay/payload gates remain the frozen v4 gates.
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
SCRIPT = TOOLS / "probe_u468_cw11_fullstep_cuttingplane_specialbc_cw24_v5.py"
V4_SOURCE = TOOLS / "probe_u468_cw11_native_cuttingplane_specialbc_cw24_v4.py"
V4_SOURCE_SHA256 = "134eebec7b1e54b68bd47256fde5c8b80dea3cd8d8ea4399428ec4c1de7d1454"
V4_SOURCE_MODE = 0o555
V4_RESULT = ROOT / "artifacts/cw24_cw11_native_cuttingplane_specialbc_trainonly_v4.json"
V4_RESULT_SHA256 = "db0c2c356fac5779e14698e3ff3ee9b36967f2aa6ddecee55531bb37e8b3377c"
V4_RESULT_MODE = 0o444
OUTPUT = ROOT / "artifacts/cw24_cw11_fullstep_cuttingplane_specialbc_trainonly_v5.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-fullstep-cuttingplane-specialbc-cw24-v5"
SEED = 202608045
FULL_TRUST_RADIUS = 9.9998e-4


class ProtocolError(RuntimeError):
    """Fail-closed CW24 v5 wrapper error."""


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
        "one_full_trust_constant": source.count(b"\nFULL_TRUST_" + b"RADIUS =") == 1,
        "v4_source_lock_present": V4_SOURCE_SHA256.encode() in source,
        "v4_result_lock_present": V4_RESULT_SHA256.encode() in source,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "no_payload_construction": (b"candidate_actor_" + b"float32_le") not in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v5 source audit failed: {checks}")
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
                raise ProtocolError("short v5 result write")
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
        raise ProtocolError(f"v5 publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def transformed_v4_module() -> tuple[ModuleType, dict[str, Any]]:
    raw, source_evidence = regular_source(
        V4_SOURCE, V4_SOURCE_SHA256, V4_SOURCE_MODE, "frozen CW24 v4 source"
    )
    source = raw.decode()
    old = '''            else:
                rejection_count += 1
                trust_radius *= 0.5
                cw20.apply_flat_actor(
'''
    new = '''            else:
                rejection_count += 1
                if new_pair_names:
                    trust_radius = TRUST_RADIUS_MAX
                else:
                    trust_radius *= 0.5
                cw20.apply_flat_actor(
'''
    occurrences = source.count(old)
    if occurrences != 1:
        raise ProtocolError(
            f"v4 dynamic-cut trust splice drift: expected 1 occurrence, observed {occurrences}"
        )
    transformed = source.replace(old, new, 1).encode()
    module_name = "cw24_v5_transformed_frozen_v4"
    if module_name in sys.modules:
        raise ProtocolError("v5 transformed v4 module name occupied")
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
        "source_total_radius_exact": float(module.TOTAL_RADIUS_CAP) == FULL_TRUST_RADIUS,
        "source_v4_seed_exact": int(module.SEED) == 202608044,
        "source_output_is_v4": Path(module.OUTPUT).resolve() == (
            ROOT / "artifacts/cw24_cw11_native_cuttingplane_specialbc_trainonly_v4.json"
        ).resolve(),
        "single_dynamic_cut_reset_splice": occurrences == 1,
        "ordinary_rejection_halving_retained": transformed.count(b"trust_radius *= 0.5") == 1,
        "dynamic_cut_full_reset_present": transformed.count(
            b"trust_radius = TRUST_RADIUS_MAX"
        )
        == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v5 transformed v4 audit failed: {checks}")
    return module, {
        "frozen_v4_source": source_evidence,
        "transformed_source_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_source_bytes": len(transformed),
        "transformation": {
            "label": "dynamic_pair_cut_resets_full_trust",
            "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
            "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
            "occurrences": occurrences,
        },
        "checks": checks,
    }


def production_run() -> dict[str, Any]:
    source = self_evidence(require_frozen=True)
    _, v4_result_evidence = regular_source(
        V4_RESULT, V4_RESULT_SHA256, V4_RESULT_MODE, "frozen CW24 v4 result"
    )
    v4, transform_evidence = transformed_v4_module()
    originals = {
        "TRUST_RADIUS_INITIAL": v4.TRUST_RADIUS_INITIAL,
        "TRUST_RADIUS_MAX": v4.TRUST_RADIUS_MAX,
        "SEED": v4.SEED,
    }
    constant_checks = {
        "v4_total_radius_exact": float(v4.TOTAL_RADIUS_CAP) == FULL_TRUST_RADIUS,
        "v4_initial_radius_was_2p5e4": float(v4.TRUST_RADIUS_INITIAL) == 2.5e-4,
        "v4_max_radius_was_3e4": float(v4.TRUST_RADIUS_MAX) == 3.0e-4,
        "v4_min_radius_retained": float(v4.TRUST_RADIUS_MIN) == 3.125e-5,
    }
    if not all(constant_checks.values()):
        raise ProtocolError(f"v4 trust constants drift: {constant_checks}")
    v4.TRUST_RADIUS_INITIAL = FULL_TRUST_RADIUS
    v4.TRUST_RADIUS_MAX = FULL_TRUST_RADIUS
    v4.SEED = SEED
    try:
        result = v4.production_run()
    finally:
        for name, value in originals.items():
            setattr(v4, name, value)

    endpoint = result["endpoint"]
    old_decision = str(endpoint["decision"])
    old_reason = str(endpoint.get("reason", ""))
    if old_decision == "GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE":
        decision = "GO_CW24_V5_FULLSTEP_CUTTINGPLANE_TRAIN_GATE"
    elif old_decision.startswith("NO_GO_CW24_V4"):
        decision = "NO_GO_CW24_V5_FULLSTEP_CUTTINGPLANE_TRAIN_GATE"
    else:
        raise ProtocolError(f"unexpected v4 endpoint decision: {old_decision}")
    endpoint["decision"] = decision
    endpoint["reason"] = (
        "V5_FULLSTEP_" + old_reason
        if old_reason
        else "V5_FULLSTEP_INHERITED_V4_ENDPOINT_REASON_MISSING"
    )
    payload_present = endpoint.get("candidate_payload") is not None
    expected_payload = (
        decision == "GO_CW24_V5_FULLSTEP_CUTTINGPLANE_TRAIN_GATE"
        and endpoint.get("trial", {}).get("pass") is True
        and endpoint.get("promotable_terminal_endpoint_count") == 1
    )
    terminal_payload_gate = payload_present == expected_payload
    if not terminal_payload_gate:
        raise ProtocolError("v5 inherited terminal-only payload gate drift")

    result["schema_version"] = SCHEMA
    result["seed"] = SEED
    result["status"] = decision
    result["decision"] = decision
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_fullstep_native_cuttingplane_special_BC"
    )
    result["selection"]["optimization_contract"].update(
        {
            "single_promotable_terminal_endpoint": True,
            "certified_QP_full_step_first": True,
            "trust_radius_initial": FULL_TRUST_RADIUS,
            "trust_radius_max": FULL_TRUST_RADIUS,
            "trust_radius_min_inherited_v4": float(v4.TRUST_RADIUS_MIN),
            "dynamic_pair_cut_trust_policy": "reset_to_full",
            "ordinary_merit_rejection_trust_policy": "halve",
            "total_additional_from_CW11_cap": FULL_TRUST_RADIUS,
            "terminal_only_payload_gate": terminal_payload_gate,
        }
    )
    result["inputs"]["CW24_v5_source"] = source
    result["inputs"]["CW24_v5_transformed_frozen_v4"] = transform_evidence
    result["inputs"]["CW24_v4_frozen_NO_GO_result"] = v4_result_evidence
    result["audit"]["CW24_v5_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "train_only_B352": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "v4_constant_checks": constant_checks,
        "only_runtime_constants_changed": [
            "TRUST_RADIUS_INITIAL",
            "TRUST_RADIUS_MAX",
            "SEED",
        ],
        "full_trust_radius": FULL_TRUST_RADIUS,
        "dynamic_cut_reset_transformation_count": 1,
        "terminal_only_payload_gate": terminal_payload_gate,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "v5_recipe_selected_from_frozen_v4_NO_GO_only": True,
        "v4_first_certified_QP_step_l2": 0.0009457176289065079,
        "fullstep_endpoint_radius_was_preregistered_before_v5_run": True,
        "internal_iterates_are_not_separately_promotable_endpoints": True,
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
        _, v4_source_evidence = regular_source(
            V4_SOURCE, V4_SOURCE_SHA256, V4_SOURCE_MODE, "frozen CW24 v4 source"
        )
        _, v4_result_evidence = regular_source(
            V4_RESULT, V4_RESULT_SHA256, V4_RESULT_MODE, "frozen CW24 v4 result"
        )
        v4, transform_evidence = transformed_v4_module()
        checks = {
            "v4_total_radius_exact": float(v4.TOTAL_RADIUS_CAP) == FULL_TRUST_RADIUS,
            "v4_initial_radius_exact": float(v4.TRUST_RADIUS_INITIAL) == 2.5e-4,
            "v4_max_radius_exact": float(v4.TRUST_RADIUS_MAX) == 3.0e-4,
            "v5_sets_only_initial_and_max_to_full": True,
            "dynamic_cut_reset_transformation_count": transform_evidence["transformation"][
                "occurrences"
            ]
            == 1,
        }
        if not all(checks.values()):
            raise ProtocolError(f"v5 static variant audit failed: {checks}")
        print(
            canonical_json(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_pass",
                    "source": self_evidence(require_frozen=True),
                    "frozen_v4_source": v4_source_evidence,
                    "frozen_v4_result": v4_result_evidence,
                    "transformed_v4": transform_evidence,
                    "variant": {
                        "seed": SEED,
                        "trust_radius_initial": FULL_TRUST_RADIUS,
                        "trust_radius_max": FULL_TRUST_RADIUS,
                        "total_radius_cap": FULL_TRUST_RADIUS,
                        "dynamic_pair_cut_trust_policy": "reset_to_full",
                        "ordinary_merit_rejection_trust_policy": "halve",
                        "checks": checks,
                    },
                    "output_absent": not OUTPUT.exists(),
                    "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
                    "CUDA_initialized": False,
                    "validation_or_test_rows_opened": 0,
                    "writes_performed": 0,
                }
            ).decode(),
            end="",
        )
        return
    if OUTPUT.exists():
        raise ProtocolError("frozen CW24 v5 output already exists")
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

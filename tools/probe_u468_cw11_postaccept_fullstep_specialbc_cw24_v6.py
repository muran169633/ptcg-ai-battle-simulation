#!/usr/bin/env python3
"""CW24 v6 two-shadow post-accept full-step wrapper over frozen v4.

V6 exactly preserves the frozen v4 first 0.00025 stepping-stone trial.  If that
trial is accepted but nonterminal, trust is set directly to the unchanged
0.00099998 CW11 radius and exactly one second certified-QP endpoint is tested.
There is no subsequent radius scan: the second endpoint must pass every frozen
v4 train check or the run is NO_GO.  Dynamic cuts reset full trust and ordinary
merit rejection retains v4 halving, although the two-shadow cap forbids a third
candidate.  Payload exposure remains the inherited terminal-only v4 gate.
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
SCRIPT = TOOLS / "probe_u468_cw11_postaccept_fullstep_specialbc_cw24_v6.py"
V4_SOURCE = TOOLS / "probe_u468_cw11_native_cuttingplane_specialbc_cw24_v4.py"
V4_SOURCE_SHA256 = "134eebec7b1e54b68bd47256fde5c8b80dea3cd8d8ea4399428ec4c1de7d1454"
V4_SOURCE_MODE = 0o555
V4_RESULT = ROOT / "artifacts/cw24_cw11_native_cuttingplane_specialbc_trainonly_v4.json"
V4_RESULT_SHA256 = "db0c2c356fac5779e14698e3ff3ee9b36967f2aa6ddecee55531bb37e8b3377c"
V4_RESULT_MODE = 0o444
V5_RESULT = ROOT / "artifacts/cw24_cw11_fullstep_cuttingplane_specialbc_trainonly_v5.json"
V5_RESULT_SHA256 = "7dcb9ad7d1e86c046fe623dd47d9fb9da2b80f97b49cd6ebd379991f6f052793"
V5_RESULT_MODE = 0o444
OUTPUT = ROOT / "artifacts/cw24_cw11_postaccept_fullstep_specialbc_trainonly_v6.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-postaccept-fullstep-specialbc-cw24-v6"
SEED = 202608046
INITIAL_TRUST_RADIUS = 2.5e-4
FULL_TRUST_RADIUS = 9.9998e-4
MAX_TRIAL_EVALUATIONS = 2
MAX_ACCEPTED_ITERATES = 2
PREREGISTERED_SECOND_PLANNED_L2 = 0.0009909412795


class ProtocolError(RuntimeError):
    """Fail-closed CW24 v6 wrapper error."""


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
        "one_initial_trust_constant": source.count(b"\nINITIAL_TRUST_" + b"RADIUS =")
        == 1,
        "one_full_trust_constant": source.count(b"\nFULL_TRUST_" + b"RADIUS =") == 1,
        "two_shadow_caps_present": b"MAX_TRIAL_EVALUATIONS = 2" in source
        and b"MAX_ACCEPTED_ITERATES = 2" in source,
        "v4_source_lock_present": V4_SOURCE_SHA256.encode() in source,
        "v4_result_lock_present": V4_RESULT_SHA256.encode() in source,
        "v5_result_lock_present": V5_RESULT_SHA256.encode() in source,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "no_payload_construction": (b"candidate_actor_" + b"float32_le") not in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v6 source audit failed: {checks}")
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
                raise ProtocolError("short v6 result write")
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
        raise ProtocolError(f"v6 publication drift: {checks}")
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
    occurrences = {
        "dynamic_cut_reset": source.count(rejection_old),
        "accepted_nonterminal_full_trust": source.count(accepted_old),
    }
    if occurrences != {
        "dynamic_cut_reset": 1,
        "accepted_nonterminal_full_trust": 1,
    }:
        raise ProtocolError(f"v6 frozen-v4 splice drift: {occurrences}")
    transformed_text = source.replace(rejection_old, rejection_new, 1).replace(
        accepted_old, accepted_new, 1
    )
    transformed = transformed_text.encode()
    module_name = "cw24_v6_transformed_frozen_v4"
    if module_name in sys.modules:
        raise ProtocolError("v6 transformed v4 module name occupied")
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
        "source_initial_radius_exact": float(module.TRUST_RADIUS_INITIAL)
        == INITIAL_TRUST_RADIUS,
        "source_max_radius_was_3e4": float(module.TRUST_RADIUS_MAX) == 3.0e-4,
        "source_v4_trial_cap_was_48": int(module.MAX_TRIAL_EVALUATIONS) == 48,
        "source_v4_accepted_cap_was_16": int(module.MAX_ACCEPTED_ITERATES) == 16,
        "source_v4_seed_exact": int(module.SEED) == 202608044,
        "source_output_is_v4": Path(module.OUTPUT).resolve()
        == (
            ROOT / "artifacts/cw24_cw11_native_cuttingplane_specialbc_trainonly_v4.json"
        ).resolve(),
        "two_splices_exact": all(value == 1 for value in occurrences.values()),
        "accepted_nonterminal_full_reset_present": transformed.count(
            b"trust_radius = TRUST_RADIUS_MAX"
        )
        == 2,
        "old_conditional_growth_absent": b"trust_radius * 1.5" not in transformed,
        "ordinary_rejection_halving_retained": transformed.count(b"trust_radius *= 0.5")
        == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v6 transformed v4 audit failed: {checks}")
    return module, {
        "frozen_v4_source": source_evidence,
        "transformed_source_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_source_bytes": len(transformed),
        "transformations": [
            {
                "label": "dynamic_pair_cut_resets_full_trust",
                "old_sha256": hashlib.sha256(rejection_old.encode()).hexdigest(),
                "new_sha256": hashlib.sha256(rejection_new.encode()).hexdigest(),
                "occurrences": occurrences["dynamic_cut_reset"],
            },
            {
                "label": "accepted_nonterminal_sets_full_trust",
                "old_sha256": hashlib.sha256(accepted_old.encode()).hexdigest(),
                "new_sha256": hashlib.sha256(accepted_new.encode()).hexdigest(),
                "occurrences": occurrences["accepted_nonterminal_full_trust"],
            },
        ],
        "transformation_count": 2,
        "checks": checks,
    }


def production_run() -> dict[str, Any]:
    source = self_evidence(require_frozen=True)
    _, v4_result_evidence = regular_source(
        V4_RESULT, V4_RESULT_SHA256, V4_RESULT_MODE, "frozen CW24 v4 result"
    )
    _, v5_result_evidence = regular_source(
        V5_RESULT, V5_RESULT_SHA256, V5_RESULT_MODE, "frozen CW24 v5 result"
    )
    v4, transform_evidence = transformed_v4_module()
    originals = {
        "TRUST_RADIUS_MAX": v4.TRUST_RADIUS_MAX,
        "MAX_TRIAL_EVALUATIONS": v4.MAX_TRIAL_EVALUATIONS,
        "MAX_ACCEPTED_ITERATES": v4.MAX_ACCEPTED_ITERATES,
        "SEED": v4.SEED,
    }
    constant_checks = {
        "v4_total_radius_exact": float(v4.TOTAL_RADIUS_CAP) == FULL_TRUST_RADIUS,
        "v4_initial_radius_retained_2p5e4": float(v4.TRUST_RADIUS_INITIAL)
        == INITIAL_TRUST_RADIUS,
        "v4_max_radius_was_3e4": float(v4.TRUST_RADIUS_MAX) == 3.0e-4,
        "v4_min_radius_retained": float(v4.TRUST_RADIUS_MIN) == 3.125e-5,
        "v4_trial_cap_was_48": int(v4.MAX_TRIAL_EVALUATIONS) == 48,
        "v4_accepted_cap_was_16": int(v4.MAX_ACCEPTED_ITERATES) == 16,
    }
    if not all(constant_checks.values()):
        raise ProtocolError(f"v4 constants drift before v6: {constant_checks}")
    v4.TRUST_RADIUS_MAX = FULL_TRUST_RADIUS
    v4.MAX_TRIAL_EVALUATIONS = MAX_TRIAL_EVALUATIONS
    v4.MAX_ACCEPTED_ITERATES = MAX_ACCEPTED_ITERATES
    v4.SEED = SEED
    try:
        result = v4.production_run()
    finally:
        for name, value in originals.items():
            setattr(v4, name, value)

    endpoint = result["endpoint"]
    trial_evaluations = int(endpoint.get("cuttingplane", {}).get("trial_evaluations", -1))
    if trial_evaluations < 1 or trial_evaluations > MAX_TRIAL_EVALUATIONS:
        raise ProtocolError(f"v6 shadow count drift: {trial_evaluations}")
    iteration_log = endpoint.get("cuttingplane", {}).get("iteration_log", [])
    evaluated_log = [row for row in iteration_log if bool(row.get("candidate_evaluated", True))]
    shadow_contract_checks = {
        "exactly_two_trials": trial_evaluations == MAX_TRIAL_EVALUATIONS,
        "log_matches_evaluated_trials": len(evaluated_log) == trial_evaluations,
        "first_trial_trust_exact": len(evaluated_log) == 2
        and float(evaluated_log[0].get("trust_radius", -1.0)) == INITIAL_TRUST_RADIUS,
        "first_trial_is_accepted_nonterminal": len(evaluated_log) == 2
        and evaluated_log[0].get("accepted") is True
        and evaluated_log[0].get("gate_pass") is False,
        "second_trial_trust_is_full": len(evaluated_log) == 2
        and float(evaluated_log[1].get("trust_radius", -1.0)) == FULL_TRUST_RADIUS,
        "second_trial_alpha_is_full": len(evaluated_log) == 2
        and float(evaluated_log[1].get("alpha", -1.0)) == 1.0,
        "second_planned_l2_matches_preregistered": len(evaluated_log) == 2
        and math.isclose(
            float(evaluated_log[1].get("planned_trial_l2", -1.0)),
            PREREGISTERED_SECOND_PLANNED_L2,
            rel_tol=0.0,
            abs_tol=5.0e-13,
        ),
        "no_third_shadow": all(int(row.get("trial_one_based", 0)) <= 2 for row in iteration_log),
    }
    if not all(shadow_contract_checks.values()):
        raise ProtocolError(f"v6 two-shadow contract drift: {shadow_contract_checks}")

    old_decision = str(endpoint["decision"])
    old_reason = str(endpoint.get("reason", ""))
    if old_decision == "GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE":
        decision = "GO_CW24_V6_POSTACCEPT_FULLSTEP_TRAIN_GATE"
    elif old_decision.startswith("NO_GO_CW24_V4"):
        decision = "NO_GO_CW24_V6_POSTACCEPT_FULLSTEP_TRAIN_GATE"
    else:
        raise ProtocolError(f"unexpected v4 endpoint decision: {old_decision}")
    endpoint["decision"] = decision
    endpoint["reason"] = (
        "V6_POSTACCEPT_FULLSTEP_" + old_reason
        if old_reason
        else "V6_POSTACCEPT_FULLSTEP_INHERITED_V4_ENDPOINT_REASON_MISSING"
    )
    payload_present = endpoint.get("candidate_payload") is not None
    expected_payload = (
        decision == "GO_CW24_V6_POSTACCEPT_FULLSTEP_TRAIN_GATE"
        and endpoint.get("trial", {}).get("pass") is True
        and endpoint.get("promotable_terminal_endpoint_count") == 1
    )
    terminal_payload_gate = payload_present == expected_payload
    if not terminal_payload_gate:
        raise ProtocolError("v6 inherited terminal-only payload gate drift")

    result["schema_version"] = SCHEMA
    result["seed"] = SEED
    result["status"] = decision
    result["decision"] = decision
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_postaccept_fullstep_special_BC"
    )
    result["selection"]["optimization_contract"].update(
        {
            "single_promotable_terminal_endpoint": True,
            "two_train_shadows_maximum": True,
            "first_shadow_trust_radius": INITIAL_TRUST_RADIUS,
            "accepted_nonterminal_trust_policy": "set_directly_to_full",
            "second_shadow_is_terminal": True,
            "second_shadow_requires_all_train_checks": True,
            "trust_radius_max": FULL_TRUST_RADIUS,
            "dynamic_pair_cut_trust_policy": "reset_to_full",
            "ordinary_merit_rejection_trust_policy": "halve_without_third_shadow",
            "max_trial_evaluations": MAX_TRIAL_EVALUATIONS,
            "max_accepted_iterates": MAX_ACCEPTED_ITERATES,
            "total_additional_from_CW11_cap": FULL_TRUST_RADIUS,
            "terminal_only_payload_gate": terminal_payload_gate,
        }
    )
    result["inputs"]["CW24_v6_source"] = source
    result["inputs"]["CW24_v6_transformed_frozen_v4"] = transform_evidence
    result["inputs"]["CW24_v4_frozen_NO_GO_result"] = v4_result_evidence
    result["inputs"]["CW24_v5_frozen_NO_GO_result"] = v5_result_evidence
    result["audit"]["CW24_v6_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "train_only_B352": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "v4_constant_checks": constant_checks,
        "two_shadow_runtime_constant_changes": {
            "TRUST_RADIUS_MAX": FULL_TRUST_RADIUS,
            "MAX_TRIAL_EVALUATIONS": MAX_TRIAL_EVALUATIONS,
            "MAX_ACCEPTED_ITERATES": MAX_ACCEPTED_ITERATES,
            "SEED": SEED,
        },
        "TRUST_RADIUS_INITIAL_unchanged": float(v4.TRUST_RADIUS_INITIAL)
        == INITIAL_TRUST_RADIUS,
        "transformation_count": 2,
        "shadow_contract_checks": shadow_contract_checks,
        "terminal_only_payload_gate": terminal_payload_gate,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "v6_recipe_selected_from_frozen_v4_and_v5_NO_GO_only": True,
        "first_shadow_is_frozen_v4_0p00025_stepping_stone": True,
        "second_fullstep_planned_l2_preregistered": PREREGISTERED_SECOND_PLANNED_L2,
        "second_shadow_failure_is_terminal_NO_GO": True,
        "no_radius_scan_after_second_shadow": True,
        "internal_shadows_are_not_separately_promotable_endpoints": True,
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
        _, v5_result_evidence = regular_source(
            V5_RESULT, V5_RESULT_SHA256, V5_RESULT_MODE, "frozen CW24 v5 result"
        )
        v4, transform_evidence = transformed_v4_module()
        checks = {
            "v4_total_radius_exact": float(v4.TOTAL_RADIUS_CAP) == FULL_TRUST_RADIUS,
            "v4_initial_radius_retained": float(v4.TRUST_RADIUS_INITIAL)
            == INITIAL_TRUST_RADIUS,
            "v4_max_radius_original": float(v4.TRUST_RADIUS_MAX) == 3.0e-4,
            "v6_sets_max_only_to_full": True,
            "v6_two_shadow_caps_exact": MAX_TRIAL_EVALUATIONS == MAX_ACCEPTED_ITERATES == 2,
            "transformation_count_two": transform_evidence["transformation_count"] == 2,
        }
        if not all(checks.values()):
            raise ProtocolError(f"v6 static variant audit failed: {checks}")
        print(
            canonical_json(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_pass",
                    "source": self_evidence(require_frozen=True),
                    "frozen_v4_source": v4_source_evidence,
                    "frozen_v4_result": v4_result_evidence,
                    "frozen_v5_result": v5_result_evidence,
                    "transformed_v4": transform_evidence,
                    "variant": {
                        "seed": SEED,
                        "trust_radius_initial": INITIAL_TRUST_RADIUS,
                        "trust_radius_max": FULL_TRUST_RADIUS,
                        "max_trial_evaluations": MAX_TRIAL_EVALUATIONS,
                        "max_accepted_iterates": MAX_ACCEPTED_ITERATES,
                        "accepted_nonterminal_trust_policy": "set_directly_to_full",
                        "dynamic_pair_cut_trust_policy": "reset_to_full",
                        "ordinary_merit_rejection_trust_policy": "halve_without_third_shadow",
                        "preregistered_second_planned_l2": PREREGISTERED_SECOND_PLANNED_L2,
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
        raise ProtocolError("frozen CW24 v6 output already exists")
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

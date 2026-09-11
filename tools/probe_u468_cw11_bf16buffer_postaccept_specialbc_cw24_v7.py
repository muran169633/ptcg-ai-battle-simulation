#!/usr/bin/env python3
"""CW24 v7 two-shadow BF16-buffered post-accept special-BC probe.

V7 preserves the frozen v4 0.00025 stepping stone.  At the second
current-point linearization only, four empirically under-delivered native
pair constraints receive the smallest preregistered robustness request:
nextafter(half of the same-point native BF16 quantum, +infinity).  Actual
training gates remain unchanged.  The complete buffered QP endpoint must fit
inside the unchanged 0.00099998 CW11 radius; it is never radius-scaled.
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
SCRIPT = TOOLS / "probe_u468_cw11_bf16buffer_postaccept_specialbc_cw24_v7.py"
V4_SOURCE = TOOLS / "probe_u468_cw11_native_cuttingplane_specialbc_cw24_v4.py"
V4_SOURCE_SHA256 = "134eebec7b1e54b68bd47256fde5c8b80dea3cd8d8ea4399428ec4c1de7d1454"
V4_SOURCE_MODE = 0o555
V4_RESULT = ROOT / "artifacts/cw24_cw11_native_cuttingplane_specialbc_trainonly_v4.json"
V4_RESULT_SHA256 = "db0c2c356fac5779e14698e3ff3ee9b36967f2aa6ddecee55531bb37e8b3377c"
V5_RESULT = ROOT / "artifacts/cw24_cw11_fullstep_cuttingplane_specialbc_trainonly_v5.json"
V5_RESULT_SHA256 = "7dcb9ad7d1e86c046fe623dd47d9fb9da2b80f97b49cd6ebd379991f6f052793"
V6_SOURCE = TOOLS / "probe_u468_cw11_postaccept_fullstep_specialbc_cw24_v6.py"
V6_SOURCE_SHA256 = "98cb3e3fb0e62ba1f8faf7185712cfcbc11639ed396433782ec8b02242d57883"
V6_RESULT = ROOT / "artifacts/cw24_cw11_postaccept_fullstep_specialbc_trainonly_v6.json"
V6_RESULT_SHA256 = "d0a33f55d11df06ce5496a84bd07f1624a56d8fed726efea50b96a8c1e5bd616"
OUTPUT = ROOT / "artifacts/cw24_cw11_bf16buffer_postaccept_specialbc_trainonly_v7.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-bf16buffer-postaccept-specialbc-cw24-v7"
SEED = 202608047
INITIAL_TRUST_RADIUS = 2.5e-4
TOTAL_RADIUS_CAP = 9.9998e-4
MAX_TRIAL_EVALUATIONS = 2
MAX_ACCEPTED_ITERATES = 2
BUFFER_FRACTION = 0.5
BUFFERED_PAIR_NAMES = (
    "pf7_boundary",
    "dominic_closest_margin",
    "historical_retention_0142a2a3d5bd",
    "historical_top1_fde6fab07449",
)


class ProtocolError(RuntimeError):
    """Fail-closed CW24 v7 wrapper error."""


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
        "four_buffer_names": len(BUFFERED_PAIR_NAMES) == 4
        and source.count(b"BUFFERED_PAIR_" + b"NAMES =") == 1,
        "half_quantum_exact": BUFFER_FRACTION == 0.5,
        "two_shadow_caps": MAX_TRIAL_EVALUATIONS == MAX_ACCEPTED_ITERATES == 2,
        "all_frozen_locks_present": all(
            value.encode() in source
            for value in (
                V4_SOURCE_SHA256,
                V4_RESULT_SHA256,
                V5_RESULT_SHA256,
                V6_SOURCE_SHA256,
                V6_RESULT_SHA256,
            )
        ),
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "no_payload_construction": (b"candidate_actor_" + b"float32_le") not in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v7 source audit failed: {checks}")
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
                raise ProtocolError("short v7 result write")
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
        raise ProtocolError(f"v7 result publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def replace_once(source: str, old: str, new: str, label: str, audits: list[dict[str, Any]]) -> str:
    count = source.count(old)
    if count != 1:
        raise ProtocolError(f"{label}: expected one source match, observed {count}")
    audits.append(
        {
            "label": label,
            "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
            "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
            "occurrences": count,
        }
    )
    return source.replace(old, new, 1)


def transformed_v4_module() -> tuple[ModuleType, dict[str, Any]]:
    raw, source_evidence = regular_source(
        V4_SOURCE, V4_SOURCE_SHA256, V4_SOURCE_MODE, "frozen CW24 v4 source"
    )
    source = raw.decode()
    audits: list[dict[str, Any]] = []
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
    source = replace_once(
        source, rejection_old, rejection_new, "dynamic_cut_resets_full_trust", audits
    )
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
    source = replace_once(
        source, accepted_old, accepted_new, "accepted_nonterminal_sets_full_trust", audits
    )
    pair_old = '''            objectives: list[tuple[str, Any, float, str]] = []
            for name, spec in pair_specs.items():
                objective = logits[spec["index"], spec["expert"]] - logits[
                    spec["index"], spec["threat"]
                ]
                objectives.append((name, objective, float(spec["target"]), "native_pair_margin"))
'''
    pair_new = '''            objectives: list[tuple[str, Any, float, str]] = []
            optimization_buffer_audit: dict[str, Any] = {}
            native_policy_cpu = outputs["policy_logits"].detach().cpu()
            for name, spec in pair_specs.items():
                objective = logits[spec["index"], spec["expert"]] - logits[
                    spec["index"], spec["threat"]
                ]
                gate_target = float(spec["target"])
                solver_target = gate_target
                if accepted_iterates >= 1 and name in BUFFERED_PAIR_NAMES:
                    expert_native = native_policy_cpu[spec["index"], spec["expert"]]
                    threat_native = native_policy_cpu[spec["index"], spec["threat"]]
                    spacings = (
                        torch.nextafter(expert_native, torch.full_like(expert_native, float("inf"))) - expert_native,
                        expert_native - torch.nextafter(expert_native, torch.full_like(expert_native, float("-inf"))),
                        torch.nextafter(threat_native, torch.full_like(threat_native, float("inf"))) - threat_native,
                        threat_native - torch.nextafter(threat_native, torch.full_like(threat_native, float("-inf"))),
                    )
                    spacing_values = [float(value.float()) for value in spacings]
                    native_quantum = max(spacing_values)
                    if not all(math.isfinite(value) and value > 0.0 for value in spacing_values):
                        raise ProtocolError(f"{name}: invalid same-point native BF16 spacing")
                    buffer = math.nextafter(BUFFER_FRACTION * native_quantum, math.inf)
                    solver_target += buffer
                    optimization_buffer_audit[name] = {
                        "gate_target": gate_target,
                        "native_BF16_spacings": spacing_values,
                        "native_BF16_quantum": native_quantum,
                        "buffer_fraction": BUFFER_FRACTION,
                        "optimization_only_buffer": buffer,
                        "solver_target": solver_target,
                    }
                objectives.append((name, objective, solver_target, "native_pair_margin"))
'''
    source = replace_once(
        source, pair_old, pair_new, "second_point_half_native_quantum_buffers", audits
    )
    audit_old = '''                "aggregate_constraint_count": len(LOSS_TARGETS),
                "constraint_semantics": (
'''
    audit_new = '''                "aggregate_constraint_count": len(LOSS_TARGETS),
                "optimization_pair_buffers": optimization_buffer_audit,
                "constraint_semantics": (
'''
    source = replace_once(
        source, audit_old, audit_new, "buffer_audit_in_linearization", audits
    )
    radius_old = '''            alpha_trust = min(1.0, trust_radius / step_l2)
            alpha_global = min(1.0, maximum_radius_alpha(current_x, step, TOTAL_RADIUS_CAP, np))
            alpha = min(alpha_trust, alpha_global)
            if alpha <= 1.0e-12:
                terminal_reason = "CW11_RADIUS_EXHAUSTED_BEFORE_GATE_PASS"
                break
'''
    radius_new = '''            alpha_trust = min(1.0, trust_radius / step_l2)
            alpha_global = min(1.0, maximum_radius_alpha(current_x, step, TOTAL_RADIUS_CAP, np))
            optimization_buffers = linearization_audit.get("optimization_pair_buffers", {})
            full_endpoint_l2 = float(np.linalg.norm(current_x + step))
            if optimization_buffers and (
                step_l2 > trust_radius + 1.0e-15
                or full_endpoint_l2 > TOTAL_RADIUS_CAP + 1.0e-15
                or alpha_trust < 1.0
                or alpha_global < 1.0
            ):
                terminal_reason = "BUFFERED_FULL_ENDPOINT_EXCEEDS_UNCHANGED_CW11_RADIUS"
                iteration_log.append(
                    {
                        "trial_one_based": trial_evaluations + 1,
                        "candidate_evaluated": False,
                        "accepted_before_trial": accepted_iterates,
                        "trust_radius": trust_radius,
                        "full_unscaled_step_l2": step_l2,
                        "full_unscaled_endpoint_l2": full_endpoint_l2,
                        "total_radius_cap": TOTAL_RADIUS_CAP,
                        "alpha_trust_would_be": alpha_trust,
                        "alpha_global_would_be": alpha_global,
                        "current_merit": float(current_state["merit"]),
                        "linearization": linearization_audit,
                        "solver": solver_audit,
                        "reason": terminal_reason,
                    }
                )
                break
            alpha = min(alpha_trust, alpha_global)
            if optimization_buffers and alpha != 1.0:
                raise ProtocolError("buffered endpoint attempted a scaled alpha")
            if alpha <= 1.0e-12:
                terminal_reason = "CW11_RADIUS_EXHAUSTED_BEFORE_GATE_PASS"
                break
'''
    source = replace_once(
        source, radius_old, radius_new, "buffered_endpoint_never_radius_scaled", audits
    )
    transformed = source.encode()
    module_name = "cw24_v7_transformed_frozen_v4"
    if module_name in sys.modules:
        raise ProtocolError("v7 transformed v4 module name occupied")
    module = types.ModuleType(module_name)
    module.__file__ = str(V4_SOURCE)
    module.__package__ = ""
    module.__dict__["BUFFER_FRACTION"] = BUFFER_FRACTION
    module.__dict__["BUFFERED_PAIR_NAMES"] = frozenset(BUFFERED_PAIR_NAMES)
    sys.modules[module_name] = module
    try:
        exec(compile(transformed, str(V4_SOURCE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    checks = {
        "source_total_radius_exact": float(module.TOTAL_RADIUS_CAP) == TOTAL_RADIUS_CAP,
        "source_initial_radius_exact": float(module.TRUST_RADIUS_INITIAL) == INITIAL_TRUST_RADIUS,
        "source_max_radius_was_3e4": float(module.TRUST_RADIUS_MAX) == 3.0e-4,
        "source_trial_cap_was48": int(module.MAX_TRIAL_EVALUATIONS) == 48,
        "source_accepted_cap_was16": int(module.MAX_ACCEPTED_ITERATES) == 16,
        "five_transforms": len(audits) == 5,
        "all_transforms_unique": all(item["occurrences"] == 1 for item in audits),
        "no_old_conditional_growth": b"trust_radius * 1.5" not in transformed,
        "buffer_names_in_module_exact": set(module.BUFFERED_PAIR_NAMES)
        == set(BUFFERED_PAIR_NAMES),
    }
    if not all(checks.values()):
        raise ProtocolError(f"v7 transformed-v4 audit failed: {checks}")
    return module, {
        "frozen_v4_source": source_evidence,
        "transformed_source_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_source_bytes": len(transformed),
        "transformations": audits,
        "transformation_count": len(audits),
        "checks": checks,
    }


def prior_evidence() -> dict[str, Any]:
    entries = {}
    for label, path, digest, mode in (
        ("v4_result", V4_RESULT, V4_RESULT_SHA256, 0o444),
        ("v5_result", V5_RESULT, V5_RESULT_SHA256, 0o444),
        ("v6_source", V6_SOURCE, V6_SOURCE_SHA256, 0o555),
        ("v6_result", V6_RESULT, V6_RESULT_SHA256, 0o444),
    ):
        _, entries[label] = regular_source(path, digest, mode, f"frozen CW24 {label}")
    return entries


def production_run() -> dict[str, Any]:
    source = self_evidence(require_frozen=True)
    previous = prior_evidence()
    v4, transform_evidence = transformed_v4_module()
    originals = {
        "TRUST_RADIUS_MAX": v4.TRUST_RADIUS_MAX,
        "MAX_TRIAL_EVALUATIONS": v4.MAX_TRIAL_EVALUATIONS,
        "MAX_ACCEPTED_ITERATES": v4.MAX_ACCEPTED_ITERATES,
        "SEED": v4.SEED,
    }
    constants_before = {
        "total_radius_exact": float(v4.TOTAL_RADIUS_CAP) == TOTAL_RADIUS_CAP,
        "initial_trust_retained": float(v4.TRUST_RADIUS_INITIAL) == INITIAL_TRUST_RADIUS,
        "max_trust_original": float(v4.TRUST_RADIUS_MAX) == 3.0e-4,
        "trial_cap_original": int(v4.MAX_TRIAL_EVALUATIONS) == 48,
        "accepted_cap_original": int(v4.MAX_ACCEPTED_ITERATES) == 16,
    }
    if not all(constants_before.values()):
        raise ProtocolError(f"v7 frozen v4 constants drift: {constants_before}")
    v4.TRUST_RADIUS_MAX = TOTAL_RADIUS_CAP
    v4.MAX_TRIAL_EVALUATIONS = MAX_TRIAL_EVALUATIONS
    v4.MAX_ACCEPTED_ITERATES = MAX_ACCEPTED_ITERATES
    v4.SEED = SEED
    try:
        result = v4.production_run()
    finally:
        for name, value in originals.items():
            setattr(v4, name, value)

    endpoint = result["endpoint"]
    cuttingplane = endpoint.get("cuttingplane", {})
    iteration_log = list(cuttingplane.get("iteration_log", []))
    evaluated = [row for row in iteration_log if bool(row.get("candidate_evaluated", True))]
    unevaluated = [row for row in iteration_log if row.get("candidate_evaluated") is False]
    first = evaluated[0] if evaluated else {}
    buffered_rows = [
        row
        for row in iteration_log
        if row.get("linearization", {}).get("optimization_pair_buffers")
    ]
    buffer_names_observed = set()
    for row in buffered_rows:
        buffer_names_observed.update(
            row["linearization"]["optimization_pair_buffers"].keys()
        )
    shadow_checks = {
        "one_or_two_evaluated_shadows": 1 <= len(evaluated) <= 2,
        "reported_shadow_count_exact": int(cuttingplane.get("trial_evaluations", -1))
        == len(evaluated),
        "first_shadow_exact_stepping_stone": len(evaluated) >= 1
        and float(first.get("trust_radius", -1.0)) == INITIAL_TRUST_RADIUS
        and first.get("accepted") is True
        and first.get("gate_pass") is False,
        "buffered_linearization_present": len(buffered_rows) == 1,
        "buffer_names_exact": buffer_names_observed == set(BUFFERED_PAIR_NAMES),
        "no_third_candidate": len(evaluated) <= MAX_TRIAL_EVALUATIONS,
        "pre_candidate_radius_close_or_second_evaluated": len(unevaluated) == 1
        or len(evaluated) == 2,
        "unevaluated_row_only_for_radius": not unevaluated
        or (
            len(unevaluated) == 1
            and unevaluated[0].get("reason")
            == "BUFFERED_FULL_ENDPOINT_EXCEEDS_UNCHANGED_CW11_RADIUS"
        ),
        "evaluated_buffered_endpoint_unscaled": len(evaluated) < 2
        or float(evaluated[1].get("alpha", -1.0)) == 1.0,
    }
    if not all(shadow_checks.values()):
        raise ProtocolError(f"v7 two-shadow/buffer contract drift: {shadow_checks}")

    old_decision = str(endpoint["decision"])
    old_reason = str(endpoint.get("reason", ""))
    if old_decision == "GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE":
        decision = "GO_CW24_V7_BF16BUFFER_POSTACCEPT_TRAIN_GATE"
    elif old_decision.startswith("NO_GO_CW24_V4"):
        decision = "NO_GO_CW24_V7_BF16BUFFER_POSTACCEPT_TRAIN_GATE"
    else:
        raise ProtocolError(f"unexpected v4 endpoint decision: {old_decision}")
    endpoint["decision"] = decision
    endpoint["reason"] = (
        "V7_BF16BUFFER_" + old_reason
        if old_reason
        else "V7_BF16BUFFER_INHERITED_ENDPOINT_REASON_MISSING"
    )
    payload_present = endpoint.get("candidate_payload") is not None
    expected_payload = (
        decision == "GO_CW24_V7_BF16BUFFER_POSTACCEPT_TRAIN_GATE"
        and endpoint.get("trial", {}).get("pass") is True
        and endpoint.get("promotable_terminal_endpoint_count") == 1
    )
    terminal_payload_gate = payload_present == expected_payload
    if not terminal_payload_gate:
        raise ProtocolError("v7 terminal-only payload gate drift")

    result["schema_version"] = SCHEMA
    result["seed"] = SEED
    result["status"] = decision
    result["decision"] = decision
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_BF16buffered_postaccept_special_BC"
    )
    result["selection"]["optimization_contract"].update(
        {
            "two_train_shadows_maximum": True,
            "first_shadow_trust_radius": INITIAL_TRUST_RADIUS,
            "second_point_optimization_only_buffer_names": list(BUFFERED_PAIR_NAMES),
            "same_point_native_BF16_quantum_fraction": BUFFER_FRACTION,
            "buffer_uses_nextafter_toward_positive_infinity": True,
            "actual_native_pair_gates_unchanged": True,
            "buffered_full_endpoint_must_fit_without_scaling": True,
            "total_additional_from_CW11_cap": TOTAL_RADIUS_CAP,
            "terminal_only_payload_gate": terminal_payload_gate,
        }
    )
    result["inputs"]["CW24_v7_source"] = source
    result["inputs"]["CW24_v7_transformed_frozen_v4"] = transform_evidence
    for name, evidence in previous.items():
        result["inputs"][f"CW24_frozen_{name}"] = evidence
    result["audit"]["CW24_v7_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "train_only_B352": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "constants_before": constants_before,
        "runtime_constant_changes": {
            "TRUST_RADIUS_MAX": TOTAL_RADIUS_CAP,
            "MAX_TRIAL_EVALUATIONS": MAX_TRIAL_EVALUATIONS,
            "MAX_ACCEPTED_ITERATES": MAX_ACCEPTED_ITERATES,
            "SEED": SEED,
        },
        "shadow_and_buffer_checks": shadow_checks,
        "buffered_candidate_evaluated": len(evaluated) == 2,
        "pre_candidate_radius_close": len(unevaluated) == 1,
        "terminal_only_payload_gate": terminal_payload_gate,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "v7_selected_only_from_frozen_v4_v5_v6_NO_GO_evidence": True,
        "buffer_values_are_same_point_native_BF16_derived_not_grid_searched": True,
        "buffered_endpoint_not_scaled_if_over_radius": True,
        "internal_shadows_are_not_separately_promotable_endpoints": True,
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
        previous = prior_evidence()
        v4, transform_evidence = transformed_v4_module()
        checks = {
            "total_radius_exact": float(v4.TOTAL_RADIUS_CAP) == TOTAL_RADIUS_CAP,
            "initial_trust_exact": float(v4.TRUST_RADIUS_INITIAL) == INITIAL_TRUST_RADIUS,
            "original_max_trust_exact": float(v4.TRUST_RADIUS_MAX) == 3.0e-4,
            "four_buffer_names_exact": set(v4.BUFFERED_PAIR_NAMES)
            == set(BUFFERED_PAIR_NAMES),
            "five_transforms": transform_evidence["transformation_count"] == 5,
            "two_shadow_caps": MAX_TRIAL_EVALUATIONS == MAX_ACCEPTED_ITERATES == 2,
        }
        if not all(checks.values()):
            raise ProtocolError(f"v7 static variant audit failed: {checks}")
        print(
            canonical_json(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_pass",
                    "source": self_evidence(require_frozen=True),
                    "previous_frozen_inputs": previous,
                    "transformed_v4": transform_evidence,
                    "variant": {
                        "seed": SEED,
                        "initial_trust_radius": INITIAL_TRUST_RADIUS,
                        "total_radius_cap": TOTAL_RADIUS_CAP,
                        "buffer_fraction": BUFFER_FRACTION,
                        "buffered_pair_names": list(BUFFERED_PAIR_NAMES),
                        "max_trial_evaluations": MAX_TRIAL_EVALUATIONS,
                        "max_accepted_iterates": MAX_ACCEPTED_ITERATES,
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
        raise ProtocolError("frozen CW24 v7 output already exists")
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
                "buffered_candidate_evaluated": result["audit"]["CW24_v7_contract"]
                ["buffered_candidate_evaluated"],
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

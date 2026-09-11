#!/usr/bin/env python3
"""CW24 v11 tie-aware nonlinear native-BF16 special-BC probe.

The frozen CW24 v4 engine is reused for its authenticated CW11 reconstruction,
B352 train-only cache, native BF16 forward path, absolute actor6 replay, radius
projection, cutting-plane loop, and terminal-only payload gate.  This variant
changes only the feasibility semantics that later evidence showed were too
strong:

* the three PokemonFan boundary repairs remain hard native requirements;
* zero-margin rows and historical retention rows are guarded by actual native
  ordered/top1 behavior, not by continuous raw-margin inequalities;
* a preservation cut is added only after an actual native flip, and asks for
  at most one native tick rather than restoration of the whole CW11 cushion;
* aggregate NLL/CE targets remain deterministic soft merit terms, never hard
  feasibility constraints.

The probe is local and train-only.  It writes no checkpoint, opens no new
validation/test rows, performs no network action, and exposes actor bytes only
if the sole terminal endpoint passes every native preservation and radius gate.
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
SCRIPT = TOOLS / "probe_u468_cw11_tieaware_native_specialbc_cw24_v11.py"
V4_SOURCE = TOOLS / "probe_u468_cw11_native_cuttingplane_specialbc_cw24_v4.py"
V4_SOURCE_SHA256 = "134eebec7b1e54b68bd47256fde5c8b80dea3cd8d8ea4399428ec4c1de7d1454"
V4_SOURCE_MODE = 0o555
V4_RESULT = ROOT / "artifacts/cw24_cw11_native_cuttingplane_specialbc_trainonly_v4.json"
V4_RESULT_SHA256 = "db0c2c356fac5779e14698e3ff3ee9b36967f2aa6ddecee55531bb37e8b3377c"
V4_RESULT_MODE = 0o444
V8_SOURCE = TOOLS / "probe_u468_cw11_absolute_residual_specialbc_cw24_v8.py"
V8_SOURCE_SHA256 = "bb22306ed4e1379d49357910485e9c1efc3fe40565a6adce67e1e5462f6177da"
V8_SOURCE_MODE = 0o555
V8_RESULT = ROOT / "artifacts/cw24_cw11_absolute_residual_specialbc_trainonly_v8.json"
V8_RESULT_SHA256 = "5341aa76c33276901bff1acb5ba5fa9810620ee4d055f78a016494b040c5e908"
V8_RESULT_MODE = 0o444
V10_SOURCE = TOOLS / "probe_u468_cw11_v1_v6_halfangle_exact_intervals_cw24_v10b.py"
V10_SOURCE_SHA256 = "58df3ad320e502f3aea2c5274bae791e86c4144d1b5847fa869c220e8ba3a9bf"
V10_SOURCE_MODE = 0o555
V10_RESULT = ROOT / "artifacts/cw24_cw11_v1_v6_halfangle_exact_intervals_cw24_v10b.json"
V10_RESULT_SHA256 = "8724c70673ef66d77e262e59511d8435498e40463a3c229ac5606efb4c062487"
V10_RESULT_MODE = 0o444

OUTPUT = ROOT / "artifacts/cw24_cw11_tieaware_native_specialbc_trainonly_v11.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-tieaware-native-specialbc-cw24-v11"
SEED = 202608051
TOTAL_RADIUS_CAP = 9.9998e-4
INITIAL_TRUST_RADIUS = 2.5e-4
MAX_TRAIN_SHADOWS = 24
MAX_ACCEPTED_ITERATES = 12
SOFT_MERIT_WEIGHT = 0.05
ONE_NATIVE_TICK = 1.0 / 512.0


class ProtocolError(RuntimeError):
    """Fail-closed CW24 v11 protocol error."""


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
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "no_network_library": (b"re" + b"quests") not in source
        and (b"url" + b"lib") not in source,
        "single_output_publication": source.count(
            b"publish_o_excl(" + b"OUTPUT"
        )
        == 1,
        "tieaware_transform_present": b"hard_pair_constraint" in source,
        "soft_loss_transform_present": b"soft_merit_only_no_hard_aggregate_rows" in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v11 source audit failed: {checks}")
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
    fd = os.open(path, flags, 0o444)
    try:
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise ProtocolError("short v11 result write")
            written += count
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
        raise ProtocolError(f"v11 publication drift: {checks}")
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

    helper_marker = "        def evaluate_outputs(candidate_outputs: Mapping[str, Any], candidate_x: Any) -> dict[str, Any]:\n"
    helper_replacement = '''        def hard_pair_constraint(spec: Mapping[str, Any]) -> bool:
            source_name = str(spec["source"])
            return source_name in {
                "fixed_PF_native_target",
                "rejected_native_PF_alternate_threat",
                "rejected_native_top1_flip",
                "rejected_native_ordered_flip",
            }

        def minimal_native_preservation_target(index: int, expert: int, threat: int) -> float:
            baseline_margin = float(
                baseline_policy_cpu[index, expert] - baseline_policy_cpu[index, threat]
            )
            if baseline_margin < 0.0:
                raise ProtocolError("baseline preservation pair was not native-correct")
            return min(baseline_margin, 1.0 / 512.0)

        def evaluate_outputs(candidate_outputs: Mapping[str, Any], candidate_x: Any) -> dict[str, Any]:
'''
    source = splice_exact_once(
        source, helper_marker, helper_replacement, "insert_tieaware_pair_semantics", audits
    )

    checks_start = "            train_checks = {\n"
    checks_end = "            merit = float(sum(value * value for value in core_deficits.values()))\n"
    checks_replacement = '''            hard_pair_names = sorted(
                name for name, spec in pair_specs.items() if hard_pair_constraint(spec)
            )
            train_checks = {
                "final_three_PF_targets_ordered_repaired": all(
                    row["ordered_correct"] for row in target_observed.values()
                ),
                "final_three_PF_targets_set_repaired": all(
                    row["set_correct"] for row in target_observed.values()
                ),
                "final_three_PF_native_margin_thresholds": all(
                    target_observed[name]["candidate_fixed_threat_margin"]
                    >= PF_FINAL_NATIVE_MARGIN[name]
                    for name in PF_FINAL_NATIVE_MARGIN
                ),
                "PF0_repairs_at_least2": repairs_by_stratum["pf_ctx0_hard"] >= 2,
                "PF7_repairs_at_least1": repairs_by_stratum["pf_ctx7_hard"] >= 1,
                "top1_guard96_zero_flips": not top1_flips,
                "retention160_zero_ordered_flips": not retention_flips,
                "six_zero_guards_stay_ordered_correct": all(
                    row["ordered_correct"] for row in zero_guard_observed
                ),
                "all_fixed_PF_native_pair_thresholds": all(
                    pair_margins[name] >= pair_specs[name]["target"]
                    for name in PF_FINAL_NATIVE_MARGIN
                ),
                "count_logits_native_exact": immutable_native["count_logits_native_exact"],
                "value_logits_native_exact": immutable_native["value_logits_native_exact"],
                "policy_logits_native_BF16": immutable_native["policy_logits_native_BF16"],
            }
            soft_objectives = {
                "classification": "soft_merit_only_no_hard_aggregate_rows",
                "targets": dict(LOSS_TARGETS),
                "observed_improvements": dict(loss_improvements),
                "weight": 0.05,
            }
            core_deficits = {
                f"hard_pair::{name}": max(
                    0.0, float(spec["target"]) - pair_margins[name]
                ) / (1.0 / 512.0)
                for name, spec in pair_specs.items()
                if hard_pair_constraint(spec)
            }
            for name, target in LOSS_TARGETS.items():
                core_deficits[f"soft::{name}"] = 0.05 * max(
                    0.0, target - loss_improvements[name]
                ) / MERIT_SCALES[name]
            merit = float(sum(value * value for value in core_deficits.values()))
'''
    source = splice_region_once(
        source,
        checks_start,
        checks_end,
        checks_replacement,
        "native_only_hard_gates_and_soft_aggregate_merit",
        audits,
    )

    return_marker = '                "loss_improvements": loss_improvements,\n'
    return_replacement = '''                "loss_improvements": loss_improvements,
                "soft_objectives": soft_objectives,
                "hard_pair_names": hard_pair_names,
'''
    source = splice_exact_once(
        source, return_marker, return_replacement, "publish_soft_and_hard_semantics", audits
    )

    pair_loop_old = '''            for name, spec in pair_specs.items():
                objective = logits[spec["index"], spec["expert"]] - logits[
                    spec["index"], spec["threat"]
                ]
                objectives.append((name, objective, float(spec["target"]), "native_pair_margin"))
'''
    pair_loop_new = '''            for name, spec in pair_specs.items():
                if not hard_pair_constraint(spec):
                    continue
                objective = logits[spec["index"], spec["expert"]] - logits[
                    spec["index"], spec["threat"]
                ]
                objectives.append((name, objective, float(spec["target"]), "native_pair_margin"))
'''
    source = splice_exact_once(
        source, pair_loop_old, pair_loop_new, "hard_pairs_only_in_tangent_system", audits
    )

    aggregate_old = '''            for name in LOSS_TARGETS:
                objectives.append((name, loss_tensors[name], LOSS_TARGETS[name], "aggregate_improvement"))
'''
    aggregate_new = '''            # Aggregate losses are deterministic soft merit terms only.  They are
            # intentionally absent from the hard tangent feasibility system.
            soft_merit_only_no_hard_aggregate_rows = tuple(loss_tensors)
            if set(soft_merit_only_no_hard_aggregate_rows) != set(LOSS_TARGETS):
                raise ProtocolError("soft aggregate objective identity drift")
'''
    source = splice_exact_once(
        source, aggregate_old, aggregate_new, "remove_aggregate_hard_rows", audits
    )
    source = splice_exact_once(
        source,
        '                "aggregate_constraint_count": len(LOSS_TARGETS),\n',
        '                "aggregate_constraint_count": 0,\n                "soft_aggregate_objective_count": len(LOSS_TARGETS),\n',
        "truthful_constraint_counts",
        audits,
    )

    target_old = '''                target = float(
                    baseline_policy_cpu[index, expert] - baseline_policy_cpu[index, threat]
                )
'''
    if source.count(target_old) != 2:
        raise ProtocolError(
            f"minimal preservation target: expected two sites, observed {source.count(target_old)}"
        )
    target_new = '''                target = minimal_native_preservation_target(
                    index, expert, threat
                )
'''
    for offset in range(2):
        expected_occurrences = 2 - offset
        observed_occurrences = source.count(target_old)
        if observed_occurrences != expected_occurrences:
            raise ProtocolError(
                "minimal preservation target site-count drift: "
                f"expected {expected_occurrences}, observed {observed_occurrences}"
            )
        audits.append(
            {
                "label": f"minimal_native_preservation_target_{offset + 1}",
                "kind": "ordered_identical_site_replacement",
                "old_sha256": hashlib.sha256(target_old.encode()).hexdigest(),
                "new_sha256": hashlib.sha256(target_new.encode()).hexdigest(),
                "occurrences_before": observed_occurrences,
            }
        )
        source = source.replace(target_old, target_new, 1)

    transformed = source.encode()
    module_name = f"_cw24_v11_transformed_v4_{hashlib.sha256(transformed).hexdigest()[:16]}"
    module = types.ModuleType(module_name)
    module.__file__ = str(V4_SOURCE)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(transformed, str(V4_SOURCE), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module, {
        "source": source_evidence,
        "transformed_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_bytes": len(transformed),
        "splice_count": len(audits),
        "splices": audits,
        "semantics": {
            "initial_hard_pairs": "three fixed PokemonFan native repair targets only",
            "zero_margin_guards": "native BF16 ordered correctness only",
            "retention_and_top1": "native no-flip plus post-flip minimal cuts",
            "dynamic_preservation_target": "min(CW11_pair_margin,1/512)",
            "aggregate_NLL_CE": "soft merit only",
        },
    }


def frozen_dependencies() -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for key, path, digest, mode, label in (
        ("v4_source", V4_SOURCE, V4_SOURCE_SHA256, V4_SOURCE_MODE, "frozen v4 source"),
        ("v4_result", V4_RESULT, V4_RESULT_SHA256, V4_RESULT_MODE, "frozen v4 result"),
        ("v8_source", V8_SOURCE, V8_SOURCE_SHA256, V8_SOURCE_MODE, "frozen v8 source"),
        ("v8_result", V8_RESULT, V8_RESULT_SHA256, V8_RESULT_MODE, "frozen v8 result"),
        ("v10_source", V10_SOURCE, V10_SOURCE_SHA256, V10_SOURCE_MODE, "frozen v10b source"),
        ("v10_result", V10_RESULT, V10_RESULT_SHA256, V10_RESULT_MODE, "frozen v10b result"),
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
        "soft_weight_exact": SOFT_MERIT_WEIGHT == 0.05,
        "native_tick_exact": ONE_NATIVE_TICK == 1.0 / 512.0,
    }
    if not all(constant_checks.values()):
        raise ProtocolError(f"v11 frozen-v4 constant drift: {constant_checks}")
    v4.TRUST_RADIUS_MAX = TOTAL_RADIUS_CAP
    v4.MAX_TRIAL_EVALUATIONS = MAX_TRAIN_SHADOWS
    v4.MAX_ACCEPTED_ITERATES = MAX_ACCEPTED_ITERATES
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
        decision = "GO_CW24_V11_TIEAWARE_NATIVE_SPECIALBC_TRAIN_GATE"
    elif old_decision.startswith("NO_GO_CW24_V4"):
        decision = "NO_GO_CW24_V11_TIEAWARE_NATIVE_SPECIALBC_TRAIN_GATE"
    else:
        raise ProtocolError(f"unexpected inherited v4 decision: {old_decision}")
    endpoint["decision"] = decision
    endpoint["reason"] = (
        "V11_TIEAWARE_NATIVE_" + old_reason
        if old_reason
        else "V11_TIEAWARE_NATIVE_INHERITED_REASON_MISSING"
    )

    cuttingplane = endpoint.get("cuttingplane", {})
    iteration_log = cuttingplane.get("iteration_log", [])
    payload_present = endpoint.get("candidate_payload") is not None
    expected_payload = (
        decision == "GO_CW24_V11_TIEAWARE_NATIVE_SPECIALBC_TRAIN_GATE"
        and endpoint.get("trial", {}).get("pass") is True
        and endpoint.get("promotable_terminal_endpoint_count") == 1
    )
    runtime_checks = {
        "terminal_only_payload_gate": payload_present == expected_payload,
        "trial_count_within_cap": 0
        <= int(cuttingplane.get("trial_evaluations", 0))
        <= MAX_TRAIN_SHADOWS,
        "accepted_count_within_cap": 0
        <= int(cuttingplane.get("accepted_iterates", 0))
        <= MAX_ACCEPTED_ITERATES,
        "every_evaluated_trial_native_preservation_reported": all(
            "top1_flip_count" in row and "retention_flip_count" in row
            for row in iteration_log
            if bool(row.get("candidate_evaluated", True))
        ),
        "no_official_candidate_consumed": True,
    }
    if not all(runtime_checks.values()):
        raise ProtocolError(f"v11 runtime contract drift: {runtime_checks}")

    result["schema_version"] = SCHEMA
    result["seed"] = SEED
    result["status"] = decision
    result["decision"] = decision
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_CW11_then_tieaware_native_nonlinear_special_BC"
    )
    result["selection"]["optimization_contract"].update(
        {
            "three_PF_native_repairs_hard": True,
            "zero_margin_guards_native_actual_forward_only": True,
            "retention_and_top1_native_no_flip_only": True,
            "post_flip_minimal_cut_target": "min(CW11_pair_margin,1/512)",
            "dynamic_pair_thresholds_optimizer_only_not_terminal_gates": True,
            "aggregate_NLL_CE_soft_merit_only": True,
            "soft_merit_weight": SOFT_MERIT_WEIGHT,
            "initial_trust_radius": INITIAL_TRUST_RADIUS,
            "maximum_trust_radius": TOTAL_RADIUS_CAP,
            "total_additional_from_CW11_cap": TOTAL_RADIUS_CAP,
            "maximum_train_shadows": MAX_TRAIN_SHADOWS,
            "maximum_accepted_iterates": MAX_ACCEPTED_ITERATES,
            "terminal_only_payload_gate": runtime_checks["terminal_only_payload_gate"],
        }
    )
    result["inputs"]["CW24_v11_source"] = source
    result["inputs"]["CW24_v11_transformed_frozen_v4"] = transform_evidence
    result["inputs"]["CW24_v11_frozen_dependencies"] = dependencies
    result["audit"]["CW24_v11_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated_mode": bool(sys.flags.isolated),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
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
            "MAX_ACCEPTED_ITERATES": MAX_ACCEPTED_ITERATES,
            "SEED": SEED,
        },
        "frozen_v4_constant_checks": constant_checks,
        "runtime_checks": runtime_checks,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "surrogate_relaxation_selected_from_frozen_v8_v10_NO_GO_evidence": True,
        "v10b_only_excluded_the_fixed_radius_v1_v6_minor_arc": True,
        "v8_did_not_prove_full_actor6_native_infeasibility": True,
        "continuous_zero_margin_sign_not_used_as_native_certificate": True,
        "historical_full_cushion_restoration_not_required": True,
        "aggregate_loss_floors_not_hard_feasibility_gates": True,
        "train_shadows_are_not_promotion_evidence": True,
        "fulltrain_then_specialist_then_broad_then_Gold_required_after_GO": True,
    }
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 2
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    canonical_json(result)
    return result


def audit_only() -> dict[str, Any]:
    dependencies = frozen_dependencies()
    v4, transform_evidence = transformed_v4_module()
    checks = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated_mode": bool(sys.flags.isolated),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
        "output_absent": not OUTPUT.exists(),
        "total_radius_exact": float(v4.TOTAL_RADIUS_CAP) == TOTAL_RADIUS_CAP,
        "initial_radius_exact": float(v4.TRUST_RADIUS_INITIAL) == INITIAL_TRUST_RADIUS,
        "original_max_radius_exact": float(v4.TRUST_RADIUS_MAX) == 3.0e-4,
        "splice_count_exact": transform_evidence["splice_count"] == 8,
        "three_initial_hard_PF_pairs": len(v4.PF_FINAL_NATIVE_MARGIN) == 3,
        "soft_loss_identity_exact": len(v4.LOSS_TARGETS) == 7,
        "no_cuda_initialized": "torch" not in sys.modules
        or not bool(getattr(sys.modules["torch"].cuda, "is_initialized", lambda: False)()),
    }
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_pass" if all(checks.values()) else "static_audit_failed",
        "checks": checks,
        "source": self_evidence(require_frozen=False),
        "frozen_dependencies": dependencies,
        "transformed_v4": transform_evidence,
        "authorization": {
            "local_train_only": True,
            "fulltrain": False,
            "specialist": False,
            "broad": False,
            "gold": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        payload = audit_only()
        print(canonical_json(payload).decode(), end="")
        if payload["status"] != "static_audit_pass":
            raise SystemExit(1)
        return
    if OUTPUT.exists():
        raise ProtocolError(f"refusing to overwrite existing v11 result: {OUTPUT}")
    result = production_run()
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "reason": result["endpoint"].get("reason"),
                "candidate_model_state_sha256": result["endpoint"]
                .get("trial", {})
                .get("candidate_model_state_sha256"),
                "candidate_payload_present": result["endpoint"].get("candidate_payload")
                is not None,
                "changed_candidate_train_shadow_count": result["endpoint"].get(
                    "changed_candidate_train_shadow_count"
                ),
                "official_unique_changed_candidate_count_consumed": 0,
                "publication": publication,
            }
        ).decode(),
        end="",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only cumulative trust-region cutting-plane SQP from raw U468.

The run starts at raw plus the frozen r0*e471 actor6 displacement.  On the 19
selected train rows it derives one metric-aware option-pair cut for every false
official obligation, solves a minimum-L2 linearized correction, caps every
correction at the frozen trust radius, and re-evaluates native BF16.  Dynamic
target cuts use the current pair's local positive BF16 nextafter quantum;
dynamic guard cuts preserve the same-process raw margin.  A duplicate key keeps
the larger threshold, and an obligation without a violated separating cut
closes the run without a candidate.

There is no sweep, training optimizer, backward call, checkpoint/result write,
network, validation access, holdout, or full-panel evaluation.  Output is one
JSON document on stdout only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v2.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-cuttingplane-sqp-v2"
SEED = 202608204

GEOMETRY = TOOLS / "probe_u468_raw_actor6_metricguard_specialbc_v1.py"
GEOMETRY_SHA256 = (
    "ddecd3a85bc2b43c28854afc56678c21943198ac3fde3613765d6a49e440eedb"
)
RAM_RUNNER = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_ram_ray_v1.py"
RAM_RUNNER_SHA256 = (
    "86b05d4f826576717907141aef2c534f140a1531b87c8e2d28aeb624a2657e4d"
)
FROZEN_MODE = 0o555
EXPECTED_DIRECTION_SHA256 = (
    "e4714580607e4c9543c026fc4851326526b33612ca9ed39818122e3ec3cffd84"
)
INITIAL_RADIUS = 0.0028737480729610368
MAX_ITER = 12
STEP_L2_CAP = 0.001
MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)
SVD_RELATIVE_RANK_TOLERANCE = 1e-12
QP_FTOL = 1e-12
QP_MAXITER = 5000
QP_DIRECT_TOLERANCE = 1e-8
PAIR_THRESHOLD_TOLERANCE = 0.0
EXPECTED_SELECTED_ROW_COUNT = 19
EXPECTED_TARGET_ROW_COUNT = 5
EXPECTED_GUARD_ROW_COUNT = 14


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_regular_bytes(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"{label} is not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    visible = os.lstat(path)
    identity = (after.st_dev, after.st_ino, after.st_size)
    if (
        (before.st_dev, before.st_ino, before.st_size) != identity
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or visible.st_nlink != 1
        or (visible.st_dev, visible.st_ino, visible.st_size) != identity
    ):
        raise RuntimeError(f"{label} changed during held-fd read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    mode = stat.S_IMODE(after.st_mode)
    if digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode": oct(mode),
        "single_link_regular_held_fd_identity_exact": True,
    }


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_attributes = {
        "backward",
        "step",
        "save",
        "savez",
        "write",
        "write_bytes",
        "write_text",
        "touch",
        "mkdir",
        "unlink",
        "rename",
        "replace",
        "copy_",
        "add_",
    }
    forbidden_names = {"open", "exec", "eval", "compile", "DataLoader"}
    forbidden_os_flags = {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"}
    forbidden_import_roots = {"requests", "urllib", "subprocess"}
    attribute_hits: list[tuple[int, str]] = []
    name_hits: list[tuple[int, str]] = []
    os_flag_hits: list[tuple[int, str]] = []
    import_hits: list[tuple[int, str]] = []
    print_sites: list[int] = []
    os_open_sites: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_import_roots:
                    import_hits.append((node.lineno, name))
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in forbidden_os_flags
        ):
            os_flag_hits.append((node.lineno, node.attr))
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr in forbidden_attributes:
                attribute_hits.append((node.lineno, function.attr))
            if (
                isinstance(function.value, ast.Name)
                and function.value.id == "os"
                and function.attr == "open"
            ):
                os_open_sites.append(node.lineno)
        elif isinstance(function, ast.Name):
            if function.id in forbidden_names:
                name_hits.append((node.lineno, function.id))
            if function.id == "print":
                print_sites.append(node.lineno)
    if attribute_hits or name_hits or os_flag_hits or import_hits:
        raise RuntimeError(
            "zero-write static audit failed: "
            f"{attribute_hits=} {name_hits=} {os_flag_hits=} {import_hits=}"
        )
    if len(print_sites) != 1 or len(os_open_sites) != 1:
        raise RuntimeError("static stdout/read call-site cardinality drift")
    return {
        "ast_parse": True,
        "forbidden_attribute_call_sites": attribute_hits,
        "forbidden_name_call_sites": name_hits,
        "forbidden_os_write_flags": os_flag_hits,
        "forbidden_network_imports": import_hits,
        "stdout_print_call_sites": print_sites,
        "os_open_read_only_call_sites": os_open_sites,
        "direct_parameter_mutation_call_sites": 0,
        "hash_bound_ram_runner_owns_only_actor6_overlay": True,
        "direct_autograd_grad_call_sites": 0,
        "hash_bound_geometry_owns_autograd": True,
        "no_optimizer_backward_step_save_write_or_network": True,
    }


def import_frozen(
    path: Path, digest: str, module_name: str
) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path, digest, module_name, expected_mode=FROZEN_MODE
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct frozen import {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def pair_key(pair: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(pair["row_index"]),
        int(pair["positive_option"]),
        int(pair["negative_option"]),
    )


def pair_margin(snapshot: Mapping[str, Any], pair: Mapping[str, Any]) -> float:
    logits = snapshot["outputs_cpu"]["policy_logits"]
    row = int(pair["row_index"])
    positive = int(pair["positive_option"])
    negative = int(pair["negative_option"])
    return float(logits[row, positive].float() - logits[row, negative].float())


def local_positive_bf16_q(
    snapshot: Mapping[str, Any],
    row_index: int,
    positive: int,
    negative: int,
    torch: Any,
) -> float:
    logits = snapshot["outputs_cpu"]["policy_logits"]
    values = (logits[row_index, positive], logits[row_index, negative])
    spacings = []
    for value in values:
        spacings.extend(
            (
                torch.nextafter(value, torch.full_like(value, float("inf")))
                - value,
                value
                - torch.nextafter(value, torch.full_like(value, float("-inf"))),
            )
        )
    result = max(float(value.float()) for value in spacings)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("dynamic target pair has invalid native-BF16 local q")
    return result


def identity_record(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "panel": descriptor["panel"],
        "member": descriptor["member"],
        "line_index_zero_based": descriptor["line_index_zero_based"],
        "line_sha256": descriptor["line_sha256"],
    }


def build_initial_pairs(
    descriptors: Sequence[Mapping[str, Any]], raw_snapshot: Mapping[str, Any]
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row_index, descriptor in enumerate(descriptors):
        pair = {
            "row_index": row_index,
            "identity": identity_record(descriptor),
            "role": str(descriptor["role"]),
            "positive_option": int(descriptor["positive_option"]),
            "negative_option": int(descriptor["negative_option"]),
            "threshold": 0.0,
            "threshold_source": "",
            "threshold_sources": [],
            "threshold_history": [],
            "origins": ["initial_specified_pair"],
            "constructions": ["frozen_initial_specified_pair"],
            "created_iteration": 0,
        }
        if descriptor["role"] == "target":
            pair["threshold"] = float(
                descriptor["expected_positive_bf16_margin"]
            )
            pair["threshold_source"] = "target_local_positive_bf16_q"
        else:
            pair["threshold"] = pair_margin(raw_snapshot, pair)
            pair["threshold_source"] = "same_process_raw_pair_margin"
            if (
                not math.isfinite(float(pair["threshold"]))
                or float(pair["threshold"]) < 0.0
            ):
                raise RuntimeError("initial guard raw pair margin is invalid")
        pair["threshold_sources"] = [str(pair["threshold_source"])]
        pair["threshold_history"] = [
            {
                "iteration": 0,
                "source": str(pair["threshold_source"]),
                "observed": float(pair["threshold"]),
                "retained_max": float(pair["threshold"]),
            }
        ]
        pairs.append(pair)
    if (
        len(pairs) != EXPECTED_SELECTED_ROW_COUNT
        or len({pair_key(pair) for pair in pairs}) != EXPECTED_SELECTED_ROW_COUNT
    ):
        raise RuntimeError("initial 19-pair ledger drift")
    return pairs


def gate_snapshot(
    descriptors: Sequence[Mapping[str, Any]],
    active_pairs: Sequence[Mapping[str, Any]],
    raw_snapshot: Mapping[str, Any],
    current: Mapping[str, Any],
    nonactor_exact: bool,
    torch: Any,
) -> dict[str, Any]:
    false_obligations: list[dict[str, Any]] = []
    target_cells: list[str] = []
    target_pair_gates = []
    guard_cells: list[str] = []
    official_row_gates: list[dict[str, Any]] = []
    for row_index, descriptor in enumerate(descriptors):
        flags = current["official_rows"][row_index]["flags"]
        if descriptor["role"] == "target":
            required_metrics = list(MAIN_METRICS)
            for metric in MAIN_METRICS:
                cell = f"{descriptor['line_sha256']}:{metric}"
                target_cells.append(cell)
                if flags[metric] is not True:
                    false_obligations.append(
                        {"row_index": row_index, "role": "target", "metric": metric}
                    )
            original_pair = {
                "row_index": row_index,
                "positive_option": descriptor["positive_option"],
                "negative_option": descriptor["negative_option"],
            }
            margin = pair_margin(current, original_pair)
            threshold = float(descriptor["expected_positive_bf16_margin"])
            target_pair_gates.append(
                {
                    "identity": identity_record(descriptor),
                    "margin": margin,
                    "threshold": threshold,
                    "pass": margin >= threshold,
                }
            )
        else:
            required_metrics = [str(value) for value in descriptor["metrics_union"]]
            for metric in descriptor["metrics_union"]:
                cell = f"{descriptor['line_sha256']}:{metric}"
                guard_cells.append(cell)
                raw_flag = raw_snapshot["official_rows"][row_index]["flags"][metric]
                if raw_flag is not True:
                    raise RuntimeError("guard metrics_union contains raw-false cell")
                if flags[metric] is not True:
                    false_obligations.append(
                        {"row_index": row_index, "role": "guard", "metric": metric}
                    )
        row_flags = {metric: flags[metric] for metric in required_metrics}
        official_row_gates.append(
            {
                "row_index": row_index,
                "identity": identity_record(descriptor),
                "role": str(descriptor["role"]),
                "required_metrics": required_metrics,
                "flags": row_flags,
                "pass": all(value is True for value in row_flags.values()),
            }
        )
    if len(target_cells) != 20 or len(set(target_cells)) != 20:
        raise RuntimeError("target 5x4 obligation ledger drift")
    if len(guard_cells) != len(set(guard_cells)):
        raise RuntimeError("guard metric-local obligation duplication")
    active_records = []
    for pair in active_pairs:
        margin = pair_margin(current, pair)
        threshold = float(pair["threshold"])
        active_records.append(
            {
                "key": list(pair_key(pair)),
                "identity": dict(pair["identity"]),
                "role": str(pair["role"]),
                "origins": list(pair["origins"]),
                "positive_option": int(pair["positive_option"]),
                "negative_option": int(pair["negative_option"]),
                "margin": margin,
                "threshold": threshold,
                "residual": margin - threshold,
                "pass": margin + PAIR_THRESHOLD_TOLERANCE >= threshold,
            }
        )
    count_exact = torch.equal(
        current["outputs_cpu"]["count_logits"],
        raw_snapshot["outputs_cpu"]["count_logits"],
    )
    value_exact = torch.equal(
        current["outputs_cpu"]["value_logits"],
        raw_snapshot["outputs_cpu"]["value_logits"],
    )
    if not count_exact or not value_exact or not nonactor_exact:
        raise RuntimeError("cutting-plane immutable/nonactor anomaly")
    targets_official = not any(
        item["role"] == "target" for item in false_obligations
    )
    guards_official = not any(
        item["role"] == "guard" for item in false_obligations
    )
    target_q = all(item["pass"] for item in target_pair_gates)
    active_feasible = all(item["pass"] for item in active_records)
    passed = targets_official and guards_official and target_q and active_feasible
    return {
        "pass": passed,
        "selected_official_row_count": len(official_row_gates),
        "selected_official_row_count_exact_19": (
            len(official_row_gates) == EXPECTED_SELECTED_ROW_COUNT
        ),
        "official_row_gates": official_row_gates,
        "target_5x4_official_all_true": targets_official,
        "target_5x4_obligation_count": len(target_cells),
        "target_5x4_obligation_unique_count": len(set(target_cells)),
        "target_specified_q_all_met": target_q,
        "target_specified_pair_gates": target_pair_gates,
        "guard_metric_local_all_true": guards_official,
        "guard_obligation_count": len(guard_cells),
        "active_pair_thresholds_all_met": active_feasible,
        "active_pair_count": len(active_records),
        "active_pair_gates": active_records,
        "active_pair_residual_min": min(
            item["residual"] for item in active_records
        ),
        "count_logits_native_exact_raw": True,
        "value_logits_native_exact_raw": True,
        "nonactor_state_exact_raw": True,
        "false_obligations": false_obligations,
    }


def first_different_pair(
    expert_order: Sequence[int], candidate_order: Sequence[int]
) -> tuple[int, int] | None:
    if len(expert_order) != len(candidate_order):
        return None
    for expected, observed in zip(expert_order, candidate_order):
        if int(expected) != int(observed):
            return int(expected), int(observed)
    return None


def set_missing_included_pair(
    expert_order: Sequence[int], candidate_order: Sequence[int]
) -> tuple[int, int] | None:
    expert_set = {int(value) for value in expert_order}
    candidate_set = {int(value) for value in candidate_order}
    missing = [int(value) for value in expert_order if int(value) not in candidate_set]
    included = [
        int(value) for value in candidate_order if int(value) not in expert_set
    ]
    if not missing or not included:
        return None
    return missing[0], included[0]


def dynamic_pair_for_obligation(
    obligation: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
    raw_snapshot: Mapping[str, Any],
    current: Mapping[str, Any],
    torch: Any,
    iteration: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    row_index = int(obligation["row_index"])
    descriptor = descriptors[row_index]
    metric = str(obligation["metric"])
    raw_row = raw_snapshot["official_rows"][row_index]
    current_row = current["official_rows"][row_index]
    expert_order = [int(value) for value in current_row["expert_order"]]
    candidate_order = [int(value) for value in current_row["predicted_order"]]
    if not expert_order:
        raise RuntimeError("dynamic policy cut cannot target an empty expert order")
    context = int(current_row["context"])
    option_pair: tuple[int, int] | None
    if metric == "top1_correct":
        positive = (
            expert_order[0]
            if descriptor["role"] == "target"
            else int(raw_row["top1_index_full_policy_vector"])
        )
        negative = int(current_row["top1_index_full_policy_vector"])
        option_pair = (positive, negative)
        construction = (
            "target_expert_or_guard_raw_full_vector_argmax"
            "_vs_current_full_vector_argmax"
        )
    elif metric == "set_exact":
        option_pair = set_missing_included_pair(expert_order, candidate_order)
        construction = "first_missing_expert_vs_first_included_nonexpert"
    elif metric in {"hybrid_order_exact", "context34_hybrid_order_exact"}:
        if metric == "context34_hybrid_order_exact" and context != 34:
            raise RuntimeError("non-context34 row has context34 hybrid obligation")
        if context == 34:
            option_pair = first_different_pair(expert_order, candidate_order)
            construction = "context34_hybrid_equals_ordered_first_difference"
        else:
            option_pair = set_missing_included_pair(expert_order, candidate_order)
            construction = "noncontext34_hybrid_set_missing_vs_included"
    elif metric in {"ordered_exact", "context34_ordered_exact"}:
        if metric == "context34_ordered_exact" and context != 34:
            raise RuntimeError("non-context34 row has context34 ordered obligation")
        option_pair = first_different_pair(expert_order, candidate_order)
        construction = "ordered_first_differing_expert_vs_candidate_stage"
    else:
        raise RuntimeError(f"unsupported false policy obligation {metric}")
    if option_pair is None:
        return None, {
            "row_index": row_index,
            "identity": identity_record(descriptor),
            "role": str(descriptor["role"]),
            "metric": metric,
            "construction": construction,
            "separating": False,
            "reason": "false_obligation_has_no_constructible_option_pair",
        }
    positive, negative = option_pair
    if positive == negative:
        return None, {
            "row_index": row_index,
            "identity": identity_record(descriptor),
            "role": str(descriptor["role"]),
            "metric": metric,
            "construction": construction,
            "separating": False,
            "reason": "false_obligation_pair_collapsed_to_one_option",
        }
    pair = {
        "row_index": row_index,
        "identity": identity_record(descriptor),
        "role": str(descriptor["role"]),
        "positive_option": int(positive),
        "negative_option": int(negative),
        "threshold": 0.0,
        "threshold_source": "",
        "threshold_sources": [],
        "threshold_history": [],
        "origins": [metric],
        "constructions": [construction],
        "created_iteration": iteration,
    }
    if descriptor["role"] == "target":
        pair["threshold"] = local_positive_bf16_q(
            current, row_index, positive, negative, torch
        )
        pair["threshold_source"] = "current_pair_native_bf16_nextafter_q"
    else:
        pair["threshold"] = pair_margin(raw_snapshot, pair)
        pair["threshold_source"] = "same_process_raw_dynamic_pair_margin"
    if not math.isfinite(float(pair["threshold"])):
        raise RuntimeError("dynamic pair threshold is not finite")
    if descriptor["role"] == "guard" and float(pair["threshold"]) < 0.0:
        raise RuntimeError("dynamic guard raw pair margin is negative")
    pair["threshold_sources"] = [str(pair["threshold_source"])]
    pair["threshold_history"] = [
        {
            "iteration": iteration,
            "source": str(pair["threshold_source"]),
            "observed": float(pair["threshold"]),
            "retained_max": float(pair["threshold"]),
        }
    ]
    return pair, {
        "row_index": row_index,
        "identity": identity_record(descriptor),
        "role": str(descriptor["role"]),
        "metric": metric,
        "construction": construction,
        "key": list(pair_key(pair)),
    }


def add_dynamic_pairs(
    active_pairs: list[dict[str, Any]],
    false_obligations: Sequence[Mapping[str, Any]],
    descriptors: Sequence[Mapping[str, Any]],
    raw_snapshot: Mapping[str, Any],
    current: Mapping[str, Any],
    torch: Any,
    iteration: int,
) -> dict[str, Any]:
    existing = {pair_key(pair): pair for pair in active_pairs}
    added: list[dict[str, Any]] = []
    strengthened: list[dict[str, Any]] = []
    separation_records: list[dict[str, Any]] = []
    for obligation in false_obligations:
        candidate, construction_audit = dynamic_pair_for_obligation(
            obligation,
            descriptors,
            raw_snapshot,
            current,
            torch,
            iteration,
        )
        if candidate is None:
            separation_records.append(construction_audit)
            continue
        key = pair_key(candidate)
        if key in existing:
            record = existing[key]
            if str(record["role"]) != str(candidate["role"]):
                raise RuntimeError("deduplicated dynamic pair role drift")
            old_threshold = float(record["threshold"])
            observed_threshold = float(candidate["threshold"])
            retained_threshold = max(old_threshold, observed_threshold)
            record["threshold"] = retained_threshold
            source = str(candidate["threshold_source"])
            if source not in record["threshold_sources"]:
                record["threshold_sources"].append(source)
            record["threshold_history"].append(
                {
                    "iteration": iteration,
                    "source": source,
                    "observed": observed_threshold,
                    "retained_max": retained_threshold,
                }
            )
            for origin in candidate["origins"]:
                if origin not in record["origins"]:
                    record["origins"].append(origin)
            for construction in candidate["constructions"]:
                if construction not in record["constructions"]:
                    record["constructions"].append(construction)
            if retained_threshold > old_threshold:
                record["threshold_source"] = source
                strengthened.append(
                    {
                        "key": list(key),
                        "old_threshold": old_threshold,
                        "observed_threshold": observed_threshold,
                        "retained_max_threshold": retained_threshold,
                    }
                )
            merged = record
        else:
            active_pairs.append(candidate)
            existing[key] = candidate
            added.append(candidate)
            merged = candidate
        margin = pair_margin(current, merged)
        threshold = float(merged["threshold"])
        separation_records.append(
            {
                **construction_audit,
                "threshold": threshold,
                "current_margin": margin,
                "residual": margin - threshold,
                "separating": margin < threshold,
                "reason": (
                    "violated_pair_threshold"
                    if margin < threshold
                    else "derived_pair_does_not_separate_current_point"
                ),
            }
        )
    if len(separation_records) != len(false_obligations):
        raise RuntimeError("false-obligation separation ledger cardinality drift")
    return {
        "added": added,
        "strengthened": strengthened,
        "separation_records": separation_records,
        "all_false_obligations_have_separating_cuts": all(
            bool(item["separating"]) for item in separation_records
        ),
        "false_obligation_count": len(false_obligations),
        "separation_record_count": len(separation_records),
    }


def solve_minimum_l2_correction(
    gradients: Any, rhs: Any, np: Any, optimize: Any
) -> tuple[Any, dict[str, Any]]:
    if gradients.ndim != 2 or rhs.shape != (gradients.shape[0],):
        raise RuntimeError("cutting-plane QP matrix/vector shape drift")
    u, singular_values, vh = np.linalg.svd(gradients, full_matrices=False)
    rank = int(
        (singular_values > singular_values[0] * SVD_RELATIVE_RANK_TOLERANCE).sum()
    )
    if rank <= 0:
        raise RuntimeError("cutting-plane gradient matrix has zero rank")
    C = u[:, :rank] * singular_values[:rank]

    def objective(y: Any) -> float:
        return 0.5 * float(y @ y)

    def jacobian(y: Any) -> Any:
        return y

    constraints = [
        {
            "type": "ineq",
            "fun": lambda y: C @ y - rhs,
            "jac": lambda y: C,
        }
    ]
    solution = optimize.minimize(
        objective,
        np.zeros(rank, dtype=np.float64),
        jac=jacobian,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": QP_FTOL, "maxiter": QP_MAXITER, "disp": False},
    )
    if not bool(solution.success):
        raise RuntimeError(
            f"cutting-plane SLSQP failed: {solution.status} {solution.message}"
        )
    correction = vh[:rank].T @ solution.x
    residual = gradients @ correction - rhs
    if float(residual.min()) < -QP_DIRECT_TOLERANCE:
        raise RuntimeError("cutting-plane direct linear constraint gate failed")
    raw_norm = float(np.linalg.norm(correction))
    if raw_norm > STEP_L2_CAP:
        applied = correction * (STEP_L2_CAP / raw_norm)
        capped = True
    else:
        applied = correction
        capped = False
    return applied, {
        "objective": "minimum actor6 L2 linearized correction",
        "gradient_shape": list(gradients.shape),
        "svd_rank": rank,
        "singular_values": [float(value) for value in singular_values],
        "solver": {
            "implementation": "scipy.optimize.minimize/SLSQP",
            "success": bool(solution.success),
            "status": int(solution.status),
            "message": str(solution.message),
            "iterations": int(solution.nit),
            "ftol": QP_FTOL,
            "maxiter": QP_MAXITER,
        },
        "direct_uncapped_residual_min": float(residual.min()),
        "uncapped_l2": raw_norm,
        "trust_region_l2_cap": STEP_L2_CAP,
        "capped": capped,
        "applied_l2": float(np.linalg.norm(applied)),
        "predicted_applied_residual_min": float(
            (gradients @ applied - rhs).min()
        ),
    }


def active_pair_gradients(
    geometry: ModuleType,
    helper: ModuleType,
    model: Any,
    batch: Mapping[str, Any],
    current: Mapping[str, Any],
    active_pairs: Sequence[Mapping[str, Any]],
    parameters: Sequence[Any],
    device: Any,
    np: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    torch = helper.torch
    outputs = helper.ppo.model_forward(model, dict(batch), device)
    if any(value.dtype != torch.bfloat16 for value in outputs.values()):
        raise RuntimeError("active-pair forward is not native BF16")
    if any(
        not torch.equal(
            outputs[key].detach().cpu(), current["outputs_cpu"][key]
        )
        for key in ("policy_logits", "count_logits", "value_logits")
    ):
        raise RuntimeError("active-pair graph/no-grad forward mismatch")
    logits = outputs["policy_logits"].float()
    margins = [
        logits[
            int(pair["row_index"]), int(pair["positive_option"])
        ]
        - logits[int(pair["row_index"]), int(pair["negative_option"])]
        for pair in active_pairs
    ]
    gradients = []
    observed = []
    for index, margin in enumerate(margins):
        flat, _ = geometry.gradient_for_margin(
            margin,
            parameters,
            torch,
            retain_graph=index + 1 < len(margins),
        )
        if not bool(np.all(np.isfinite(flat))):
            raise RuntimeError("active-pair gradient is not finite")
        gradients.append(flat)
        observed.append(float(margin.detach().cpu()))
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("autograd.grad materialized actor6 .grad")
    matrix = np.stack(gradients, axis=0)
    observed_array = np.asarray(observed, dtype=np.float64)
    thresholds = np.asarray(
        [float(pair["threshold"]) for pair in active_pairs], dtype=np.float64
    )
    rhs = thresholds - observed_array
    return matrix, rhs, {
        "active_pair_count": len(active_pairs),
        "current_margin_min": float(observed_array.min()),
        "rhs_min": float(rhs.min()),
        "rhs_max": float(rhs.max()),
        "violated_linear_constraints": int((rhs > 0.0).sum()),
    }


def vector_to_tensors(
    vector: Any, parameters: Sequence[Any], torch: Any
) -> list[Any]:
    tensors = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        value = torch.from_numpy(vector[offset : offset + count].copy())
        tensors.append(
            value.to(device=parameter.device, dtype=parameter.dtype).reshape(
                parameter.shape
            )
        )
        offset += count
    if offset != int(vector.size):
        raise RuntimeError("actor6 vector slice accounting drift")
    return tensors


def apply_cumulative_from_raw(
    ram: ModuleType,
    parameters: Sequence[Any],
    raw_actor: Sequence[Any],
    cumulative: Any,
    torch: Any,
) -> None:
    ram.set_actor_overlay(
        parameters,
        raw_actor,
        vector_to_tensors(cumulative, parameters, torch),
        1.0,
        torch,
    )


def restore_raw_actor(
    ram: ModuleType,
    parameters: Sequence[Any],
    raw_actor: Sequence[Any],
    torch: Any,
) -> None:
    # The frozen RAM helper zips all three sequences even when radius is zero.
    ram.set_actor_overlay(parameters, raw_actor, raw_actor, 0.0, torch)


def canonical_active_pair_ledger(
    active_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for pair in sorted(active_pairs, key=pair_key):
        records.append(
            {
                "key": list(pair_key(pair)),
                "identity": dict(pair["identity"]),
                "role": str(pair["role"]),
                "positive_option": int(pair["positive_option"]),
                "negative_option": int(pair["negative_option"]),
                "threshold": float(pair["threshold"]),
                "threshold_source": str(pair["threshold_source"]),
                "threshold_sources": list(pair["threshold_sources"]),
                "threshold_history": list(pair["threshold_history"]),
                "origins": list(pair["origins"]),
                "construction": list(pair["constructions"]),
                "created_iteration": int(pair["created_iteration"]),
            }
        )
    if len(records) != len({tuple(record["key"]) for record in records}):
        raise RuntimeError("canonical active-pair ledger has duplicate keys")
    return records


def active_pair_ledger_audit(
    active_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ledger = canonical_active_pair_ledger(active_pairs)
    return {
        "count": len(ledger),
        "canonical_json_sha256": sha256_bytes(canonical_json(ledger)),
    }


def raw_restore_integrity(
    helper: ModuleType,
    ram: ModuleType,
    model: Any,
    parameters: Sequence[Any],
    nonactor_names: Sequence[str],
    raw_state_sha: str,
    raw_nonactor_sha: str,
    raw_snapshot: Mapping[str, Any],
    batch: Mapping[str, Any],
    cpu_batch: Mapping[str, Any],
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    final_state_sha = helper.model_state_sha256(model.state_dict())
    final_nonactor_sha = helper.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )
    final_raw = ram.snapshot_forward(helper, model, batch, cpu_batch, device)
    forward_exact = (
        final_raw["actions"] == raw_snapshot["actions"]
        and final_raw["official_rows"] == raw_snapshot["official_rows"]
        and all(
            torch.equal(
                final_raw["outputs_cpu"][key], raw_snapshot["outputs_cpu"][key]
            )
            for key in ("policy_logits", "count_logits", "value_logits")
        )
    )
    grad_none = all(parameter.grad is None for parameter in parameters)
    passed = (
        final_state_sha == raw_state_sha
        and final_nonactor_sha == raw_nonactor_sha
        and forward_exact
        and grad_none
    )
    return {
        "pass": passed,
        "raw_model_state_sha256": raw_state_sha,
        "final_model_state_sha256": final_state_sha,
        "raw_nonactor_sha256": raw_nonactor_sha,
        "final_nonactor_sha256": final_nonactor_sha,
        "raw_state_and_nonactor_final_bit_exact": (
            final_state_sha == raw_state_sha and final_nonactor_sha == raw_nonactor_sha
        ),
        "same_process_raw_pre_post_forward_exact": forward_exact,
        "all_actor6_grad_buffers_none": grad_none,
    }


def run_cuttingplane(
    source: bytes,
    static: Mapping[str, Any],
    geometry: ModuleType,
    geometry_evidence: Mapping[str, Any],
    ram: ModuleType,
    ram_evidence: Mapping[str, Any],
    candidate_consumer: Any | None = None,
) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    helper, helper_evidence = geometry.load_helper()
    torch = helper.torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("requires CUDA with native BF16 support")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")

    formal_payload, formal_evidence = geometry.read_regular_bytes(
        geometry.FORMAL_RESULT,
        geometry.FORMAL_RESULT_SHA256,
        "formal v3 ray result",
    )
    formal = geometry.strict_json_bytes(formal_payload, "formal v3 ray result")
    if formal.get("status") != "closed_no_candidate":
        raise RuntimeError("formal v3 result status drift")
    descriptors = [*geometry.extract_targets(formal), *geometry.extract_guards(formal)]
    if len(descriptors) != 19:
        raise RuntimeError("selected descriptor count drift")

    parent_payload, parent_evidence = geometry.read_regular_bytes(
        geometry.PARENT,
        geometry.PARENT_FILE_SHA256,
        "raw U468 checkpoint",
    )
    checkpoint = torch.load(
        io.BytesIO(parent_payload), map_location="cpu", weights_only=False
    )
    model, model_config, kind = helper.instantiate_checkpoint(checkpoint, device)
    if (
        kind != "ppo"
        or helper.model_state_sha256(model.state_dict())
        != geometry.PARENT_MODEL_STATE_SHA256
    ):
        raise RuntimeError("raw U468 model construction/hash drift")
    raw_state_sha = helper.model_state_sha256(model.state_dict())
    parameters = geometry.configure_actor6(model)
    raw_actor = [parameter.detach().clone() for parameter in parameters]
    nonactor_names = sorted(set(model.state_dict()) - set(geometry.ACTOR6_NAMES))
    raw_nonactor_sha = helper.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )
    selected_rows, archive_evidence = geometry.load_selected_rows(
        helper, descriptors, model_config
    )
    cpu_batch = helper.evaluator.collate_ordered(
        [item["features"] for item in selected_rows],
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    model.eval()
    raw_snapshot = ram.snapshot_forward(helper, model, batch, cpu_batch, device)

    raw_outputs = helper.ppo.model_forward(model, batch, device)
    raw_logits = raw_outputs["policy_logits"].float()
    raw_margins = []
    raw_gradient_norms = []
    normalized_gradients = []
    for row_index, descriptor in enumerate(descriptors):
        positive = int(descriptor["positive_option"])
        negative = int(descriptor["negative_option"])
        margin = raw_logits[row_index, positive] - raw_logits[row_index, negative]
        observed = float(margin.detach().cpu())
        if descriptor["role"] == "target" and observed != descriptor["expected_raw_margin"]:
            raise RuntimeError("target raw margin drift")
        raw_margins.append(margin)
    for index, margin in enumerate(raw_margins):
        flat, _ = geometry.gradient_for_margin(
            margin,
            parameters,
            torch,
            retain_graph=index + 1 < len(raw_margins),
        )
        norm = float(np.linalg.norm(flat))
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("raw selected-row gradient norm invalid")
        raw_gradient_norms.append(norm)
        normalized_gradients.append(flat / norm)
    A = np.stack(normalized_gradients, axis=0)
    target_required = np.asarray(
        [float(item["required_positive_margin"]) for item in descriptors[:5]],
        dtype=np.float64,
    )
    direction, direction_audit = geometry.solve_closure_rate_direction(
        A,
        np.asarray(raw_gradient_norms[:5], dtype=np.float64),
        target_required,
        np,
        optimize,
    )
    direction_sha = geometry.vector_sha256_float64_le(direction, np)
    if direction_sha != EXPECTED_DIRECTION_SHA256:
        raise RuntimeError("frozen e471 direction SHA drift")
    del raw_outputs, raw_logits, raw_margins

    active_pairs = build_initial_pairs(descriptors, raw_snapshot)
    cumulative = INITIAL_RADIUS * direction
    ledger: list[dict[str, Any]] = []
    success_iteration: int | None = None
    close_reason = "maximum_correction_count_reached_without_pass"
    overlay_started = False
    candidate_consumer_called = False
    terminal_cumulative = cumulative.copy()
    final_integrity: dict[str, Any] = {}

    try:
        # Set this before the call so even a partially completed helper overlay is
        # covered by the unconditional raw restore below.
        overlay_started = True
        apply_cumulative_from_raw(ram, parameters, raw_actor, cumulative, torch)
        current = ram.snapshot_forward(helper, model, batch, cpu_batch, device)
        nonactor_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in nonactor_names}
        )
        current_gate = gate_snapshot(
            descriptors,
            active_pairs,
            raw_snapshot,
            current,
            nonactor_sha == raw_nonactor_sha,
            torch,
        )
        ledger.append(
            {
                "iteration": 0,
                "kind": "fixed_initial_raw_plus_r0_e471",
                "new_pairs": [],
                "strengthened_pairs": [],
                "active_pair_ledger": active_pair_ledger_audit(active_pairs),
                "step_l2": INITIAL_RADIUS,
                "cumulative_l2": float(np.linalg.norm(cumulative)),
                "cumulative_float64_le_sha256": (
                    geometry.vector_sha256_float64_le(cumulative, np)
                ),
                "post_gate": current_gate,
            }
        )
        if current_gate["pass"]:
            success_iteration = 0
            close_reason = "all_selected_rows_and_active_thresholds_pass"

        for iteration in range(1, MAX_ITER + 1):
            if success_iteration is not None:
                break
            pair_update = add_dynamic_pairs(
                active_pairs,
                current_gate["false_obligations"],
                descriptors,
                raw_snapshot,
                current,
                torch,
                iteration,
            )
            nonactor_sha = helper.model_state_sha256(
                {name: model.state_dict()[name] for name in nonactor_names}
            )
            pre_gate = gate_snapshot(
                descriptors,
                active_pairs,
                raw_snapshot,
                current,
                nonactor_sha == raw_nonactor_sha,
                torch,
            )
            active_violated = sum(
                not bool(record["pass"])
                for record in pre_gate["active_pair_gates"]
            )
            has_separating_cut = (
                bool(pair_update["all_false_obligations_have_separating_cuts"])
                and active_violated > 0
            )
            if not has_separating_cut:
                close_reason = "fail_closed_no_violated_separating_cut"
                ledger.append(
                    {
                        "iteration": iteration,
                        "kind": "fail_closed_without_correction",
                        "new_pairs": pair_update["added"],
                        "strengthened_pairs": pair_update["strengthened"],
                        "separation_audit": pair_update,
                        "active_pair_ledger": active_pair_ledger_audit(active_pairs),
                        "pre_gate": pre_gate,
                        "step_l2": 0.0,
                        "cumulative_l2": float(np.linalg.norm(cumulative)),
                        "cumulative_float64_le_sha256": (
                            geometry.vector_sha256_float64_le(cumulative, np)
                        ),
                    }
                )
                break

            gradients, rhs, gradient_audit = active_pair_gradients(
                geometry,
                helper,
                model,
                batch,
                current,
                active_pairs,
                parameters,
                device,
                np,
            )
            correction, qp_audit = solve_minimum_l2_correction(
                gradients, rhs, np, optimize
            )
            correction_l2 = float(np.linalg.norm(correction))
            if (
                not math.isfinite(correction_l2)
                or correction_l2 <= 0.0
                or correction_l2 > STEP_L2_CAP + 1e-12
            ):
                raise RuntimeError("correction trust-region L2 hard gate failed")
            cumulative = cumulative + correction

            # Every endpoint is reconstructed from the immutable raw actor.
            restore_raw_actor(ram, parameters, raw_actor, torch)
            if helper.model_state_sha256(model.state_dict()) != raw_state_sha:
                raise RuntimeError("per-iteration raw restoration failed")
            apply_cumulative_from_raw(
                ram, parameters, raw_actor, cumulative, torch
            )
            current = ram.snapshot_forward(
                helper, model, batch, cpu_batch, device
            )
            nonactor_sha = helper.model_state_sha256(
                {name: model.state_dict()[name] for name in nonactor_names}
            )
            current_gate = gate_snapshot(
                descriptors,
                active_pairs,
                raw_snapshot,
                current,
                nonactor_sha == raw_nonactor_sha,
                torch,
            )
            ledger.append(
                {
                    "iteration": iteration,
                    "kind": "cumulative_cuttingplane_correction",
                    "new_pairs": pair_update["added"],
                    "strengthened_pairs": pair_update["strengthened"],
                    "separation_audit": pair_update,
                    "active_pair_ledger": active_pair_ledger_audit(active_pairs),
                    "pre_gate": pre_gate,
                    "gradient_audit": gradient_audit,
                    "qp": qp_audit,
                    "step_l2": correction_l2,
                    "cumulative_l2": float(np.linalg.norm(cumulative)),
                    "cumulative_float64_le_sha256": (
                        geometry.vector_sha256_float64_le(cumulative, np)
                    ),
                    "post_gate": current_gate,
                }
            )
            if current_gate["pass"]:
                success_iteration = iteration
                close_reason = "all_selected_rows_and_active_thresholds_pass"

        terminal_cumulative = cumulative.copy()
        if success_iteration is not None and candidate_consumer is not None:
            candidate_consumer_called = True
            candidate_consumer(
                {
                    "helper": helper,
                    "model": model,
                    "checkpoint": checkpoint,
                    "model_config": model_config,
                    "raw_actor": raw_actor,
                    "raw_model_state_sha256": raw_state_sha,
                    "raw_nonactor_sha256": raw_nonactor_sha,
                    "terminal_cumulative_float64": terminal_cumulative.copy(),
                    "terminal_cumulative_float64_le_sha256": (
                        geometry.vector_sha256_float64_le(
                            terminal_cumulative, np
                        )
                    ),
                    "success_iteration": success_iteration,
                    "selected_row_gate": current_gate,
                    "active_pair_ledger": canonical_active_pair_ledger(
                        active_pairs
                    ),
                }
            )
    finally:
        if overlay_started:
            restore_raw_actor(ram, parameters, raw_actor, torch)
            final_integrity = raw_restore_integrity(
                helper,
                ram,
                model,
                parameters,
                nonactor_names,
                raw_state_sha,
                raw_nonactor_sha,
                raw_snapshot,
                batch,
                cpu_batch,
                device,
                torch,
            )
            if not bool(final_integrity["pass"]):
                raise RuntimeError("finally raw state/nonactor/forward/grad restore failed")

    terminal_cumulative_sha = geometry.vector_sha256_float64_le(
        terminal_cumulative, np
    )
    status = (
        "selected_row_adaptive_optimization_success"
        if success_iteration is not None
        else "closed_no_candidate"
    )
    final_active_ledger = canonical_active_pair_ledger(active_pairs)
    final_active_ledger_sha = sha256_bytes(canonical_json(final_active_ledger))
    return {
        "schema_version": SCHEMA,
        "status": status,
        "scope": {
            "selected_train_rows_only": True,
            "full_panel_or_holdout": False,
            "max_correction_iterations": MAX_ITER,
            "fixed_step_l2_cap": STEP_L2_CAP,
            "hyperparameter_sweep": False,
            "result_class": "selected-row adaptive optimization",
            "training_optimizer_backward_save_write_network": False,
            "stdout_only": True,
            "standalone_candidate_consumer_is_none": candidate_consumer is None,
        },
        "input_lock": {
            "self": {
                "path": str(SCRIPT.relative_to(ROOT)),
                "sha256": sha256_bytes(source),
            },
            "geometry": dict(geometry_evidence),
            "ram_runner": dict(ram_evidence),
            "parent": parent_evidence,
            "formal_result": formal_evidence,
            "helper": helper_evidence,
            "train_archives": archive_evidence,
        },
        "initial_direction": {
            "construction": "raw_plus_r0_times_frozen_e471",
            "radius": INITIAL_RADIUS,
            "direction_sha256": direction_sha,
            "direction_audit": direction_audit,
        },
        "active_pair_contract": {
            "initial_count": EXPECTED_SELECTED_ROW_COUNT,
            "final_count": len(active_pairs),
            "initial_target_threshold": "frozen target local positive BF16 q",
            "initial_guard_threshold": "same-process raw observed-pair margin",
            "dynamic_thresholds": {
                "target": "current exact option pair native-BF16 nextafter local q",
                "guard": "same-process raw exact dynamic-pair margin",
            },
            "duplicate_key_rule": "retain max(old threshold, newly observed threshold)",
            "dedup_key": ["row_index", "positive_option", "negative_option"],
            "final_canonical_ledger_sha256": final_active_ledger_sha,
            "final_canonical_ledger": final_active_ledger,
        },
        "iterations": ledger,
        "decision": {
            "status": status,
            "close_reason": close_reason,
            "success_iteration": success_iteration,
            "first_passing_iteration_selected": success_iteration is not None,
            "closed_after_max_iter_if_absent": (
                success_iteration is None
                and close_reason == "maximum_correction_count_reached_without_pass"
            ),
            "terminal_cumulative_l2": float(np.linalg.norm(terminal_cumulative)),
            "terminal_cumulative_float64_le_sha256": terminal_cumulative_sha,
            "model_materialized": False,
            "candidate_consumer_called_before_finally_restore": (
                candidate_consumer_called
            ),
        },
        "final_integrity": {
            **final_integrity,
            "finally_restore_executed_after_overlay": overlay_started,
            "static_zero_write_audit": dict(static),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="run")
    args = parser.parse_args()
    validate_runtime()
    source, self_evidence = read_regular_bytes(
        SCRIPT, sha256_bytes(SCRIPT.read_bytes()), "cutting-plane runner"
    )
    static = static_audit(source)
    geometry, geometry_evidence = import_frozen(
        GEOMETRY, GEOMETRY_SHA256, "u468_metricguard_geometry_v2"
    )
    ram, ram_evidence = import_frozen(
        RAM_RUNNER, RAM_RUNNER_SHA256, "u468_metricguard_ram_ray_v1"
    )
    if (
        geometry.EXPECTED_DIRECTION_SHA256 != EXPECTED_DIRECTION_SHA256
        or ram.EXPECTED_DIRECTION_SHA256 != EXPECTED_DIRECTION_SHA256
        or tuple(ram.MAIN_METRICS) != MAIN_METRICS
    ):
        raise RuntimeError("frozen geometry/RAM exported contract drift")
    if args.mode == "static":
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "geometry": geometry_evidence,
            "ram_runner": ram_evidence,
            "constants": {
                "initial_radius": INITIAL_RADIUS,
                "max_iter": MAX_ITER,
                "step_l2_cap": STEP_L2_CAP,
                "direction_sha256": EXPECTED_DIRECTION_SHA256,
            },
            "audit": static,
        }
    else:
        result = run_cuttingplane(
            source,
            static,
            geometry,
            geometry_evidence,
            ram,
            ram_evidence,
        )
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()

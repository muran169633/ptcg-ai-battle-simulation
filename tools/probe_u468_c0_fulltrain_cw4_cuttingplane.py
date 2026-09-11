#!/usr/bin/env python3
"""RAM-only continuation of frozen U468 C0 with four full-train CW guards.

The frozen v2 cutting-plane runner first reconstructs its iteration-11 C0
endpoint.  A consumer then appends four hash-bound train rows, preserves the
original active-20 absolute thresholds, and performs at most twelve additional
minimum-L2 cutting-plane corrections (L2 cap 0.001 each).  Every endpoint is
rebuilt from raw and checked with native BF16 policy/count/value outputs.

This is a feasibility probe: stdout only, no checkpoint/result write,
materialization, optimizer, network access, upload, or submission.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
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
SCRIPT = TOOLS / "probe_u468_c0_fulltrain_cw4_cuttingplane.py"
SCHEMA = "ptcg-u468-c0-fulltrain-cw4-cuttingplane-feasibility-v1"
SEED = 202608205

CUTTING = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v2.py"
CUTTING_SHA256 = "c2866ba8b00eba6b424197a520419a5717401335cc47202b4fcc711611f67503"
FROZEN_MODE = 0o555
EXPECTED_C0_ITERATION = 11
EXPECTED_C0_CUMULATIVE_SHA256 = (
    "47a001d201db1bba7255ce0b8ccb993cce1649f78f14826782cc0b37e0b31ffc"
)
EXPECTED_C0_MODEL_SHA256 = (
    "c79ce4a258f673248c7abab6e7ccffa10bd9dd4250f43d0e841fd54e44586c82"
)
EXPECTED_C0_L2 = 0.006294165313358753
EXPECTED_ORIGINAL_ACTIVE_COUNT = 20
EXPECTED_COMBINED_ROW_COUNT = 23
MAX_ADDITIONAL_CORRECTIONS = 12
STEP_L2_CAP = 0.001

MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)

NEW_GUARDS = (
    {
        "role": "guard",
        "panel": "flg",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 4545,
        "line_sha256": "f5c29fe63df23a40aff98d58447af577ba7244af2c762a288555041dd7fdfac4",
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": list(MAIN_METRICS),
        "positive_option": 9,
        "negative_option": 10,
        "formal_raw_order": [9],
        "formal_threat_order": [10],
    },
    {
        "role": "guard",
        "panel": "flg",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 5063,
        "line_sha256": "9417c4667da3af128f684cf5b847952b47686a329af28a94cbf01615f3de8644",
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": ["ordered_exact"],
        "positive_option": 1,
        "negative_option": 3,
        "formal_raw_order": [1, 3],
        "formal_threat_order": [3, 1],
    },
    {
        "role": "guard",
        "panel": "pokemonfan",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 5092,
        "line_sha256": "989b079b931a1f256b4ddad8e7f1d42986f3f98464f993a3afe994b5cca56d13",
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": list(MAIN_METRICS),
        "positive_option": 0,
        "negative_option": 1,
        "formal_raw_order": [0],
        "formal_threat_order": [1],
    },
    {
        "role": "guard",
        "panel": "core5",
        "member": "train/part-00007.jsonl",
        "line_index_zero_based": 209,
        "line_sha256": "010dca17bea03dab885587766989072a358f8df36e6485cc358321dd83f6d483",
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": list(MAIN_METRICS),
        "positive_option": 5,
        "negative_option": 4,
        "formal_raw_order": [5],
        "formal_threat_order": [4],
    },
)


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


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")


def read_regular_bytes(
    path: Path,
    expected_sha256: str | None,
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
    if expected_sha256 is not None and digest != expected_sha256:
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


def import_frozen(path: Path, digest: str, name: str) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path, digest, name, expected_mode=FROZEN_MODE
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct import for {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_calls = {
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
        "add_",
        "copy_",
    }
    forbidden_names = {"open", "exec", "eval", "compile", "DataLoader"}
    forbidden_imports = {"requests", "urllib", "subprocess"}
    hits: list[list[Any]] = []
    prints = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            for value in roots:
                if value.split(".", 1)[0] in forbidden_imports:
                    hits.append([node.lineno, value])
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
            hits.append([node.lineno, node.func.attr])
        if isinstance(node.func, ast.Name):
            if node.func.id in forbidden_names:
                hits.append([node.lineno, node.func.id])
            if node.func.id == "print":
                prints += 1
    if hits or prints != 1:
        raise RuntimeError(f"zero-write static audit failed: {hits=} {prints=}")
    return {
        "ast_parse": True,
        "forbidden_write_optimizer_network_calls": hits,
        "stdout_print_call_count": prints,
        "zero_write": True,
    }


def load_original_descriptors(geometry: ModuleType) -> list[dict[str, Any]]:
    payload, _ = geometry.read_regular_bytes(
        geometry.FORMAL_RESULT,
        geometry.FORMAL_RESULT_SHA256,
        "formal v3 ray result",
    )
    formal = geometry.strict_json_bytes(payload, "formal v3 ray result")
    descriptors = [*geometry.extract_targets(formal), *geometry.extract_guards(formal)]
    if len(descriptors) != 19:
        raise RuntimeError("original descriptor count drift")
    return descriptors


def canonical_to_internal_pairs(
    ledger: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for record in ledger:
        key = [int(value) for value in record["key"]]
        if len(key) != 3:
            raise RuntimeError("canonical active pair key shape drift")
        pair = {
            "row_index": key[0],
            "identity": copy.deepcopy(record["identity"]),
            "role": str(record["role"]),
            "positive_option": key[1],
            "negative_option": key[2],
            "threshold": float(record["threshold"]),
            "threshold_source": str(record["threshold_source"]),
            "threshold_sources": list(record["threshold_sources"]),
            "threshold_history": copy.deepcopy(record["threshold_history"]),
            "origins": list(record["origins"]),
            "constructions": list(record["construction"]),
            "created_iteration": int(record["created_iteration"]),
        }
        if tuple(key) != tuple(int(value) for value in record["key"]):
            raise RuntimeError("active pair remap key drift")
        pairs.append(pair)
    if len(pairs) != EXPECTED_ORIGINAL_ACTIVE_COUNT:
        raise RuntimeError("original active pair count drift")
    if len({tuple(cutting_key(pair)) for pair in pairs}) != len(pairs):
        raise RuntimeError("original active pair remap has duplicate keys")
    return pairs


def cutting_key(pair: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(pair["row_index"]),
        int(pair["positive_option"]),
        int(pair["negative_option"]),
    )


def append_new_guard_pairs(
    cutting: ModuleType,
    active_pairs: list[dict[str, Any]],
    descriptors: Sequence[Mapping[str, Any]],
    raw_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    appended: list[dict[str, Any]] = []
    for row_index in range(19, 23):
        descriptor = descriptors[row_index]
        pair: dict[str, Any] = {
            "row_index": row_index,
            "identity": cutting.identity_record(descriptor),
            "role": "guard",
            "positive_option": int(descriptor["positive_option"]),
            "negative_option": int(descriptor["negative_option"]),
            "threshold": 0.0,
            "threshold_source": "same_process_raw_fulltrain_cw_pair_margin",
            "threshold_sources": ["same_process_raw_fulltrain_cw_pair_margin"],
            "threshold_history": [],
            "origins": ["fulltrain_CW_guard"],
            "constructions": ["specified_raw_safe_vs_C0_threat_pair"],
            "created_iteration": EXPECTED_C0_ITERATION,
        }
        threshold = cutting.pair_margin(raw_snapshot, pair)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise RuntimeError("new full-train guard raw margin invalid")
        pair["threshold"] = threshold
        pair["threshold_history"] = [
            {
                "iteration": EXPECTED_C0_ITERATION,
                "source": pair["threshold_source"],
                "observed": threshold,
                "retained_max": threshold,
            }
        ]
        active_pairs.append(pair)
        appended.append(copy.deepcopy(pair))
    if len(active_pairs) != 24 or len({cutting_key(pair) for pair in active_pairs}) != 24:
        raise RuntimeError("combined initial active-pair ledger drift")
    return appended


def short_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pass": bool(gate["pass"]),
        "false_obligation_count": len(gate["false_obligations"]),
        "false_obligations": copy.deepcopy(gate["false_obligations"]),
        "targets_official": bool(gate["target_5x4_official_all_true"]),
        "target_q": bool(gate["target_specified_q_all_met"]),
        "guards_official": bool(gate["guard_metric_local_all_true"]),
        "active_thresholds": bool(gate["active_pair_thresholds_all_met"]),
        "active_pair_count": int(gate["active_pair_count"]),
        "active_pair_residual_min": float(gate["active_pair_residual_min"]),
        "count_logits_exact_raw": bool(gate["count_logits_native_exact_raw"]),
        "value_logits_exact_raw": bool(gate["value_logits_native_exact_raw"]),
        "nonactor_exact_raw": bool(gate["nonactor_state_exact_raw"]),
    }


def verify_new_guard_endpoint(
    snapshot: Mapping[str, Any], descriptors: Sequence[Mapping[str, Any]], *, raw: bool
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index in range(19, 23):
        descriptor = descriptors[row_index]
        row = snapshot["official_rows"][row_index]
        expected_order = (
            descriptor["formal_raw_order"] if raw else descriptor["formal_threat_order"]
        )
        observed_order = [int(value) for value in row["predicted_order"]]
        if observed_order != expected_order:
            raise RuntimeError(
                f"new guard {'raw' if raw else 'C0'} order drift at row {row_index}: "
                f"{observed_order} != {expected_order}"
            )
        flags = {metric: bool(row["flags"][metric]) for metric in descriptor["metrics_union"]}
        if raw and not all(flags.values()):
            raise RuntimeError("new guard is not raw-correct")
        if not raw:
            expected_false = set(descriptor["expected_c0_false_metrics"])
            observed_false = {metric for metric, value in flags.items() if not value}
            if observed_false != expected_false:
                raise RuntimeError(
                    "declared C0 CW metric set drift: "
                    f"{observed_false} != {expected_false}"
                )
        records.append(
            {
                "row_index": row_index,
                "identity": cutting_identity(descriptor),
                "predicted_order": observed_order,
                "metric_flags": flags,
            }
        )
    return records


def cutting_identity(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "panel": descriptor["panel"],
        "member": descriptor["member"],
        "line_index_zero_based": int(descriptor["line_index_zero_based"]),
        "line_sha256": descriptor["line_sha256"],
    }


def continue_from_c0(
    context: Mapping[str, Any],
    cutting: ModuleType,
    geometry: ModuleType,
    ram: ModuleType,
) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    helper = context["helper"]
    torch = helper.torch
    model = context["model"]
    model_config = context["model_config"]
    terminal = np.asarray(context["terminal_cumulative_float64"], dtype=np.float64).copy()
    c0_checks = {
        "success_iteration_11": int(context["success_iteration"]) == EXPECTED_C0_ITERATION,
        "cumulative_sha_exact": context["terminal_cumulative_float64_le_sha256"]
        == EXPECTED_C0_CUMULATIVE_SHA256,
        "cumulative_l2_exact": math.isclose(
            float(np.linalg.norm(terminal)), EXPECTED_C0_L2, rel_tol=0.0, abs_tol=1e-15
        ),
        "model_sha_exact": helper.model_state_sha256(model.state_dict())
        == EXPECTED_C0_MODEL_SHA256,
        "selected_gate_pass": bool(context["selected_row_gate"]["pass"]),
        "original_active20": len(context["active_pair_ledger"])
        == EXPECTED_ORIGINAL_ACTIVE_COUNT,
    }
    if not all(c0_checks.values()):
        raise RuntimeError(f"frozen C0 contract drift: {c0_checks}")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = next(model.parameters()).device
    parameters = geometry.configure_actor6(model)
    raw_actor = context["raw_actor"]
    raw_state_sha = str(context["raw_model_state_sha256"])
    raw_nonactor_sha = str(context["raw_nonactor_sha256"])
    nonactor_names = sorted(set(model.state_dict()) - set(geometry.ACTOR6_NAMES))

    original_descriptors = load_original_descriptors(geometry)
    descriptors = [*original_descriptors, *[copy.deepcopy(item) for item in NEW_GUARDS]]
    for ordinal, descriptor in enumerate(descriptors[19:], start=14):
        descriptor["ordinal"] = ordinal
        descriptor["first_different_ranking_stage"] = 0
        descriptor["margin_semantics"] = "raw_safe_option_minus_C0_fulltrain_threat"

    old_geometry_rows = geometry.EXPECTED_ROW_COUNT
    old_cutting_rows = cutting.EXPECTED_SELECTED_ROW_COUNT
    geometry.EXPECTED_ROW_COUNT = EXPECTED_COMBINED_ROW_COUNT
    cutting.EXPECTED_SELECTED_ROW_COUNT = EXPECTED_COMBINED_ROW_COUNT
    try:
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

        cutting.restore_raw_actor(ram, parameters, raw_actor, torch)
        if helper.model_state_sha256(model.state_dict()) != raw_state_sha:
            raise RuntimeError("raw restoration before 23-row snapshot failed")
        raw_snapshot = ram.snapshot_forward(helper, model, batch, cpu_batch, device)
        raw_guard_records = verify_new_guard_endpoint(
            raw_snapshot, descriptors, raw=True
        )

        cutting.apply_cumulative_from_raw(ram, parameters, raw_actor, terminal, torch)
        if helper.model_state_sha256(model.state_dict()) != EXPECTED_C0_MODEL_SHA256:
            raise RuntimeError("23-row C0 reconstruction model SHA drift")
        current = ram.snapshot_forward(helper, model, batch, cpu_batch, device)
        c0_guard_records = verify_new_guard_endpoint(current, descriptors, raw=False)

        active_pairs = canonical_to_internal_pairs(context["active_pair_ledger"])
        original_ledger_before = cutting.canonical_active_pair_ledger(active_pairs)
        original_ledger_sha_before = sha256_bytes(canonical_json(original_ledger_before))
        expected_original_sha = sha256_bytes(
            canonical_json(list(context["active_pair_ledger"]))
        )
        if original_ledger_sha_before != expected_original_sha:
            raise RuntimeError("canonical active20 remap changed absolute ledger")
        raw23_guard_threshold_comparison = []
        for pair in active_pairs:
            if pair["role"] != "guard":
                continue
            raw23_margin = cutting.pair_margin(raw_snapshot, pair)
            canonical_threshold = float(pair["threshold"])
            raw23_guard_threshold_comparison.append(
                {
                    "key": list(cutting_key(pair)),
                    "created_iteration": int(pair["created_iteration"]),
                    "canonical_threshold": canonical_threshold,
                    "same_process_raw23_margin": raw23_margin,
                    "raw23_minus_canonical": raw23_margin - canonical_threshold,
                    "would_raise_by_max_rule": raw23_margin > canonical_threshold,
                }
            )
        appended_pairs = append_new_guard_pairs(
            cutting, active_pairs, descriptors, raw_snapshot
        )

        def nonactor_exact() -> bool:
            return helper.model_state_sha256(
                {name: model.state_dict()[name] for name in nonactor_names}
            ) == raw_nonactor_sha

        current_gate = cutting.gate_snapshot(
            descriptors,
            active_pairs,
            raw_snapshot,
            current,
            nonactor_exact(),
            torch,
        )
        if current_gate["pass"] or len(current_gate["false_obligations"]) != 13:
            raise RuntimeError("C0 four-CW guard obligation ledger drift")

        iterations: list[dict[str, Any]] = [
            {
                "continuation_iteration": 0,
                "absolute_iteration": EXPECTED_C0_ITERATION,
                "kind": "frozen_C0_plus_four_fulltrain_guards",
                "step_l2": 0.0,
                "cumulative_l2": float(np.linalg.norm(terminal)),
                "cumulative_float64_le_sha256": geometry.vector_sha256_float64_le(
                    terminal, np
                ),
                "gate": short_gate(current_gate),
            }
        ]
        success_iteration: int | None = None
        close_reason = "maximum_additional_corrections_reached"

        for correction_index in range(1, MAX_ADDITIONAL_CORRECTIONS + 1):
            absolute_iteration = EXPECTED_C0_ITERATION + correction_index
            pair_update = cutting.add_dynamic_pairs(
                active_pairs,
                current_gate["false_obligations"],
                descriptors,
                raw_snapshot,
                current,
                torch,
                absolute_iteration,
            )
            pre_gate = cutting.gate_snapshot(
                descriptors,
                active_pairs,
                raw_snapshot,
                current,
                nonactor_exact(),
                torch,
            )
            active_violated = sum(
                not bool(record["pass"])
                for record in pre_gate["active_pair_gates"]
            )
            if (
                not bool(pair_update["all_false_obligations_have_separating_cuts"])
                or active_violated <= 0
            ):
                close_reason = "fail_closed_no_violated_separating_cut"
                iterations.append(
                    {
                        "continuation_iteration": correction_index,
                        "absolute_iteration": absolute_iteration,
                        "kind": "fail_closed_without_correction",
                        "new_pair_count": len(pair_update["added"]),
                        "strengthened_pair_count": len(pair_update["strengthened"]),
                        "active_violated": active_violated,
                        "step_l2": 0.0,
                        "cumulative_l2": float(np.linalg.norm(terminal)),
                        "cumulative_float64_le_sha256": geometry.vector_sha256_float64_le(
                            terminal, np
                        ),
                        "gate": short_gate(pre_gate),
                    }
                )
                break

            gradients, rhs, gradient_audit = cutting.active_pair_gradients(
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
            correction, qp_audit = cutting.solve_minimum_l2_correction(
                gradients, rhs, np, optimize
            )
            correction_l2 = float(np.linalg.norm(correction))
            if (
                not math.isfinite(correction_l2)
                or correction_l2 <= 0.0
                or correction_l2 > STEP_L2_CAP + 1e-12
            ):
                raise RuntimeError("additional correction trust-region gate failed")
            terminal = terminal + correction

            cutting.restore_raw_actor(ram, parameters, raw_actor, torch)
            if helper.model_state_sha256(model.state_dict()) != raw_state_sha:
                raise RuntimeError("per-correction raw reconstruction failed")
            cutting.apply_cumulative_from_raw(
                ram, parameters, raw_actor, terminal, torch
            )
            current = ram.snapshot_forward(helper, model, batch, cpu_batch, device)
            current_gate = cutting.gate_snapshot(
                descriptors,
                active_pairs,
                raw_snapshot,
                current,
                nonactor_exact(),
                torch,
            )
            iterations.append(
                {
                    "continuation_iteration": correction_index,
                    "absolute_iteration": absolute_iteration,
                    "kind": "minimum_l2_dynamic_cuttingplane_correction",
                    "new_pair_count": len(pair_update["added"]),
                    "new_pair_keys": [
                        list(cutting_key(pair)) for pair in pair_update["added"]
                    ],
                    "strengthened_pair_count": len(pair_update["strengthened"]),
                    "active_pair_count": len(active_pairs),
                    "active_violated_before": active_violated,
                    "gradient_rank": int(qp_audit["svd_rank"]),
                    "uncapped_l2": float(qp_audit["uncapped_l2"]),
                    "step_capped": bool(qp_audit["capped"]),
                    "step_l2": correction_l2,
                    "cumulative_l2": float(np.linalg.norm(terminal)),
                    "cumulative_float64_le_sha256": geometry.vector_sha256_float64_le(
                        terminal, np
                    ),
                    "gradient_rhs_max": float(gradient_audit["rhs_max"]),
                    "gate": short_gate(current_gate),
                }
            )
            if current_gate["pass"]:
                success_iteration = correction_index
                close_reason = "all_23_rows_and_active_thresholds_pass"
                break

        terminal_sha = geometry.vector_sha256_float64_le(terminal, np)
        terminal_model_sha = helper.model_state_sha256(model.state_dict())
        terminal_nonactor_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in nonactor_names}
        )
        final_ledger = cutting.canonical_active_pair_ledger(active_pairs)
        original_after = [
            record for record in final_ledger if int(record["key"][0]) < 19
        ]
        # Dynamic duplicate handling may append history/origins to an original
        # pair, but its absolute threshold must never decrease or be recomputed.
        before_thresholds = {
            tuple(record["key"]): float(record["threshold"])
            for record in original_ledger_before
        }
        after_thresholds = {
            tuple(record["key"]): float(record["threshold"])
            for record in original_after
        }
        if before_thresholds != after_thresholds:
            raise RuntimeError("original active20 absolute thresholds changed")
        return {
            "status": "feasible_23row_terminal" if success_iteration is not None else "closed_no_23row_terminal",
            "c0_contract": c0_checks,
            "archives": archive_evidence,
            "new_guard_raw_records": raw_guard_records,
            "new_guard_c0_records": c0_guard_records,
            "new_guard_initial_pairs": appended_pairs,
            "original_active20": {
                "count": len(original_ledger_before),
                "canonical_ledger_sha256": original_ledger_sha_before,
                "absolute_thresholds_preserved": before_thresholds == after_thresholds,
                "raw23_guard_threshold_comparison": raw23_guard_threshold_comparison,
                "raw23_max_rule_raise_count": sum(
                    bool(item["would_raise_by_max_rule"])
                    for item in raw23_guard_threshold_comparison
                ),
                "raw23_max_rule_max_positive_delta": max(
                    [
                        float(item["raw23_minus_canonical"])
                        for item in raw23_guard_threshold_comparison
                        if bool(item["would_raise_by_max_rule"])
                    ],
                    default=0.0,
                ),
            },
            "initial_active_pair_count": 24,
            "iterations": iterations,
            "decision": {
                "success_continuation_iteration": success_iteration,
                "success_absolute_iteration": (
                    None
                    if success_iteration is None
                    else EXPECTED_C0_ITERATION + success_iteration
                ),
                "close_reason": close_reason,
                "terminal_cumulative_l2": float(np.linalg.norm(terminal)),
                "terminal_cumulative_float64_le_sha256": terminal_sha,
                "terminal_model_state_sha256": terminal_model_sha,
                "terminal_nonactor_sha256": terminal_nonactor_sha,
                "terminal_nonactor_exact_raw": terminal_nonactor_sha == raw_nonactor_sha,
                "final_active_pair_count": len(active_pairs),
                "final_active_pair_ledger_sha256": sha256_bytes(
                    canonical_json(final_ledger)
                ),
                "final_gate": short_gate(current_gate),
                "model_materialized": False,
                "submission_performed": False,
            },
        }
    finally:
        geometry.EXPECTED_ROW_COUNT = old_geometry_rows
        cutting.EXPECTED_SELECTED_ROW_COUNT = old_cutting_rows


def run_probe(
    source: bytes,
    static: Mapping[str, Any],
    cutting: ModuleType,
    cutting_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    geometry, geometry_evidence = cutting.import_frozen(
        cutting.GEOMETRY,
        cutting.GEOMETRY_SHA256,
        "cw4_geometry_frozen",
    )
    ram, ram_evidence = cutting.import_frozen(
        cutting.RAM_RUNNER,
        cutting.RAM_RUNNER_SHA256,
        "cw4_ram_frozen",
    )
    holder: dict[str, Any] = {}

    def consumer(context: Mapping[str, Any]) -> None:
        if holder:
            raise RuntimeError("C0 consumer called more than once")
        holder.update(continue_from_c0(context, cutting, geometry, ram))

    frozen_source, _ = cutting.read_regular_bytes(
        cutting.SCRIPT,
        CUTTING_SHA256,
        "frozen cutting-plane v2",
        expected_mode=FROZEN_MODE,
    )
    frozen_static = cutting.static_audit(frozen_source)
    base = cutting.run_cuttingplane(
        frozen_source,
        frozen_static,
        geometry,
        geometry_evidence,
        ram,
        ram_evidence,
        candidate_consumer=consumer,
    )
    if not holder:
        raise RuntimeError("frozen C0 reconstruction did not invoke consumer")
    if not bool(base["final_integrity"]["pass"]):
        raise RuntimeError("frozen runner finally raw restore failed")
    return {
        "schema_version": SCHEMA,
        "status": holder["status"],
        "input_lock": {
            "self": {
                "path": str(SCRIPT.relative_to(ROOT)),
                "sha256": sha256_bytes(source),
            },
            "cuttingplane_v2": dict(cutting_evidence),
            "geometry": dict(geometry_evidence),
            "ram_runner": dict(ram_evidence),
        },
        "frozen_c0_reconstruction": {
            "status": base["status"],
            "success_iteration": base["decision"]["success_iteration"],
            "terminal_cumulative_l2": base["decision"]["terminal_cumulative_l2"],
            "terminal_cumulative_float64_le_sha256": base["decision"][
                "terminal_cumulative_float64_le_sha256"
            ],
            "active_pair_count": base["active_pair_contract"]["final_count"],
            "finally_raw_restore": base["final_integrity"],
        },
        "continuation": holder,
        "scope": {
            "original_19_plus_four_hash_bound_train_rows": True,
            "validation_or_holdout_access": False,
            "maximum_additional_corrections": MAX_ADDITIONAL_CORRECTIONS,
            "step_l2_cap": STEP_L2_CAP,
            "training_optimizer_backward": False,
            "checkpoint_or_result_writes": 0,
            "stdout_only": True,
            "network_upload_submission": False,
        },
        "static_audit": dict(static),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="run")
    args = parser.parse_args()
    validate_runtime()
    source, self_evidence = read_regular_bytes(SCRIPT, None, "probe script")
    static = static_audit(source)
    cutting, cutting_evidence = import_frozen(
        CUTTING, CUTTING_SHA256, "cw4_cuttingplane_v2_frozen"
    )
    if args.mode == "static":
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "cuttingplane_v2": cutting_evidence,
            "audit": static,
            "run_executed": False,
        }
    else:
        result = run_probe(source, static, cutting, cutting_evidence)
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()

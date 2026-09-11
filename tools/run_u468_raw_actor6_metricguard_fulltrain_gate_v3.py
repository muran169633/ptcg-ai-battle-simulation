#!/usr/bin/env python3
"""Read-only full-train gate for the expanded CW4 cutting-plane candidate.

The hash-bound expanded runner reconstructs its preregistered iteration-5
terminal in RAM.  Its optional consumer copies that live state to CPU before
the runner's mandatory ``finally`` raw restoration.  Only after that restore
passes does this wrapper reuse the frozen formal-v3 evaluator for all 24,050
train rows in FLG, PokemonFan, and core5 (two states by three panels, B256).

There is no validation access, training optimizer, backward call, sweep,
checkpoint/result write, materialization, network access, upload, or
submission.  Output is one JSON document on stdout only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v3.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-fulltrain-gate-v3"

CUTTING = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_cw4_cuttingplane_v1.py"
CUTTING_SHA256 = (
    "40715927549457b161fc907757279e4d85466ded1d1e6bfed62f67c13b529e7c"
)
FORMAL = TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v3.py"
FORMAL_SHA256 = (
    "2d12ce9e76f6672911948dcc6458573899ec45fc39e748510544001025ede939"
)
FROZEN_MODE = 0o555

PANEL_ORDER = ("flg", "pokemonfan", "core5")
MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)
SAFETY_METRICS = MAIN_METRICS + (
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
PF_MINIMUM_WC = {
    "set_exact": 3,
    "hybrid_order_exact": 3,
    "ordered_exact": 5,
    "top1_correct": 3,
}
EXPECTED_ROWS = {"flg": 9443, "pokemonfan": 9487, "core5": 5120}
EXPECTED_TOTAL_ROWS = 24050
EXPECTED_EVALUATIONS = 6
EXPECTED_BATCH_SIZE = 256
EXPECTED_SELECTED_SUCCESS_ITERATION = 5
EXPECTED_TERMINAL_CUMULATIVE_L2 = 0.007014818833558696
EXPECTED_TERMINAL_CUMULATIVE_SHA256 = (
    "f2f3d554cf1cffa8f81c7b637ca4a2e531e97777188446b2dc999592800e4b9e"
)
EXPECTED_TERMINAL_MODEL_STATE_SHA256 = (
    "f1130d1701619609737c5a277502a41367c05a261115f5daaaaccf967822fd6b"
)
EXPECTED_FINAL_ACTIVE_PAIR_COUNT = 24
EXPECTED_FINAL_ACTIVE_PAIR_LEDGER_SHA256 = (
    "ad48597cef3e0d43a6e1e2dd34dc00d585e1bd1b1ef904972f8436aa0455508e"
)
EXPECTED_ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
EXPECTED_CHANGED_ACTOR_NAMES = EXPECTED_ACTOR6_NAMES[:-1]
EXPECTED_UNCHANGED_ACTOR_NAMES = ("actor_residual.2.bias",)


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


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")


def local_static_audit(source: bytes) -> dict[str, Any]:
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
        "checkpoint_or_result_write_call_sites": 0,
        "no_training_optimizer_backward_write_network_or_submission": True,
    }


def interface_audit(cutting: ModuleType, formal: ModuleType) -> dict[str, Any]:
    cutting_parameters = inspect.signature(cutting.run_cuttingplane).parameters
    formal_parameters = tuple(
        inspect.signature(formal.evaluate_all_states).parameters
    )
    if "candidate_consumer" not in cutting_parameters:
        raise RuntimeError("frozen cutting-plane lacks candidate_consumer")
    if cutting_parameters["candidate_consumer"].default is not None:
        raise RuntimeError("standalone candidate_consumer default drift")
    expected_formal_parameters = (
        "helper",
        "design",
        "states",
        "raw_checkpoint",
        "archive_payloads",
        "frozen_summaries",
    )
    if formal_parameters != expected_formal_parameters:
        raise RuntimeError("formal evaluate_all_states signature drift")
    if (
        tuple(formal.PANEL_ORDER) != PANEL_ORDER
        or tuple(formal.MAIN_METRICS) != MAIN_METRICS
        or tuple(formal.SAFETY_METRICS) != SAFETY_METRICS
        or tuple(cutting.MAIN_METRICS) != MAIN_METRICS
    ):
        raise RuntimeError("panel/metric contract drift")
    expanded_constants = {
        "success_iteration": int(cutting.EXPECTED_SUCCESS_CONTINUATION_ITERATION)
        == EXPECTED_SELECTED_SUCCESS_ITERATION,
        "terminal_l2": math.isclose(
            float(cutting.EXPECTED_TERMINAL_CUMULATIVE_L2),
            EXPECTED_TERMINAL_CUMULATIVE_L2,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "terminal_cumulative_sha": str(
            cutting.EXPECTED_TERMINAL_CUMULATIVE_SHA256
        )
        == EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "terminal_model_sha": str(cutting.EXPECTED_TERMINAL_MODEL_STATE_SHA256)
        == EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "terminal_active_ledger_sha": str(
            cutting.EXPECTED_TERMINAL_ACTIVE_LEDGER_SHA256
        )
        == EXPECTED_FINAL_ACTIVE_PAIR_LEDGER_SHA256,
        "terminal_active_count": int(cutting.EXPECTED_INITIAL_ACTIVE_COUNT)
        == EXPECTED_FINAL_ACTIVE_PAIR_COUNT,
    }
    if not all(expanded_constants.values()):
        raise RuntimeError(
            f"expanded runner preregistered constants drift: {expanded_constants}"
        )
    return {
        "cutting_candidate_consumer_present": True,
        "cutting_candidate_consumer_default_none": True,
        "formal_evaluate_all_states_signature": list(formal_parameters),
        "panels_exact": list(PANEL_ORDER),
        "safety_metrics_exact": list(SAFETY_METRICS),
        "expanded_preregistered_constants": expanded_constants,
    }


def audit_candidate_state(
    helper: ModuleType,
    geometry: ModuleType,
    raw_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    expected_raw_nonactor_sha256: str,
) -> dict[str, Any]:
    torch = helper.torch
    allowed_actor_names = tuple(geometry.ACTOR6_NAMES)
    if allowed_actor_names != EXPECTED_ACTOR6_NAMES:
        raise RuntimeError("cutting geometry actor6 name/order drift")
    if set(candidate_state) != set(raw_state):
        raise RuntimeError("candidate model-state key set differs from raw")

    changed_set: set[str] = set()
    raw_nonfinite: list[str] = []
    candidate_nonfinite: list[str] = []
    for name in sorted(raw_state):
        raw_tensor = raw_state[name]
        candidate_tensor = candidate_state[name]
        if not isinstance(raw_tensor, torch.Tensor) or not isinstance(
            candidate_tensor, torch.Tensor
        ):
            raise RuntimeError(f"non-tensor model state entry at {name}")
        if (
            raw_tensor.shape != candidate_tensor.shape
            or raw_tensor.dtype != candidate_tensor.dtype
        ):
            raise RuntimeError(f"candidate tensor schema differs at {name}")
        if (raw_tensor.is_floating_point() or raw_tensor.is_complex()) and not bool(
            torch.isfinite(raw_tensor).all()
        ):
            raw_nonfinite.append(name)
        if (
            candidate_tensor.is_floating_point() or candidate_tensor.is_complex()
        ) and not bool(torch.isfinite(candidate_tensor).all()):
            candidate_nonfinite.append(name)
        if not torch.equal(raw_tensor, candidate_tensor):
            changed_set.add(name)
    if raw_nonfinite or candidate_nonfinite:
        raise RuntimeError(
            f"nonfinite state tensor: {raw_nonfinite=} {candidate_nonfinite=}"
        )

    allowed_set = set(allowed_actor_names)
    changed_actor_names = tuple(
        name for name in allowed_actor_names if name in changed_set
    )
    unchanged_actor_names = tuple(
        name for name in allowed_actor_names if name not in changed_set
    )
    scope_checks = {
        "changed_set_nonempty": bool(changed_set),
        "changed_set_strict_subset_of_cutting_geometry_actor6": (
            bool(changed_set)
            and changed_set < allowed_set
        ),
        "every_changed_tensor_in_cutting_geometry_actor6": changed_set <= allowed_set,
        "changed_actor_set_exact_expected_five": set(changed_actor_names)
        == set(EXPECTED_CHANGED_ACTOR_NAMES),
        "unchanged_actor_set_exact_residual2_bias": unchanged_actor_names
        == EXPECTED_UNCHANGED_ACTOR_NAMES,
    }
    if not all(scope_checks.values()):
        raise RuntimeError(f"candidate actor scope drift: {scope_checks}")

    unchanged_bias = EXPECTED_UNCHANGED_ACTOR_NAMES[0]
    if (
        raw_state[unchanged_bias].numel() != 1
        or not torch.equal(raw_state[unchanged_bias], candidate_state[unchanged_bias])
    ):
        raise RuntimeError("common actor residual output bias structure drift")

    nonactor_names = sorted(set(raw_state) - allowed_set)
    nonactor_mismatches = [
        name
        for name in nonactor_names
        if not torch.equal(raw_state[name], candidate_state[name])
    ]
    raw_nonactor_sha = helper.model_state_sha256(
        {name: raw_state[name] for name in nonactor_names}
    )
    candidate_nonactor_sha = helper.model_state_sha256(
        {name: candidate_state[name] for name in nonactor_names}
    )
    if (
        nonactor_mismatches
        or raw_nonactor_sha != expected_raw_nonactor_sha256
        or candidate_nonactor_sha != expected_raw_nonactor_sha256
    ):
        raise RuntimeError("candidate nonactor bit-exact/SHA gate failed")
    return {
        "allowed_parameter_names": list(allowed_actor_names),
        "changed_parameter_names": list(changed_actor_names),
        "changed_parameter_count": len(changed_actor_names),
        "unchanged_allowed_parameter_names": list(unchanged_actor_names),
        "unchanged_allowed_parameter_count": len(unchanged_actor_names),
        "scope_checks": scope_checks,
        "all_state_tensor_keys_shapes_dtypes_exact_raw": True,
        "all_raw_and_candidate_state_tensors_finite": True,
        "raw_state_tensor_count": len(raw_state),
        "candidate_state_tensor_count": len(candidate_state),
        "nonactor_tensor_count": len(nonactor_names),
        "nonactor_mismatch_names": nonactor_mismatches,
        "all_nonactor_tensors_bit_exact_raw": True,
        "raw_nonactor_sha256": raw_nonactor_sha,
        "candidate_nonactor_sha256": candidate_nonactor_sha,
        "unchanged_actor_residual_2_bias": {
            "name": unchanged_bias,
            "numel": 1,
            "tensor_exact_raw": True,
            "structural_explanation": (
                "actor_residual.2.bias is one common scalar added to every "
                "option logit; it cancels exactly in every option-pair margin, "
                "so its cutting-plane margin gradient and update are zero"
            ),
        },
    }


def capture_candidate(
    context: Mapping[str, Any],
    holder: dict[str, Any],
    geometry: ModuleType,
) -> None:
    if holder:
        raise RuntimeError("candidate consumer called more than once")
    required = {
        "helper",
        "model",
        "checkpoint",
        "raw_model_state_sha256",
        "raw_nonactor_sha256",
        "terminal_cumulative_float64",
        "terminal_cumulative_float64_le_sha256",
        "success_iteration",
        "selected_row_gate",
        "active_pair_ledger",
    }
    if not required.issubset(context):
        raise RuntimeError("candidate consumer context schema drift")
    helper = context["helper"]
    model = context["model"]
    checkpoint = context["checkpoint"]
    raw_state_source = checkpoint.get("model_state_dict")
    if not isinstance(raw_state_source, Mapping):
        raise RuntimeError("consumer checkpoint has no model_state_dict")
    raw_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in raw_state_source.items()
    }
    candidate_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    raw_sha = helper.model_state_sha256(raw_state)
    candidate_sha = helper.model_state_sha256(candidate_state)
    live_candidate_sha = helper.model_state_sha256(model.state_dict())
    if raw_sha != context["raw_model_state_sha256"]:
        raise RuntimeError("consumer raw model-state SHA drift")
    if candidate_sha != live_candidate_sha:
        raise RuntimeError("consumer CPU candidate differs from live candidate")
    if candidate_sha != EXPECTED_TERMINAL_MODEL_STATE_SHA256:
        raise RuntimeError(
            f"consumer candidate model-state SHA drift: {candidate_sha}"
        )
    actor_audit = audit_candidate_state(
        helper,
        geometry,
        raw_state,
        candidate_state,
        str(context["raw_nonactor_sha256"]),
    )
    cumulative = context["terminal_cumulative_float64"]
    cumulative_l2 = float((cumulative @ cumulative) ** 0.5)
    if not math.isfinite(cumulative_l2) or cumulative_l2 <= 0.0:
        raise RuntimeError("consumer terminal cumulative norm invalid")
    selected_gate = context["selected_row_gate"]
    if not isinstance(selected_gate, Mapping) or not bool(selected_gate.get("pass")):
        raise RuntimeError("consumer invoked without selected-row pass")
    holder.update(
        {
            "raw_state": raw_state,
            "candidate_state": candidate_state,
            "checkpoint": checkpoint,
            "audit": {
                "consumer_call_count": 1,
                "success_iteration": int(context["success_iteration"]),
                "raw_model_state_sha256": raw_sha,
                "candidate_model_state_sha256": candidate_sha,
                "raw_nonactor_sha256": context["raw_nonactor_sha256"],
                "candidate_nonactor_sha256": actor_audit[
                    "candidate_nonactor_sha256"
                ],
                "actor_scope": actor_audit,
                "terminal_cumulative_l2": cumulative_l2,
                "terminal_cumulative_float64_le_sha256": context[
                    "terminal_cumulative_float64_le_sha256"
                ],
                "selected_row_gate_pass": True,
                "active_pair_count": len(context["active_pair_ledger"]),
                "candidate_model_state_sha256_exact_preregistered": True,
                "RAM_only_CPU_clone_before_expanded_finally_restore": True,
            },
        }
    )


def load_fulltrain_inputs(
    formal: ModuleType,
) -> tuple[ModuleType, dict[str, bytes], dict[str, Any], dict[str, Any]]:
    design, _, design_evidence = formal.verify_frozen_design()
    if (
        tuple(design.DATASETS) != PANEL_ORDER
        or {name: int(value["rows"]) for name, value in design.EXPECTED_TRAIN.items()}
        != EXPECTED_ROWS
        or sum(EXPECTED_ROWS.values()) != EXPECTED_TOTAL_ROWS
    ):
        raise RuntimeError("frozen full-train dataset cardinality drift")
    archive_payloads: dict[str, bytes] = {}
    archive_evidence: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        expected_mode = int(
            str(design_evidence["fixed_inputs"]["datasets"][panel]["mode"]), 8
        )
        payload, evidence = formal.read_regular_bytes(
            design.DATASETS[panel],
            design.DATA_SHA256[panel],
            f"{panel} full-train archive",
            expected_mode=expected_mode,
        )
        archive_payloads[panel] = payload
        archive_evidence[panel] = evidence
    frozen_summaries, frozen_summary_evidence = formal.load_frozen_metric_summaries(
        design
    )
    return design, archive_payloads, {
        "design": {
            "tool": design_evidence["tool"],
            "artifact": design_evidence["artifact"],
            "cache_canonical_sha256": sha256_bytes(
                canonical_json(design_evidence["cache"])
            ),
        },
        "train_archives": archive_evidence,
        "frozen_official_summaries": frozen_summary_evidence,
    }, frozen_summaries


def evaluate_raw_candidate(
    formal: ModuleType,
    design: ModuleType,
    states: Mapping[str, Mapping[str, Any]],
    raw_checkpoint: Mapping[str, Any],
    archive_payloads: Mapping[str, bytes],
    frozen_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    helper = formal.load_helper()
    if (
        helper.model_state_sha256(states["raw"])
        != design.PARENT_MODEL_STATE_SHA256
    ):
        raise RuntimeError("full-train raw state/design binding drift")
    original_order = formal.MODEL_ORDER
    original_alphas = formal.ALPHAS
    original_copy = formal.copy_actor_state_to_model
    live: dict[str, Any] = {}
    restore: dict[str, Any] = {
        "evaluation_model_instantiated": False,
        "finally_raw_restore_attempted": False,
        "finally_raw_restore_pass": False,
    }

    def tracked_copy(model: Any, state: Mapping[str, Any], module: ModuleType) -> None:
        live["model"] = model
        original_copy(model, state, module)

    formal.MODEL_ORDER = ("raw", "candidate")
    formal.ALPHAS = {"raw": 0, "candidate": 1}
    formal.copy_actor_state_to_model = tracked_copy
    try:
        panel_results, advisory, execution = formal.evaluate_all_states(
            helper,
            design,
            states,
            raw_checkpoint,
            archive_payloads,
            frozen_summaries,
        )
    finally:
        try:
            model = live.get("model")
            if model is not None:
                restore["evaluation_model_instantiated"] = True
                restore["finally_raw_restore_attempted"] = True
                original_copy(model, states["raw"], helper)
                formal.verify_live_actor_state(model, states["raw"], helper)
                observed = helper.model_state_sha256(model.state_dict())
                expected = helper.model_state_sha256(states["raw"])
                if observed != expected:
                    raise RuntimeError("full-train evaluator final raw hash mismatch")
                restore["final_raw_model_state_sha256"] = observed
                restore["finally_raw_restore_pass"] = True
        finally:
            formal.MODEL_ORDER = original_order
            formal.ALPHAS = original_alphas
            formal.copy_actor_state_to_model = original_copy
    if not restore["finally_raw_restore_pass"]:
        raise RuntimeError("full-train evaluator did not complete finally raw restore")
    return panel_results, execution, restore


def decide_fulltrain(panel_results: Mapping[str, Any]) -> dict[str, Any]:
    panel_gates: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        result = panel_results["candidate"][panel]
        transitions = result["raw_transition_evidence"]
        count_value = result["count_value_integrity"]
        safety_cw = {
            metric: int(transitions[metric]["cells"]["cw"]["count"])
            for metric in SAFETY_METRICS
        }
        correctness = count_value["correctness_transitions"]
        correctness_changes = {
            metric: {
                cell: int(correctness[metric]["cells"][cell]["count"])
                for cell in ("cw", "wc")
            }
            for metric in ("count_correct", "value_correct")
        }
        checks = {
            "six_policy_metrics_zero_cw": all(
                value == 0 for value in safety_cw.values()
            ),
            "native_count_logits_tensor_exact_raw_every_row": (
                int(count_value["native_count_logits_mismatch_count"]) == 0
            ),
            "native_value_logits_tensor_exact_raw_every_row": (
                int(count_value["native_value_logits_mismatch_count"]) == 0
            ),
            "count_value_correctness_has_no_transition": all(
                values["cw"] == 0 and values["wc"] == 0
                for values in correctness_changes.values()
            ),
            "all_count_value_fingerprints_exact_raw": all(
                bool(value)
                for value in count_value["fingerprint_exact_raw"].values()
            ),
        }
        panel_gates[panel] = {
            "rows": int(result["rows"]),
            "safety_cw_counts": safety_cw,
            "count_value_correctness_changed_cells": correctness_changes,
            "count_value_fingerprint_exact_raw": count_value[
                "fingerprint_exact_raw"
            ],
            "checks": checks,
            "pass": all(checks.values()),
        }
    pf_transitions = panel_results["candidate"]["pokemonfan"][
        "raw_transition_evidence"
    ]
    pf_wc = {
        metric: int(pf_transitions[metric]["cells"]["wc"]["count"])
        for metric in MAIN_METRICS
    }
    pf_net = {
        metric: int(pf_transitions[metric]["candidate_correct_minus_raw_correct"])
        for metric in MAIN_METRICS
    }
    pf_checks = {
        metric: (
            pf_wc[metric] >= minimum
            and pf_net[metric] >= minimum
            and int(pf_transitions[metric]["cells"]["cw"]["count"]) == 0
        )
        for metric, minimum in PF_MINIMUM_WC.items()
    }
    checks = {
        "all_three_panels_six_policy_metrics_zero_cw_and_count_value_exact": all(
            item["pass"] for item in panel_gates.values()
        ),
        "pokemonfan_main_wc_and_net_close_raw_gap": all(pf_checks.values()),
    }
    passed = all(checks.values())
    return {
        "status": "full_train_gate_pass" if passed else "full_train_gate_fail",
        "checks": checks,
        "panel_gates": panel_gates,
        "pokemonfan_minimum_wc": dict(PF_MINIMUM_WC),
        "pokemonfan_observed_wc": pf_wc,
        "pokemonfan_observed_net_gains": pf_net,
        "pokemonfan_threshold_checks": pf_checks,
        "pass": passed,
        "materialization_allowed_next": passed,
        "model_materialized": False,
        "submission_performed": False,
    }


def load_expanded_dependencies(
    expanded: ModuleType,
) -> tuple[
    ModuleType,
    dict[str, Any],
    ModuleType,
    dict[str, Any],
    ModuleType,
    dict[str, Any],
    ModuleType,
    dict[str, Any],
]:
    base_cutting, base_cutting_evidence = expanded.import_frozen(
        expanded.CUTTING,
        expanded.CUTTING_SHA256,
        "fulltrain_v3_base_cutting_v2",
    )
    fulltrain_v2, fulltrain_v2_evidence = expanded.import_frozen(
        expanded.FULLTRAIN,
        expanded.FULLTRAIN_SHA256,
        "fulltrain_v3_old_fulltrain_v2",
    )
    geometry, geometry_evidence = base_cutting.import_frozen(
        base_cutting.GEOMETRY,
        base_cutting.GEOMETRY_SHA256,
        "fulltrain_v3_geometry",
    )
    ram, ram_evidence = base_cutting.import_frozen(
        base_cutting.RAM_RUNNER,
        base_cutting.RAM_RUNNER_SHA256,
        "fulltrain_v3_ram_runner",
    )
    return (
        base_cutting,
        base_cutting_evidence,
        fulltrain_v2,
        fulltrain_v2_evidence,
        geometry,
        geometry_evidence,
        ram,
        ram_evidence,
    )


def run_gate(
    source: bytes,
    static: Mapping[str, Any],
    cutting: ModuleType,
    cutting_evidence: Mapping[str, Any],
    formal: ModuleType,
    formal_evidence: Mapping[str, Any],
    interfaces: Mapping[str, Any],
) -> dict[str, Any]:
    (
        base_cutting,
        base_cutting_evidence,
        fulltrain_v2,
        fulltrain_v2_evidence,
        geometry,
        geometry_evidence,
        ram,
        ram_evidence,
    ) = load_expanded_dependencies(cutting)
    cutting_source, _ = cutting.read_regular_bytes(
        cutting.SCRIPT,
        CUTTING_SHA256,
        "fulltrain cutting-plane runner",
        expected_mode=FROZEN_MODE,
    )
    cutting_static = cutting.static_audit(cutting_source)
    captured: dict[str, Any] = {}

    def consumer(context: Mapping[str, Any]) -> None:
        capture_candidate(context, captured, geometry)

    selected_result = cutting.run_cuttingplane(
        cutting_source,
        cutting_static,
        base_cutting,
        base_cutting_evidence,
        fulltrain_v2,
        fulltrain_v2_evidence,
        geometry,
        geometry_evidence,
        ram,
        ram_evidence,
        formal,
        formal_evidence,
        candidate_consumer=consumer,
    )
    if (
        selected_result.get("status")
        != "expanded_selected_row_adaptive_optimization_success"
    ):
        if captured:
            raise RuntimeError("closed selected-row run unexpectedly invoked consumer")
        return {
            "schema_version": SCHEMA,
            "status": "closed_without_selected_row_candidate",
            "input_lock": {
                "self": {"sha256": sha256_bytes(source)},
                "cutting": dict(cutting_evidence),
                "formal_evaluator": dict(formal_evidence),
            },
            "interface_audit": dict(interfaces),
            "selected_row_result": selected_result,
            "full_train_evaluated": False,
            "model_materialized": False,
            "submission_performed": False,
        }
    selected_decision = selected_result["decision"]
    selected_contract_checks = {
        "success_iteration_exact_5": int(
            selected_decision["success_iteration_after_c0"]
        )
        == EXPECTED_SELECTED_SUCCESS_ITERATION,
        "terminal_cumulative_l2_exact": math.isclose(
            float(selected_decision["terminal_cumulative_l2"]),
            EXPECTED_TERMINAL_CUMULATIVE_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "terminal_cumulative_sha256_exact": selected_decision[
            "terminal_cumulative_float64_le_sha256"
        ]
        == EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "terminal_model_state_sha256_exact": selected_decision[
            "candidate_model_state_sha256_before_restore"
        ]
        == EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "final_active_pair_count_exact_24": int(
            selected_result["active_pair_contract"]["final_count"]
        )
        == EXPECTED_FINAL_ACTIVE_PAIR_COUNT,
        "final_active_pair_ledger_sha256_exact": selected_result[
            "active_pair_contract"
        ]["final_canonical_ledger_sha256"]
        == EXPECTED_FINAL_ACTIVE_PAIR_LEDGER_SHA256,
    }
    if not all(selected_contract_checks.values()):
        raise RuntimeError(
            f"selected-row frozen result drift: {selected_contract_checks}"
        )
    if not captured or not bool(selected_result["final_integrity"]["pass"]):
        raise RuntimeError("selected candidate capture/finally restoration failed")
    if not bool(
        selected_result["decision"][
            "candidate_consumer_called_before_finally_restore"
        ]
    ):
        raise RuntimeError("selected result did not record consumer invocation")
    if (
        captured["audit"]["terminal_cumulative_float64_le_sha256"]
        != selected_result["decision"]["terminal_cumulative_float64_le_sha256"]
    ):
        raise RuntimeError("captured candidate cumulative SHA differs from result")
    captured_contract_checks = {
        "consumer_call_count_exact_1": int(captured["audit"]["consumer_call_count"])
        == 1,
        "success_iteration_exact_5": int(captured["audit"]["success_iteration"])
        == EXPECTED_SELECTED_SUCCESS_ITERATION,
        "terminal_l2_exact": math.isclose(
            float(captured["audit"]["terminal_cumulative_l2"]),
            EXPECTED_TERMINAL_CUMULATIVE_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "terminal_cumulative_sha_exact": captured["audit"][
            "terminal_cumulative_float64_le_sha256"
        ]
        == EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "candidate_model_sha_exact": captured["audit"][
            "candidate_model_state_sha256"
        ]
        == EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "active_pair_count_exact_24": int(captured["audit"]["active_pair_count"])
        == EXPECTED_FINAL_ACTIVE_PAIR_COUNT,
        "candidate_nonactor_exact_raw": captured["audit"][
            "candidate_nonactor_sha256"
        ]
        == captured["audit"]["raw_nonactor_sha256"],
        "changed_actor_set_exact_expected_five": bool(
            captured["audit"]["actor_scope"]["scope_checks"]
            ["changed_actor_set_exact_expected_five"]
        ),
        "residual2_bias_unchanged": captured["audit"]["actor_scope"][
            "unchanged_allowed_parameter_names"
        ]
        == list(EXPECTED_UNCHANGED_ACTOR_NAMES),
    }
    if not all(captured_contract_checks.values()):
        raise RuntimeError(
            f"captured expanded candidate contract drift: {captured_contract_checks}"
        )

    design, archives, fulltrain_evidence, frozen_summaries = load_fulltrain_inputs(
        formal
    )
    states = {
        "raw": captured["raw_state"],
        "candidate": captured["candidate_state"],
    }
    panel_results, execution, evaluator_restore = evaluate_raw_candidate(
        formal,
        design,
        states,
        captured["checkpoint"],
        archives,
        frozen_summaries,
    )
    expected_ledger = [
        {"ordinal": ordinal, "model": model, "panel": panel}
        for ordinal, (model, panel) in enumerate(
            (
                (model, panel)
                for model in ("raw", "candidate")
                for panel in PANEL_ORDER
            ),
            start=1,
        )
    ]
    observed_ledger = [
        {key: item[key] for key in ("ordinal", "model", "panel")}
        for item in execution["completion_ledger"]
    ]
    execution_checks = {
        "batch_size_exact_256": int(execution["batch_size"])
        == EXPECTED_BATCH_SIZE,
        "evaluation_count_exact_6": (
            int(execution["evaluation_count_exact"]) == EXPECTED_EVALUATIONS
        ),
        "completion_ledger_exact_raw_then_candidate_three_panels": (
            observed_ledger == expected_ledger
        ),
        "all_panel_rows_exact": all(
            int(panel_results[model][panel]["rows"]) == EXPECTED_ROWS[panel]
            for model in ("raw", "candidate")
            for panel in PANEL_ORDER
        ),
        "total_unique_train_rows_24050": sum(EXPECTED_ROWS.values())
        == EXPECTED_TOTAL_ROWS,
        "validation_members_not_opened": not bool(
            execution["validation_member_payloads_opened"]
        ),
        "evaluator_finally_raw_restore_pass": bool(
            evaluator_restore["finally_raw_restore_pass"]
        ),
    }
    if not all(execution_checks.values()):
        raise RuntimeError(f"full-train execution integrity failed: {execution_checks}")
    decision = decide_fulltrain(panel_results)
    return {
        "schema_version": SCHEMA,
        "status": decision["status"],
        "input_lock": {
            "self": {
                "path": str(SCRIPT.relative_to(ROOT)),
                "sha256": sha256_bytes(source),
            },
            "cutting": dict(cutting_evidence),
            "base_cutting_v2": dict(base_cutting_evidence),
            "fulltrain_v2": dict(fulltrain_v2_evidence),
            "geometry": dict(geometry_evidence),
            "ram_runner": dict(ram_evidence),
            "formal_evaluator": dict(formal_evidence),
            "full_train": fulltrain_evidence,
        },
        "interface_audit": dict(interfaces),
        "selected_row_result": selected_result,
        "selected_row_frozen_result_checks": selected_contract_checks,
        "candidate_capture": captured["audit"],
        "candidate_capture_contract_checks": captured_contract_checks,
        "execution": {
            "device": execution["device"],
            "batch_size": execution["batch_size"],
            "workers": execution["workers"],
            "evaluation_count_exact": execution["evaluation_count_exact"],
            "completion_ledger": execution["completion_ledger"],
            "dataset_audits": execution["dataset_audits"],
            "checks": execution_checks,
            "candidate_generator_finally_raw_restore": selected_result[
                "final_integrity"
            ],
            "fulltrain_evaluator_finally_raw_restore": evaluator_restore,
        },
        "evaluations": panel_results,
        "decision": decision,
        "scope_audit": {
            "selected_train_rows_then_all_24050_train_rows": True,
            "validation_members_opened": False,
            "validation_results_read": False,
            "training_optimizer_backward": False,
            "hyperparameter_sweep": False,
            "candidate_RAM_only": True,
            "model_artifact_writes": 0,
            "evidence_file_writes": 0,
            "stdout_only": True,
            "network_upload_submission": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="run")
    args = parser.parse_args()
    validate_runtime()
    source, self_evidence = read_regular_bytes(
        SCRIPT, sha256_bytes(SCRIPT.read_bytes()), "full-train wrapper"
    )
    static = local_static_audit(source)
    cutting, cutting_evidence = import_frozen(
        CUTTING, CUTTING_SHA256, "u468_cuttingplane_v2_frozen"
    )
    formal, formal_evidence = import_frozen(
        FORMAL, FORMAL_SHA256, "u468_formal_v3_evaluator_frozen"
    )
    interfaces = interface_audit(cutting, formal)
    if args.mode == "static":
        (
            base_cutting,
            base_cutting_evidence,
            fulltrain_v2,
            fulltrain_v2_evidence,
            geometry,
            geometry_evidence,
            ram,
            ram_evidence,
        ) = load_expanded_dependencies(cutting)
        expanded_interfaces = cutting.interface_audit(
            base_cutting, fulltrain_v2, geometry, ram, formal
        )
        cutting_static = cutting.static_audit(CUTTING.read_bytes())
        base_cutting_static = base_cutting.static_audit(
            base_cutting.SCRIPT.read_bytes()
        )
        fulltrain_v2_static = fulltrain_v2.local_static_audit(
            fulltrain_v2.SCRIPT.read_bytes()
        )
        geometry_static = geometry.static_audit(geometry.SCRIPT.read_bytes())
        ram_static = ram.static_audit(ram.SCRIPT.read_bytes())
        formal_static = formal.ast_audit()
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "dependencies": {
                "cutting": cutting_evidence,
                "base_cutting_v2": base_cutting_evidence,
                "fulltrain_v2": fulltrain_v2_evidence,
                "geometry": geometry_evidence,
                "ram_runner": ram_evidence,
                "formal_evaluator": formal_evidence,
            },
            "interface_audit": interfaces,
            "expanded_interface_audit": expanded_interfaces,
            "audit": {
                "local": static,
                "cutting": cutting_static,
                "base_cutting_v2": base_cutting_static,
                "fulltrain_v2": fulltrain_v2_static,
                "geometry": geometry_static,
                "ram_runner": ram_static,
                "formal_evaluator": formal_static,
            },
            "run_executed": False,
            "writes_performed": False,
        }
    else:
        result = run_gate(
            source,
            static,
            cutting,
            cutting_evidence,
            formal,
            formal_evidence,
            interfaces,
        )
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()

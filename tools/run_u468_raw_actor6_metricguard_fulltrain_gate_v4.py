#!/usr/bin/env python3
"""Read-only formal full-train gate for the exploratory CW9 candidate.

The hash-bound CW9 probe reconstructs one preregistered terminal candidate in
RAM and invokes a trusted consumer before its mandatory raw restoration.  This
wrapper CPU-clones that state, verifies its exact model/vector/ledger identity,
then evaluates raw and candidate over all 24,050 training rows in the frozen
formal B256 protocol.  No candidate is materialized here.

There are no validation reads, optimizer/backward calls, checkpoint/result
writes, network calls, uploads, or submissions.  Output is JSON on stdout only.
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
SCRIPT = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v4.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-fulltrain-gate-v4"

PROBE = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw9_cuttingplane_v1.py"
PROBE_SHA256 = "f77a76f2ac91732e8bc8d9a436ca7a49058b5de50f7b3114db14cffa1fbf5292"
BASE_GATE = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v3.py"
BASE_GATE_SHA256 = "f32c077c0d577bcc7c0ad2e42bdfda1ec1641244004b29efd5c82008cfe92338"
FROZEN_MODE = 0o555

EXPECTED_RAW_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
EXPECTED_RAW_NONACTOR_SHA256 = (
    "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
)
EXPECTED_SUCCESS_ITERATION = 6
EXPECTED_TERMINAL_CUMULATIVE_L2 = 0.007231848125042783
EXPECTED_TERMINAL_CUMULATIVE_SHA256 = (
    "846f3d929c6dc9f3b94f8e5bd49539f262d8393881f7fcde92e891b6304c5314"
)
EXPECTED_TERMINAL_MODEL_STATE_SHA256 = (
    "bfca16815a2d58506c4afd202aa47e1c12c47a5d7a786d93c6377438f2686cd9"
)
EXPECTED_ACTIVE_PAIR_COUNT = 29
EXPECTED_ACTIVE_PAIR_LEDGER_SHA256 = (
    "ffff3d9e77f9cf15bfc1cea3be8051df9539c5c0c9a75a5a0dfade74abc63ceb"
)
EXPECTED_ROWS = {"flg": 9443, "pokemonfan": 9487, "core5": 5120}
EXPECTED_TOTAL_ROWS = 24050
PANEL_ORDER = ("flg", "pokemonfan", "core5")
EXPECTED_BATCH_SIZE = 256
EXPECTED_EVALUATIONS = 6


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
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    mode = stat.S_IMODE(after.st_mode)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or after.st_size != len(payload)
    ):
        raise RuntimeError(f"{label} changed while reading")
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
    }


def import_frozen(path: Path, expected_sha256: str, label: str) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path, expected_sha256, label, expected_mode=FROZEN_MODE
    )
    spec = importlib.util.spec_from_file_location(
        f"_fulltrain_v4_{path.stem}_{expected_sha256[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {label}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def validate_runtime() -> None:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    forbidden_calls = {
        "save",
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
        "replace",
        "rename",
        "unlink",
        "remove",
        "rmtree",
        "backward",
        "step",
    }
    forbidden_import_roots = {
        "requests",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "kaggle",
    }
    call_hits: list[dict[str, Any]] = []
    import_hits: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in forbidden_calls:
                call_hits.append({"name": name, "line": node.lineno})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_import_roots:
                    import_hits.append({"name": name, "line": node.lineno})
    checks = {
        "ast_parse": True,
        "no_write_optimizer_backward_or_network_calls": not call_hits,
        "no_network_or_submission_imports": not import_hits,
        "expected_rows_sum_24050": sum(EXPECTED_ROWS.values())
        == EXPECTED_TOTAL_ROWS,
        "terminal_hashes_hex64": all(
            len(value) == 64
            for value in (
                EXPECTED_TERMINAL_CUMULATIVE_SHA256,
                EXPECTED_TERMINAL_MODEL_STATE_SHA256,
                EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"fulltrain v4 static audit failed: {checks}")
    return {
        "checks": checks,
        "forbidden_call_hits": call_hits,
        "forbidden_import_hits": import_hits,
        "stdout_only": True,
    }


def interface_audit(probe: ModuleType, base: ModuleType) -> dict[str, Any]:
    parameters = inspect.signature(probe.run_probe).parameters
    checks = {
        "probe_candidate_consumer_present": "candidate_consumer" in parameters,
        "probe_candidate_consumer_default_none": (
            "candidate_consumer" in parameters
            and parameters["candidate_consumer"].default is None
        ),
        "base_panel_order_exact": tuple(base.PANEL_ORDER) == PANEL_ORDER,
        "base_rows_exact": dict(base.EXPECTED_ROWS) == EXPECTED_ROWS,
        "base_batch_size_exact": int(base.EXPECTED_BATCH_SIZE)
        == EXPECTED_BATCH_SIZE,
        "base_evaluation_count_exact": int(base.EXPECTED_EVALUATIONS)
        == EXPECTED_EVALUATIONS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"fulltrain v4 interface drift: {checks}")
    return checks


def capture_candidate(
    context: Mapping[str, Any],
    holder: dict[str, Any],
    base: ModuleType,
    geometry: ModuleType,
) -> None:
    if holder:
        raise RuntimeError("CW9 candidate consumer called more than once")
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
        "expanded_row_count",
    }
    if not required.issubset(context):
        raise RuntimeError("CW9 candidate context schema drift")
    helper = context["helper"]
    model = context["model"]
    checkpoint = context["checkpoint"]
    raw_source = checkpoint.get("model_state_dict")
    if not isinstance(raw_source, Mapping):
        raise RuntimeError("CW9 checkpoint lacks model_state_dict")
    raw_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in raw_source.items()
    }
    candidate_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    raw_sha = helper.model_state_sha256(raw_state)
    candidate_sha = helper.model_state_sha256(candidate_state)
    live_sha = helper.model_state_sha256(model.state_dict())
    cumulative = context["terminal_cumulative_float64"]
    cumulative_l2 = float((cumulative @ cumulative) ** 0.5)
    ledger = context["active_pair_ledger"]
    ledger_sha = sha256_bytes(canonical_json(ledger))
    selected_gate = context["selected_row_gate"]
    checks = {
        "raw_model_sha_exact": raw_sha == EXPECTED_RAW_MODEL_STATE_SHA256,
        "context_raw_model_sha_exact": context["raw_model_state_sha256"]
        == EXPECTED_RAW_MODEL_STATE_SHA256,
        "candidate_cpu_live_sha_exact": candidate_sha == live_sha,
        "candidate_model_sha_exact": candidate_sha
        == EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "success_iteration_exact_6": int(context["success_iteration"])
        == EXPECTED_SUCCESS_ITERATION,
        "terminal_l2_exact": math.isclose(
            cumulative_l2,
            EXPECTED_TERMINAL_CUMULATIVE_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "terminal_vector_sha_exact": context[
            "terminal_cumulative_float64_le_sha256"
        ]
        == EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "selected_28row_gate_pass": isinstance(selected_gate, Mapping)
        and bool(selected_gate.get("pass")),
        "expanded_row_count_exact_28": int(context["expanded_row_count"])
        == 28,
        "active_pair_count_exact_29": len(ledger)
        == EXPECTED_ACTIVE_PAIR_COUNT,
        "active_pair_ledger_sha_exact": ledger_sha
        == EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
        "raw_nonactor_context_sha_exact": context["raw_nonactor_sha256"]
        == EXPECTED_RAW_NONACTOR_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW9 candidate capture drift: {checks}")
    actor_scope = base.audit_candidate_state(
        helper,
        geometry,
        raw_state,
        candidate_state,
        EXPECTED_RAW_NONACTOR_SHA256,
    )
    holder.update(
        {
            "raw_state": raw_state,
            "candidate_state": candidate_state,
            "checkpoint": checkpoint,
            "audit": {
                "checks": checks,
                "consumer_call_count": 1,
                "raw_model_state_sha256": raw_sha,
                "candidate_model_state_sha256": candidate_sha,
                "terminal_cumulative_l2": cumulative_l2,
                "terminal_cumulative_float64_le_sha256": context[
                    "terminal_cumulative_float64_le_sha256"
                ],
                "active_pair_count": len(ledger),
                "active_pair_ledger_sha256": ledger_sha,
                "actor_scope": actor_scope,
                "RAM_only_CPU_clone_before_probe_finally_restore": True,
            },
        }
    )


def run_gate(
    source: bytes,
    static: Mapping[str, Any],
    probe: ModuleType,
    probe_evidence: Mapping[str, Any],
    base: ModuleType,
    base_evidence: Mapping[str, Any],
    interfaces: Mapping[str, Any],
) -> dict[str, Any]:
    primary, primary_evidence = probe.import_frozen(
        probe.PRIMARY,
        probe.PRIMARY_SHA256,
        "fulltrain v4 frozen CW4 primary",
    )
    primary_source, _ = probe.read_regular_bytes(
        probe.PRIMARY,
        probe.PRIMARY_SHA256,
        "fulltrain v4 frozen CW4 primary source",
        expected_mode=FROZEN_MODE,
    )
    cutting, cutting_evidence = primary.import_frozen(
        primary.CUTTING,
        primary.CUTTING_SHA256,
        "fulltrain v4 frozen base cutting",
    )
    fulltrain_v2, fulltrain_v2_evidence = primary.import_frozen(
        primary.FULLTRAIN,
        primary.FULLTRAIN_SHA256,
        "fulltrain v4 frozen fulltrain v2",
    )
    geometry, geometry_evidence = cutting.import_frozen(
        cutting.GEOMETRY,
        cutting.GEOMETRY_SHA256,
        "fulltrain v4 frozen geometry",
    )
    formal, formal_evidence = fulltrain_v2.import_frozen(
        fulltrain_v2.FORMAL,
        fulltrain_v2.FORMAL_SHA256,
        "fulltrain v4 frozen formal evaluator",
    )
    holder: dict[str, Any] = {}

    def consumer(context: Mapping[str, Any]) -> None:
        capture_candidate(context, holder, base, geometry)

    probe_result = probe.run_probe(
        primary,
        primary_source,
        primary_evidence,
        candidate_consumer=consumer,
    )
    probe_checks = {
        "probe_status_success": probe_result.get("status")
        == "exploratory_28row_29pair_optimization_success",
        "probe_success_iteration_exact_6": int(
            probe_result["decision"]["success_iteration_after_c0"]
        )
        == EXPECTED_SUCCESS_ITERATION,
        "probe_terminal_l2_exact": math.isclose(
            float(probe_result["decision"]["terminal_cumulative_l2"]),
            EXPECTED_TERMINAL_CUMULATIVE_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "probe_terminal_vector_sha_exact": probe_result["decision"][
            "terminal_cumulative_float64_le_sha256"
        ]
        == EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "probe_terminal_model_sha_exact": probe_result["decision"][
            "candidate_model_state_sha256_before_restore"
        ]
        == EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "probe_terminal_active_ledger_sha_exact": probe_result["decision"][
            "terminal_active_pair_ledger_sha256"
        ]
        == EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
        "probe_consumer_called": bool(
            probe_result["decision"][
                "candidate_consumer_called_before_finally_restore"
            ]
        ),
        "probe_finally_raw_restore_pass": bool(
            probe_result["final_integrity"]["pass"]
        ),
        "candidate_captured_once": bool(holder)
        and int(holder["audit"]["consumer_call_count"]) == 1,
    }
    if not all(probe_checks.values()):
        raise RuntimeError(f"CW9 frozen reproduction drift: {probe_checks}")

    design, archives, fulltrain_evidence, frozen_summaries = (
        base.load_fulltrain_inputs(formal)
    )
    states = {
        "raw": holder["raw_state"],
        "candidate": holder["candidate_state"],
    }
    panel_results, execution, evaluator_restore = base.evaluate_raw_candidate(
        formal,
        design,
        states,
        holder["checkpoint"],
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
        "evaluation_count_exact_6": int(execution["evaluation_count_exact"])
        == EXPECTED_EVALUATIONS,
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
        raise RuntimeError(
            f"fulltrain v4 execution integrity failed: {execution_checks}"
        )
    decision = base.decide_fulltrain(panel_results)
    return {
        "schema_version": SCHEMA,
        "status": decision["status"],
        "input_lock": {
            "self": {
                "path": str(SCRIPT.relative_to(ROOT)),
                "sha256": sha256_bytes(source),
            },
            "probe": dict(probe_evidence),
            "base_gate_v3": dict(base_evidence),
            "primary_cw4": dict(primary_evidence),
            "base_cutting": dict(cutting_evidence),
            "fulltrain_v2": dict(fulltrain_v2_evidence),
            "geometry": dict(geometry_evidence),
            "formal_evaluator": dict(formal_evidence),
            "full_train": fulltrain_evidence,
        },
        "interface_audit": dict(interfaces),
        "probe_frozen_reproduction_checks": probe_checks,
        "probe_result": probe_result,
        "candidate_capture": holder["audit"],
        "execution": {
            "device": execution["device"],
            "batch_size": execution["batch_size"],
            "workers": execution["workers"],
            "evaluation_count_exact": execution["evaluation_count_exact"],
            "completion_ledger": execution["completion_ledger"],
            "dataset_audits": execution["dataset_audits"],
            "checks": execution_checks,
            "candidate_generator_finally_raw_restore": probe_result[
                "final_integrity"
            ],
            "fulltrain_evaluator_finally_raw_restore": evaluator_restore,
        },
        "evaluations": panel_results,
        "decision": decision,
        "scope_audit": {
            "selected_28_train_rows_then_all_24050_train_rows": True,
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
        SCRIPT, None, "fulltrain gate v4", expected_mode=FROZEN_MODE
    )
    static = static_audit(source)
    probe, probe_evidence = import_frozen(
        PROBE, PROBE_SHA256, "frozen CW9 probe"
    )
    base, base_evidence = import_frozen(
        BASE_GATE, BASE_GATE_SHA256, "frozen fulltrain gate v3 utilities"
    )
    interfaces = interface_audit(probe, base)
    if args.mode == "static":
        probe_source, _ = read_regular_bytes(
            PROBE, PROBE_SHA256, "static CW9 probe", expected_mode=FROZEN_MODE
        )
        base_source, _ = read_regular_bytes(
            BASE_GATE,
            BASE_GATE_SHA256,
            "static fulltrain v3 utilities",
            expected_mode=FROZEN_MODE,
        )
        result = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "dependencies": {
                "probe": probe_evidence,
                "base_gate_v3": base_evidence,
            },
            "interface_audit": interfaces,
            "terminal_contract": {
                "success_iteration": EXPECTED_SUCCESS_ITERATION,
                "cumulative_l2": EXPECTED_TERMINAL_CUMULATIVE_L2,
                "cumulative_sha256": EXPECTED_TERMINAL_CUMULATIVE_SHA256,
                "model_state_sha256": EXPECTED_TERMINAL_MODEL_STATE_SHA256,
                "active_pair_count": EXPECTED_ACTIVE_PAIR_COUNT,
                "active_pair_ledger_sha256": EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
            },
            "audit": {
                "local": static,
                "probe": probe.static_audit(probe_source),
                "base_gate_v3": base.local_static_audit(base_source),
            },
            "run_executed": False,
            "cuda_accessed": False,
            "writes_performed": False,
        }
    else:
        result = run_gate(
            source,
            static,
            probe,
            probe_evidence,
            base,
            base_evidence,
            interfaces,
        )
        result["input_lock"]["self"] = self_evidence
        result["local_static_audit"] = static
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()

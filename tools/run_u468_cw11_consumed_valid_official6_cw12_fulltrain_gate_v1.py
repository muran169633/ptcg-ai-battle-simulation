#!/usr/bin/env python3
"""Static-only CW12 formal full-train revalidation adapter skeleton.

The future run path is deliberately fail-closed until one immutable one-shot
stdout record is bound by path, file SHA, terminal model SHA, terminal total
vector SHA, and terminal active-cut ledger SHA.  Once armed, the adapter is
designed to reconstruct the frozen CW11 B33 terminal exactly once, decode and
apply only the terminal delta carried by that record, and then reuse the
frozen v5 -> v4 -> v3 six-evaluation/24,050-row formal gate.

This module must never import or invoke the CW12 official-six optimization
probe.  In particular it must not call its run/outer/stream-loader/snapshot/
oracle/gradient functions.  The consumed specialist views are therefore not
reopened during formal revalidation.  Static mode performs no CUDA work and
the current unbound run mode refuses before reconstructing CW11.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import lzma
import math
import os
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
SCRIPT = TOOLS / (
    "run_u468_cw11_consumed_valid_official6_cw12_fulltrain_gate_v1.py"
)
SCHEMA = "ptcg-u468-cw11-consumed-valid-official6-cw12-fulltrain-gate-v1"
FROZEN_MODE = 0o555

CW11_FORMAL = TOOLS / (
    "run_u468_raw_actor6_metricguard_specialist_valid_"
    "cw11_fulltrain_gate_v1.py"
)
CW11_FORMAL_SHA256 = (
    "3ef943434bc2699b125a7bd907892c877d3645fe8bed92ba576ef746c91fe695"
)
CW11_PROBE = TOOLS / (
    "probe_u468_raw_actor6_metricguard_specialist_valid_"
    "cw11_cuttingplane_v1.py"
)
CW11_PROBE_SHA256 = (
    "23bc74022210942115eaf339a38e710e88ad81ba75d62237e8eb9e2c11e074ee"
)
FULLTRAIN_V5 = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v5.py"
FULLTRAIN_V5_SHA256 = (
    "90ab439a0ca75e7f0a2cf2c8e7c117a07ee5ff9b8b6462ffc5e8380114b404a0"
)
FULLTRAIN_V4 = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v4.py"
FULLTRAIN_V4_SHA256 = (
    "75974150f83455c10bbf6a90b3c571e4daf319f4e06e9c77f67673fbf34b9ea8"
)
FULLTRAIN_V3 = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v3.py"
FULLTRAIN_V3_SHA256 = (
    "f32c077c0d577bcc7c0ad2e42bdfda1ec1641244004b29efd5c82008cfe92338"
)

CUTTING = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v2.py"
CUTTING_SHA256 = (
    "c2866ba8b00eba6b424197a520419a5717401335cc47202b4fcc711611f67503"
)
GEOMETRY = TOOLS / "probe_u468_raw_actor6_metricguard_specialbc_v1.py"
GEOMETRY_SHA256 = (
    "ddecd3a85bc2b43c28854afc56678c21943198ac3fde3613765d6a49e440eedb"
)
RAM = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_ram_ray_v1.py"
RAM_SHA256 = (
    "86b05d4f826576717907141aef2c534f140a1531b87c8e2d28aeb624a2657e4d"
)

EXPECTED_RAW_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
EXPECTED_RAW_NONACTOR_SHA256 = (
    "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
)
EXPECTED_CW11_MODEL_STATE_SHA256 = (
    "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
)
EXPECTED_CW11_TOTAL_FLOAT64_LE_SHA256 = (
    "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
)
EXPECTED_CW11_ACTIVE_LEDGER_SHA256 = (
    "6ae0c07a94627cb6fd0155f265f75d13bd7505b0e5f62cd62c3e26ceb7714436"
)
EXPECTED_CW11_SELECTED_ROWS = 33
EXPECTED_CW11_ACTIVE_PAIRS = 34

ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
ACTOR6_LAYOUT_SHA256 = (
    "b86476b9ccbbeeac7b46754f7f15e349e398aa627fc3dc6060d6c3d6e3bd26cb"
)
ACTOR6_FLAT_LENGTH = 65793
ACTOR6_FLOAT64_BYTES = 526344

EXPECTED_PANEL_ORDER = ("flg", "pokemonfan", "core5")
EXPECTED_ROWS = {"flg": 9443, "pokemonfan": 9487, "core5": 5120}
EXPECTED_TOTAL_ROWS = 24050
EXPECTED_BATCH_SIZE = 256
EXPECTED_EVALUATIONS = 6
EXPECTED_MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)
EXPECTED_SAFETY_METRICS = EXPECTED_MAIN_METRICS + (
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
EXPECTED_PF_MINIMUM_WC = {
    "set_exact": 3,
    "hybrid_order_exact": 3,
    "ordered_exact": 5,
    "top1_correct": 3,
}

# Fail-closed placeholders.  A separate audit must replace every None with the
# immutable identity emitted by the one and only authorized stdout attempt.
RUN_IMPLEMENTATION_ARMED = False
TERMINAL_STDOUT_RECORD: Path | None = None
TERMINAL_STDOUT_RECORD_SHA256: str | None = None
EXPECTED_CW12_MODEL_STATE_SHA256: str | None = None
EXPECTED_CW12_TOTAL_FLOAT64_LE_SHA256: str | None = None
EXPECTED_CW12_ACTIVE_CUT_LEDGER_SHA256: str | None = None
EXPECTED_CW12_ACTIVE_CUT_COUNT: int | None = None
EXPECTED_CW12_TERMINAL_ITERATION: int | None = None
EXPECTED_CW12_TOTAL_FROM_RAW_L2: float | None = None

CW12_REPLAY_STATUS = "cw12_reconstructed_from_frozen_one_shot_stdout"
LEGACY_V4_EXPECTED_STATUS = "exploratory_28row_29pair_optimization_success"

FORBIDDEN_OFFICIAL6_CALL_NAMES = frozenset(
    {
        "run_outer_cutting_plane",
        "run_cw12_consumer",
        "load_official_six_streams",
        "snapshot_official_six_views",
        "full_stream_separation_oracle",
        "official_context_gradients",
        "fixed_repair_gates_and_cuts",
        "terminal_acceptance",
    }
)
FORBIDDEN_WRITE_OR_TRAIN_CALL_NAMES = frozenset(
    {
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
        raise RuntimeError(f"{label} SHA drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
    }


def import_frozen(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path,
        expected_sha256,
        label,
        expected_mode=FROZEN_MODE,
    )
    spec = importlib.util.spec_from_file_location(
        f"_cw12_formal_{path.stem}_{expected_sha256[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {label}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def validate_runtime() -> dict[str, bool]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated_exact": sys.flags.isolated == 1,
        "dont_write_bytecode_exact": sys.flags.dont_write_bytecode == 1,
        "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        == ":4096:8",
    }
    if not all(checks.values()):
        raise RuntimeError(f"runtime contract failed: {checks}")
    return checks


def call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def is_cw11_run_probe_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_probe"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cw11"
    )


def terminal_binding_audit() -> dict[str, Any]:
    bindings = {
        "terminal_stdout_record_path": TERMINAL_STDOUT_RECORD,
        "terminal_stdout_record_file_sha256": TERMINAL_STDOUT_RECORD_SHA256,
        "terminal_model_state_sha256": EXPECTED_CW12_MODEL_STATE_SHA256,
        "terminal_total_float64_le_sha256": (
            EXPECTED_CW12_TOTAL_FLOAT64_LE_SHA256
        ),
        "terminal_active_cut_ledger_sha256": (
            EXPECTED_CW12_ACTIVE_CUT_LEDGER_SHA256
        ),
        "terminal_active_cut_count": EXPECTED_CW12_ACTIVE_CUT_COUNT,
        "terminal_iteration": EXPECTED_CW12_TERMINAL_ITERATION,
        "terminal_total_from_raw_l2": EXPECTED_CW12_TOTAL_FROM_RAW_L2,
    }
    unbound = [name for name, value in bindings.items() if value is None]
    return {
        "run_implementation_armed": RUN_IMPLEMENTATION_ARMED,
        "bindings": {
            name: None if value is None else str(value)
            for name, value in bindings.items()
        },
        "unbound_fields": unbound,
        "complete": RUN_IMPLEMENTATION_ARMED and not unbound,
        "fail_closed": not RUN_IMPLEMENTATION_ARMED or bool(unbound),
    }


def static_source_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    official6_hits: list[dict[str, Any]] = []
    write_or_train_hits: list[dict[str, Any]] = []
    forbidden_import_hits: list[dict[str, Any]] = []
    cw11_calls: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = call_name(node)
            if name in FORBIDDEN_OFFICIAL6_CALL_NAMES:
                official6_hits.append({"name": name, "line": node.lineno})
            if name in FORBIDDEN_WRITE_OR_TRAIN_CALL_NAMES:
                write_or_train_hits.append({"name": name, "line": node.lineno})
            if is_cw11_run_probe_call(node):
                cw11_calls.append(node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".", 1)[0] in {
                    "requests",
                    "urllib",
                    "http",
                    "socket",
                    "subprocess",
                    "kaggle",
                }:
                    forbidden_import_hits.append(
                        {"name": name, "line": node.lineno}
                    )
    source_text = source.decode("utf-8")
    solver_filename = (
        "probe_u468_cw11_consumed_valid_official6_"
        "cw12_cuttingplane_v1.py"
    )
    checks = {
        "ast_parse": True,
        "official6_call_sites_exact_0": len(official6_hits) == 0,
        "cw11_B33_run_probe_call_sites_exact_1": len(cw11_calls) == 1,
        "final_CW12_solver_filename_absent": solver_filename not in source_text,
        "no_write_optimizer_backward_calls": not write_or_train_hits,
        "no_network_or_submission_imports": not forbidden_import_hits,
        "current_terminal_bindings_fail_closed": terminal_binding_audit()[
            "fail_closed"
        ],
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW12 formal skeleton source audit failed: {checks}")
    return {
        "checks": checks,
        "official6_call_sites": len(official6_hits),
        "official6_call_hits": official6_hits,
        "cw11_B33_run_probe_call_sites": len(cw11_calls),
        "cw11_B33_run_probe_call_lines": cw11_calls,
        "write_or_train_call_hits": write_or_train_hits,
        "forbidden_import_hits": forbidden_import_hits,
        "stdout_only": True,
    }


def frozen_formal_contract_audit(
    cw11_formal: ModuleType,
    cw11: ModuleType,
    v5: ModuleType,
    v4: ModuleType,
    base: ModuleType,
) -> dict[str, bool]:
    checks = {
        "cw11_formal_to_probe_exact": Path(cw11_formal.CW11) == CW11_PROBE
        and str(cw11_formal.CW11_SHA256) == CW11_PROBE_SHA256,
        "cw11_formal_to_v5_exact": Path(cw11_formal.FULLTRAIN_V5)
        == FULLTRAIN_V5
        and str(cw11_formal.FULLTRAIN_V5_SHA256) == FULLTRAIN_V5_SHA256,
        "v5_to_v4_exact": Path(v5.V4) == FULLTRAIN_V4
        and str(v5.V4_SHA256) == FULLTRAIN_V4_SHA256,
        "v4_to_v3_exact": Path(v4.BASE_GATE) == FULLTRAIN_V3
        and str(v4.BASE_GATE_SHA256) == FULLTRAIN_V3_SHA256,
        "cw11_raw_model_anchor_exact": cw11.EXPECTED_RAW_MODEL_STATE_SHA256
        == EXPECTED_RAW_MODEL_STATE_SHA256,
        "cw11_raw_nonactor_anchor_exact": cw11.EXPECTED_RAW_NONACTOR_SHA256
        == EXPECTED_RAW_NONACTOR_SHA256,
        "cw11_model_anchor_exact": cw11_formal.EXPECTED_TERMINAL_MODEL_STATE_SHA256
        == EXPECTED_CW11_MODEL_STATE_SHA256,
        "cw11_vector_anchor_exact": cw11_formal.EXPECTED_TERMINAL_CUMULATIVE_SHA256
        == EXPECTED_CW11_TOTAL_FLOAT64_LE_SHA256,
        "cw11_ledger_anchor_exact": cw11_formal.EXPECTED_ACTIVE_PAIR_LEDGER_SHA256
        == EXPECTED_CW11_ACTIVE_LEDGER_SHA256,
        "cw11_B33_rows_exact": int(cw11_formal.EXPECTED_SELECTED_ROW_COUNT)
        == EXPECTED_CW11_SELECTED_ROWS,
        "cw11_B33_pairs_exact": int(cw11_formal.EXPECTED_ACTIVE_PAIR_COUNT)
        == EXPECTED_CW11_ACTIVE_PAIRS,
        "panel_order_exact": tuple(base.PANEL_ORDER) == EXPECTED_PANEL_ORDER,
        "panel_rows_exact": dict(base.EXPECTED_ROWS) == EXPECTED_ROWS,
        "total_unique_rows_exact_24050": int(base.EXPECTED_TOTAL_ROWS)
        == EXPECTED_TOTAL_ROWS
        and sum(EXPECTED_ROWS.values()) == EXPECTED_TOTAL_ROWS,
        "batch_size_exact_256": int(base.EXPECTED_BATCH_SIZE)
        == EXPECTED_BATCH_SIZE,
        "evaluation_count_exact_6": int(base.EXPECTED_EVALUATIONS)
        == EXPECTED_EVALUATIONS,
        "main_metrics_exact": tuple(base.MAIN_METRICS) == EXPECTED_MAIN_METRICS,
        "six_policy_safety_metrics_exact": tuple(base.SAFETY_METRICS)
        == EXPECTED_SAFETY_METRICS,
        "pokemonfan_thresholds_exact_3_3_5_3": dict(base.PF_MINIMUM_WC)
        == EXPECTED_PF_MINIMUM_WC,
        "v5_run_gate_reused": callable(v5.run_gate),
        "v4_capture_reused": callable(v4.capture_candidate),
        "v3_evaluator_reused": callable(base.evaluate_raw_candidate),
        "v3_decision_reused": callable(base.decide_fulltrain),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen formal contract drift: {checks}")
    return checks


def decode_xz_base64_payload(value: Mapping[str, Any]) -> bytes:
    if str(value.get("base64_variant")) != "standard_RFC4648":
        raise RuntimeError("terminal payload base64 variant drift")
    encoded = "".join(str(chunk) for chunk in value["base64_chunks_76"])
    compressed = base64.b64decode(encoded, altchars=None, validate=True)
    if sha256_bytes(compressed) != str(value["compressed_sha256"]):
        raise RuntimeError("terminal compressed payload SHA drift")
    raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    if (
        len(raw) != int(value["raw_bytes"])
        or sha256_bytes(raw) != str(value["raw_sha256"])
    ):
        raise RuntimeError("terminal decoded payload drift")
    return raw


def actor6_layout(parameters: Sequence[Any]) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    offset = 0
    for name, parameter in zip(ACTOR6_NAMES, parameters, strict=True):
        count = int(parameter.numel())
        layout.append(
            {
                "name": name,
                "shape": [int(value) for value in parameter.shape],
                "numel": count,
                "start": offset,
                "stop": offset + count,
            }
        )
        offset += count
    if (
        offset != ACTOR6_FLAT_LENGTH
        or sha256_bytes(canonical_json(layout)) != ACTOR6_LAYOUT_SHA256
        or any(str(parameter.dtype) != "torch.float32" for parameter in parameters)
    ):
        raise RuntimeError("actor6 layout/dtype drift")
    return layout


def load_bound_terminal_record() -> tuple[dict[str, Any], dict[str, Any]]:
    audit = terminal_binding_audit()
    if not audit["complete"]:
        raise RuntimeError(
            "CW12 formal remains fail-closed: bind and audit one frozen terminal record"
        )
    assert TERMINAL_STDOUT_RECORD is not None
    assert TERMINAL_STDOUT_RECORD_SHA256 is not None
    payload, evidence = read_regular_bytes(
        TERMINAL_STDOUT_RECORD,
        TERMINAL_STDOUT_RECORD_SHA256,
        "frozen CW12 one-shot stdout terminal record",
        expected_mode=0o444,
    )
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise RuntimeError("terminal stdout record must be a JSON object")
    return record, evidence


def validate_and_decode_terminal(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    terminal = record.get("terminal")
    if not isinstance(terminal, Mapping):
        raise RuntimeError("terminal stdout record lacks terminal object")
    decision = terminal.get("decision")
    payload = terminal.get("reconstruction_payload")
    ledger = terminal.get("active_cut_ledger")
    if not isinstance(decision, Mapping) or not isinstance(payload, Mapping):
        raise RuntimeError("terminal decision/reconstruction payload missing")
    if not isinstance(ledger, list):
        raise RuntimeError("terminal active-cut ledger missing")
    assert EXPECTED_CW12_MODEL_STATE_SHA256 is not None
    assert EXPECTED_CW12_TOTAL_FLOAT64_LE_SHA256 is not None
    assert EXPECTED_CW12_ACTIVE_CUT_LEDGER_SHA256 is not None
    assert EXPECTED_CW12_ACTIVE_CUT_COUNT is not None
    assert EXPECTED_CW12_TERMINAL_ITERATION is not None
    assert EXPECTED_CW12_TOTAL_FROM_RAW_L2 is not None
    additional_raw = decode_xz_base64_payload(
        payload["additional_from_CW11_float64_le"]
    )
    total_raw = decode_xz_base64_payload(
        payload["terminal_total_from_raw_float64_le"]
    )
    additional = np.frombuffer(additional_raw, dtype="<f8").astype(
        np.float64, copy=True
    )
    total = np.frombuffer(total_raw, dtype="<f8").astype(np.float64, copy=True)
    ledger_sha = sha256_bytes(canonical_json(ledger))
    checks = {
        "one_shot_status_success": record.get("status")
        == "consumed_valid_optimization_closure_first_feasible",
        "one_shot_run_executed": bool(record.get("run_executed")),
        "one_shot_writes_zero": record.get("writes_performed") is False,
        "terminal_status_success": terminal.get("status")
        == "consumed_valid_optimization_closure_first_feasible",
        "eligible_only_for_formal": bool(
            decision.get("eligible_only_for_formal_fulltrain_revalidation")
        ),
        "not_promotion_evidence": decision.get("eligible_as_promotion_evidence")
        is False,
        "terminal_iteration_exact": int(decision.get("terminal_iteration", -1))
        == EXPECTED_CW12_TERMINAL_ITERATION,
        "terminal_model_sha_exact": payload["anchor"][
            "terminal_model_state_sha256"
        ]
        == EXPECTED_CW12_MODEL_STATE_SHA256,
        "raw_anchor_exact": payload["anchor"]["raw_model_state_sha256"]
        == EXPECTED_RAW_MODEL_STATE_SHA256,
        "cw11_model_anchor_exact": payload["anchor"]["cw11_model_state_sha256"]
        == EXPECTED_CW11_MODEL_STATE_SHA256,
        "cw11_vector_anchor_exact": payload["anchor"][
            "cw11_total_float64_le_sha256"
        ]
        == EXPECTED_CW11_TOTAL_FLOAT64_LE_SHA256,
        "actor_layout_sha_exact": payload["actor_layout_sha256"]
        == ACTOR6_LAYOUT_SHA256,
        "additional_shape_exact": additional.shape == (ACTOR6_FLAT_LENGTH,),
        "total_shape_exact": total.shape == (ACTOR6_FLAT_LENGTH,),
        "total_bytes_exact": len(total_raw) == ACTOR6_FLOAT64_BYTES,
        "total_vector_sha_exact": sha256_bytes(total_raw)
        == EXPECTED_CW12_TOTAL_FLOAT64_LE_SHA256,
        "terminal_total_l2_exact": math.isclose(
            float(np.linalg.norm(total)),
            EXPECTED_CW12_TOTAL_FROM_RAW_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "active_cut_count_exact": len(ledger) == EXPECTED_CW12_ACTIVE_CUT_COUNT,
        "active_cut_ledger_sha_exact": ledger_sha
        == EXPECTED_CW12_ACTIVE_CUT_LEDGER_SHA256
        == terminal.get("active_cut_ledger_sha256"),
        "downstream_no_solver_rerun": bool(
            terminal.get("downstream_contract", {}).get(
                "formal_must_not_call_official6_solver_again"
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"bound CW12 terminal record drift: {checks}")
    return {
        "terminal": terminal,
        "decision": decision,
        "payload": payload,
        "ledger": ledger,
        "ledger_sha256": ledger_sha,
        "additional": additional,
        "total": total,
        "checks": checks,
    }


def apply_bound_terminal_inside_cw11(
    context: Mapping[str, Any],
    decoded: Mapping[str, Any],
    *,
    cutting: ModuleType,
    geometry: ModuleType,
    ram: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    helper = context["helper"]
    model = context["model"]
    torch = helper.torch
    parameters = geometry.configure_actor6(model)
    layout = actor6_layout(parameters)
    cw11_total = np.asarray(
        context["terminal_cumulative_float64"], dtype=np.float64
    )
    additional = np.asarray(decoded["additional"], dtype=np.float64)
    total = np.asarray(decoded["total"], dtype=np.float64)
    computed_total = np.add(cw11_total, additional, dtype=np.float64)
    checks = {
        "incoming_CW11_model_exact": helper.model_state_sha256(model.state_dict())
        == EXPECTED_CW11_MODEL_STATE_SHA256,
        "incoming_CW11_vector_exact": geometry.vector_sha256_float64_le(
            cw11_total, np
        )
        == EXPECTED_CW11_TOTAL_FLOAT64_LE_SHA256,
        "incoming_CW11_rows_exact_33": int(context["expanded_row_count"])
        == EXPECTED_CW11_SELECTED_ROWS,
        "incoming_CW11_pairs_exact_34": len(context["active_pair_ledger"])
        == EXPECTED_CW11_ACTIVE_PAIRS,
        "incoming_CW11_ledger_exact": sha256_bytes(
            canonical_json(context["active_pair_ledger"])
        )
        == EXPECTED_CW11_ACTIVE_LEDGER_SHA256,
        "additional_plus_CW11_exact_total": computed_total.astype(
            "<f8", copy=False
        ).tobytes()
        == total.astype("<f8", copy=False).tobytes(),
        "payload_actor_layout_exact": layout == decoded["payload"]["actor_layout"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW11-to-CW12 replay precondition drift: {checks}")
    cutting.restore_raw_actor(ram, parameters, context["raw_actor"], torch)
    cutting.apply_cumulative_from_raw(
        ram,
        parameters,
        context["raw_actor"],
        total,
        torch,
    )
    observed_model_sha = helper.model_state_sha256(model.state_dict())
    assert EXPECTED_CW12_MODEL_STATE_SHA256 is not None
    if observed_model_sha != EXPECTED_CW12_MODEL_STATE_SHA256:
        raise RuntimeError("CW12 mathematical replay model SHA mismatch")
    replayed = dict(context)
    replayed.update(
        {
            "terminal_cumulative_float64": total.copy(),
            "terminal_cumulative_float64_le_sha256": (
                EXPECTED_CW12_TOTAL_FLOAT64_LE_SHA256
            ),
            "success_iteration": EXPECTED_CW12_TERMINAL_ITERATION,
            "active_pair_ledger": list(decoded["ledger"]),
            "expanded_row_count": EXPECTED_CW11_SELECTED_ROWS,
            "candidate_model_state_sha256": EXPECTED_CW12_MODEL_STATE_SHA256,
        }
    )
    return replayed, {
        "checks": checks,
        "model_state_sha256": observed_model_sha,
        "terminal_total_float64_le_sha256": (
            EXPECTED_CW12_TOTAL_FLOAT64_LE_SHA256
        ),
        "active_cut_ledger_sha256": decoded["ledger_sha256"],
        "consumed_specialist_data_reopened": False,
    }


def reconstruct_cw11_b33_once(
    cw11: ModuleType,
    primary: ModuleType,
    primary_source: bytes,
    primary_evidence: Mapping[str, Any],
    consumer: Any,
) -> dict[str, Any]:
    """The sole source-level CW11 B33 reconstruction call site."""
    return cw11.run_probe(
        primary,
        primary_source,
        primary_evidence,
        candidate_consumer=consumer,
    )


def run_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Future adapter entrypoint; unbound skeleton intentionally refuses."""
    del args, kwargs
    audit = terminal_binding_audit()
    if not audit["complete"]:
        raise RuntimeError(
            "CW12 formal run is not armed; terminal path/hash/model/vector/ledger "
            f"bindings remain fail-closed: {audit['unbound_fields']}"
        )
    raise RuntimeError(
        "CW12 formal skeleton is static-only; bind the terminal and complete the "
        "audited v5 compatibility adapter before enabling a run"
    )


def static_result() -> dict[str, Any]:
    runtime = validate_runtime()
    source, self_evidence = read_regular_bytes(
        SCRIPT,
        None,
        "CW12 formal skeleton",
        expected_mode=FROZEN_MODE,
    )
    source_audit = static_source_audit(source)
    cw11_formal, cw11_formal_evidence = import_frozen(
        CW11_FORMAL, CW11_FORMAL_SHA256, "frozen CW11 formal gate"
    )
    cw11, cw11_evidence = import_frozen(
        CW11_PROBE, CW11_PROBE_SHA256, "frozen CW11 B33 probe"
    )
    v5, v5_evidence = import_frozen(
        FULLTRAIN_V5, FULLTRAIN_V5_SHA256, "frozen fulltrain v5 adapter"
    )
    v4, v4_evidence = import_frozen(
        FULLTRAIN_V4, FULLTRAIN_V4_SHA256, "frozen fulltrain v4 implementation"
    )
    base, base_evidence = import_frozen(
        FULLTRAIN_V3, FULLTRAIN_V3_SHA256, "frozen fulltrain v3 utilities"
    )
    contract = frozen_formal_contract_audit(cw11_formal, cw11, v5, v4, base)
    return {
        "schema_version": SCHEMA,
        "status": "static_only_fail_closed_terminal_placeholders",
        "self": self_evidence,
        "dependencies": {
            "cw11_formal": cw11_formal_evidence,
            "cw11_B33_probe": cw11_evidence,
            "fulltrain_v5_adapter": v5_evidence,
            "fulltrain_v4_implementation": v4_evidence,
            "fulltrain_v3_utilities": base_evidence,
        },
        "runtime": runtime,
        "source_audit": source_audit,
        "terminal_binding_audit": terminal_binding_audit(),
        "frozen_formal_contract": contract,
        "planned_run_contract": {
            "CW11_B33_reconstructions": 1,
            "official6_solver_calls": 0,
            "models": ["raw", "candidate"],
            "panels": list(EXPECTED_PANEL_ORDER),
            "evaluation_count": EXPECTED_EVALUATIONS,
            "panel_rows": dict(EXPECTED_ROWS),
            "total_unique_train_rows": EXPECTED_TOTAL_ROWS,
            "batch_size": EXPECTED_BATCH_SIZE,
            "pokemonfan_minimum_wc_and_net": dict(EXPECTED_PF_MINIMUM_WC),
            "all_three_panels_six_policy_metrics_zero_cw": True,
            "count_value_logits_fingerprints_exact_raw": True,
            "consumed_specialist_views_reopened": False,
            "candidate_RAM_only": True,
            "training_optimizer_backward": False,
            "model_or_result_writes": 0,
            "network_upload_submission": False,
            "promotion_evidence": False,
        },
        "run_executed": False,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="static")
    args = parser.parse_args()
    if args.mode == "static":
        result = static_result()
    else:
        validate_runtime()
        result = run_gate()
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-shot train-only gate for the guarded U468 exact-P12 endpoint.

The frozen formal-v3 evaluator supplies the identity-strict train loader,
official CUDA BF16 forward path, per-row transition books and fingerprints.
This adapter replaces its historical actor6 sweep with exactly two complete
checkpoint states: guarded U468 and its exact-P12 child.  The only live model
overlay is the preregistered mutable10 scope (actor6 plus count-head4).

Audit mode is read-only.  Formal mode consumes one O_EXCL attempt marker and
publishes one O_EXCL JSON result.  It never trains, opens validation/broad/Gold
data, materializes a model, uses the network, packages, uploads, or submits.
"""

from __future__ import annotations

import argparse
import ast
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
from typing import Any, Mapping


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_u456_g8_guarded3x4_u468_exactp12_fulltrain24050_gate_v1.py"
SCHEMA = "ptcg-u456-g8-guarded3x4-u468-exactp12-fulltrain24050-gate-v1"
PREREG_SCHEMA = f"{SCHEMA}-execution-preregistration-v1"

MASTER = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141.master_preregistration.json"
MASTER_SHA256 = "e5dd52ba5e28f5672ea0199ba3b55c4da3fd8a6f45ebd6062b77fca41a5d37c7"
P12_DECISION = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141.p12_training_integrity_decision.json"
P12_DECISION_SHA256 = "c494937e56ee08b3996efde9717e09d9370989dcfeb5ffb4a2e7943aba36d658"
P12_DESIGN = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141.p12_design_preregistration.json"
P12_DESIGN_SHA256 = "6b6da1d3b2896e09bcf0a7915e981cbeaaf36cf0a70be7b7b75193484f60bb1f"
PARENT = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141/ppo_stage/block3/B_gold_league/seed-202608141/checkpoints/update-0468.pt"
PARENT_FILE_SHA256 = "a9290745b8eb58704c4617594ac5ab49500c6e65ecd9f273a8285c12547fbb83"
PARENT_MODEL_SHA256 = "832c724277d6c2263d76ddbf41622f4ba06621ec3155fa8b7ae2591203db1c5e"
CANDIDATE = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141/special_stage/special-bc-actorheadonly-pokemonfan-exactp12-0012.pt"
CANDIDATE_FILE_SHA256 = "6eb065efd36dd5c258d07a202ca6bd034b145466764175ad2f5bfbd4fff8e151"
CANDIDATE_MODEL_SHA256 = "fe34a72e91e42b71daab884ebbf188eb492c6a8e578b38d1d266743e0e2ab754"

FORMAL = TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v3.py"
FORMAL_SHA256 = "2d12ce9e76f6672911948dcc6458573899ec45fc39e748510544001025ede939"
REFERENCE_GATE = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v3.py"
REFERENCE_GATE_SHA256 = "f32c077c0d577bcc7c0ad2e42bdfda1ec1641244004b29efd5c82008cfe92338"

PREREGISTRATION = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141.fulltrain24050_execution_preregistration.json"
ATTEMPT_MARKER = ROOT / ".ptcg-u456-g8-guarded3x4-u468-exactp12-fulltrain24050-attempt-v1.json"
RESULT = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141.fulltrain24050_result.json"

PANEL_ORDER = ("flg", "pokemonfan", "core5")
EXPECTED_ROWS = {"flg": 9443, "pokemonfan": 9487, "core5": 5120}
EXPECTED_CONTEXT34_ROWS = {"flg": 42, "pokemonfan": 38, "core5": 20}
EXPECTED_TOTAL_ROWS = 24050
DATA_SHA256 = {
    "flg": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    "pokemonfan": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}
POLICY_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
PF_MINIMUM_NET = {
    "set_exact": 3,
    "hybrid_order_exact": 3,
    "ordered_exact": 5,
    "top1_correct": 3,
    "context34_hybrid_order_exact": 0,
    "context34_ordered_exact": 0,
    "count_correct": 0,
}
MUTABLE10 = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
    "count_head.0.weight",
    "count_head.0.bias",
    "count_head.2.weight",
    "count_head.2.bias",
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
    visible = os.lstat(path)
    payload = b"".join(chunks)
    identity = (after.st_dev, after.st_ino, after.st_size)
    if (
        (before.st_dev, before.st_ino, before.st_size) != identity
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or visible.st_nlink != 1
        or (visible.st_dev, visible.st_ino, visible.st_size) != identity
        or len(payload) != after.st_size
    ):
        raise RuntimeError(f"{label} changed during held-fd read")
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
        "mode_octal": format(mode, "04o"),
        "single_link_regular_held_fd_identity_exact": True,
    }


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    if path.parent != ROOT and not path.parent.exists():
        raise RuntimeError(f"publication parent must already exist: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        observed = os.fstat(fd)
    finally:
        os.close(fd)
    reloaded, evidence = read_regular_bytes(
        path, sha256_bytes(payload), str(path), expected_mode=0o444
    )
    if reloaded != payload or observed.st_size != len(payload):
        raise RuntimeError(f"O_EXCL publication verification failed: {path}")
    return evidence


def import_frozen(path: Path, digest: str, name: str) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(path, digest, name, expected_mode=0o555)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct frozen import {name}")
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


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    forbidden_imports = {"requests", "urllib", "http", "socket", "subprocess", "kaggle"}
    forbidden_calls = {"backward", "step", "save", "savez", "unlink", "rename", "replace", "rmtree"}
    import_hits: list[dict[str, Any]] = []
    call_hits: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_imports:
                    import_hits.append({"line": node.lineno, "name": name})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden_calls:
                call_hits.append({"line": node.lineno, "name": node.func.attr})
    checks = {
        "ast_parse": True,
        "no_network_or_submission_imports": not import_hits,
        "no_optimizer_backward_checkpoint_or_destructive_calls": not call_hits,
        "rows_sum_24050": sum(EXPECTED_ROWS.values()) == EXPECTED_TOTAL_ROWS,
        "mutable_scope_exact_10": len(MUTABLE10) == 10 and len(set(MUTABLE10)) == 10,
        "positive_pf_thresholds_historical": {
            key: PF_MINIMUM_NET[key]
            for key in ("set_exact", "hybrid_order_exact", "ordered_exact", "top1_correct")
        }
        == {"set_exact": 3, "hybrid_order_exact": 3, "ordered_exact": 5, "top1_correct": 3},
    }
    if not all(checks.values()):
        raise RuntimeError(f"static audit failed: {checks}")
    return {"checks": checks, "import_hits": import_hits, "call_hits": call_hits}


def output_absence() -> dict[str, bool]:
    return {
        "preregistration_present": PREREGISTRATION.is_file() and not PREREGISTRATION.is_symlink(),
        "attempt_marker_absent": not (ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink()),
        "result_absent": not (RESULT.exists() or RESULT.is_symlink()),
    }


def expected_gate_contract() -> dict[str, Any]:
    return {
        "data": {
            "split": "train_only",
            "panel_order": list(PANEL_ORDER),
            "rows": dict(EXPECTED_ROWS),
            "context34_rows": dict(EXPECTED_CONTEXT34_ROWS),
            "total_unique_rows": EXPECTED_TOTAL_ROWS,
            "validation_test_broad_gold_opened": False,
        },
        "evaluation": {
            "model_order_within_every_batch": ["parent", "candidate"],
            "same_model_instance": True,
            "batch_size": 256,
            "workers": 0,
            "device": "cuda:0",
            "native_output_dtype": "torch.bfloat16",
            "deterministic_algorithms": True,
            "all_six_logical_evaluations_before_decision": True,
        },
        "checkpoint_scope": {
            "changed_exactly": list(MUTABLE10),
            "all_other_tensors_byte_equal_parent": True,
            "both_states_finite": True,
        },
        "policy_and_count_gate": {
            "pokemonfan_minimum_net": dict(PF_MINIMUM_NET),
            "pokemonfan_positive_threshold_metrics_also_require_wc_at_least_threshold": True,
            "flg_minimum_net_all_seven_metrics": 0,
            "core5_minimum_net_all_seven_metrics": 0,
            "policy_and_count_cw_reported_but_not_zero_cw_hard_gate": True,
            "cross_panel_offset_forbidden": True,
        },
        "value_gate": {
            "native_logits_bit_exact_every_row": True,
            "value_logits_value_sign_value_correct_fingerprints_exact": True,
            "value_correct_cw_and_wc_zero": True,
            "aggregate_value_correct_exact": True,
        },
        "decision": {
            "pass": "GO_SPECIALIST",
            "fail": "FULLTRAIN_GATE_FAIL_CLOSE_P12_LINEAGE",
            "threshold_adjustment_retry_or_endpoint_selection": False,
            "official_unique_changed_candidate_count_consumed": 0,
        },
    }


def expected_bindings(self_sha256: str) -> dict[str, Any]:
    return {
        "runner": {
            "path": str(SCRIPT.relative_to(ROOT)),
            "sha256": self_sha256,
            "mode_octal": "0555",
        },
        "master": {"path": str(MASTER.relative_to(ROOT)), "sha256": MASTER_SHA256},
        "p12_integrity_decision": {
            "path": str(P12_DECISION.relative_to(ROOT)),
            "sha256": P12_DECISION_SHA256,
            "status": "GO_FULLTRAIN",
        },
        "p12_design": {"path": str(P12_DESIGN.relative_to(ROOT)), "sha256": P12_DESIGN_SHA256},
        "parent": {
            "path": str(PARENT.relative_to(ROOT)),
            "sha256": PARENT_FILE_SHA256,
            "runtime_model_state_sha256": PARENT_MODEL_SHA256,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": CANDIDATE_FILE_SHA256,
            "runtime_model_state_sha256": CANDIDATE_MODEL_SHA256,
        },
        "formal_v3": {"path": str(FORMAL.relative_to(ROOT)), "sha256": FORMAL_SHA256, "mode_octal": "0555"},
        "historical_gate_reference": {
            "path": str(REFERENCE_GATE.relative_to(ROOT)),
            "sha256": REFERENCE_GATE_SHA256,
            "mode_octal": "0555",
        },
        "train_archives": {
            panel: {"path": None, "sha256": DATA_SHA256[panel]} for panel in PANEL_ORDER
        },
    }


def tensor_equal(left: Any, right: Any, torch: Any) -> bool:
    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    )


def load_and_audit_checkpoints(
    helper: ModuleType,
    parent_payload: bytes,
    candidate_payload: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent = helper.checkpoint_from_bytes(parent_payload, "guarded U468 parent")
    candidate = helper.checkpoint_from_bytes(candidate_payload, "guarded exact-P12 candidate")
    if helper.evaluator.checkpoint_kind(parent) != "ppo" or helper.evaluator.checkpoint_kind(candidate) != "ppo":
        raise RuntimeError("both fulltrain arms must be PPO checkpoints")
    if parent.get("model_config") != candidate.get("model_config"):
        raise RuntimeError("parent/candidate inference configuration drift")
    parent_state = parent["model_state_dict"]
    candidate_state = candidate["model_state_dict"]
    if set(parent_state) != set(candidate_state):
        raise RuntimeError("parent/candidate model key drift")
    parent_hash = helper.model_state_sha256(parent_state)
    candidate_hash = helper.model_state_sha256(candidate_state)
    if parent_hash != PARENT_MODEL_SHA256 or candidate_hash != CANDIDATE_MODEL_SHA256:
        raise RuntimeError(f"checkpoint model identity drift: {parent_hash=} {candidate_hash=}")
    changed = sorted(
        name for name in parent_state if not tensor_equal(parent_state[name], candidate_state[name], helper.torch)
    )
    if changed != sorted(MUTABLE10):
        raise RuntimeError(f"candidate changed scope drift: {changed}")
    if not formal_all_finite(parent_state, helper.torch) or not formal_all_finite(candidate_state, helper.torch):
        raise RuntimeError("non-finite checkpoint state")
    scope = {
        "parent_model_state_sha256": parent_hash,
        "candidate_model_state_sha256": candidate_hash,
        "state_key_count": len(parent_state),
        "changed_parameter_count": len(changed),
        "changed_parameter_names": changed,
        "changed_exactly_mutable10": True,
        "all_other_tensors_byte_equal_parent": True,
        "both_states_all_finite": True,
    }
    return parent, candidate, scope


def formal_all_finite(state: Mapping[str, Any], torch: Any) -> bool:
    return all(
        isinstance(value, torch.Tensor)
        and (not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all()))
        for value in state.values()
    )


def load_inputs(formal: ModuleType) -> dict[str, Any]:
    source, self_evidence = read_regular_bytes(
        SCRIPT, None, "fulltrain wrapper", expected_mode=0o555
    )
    master, master_evidence = read_regular_bytes(MASTER, MASTER_SHA256, "master", expected_mode=0o444)
    decision, decision_evidence = read_regular_bytes(P12_DECISION, P12_DECISION_SHA256, "P12 integrity decision", expected_mode=0o444)
    _, p12_design_evidence = read_regular_bytes(P12_DESIGN, P12_DESIGN_SHA256, "P12 design", expected_mode=0o444)
    parent_payload, parent_evidence = read_regular_bytes(PARENT, PARENT_FILE_SHA256, "guarded U468 parent", expected_mode=0o444)
    candidate_payload, candidate_evidence = read_regular_bytes(CANDIDATE, CANDIDATE_FILE_SHA256, "guarded P12 candidate", expected_mode=0o444)
    _, reference_evidence = read_regular_bytes(REFERENCE_GATE, REFERENCE_GATE_SHA256, "historical fulltrain reference", expected_mode=0o555)
    master_json = json.loads(master)
    decision_json = json.loads(decision)
    if master_json.get("downstream_order_if_P12_passes", [None])[0] != "fulltrain_24050_rows":
        raise RuntimeError("master downstream order drift")
    if decision_json.get("status") != "GO_FULLTRAIN" or not decision_json.get("decision", {}).get("fulltrain_execution_preregistration_authorized"):
        raise RuntimeError("P12 decision does not authorize fulltrain preregistration")
    design = formal.load_module(
        formal.DESIGN_TOOL,
        formal.DESIGN_TOOL_SHA256,
        "guarded_fulltrain_data_contract",
        0o555,
    )
    if tuple(design.DATASETS) != PANEL_ORDER:
        raise RuntimeError("fulltrain panel order drift")
    archives: dict[str, bytes] = {}
    archive_evidence: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        if int(design.EXPECTED_TRAIN[panel]["rows"]) != EXPECTED_ROWS[panel]:
            raise RuntimeError(f"{panel} expected row drift")
        if int(design.EXPECTED_TRAIN[panel]["context34_rows"]) != EXPECTED_CONTEXT34_ROWS[panel]:
            raise RuntimeError(f"{panel} context34 row drift")
        payload, evidence = read_regular_bytes(
            design.DATASETS[panel], DATA_SHA256[panel], f"{panel} train archive"
        )
        archives[panel] = payload
        archive_evidence[panel] = evidence
    bindings = expected_bindings(self_evidence["sha256"])
    for panel in PANEL_ORDER:
        bindings["train_archives"][panel]["path"] = str(design.DATASETS[panel].relative_to(ROOT))
    return {
        "source": source,
        "self": self_evidence,
        "master": master_evidence,
        "decision": decision_evidence,
        "p12_design": p12_design_evidence,
        "parent_payload": parent_payload,
        "parent": parent_evidence,
        "candidate_payload": candidate_payload,
        "candidate": candidate_evidence,
        "reference": reference_evidence,
        "design": design,
        "archives": archives,
        "archive_evidence": archive_evidence,
        "expected_bindings": bindings,
    }


def verify_preregistration(
    expected_sha256: str,
    self_sha256: str,
    bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, evidence = read_regular_bytes(
        PREREGISTRATION,
        expected_sha256,
        "fulltrain execution preregistration",
        expected_mode=0o444,
    )
    value = json.loads(payload)
    expected_template = [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        str(SCRIPT),
        "--mode",
        "formal",
        "--expected-tool-sha256",
        self_sha256,
        "--expected-preregistration-sha256",
        "<LOCKED_PREREGISTRATION_SHA256>",
    ]
    expected_template_sha256 = sha256_bytes(canonical_json(expected_template))
    checks = {
        "schema": value.get("schema_version") == PREREG_SCHEMA,
        "status": value.get("status") == "locked_before_only_formal_attempt",
        "all_bindings_exact": value.get("bindings") == dict(bindings),
        "gate_contract_exact": value.get("gate_contract") == expected_gate_contract(),
        "result_path": value.get("output_contract", {}).get("result") == str(RESULT.relative_to(ROOT)),
        "marker_path": value.get("output_contract", {}).get("attempt_marker") == str(ATTEMPT_MARKER.relative_to(ROOT)),
        "attempts": value.get("output_contract", {}).get("formal_attempts_authorized") == 1,
        "targets_absent_at_lock": value.get("output_contract", {}).get("targets_absent_at_lock") is True,
        "retry_forbidden": value.get("output_contract", {}).get("retry_authorized") is False,
        "formal_command_template": value.get("formal_command_template") == expected_template,
        "formal_command_template_sha256": value.get("formal_command_template_sha256") == expected_template_sha256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"preregistration binding drift: {checks}")
    return value, {**evidence, "checks": checks}


def same_run_parent_summary_hook(
    summary: Mapping[str, Any],
    frozen: Mapping[str, Any],
    panel: str,
    model_name: str,
    formal: ModuleType,
) -> dict[str, Any]:
    del frozen
    leaves = formal.typed_leaf_map(summary)
    if model_name != "raw" or panel not in PANEL_ORDER or len(leaves) != 572:
        raise RuntimeError("same-run parent summary structure drift")
    return {
        "authority": "same_run_guarded_parent",
        "old_raw_u468_frozen_summary_not_used": True,
        "typed_scalar_leaf_count": len(leaves),
        "canonical_sha256": sha256_bytes(canonical_json(summary)),
    }


def evaluate_two_states(
    formal: ModuleType,
    helper: ModuleType,
    design: ModuleType,
    parent: dict[str, Any],
    candidate: dict[str, Any],
    archives: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch = helper.torch
    states = {
        "raw": {name: value.detach().cpu().clone() for name, value in parent["model_state_dict"].items()},
        "candidate": {name: value.detach().cpu().clone() for name, value in candidate["model_state_dict"].items()},
    }
    original_order = formal.MODEL_ORDER
    original_alphas = formal.ALPHAS
    original_copy = formal.copy_actor_state_to_model
    original_verify = formal.verify_live_actor_state
    original_summary = formal.assert_frozen_summary_exact
    original_parent_hash = design.PARENT_MODEL_STATE_SHA256
    live: dict[str, Any] = {}
    restore = {"live_model_seen": False, "finally_parent_restore_attempted": False, "finally_parent_restore_pass": False}

    def copy_mutable10(model: Any, state: Mapping[str, Any], module: ModuleType) -> None:
        del module
        live["model"] = model
        current = model.state_dict()
        if set(current) != set(state):
            raise RuntimeError("live/state key drift during mutable10 overlay")
        with torch.no_grad():
            for name in MUTABLE10:
                target = current[name]
                source = state[name].to(device=target.device, dtype=target.dtype)
                target.copy_(source)

    def verify_full_live_state(model: Any, state: Mapping[str, Any], module: ModuleType) -> None:
        observed = module.model_state_sha256(model.state_dict())
        expected = module.model_state_sha256(state)
        if observed != expected:
            raise RuntimeError(f"full live mutable10 overlay hash mismatch: {observed} != {expected}")

    def summary_hook(summary: Mapping[str, Any], frozen: Mapping[str, Any], panel: str, model_name: str) -> dict[str, Any]:
        return same_run_parent_summary_hook(summary, frozen, panel, model_name, formal)

    formal.MODEL_ORDER = ("raw", "candidate")
    formal.ALPHAS = {"raw": 0, "candidate": 1}
    formal.copy_actor_state_to_model = copy_mutable10
    formal.verify_live_actor_state = verify_full_live_state
    formal.assert_frozen_summary_exact = summary_hook
    design.PARENT_MODEL_STATE_SHA256 = PARENT_MODEL_SHA256
    try:
        panels, advisory, execution = formal.evaluate_all_states(
            helper,
            design,
            states,
            parent,
            archives,
            {"raw": {panel: {} for panel in PANEL_ORDER}},
        )
    finally:
        try:
            model = live.get("model")
            if model is not None:
                restore["live_model_seen"] = True
                restore["finally_parent_restore_attempted"] = True
                copy_mutable10(model, states["raw"], helper)
                verify_full_live_state(model, states["raw"], helper)
                restore["final_parent_model_state_sha256"] = helper.model_state_sha256(model.state_dict())
                restore["finally_parent_restore_pass"] = restore["final_parent_model_state_sha256"] == PARENT_MODEL_SHA256
        finally:
            formal.MODEL_ORDER = original_order
            formal.ALPHAS = original_alphas
            formal.copy_actor_state_to_model = original_copy
            formal.verify_live_actor_state = original_verify
            formal.assert_frozen_summary_exact = original_summary
            design.PARENT_MODEL_STATE_SHA256 = original_parent_hash
    if not restore["finally_parent_restore_pass"]:
        raise RuntimeError("fulltrain evaluator final parent restoration failed")
    execution["mutable10_copy_operations"] = execution.pop("actor6_copy_operations")
    execution["all_two_live_mutable10_overlays_verified_after_copy"] = execution.pop(
        "all_ten_live_actor_overlays_verified_after_copy"
    )
    execution["same_batch_parent_then_candidate"] = execution.pop("same_batch_all_ten_states")
    execution["all_six_completed_before_decision"] = (
        int(execution["evaluation_count_exact"]) == 6
    )
    execution.pop("all_30_completed_before_decision", None)
    return panels, execution, {"advisory": advisory, "restore": restore}


def transition_net(result: Mapping[str, Any], metric: str) -> tuple[int, int, int]:
    if metric == "count_correct":
        item = result["count_value_integrity"]["correctness_transitions"][metric]
    else:
        item = result["raw_transition_evidence"][metric]
    wc = int(item["cells"]["wc"]["count"])
    cw = int(item["cells"]["cw"]["count"])
    net = int(item["candidate_correct_minus_raw_correct"])
    if net != wc - cw:
        raise RuntimeError(f"transition net identity drift for {metric}")
    return wc, cw, net


def decide(panel_results: Mapping[str, Any]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        raw = panel_results["raw"][panel]
        candidate = panel_results["candidate"][panel]
        observed: dict[str, Any] = {}
        metric_checks: dict[str, bool] = {}
        thresholds = PF_MINIMUM_NET if panel == "pokemonfan" else {metric: 0 for metric in (*POLICY_METRICS, "count_correct")}
        for metric in (*POLICY_METRICS, "count_correct"):
            wc, cw, net = transition_net(candidate, metric)
            minimum = int(thresholds[metric])
            wc_minimum = minimum if panel == "pokemonfan" and minimum > 0 else 0
            passed = net >= minimum and wc >= wc_minimum
            observed[metric] = {"wc": wc, "cw": cw, "net": net, "minimum_net": minimum, "minimum_wc": wc_minimum, "pass": passed}
            metric_checks[metric] = passed
        value_integrity = candidate["count_value_integrity"]
        value_transition = value_integrity["correctness_transitions"]["value_correct"]
        value_fingerprints = value_integrity["fingerprint_exact_raw"]
        value_checks = {
            "native_value_logits_bit_exact_every_row": int(value_integrity["native_value_logits_mismatch_count"]) == 0,
            "value_logits_fingerprint_exact": bool(value_fingerprints["value_logits_float32_le"]),
            "value_sign_fingerprint_exact": bool(value_fingerprints["value_sign"]),
            "value_correct_fingerprint_exact": bool(value_fingerprints["value_correct"]),
            "value_correct_no_cw": int(value_transition["cells"]["cw"]["count"]) == 0,
            "value_correct_no_wc": int(value_transition["cells"]["wc"]["count"]) == 0,
            "aggregate_value_correct_exact": int(raw["official_metrics"]["value_correct"]) == int(candidate["official_metrics"]["value_correct"]),
        }
        shape_checks = {
            "parent_rows_exact": int(raw["rows"]) == EXPECTED_ROWS[panel],
            "candidate_rows_exact": int(candidate["rows"]) == EXPECTED_ROWS[panel],
            "parent_typed_leaves_572": int(raw["official_metrics_typed_scalar_leaf_count"]) == 572,
            "candidate_typed_leaves_572": int(candidate["official_metrics_typed_scalar_leaf_count"]) == 572,
        }
        passed = all(metric_checks.values()) and all(value_checks.values()) and all(shape_checks.values())
        gates[panel] = {
            "threshold_source": "historical_exact_P12_contract_locked_before_guarded_results",
            "policy_and_count_transitions": observed,
            "value_integrity": value_checks,
            "shape_integrity": shape_checks,
            "policy_and_count_CW_reported_but_not_zero_CW_hard_gate": True,
            "pass": passed,
        }
    passed = all(gates[panel]["pass"] for panel in PANEL_ORDER)
    return {
        "status": "GO_SPECIALIST" if passed else "FULLTRAIN_GATE_FAIL_CLOSE_P12_LINEAGE",
        "panel_gates": gates,
        "all_three_panels_pass": passed,
        "specialist_execution_preregistration_authorized": passed,
        "specialist_execution_authorized_now": False,
        "official_unique_changed_candidate_count_consumed": 0,
        "broad_gold_package_upload_submission_authorized": False,
    }


def run_formal(formal: ModuleType, inputs: Mapping[str, Any], prereg_sha256: str) -> dict[str, Any]:
    prereg, prereg_evidence = verify_preregistration(
        prereg_sha256,
        inputs["self"]["sha256"],
        inputs["expected_bindings"],
    )
    absence = output_absence()
    if not all((absence["preregistration_present"], absence["attempt_marker_absent"], absence["result_absent"])):
        raise RuntimeError(f"formal output absence failed: {absence}")
    helper = formal.load_helper()
    cuda_runtime = formal.cuda_runtime(helper)
    marker_payload = canonical_json({
        "schema_version": f"{SCHEMA}-attempt-v1",
        "attempt": 1,
        "attempts_authorized": 1,
        "preregistration_sha256": prereg_sha256,
        "runner_sha256": inputs["self"]["sha256"],
        "parent_sha256": PARENT_FILE_SHA256,
        "candidate_sha256": CANDIDATE_FILE_SHA256,
        "retry_authorized": False,
    })
    marker_evidence = publish_o_excl(ATTEMPT_MARKER, marker_payload)
    random.seed(formal.SEED)
    helper.torch.manual_seed(formal.SEED)
    helper.torch.cuda.manual_seed_all(formal.SEED)
    parent, candidate, scope = load_and_audit_checkpoints(
        helper, inputs["parent_payload"], inputs["candidate_payload"]
    )
    panels, execution, runtime_integrity = evaluate_two_states(
        formal, helper, inputs["design"], parent, candidate, inputs["archives"]
    )
    execution_checks = {
        "device_cuda0": execution["device"] == "cuda:0",
        "model_kind_ppo": execution["model_kind"] == "ppo",
        "one_model_instance": int(execution["model_instance_count"]) == 1,
        "workers_zero": int(execution["workers"]) == 0,
        "batch_size_256": int(execution["batch_size"]) == 256,
        "evaluation_count_6": int(execution["evaluation_count_exact"]) == 6,
        "all_panel_rows_exact": all(int(panels[model][panel]["rows"]) == EXPECTED_ROWS[panel] for model in ("raw", "candidate") for panel in PANEL_ORDER),
        "all_train_rows_24050": sum(EXPECTED_ROWS.values()) == EXPECTED_TOTAL_ROWS,
        "validation_members_not_opened": not bool(execution["validation_member_payloads_opened"]),
        "finally_parent_restore": bool(runtime_integrity["restore"]["finally_parent_restore_pass"]),
    }
    if not all(execution_checks.values()):
        raise RuntimeError(f"formal execution integrity failed: {execution_checks}")
    decision = decide(panels)
    result = {
        "schema_version": SCHEMA,
        "status": decision["status"],
        "preregistration": prereg_evidence,
        "attempt_marker": marker_evidence,
        "input_lock": {
            "runner": inputs["self"],
            "master": inputs["master"],
            "p12_integrity_decision": inputs["decision"],
            "p12_design": inputs["p12_design"],
            "parent": inputs["parent"],
            "candidate": inputs["candidate"],
            "formal_v3": {"path": str(FORMAL.relative_to(ROOT)), "sha256": FORMAL_SHA256},
            "historical_gate_reference": inputs["reference"],
            "train_archives": inputs["archive_evidence"],
        },
        "checkpoint_scope_audit": scope,
        "execution": {**execution, "checks": execution_checks},
        "runtime_integrity": runtime_integrity,
        "cuda_runtime": cuda_runtime,
        "evaluations": panels,
        "decision": decision,
        "scope_audit": {
            "all_24050_train_rows": True,
            "validation_test_broad_gold_opened": False,
            "training_optimizer_backward": False,
            "model_artifact_writes": 0,
            "evidence_result_writes": 1,
            "candidate_endpoint_selection_or_sweep": False,
            "network_package_upload_submission": False,
        },
        "preregistered_contract": prereg.get("gate_contract"),
    }
    result_payload = canonical_json(result)
    result_evidence = publish_o_excl(RESULT, result_payload)
    return {"status": result["status"], "result": result_evidence, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "formal"), default="audit")
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--expected-preregistration-sha256")
    args = parser.parse_args()
    validate_runtime()
    formal, formal_evidence = import_frozen(FORMAL, FORMAL_SHA256, "guarded_fulltrain_formal_v3")
    inputs = load_inputs(formal)
    if args.expected_tool_sha256 != inputs["self"]["sha256"]:
        raise RuntimeError(
            f"runner SHA-256 mismatch: {inputs['self']['sha256']}"
        )
    static = static_audit(inputs["source"])
    if args.mode == "audit":
        helper = formal.load_helper()
        cuda_runtime = formal.cuda_runtime(helper)
        _, _, scope = load_and_audit_checkpoints(
            helper, inputs["parent_payload"], inputs["candidate_payload"]
        )
        print(canonical_json({
            "schema_version": f"{SCHEMA}-audit-v1",
            "status": "AUDIT_PASS",
            "runner": inputs["self"],
            "formal_v3": formal_evidence,
            "static": static,
            "checkpoint_scope_audit": scope,
            "cuda_runtime": cuda_runtime,
            "output_absence": output_absence(),
            "optimizer_steps": 0,
            "evaluation_rows": 0,
            "checkpoint_writes": 0,
            "network_package_upload_submission": False,
        }).decode("utf-8"), end="")
        return
    if not args.expected_preregistration_sha256:
        raise RuntimeError("formal mode requires --expected-preregistration-sha256")
    expected_argv = [
        str(SCRIPT),
        "--mode",
        "formal",
        "--expected-tool-sha256",
        inputs["self"]["sha256"],
        "--expected-preregistration-sha256",
        args.expected_preregistration_sha256,
    ]
    observed_argv = [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    if observed_argv != expected_argv:
        raise RuntimeError(
            f"formal argv/order drift: {observed_argv} != {expected_argv}"
        )
    receipt = run_formal(formal, inputs, args.expected_preregistration_sha256)
    print(canonical_json(receipt).decode("utf-8"), end="")


if __name__ == "__main__":
    main()

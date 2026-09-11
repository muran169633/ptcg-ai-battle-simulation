#!/usr/bin/env python3
"""Projected nonlinear actor6 special-BC probe for exact CW11 and frozen B352.

This is an unfrozen, audit-first CW24 v12 draft.  Production reuses the
authenticated historical callback only to reconstruct exact CW11 in RAM, then
runs at most 48 plain-SGD actor6 updates on the frozen train-only B352 panel.
Every update is projected back to a strict ball around CW11 and evaluated on
native BF16 logits.  Dynamic false-obligation pairs are sticky training terms,
never terminal feasibility constraints.  The first endpoint satisfying all
native behavior, preservation, radius, immutable-head, and soft-improvement
gates is selected.  No checkpoint is written and NO_GO never exposes actor
bytes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import sys
import types
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_projected_nonlinear_specialbc_cw24_v12.py"
OUTPUT = ROOT / "artifacts/cw24_cw11_projected_nonlinear_specialbc_trainonly_v12.json"
ATTEMPT_MARKER = ROOT / (
    "artifacts/.ptcg-cw24-cw11-projected-nonlinear-specialbc-v12-attempt.json"
)
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-projected-nonlinear-specialbc-cw24-v12"
SEED = 202608052

V1 = TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py"
V1_SHA256 = "a3c96eed95a6b558c61303e2190eaa4f353970add4509e8770c3dfcc1621db39"
CW22 = TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py"
CW22_SHA256 = "f307f3f26355125f97c857f14b97a73d9a7a4f2ea39ca8b0a578e801055e82b4"
CW23 = TOOLS / "probe_u468_cw11_boundary_qp_specialbc_cw23_v1.py"
CW23_SHA256 = "4a2d209af71a7ea5d955bd373f1fbada936688fc5bc597615c41192dc360b4b8"
EQUALBLEND = TOOLS / "run_u468_raw_actor6_equalblend_sgd512_shadow.py"
EQUALBLEND_SHA256 = "e04f7b7579ef42d0c6f643db833779fb837ae86e3205d948b5564d08f8b30f8e"
RAWCUT = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v1.py"
RAWCUT_SHA256 = "9c5a7377d82b18e437ca89d064ce8be60554646573e3aca27f4e40f4989ab77c"
SELECTION = ROOT / "artifacts/cw24_top1_b352_selection_v1.json"
SELECTION_SHA256 = "1f72b26eccd436ca0d341f837ec42949f5823bf7f0be73461aaeab28166cc222"

SOURCE_MODE = 0o555
SELECTION_MODE = 0o444
LEARNING_RATE = 5.0e-5
MAX_GRAD_NORM = 0.5
MAX_STEPS = 48
PROJECT_RADIUS = 9.99975e-4
HARD_RADIUS = 9.9998e-4
MAX_DYNAMIC_PAIRS = 64
SOFT_IMPROVEMENT_FLOOR = 1.0e-8
PAIR_WEIGHT = 0.25
KL_WEIGHT = 0.012
ORDER_CONTEXT_WEIGHT = 8.0
PF0_FINAL_MARGIN = 1.0 / 512.0
PF7_FINAL_MARGIN = 0.0


class ProtocolError(RuntimeError):
    """Fail-closed CW24 v12 protocol error."""


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
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
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


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    calls: list[tuple[str, int]] = []
    imports: list[str] = []
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append((node.func.id, int(node.lineno)))
            elif isinstance(node.func, ast.Attribute):
                calls.append((node.func.attr, int(node.lineno)))
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    call_counts = Counter(name for name, _ in calls)
    checks = {
        "syntax_valid": True,
        "exact_one_SGD_constructor": call_counts["SGD"] == 1,
        "exact_one_optimizer_step_site": call_counts["step"] == 1,
        "exact_one_backward_site": call_counts["backward"] == 1,
        "exact_one_gradient_clip_site": call_counts["clip_grad_norm_"] == 1,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_network_import": not any(
            name.split(".")[0] in {"requests", "urllib", "httpx", "socket"}
            for name in imports
        ),
        "single_result_publication_site": source.count(
            b"publish_o_excl(" + b"OUTPUT"
        )
        == 1,
        "single_attempt_claim_site": source.count(
            b"publish_o_excl(" + b"ATTEMPT_MARKER"
        )
        == 1,
        "project_and_hard_radii_literal": constants.get("PROJECT_RADIUS")
        == 9.99975e-4
        and constants.get("HARD_RADIUS") == 9.9998e-4,
        "step_budget_literal": constants.get("MAX_STEPS") == 48,
        "dynamic_pair_cap_literal": constants.get("MAX_DYNAMIC_PAIRS") == 64,
        "no_active_pair_terminal_gate": (
            "all_active_pair_" + "thresholds"
        ).encode() not in source,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v12 source audit failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "call_counts": {
            name: call_counts[name]
            for name in ("SGD", "step", "backward", "clip_grad_norm_")
        },
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
    }


def self_evidence(require_frozen: bool) -> dict[str, Any]:
    before = SCRIPT.lstat()
    source = SCRIPT.read_bytes()
    after = SCRIPT.lstat()
    mode = stat.S_IMODE(after.st_mode)
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
        "mode_allowed": mode == 0o555 if require_frozen else mode in {0o644, 0o664, 0o555},
        "static_source_audit": source_audit()["pass"],
    }
    if not all(checks.values()):
        raise ProtocolError(f"v12 self evidence failed: {checks}")
    return {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "mode_octal": format(mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def dependency_evidence() -> dict[str, Any]:
    specifications = {
        "frozen_CW24_v1": (V1, V1_SHA256, SOURCE_MODE),
        "frozen_CW22": (CW22, CW22_SHA256, SOURCE_MODE),
        "frozen_CW23": (CW23, CW23_SHA256, SOURCE_MODE),
        "plain_SGD_reference": (EQUALBLEND, EQUALBLEND_SHA256, SOURCE_MODE),
        "dynamic_pair_reference": (RAWCUT, RAWCUT_SHA256, SOURCE_MODE),
        "frozen_B352_selection": (SELECTION, SELECTION_SHA256, SELECTION_MODE),
    }
    return {
        label: regular_source(path, digest, mode, label)[1]
        for label, (path, digest, mode) in specifications.items()
    }


def path_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    return False


def binding_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(record["path"]),
        "sha256": str(record["sha256"]),
        "bytes": int(record["bytes"]),
        "mode_octal": str(record["mode_octal"]),
    }


def validate_pre_cuda_runtime() -> dict[str, Any]:
    checks = {
        "repo_root": Path.cwd().resolve() == ROOT,
        "my_project_env": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1
        and sys.dont_write_bytecode is True,
        "pycache_prefix_dev_null": sys.pycache_prefix == "/dev/null",
        "torch_not_imported_before_claim": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        raise ProtocolError(f"pre-CUDA runtime drift: {checks}")
    return {
        "python": str(Path(sys.executable).resolve()),
        "checks": checks,
        "pass": True,
    }


def load_locked_module(
    path: Path, expected_sha: str, expected_mode: int, module_name: str
) -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = regular_source(path, expected_sha, expected_mode, module_name)
    if module_name in sys.modules:
        raise ProtocolError(f"module name already occupied: {module_name}")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, evidence


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
                raise ProtocolError("short v12 result write")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_fd = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
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
        raise ProtocolError(f"v12 result publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def claim_attempt() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v12 output already exists before attempt claim")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v12 one-shot attempt was already consumed")
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    payload = {
        "schema_version": "ptcg-u468-cw11-projected-nonlinear-specialbc-cw24-v12-attempt",
        "status": "claimed_before_module_exec_or_CUDA",
        "source": {
            **binding_summary(source),
        },
        "dependencies": {
            name: binding_summary(record)
            for name, record in dependencies.items()
        },
        "runtime": runtime,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent_by_lstat_at_claim": True,
        "CUDA_preflight_state": "not_imported_pending_production_validation",
        "single_seed": SEED,
        "maximum_steps": MAX_STEPS,
        "retry_authorized": False,
        "network_package_upload_submission": False,
    }
    publication = publish_o_excl(ATTEMPT_MARKER, canonical_json(payload))
    return {"payload": payload, "publication": publication}


def masked_batch(batch: Mapping[str, Any], mask: Any) -> dict[str, Any]:
    if mask.shape != batch["sample_weights"].shape:
        raise ProtocolError("objective mask shape drift")
    result = dict(batch)
    result["sample_weights"] = batch["sample_weights"] * mask.to(
        device=batch["sample_weights"].device,
        dtype=batch["sample_weights"].dtype,
    )
    return result


def complete_ordered_scalar(
    outputs: Mapping[str, Any], batch: Mapping[str, Any], ppo: ModuleType
) -> Any:
    scalar, _ = ppo.bc_expert_actor_loss(
        dict(outputs),
        dict(batch),
        loss_mode="ordered",
        order_context_weight=ORDER_CONTEXT_WEIGHT,
        non_context34_fixed_multi_action_order_weight=1.0,
    )
    if scalar.ndim != 0 or not bool(__import__("torch").isfinite(scalar)):
        raise ProtocolError("ordered BC scalar is nonfinite")
    return scalar


def top1_ce_per_row(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> Any:
    import torch

    logits = outputs["policy_logits"].float()
    option_mask = batch["option_mask"].bool()
    expert_first = batch["action_sequences"][:, 0]
    log_probs = torch.log_softmax(logits.masked_fill(~option_mask, -1e9), dim=1)
    return -log_probs.gather(1, expert_first.unsqueeze(1)).squeeze(1)


def mean_indices(values: Any, indices: Sequence[int], device: Any) -> Any:
    import torch

    if not indices:
        raise ProtocolError("empty objective index set")
    index = torch.tensor(indices, dtype=torch.long, device=device)
    return values[index].mean()


def float64_sha(value: Any) -> str:
    contiguous = value.astype("<f8", copy=False)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def pair_key(pair: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(pair["row_index"]),
        int(pair["positive_option"]),
        int(pair["negative_option"]),
    )


def native_pair_margin(outputs: Mapping[str, Any], pair: Mapping[str, Any]) -> float:
    logits = outputs["policy_logits"]
    row, positive, negative = pair_key(pair)
    return float((logits[row, positive].float() - logits[row, negative].float()).detach().cpu())


def local_positive_bf16_q(
    outputs: Mapping[str, Any], row: int, positive: int, negative: int
) -> float:
    import torch

    logits = outputs["policy_logits"]
    spacings = []
    for value in (logits[row, positive], logits[row, negative]):
        # Preserve the exact native BF16 cell value while avoiding dependence on
        # CUDA nextafter kernel availability during the production trajectory.
        detached = value.detach().cpu()
        spacings.extend(
            (
                torch.nextafter(detached, torch.full_like(detached, float("inf")))
                - detached,
                detached
                - torch.nextafter(detached, torch.full_like(detached, float("-inf"))),
            )
        )
    result = max(float(value.float().cpu()) for value in spacings)
    if not math.isfinite(result) or result <= 0.0:
        raise ProtocolError("invalid local native-BF16 positive quantum")
    return result


def first_different_pair(
    expert_order: Sequence[int], predicted_order: Sequence[int]
) -> tuple[int, int]:
    for expected, observed in zip(expert_order, predicted_order):
        if int(expected) != int(observed):
            return int(expected), int(observed)
    raise ProtocolError("false ordered obligation lacks a differing stage")


def evaluate_snapshot(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    ppo: ModuleType,
    cw22: ModuleType,
) -> dict[str, Any]:
    report, ordered_correct, predictions = cw22.evaluate_outputs(
        outputs, batch, rows, ppo
    )
    nll = cw22.ordered_nll_per_row(outputs, batch).detach().cpu()
    top1_ce = top1_ce_per_row(outputs, batch).detach().cpu()
    if len(predictions) != 352 or int(nll.shape[0]) != 352:
        raise ProtocolError("B352 snapshot row count drift")
    return {
        "report": report,
        "ordered_correct": ordered_correct,
        "predictions": [[int(value) for value in row] for row in predictions],
        "ordered_nll_cpu": nll,
        "top1_ce_cpu": top1_ce,
    }


def build_row_contract(
    rows: Sequence[Mapping[str, Any]],
    batch: Mapping[str, Any],
    baseline_outputs: Mapping[str, Any],
    baseline_snapshot: Mapping[str, Any],
    cw23: ModuleType,
    v1: ModuleType,
) -> dict[str, Any]:
    import torch

    row_by_sha = {str(row["line_sha256"]): index for index, row in enumerate(rows)}
    if len(row_by_sha) != 352:
        raise ProtocolError("B352 line identity duplication")
    hard = [index for index, row in enumerate(rows) if row["category"] == "hard"]
    retention = [
        index
        for index, row in enumerate(rows)
        if row["category"] not in {"hard", "top1_guard"}
    ]
    top1 = [
        index for index, row in enumerate(rows) if row["category"] == "top1_guard"
    ]
    baseline_top1 = {
        str(rows[index]["line_sha256"]): bool(baseline_snapshot["predictions"][index])
        and int(baseline_snapshot["predictions"][index][0])
        == int(rows[index]["expert_order"][0])
        for index in top1
    }
    baseline_checks = {
        "rows352": baseline_snapshot["report"]["rows"] == 352,
        "hard96": len(hard) == 96,
        "hard96_ordered_wrong": all(
            not baseline_snapshot["ordered_correct"][str(rows[index]["line_sha256"])]
            for index in hard
        ),
        "retention160": len(retention) == 160,
        "retention160_ordered_correct": all(
            baseline_snapshot["ordered_correct"][str(rows[index]["line_sha256"])]
            for index in retention
        ),
        "top1_guard96": len(top1) == 96,
        "top1_guard96_top1_correct": all(baseline_top1.values()),
        "native_policy_BF16": baseline_outputs["policy_logits"].dtype
        == torch.bfloat16,
    }
    if not all(baseline_checks.values()):
        raise ProtocolError(f"exact CW11 B352 baseline failed: {baseline_checks}")
    if not all(sha in row_by_sha for _, sha, _ in v1.PF_TARGETS):
        raise ProtocolError("three fixed PF identities missing")
    if not all(sha in row_by_sha for sha in v1.ZERO_MARGIN_SHA256):
        raise ProtocolError("six zero-margin identities missing")

    logits = baseline_outputs["policy_logits"].float()
    fixed_pairs: list[dict[str, Any]] = []
    pf_contract: dict[str, dict[str, Any]] = {}
    for name, sha, expected_baseline in v1.PF_TARGETS:
        index = row_by_sha[sha]
        margin, detail = cw23.fixed_threat_margin(
            logits[index], batch["option_mask"][index], rows[index]["expert_order"]
        )
        observed = float(margin.detach().cpu())
        if observed != float(expected_baseline):
            raise ProtocolError(f"{name}: exact CW11 fixed margin drift")
        threshold = PF7_FINAL_MARGIN if name == "pf7_boundary" else PF0_FINAL_MARGIN
        pair = {
            "row_index": index,
            "line_sha256": sha,
            "positive_option": int(detail["expert"]),
            "negative_option": int(detail["fixed_threat"]),
            "threshold": threshold,
            "threshold_source": "terminal_native_PF_requirement",
            "origins": [name],
            "fixed": True,
        }
        fixed_pairs.append(pair)
        pf_contract[name] = {
            **pair,
            "baseline_margin": observed,
            "terminal_margin": threshold,
        }

    zero_contract: list[dict[str, Any]] = []
    for offset, sha in enumerate(v1.ZERO_MARGIN_SHA256):
        index = row_by_sha[sha]
        margin, detail = cw23.fixed_threat_margin(
            logits[index], batch["option_mask"][index], rows[index]["expert_order"]
        )
        observed = float(margin.detach().cpu())
        if observed != 0.0:
            raise ProtocolError("zero-margin exact CW11 anchor drift")
        zero_contract.append(
            {
                "name": f"zero_margin_guard_{offset}",
                "row_index": index,
                "line_sha256": sha,
                "positive_option": int(detail["expert"]),
                "negative_option": int(detail["fixed_threat"]),
                "baseline_margin": observed,
            }
        )

    masks = {
        "union": torch.ones(352, dtype=torch.bool, device=batch["sample_weights"].device),
        "pf0": torch.tensor(
            [str(row["stratum"]) == "pf_ctx0_hard" for row in rows],
            dtype=torch.bool,
            device=batch["sample_weights"].device,
        ),
        "pf7": torch.tensor(
            [str(row["stratum"]) == "pf_ctx7_hard" for row in rows],
            dtype=torch.bool,
            device=batch["sample_weights"].device,
        ),
        "dominic": torch.tensor(
            [str(row["stratum"]) == "dominic_ctx0_hard" for row in rows],
            dtype=torch.bool,
            device=batch["sample_weights"].device,
        ),
        "protected": torch.tensor(
            [index in set(retention + top1) for index in range(352)],
            dtype=torch.bool,
            device=batch["sample_weights"].device,
        ),
    }
    mask_counts = {name: int(mask.sum().detach().cpu()) for name, mask in masks.items()}
    if mask_counts != {
        "union": 352,
        "pf0": 32,
        "pf7": 32,
        "dominic": 32,
        "protected": 256,
    }:
        raise ProtocolError(f"objective mask count drift: {mask_counts}")
    return {
        "row_by_sha": row_by_sha,
        "hard_indices": hard,
        "retention_indices": retention,
        "top1_indices": top1,
        "baseline_top1_correct": baseline_top1,
        "baseline_checks": baseline_checks,
        "fixed_pairs": fixed_pairs,
        "pf_contract": pf_contract,
        "zero_contract": zero_contract,
        "masks": masks,
        "mask_counts": mask_counts,
    }


def objective_components(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    teacher_policy_logits: Any,
    fixed_pairs: Sequence[Mapping[str, Any]],
    dynamic_pairs: Sequence[Mapping[str, Any]],
    masks: Mapping[str, Any],
    ppo: ModuleType,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    union = complete_ordered_scalar(outputs, masked_batch(batch, masks["union"]), ppo)
    pf0 = complete_ordered_scalar(outputs, masked_batch(batch, masks["pf0"]), ppo)
    pf7 = complete_ordered_scalar(outputs, masked_batch(batch, masks["pf7"]), ppo)
    dominic = complete_ordered_scalar(
        outputs, masked_batch(batch, masks["dominic"]), ppo
    )
    bc_loss = 0.5 * union + (pf0 + pf7 + dominic) / 6.0

    pair_terms = []
    policy_logits = outputs["policy_logits"].float()
    for pair in list(fixed_pairs) + list(dynamic_pairs):
        row, positive, negative = pair_key(pair)
        margin = policy_logits[row, positive] - policy_logits[row, negative]
        threshold = policy_logits.new_tensor(float(pair["threshold"]))
        pair_terms.append(functional.softplus(threshold - margin))
    if not pair_terms:
        raise ProtocolError("pair objective unexpectedly empty")
    pair_loss = torch.stack(pair_terms).mean()

    option_mask = batch["option_mask"].bool()
    teacher = teacher_policy_logits.float().masked_fill(~option_mask, -1e9)
    current = policy_logits.masked_fill(~option_mask, -1e9)
    teacher_log_probs = torch.log_softmax(teacher, dim=1)
    teacher_probs = teacher_log_probs.exp()
    current_log_probs = torch.log_softmax(current, dim=1)
    per_row_kl = (teacher_probs * (teacher_log_probs - current_log_probs)).sum(dim=1)
    protected_weights = (
        batch["sample_weights"].float() * masks["protected"].float()
    )
    kl_loss = (per_row_kl * protected_weights).sum() / protected_weights.sum()
    total = bc_loss + PAIR_WEIGHT * pair_loss + KL_WEIGHT * kl_loss
    tensors = (union, pf0, pf7, dominic, bc_loss, pair_loss, kl_loss, total)
    if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in tensors):
        raise ProtocolError("nonfinite/non-scalar nonlinear objective component")
    audit = {
        "L_union": float(union.detach().cpu()),
        "L_pf0": float(pf0.detach().cpu()),
        "L_pf7": float(pf7.detach().cpu()),
        "L_dominic": float(dominic.detach().cpu()),
        "L_bc": float(bc_loss.detach().cpu()),
        "base_soft_loss": float(bc_loss.detach().cpu()),
        "L_pair": float(pair_loss.detach().cpu()),
        "L_kl": float(kl_loss.detach().cpu()),
        "L_total": float(total.detach().cpu()),
        "fixed_pair_count": len(fixed_pairs),
        "dynamic_pair_count": len(dynamic_pairs),
    }
    return total, audit


def dynamic_obligations(
    outputs: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    contract: Mapping[str, Any],
    v1: ModuleType,
) -> list[dict[str, Any]]:
    obligations: list[dict[str, Any]] = []
    fixed_keys = {pair_key(pair) for pair in contract["fixed_pairs"]}

    def make_pair(index: int, positive: int, negative: int, origin: str) -> None:
        if positive == negative:
            raise ProtocolError("dynamic false-obligation pair collapsed")
        candidate = {
            "row_index": int(index),
            "line_sha256": str(rows[index]["line_sha256"]),
            "positive_option": int(positive),
            "negative_option": int(negative),
            "threshold": local_positive_bf16_q(outputs, index, positive, negative),
            "threshold_source": "current_native_BF16_local_positive_q",
            "origins": [origin],
            "fixed": False,
        }
        if pair_key(candidate) not in fixed_keys:
            obligations.append(candidate)

    for name, sha, _ in v1.PF_TARGETS:
        index = contract["row_by_sha"][sha]
        if not current["ordered_correct"][sha]:
            positive, negative = first_different_pair(
                rows[index]["expert_order"], current["predictions"][index]
            )
            make_pair(index, positive, negative, f"PF_false:{name}")

    for spec in contract["zero_contract"]:
        index = int(spec["row_index"])
        sha = str(spec["line_sha256"])
        pair = {
            "row_index": index,
            "positive_option": spec["positive_option"],
            "negative_option": spec["negative_option"],
        }
        margin = native_pair_margin(outputs, pair)
        passed = margin > 0.0 or (margin == 0.0 and current["ordered_correct"][sha])
        if not passed:
            positive, negative = first_different_pair(
                rows[index]["expert_order"], current["predictions"][index]
            )
            make_pair(index, positive, negative, f"zero_guard_false:{spec['name']}")

    for index in contract["retention_indices"]:
        sha = str(rows[index]["line_sha256"])
        if baseline["ordered_correct"][sha] and not current["ordered_correct"][sha]:
            positive, negative = first_different_pair(
                rows[index]["expert_order"], current["predictions"][index]
            )
            make_pair(index, positive, negative, "retention160_ordered_flip")

    for index in contract["top1_indices"]:
        sha = str(rows[index]["line_sha256"])
        predicted = current["predictions"][index]
        still_correct = bool(predicted) and int(predicted[0]) == int(
            rows[index]["expert_order"][0]
        )
        if contract["baseline_top1_correct"][sha] and not still_correct:
            if not predicted:
                raise ProtocolError("top1 false obligation has empty prediction")
            make_pair(
                index,
                int(rows[index]["expert_order"][0]),
                int(predicted[0]),
                "top1_guard96_flip",
            )
    obligations.sort(key=lambda item: (pair_key(item), tuple(item["origins"])))
    return obligations


def merge_dynamic_pairs(
    ledger: list[dict[str, Any]], candidates: Sequence[Mapping[str, Any]], step: int
) -> list[dict[str, Any]]:
    existing = {pair_key(pair): pair for pair in ledger}
    added: list[dict[str, Any]] = []
    for candidate_value in candidates:
        candidate = dict(candidate_value)
        key = pair_key(candidate)
        if key in existing:
            record = existing[key]
            record["threshold"] = max(
                float(record["threshold"]), float(candidate["threshold"])
            )
            record["origins"] = sorted(
                set(str(value) for value in record["origins"])
                | set(str(value) for value in candidate["origins"])
            )
            continue
        candidate["discovered_step"] = int(step)
        ledger.append(candidate)
        existing[key] = candidate
        added.append(candidate)
    ledger.sort(key=pair_key)
    if len(ledger) > MAX_DYNAMIC_PAIRS:
        raise ProtocolError("dynamic pair ledger exceeded fail-closed cap64")
    return added


def soft_floor_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    cw22: ModuleType,
) -> dict[str, Any]:
    baseline_nll = baseline["ordered_nll_cpu"]
    current_nll = current["ordered_nll_cpu"]
    baseline_ce = baseline["top1_ce_cpu"]
    current_ce = current["top1_ce_cpu"]

    def indices_for(stratum: str) -> list[int]:
        return [
            index for index, row in enumerate(rows) if str(row["stratum"]) == stratum
        ]

    def mean_improvement(before: Any, after: Any, indices: Sequence[int]) -> float:
        if not indices:
            raise ProtocolError("soft diagnostic has empty stratum")
        return float(before[list(indices)].mean() - after[list(indices)].mean())

    pf0 = indices_for("pf_ctx0_hard")
    pf7 = indices_for("pf_ctx7_hard")
    dominic = indices_for("dominic_ctx0_hard")
    special9 = [index for index in dominic if cw22.is_dominic_special(rows[index])]
    szlach = indices_for("top1_guard_szlach")
    core_other = indices_for("top1_guard_core_other")
    retention = [
        index
        for index, row in enumerate(rows)
        if row["category"] not in {"hard", "top1_guard"}
    ]
    counts = {
        "pf0": len(pf0),
        "pf7": len(pf7),
        "dominic": len(dominic),
        "dominic_special": len(special9),
        "szlach": len(szlach),
        "core_other": len(core_other),
        "retention": len(retention),
    }
    if counts != {
        "pf0": 32,
        "pf7": 32,
        "dominic": 32,
        "dominic_special": 9,
        "szlach": 15,
        "core_other": 18,
        "retention": 160,
    }:
        raise ProtocolError(f"soft diagnostic count drift: {counts}")
    improvements = {
        "pf0_ordered_NLL": mean_improvement(baseline_nll, current_nll, pf0),
        "pf7_ordered_NLL": mean_improvement(baseline_nll, current_nll, pf7),
        "dominic_ordered_NLL": mean_improvement(
            baseline_nll, current_nll, dominic
        ),
        "dominic_special9_ordered_NLL": mean_improvement(
            baseline_nll, current_nll, special9
        ),
        "szlach15_top1_CE": mean_improvement(baseline_ce, current_ce, szlach),
        "core_other18_top1_CE": mean_improvement(
            baseline_ce, current_ce, core_other
        ),
        "retention160_ordered_NLL": mean_improvement(
            baseline_nll, current_nll, retention
        ),
    }
    old_floor_checks = {
        "pf0_at_least_2e4": improvements["pf0_ordered_NLL"] >= 2.0e-4,
        "pf7_at_least_3e4": improvements["pf7_ordered_NLL"] >= 3.0e-4,
        "dominic_at_least_2p5e4": improvements["dominic_ordered_NLL"] >= 2.5e-4,
        "dominic_special9_at_least_8e5": improvements[
            "dominic_special9_ordered_NLL"
        ]
        >= 8.0e-5,
        "szlach15_positive": improvements["szlach15_top1_CE"] > 0.0,
        "core_other18_positive": improvements["core_other18_top1_CE"] > 0.0,
        "retention160_at_least_minus_1e6": improvements[
            "retention160_ordered_NLL"
        ]
        >= -1.0e-6,
    }
    return {
        "counts": counts,
        "improvements": improvements,
        "old_floor_checks": old_floor_checks,
        "hard_gate_inclusion": False,
        "semantics": "diagnostic_soft_floors_only",
    }


def terminal_gate(
    *,
    outputs: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    baseline_native: Mapping[str, Any],
    contract: Mapping[str, Any],
    base_soft_loss: float,
    baseline_base_soft_loss: float,
    actual_delta: Any,
    candidate_hash: str,
    changed_names: Sequence[str],
    model: Any,
    parameters: Mapping[str, Any],
    helper: Any,
    cw20: ModuleType,
    cw22: ModuleType,
) -> dict[str, Any]:
    import numpy as np
    import torch

    actual_l2 = float(np.linalg.norm(actual_delta))
    pf_records: dict[str, Any] = {}
    for name, spec in contract["pf_contract"].items():
        sha = str(spec["line_sha256"])
        index = int(spec["row_index"])
        margin = native_pair_margin(outputs, spec)
        predicted = current["predictions"][index]
        expert = [int(value) for value in rows[index]["expert_order"]]
        ordered = current["ordered_correct"][sha]
        set_correct = set(predicted) == set(expert)
        threshold = float(spec["terminal_margin"])
        margin_pass = margin >= threshold
        if name == "pf7_boundary" and margin == 0.0:
            margin_pass = ordered and set_correct
        pf_records[name] = {
            "line_sha256": sha,
            "native_fixed_pair_margin": margin,
            "threshold": threshold,
            "ordered_correct": ordered,
            "set_correct": set_correct,
            "pass": margin_pass and ordered and set_correct,
        }

    zero_records: list[dict[str, Any]] = []
    for spec in contract["zero_contract"]:
        sha = str(spec["line_sha256"])
        margin = native_pair_margin(outputs, spec)
        ordered = current["ordered_correct"][sha]
        passed = margin > 0.0 or (margin == 0.0 and ordered)
        zero_records.append(
            {
                "name": spec["name"],
                "line_sha256": sha,
                "native_fixed_pair_margin": margin,
                "ordered_correct": ordered,
                "rule": "positive_or_zero_with_full_deterministic_order",
                "pass": passed,
            }
        )

    retention_flips = [
        str(rows[index]["line_sha256"])
        for index in contract["retention_indices"]
        if baseline["ordered_correct"][str(rows[index]["line_sha256"])]
        and not current["ordered_correct"][str(rows[index]["line_sha256"])]
    ]
    top1_flips = []
    for index in contract["top1_indices"]:
        sha = str(rows[index]["line_sha256"])
        predicted = current["predictions"][index]
        still_correct = bool(predicted) and int(predicted[0]) == int(
            rows[index]["expert_order"][0]
        )
        if contract["baseline_top1_correct"][sha] and not still_correct:
            top1_flips.append(sha)

    model_finite = all(
        bool(torch.isfinite(value).all()) for value in model.state_dict().values()
    )
    allowed = set(cw22.EXPECTED_ACTOR_NAMES)
    changed = set(str(name) for name in changed_names)
    nonactor_exact = (
        cw20.nonactor_sha(model, cw22.EXPECTED_ACTOR_NAMES, helper)
        == cw22.RAW_NONACTOR_SHA256
    )
    checks = {
        "native_policy_BF16": outputs["policy_logits"].dtype == torch.bfloat16,
        "three_PF_native_order_set_and_margin": all(
            record["pass"] for record in pf_records.values()
        ),
        "six_zero_margin_native_rules": all(record["pass"] for record in zero_records),
        "retention160_zero_ordered_flips": not retention_flips,
        "top1_guard96_zero_top1_flips": not top1_flips,
        "count_logits_native_exact_CW11": bool(
            torch.equal(outputs["count_logits"], baseline_native["count_logits"])
        ),
        "value_logits_native_exact_CW11": bool(
            torch.equal(outputs["value_logits"], baseline_native["value_logits"])
        ),
        "actual_actor_delta_finite": bool(np.isfinite(actual_delta).all()),
        "actual_actor_delta_within_hard_radius": math.isfinite(actual_l2)
        and actual_l2 <= HARD_RADIUS,
        "candidate_changed": candidate_hash != cw22.CW11_MODEL_SHA256,
        "changed_scope_actor6_only": bool(changed) and changed.issubset(allowed),
        "nonactor_exact_raw": nonactor_exact,
        "candidate_model_all_finite": model_finite,
        "base_soft_loss_finite": math.isfinite(base_soft_loss),
        "base_soft_loss_improves_by_floor": baseline_base_soft_loss
        - base_soft_loss
        >= SOFT_IMPROVEMENT_FLOOR,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "actual_additional_from_CW11_l2": actual_l2,
        "hard_radius": HARD_RADIUS,
        "PF_native": pf_records,
        "zero_margin_native": zero_records,
        "retention160_correct_to_wrong_count": len(retention_flips),
        "retention160_correct_to_wrong_line_sha256": retention_flips,
        "top1_guard96_correct_to_wrong_count": len(top1_flips),
        "top1_guard96_correct_to_wrong_line_sha256": top1_flips,
        "baseline_base_soft_loss": baseline_base_soft_loss,
        "candidate_base_soft_loss": base_soft_loss,
        "base_soft_loss_improvement": baseline_base_soft_loss - base_soft_loss,
        "dynamic_pair_thresholds_are_terminal_gates": False,
    }


def project_actor6(
    parameters: Mapping[str, Any],
    names: Sequence[str],
    cw11_flat: Any,
    cw20: ModuleType,
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    before_projection = cw20.flat_actor(parameters, names, np)
    raw_delta = before_projection - cw11_flat
    raw_l2 = float(np.linalg.norm(raw_delta))
    if not bool(np.isfinite(raw_delta).all()) or not math.isfinite(raw_l2):
        raise ProtocolError("nonfinite actor6 update before projection")
    projected = raw_l2 > PROJECT_RADIUS
    if projected:
        cw20.apply_flat_actor(
            parameters,
            names,
            cw11_flat + raw_delta * (PROJECT_RADIUS / raw_l2),
            torch,
        )
    actual = cw20.flat_actor(parameters, names, np)
    actual_delta = actual - cw11_flat
    actual_l2 = float(np.linalg.norm(actual_delta))
    if not bool(np.isfinite(actual_delta).all()) or actual_l2 > HARD_RADIUS:
        raise ProtocolError("post-projection actor6 exceeds finite hard radius")
    return {
        "raw_step_l2": raw_l2,
        "projection_applied": projected,
        "actual_l2": actual_l2,
        "actual_delta": actual_delta,
        "actual_delta_float64_le_sha256": float64_sha(actual_delta),
    }


def run_projected_core(
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cw20: ModuleType,
    cw19: ModuleType,
    cw15: ModuleType,
    modules: Mapping[str, ModuleType],
    *,
    cw22: ModuleType,
    cw23: ModuleType,
    v1: ModuleType,
) -> dict[str, Any]:
    import numpy as np
    import torch

    context_checks = cw22.validate_exact_context(context, cw20, cw15)
    helper = context["helper"]
    model = context["model"]
    ppo = helper.ppo
    dependency_checks = {
        "BC_source_exact_CW15": sha256_file(Path(helper.bc.__file__))
        == cw15.MODULE_SHAS[cw15.BC],
        "PPO_source_exact_CW15": sha256_file(Path(ppo.__file__))
        == cw15.MODULE_SHAS[cw15.PPO],
    }
    if not all(dependency_checks.values()):
        raise ProtocolError(f"BC/PPO dependency drift: {dependency_checks}")
    batch_cpu, cache_audit = v1.load_train_b352(
        rows, context["model_config"], helper.bc, ppo, cw22
    )
    device = next(model.parameters()).device
    named = dict(model.named_parameters())
    requires_grad_before = {
        name: bool(parameter.requires_grad) for name, parameter in named.items()
    }
    for parameter in named.values():
        parameter.requires_grad_(False)
        parameter.grad = None
    for name in cw22.EXPECTED_ACTOR_NAMES:
        named[name].requires_grad_(True)
    parameters = {name: named[name] for name in cw22.EXPECTED_ACTOR_NAMES}
    parameter_sequence = modules["geometry"].configure_actor6(model)
    if len(parameter_sequence) != len(cw22.EXPECTED_ACTOR_NAMES) or any(
        parameters[name] is not value
        for name, value in zip(cw22.EXPECTED_ACTOR_NAMES, parameter_sequence)
    ):
        raise ProtocolError("actor6 implementation tensor identity drift")
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        raise ProtocolError("actor6 storage is not float32")
    cw11_actor = cw20.clone_actor(parameters, cw22.EXPECTED_ACTOR_NAMES)
    cw11_flat = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
    try:
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch_cpu.items()
        }
        outputs = ppo.model_forward(model, batch, device)
        with torch.no_grad():
            baseline = evaluate_snapshot(outputs, batch, rows, ppo, cw22)
        baseline_native = {
            "count_logits": outputs["count_logits"].detach().clone(),
            "value_logits": outputs["value_logits"].detach().clone(),
        }
        teacher_policy_logits = outputs["policy_logits"].detach().clone()
        contract = build_row_contract(
            rows, batch, outputs, baseline, cw23, v1
        )
        dynamic_pairs: list[dict[str, Any]] = []
        baseline_obligations = dynamic_obligations(
            outputs, rows, baseline, baseline, contract, v1
        )
        baseline_added = merge_dynamic_pairs(dynamic_pairs, baseline_obligations, 0)
        total_loss, loss_audit = objective_components(
            outputs,
            batch,
            teacher_policy_logits,
            contract["fixed_pairs"],
            dynamic_pairs,
            contract["masks"],
            ppo,
        )
        baseline_loss_audit = dict(loss_audit)
        baseline_base_soft_loss = float(loss_audit["base_soft_loss"])
        if not math.isfinite(baseline_base_soft_loss):
            raise ProtocolError("baseline base soft loss is nonfinite")

        optimizer = torch.optim.SGD(
            list(parameters.values()),
            lr=LEARNING_RATE,
            momentum=0.0,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            maximize=False,
            foreach=False,
            differentiable=False,
            fused=False,
        )
        if optimizer.state or optimizer.state_dict()["state"]:
            raise ProtocolError("fresh plain SGD unexpectedly has state")

        actor_hashes = {
            hashlib.sha256(
                cw20.actor_bytes(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
            ).hexdigest()
        }
        trajectory: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        last_trial: dict[str, Any] | None = None
        for step_index in range(1, MAX_STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradients = [parameters[name].grad for name in cw22.EXPECTED_ACTOR_NAMES]
            if any(gradient is None for gradient in gradients) or any(
                not bool(torch.isfinite(gradient).all())
                for gradient in gradients
                if gradient is not None
            ):
                raise ProtocolError("actor6 backward produced missing/nonfinite gradient")
            preclip_norm = torch.nn.utils.clip_grad_norm_(
                list(parameters.values()), MAX_GRAD_NORM
            )
            preclip_norm_value = float(preclip_norm.detach().cpu())
            if not math.isfinite(preclip_norm_value):
                raise ProtocolError("nonfinite actor6 gradient norm")
            optimizer.step()
            if optimizer.state or optimizer.state_dict()["state"]:
                raise ProtocolError("plain SGD acquired forbidden optimizer state")

            projection = project_actor6(
                parameters,
                cw22.EXPECTED_ACTOR_NAMES,
                cw11_flat,
                cw20,
                np,
                torch,
            )
            actor_bytes = cw20.actor_bytes(
                parameters, cw22.EXPECTED_ACTOR_NAMES, np
            )
            actor_hash = hashlib.sha256(actor_bytes).hexdigest()
            if actor_hash in actor_hashes:
                raise ProtocolError("repeated actor6 state in nonlinear trajectory")
            actor_hashes.add(actor_hash)
            candidate_hash = helper.model_state_sha256(model.state_dict())
            outputs = ppo.model_forward(model, batch, device)
            with torch.no_grad():
                current = evaluate_snapshot(outputs, batch, rows, ppo, cw22)
            obligations = dynamic_obligations(
                outputs, rows, baseline, current, contract, v1
            )
            added = merge_dynamic_pairs(dynamic_pairs, obligations, step_index)
            total_loss, loss_audit = objective_components(
                outputs,
                batch,
                teacher_policy_logits,
                contract["fixed_pairs"],
                dynamic_pairs,
                contract["masks"],
                ppo,
            )
            changed_names = [
                name
                for name in cw22.EXPECTED_ACTOR_NAMES
                if not torch.equal(parameters[name], cw11_actor[name])
            ]
            gate = terminal_gate(
                outputs=outputs,
                rows=rows,
                current=current,
                baseline=baseline,
                baseline_native=baseline_native,
                contract=contract,
                base_soft_loss=float(loss_audit["base_soft_loss"]),
                baseline_base_soft_loss=baseline_base_soft_loss,
                actual_delta=projection["actual_delta"],
                candidate_hash=candidate_hash,
                changed_names=changed_names,
                model=model,
                parameters=parameters,
                helper=helper,
                cw20=cw20,
                cw22=cw22,
            )
            diagnostics = soft_floor_diagnostics(rows, baseline, current, cw22)
            trial = {
                "step": step_index,
                "candidate_model_state_sha256": candidate_hash,
                "candidate_actor_float32_le_sha256": actor_hash,
                "gradient_norm_before_clip": preclip_norm_value,
                "gradient_clip_max_norm": MAX_GRAD_NORM,
                "loss": loss_audit,
                "projection": {
                    key: value
                    for key, value in projection.items()
                    if key != "actual_delta"
                },
                "dynamic_pairs_added": [dict(value) for value in added],
                "dynamic_pair_count": len(dynamic_pairs),
                "native_terminal_gate": gate,
                "soft_floor_diagnostics": diagnostics,
                "pass": gate["pass"],
            }
            trajectory.append(trial)
            last_trial = trial
            if gate["pass"]:
                selected = {
                    "step": step_index,
                    "candidate_model_state_sha256": candidate_hash,
                    "actor_bytes": actor_bytes,
                    "actor_hash": actor_hash,
                    "actual_delta": projection["actual_delta"].copy(),
                    "actual_delta_sha256": projection[
                        "actual_delta_float64_le_sha256"
                    ],
                    "changed_names": changed_names,
                    "loss": dict(loss_audit),
                    "gate": gate,
                    "diagnostics": diagnostics,
                    "dynamic_pairs": [dict(value) for value in dynamic_pairs],
                }
                break

        common = {
            "context_checks": context_checks,
            "dependency_checks": dependency_checks,
            "cache": cache_audit,
            "baseline_checks": contract["baseline_checks"],
            "objective_mask_counts": contract["mask_counts"],
            "baseline_loss": baseline_loss_audit,
            "baseline_dynamic_pairs_added": [dict(value) for value in baseline_added],
            "fixed_pair_contract": [dict(value) for value in contract["fixed_pairs"]],
            "zero_margin_contract": [dict(value) for value in contract["zero_contract"]],
            "optimizer_contract": {
                "name": "SGD",
                "plain": True,
                "learning_rate": LEARNING_RATE,
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
                "maximize": False,
                "foreach": False,
                "differentiable": False,
                "fused": False,
                "max_grad_norm": MAX_GRAD_NORM,
                "state_entries_terminal": len(optimizer.state),
            },
            "projection_contract": {
                "center": "exact_CW11_actor6_float32",
                "projection_radius": PROJECT_RADIUS,
                "hard_actual_radius": HARD_RADIUS,
                "project_after_every_step": True,
                "actual_float32_actor_reread_after_projection": True,
            },
            "objective_contract": {
                "formula": "L_bc + 0.25*L_pair + 0.012*L_kl",
                "L_bc": "0.5*L_union + (L_pf0+L_pf7+L_dominic)/6",
                "L_pair": "mean softplus(threshold-current_pair_margin)",
                "L_kl": "CW11 teacher masked policy KL on retention160 plus top1_guard96",
                "PPO_semantics": "CW11_PPO_policy_preservation_not_new_PPO_rollout_or_update",
                "dynamic_pairs": "training_only_sticky_deduplicated_cap64",
                "old_NLL_CE_floors": "diagnostic_only",
            },
            "trajectory": trajectory,
            "changed_candidate_train_shadow_count": len(trajectory),
            "first_hard_pass_stop": True,
            "max_steps": MAX_STEPS,
        }
        if selected is None:
            return {
                "decision": "NO_GO_CW24_PROJECTED_NONLINEAR_TRAIN_GATE",
                "reason": "NO_NATIVE_HARD_PASS_WITHIN_48_PROJECTED_SGD_STEPS",
                **common,
                "trial": last_trial,
                "candidate_payload": None,
            }

        if any(record["pass"] for record in trajectory[: selected["step"] - 1]):
            raise ProtocolError("selected endpoint is not first native hard pass")
        layout = cw20.actor_layout(parameters, cw22.EXPECTED_ACTOR_NAMES)
        payload = {
            "anchor": {
                "reconstruction_base": "original_raw_U468",
                "raw_checkpoint": str(cw23.RAW.relative_to(ROOT)),
                "raw_checkpoint_file_sha256": cw23.RAW_SHA256,
                "raw_model_state_sha256": cw22.RAW_MODEL_SHA256,
                "CW11_provenance_model_state_sha256": cw22.CW11_MODEL_SHA256,
                "CW11_provenance_vector_float64_le_sha256": cw22.CW11_VECTOR_SHA256,
                "CW11_provenance_active_pair_ledger_sha256": cw22.CW11_LEDGER_SHA256,
                "CW11_materialized_eval_only_checkpoint_used": False,
                "terminal_model_state_sha256": selected[
                    "candidate_model_state_sha256"
                ],
            },
            "formula": "load_original_raw_U468_then_replace_all_six_absolute_actor_float32_tensors_from_frozen_payload",
            "actor_names": list(cw22.EXPECTED_ACTOR_NAMES),
            "changed_actor_names": selected["changed_names"],
            "actor_layout": layout,
            "actor_layout_sha256": hashlib.sha256(
                json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "candidate_actor_float32_le": cw20.xz_payload(selected["actor_bytes"]),
            "actual_additional_from_CW11_l2": selected["gate"][
                "actual_additional_from_CW11_l2"
            ],
            "actual_delta_float64_le_sha256": selected["actual_delta_sha256"],
            "selected_step": selected["step"],
            "selection_sha256": v1.ROWS_CANONICAL_SHA256,
            "cache_sha256": cache_audit["cache_sha256"],
        }

        modules["cutting"].restore_raw_actor(
            modules["ram"], parameter_sequence, context["raw_actor"], torch
        )
        raw_restored_hash = helper.model_state_sha256(model.state_dict())
        decoded = cw20.decode_xz(payload["candidate_actor_float32_le"])
        cw20.copy_actor_bytes(
            parameters,
            cw22.EXPECTED_ACTOR_NAMES,
            layout,
            decoded,
            np,
            torch,
        )
        reconstructed_hash = helper.model_state_sha256(model.state_dict())
        replay_flat = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
        replay_delta = replay_flat - cw11_flat
        replay_outputs = ppo.model_forward(model, batch, device)
        with torch.no_grad():
            replay_snapshot = evaluate_snapshot(
                replay_outputs, batch, rows, ppo, cw22
            )
        _, replay_loss = objective_components(
            replay_outputs,
            batch,
            teacher_policy_logits,
            contract["fixed_pairs"],
            selected["dynamic_pairs"],
            contract["masks"],
            ppo,
        )
        replay_changed_names = [
            name
            for name in cw22.EXPECTED_ACTOR_NAMES
            if not torch.equal(parameters[name], cw11_actor[name])
        ]
        replay_gate = terminal_gate(
            outputs=replay_outputs,
            rows=rows,
            current=replay_snapshot,
            baseline=baseline,
            baseline_native=baseline_native,
            contract=contract,
            base_soft_loss=float(replay_loss["base_soft_loss"]),
            baseline_base_soft_loss=baseline_base_soft_loss,
            actual_delta=replay_delta,
            candidate_hash=reconstructed_hash,
            changed_names=replay_changed_names,
            model=model,
            parameters=parameters,
            helper=helper,
            cw20=cw20,
            cw22=cw22,
        )
        replay_diagnostics = soft_floor_diagnostics(
            rows, baseline, replay_snapshot, cw22
        )
        reconstruction_checks = {
            "raw_U468_restore_exact": raw_restored_hash == cw22.RAW_MODEL_SHA256,
            "candidate_hash_exact": reconstructed_hash
            == selected["candidate_model_state_sha256"],
            "candidate_actor_bytes_exact": cw20.actor_bytes(
                parameters, cw22.EXPECTED_ACTOR_NAMES, np
            )
            == selected["actor_bytes"],
            "actual_delta_sha_exact": float64_sha(replay_delta)
            == selected["actual_delta_sha256"],
            "nonactor_exact_raw": cw20.nonactor_sha(
                model, cw22.EXPECTED_ACTOR_NAMES, helper
            )
            == cw22.RAW_NONACTOR_SHA256,
            "repeat_B352_native_terminal_gate": replay_gate["pass"],
            "repeat_base_soft_loss_exact": replay_loss["base_soft_loss"]
            == selected["loss"]["base_soft_loss"],
            "repeat_soft_diagnostics_exact": replay_diagnostics
            == selected["diagnostics"],
        }
        passed = all(reconstruction_checks.values())
        final_checks = {
            "exact_CW11_context": all(context_checks.values()),
            "dependency_sources": all(dependency_checks.values()),
            "B352_exact_train_only_cache": all(cache_audit["checks"].values()),
            "first_native_hard_pass": selected["gate"]["pass"],
            "absolute_payload_reconstruction_and_repeat_gate": passed,
        }
        final_pass = passed and all(final_checks.values())
        return {
            "decision": "GO_CW24_PROJECTED_NONLINEAR_TRAIN_GATE"
            if final_pass
            else "NO_GO_CW24_PROJECTED_NONLINEAR_TRAIN_GATE",
            "reason": "FIRST_PROJECTED_SGD_ENDPOINT_PASSED_AND_REPLAYED"
            if final_pass
            else "ABSOLUTE_PAYLOAD_RECONSTRUCTION_OR_REPEAT_GATE_FAILED",
            **common,
            "trial": trajectory[selected["step"] - 1],
            "pure_payload_reconstruction": {
                "checks": reconstruction_checks,
                "pass": passed,
                "raw_model_state_sha256": raw_restored_hash,
                "candidate_model_state_sha256": reconstructed_hash,
                "repeat_native_terminal_gate": replay_gate,
                "repeat_loss": replay_loss,
                "repeat_soft_floor_diagnostics": replay_diagnostics,
            },
            "final_checks": final_checks,
            "candidate_payload": payload if final_pass else None,
        }
    finally:
        cw20.restore_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, cw11_actor, torch)
        for name, parameter in named.items():
            parameter.requires_grad_(requires_grad_before[name])
            parameter.grad = None


def production_run() -> dict[str, Any]:
    pre_cuda_runtime = validate_pre_cuda_runtime()
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    v1, v1_evidence = load_locked_module(V1, V1_SHA256, SOURCE_MODE, "cw24_v12_v1")
    rows, selection_audit = v1.load_selection()
    cw23, cw23_evidence = v1.load_locked_module(
        v1.CW23, v1.CW23_SHA256, v1.CW23_MODE, "cw24_v12_cw23"
    )
    cw22, cw22_evidence = cw23.load_cw22()
    original_core = cw22.run_targeted_core
    original_seed = cw22.SEED

    def bound_core(
        context: Mapping[str, Any],
        ignored_rows: Sequence[Mapping[str, Any]],
        cw20: ModuleType,
        cw19: ModuleType,
        cw15: ModuleType,
        modules: Mapping[str, ModuleType],
    ) -> dict[str, Any]:
        if len(ignored_rows) != 256:
            raise ProtocolError("frozen CW22 reconstruction callback row drift")
        return run_projected_core(
            context,
            rows,
            cw20,
            cw19,
            cw15,
            modules,
            cw22=cw22,
            cw23=cw23,
            v1=v1,
        )

    cw22.run_targeted_core = bound_core
    cw22.SEED = SEED
    try:
        result = cw22.production_run()
    finally:
        cw22.run_targeted_core = original_core
        cw22.SEED = original_seed

    post_source = self_evidence(require_frozen=True)
    post_dependencies = dependency_evidence()
    rehash_checks = {
        "source_byte_identity": post_source == source,
        "all_direct_dependencies_byte_identity": post_dependencies == dependencies,
    }
    if not all(rehash_checks.values()):
        raise ProtocolError(f"post-run immutable input drift: {rehash_checks}")

    endpoint = result["endpoint"]
    decision = str(endpoint["decision"])
    payload_present = endpoint.get("candidate_payload") is not None
    if (decision.startswith("GO_")) != payload_present:
        raise ProtocolError("GO/payload equivalence failed")
    if decision.startswith("NO_GO") and payload_present:
        raise ProtocolError("NO_GO exposed forbidden candidate payload")
    if int(endpoint.get("changed_candidate_train_shadow_count", -1)) > MAX_STEPS:
        raise ProtocolError("changed train-shadow budget exceeded")

    frozen_engine_selection = result["selection"]
    result["schema_version"] = SCHEMA
    result["status"] = decision
    result["decision"] = decision
    result["seed"] = SEED
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_projected_nonlinear_special_BC"
    )
    result["selection"] = {
        "rows": 352,
        "sha256": v1.ROWS_CANONICAL_SHA256,
        "checks": selection_audit["checks"],
        "counts": selection_audit["counts"],
        "scope": selection_audit["scope"],
        "optimization_contract": {
            "single_seed": SEED,
            "plain_actor6_SGD": True,
            "learning_rate": LEARNING_RATE,
            "maximum_steps": MAX_STEPS,
            "projection_radius": PROJECT_RADIUS,
            "hard_actual_radius": HARD_RADIUS,
            "first_native_hard_pass_stop": True,
            "dynamic_pairs_training_only": True,
            "old_NLL_CE_floors_soft_only": True,
        },
        "frozen_engine_B256_reconstruction_selection": frozen_engine_selection,
    }
    historical = result["historical_exact_CW11_replay"]
    historical.pop("new_validation_rows_opened_for_CW22_selection_or_candidate", None)
    historical["new_validation_rows_opened_for_CW24_v12_selection_or_candidate"] = 0
    result.setdefault("inputs", {})["CW24_v12_source"] = source
    result["inputs"]["CW24_v12_source_post_run"] = post_source
    result["inputs"]["CW24_v12_dependencies"] = dependencies
    result["inputs"]["CW24_v12_dependencies_post_run"] = post_dependencies
    result["inputs"]["CW24_v12_v1_loader"] = v1_evidence
    result["inputs"]["CW24_v12_CW23_loader"] = cw23_evidence
    result["inputs"]["CW24_v12_CW22_engine"] = cw22_evidence
    audit = result.setdefault("audit", {})
    audit["source"] = source
    audit["CW24_v12_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "pre_cuda_runtime": pre_cuda_runtime,
        "post_run_rehash_checks": rehash_checks,
        "train_only": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 1,
        "optimizer_step_calls": endpoint.get("changed_candidate_train_shadow_count"),
        "optimizer_step_calls_within_48": int(
            endpoint.get("changed_candidate_train_shadow_count", -1)
        )
        <= MAX_STEPS,
        "official_unique_changed_candidate_count_consumed": 0,
        "candidate_payload_present_iff_GO": payload_present
        == decision.startswith("GO_"),
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "train_only_result_is_not_promotion_evidence": True,
        "PPO_term_is_policy_KL_not_new_rollout": True,
        "fulltrain_exact_CW11_comparator_required_after_GO": True,
        "specialist_then_broad_then_Gold_required_after_fulltrain_GO": True,
    }
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 2
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    canonical_json(result)
    return result


def audit_only() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v12 output target must be absent during static audit")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v12 attempt marker must be absent during static audit")
    source = self_evidence(require_frozen=False)
    dependencies = dependency_evidence()
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_pass_unfrozen_draft",
        "source": source,
        "source_audit": source_audit(),
        "dependencies": dependencies,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent": True,
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
        "attempt_marker_absent": True,
        "pre_cuda_runtime": runtime,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "CUDA_initialized": False,
        "production_modules_executed": False,
        "writes_performed": 0,
        "draft_freeze_required_before_production": source["mode_octal"] != "0555",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        raise ProtocolError("output path is not the exact v12 target")
    if args.audit_only:
        print(canonical_json(audit_only()).decode("utf-8"), end="")
        return
    if not path_absent(OUTPUT):
        raise ProtocolError("v12 O_EXCL output already exists")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v12 one-shot attempt marker already exists")
    attempt = claim_attempt()
    result = production_run()
    attempt_post = regular_source(
        ATTEMPT_MARKER,
        attempt["publication"]["sha256"],
        0o444,
        "v12 attempt marker post-run",
    )[1]
    final_source = self_evidence(require_frozen=True)
    final_dependencies = dependency_evidence()
    attempt_source = attempt["payload"]["source"]
    attempt_dependencies = attempt["payload"]["dependencies"]
    production_source_pre = result["inputs"]["CW24_v12_source"]
    production_source_post = result["inputs"]["CW24_v12_source_post_run"]
    production_dependencies_pre = result["inputs"]["CW24_v12_dependencies"]
    production_dependencies_post = result["inputs"][
        "CW24_v12_dependencies_post_run"
    ]
    binding_checks = {
        "attempt_source_equals_production_pre": attempt_source
        == binding_summary(production_source_pre),
        "attempt_source_equals_production_post": attempt_source
        == binding_summary(production_source_post),
        "attempt_source_equals_final_rehash": attempt_source
        == binding_summary(final_source),
        "attempt_dependencies_equal_production_pre": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in production_dependencies_pre.items()
        },
        "attempt_dependencies_equal_production_post": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in production_dependencies_post.items()
        },
        "attempt_dependencies_equal_final_rehash": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in final_dependencies.items()
        },
    }
    if not all(binding_checks.values()):
        raise ProtocolError(f"attempt-to-publication binding drift: {binding_checks}")
    result["inputs"]["CW24_v12_source_final_before_publication"] = final_source
    result["inputs"][
        "CW24_v12_dependencies_final_before_publication"
    ] = final_dependencies
    result.setdefault("audit", {})["one_shot_attempt"] = {
        "claim": attempt,
        "post_run": attempt_post,
        "unchanged": attempt_post["sha256"]
        == attempt["publication"]["sha256"],
        "claim_to_publication_binding_checks": binding_checks,
    }
    canonical_json(result)
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    endpoint = result["endpoint"]
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "candidate_model_state_sha256": endpoint.get("trial", {}).get(
                    "candidate_model_state_sha256"
                ),
                "candidate_payload_present": endpoint.get("candidate_payload")
                is not None,
                "changed_candidate_train_shadow_count": endpoint.get(
                    "changed_candidate_train_shadow_count"
                ),
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
                "output": publication,
            }
        ).decode("utf-8"),
        end="",
    )


if __name__ == "__main__":
    main()

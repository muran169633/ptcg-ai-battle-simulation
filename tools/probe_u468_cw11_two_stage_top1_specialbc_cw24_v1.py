#!/usr/bin/env python3
"""CW24 two-stage actor6 special-BC train-only probe.

The historical frozen callback reconstructs exact CW11 in RAM.  This probe
loads the frozen train-only B352 panel, solves one PF-priority minimum-norm
boundary update, adds one null-space auxiliary direction for Dominic and top1
retention, evaluates exactly one changed train shadow, and restores CW11.  It
never materializes a model checkpoint and never opens a new evaluation row.
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
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py"
CW23 = TOOLS / "probe_u468_cw11_boundary_qp_specialbc_cw23_v1.py"
CW23_SHA256 = "4a2d209af71a7ea5d955bd373f1fbada936688fc5bc597615c41192dc360b4b8"
CW23_MODE = 0o555
SELECTOR = TOOLS / "select_cw24_top1_b352_from_cw22_v1.py"
SELECTOR_SHA256 = "6af386d162f4a36529d3c55b917db079abe247f3925e4f399007b842f7469ce0"
SELECTION = ROOT / "artifacts/cw24_top1_b352_selection_v1.json"
SELECTION_SHA256 = "1f72b26eccd436ca0d341f837ec42949f5823bf7f0be73461aaeab28166cc222"
ROWS_CANONICAL_SHA256 = "2d1047b6984d31fb55a6e20e33ef5abb23c826a4c5ff13c6a7716ccd57500dd0"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
OUTPUT = ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v1.json"
SCHEMA = "ptcg-u468-cw11-two-stage-top1-specialbc-cw24-v1"
SEED = 202608041

PF_TARGETS = (
    ("pf0_boundary_a", "92b3f535ff3ac2c1aa4e95a3ba1c71fc23fba4e3aa0a93f32ce4e2c4ea180379", -1.0 / 512.0),
    ("pf0_boundary_b", "c691171c4e7093f498f29fda034e456a52e277c1d27defffc886217f08ca935b", -1.0 / 512.0),
    ("pf7_boundary", "d35040093be24508b97bbfb4a9f0ea98d0f2ef2d7f6af1c9d40119d31de8f123", -3.0 / 512.0),
)
ZERO_MARGIN_SHA256 = (
    "061627395d4bb484f2e21d1a647f6d69fad043016efc102fe093aa55815f4a7f",
    "51eb642eb4c5a82cad4ac22c82bca296841ee8e348a19028aa847b9b658b4305",
    "7bf6fa80288ce4f38874d11da426030af291cdd3880a0242458cd111632e537d",
    "3b8e123d82ea6b8233d3d6088830527ec635f0ac363e72bb95bbd8a99d99cd21",
    "78f8d0275a2f38f5a043ffee28df5901001d66ab964422bbfaba93c1403a65dc",
    "10192657ae504eed3bc9e3fd59f80f7e9499d6022e38bdddd6c5da69e78afd81",
)
DOMINIC_CLOSEST_SHA256 = "d3c16aacdf1963bbc02247ed3dbd2d8ebcf14f39bc84fd0207794d7e143d452b"
TARGET_FINAL_MARGIN = 1.0 / 512.0
PF_RADIUS_CAP = 9.2e-4
TOTAL_RADIUS_CAP = 1.0e-3
PINV_RCOND = 1.0e-12
GRAM_CONDITION_LIMIT = 1.0e8
AUX_WEIGHTS = {
    "dominic32_nll_descent": 0.35,
    "dominic_closest_margin_ascent": 0.25,
    "special9_nll_descent": 0.20,
    "szlach_top1_ce_descent": 0.12,
    "core_other_top1_ce_descent": 0.08,
}


class ProtocolError(RuntimeError):
    """Fail-closed CW24 protocol error."""


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


def regular_source(path: Path, expected_sha: str, expected_mode: int, label: str) -> tuple[bytes, dict[str, Any]]:
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
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "one_candidate_formula": (b"for radius" + b" in") not in source
        and (b"candidate_grid" + b"=") not in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"CW24 source audit failed: {checks}")
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
                raise ProtocolError("short CW24 result write")
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
        raise ProtocolError(f"CW24 result publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def load_locked_module(path: Path, expected_sha: str, expected_mode: int, name: str) -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = regular_source(path, expected_sha, expected_mode, name)
    module_name = f"cw24_locked_{name}"
    if module_name in sys.modules:
        raise ProtocolError(f"module name occupied: {module_name}")
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


def load_selection() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, selector_evidence = regular_source(SELECTOR, SELECTOR_SHA256, 0o555, "CW24 selector")
    raw, selection_evidence = regular_source(SELECTION, SELECTION_SHA256, 0o444, "CW24 B352 selection")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid B352 JSON: {exc}") from exc
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    checks = {
        "schema_exact": payload.get("schema_version") == "ptcg-cw24-top1-b352-train-selection-v1",
        "status_exact": payload.get("status") == "completed_frozen_train_only_B352_selection",
        "rows352": isinstance(rows, list) and len(rows) == 352,
        "rows_sha_exact": isinstance(rows, list)
        and hashlib.sha256(canonical_json(rows)).hexdigest() == ROWS_CANONICAL_SHA256,
        "unique_line_sha352": isinstance(rows, list)
        and len({str(row["line_sha256"]) for row in rows}) == 352,
        "all_train_members": isinstance(rows, list)
        and all(str(row["member"]).startswith("train/") for row in rows),
        "top1_guard96": isinstance(rows, list)
        and sum(str(row.get("category")) == "top1_guard" for row in rows) == 96,
        "terminal_checks_all_true": all(payload.get("terminal_checks", {}).values()),
        "zero_evaluation_scope": payload.get("contract", {}).get("validation_or_test_rows_opened") == 0
        and payload.get("contract", {}).get("official_evaluation_count") == 0,
    }
    if not all(checks.values()):
        raise ProtocolError(f"B352 selection drift: {checks}")
    return [dict(row) for row in rows], {
        "selector": selector_evidence,
        "selection": selection_evidence,
        "checks": checks,
        "counts": payload["counts"],
        "rows_canonical_sha256": ROWS_CANONICAL_SHA256,
        "scope": payload["contract"],
    }


def load_train_b352(
    rows: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
    bc: ModuleType,
    ppo: ModuleType,
    cw22: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wanted: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {
        source: {} for source in cw22.DATASETS
    }
    for row in rows:
        source = str(row["source"])
        key = (str(row["member"]), int(row["line_index"]))
        if source not in wanted or key in wanted[source]:
            raise ProtocolError("B352 source/key drift")
        wanted[source][key] = row
    features: dict[tuple[str, str, int], dict[str, Any]] = {}
    opened: dict[str, list[str]] = {}
    for source, path in cw22.DATASETS.items():
        by_member: dict[str, set[int]] = defaultdict(set)
        for member, line_index in wanted[source]:
            if not member.startswith("train/") or not member.endswith(".jsonl") or ".." in member.split("/"):
                raise ProtocolError("attempted noncanonical train member")
            by_member[member].add(line_index)
        opened[source] = []
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ProtocolError(f"{source}: duplicate ZIP members")
            for member in sorted(by_member):
                archive.getinfo(member)
                opened[source].append(member)
                remaining = set(by_member[member])
                with archive.open(member) as handle:
                    for line_index, raw in enumerate(handle):
                        if line_index not in remaining:
                            continue
                        identity = wanted[source][(member, line_index)]
                        if hashlib.sha256(raw).hexdigest() != identity["line_sha256"]:
                            raise ProtocolError("selected train line SHA drift")
                        source_row = json.loads(raw)
                        if str(source_row.get("split", "")) != "train":
                            raise ProtocolError("selected source row is not train")
                        if str(source_row.get("episode_id", "")) != str(identity["episode_id"]):
                            raise ProtocolError("selected episode drift")
                        if str(source_row.get("team_name", "")) != str(identity["team_name"]):
                            raise ProtocolError("selected team drift")
                        previous_max = bc.MAX_ACTION_COUNT
                        bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
                        try:
                            feature = bc.featurize_row(
                                source_row,
                                int(model_config["hash_size"]),
                                int(model_config["max_state_entities"]),
                            )
                        finally:
                            bc.MAX_ACTION_COUNT = previous_max
                        if feature is None:
                            raise ProtocolError("selected row no longer featurizes")
                        expert = [int(value) for value in source_row.get("action", [])]
                        decision_checks = {
                            "expert": expert == [int(value) for value in identity["expert_order"]],
                            "context": int(feature["context"]) == int(identity["context"]),
                            "min_count": int(feature["min_count"]) == int(identity["min_count"]),
                            "max_count": int(feature["max_count"]) == int(identity["max_count"]),
                        }
                        if not all(decision_checks.values()):
                            raise ProtocolError(f"selected decision drift: {decision_checks}")
                        feature["action_sequence"] = expert
                        feature["sample_weight"] = (
                            cw22.CONTEXT34_SAMPLE_WEIGHT
                            if int(feature["context"]) == ppo.SKILL_ORDER_CONTEXT
                            else 1.0
                        )
                        features[(source, member, line_index)] = feature
                        remaining.remove(line_index)
                        if not remaining:
                            break
                if remaining:
                    raise ProtocolError(f"selected lines missing: {source}/{member}")
    ordered_features = [
        features[(str(row["source"]), str(row["member"]), int(row["line_index"]))]
        for row in rows
    ]
    batch = bc.collate_decisions(
        ordered_features,
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    import torch

    context34_mask = batch["contexts"] == ppo.SKILL_ORDER_CONTEXT
    checks = {
        "batch_rows352": int(batch["action_counts"].shape[0]) == 352,
        "context34_rows16": int(context34_mask.sum()) == 16,
        "context34_weights_one_third": bool(
            torch.allclose(
                batch["sample_weights"][context34_mask],
                torch.full_like(batch["sample_weights"][context34_mask], cw22.CONTEXT34_SAMPLE_WEIGHT),
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "other_weights_one": bool(
            torch.allclose(
                batch["sample_weights"][~context34_mask],
                torch.ones_like(batch["sample_weights"][~context34_mask]),
                rtol=0.0,
                atol=0.0,
            )
        ),
        "all_action_counts_positive": bool((batch["action_counts"] > 0).all()),
        "opened_train_only": all(
            members and all(member.startswith("train/") for member in members)
            for members in opened.values()
        ),
        "source_counts_exact": Counter(str(row["source"]) for row in rows)
        == Counter({"core5": 119, "flg": 48, "pokemonfan": 185}),
    }
    if not all(checks.values()):
        raise ProtocolError(f"B352 collate gate failed: {checks}")
    cache_sha, manifest = cw22.tensor_batch_manifest(batch)
    return batch, {
        "checks": checks,
        "opened_train_members": opened,
        "non_train_members_opened": False,
        "cache_sha256": cache_sha,
        "tensor_manifest": manifest,
    }


def array_sha(value: Any) -> str:
    contiguous = value.astype("<f8", copy=False)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def top1_ce_per_row(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> Any:
    import torch

    logits = outputs["policy_logits"].float()
    mask = batch["option_mask"].bool()
    expert_first = batch["action_sequences"][:, 0]
    log_probs = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=1)
    return -log_probs.gather(1, expert_first.unsqueeze(1)).squeeze(1)


def mean_at(values: Any, indices: Sequence[int], device: Any) -> Any:
    import torch

    if not indices:
        raise ProtocolError("empty objective index set")
    index = torch.tensor(indices, dtype=torch.long, device=device)
    return values[index].mean()


def run_two_stage_core(
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cw20: ModuleType,
    cw19: ModuleType,
    cw15: ModuleType,
    modules: Mapping[str, ModuleType],
    *,
    cw22: ModuleType,
    cw23: ModuleType,
) -> dict[str, Any]:
    import numpy as np
    import torch

    context_checks = cw22.validate_exact_context(context, cw20, cw15)
    helper = context["helper"]
    model = context["model"]
    ppo = helper.ppo
    dependency_checks = {
        "BC_source_exact": sha256_file(Path(helper.bc.__file__)) == cw15.MODULE_SHAS[cw15.BC],
        "PPO_source_exact": sha256_file(Path(ppo.__file__)) == cw15.MODULE_SHAS[cw15.PPO],
    }
    if not all(dependency_checks.values()):
        raise ProtocolError(f"BC/PPO dependency drift: {dependency_checks}")
    batch_cpu, cache_audit = load_train_b352(rows, context["model_config"], helper.bc, ppo, cw22)
    device = next(model.parameters()).device
    named = dict(model.named_parameters())
    requires_grad_before = {name: bool(parameter.requires_grad) for name, parameter in named.items()}
    for parameter in named.values():
        parameter.requires_grad_(False)
        parameter.grad = None
    for name in cw22.EXPECTED_ACTOR_NAMES:
        named[name].requires_grad_(True)
    parameters = {name: named[name] for name in cw22.EXPECTED_ACTOR_NAMES}
    parameter_sequence = modules["geometry"].configure_actor6(model)
    if any(
        parameters[name] is not value
        for name, value in zip(cw22.EXPECTED_ACTOR_NAMES, parameter_sequence)
    ):
        raise ProtocolError("actor6 tensor identity drift")
    cw11_actor = cw20.clone_actor(parameters, cw22.EXPECTED_ACTOR_NAMES)
    cw11_flat = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
    try:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch_cpu.items()}
        outputs = ppo.model_forward(model, batch, device)
        baseline, baseline_ordered_correct, baseline_predictions = cw22.evaluate_outputs(
            outputs, batch, rows, ppo
        )
        baseline_native = {
            "count_logits": outputs["count_logits"].detach().clone(),
            "value_logits": outputs["value_logits"].detach().clone(),
        }
        original_retention_indices = [
            index
            for index, row in enumerate(rows)
            if row["category"] not in {"hard", "top1_guard"}
        ]
        top1_guard_indices = [
            index for index, row in enumerate(rows) if row["category"] == "top1_guard"
        ]
        hard_indices = [index for index, row in enumerate(rows) if row["category"] == "hard"]
        baseline_top1_correct = {
            str(rows[index]["line_sha256"]): bool(baseline_predictions[index])
            and int(baseline_predictions[index][0]) == int(rows[index]["expert_order"][0])
            for index in top1_guard_indices
        }
        baseline_checks = {
            "rows352": baseline["rows"] == 352,
            "hard96": len(hard_indices) == 96,
            "hard96_ordered_wrong": all(
                not baseline_ordered_correct[str(rows[index]["line_sha256"])]
                for index in hard_indices
            ),
            "retention160": len(original_retention_indices) == 160,
            "retention160_ordered_correct": all(
                baseline_ordered_correct[str(rows[index]["line_sha256"])]
                for index in original_retention_indices
            ),
            "top1_guard96": len(top1_guard_indices) == 96,
            "top1_guard96_top1_correct": all(baseline_top1_correct.values()),
            "top1_guard96_ordered_wrong": all(
                not baseline_ordered_correct[str(rows[index]["line_sha256"])]
                for index in top1_guard_indices
            ),
            "native_policy_BF16": outputs["policy_logits"].dtype == torch.bfloat16,
        }
        if not all(baseline_checks.values()):
            return {
                "decision": "NO_GO_CW24_BASELINE",
                "reason": "B352_BASELINE_CERTIFICATION_FAILED",
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline_checks": baseline_checks,
                "changed_candidate_train_shadow_count": 0,
                "candidate_payload": None,
            }

        row_by_sha = {str(row["line_sha256"]): index for index, row in enumerate(rows)}
        identity_checks = {
            "unique352": len(row_by_sha) == 352,
            "three_PF_targets": all(sha in row_by_sha for _, sha, _ in PF_TARGETS),
            "six_zero_guards": all(sha in row_by_sha for sha in ZERO_MARGIN_SHA256),
            "dominic_closest": DOMINIC_CLOSEST_SHA256 in row_by_sha,
        }
        if not all(identity_checks.values()):
            raise ProtocolError(f"fixed identity drift: {identity_checks}")

        logits = outputs["policy_logits"].float()
        parameter_tuple = tuple(parameters[name] for name in cw22.EXPECTED_ACTOR_NAMES)
        pf_objectives: list[Any] = []
        pf_names: list[str] = []
        threat_contract: dict[str, Any] = {}
        for name, sha, expected_margin in PF_TARGETS:
            index = row_by_sha[sha]
            objective, detail = cw23.fixed_threat_margin(
                logits[index], batch["option_mask"][index], rows[index]["expert_order"]
            )
            if float(objective.detach().cpu()) != expected_margin:
                raise ProtocolError(f"{name}: baseline margin drift")
            pf_objectives.append(objective)
            pf_names.append(name)
            threat_contract[name] = {"line_sha256": sha, "index": index, **detail}
        for offset, sha in enumerate(ZERO_MARGIN_SHA256):
            name = f"zero_margin_guard_{offset}"
            index = row_by_sha[sha]
            objective, detail = cw23.fixed_threat_margin(
                logits[index], batch["option_mask"][index], rows[index]["expert_order"]
            )
            if float(objective.detach().cpu()) != 0.0:
                raise ProtocolError(f"{name}: baseline margin drift")
            pf_objectives.append(objective)
            pf_names.append(name)
            threat_contract[name] = {"line_sha256": sha, "index": index, **detail}

        gradient_audit: dict[str, Any] = {}

        def flat_grad(name: str, objective: Any, retain_graph: bool) -> Any:
            values = torch.autograd.grad(
                objective,
                parameter_tuple,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
                materialize_grads=False,
            )
            flat = cw23.flat_gradient(values, cw22.EXPECTED_ACTOR_NAMES, np)
            norm = float(np.linalg.norm(flat))
            if not bool(np.isfinite(flat).all()) or norm <= 0.0:
                raise ProtocolError(f"{name}: invalid gradient")
            gradient_audit[name] = {
                "l2": norm,
                "max_abs": float(np.max(np.abs(flat))),
                "nonzero_elements": int(np.count_nonzero(flat)),
                "float64_le_sha256": array_sha(flat),
            }
            return flat

        pf_gradients = [
            flat_grad(name, objective, True)
            for name, objective in zip(pf_names, pf_objectives)
        ]
        matrix = np.stack(pf_gradients, axis=0)
        gram = matrix @ matrix.T
        gram_rank = int(np.linalg.matrix_rank(gram))
        gram_condition = float(np.linalg.cond(gram))
        pf_feasibility_checks = {
            "matrix_shape_9x65793": matrix.shape == (9, 65793),
            "rank9": gram_rank == 9,
            "condition_within_limit": gram_condition <= GRAM_CONDITION_LIMIT,
        }
        if not all(pf_feasibility_checks.values()):
            return {
                "decision": "NO_GO_CW24_PF_FEASIBILITY",
                "reason": "PF_BOUNDARY_GRAM_FAILED",
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline_checks": baseline_checks,
                "identity_checks": identity_checks,
                "gradient_audit": gradient_audit,
                "PF_feasibility": {
                    "checks": pf_feasibility_checks,
                    "rank": gram_rank,
                    "condition_number": gram_condition,
                    "gram_float64_le_sha256": array_sha(gram),
                },
                "changed_candidate_train_shadow_count": 0,
                "candidate_payload": None,
            }
        gram_pinv = np.linalg.pinv(gram, rcond=PINV_RCOND)
        requested = np.array(
            [
                TARGET_FINAL_MARGIN - expected_margin
                for _, _, expected_margin in PF_TARGETS
            ]
            + [0.0] * len(ZERO_MARGIN_SHA256),
            dtype=np.float64,
        )
        nominal_pf = matrix.T @ (gram_pinv @ requested)
        nominal_pf_l2 = float(np.linalg.norm(nominal_pf))
        if not math.isfinite(nominal_pf_l2) or nominal_pf_l2 <= 0.0:
            raise ProtocolError("invalid PF minimum-norm solution")
        pf_scale = min(1.0, PF_RADIUS_CAP / nominal_pf_l2)
        delta_pf = nominal_pf * pf_scale
        delta_pf_l2 = float(np.linalg.norm(delta_pf))

        per_row_nll = cw22.ordered_nll_per_row(outputs, batch)
        top1_ce = top1_ce_per_row(outputs, batch)
        dominic32_indices = [
            index for index, row in enumerate(rows) if row["stratum"] == "dominic_ctx0_hard"
        ]
        special9_indices = [index for index in dominic32_indices if cw22.is_dominic_special(rows[index])]
        szlach_indices = [
            index for index, row in enumerate(rows) if row["stratum"] == "top1_guard_szlach"
        ]
        core_other_indices = [
            index for index, row in enumerate(rows) if row["stratum"] == "top1_guard_core_other"
        ]
        if (len(dominic32_indices), len(special9_indices), len(szlach_indices), len(core_other_indices)) != (32, 9, 15, 18):
            raise ProtocolError("auxiliary stratum count drift")
        dominic_index = row_by_sha[DOMINIC_CLOSEST_SHA256]
        dominic_margin, dominic_detail = cw23.fixed_threat_margin(
            logits[dominic_index], batch["option_mask"][dominic_index], rows[dominic_index]["expert_order"]
        )
        if float(dominic_margin.detach().cpu()) != -1.0 / 32.0:
            raise ProtocolError("closest Dominic baseline margin drift")
        retention_tensor = torch.tensor(original_retention_indices, dtype=torch.long, device=device)
        retention_weights = batch["sample_weights"][retention_tensor].float()
        retention_loss = (
            per_row_nll[retention_tensor] * retention_weights
        ).sum() / retention_weights.sum().clamp_min(1.0)
        aux_objectives = {
            "dominic32_nll_descent": mean_at(per_row_nll, dominic32_indices, device),
            "dominic_closest_margin_ascent": dominic_margin,
            "special9_nll_descent": mean_at(per_row_nll, special9_indices, device),
            "szlach_top1_ce_descent": mean_at(top1_ce, szlach_indices, device),
            "core_other_top1_ce_descent": mean_at(top1_ce, core_other_indices, device),
            "retention160_nll_guard": retention_loss,
        }
        aux_gradients: dict[str, Any] = {}
        aux_names = list(aux_objectives)
        for offset, name in enumerate(aux_names):
            aux_gradients[name] = flat_grad(
                name,
                aux_objectives[name],
                offset + 1 < len(aux_names),
            )
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ProtocolError("autograd.grad materialized .grad buffers")

        def project_null(vector: Any) -> Any:
            return vector - matrix.T @ (gram_pinv @ (matrix @ vector))

        projected: dict[str, Any] = {}
        direction = np.zeros_like(delta_pf)
        for name, weight in AUX_WEIGHTS.items():
            value = project_null(aux_gradients[name])
            norm = float(np.linalg.norm(value))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ProtocolError(f"{name}: zero null-space projection")
            projected[name] = value
            sign = 1.0 if name.endswith("margin_ascent") else -1.0
            direction += sign * float(weight) * value / norm
        retention_projected = project_null(aux_gradients["retention160_nll_guard"])
        retention_square = float(retention_projected @ retention_projected)
        if not math.isfinite(retention_square) or retention_square <= 0.0:
            raise ProtocolError("retention null-space projection is zero")
        retention_dot_before = float(retention_projected @ direction)
        retention_projection_applied = retention_dot_before > 0.0
        if retention_projection_applied:
            direction = direction - (retention_dot_before / retention_square) * retention_projected
        retention_dot_after = float(retention_projected @ direction)
        direction_l2 = float(np.linalg.norm(direction))
        if not math.isfinite(direction_l2) or direction_l2 <= 0.0:
            raise ProtocolError("auxiliary direction is zero")
        aux_radius = math.sqrt(max(0.0, TOTAL_RADIUS_CAP**2 - delta_pf_l2**2))
        delta_aux = direction * (aux_radius / direction_l2)
        planned_delta = delta_pf + delta_aux
        planned_l2 = float(np.linalg.norm(planned_delta))
        geometry_checks = {
            "PF_nominal_residual_small": float(np.max(np.abs(matrix @ nominal_pf - requested))) <= 1e-9,
            "PF_radius_within_cap": delta_pf_l2 <= PF_RADIUS_CAP + 1e-15,
            "aux_in_PF_nullspace": float(np.max(np.abs(matrix @ delta_aux))) <= 1e-10,
            "PF_aux_orthogonal": abs(float(delta_pf @ delta_aux)) <= 1e-12,
            "retention_first_order_nondegrade": retention_dot_after <= 1e-12,
            "planned_total_radius": abs(planned_l2 - TOTAL_RADIUS_CAP) <= 1e-12,
            "weights_sum_one": abs(sum(AUX_WEIGHTS.values()) - 1.0) <= 1e-15,
        }
        if not all(geometry_checks.values()):
            raise ProtocolError(f"two-stage geometry failed: {geometry_checks}")

        cw20.apply_flat_actor(
            parameters,
            cw22.EXPECTED_ACTOR_NAMES,
            cw11_flat + delta_pf,
            torch,
        )
        with torch.no_grad():
            pf_only_outputs = ppo.model_forward(model, batch, device)
            pf_only_report, pf_only_correct, _ = cw22.evaluate_outputs(
                pf_only_outputs, batch, rows, ppo
            )
        pf_only_target_repairs = {
            name: bool(pf_only_correct[sha]) for name, sha, _ in PF_TARGETS
        }

        cw20.apply_flat_actor(
            parameters,
            cw22.EXPECTED_ACTOR_NAMES,
            cw11_flat + planned_delta,
            torch,
        )
        actual_flat = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
        actual_delta = actual_flat - cw11_flat
        actual_l2 = float(np.linalg.norm(actual_delta))
        candidate_hash = helper.model_state_sha256(model.state_dict())
        selected_actor_bytes = cw20.actor_bytes(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
        with torch.no_grad():
            candidate_outputs = ppo.model_forward(model, batch, device)
            candidate, candidate_ordered_correct, candidate_predictions = cw22.evaluate_outputs(
                candidate_outputs, batch, rows, ppo
            )
            candidate_nll = cw22.ordered_nll_per_row(candidate_outputs, batch)
            candidate_top1_ce = top1_ce_per_row(candidate_outputs, batch)
            immutable_native = {
                "count_logits_native_exact": bool(torch.equal(candidate_outputs["count_logits"], baseline_native["count_logits"])),
                "value_logits_native_exact": bool(torch.equal(candidate_outputs["value_logits"], baseline_native["value_logits"])),
                "policy_logits_native_BF16": candidate_outputs["policy_logits"].dtype == torch.bfloat16,
            }

        hard_repairs = [
            str(rows[index]["line_sha256"])
            for index in hard_indices
            if not baseline_ordered_correct[str(rows[index]["line_sha256"])]
            and candidate_ordered_correct[str(rows[index]["line_sha256"])]
        ]
        repairs_by_stratum = {
            stratum: sum(
                str(rows[index]["line_sha256"]) in hard_repairs
                for index in hard_indices
                if rows[index]["stratum"] == stratum
            )
            for stratum in ("pf_ctx0_hard", "pf_ctx7_hard", "dominic_ctx0_hard")
        }
        target_observed: dict[str, Any] = {}
        candidate_policy_cpu = candidate_outputs["policy_logits"].float().detach().cpu()
        for name, sha, _ in PF_TARGETS:
            detail = threat_contract[name]
            observed = float(
                candidate_policy_cpu[detail["index"], detail["expert"]]
                - candidate_policy_cpu[detail["index"], detail["fixed_threat"]]
            )
            target_observed[name] = {
                "line_sha256": sha,
                "baseline_margin": detail["margin"],
                "candidate_fixed_threat_margin": observed,
                "ordered_correct": candidate_ordered_correct[sha],
                "set_correct": set(candidate_predictions[detail["index"]])
                == set(int(value) for value in rows[detail["index"]]["expert_order"]),
            }
        zero_guard_observed = []
        for offset, sha in enumerate(ZERO_MARGIN_SHA256):
            detail = threat_contract[f"zero_margin_guard_{offset}"]
            observed = float(
                candidate_policy_cpu[detail["index"], detail["expert"]]
                - candidate_policy_cpu[detail["index"], detail["fixed_threat"]]
            )
            zero_guard_observed.append({
                "line_sha256": sha,
                "baseline_margin": detail["margin"],
                "candidate_fixed_threat_margin": observed,
                "ordered_correct": candidate_ordered_correct[sha],
                "stratum": rows[detail["index"]]["stratum"],
            })

        baseline_nll_cpu = per_row_nll.detach().cpu()
        candidate_nll_cpu = candidate_nll.detach().cpu()

        def nll_improvement(indices: Sequence[int]) -> float:
            return float(baseline_nll_cpu[list(indices)].mean() - candidate_nll_cpu[list(indices)].mean())

        hard_nll_improvements = {
            stratum: nll_improvement([
                index for index in hard_indices if rows[index]["stratum"] == stratum
            ])
            for stratum in ("pf_ctx0_hard", "pf_ctx7_hard", "dominic_ctx0_hard")
        }
        special9_improvement = nll_improvement(special9_indices)
        retention_improvement = nll_improvement(original_retention_indices)
        baseline_dominic_margin = float(dominic_margin.detach().cpu())
        dominic_observed_margin = float(
            candidate_policy_cpu[dominic_index, dominic_detail["expert"]]
            - candidate_policy_cpu[dominic_index, dominic_detail["fixed_threat"]]
        )
        top1_flips = []
        for index in top1_guard_indices:
            prediction = candidate_predictions[index]
            still_correct = bool(prediction) and int(prediction[0]) == int(rows[index]["expert_order"][0])
            if baseline_top1_correct[str(rows[index]["line_sha256"])] and not still_correct:
                top1_flips.append(str(rows[index]["line_sha256"]))
        retention_flips = [
            str(rows[index]["line_sha256"])
            for index in original_retention_indices
            if baseline_ordered_correct[str(rows[index]["line_sha256"])]
            and not candidate_ordered_correct[str(rows[index]["line_sha256"])]
        ]
        top1_ce_improvements = {
            "szlach15": float(top1_ce[szlach_indices].mean().detach().cpu() - candidate_top1_ce[szlach_indices].mean().cpu()),
            "core_other18": float(top1_ce[core_other_indices].mean().detach().cpu() - candidate_top1_ce[core_other_indices].mean().cpu()),
        }
        changed_names = [
            name for name in cw22.EXPECTED_ACTOR_NAMES if not torch.equal(parameters[name], cw11_actor[name])
        ]
        train_checks = {
            "PF_only_three_targets_repaired": all(pf_only_target_repairs.values()),
            "final_three_PF_targets_ordered_repaired": all(
                row["ordered_correct"] for row in target_observed.values()
            ),
            "final_three_PF_targets_set_repaired": all(
                row["set_correct"] for row in target_observed.values()
            ),
            "PF0_repairs_at_least2": repairs_by_stratum["pf_ctx0_hard"] >= 2,
            "PF7_repairs_at_least1": repairs_by_stratum["pf_ctx7_hard"] >= 1,
            "PF0_NLL_improvement_at_least_2e4": hard_nll_improvements["pf_ctx0_hard"] >= 2.0e-4,
            "PF7_NLL_improvement_at_least_3e4": hard_nll_improvements["pf_ctx7_hard"] >= 3.0e-4,
            "Dominic32_NLL_improvement_at_least_2p5e4": hard_nll_improvements["dominic_ctx0_hard"] >= 2.5e-4,
            "special9_NLL_improvement_at_least_8e5": special9_improvement >= 8.0e-5,
            "Dominic_closest_margin_improvement_at_least_1over512": dominic_observed_margin - baseline_dominic_margin >= 1.0 / 512.0,
            "top1_guard96_zero_flips": not top1_flips,
            "szlach15_mean_top1_CE_improves": top1_ce_improvements["szlach15"] > 0.0,
            "core_other18_mean_top1_CE_improves": top1_ce_improvements["core_other18"] > 0.0,
            "retention160_zero_ordered_flips": not retention_flips,
            "retention160_NLL_nondegrade": retention_improvement >= -1.0e-6,
            "six_zero_guards_stay_ordered_correct": all(row["ordered_correct"] for row in zero_guard_observed),
            "count_logits_native_exact": immutable_native["count_logits_native_exact"],
            "value_logits_native_exact": immutable_native["value_logits_native_exact"],
            "policy_logits_native_BF16": immutable_native["policy_logits_native_BF16"],
        }
        integrity = {
            "actual_additional_from_CW11_within_cap": actual_l2 <= TOTAL_RADIUS_CAP + 1e-12,
            "actual_delta_finite": bool(np.isfinite(actual_delta).all()),
            "candidate_changed": candidate_hash != cw22.CW11_MODEL_SHA256,
            "changed_scope_exact_actor6": set(changed_names) == set(cw22.EXPECTED_ACTOR_NAMES),
            "nonactor_exact_raw": cw20.nonactor_sha(model, cw22.EXPECTED_ACTOR_NAMES, helper)
            == cw22.RAW_NONACTOR_SHA256,
            "candidate_all_finite": all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()),
        }
        candidate_pass = all(train_checks.values()) and all(integrity.values())
        trial = {
            "candidate_model_state_sha256": candidate_hash,
            "actual_additional_from_CW11_l2": actual_l2,
            "actual_delta_float64_le_sha256": array_sha(actual_delta),
            "changed_parameter_names": changed_names,
            "integrity": integrity,
            "PF_only_target_repairs": pf_only_target_repairs,
            "PF_only_report": {
                "hard_ordered_correct": pf_only_report["hard"]["ordered_correct"],
                "hard_set_correct": pf_only_report["hard"]["set_correct"],
            },
            "target_observed": target_observed,
            "zero_guard_observed": zero_guard_observed,
            "hard_repairs": len(hard_repairs),
            "hard_repairs_by_stratum": repairs_by_stratum,
            "hard_repair_line_sha256": hard_repairs,
            "hard_exact_NLL_improvements": hard_nll_improvements,
            "special9_exact_NLL_improvement": special9_improvement,
            "Dominic_closest_margin": {
                "line_sha256": DOMINIC_CLOSEST_SHA256,
                "baseline": baseline_dominic_margin,
                "candidate": dominic_observed_margin,
                "improvement": dominic_observed_margin - baseline_dominic_margin,
            },
            "top1_guard_correct_to_wrong_count": len(top1_flips),
            "top1_guard_correct_to_wrong_line_sha256": top1_flips,
            "top1_CE_improvements": top1_ce_improvements,
            "retention160_correct_to_wrong_count": len(retention_flips),
            "retention160_correct_to_wrong_line_sha256": retention_flips,
            "retention160_exact_NLL_improvement": retention_improvement,
            "immutable_native": immutable_native,
            "train_checks": train_checks,
            "pass": candidate_pass,
        }
        geometry = {
            "PF_objective_order": pf_names,
            "PF_requested_linear_changes": requested.tolist(),
            "PF_gram_rank": gram_rank,
            "PF_gram_condition_number": gram_condition,
            "PF_gram_float64_le_sha256": array_sha(gram),
            "PF_nominal_l2": nominal_pf_l2,
            "PF_nominal_delta_float64_le_sha256": array_sha(nominal_pf),
            "PF_scale": pf_scale,
            "PF_planned_l2": delta_pf_l2,
            "PF_planned_delta_float64_le_sha256": array_sha(delta_pf),
            "PF_predicted_changes": (matrix @ delta_pf).tolist(),
            "aux_weights": AUX_WEIGHTS,
            "aux_radius": aux_radius,
            "aux_direction_pre_normalize_l2": direction_l2,
            "aux_delta_float64_le_sha256": array_sha(delta_aux),
            "retention_conflict_projection_applied": retention_projection_applied,
            "retention_dot_before": retention_dot_before,
            "retention_dot_after": retention_dot_after,
            "planned_total_l2": planned_l2,
            "planned_total_delta_float64_le_sha256": array_sha(planned_delta),
            "checks": geometry_checks,
            "single_candidate_only": True,
            "optimizer_instances_created": 0,
            "optimizer_step_calls": 0,
            "autograd_grad_calls": len(pf_objectives) + len(aux_objectives),
        }
        if not candidate_pass:
            return {
                "decision": "NO_GO_CW24_TWO_STAGE_TRAIN_GATE",
                "reason": "SOLE_TWO_STAGE_CANDIDATE_FAILED_TARGETED_TRAIN_GATE",
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline_checks": baseline_checks,
                "identity_checks": identity_checks,
                "threat_contract": threat_contract,
                "gradient_audit": gradient_audit,
                "two_stage_geometry": geometry,
                "trial": trial,
                "changed_candidate_train_shadow_count": 1,
                "candidate_payload": None,
            }

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
                "terminal_model_state_sha256": candidate_hash,
            },
            "formula": "load_original_raw_U468_then_replace_all_six_absolute_actor_float32_tensors_from_frozen_payload",
            "actor_names": list(cw22.EXPECTED_ACTOR_NAMES),
            "changed_actor_names": changed_names,
            "actor_layout": layout,
            "actor_layout_sha256": hashlib.sha256(
                json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "candidate_actor_float32_le": cw20.xz_payload(selected_actor_bytes),
            "actual_additional_from_CW11_l2": actual_l2,
            "two_stage_planned_delta_float64_le_sha256": array_sha(planned_delta),
            "selection_sha256": ROWS_CANONICAL_SHA256,
            "cache_sha256": cache_audit["cache_sha256"],
        }
        modules["cutting"].restore_raw_actor(
            modules["ram"], parameter_sequence, context["raw_actor"], torch
        )
        raw_restored_hash = helper.model_state_sha256(model.state_dict())
        decoded = cw20.decode_xz(payload["candidate_actor_float32_le"])
        cw20.copy_actor_bytes(
            parameters, cw22.EXPECTED_ACTOR_NAMES, layout, decoded, np, torch
        )
        reconstructed_hash = helper.model_state_sha256(model.state_dict())
        reconstruction_checks = {
            "raw_U468_restore_exact": raw_restored_hash == cw22.RAW_MODEL_SHA256,
            "candidate_hash_exact": reconstructed_hash == candidate_hash,
            "candidate_actor_bytes_exact": cw20.actor_bytes(
                parameters, cw22.EXPECTED_ACTOR_NAMES, np
            )
            == selected_actor_bytes,
            "nonactor_exact_raw": cw20.nonactor_sha(model, cw22.EXPECTED_ACTOR_NAMES, helper)
            == cw22.RAW_NONACTOR_SHA256,
        }
        final_checks = {
            "exact_CW11_context": all(context_checks.values()),
            "dependency_sources": all(dependency_checks.values()),
            "B352_baseline": all(baseline_checks.values()),
            "sole_two_stage_trial": trial["pass"],
            "absolute_payload_reconstruction": all(reconstruction_checks.values()),
        }
        passed = all(final_checks.values())
        return {
            "decision": "GO_CW24_TWO_STAGE_TRAIN_GATE" if passed else "NO_GO_CW24_TWO_STAGE_TRAIN_GATE",
            "reason": "SOLE_TWO_STAGE_CANDIDATE_PASSED_TARGETED_TRAIN_GATE"
            if passed
            else "ABSOLUTE_PAYLOAD_RECONSTRUCTION_FAILED",
            "context_checks": context_checks,
            "dependency_checks": dependency_checks,
            "cache": cache_audit,
            "baseline_checks": baseline_checks,
            "identity_checks": identity_checks,
            "threat_contract": threat_contract,
            "gradient_audit": gradient_audit,
            "two_stage_geometry": geometry,
            "trial": trial,
            "pure_payload_reconstruction": {
                "checks": reconstruction_checks,
                "pass": all(reconstruction_checks.values()),
                "raw_model_state_sha256": raw_restored_hash,
                "candidate_model_state_sha256": reconstructed_hash,
            },
            "final_checks": final_checks,
            "changed_candidate_train_shadow_count": 1,
            "candidate_payload": payload if passed else None,
        }
    finally:
        cw20.restore_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, cw11_actor, torch)
        for name, parameter in named.items():
            parameter.requires_grad_(requires_grad_before[name])
            parameter.grad = None


def production_run() -> dict[str, Any]:
    import numpy as np
    import torch

    source = self_evidence(require_frozen=True)
    rows, selection_audit = load_selection()
    cw23, cw23_evidence = load_locked_module(CW23, CW23_SHA256, CW23_MODE, "frozen_CW23")
    cw22, cw22_evidence = cw23.load_cw22()
    original = cw22.run_targeted_core
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
            raise ProtocolError("frozen engine B256 callback drift")
        return run_two_stage_core(
            context, rows, cw20, cw19, cw15, modules, cw22=cw22, cw23=cw23
        )

    cw22.run_targeted_core = bound_core
    cw22.SEED = SEED
    try:
        result = cw22.production_run()
    finally:
        cw22.run_targeted_core = original
        cw22.SEED = original_seed
    frozen_engine_selection = result["selection"]
    result["schema_version"] = SCHEMA
    result["status"] = result["endpoint"]["decision"]
    result["decision"] = result["endpoint"]["decision"]
    result["seed"] = SEED
    result["base"]["kind"] = "general_BC_plus_PPO_plus_exact_CW11_then_two_stage_top1_special_BC"
    result["selection"] = {
        "rows": 352,
        "sha256": ROWS_CANONICAL_SHA256,
        "checks": selection_audit["checks"],
        "counts": selection_audit["counts"],
        "scope": selection_audit["scope"],
        "optimization_contract": {
            "single_candidate_only": True,
            "PF_priority": "three fixed boundary margins plus six zero-margin guards",
            "PF_radius_cap": PF_RADIUS_CAP,
            "auxiliary_geometry": "weighted unit VJPs projected into PF nullspace then retention conflict projection",
            "auxiliary_weights": AUX_WEIGHTS,
            "total_additional_from_CW11_cap": TOTAL_RADIUS_CAP,
            "native_gate": "deterministic BF16 actions plus exact ordered PL NLL and top1 CE",
        },
        "frozen_engine_B256_reconstruction_selection": frozen_engine_selection,
    }
    result["historical_exact_CW11_replay"].pop(
        "new_validation_rows_opened_for_CW22_selection_or_candidate", None
    )
    result["historical_exact_CW11_replay"][
        "new_validation_rows_opened_for_CW24_selection_or_candidate"
    ] = 0
    result["inputs"]["CW24_source"] = source
    result["inputs"]["CW24_selector"] = selection_audit["selector"]
    result["inputs"]["CW24_selection"] = selection_audit["selection"]
    result["inputs"]["frozen_CW23_module"] = cw23_evidence
    result["inputs"]["frozen_CW22_engine"] = cw22_evidence
    result["audit"]["source"] = source
    result["audit"]["CW24_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "train_only": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "changed_candidate_train_shadow_count": result["endpoint"].get(
            "changed_candidate_train_shadow_count"
        ),
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "recipe_selected_after_CW23_specialist_failure": True,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        _, cw23_evidence = regular_source(CW23, CW23_SHA256, CW23_MODE, "frozen CW23")
        _, selector_evidence = regular_source(SELECTOR, SELECTOR_SHA256, 0o555, "CW24 selector")
        _, selection_evidence = regular_source(SELECTION, SELECTION_SHA256, 0o444, "CW24 selection")
        print(canonical_json({
            "schema_version": SCHEMA,
            "status": "static_audit_pass",
            "source": self_evidence(require_frozen=True),
            "CW23": cw23_evidence,
            "selector": selector_evidence,
            "selection": selection_evidence,
            "output_absent": not OUTPUT.exists(),
            "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
            "CUDA_initialized": False,
            "writes_performed": 0,
        }).decode(), end="")
        return
    output = args.output.resolve()
    if output != OUTPUT.resolve():
        raise ProtocolError("output path is not the frozen CW24 target")
    if OUTPUT.exists():
        raise ProtocolError("frozen CW24 output already exists")
    result = production_run()
    payload = canonical_json(result)
    publication = publish_o_excl(OUTPUT, payload)
    print(canonical_json({
        "schema_version": SCHEMA,
        "status": result["status"],
        "decision": result["decision"],
        "candidate_model_state_sha256": result["endpoint"].get("trial", {}).get(
            "candidate_model_state_sha256"
        ),
        "candidate_payload_present": result["endpoint"].get("candidate_payload") is not None,
        "changed_candidate_train_shadow_count": result["endpoint"].get(
            "changed_candidate_train_shadow_count"
        ),
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
        "output": publication,
    }).decode(), end="")


if __name__ == "__main__":
    main()

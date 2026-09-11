#!/usr/bin/env python3
"""CW24 v4 sequential native-BF16 cutting-plane train-only probe.

This wrapper reuses the frozen CW24 v1 reconstruction/cache/payload scaffold,
but replaces its one-shot two-stage update with a deterministic sequential
active-set method.  Every trial is replayed as exact CW11 + absolute actor6
delta, every accepted point is checked on the frozen train-only B352 panel,
and a payload is exposed only when the sole terminal endpoint passes every
native, loss, retention, scope, radius, and reconstruction gate.
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
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_native_cuttingplane_specialbc_cw24_v4.py"
ENGINE = TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py"
ENGINE_SHA256 = "a3c96eed95a6b558c61303e2190eaa4f353970add4509e8770c3dfcc1621db39"
ENGINE_MODE = 0o555
V1_RESULT = ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v1.json"
V1_RESULT_SHA256 = "9e3be911bfba95338538f315d47d3f7ce9f598e830ace607332d46697cb0cda5"
V2_RESULT = ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v2.json"
V2_RESULT_SHA256 = "36cae87fbaf4947f1caf1d6229a9d8ade75044eb25bc33398b731fd429d9e465"
V3_RESULT = ROOT / "artifacts/cw24_cw11_inequality_top1_specialbc_trainonly_v3.json"
V3_RESULT_SHA256 = "6a8affc789b3642d8ae4edf7a639f96526f7f211ad4899efb4b7d0fdfece3f34"
OUTPUT = ROOT / "artifacts/cw24_cw11_native_cuttingplane_specialbc_trainonly_v4.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-native-cuttingplane-specialbc-cw24-v4"
SEED = 202608044

# The hard user-facing cap is 1e-3.  Keep a deterministic 2e-8 planning and
# float32-copy buffer; the observed actor delta must also pass this tighter cap.
TOTAL_RADIUS_CAP = 9.9998e-4
TRUST_RADIUS_INITIAL = 2.5e-4
TRUST_RADIUS_MIN = 3.125e-5
TRUST_RADIUS_MAX = 3.0e-4
MAX_ACCEPTED_ITERATES = 16
MAX_TRIAL_EVALUATIONS = 48
MAX_ACTIVE_PAIR_CONSTRAINTS = 64
DUAL_MAX_SWEEPS = 20000
DUAL_TOL = 1.0e-11
KKT_TOL = 5.0e-9
MERIT_TOL = 1.0e-12

PF_FINAL_NATIVE_MARGIN = {
    "pf0_boundary_a": 1.0 / 512.0,
    "pf0_boundary_b": 1.0 / 512.0,
    "pf7_boundary": 0.0,
}
LOSS_TARGETS = {
    "pf0_nll_improvement": 2.0e-4,
    "pf7_nll_improvement": 3.0e-4,
    "dominic32_nll_improvement": 2.5e-4,
    "special9_nll_improvement": 8.0e-5,
    "retention160_nll_improvement": 0.0,
    "szlach15_top1_ce_improvement": 1.0e-7,
    "core_other18_top1_ce_improvement": 1.0e-7,
}
MERIT_SCALES = {
    "pf0_nll_improvement": 2.0e-4,
    "pf7_nll_improvement": 3.0e-4,
    "dominic32_nll_improvement": 2.5e-4,
    "special9_nll_improvement": 8.0e-5,
    "retention160_nll_improvement": 1.0e-5,
    "szlach15_top1_ce_improvement": 1.0e-5,
    "core_other18_top1_ce_improvement": 1.0e-5,
}

# These are not selected from a new sweep.  They are the sole native failures
# already disclosed by frozen CW24 v1-v3 and are pre-seeded as retention cuts.
KNOWN_RETENTION_FLIP_SHA256 = (
    "0142a2a3d5bdf5a9a7edc19761db708549701625055a07a4e90013884c752f43",
    "243f4a21d6c0b178a6306bdb63ac738aeb5bb13f6be0ccfaa68eebe02eac2d99",
)
KNOWN_TOP1_FLIP_SHA256 = (
    "fde6fab074495c238d4c4ef42a2f12ec595da78e35836578f7a3828002037b88",
)


class ProtocolError(RuntimeError):
    """Fail-closed CW24 v4 protocol error."""


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
        "single_terminal_payload_site": source.count(
            b'"candidate_' + b'payload": payload'
        )
        == 1,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
        "absolute_replay_present": b"cw11_flat + planned_trial_x" in source,
        "native_relinearization_present": b"build_linearized_constraints" in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v4 source audit failed: {checks}")
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
                raise ProtocolError("short v4 result write")
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
        raise ProtocolError(f"v4 publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def load_engine() -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = regular_source(ENGINE, ENGINE_SHA256, ENGINE_MODE, "frozen CW24 v1 engine")
    name = "cw24_v4_locked_v1_engine"
    if name in sys.modules:
        raise ProtocolError("v4 frozen engine module name occupied")
    module = types.ModuleType(name)
    module.__file__ = str(ENGINE)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, str(ENGINE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, evidence


def dual_hildreth_minimum_norm(matrix: Any, rhs: Any, np: Any) -> tuple[Any, dict[str, Any]]:
    """Solve min 0.5||p||^2 subject to matrix@p >= rhs.

    Rows are normalized by the caller.  Hildreth coordinate ascent solves the
    nonnegative dual and exposes the nonzero multipliers as the active set.
    """

    if matrix.ndim != 2 or rhs.shape != (matrix.shape[0],):
        raise ProtocolError("cutting-plane matrix shape drift")
    gram = matrix @ matrix.T
    diagonal = np.diag(gram)
    if not bool(np.isfinite(gram).all()) or float(np.min(diagonal)) <= 0.0:
        raise ProtocolError("cutting-plane Gram is invalid")
    multipliers = np.zeros(len(rhs), dtype=np.float64)
    converged = False
    sweeps = 0
    for sweep in range(DUAL_MAX_SWEEPS):
        max_change = 0.0
        for index in range(len(rhs)):
            residual = float(rhs[index] - gram[index] @ multipliers)
            updated = max(0.0, float(multipliers[index]) + residual / float(diagonal[index]))
            max_change = max(max_change, abs(updated - float(multipliers[index])))
            multipliers[index] = updated
        sweeps = sweep + 1
        if max_change <= DUAL_TOL:
            step = matrix.T @ multipliers
            slack = matrix @ step - rhs
            if float(np.min(slack)) >= -5.0e-9:
                converged = True
                break
    step = matrix.T @ multipliers
    slack = matrix @ step - rhs
    active = [int(index) for index in np.flatnonzero(multipliers > 1.0e-12)]
    active_condition = float(
        np.linalg.cond(gram[np.ix_(active, active)]) if active else 1.0
    )
    stationarity = step - matrix.T @ multipliers
    primal_objective = 0.5 * float(step @ step)
    dual_objective = float(rhs @ multipliers - 0.5 * multipliers @ gram @ multipliers)
    duality_gap = primal_objective - dual_objective
    complementarity_linf = float(np.max(np.abs(multipliers * slack)))
    kkt_checks = {
        "primal_feasible": float(np.min(slack)) >= -KKT_TOL,
        "dual_feasible": float(np.min(multipliers)) >= -KKT_TOL,
        "stationarity": float(np.max(np.abs(stationarity))) <= 1.0e-12,
        "complementarity": complementarity_linf <= KKT_TOL,
        "strong_duality": abs(duality_gap) <= KKT_TOL,
        "objectives_finite": math.isfinite(primal_objective)
        and math.isfinite(dual_objective)
        and math.isfinite(duality_gap),
    }
    certified = converged and all(kkt_checks.values())
    audit = {
        "certified": certified,
        "converged": converged,
        "sweeps": sweeps,
        "constraint_count": int(len(rhs)),
        "active_indices": active,
        "active_count": len(active),
        "minimum_slack": float(np.min(slack)),
        "step_l2": float(np.linalg.norm(step)),
        "gram_rank": int(np.linalg.matrix_rank(gram)),
        "gram_condition_active": active_condition if math.isfinite(active_condition) else None,
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "duality_gap": duality_gap,
        "stationarity_linf": float(np.max(np.abs(stationarity))),
        "complementarity_linf": complementarity_linf,
        "minimum_multiplier": float(np.min(multipliers)),
        "KKT_checks": kkt_checks,
        "multipliers_float64_le_sha256": hashlib.sha256(
            np.asarray(multipliers, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
    }
    if not bool(np.isfinite(step).all()):
        raise ProtocolError(f"cutting-plane dual produced nonfinite step: {audit}")
    return step, audit


def maximum_radius_alpha(x: Any, step: Any, radius: float, np: Any) -> float:
    quadratic = float(step @ step)
    if quadratic <= 0.0:
        return 0.0
    linear = 2.0 * float(x @ step)
    constant = float(x @ x) - radius**2
    discriminant = linear**2 - 4.0 * quadratic * constant
    if discriminant < -1.0e-24:
        raise ProtocolError("negative trust/radius intersection discriminant")
    root = (-linear + math.sqrt(max(0.0, discriminant))) / (2.0 * quadratic)
    return max(0.0, float(root))


def run_native_cuttingplane_core(
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cw20: ModuleType,
    cw19: ModuleType,
    cw15: ModuleType,
    modules: Mapping[str, ModuleType],
    *,
    cw22: ModuleType,
    cw23: ModuleType,
    engine: ModuleType,
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
    batch_cpu, cache_audit = engine.load_train_b352(
        rows, context["model_config"], helper.bc, ppo, cw22
    )
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
        with torch.no_grad():
            baseline_outputs = ppo.model_forward(model, batch, device)
            baseline, baseline_ordered_correct, baseline_predictions = cw22.evaluate_outputs(
                baseline_outputs, batch, rows, ppo
            )
            baseline_nll_tensor = cw22.ordered_nll_per_row(baseline_outputs, batch).detach()
            baseline_top1_tensor = engine.top1_ce_per_row(baseline_outputs, batch).detach()
        baseline_policy_cpu = baseline_outputs["policy_logits"].float().detach().cpu()
        baseline_native = {
            "count_logits": baseline_outputs["count_logits"].detach().clone(),
            "value_logits": baseline_outputs["value_logits"].detach().clone(),
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
            "native_policy_BF16": baseline_outputs["policy_logits"].dtype == torch.bfloat16,
        }
        if not all(baseline_checks.values()):
            return {
                "decision": "NO_GO_CW24_V4_BASELINE",
                "reason": "B352_BASELINE_CERTIFICATION_FAILED",
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline_checks": baseline_checks,
                "changed_candidate_train_shadow_count": 0,
                "candidate_payload": None,
            }

        row_by_sha = {str(row["line_sha256"]): index for index, row in enumerate(rows)}
        required_shas = (
            tuple(sha for _, sha, _ in engine.PF_TARGETS)
            + tuple(engine.ZERO_MARGIN_SHA256)
            + (engine.DOMINIC_CLOSEST_SHA256,)
            + KNOWN_RETENTION_FLIP_SHA256
            + KNOWN_TOP1_FLIP_SHA256
        )
        identity_checks = {
            "unique352": len(row_by_sha) == 352,
            "all_fixed_rows_present": all(sha in row_by_sha for sha in required_shas),
            "three_PF_targets": all(sha in row_by_sha for _, sha, _ in engine.PF_TARGETS),
            "three_PF_targets_single_action": all(
                len(rows[row_by_sha[sha]]["expert_order"]) == 1
                for _, sha, _ in engine.PF_TARGETS
            ),
            "six_zero_guards": all(sha in row_by_sha for sha in engine.ZERO_MARGIN_SHA256),
            "three_historical_native_failures": all(
                sha in row_by_sha
                for sha in KNOWN_RETENTION_FLIP_SHA256 + KNOWN_TOP1_FLIP_SHA256
            ),
        }
        if not all(identity_checks.values()):
            raise ProtocolError(f"v4 fixed identity drift: {identity_checks}")

        group_indices = {
            "pf0": [index for index in hard_indices if rows[index]["stratum"] == "pf_ctx0_hard"],
            "pf7": [index for index in hard_indices if rows[index]["stratum"] == "pf_ctx7_hard"],
            "dominic32": [
                index for index in hard_indices if rows[index]["stratum"] == "dominic_ctx0_hard"
            ],
            "szlach15": [
                index for index in top1_guard_indices if rows[index]["stratum"] == "top1_guard_szlach"
            ],
            "core_other18": [
                index
                for index in top1_guard_indices
                if rows[index]["stratum"] == "top1_guard_core_other"
            ],
            "retention160": list(original_retention_indices),
        }
        group_indices["special9"] = [
            index for index in group_indices["dominic32"] if cw22.is_dominic_special(rows[index])
        ]
        expected_group_sizes = {
            "pf0": 32,
            "pf7": 32,
            "dominic32": 32,
            "special9": 9,
            "szlach15": 15,
            "core_other18": 18,
            "retention160": 160,
        }
        observed_group_sizes = {name: len(indices) for name, indices in group_indices.items()}
        if observed_group_sizes != expected_group_sizes:
            raise ProtocolError(f"v4 group count drift: {observed_group_sizes}")

        baseline_nll_cpu = baseline_nll_tensor.detach().cpu()
        baseline_top1_cpu = baseline_top1_tensor.detach().cpu()
        baseline_means = {
            "pf0_nll_improvement": float(baseline_nll_cpu[group_indices["pf0"]].mean()),
            "pf7_nll_improvement": float(baseline_nll_cpu[group_indices["pf7"]].mean()),
            "dominic32_nll_improvement": float(
                baseline_nll_cpu[group_indices["dominic32"]].mean()
            ),
            "special9_nll_improvement": float(
                baseline_nll_cpu[group_indices["special9"]].mean()
            ),
            "retention160_nll_improvement": float(
                baseline_nll_cpu[group_indices["retention160"]].mean()
            ),
            "szlach15_top1_ce_improvement": float(
                baseline_top1_cpu[group_indices["szlach15"]].mean()
            ),
            "core_other18_top1_ce_improvement": float(
                baseline_top1_cpu[group_indices["core_other18"]].mean()
            ),
        }

        # Each pair constraint is an absolute native-BF16 margin target.  Rows
        # discovered by a rejected trial retain their exact CW11 pair margin.
        pair_specs: dict[str, dict[str, Any]] = {}

        def add_pair(
            name: str,
            index: int,
            expert: int,
            threat: int,
            target: float,
            source: str,
        ) -> bool:
            if expert == threat:
                raise ProtocolError(f"{name}: expert equals threat")
            existing = pair_specs.get(name)
            spec = {
                "name": name,
                "index": int(index),
                "line_sha256": str(rows[index]["line_sha256"]),
                "expert": int(expert),
                "threat": int(threat),
                "target": float(target),
                "source": source,
            }
            if existing is not None:
                if existing != spec:
                    raise ProtocolError(f"{name}: active-pair identity collision")
                return False
            equivalent = (
                int(index),
                int(expert),
                int(threat),
                float(target),
            )
            if any(
                (
                    int(value["index"]),
                    int(value["expert"]),
                    int(value["threat"]),
                    float(value["target"]),
                )
                == equivalent
                for value in pair_specs.values()
            ):
                return False
            if len(pair_specs) >= MAX_ACTIVE_PAIR_CONSTRAINTS:
                raise ProtocolError("active-pair cap exhausted")
            pair_specs[name] = spec
            return True

        baseline_logits_for_margin = baseline_outputs["policy_logits"].float()
        threat_contract: dict[str, Any] = {}
        for name, sha, expected_margin in engine.PF_TARGETS:
            index = row_by_sha[sha]
            objective, detail = cw23.fixed_threat_margin(
                baseline_logits_for_margin[index],
                batch["option_mask"][index],
                rows[index]["expert_order"],
            )
            observed = float(objective.detach().cpu())
            if observed != expected_margin:
                raise ProtocolError(f"{name}: baseline margin drift")
            add_pair(
                name,
                index,
                int(detail["expert"]),
                int(detail["fixed_threat"]),
                PF_FINAL_NATIVE_MARGIN[name],
                "fixed_PF_native_target",
            )
            threat_contract[name] = {"line_sha256": sha, "index": index, **detail}
        for offset, sha in enumerate(engine.ZERO_MARGIN_SHA256):
            name = f"zero_margin_guard_{offset}"
            index = row_by_sha[sha]
            objective, detail = cw23.fixed_threat_margin(
                baseline_logits_for_margin[index],
                batch["option_mask"][index],
                rows[index]["expert_order"],
            )
            observed = float(objective.detach().cpu())
            if observed != 0.0:
                raise ProtocolError(f"{name}: baseline zero-margin drift")
            add_pair(
                name,
                index,
                int(detail["expert"]),
                int(detail["fixed_threat"]),
                0.0,
                "fixed_zero_margin_guard",
            )
            threat_contract[name] = {"line_sha256": sha, "index": index, **detail}
        dominic_index = row_by_sha[engine.DOMINIC_CLOSEST_SHA256]
        dominic_objective, dominic_detail = cw23.fixed_threat_margin(
            baseline_logits_for_margin[dominic_index],
            batch["option_mask"][dominic_index],
            rows[dominic_index]["expert_order"],
        )
        baseline_dominic_margin = float(dominic_objective.detach().cpu())
        if baseline_dominic_margin != -1.0 / 32.0:
            raise ProtocolError("closest Dominic baseline margin drift")
        add_pair(
            "dominic_closest_margin",
            dominic_index,
            int(dominic_detail["expert"]),
            int(dominic_detail["fixed_threat"]),
            baseline_dominic_margin + 1.0 / 512.0,
            "fixed_Dominic_one_native_tick",
        )
        threat_contract["dominic_closest_margin"] = {
            "line_sha256": engine.DOMINIC_CLOSEST_SHA256,
            "index": dominic_index,
            **dominic_detail,
        }

        def strongest_baseline_threat(index: int, expert: int) -> int:
            valid = [
                int(candidate)
                for candidate in torch.nonzero(batch["option_mask"][index], as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
                if int(candidate) != int(expert)
            ]
            if not valid:
                raise ProtocolError("retention cut has no competing option")
            return max(valid, key=lambda candidate: (float(baseline_policy_cpu[index, candidate]), -candidate))

        for sha in KNOWN_RETENTION_FLIP_SHA256:
            index = row_by_sha[sha]
            expert = int(rows[index]["expert_order"][0])
            threat = strongest_baseline_threat(index, expert)
            target = float(baseline_policy_cpu[index, expert] - baseline_policy_cpu[index, threat])
            add_pair(
                f"historical_retention_{sha[:12]}",
                index,
                expert,
                threat,
                target,
                "frozen_v1_v3_native_flip_cut",
            )
        for sha in KNOWN_TOP1_FLIP_SHA256:
            index = row_by_sha[sha]
            expert = int(rows[index]["expert_order"][0])
            threat = strongest_baseline_threat(index, expert)
            target = float(baseline_policy_cpu[index, expert] - baseline_policy_cpu[index, threat])
            add_pair(
                f"historical_top1_{sha[:12]}",
                index,
                expert,
                threat,
                target,
                "frozen_v2_native_flip_cut",
            )

        def evaluate_outputs(candidate_outputs: Mapping[str, Any], candidate_x: Any) -> dict[str, Any]:
            report, ordered_correct, predictions = cw22.evaluate_outputs(
                candidate_outputs, batch, rows, ppo
            )
            candidate_nll = cw22.ordered_nll_per_row(candidate_outputs, batch).detach().cpu()
            candidate_top1 = engine.top1_ce_per_row(candidate_outputs, batch).detach().cpu()
            policy_cpu = candidate_outputs["policy_logits"].float().detach().cpu()
            pair_margins = {
                name: float(
                    policy_cpu[spec["index"], spec["expert"]]
                    - policy_cpu[spec["index"], spec["threat"]]
                )
                for name, spec in pair_specs.items()
            }
            loss_improvements = {
                "pf0_nll_improvement": baseline_means["pf0_nll_improvement"]
                - float(candidate_nll[group_indices["pf0"]].mean()),
                "pf7_nll_improvement": baseline_means["pf7_nll_improvement"]
                - float(candidate_nll[group_indices["pf7"]].mean()),
                "dominic32_nll_improvement": baseline_means["dominic32_nll_improvement"]
                - float(candidate_nll[group_indices["dominic32"]].mean()),
                "special9_nll_improvement": baseline_means["special9_nll_improvement"]
                - float(candidate_nll[group_indices["special9"]].mean()),
                "retention160_nll_improvement": baseline_means["retention160_nll_improvement"]
                - float(candidate_nll[group_indices["retention160"]].mean()),
                "szlach15_top1_ce_improvement": baseline_means["szlach15_top1_ce_improvement"]
                - float(candidate_top1[group_indices["szlach15"]].mean()),
                "core_other18_top1_ce_improvement": baseline_means[
                    "core_other18_top1_ce_improvement"
                ]
                - float(candidate_top1[group_indices["core_other18"]].mean()),
            }
            hard_repairs = [
                str(rows[index]["line_sha256"])
                for index in hard_indices
                if ordered_correct[str(rows[index]["line_sha256"])]
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
            for name, sha, _ in engine.PF_TARGETS:
                spec = pair_specs[name]
                prediction = predictions[spec["index"]]
                target_observed[name] = {
                    "line_sha256": sha,
                    "baseline_margin": float(threat_contract[name]["margin"]),
                    "required_native_margin": float(spec["target"]),
                    "candidate_fixed_threat_margin": pair_margins[name],
                    "ordered_correct": bool(ordered_correct[sha]),
                    "set_correct": set(prediction)
                    == set(int(value) for value in rows[spec["index"]]["expert_order"]),
                }
            zero_guard_observed = []
            for offset, sha in enumerate(engine.ZERO_MARGIN_SHA256):
                name = f"zero_margin_guard_{offset}"
                zero_guard_observed.append(
                    {
                        "line_sha256": sha,
                        "candidate_fixed_threat_margin": pair_margins[name],
                        "ordered_correct": bool(ordered_correct[sha]),
                        "stratum": rows[pair_specs[name]["index"]]["stratum"],
                    }
                )
            top1_flips = []
            for index in top1_guard_indices:
                prediction = predictions[index]
                still_correct = bool(prediction) and int(prediction[0]) == int(
                    rows[index]["expert_order"][0]
                )
                sha = str(rows[index]["line_sha256"])
                if baseline_top1_correct[sha] and not still_correct:
                    top1_flips.append(sha)
            retention_flips = [
                str(rows[index]["line_sha256"])
                for index in original_retention_indices
                if baseline_ordered_correct[str(rows[index]["line_sha256"])]
                and not ordered_correct[str(rows[index]["line_sha256"])]
            ]
            immutable_native = {
                "count_logits_native_exact": bool(
                    torch.equal(candidate_outputs["count_logits"], baseline_native["count_logits"])
                ),
                "value_logits_native_exact": bool(
                    torch.equal(candidate_outputs["value_logits"], baseline_native["value_logits"])
                ),
                "policy_logits_native_BF16": candidate_outputs["policy_logits"].dtype
                == torch.bfloat16,
            }
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
                "PF0_NLL_improvement_at_least_2e4": loss_improvements[
                    "pf0_nll_improvement"
                ]
                >= LOSS_TARGETS["pf0_nll_improvement"],
                "PF7_NLL_improvement_at_least_3e4": loss_improvements[
                    "pf7_nll_improvement"
                ]
                >= LOSS_TARGETS["pf7_nll_improvement"],
                "Dominic32_NLL_improvement_at_least_2p5e4": loss_improvements[
                    "dominic32_nll_improvement"
                ]
                >= LOSS_TARGETS["dominic32_nll_improvement"],
                "special9_NLL_improvement_at_least_8e5": loss_improvements[
                    "special9_nll_improvement"
                ]
                >= LOSS_TARGETS["special9_nll_improvement"],
                "Dominic_closest_margin_improvement_at_least_1over512": pair_margins[
                    "dominic_closest_margin"
                ]
                - baseline_dominic_margin
                >= 1.0 / 512.0,
                "top1_guard96_zero_flips": not top1_flips,
                "szlach15_mean_top1_CE_improves": loss_improvements[
                    "szlach15_top1_ce_improvement"
                ]
                > 0.0,
                "core_other18_mean_top1_CE_improves": loss_improvements[
                    "core_other18_top1_ce_improvement"
                ]
                > 0.0,
                "retention160_zero_ordered_flips": not retention_flips,
                "retention160_NLL_nondegrade": loss_improvements[
                    "retention160_nll_improvement"
                ]
                >= -1.0e-6,
                "six_zero_guards_stay_ordered_correct": all(
                    row["ordered_correct"] for row in zero_guard_observed
                ),
                "all_active_native_pair_thresholds": all(
                    pair_margins[name] >= spec["target"]
                    for name, spec in pair_specs.items()
                ),
                "count_logits_native_exact": immutable_native["count_logits_native_exact"],
                "value_logits_native_exact": immutable_native["value_logits_native_exact"],
                "policy_logits_native_BF16": immutable_native["policy_logits_native_BF16"],
            }
            core_deficits = {
                f"active_pair::{name}": max(0.0, float(spec["target"]) - pair_margins[name])
                / (1.0 / 512.0)
                for name, spec in pair_specs.items()
            }
            for name, target in LOSS_TARGETS.items():
                core_deficits[name] = max(0.0, target - loss_improvements[name]) / MERIT_SCALES[name]
            merit = float(sum(value * value for value in core_deficits.values()))
            return {
                "report": report,
                "ordered_correct": ordered_correct,
                "predictions": predictions,
                "policy_cpu": policy_cpu,
                "pair_margins": pair_margins,
                "loss_improvements": loss_improvements,
                "hard_repairs": hard_repairs,
                "hard_repairs_by_stratum": repairs_by_stratum,
                "target_observed": target_observed,
                "zero_guard_observed": zero_guard_observed,
                "top1_flips": sorted(top1_flips),
                "retention_flips": sorted(retention_flips),
                "immutable_native": immutable_native,
                "train_checks": train_checks,
                "core_deficits": core_deficits,
                "merit": merit,
                "actor_l2": float(np.linalg.norm(candidate_x)),
            }

        parameter_tuple = tuple(parameters[name] for name in cw22.EXPECTED_ACTOR_NAMES)

        def build_linearized_constraints() -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:
            outputs = ppo.model_forward(model, batch, device)
            if outputs["policy_logits"].dtype != torch.bfloat16:
                raise ProtocolError("native policy dtype drift during relinearization")
            with torch.no_grad():
                native_snapshot = ppo.model_forward(model, batch, device)
            graph_snapshot_checks = {
                "keys_exact": set(outputs) == set(native_snapshot),
                "all_tensor_values_bit_exact": set(outputs) == set(native_snapshot)
                and all(
                    isinstance(outputs[name], torch.Tensor)
                    and isinstance(native_snapshot[name], torch.Tensor)
                    and outputs[name].dtype == native_snapshot[name].dtype
                    and tuple(outputs[name].shape) == tuple(native_snapshot[name].shape)
                    and bool(torch.equal(outputs[name].detach(), native_snapshot[name]))
                    for name in outputs
                ),
                "policy_native_BF16": native_snapshot["policy_logits"].dtype
                == torch.bfloat16,
            }
            if not all(graph_snapshot_checks.values()):
                raise ProtocolError(
                    f"graph/no-grad same-point native snapshot drift: {graph_snapshot_checks}"
                )
            logits = outputs["policy_logits"].float()
            per_row_nll = cw22.ordered_nll_per_row(outputs, batch)
            top1_ce = engine.top1_ce_per_row(outputs, batch)
            objectives: list[tuple[str, Any, float, str]] = []
            for name, spec in pair_specs.items():
                objective = logits[spec["index"], spec["expert"]] - logits[
                    spec["index"], spec["threat"]
                ]
                objectives.append((name, objective, float(spec["target"]), "native_pair_margin"))

            loss_tensors = {
                "pf0_nll_improvement": baseline_means["pf0_nll_improvement"]
                - per_row_nll[group_indices["pf0"]].mean(),
                "pf7_nll_improvement": baseline_means["pf7_nll_improvement"]
                - per_row_nll[group_indices["pf7"]].mean(),
                "dominic32_nll_improvement": baseline_means["dominic32_nll_improvement"]
                - per_row_nll[group_indices["dominic32"]].mean(),
                "special9_nll_improvement": baseline_means["special9_nll_improvement"]
                - per_row_nll[group_indices["special9"]].mean(),
                "retention160_nll_improvement": baseline_means["retention160_nll_improvement"]
                - per_row_nll[group_indices["retention160"]].mean(),
                "szlach15_top1_ce_improvement": baseline_means[
                    "szlach15_top1_ce_improvement"
                ]
                - top1_ce[group_indices["szlach15"]].mean(),
                "core_other18_top1_ce_improvement": baseline_means[
                    "core_other18_top1_ce_improvement"
                ]
                - top1_ce[group_indices["core_other18"]].mean(),
            }
            for name in LOSS_TARGETS:
                objectives.append((name, loss_tensors[name], LOSS_TARGETS[name], "aggregate_improvement"))

            rows_normalized = []
            rhs_normalized = []
            details = []
            for offset, (name, objective, target, kind) in enumerate(objectives):
                gradients = torch.autograd.grad(
                    objective,
                    parameter_tuple,
                    retain_graph=offset + 1 < len(objectives),
                    create_graph=False,
                    allow_unused=False,
                    materialize_grads=False,
                )
                flat = cw23.flat_gradient(gradients, cw22.EXPECTED_ACTOR_NAMES, np)
                norm = float(np.linalg.norm(flat))
                current = float(objective.detach().cpu())
                if not bool(np.isfinite(flat).all()) or norm <= 0.0:
                    raise ProtocolError(f"{name}: invalid sequential gradient")
                rows_normalized.append(flat / norm)
                rhs_normalized.append((float(target) - current) / norm)
                details.append(
                    {
                        "name": name,
                        "kind": kind,
                        "current": current,
                        "target": float(target),
                        "gradient_l2": norm,
                        "gradient_float64_le_sha256": engine.array_sha(flat),
                    }
                )
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise ProtocolError("sequential autograd materialized .grad buffers")
            matrix = np.stack(rows_normalized, axis=0)
            rhs = np.asarray(rhs_normalized, dtype=np.float64)
            audit = {
                "constraint_count": len(details),
                "pair_constraint_count": len(pair_specs),
                "aggregate_constraint_count": len(LOSS_TARGETS),
                "constraint_semantics": (
                    "persistent active identities, current-point tangent only; "
                    "no global anchor-tangent retention claim"
                ),
                "graph_no_grad_same_point_checks": graph_snapshot_checks,
                "matrix_shape": list(matrix.shape),
                "matrix_float64_le_sha256": engine.array_sha(matrix),
                "rhs_float64_le_sha256": engine.array_sha(rhs),
                "constraints": details,
            }
            return matrix, rhs, details, audit

        def discover_failure_pairs(state: Mapping[str, Any]) -> list[dict[str, Any]]:
            discovered = []
            for name, sha, _ in engine.PF_TARGETS:
                observed = state["target_observed"][name]
                fixed_margin_cleared = (
                    float(observed["candidate_fixed_threat_margin"])
                    >= PF_FINAL_NATIVE_MARGIN[name]
                )
                if (
                    fixed_margin_cleared
                    and (not observed["ordered_correct"] or not observed["set_correct"])
                ):
                    index = row_by_sha[sha]
                    prediction = state["predictions"][index]
                    if not prediction:
                        raise ProtocolError("PF failure has empty native prediction")
                    expert = int(rows[index]["expert_order"][0])
                    threat = int(prediction[0])
                    if threat == expert:
                        raise ProtocolError(
                            "PF fixed threat cleared but native failure has no alternate threat"
                        )
                    discovered.append(
                        {
                            "name": f"dynamic_PF_{name}_{sha[:12]}_e{expert}_t{threat}",
                            "index": index,
                            "expert": expert,
                            "threat": threat,
                            "target": PF_FINAL_NATIVE_MARGIN[name],
                            "source": "rejected_native_PF_alternate_threat",
                        }
                    )
            for sha in state["top1_flips"]:
                index = row_by_sha[sha]
                prediction = state["predictions"][index]
                if not prediction:
                    raise ProtocolError("top1 failure has empty prediction")
                expert = int(rows[index]["expert_order"][0])
                threat = int(prediction[0])
                target = float(
                    baseline_policy_cpu[index, expert] - baseline_policy_cpu[index, threat]
                )
                discovered.append(
                    {
                        "name": f"dynamic_top1_{sha[:12]}_e{expert}_t{threat}",
                        "index": index,
                        "expert": expert,
                        "threat": threat,
                        "target": target,
                        "source": "rejected_native_top1_flip",
                    }
                )
            for sha in state["retention_flips"]:
                index = row_by_sha[sha]
                prediction = list(state["predictions"][index])
                expert_order = [int(value) for value in rows[index]["expert_order"]]
                mismatch = next(
                    (
                        step
                        for step, (expert, observed) in enumerate(zip(expert_order, prediction))
                        if int(expert) != int(observed)
                    ),
                    None,
                )
                if mismatch is None:
                    raise ProtocolError("retention failure lacks a divergent native step")
                expert = expert_order[mismatch]
                threat = int(prediction[mismatch])
                target = float(
                    baseline_policy_cpu[index, expert] - baseline_policy_cpu[index, threat]
                )
                discovered.append(
                    {
                        "name": f"dynamic_retention_{sha[:12]}_s{mismatch}_e{expert}_t{threat}",
                        "index": index,
                        "expert": expert,
                        "threat": threat,
                        "target": target,
                        "source": "rejected_native_ordered_flip",
                    }
                )
            return sorted(discovered, key=lambda item: item["name"])

        current_x = np.zeros_like(cw11_flat, dtype=np.float64)
        with torch.no_grad():
            current_outputs = ppo.model_forward(model, batch, device)
        current_state = evaluate_outputs(current_outputs, current_x)
        trust_radius = TRUST_RADIUS_INITIAL
        accepted_iterates = 0
        trial_evaluations = 0
        rejection_count = 0
        cutting_plane_additions = 0
        iteration_log: list[dict[str, Any]] = []
        terminal_solver_audit: dict[str, Any] | None = None
        terminal_reason = "MAX_TRIAL_EVALUATIONS_EXHAUSTED"

        while trial_evaluations < MAX_TRIAL_EVALUATIONS:
            if all(current_state["train_checks"].values()):
                terminal_reason = "ALL_TARGETED_TRAIN_GATES_PASSED"
                break
            if accepted_iterates >= MAX_ACCEPTED_ITERATES:
                terminal_reason = "MAX_ACCEPTED_ITERATES_EXHAUSTED"
                break
            matrix, rhs, _, linearization_audit = build_linearized_constraints()
            step, solver_audit = dual_hildreth_minimum_norm(matrix, rhs, np)
            if not solver_audit["certified"]:
                terminal_solver_audit = solver_audit
                terminal_reason = "CURRENT_TANGENT_SOLVER_UNCERTIFIED"
                iteration_log.append(
                    {
                        "trial_one_based": trial_evaluations + 1,
                        "accepted_before_trial": accepted_iterates,
                        "trust_radius": trust_radius,
                        "candidate_evaluated": False,
                        "accepted": False,
                        "terminal_solver_failure": True,
                        "linearization": linearization_audit,
                        "solver": solver_audit,
                    }
                )
                break
            step_l2 = float(np.linalg.norm(step))
            if step_l2 <= 1.0e-15:
                terminal_reason = "ZERO_SEQUENTIAL_STEP_BEFORE_GATE_PASS"
                break
            alpha_trust = min(1.0, trust_radius / step_l2)
            alpha_global = min(1.0, maximum_radius_alpha(current_x, step, TOTAL_RADIUS_CAP, np))
            alpha = min(alpha_trust, alpha_global)
            if alpha <= 1.0e-12:
                terminal_reason = "CW11_RADIUS_EXHAUSTED_BEFORE_GATE_PASS"
                break
            planned_trial_x = current_x + alpha * step
            if float(np.linalg.norm(planned_trial_x)) > TOTAL_RADIUS_CAP + 1.0e-15:
                raise ProtocolError("planned sequential trial exceeds radius cap")
            cw20.apply_flat_actor(
                parameters,
                cw22.EXPECTED_ACTOR_NAMES,
                cw11_flat + planned_trial_x,
                torch,
            )
            actual_trial_x = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np) - cw11_flat
            actual_trial_l2 = float(np.linalg.norm(actual_trial_x))
            with torch.no_grad():
                trial_outputs = ppo.model_forward(model, batch, device)
            trial_state = evaluate_outputs(trial_outputs, actual_trial_x)
            trial_evaluations += 1

            discovered = discover_failure_pairs(trial_state)
            new_pair_names = []
            for item in discovered:
                if add_pair(**item):
                    new_pair_names.append(item["name"])
            cutting_plane_additions += len(new_pair_names)
            preservation_failed = bool(trial_state["top1_flips"] or trial_state["retention_flips"])
            immutable_failed = not all(trial_state["immutable_native"].values())
            radius_failed = actual_trial_l2 > TOTAL_RADIUS_CAP
            merit_reduction = float(current_state["merit"] - trial_state["merit"])
            gate_pass = all(trial_state["train_checks"].values())
            accept = (
                not preservation_failed
                and not immutable_failed
                and not radius_failed
                and not new_pair_names
                and (gate_pass or merit_reduction > MERIT_TOL)
            )
            log_entry = {
                "trial_one_based": trial_evaluations,
                "accepted_before_trial": accepted_iterates,
                "trust_radius": trust_radius,
                "alpha": alpha,
                "alpha_trust": alpha_trust,
                "alpha_global": alpha_global,
                "planned_trial_l2": float(np.linalg.norm(planned_trial_x)),
                "actual_trial_l2": actual_trial_l2,
                "current_merit": float(current_state["merit"]),
                "trial_merit": float(trial_state["merit"]),
                "merit_reduction": merit_reduction,
                "PF_repairs": {
                    name: bool(trial_state["target_observed"][name]["ordered_correct"])
                    for name in PF_FINAL_NATIVE_MARGIN
                },
                "top1_flip_count": len(trial_state["top1_flips"]),
                "retention_flip_count": len(trial_state["retention_flips"]),
                "new_pair_names": new_pair_names,
                "accepted": accept,
                "gate_pass": gate_pass,
                "linearization": linearization_audit,
                "solver": solver_audit,
            }
            iteration_log.append(log_entry)
            if accept:
                current_x = actual_trial_x
                current_state = trial_state
                accepted_iterates += 1
                if gate_pass:
                    terminal_reason = "ALL_TARGETED_TRAIN_GATES_PASSED"
                    break
                reduction_fraction = merit_reduction / max(
                    float(log_entry["current_merit"]), MERIT_TOL
                )
                if reduction_fraction >= 0.25 and not new_pair_names:
                    trust_radius = min(TRUST_RADIUS_MAX, trust_radius * 1.5)
            else:
                rejection_count += 1
                trust_radius *= 0.5
                cw20.apply_flat_actor(
                    parameters,
                    cw22.EXPECTED_ACTOR_NAMES,
                    cw11_flat + current_x,
                    torch,
                )
                restored_x = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np) - cw11_flat
                if not bool(np.array_equal(restored_x, current_x)):
                    raise ProtocolError("absolute CW11+x rejected-trial restore drift")
                with torch.no_grad():
                    restored_outputs = ppo.model_forward(model, batch, device)
                current_state = evaluate_outputs(restored_outputs, restored_x)
                if trust_radius < TRUST_RADIUS_MIN:
                    terminal_reason = "TRUST_RADIUS_FLOOR_REACHED"
                    break

        passed = all(current_state["train_checks"].values())
        if not passed:
            return {
                "decision": "NO_GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE",
                "reason": terminal_reason,
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline_checks": baseline_checks,
                "identity_checks": identity_checks,
                "threat_contract": threat_contract,
                "cuttingplane": {
                    "total_radius_cap": TOTAL_RADIUS_CAP,
                    "accepted_iterates": accepted_iterates,
                    "trial_evaluations": trial_evaluations,
                    "rejection_count": rejection_count,
                    "cutting_plane_additions": cutting_plane_additions,
                    "active_pair_count": len(pair_specs),
                    "active_pairs": list(pair_specs.values()),
                    "terminal_reason": terminal_reason,
                    "terminal_solver_audit": terminal_solver_audit,
                    "iteration_log": iteration_log,
                    "optimizer_instances_created": 0,
                    "optimizer_step_calls": 0,
                },
                "trial": {
                    "actual_additional_from_CW11_l2": float(np.linalg.norm(current_x)),
                    "target_observed": current_state["target_observed"],
                    "hard_repairs": len(current_state["hard_repairs"]),
                    "hard_repairs_by_stratum": current_state["hard_repairs_by_stratum"],
                    "hard_repair_line_sha256": current_state["hard_repairs"],
                    "loss_improvements": current_state["loss_improvements"],
                    "top1_guard_correct_to_wrong_line_sha256": current_state["top1_flips"],
                    "retention160_correct_to_wrong_line_sha256": current_state[
                        "retention_flips"
                    ],
                    "immutable_native": current_state["immutable_native"],
                    "train_checks": current_state["train_checks"],
                    "pass": False,
                },
                "changed_candidate_train_shadow_count": trial_evaluations,
                "promotable_terminal_endpoint_count": 0,
                "candidate_payload": None,
            }

        # Replay the sole terminal endpoint from CW11 + absolute x.  This is the
        # only point for which actor bytes, model hash, or a payload are exposed.
        cw20.apply_flat_actor(
            parameters,
            cw22.EXPECTED_ACTOR_NAMES,
            cw11_flat + current_x,
            torch,
        )
        replay_x = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np) - cw11_flat
        with torch.no_grad():
            replay_outputs = ppo.model_forward(model, batch, device)
        replay_state = evaluate_outputs(replay_outputs, replay_x)
        replay_checks = {
            "absolute_actor_delta_exact": bool(np.array_equal(replay_x, current_x)),
            "all_train_checks_repeat": all(replay_state["train_checks"].values()),
            "target_observed_repeat": canonical_json(replay_state["target_observed"])
            == canonical_json(current_state["target_observed"]),
            "loss_improvements_repeat": canonical_json(replay_state["loss_improvements"])
            == canonical_json(current_state["loss_improvements"]),
            "top1_flips_repeat_zero": not replay_state["top1_flips"],
            "retention_flips_repeat_zero": not replay_state["retention_flips"],
            "actual_radius_within_cap": float(np.linalg.norm(replay_x)) <= TOTAL_RADIUS_CAP,
        }
        if not all(replay_checks.values()):
            raise ProtocolError(f"terminal absolute replay failed: {replay_checks}")

        actual_l2 = float(np.linalg.norm(replay_x))
        candidate_hash = helper.model_state_sha256(model.state_dict())
        selected_actor_bytes = cw20.actor_bytes(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
        changed_names = [
            name
            for name in cw22.EXPECTED_ACTOR_NAMES
            if not torch.equal(parameters[name], cw11_actor[name])
        ]
        integrity = {
            "actual_additional_from_CW11_within_cap": actual_l2 <= TOTAL_RADIUS_CAP,
            "actual_delta_finite": bool(np.isfinite(replay_x).all()),
            "candidate_changed": candidate_hash != cw22.CW11_MODEL_SHA256,
            "changed_scope_exact_actor6": set(changed_names) == set(cw22.EXPECTED_ACTOR_NAMES),
            "nonactor_exact_raw": cw20.nonactor_sha(model, cw22.EXPECTED_ACTOR_NAMES, helper)
            == cw22.RAW_NONACTOR_SHA256,
            "candidate_all_finite": all(
                bool(torch.isfinite(value).all()) for value in model.state_dict().values()
            ),
        }
        if not all(integrity.values()):
            raise ProtocolError(f"terminal candidate integrity failed: {integrity}")
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
            "native_cuttingplane_delta_float64_le_sha256": engine.array_sha(replay_x),
            "selection_sha256": engine.ROWS_CANONICAL_SHA256,
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
            "terminal_train_gate": all(replay_state["train_checks"].values()),
            "terminal_integrity": all(integrity.values()),
            "absolute_CW11_replay": all(replay_checks.values()),
            "absolute_payload_reconstruction": all(reconstruction_checks.values()),
            "sole_promotable_endpoint": True,
        }
        final_pass = all(final_checks.values())
        trial = {
            "candidate_model_state_sha256": candidate_hash,
            "actual_additional_from_CW11_l2": actual_l2,
            "actual_delta_float64_le_sha256": engine.array_sha(replay_x),
            "changed_parameter_names": changed_names,
            "integrity": integrity,
            "target_observed": replay_state["target_observed"],
            "zero_guard_observed": replay_state["zero_guard_observed"],
            "hard_repairs": len(replay_state["hard_repairs"]),
            "hard_repairs_by_stratum": replay_state["hard_repairs_by_stratum"],
            "hard_repair_line_sha256": replay_state["hard_repairs"],
            "loss_improvements": replay_state["loss_improvements"],
            "Dominic_closest_margin": {
                "line_sha256": engine.DOMINIC_CLOSEST_SHA256,
                "baseline": baseline_dominic_margin,
                "candidate": replay_state["pair_margins"]["dominic_closest_margin"],
                "improvement": replay_state["pair_margins"]["dominic_closest_margin"]
                - baseline_dominic_margin,
            },
            "top1_guard_correct_to_wrong_count": len(replay_state["top1_flips"]),
            "top1_guard_correct_to_wrong_line_sha256": replay_state["top1_flips"],
            "retention160_correct_to_wrong_count": len(replay_state["retention_flips"]),
            "retention160_correct_to_wrong_line_sha256": replay_state["retention_flips"],
            "immutable_native": replay_state["immutable_native"],
            "train_checks": replay_state["train_checks"],
            "pass": final_pass,
        }
        return {
            "decision": "GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE"
            if final_pass
            else "NO_GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE",
            "reason": "SOLE_NATIVE_CUTTINGPLANE_ENDPOINT_PASSED_TARGETED_TRAIN_GATE"
            if final_pass
            else "SOLE_NATIVE_CUTTINGPLANE_ENDPOINT_RECONSTRUCTION_FAILED",
            "context_checks": context_checks,
            "dependency_checks": dependency_checks,
            "cache": cache_audit,
            "baseline_checks": baseline_checks,
            "identity_checks": identity_checks,
            "threat_contract": threat_contract,
            "cuttingplane": {
                "total_radius_cap": TOTAL_RADIUS_CAP,
                "accepted_iterates": accepted_iterates,
                "trial_evaluations": trial_evaluations,
                "rejection_count": rejection_count,
                "cutting_plane_additions": cutting_plane_additions,
                "active_pair_count": len(pair_specs),
                "active_pairs": list(pair_specs.values()),
                "terminal_reason": terminal_reason,
                "terminal_solver_audit": terminal_solver_audit,
                "iteration_log": iteration_log,
                "optimizer_instances_created": 0,
                "optimizer_step_calls": 0,
            },
            "trial": trial,
            "terminal_absolute_replay": {
                "checks": replay_checks,
                "pass": all(replay_checks.values()),
            },
            "pure_payload_reconstruction": {
                "checks": reconstruction_checks,
                "pass": all(reconstruction_checks.values()),
                "raw_model_state_sha256": raw_restored_hash,
                "candidate_model_state_sha256": reconstructed_hash,
            },
            "final_checks": final_checks,
            "changed_candidate_train_shadow_count": trial_evaluations,
            "promotable_terminal_endpoint_count": 1 if final_pass else 0,
            "candidate_payload": payload if final_pass else None,
        }
    finally:
        cw20.restore_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, cw11_actor, torch)
        for name, parameter in named.items():
            parameter.requires_grad_(requires_grad_before[name])
            parameter.grad = None


def production_run() -> dict[str, Any]:
    source = self_evidence(require_frozen=True)
    _, v1_evidence = regular_source(V1_RESULT, V1_RESULT_SHA256, 0o444, "frozen v1 result")
    _, v2_evidence = regular_source(V2_RESULT, V2_RESULT_SHA256, 0o444, "frozen v2 result")
    _, v3_evidence = regular_source(V3_RESULT, V3_RESULT_SHA256, 0o444, "frozen v3 result")
    engine, engine_evidence = load_engine()
    original_core = engine.run_two_stage_core
    original_seed = engine.SEED

    def bound_v4_core(
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
        return run_native_cuttingplane_core(
            context,
            rows,
            cw20,
            cw19,
            cw15,
            modules,
            cw22=cw22,
            cw23=cw23,
            engine=engine,
        )

    engine.run_two_stage_core = bound_v4_core
    engine.SEED = SEED
    try:
        result = engine.production_run()
    finally:
        engine.run_two_stage_core = original_core
        engine.SEED = original_seed
    endpoint = result["endpoint"]
    payload_present = endpoint.get("candidate_payload") is not None
    payload_gate = (
        payload_present
        == (
            endpoint.get("decision") == "GO_CW24_V4_NATIVE_CUTTINGPLANE_TRAIN_GATE"
            and endpoint.get("trial", {}).get("pass") is True
            and endpoint.get("promotable_terminal_endpoint_count") == 1
        )
    )
    if not payload_gate:
        raise ProtocolError("v4 terminal-only payload gate drift")
    result["schema_version"] = SCHEMA
    result["status"] = endpoint["decision"]
    result["decision"] = endpoint["decision"]
    result["seed"] = SEED
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_native_cuttingplane_special_BC"
    )
    result["selection"]["optimization_contract"] = {
        "single_promotable_terminal_endpoint": True,
        "native_BF16_relinearized_each_accepted_iteration": True,
        "absolute_CW11_plus_x_replay_every_trial": True,
        "PF_native_targets": PF_FINAL_NATIVE_MARGIN,
        "Dominic_closest_native_improvement": 1.0 / 512.0,
        "aggregate_improvement_targets": LOSS_TARGETS,
        "top1_and_retention_flips_rejected_and_added_as_pair_cuts": True,
        "total_additional_from_CW11_cap": TOTAL_RADIUS_CAP,
        "payload_only_after_all_terminal_train_gates": True,
    }
    result["historical_exact_CW11_replay"].pop(
        "new_validation_rows_opened_for_CW24_selection_or_candidate", None
    )
    result["historical_exact_CW11_replay"][
        "new_validation_rows_opened_for_CW24_v4_selection_or_candidate"
    ] = 0
    result["inputs"]["CW24_v4_source"] = source
    result["inputs"]["CW24_v4_frozen_v1_engine"] = engine_evidence
    result["inputs"]["CW24_v1_NO_GO_result"] = v1_evidence
    result["inputs"]["CW24_v2_NO_GO_result"] = v2_evidence
    result["inputs"]["CW24_v3_NO_GO_result"] = v3_evidence
    result["audit"]["CW24_v4_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "train_only_B352": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 0,
        "optimizer_step_calls": 0,
        "total_radius_cap": TOTAL_RADIUS_CAP,
        "terminal_only_payload_gate": payload_gate,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "v4_recipe_selected_after_frozen_v1_v2_v3_train_failures": True,
        "internal_iterates_are_not_separately_promotable_endpoints": True,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        _, engine_evidence = regular_source(
            ENGINE, ENGINE_SHA256, ENGINE_MODE, "frozen CW24 v1 engine"
        )
        previous = []
        for path, digest, label in (
            (V1_RESULT, V1_RESULT_SHA256, "v1"),
            (V2_RESULT, V2_RESULT_SHA256, "v2"),
            (V3_RESULT, V3_RESULT_SHA256, "v3"),
        ):
            _, evidence = regular_source(path, digest, 0o444, f"frozen {label} result")
            previous.append(evidence)
        print(
            canonical_json(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_pass",
                    "source": self_evidence(require_frozen=True),
                    "engine": engine_evidence,
                    "previous_NO_GO_results": previous,
                    "variant": {
                        "total_radius_cap": TOTAL_RADIUS_CAP,
                        "trust_radius_initial": TRUST_RADIUS_INITIAL,
                        "trust_radius_min": TRUST_RADIUS_MIN,
                        "trust_radius_max": TRUST_RADIUS_MAX,
                        "PF_final_native_margin": PF_FINAL_NATIVE_MARGIN,
                        "loss_targets": LOSS_TARGETS,
                    },
                    "output_absent": not OUTPUT.exists(),
                    "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
                    "CUDA_initialized": False,
                    "validation_or_test_rows_opened": 0,
                    "writes_performed": 0,
                }
            ).decode(),
            end="",
        )
        return
    if OUTPUT.exists():
        raise ProtocolError("frozen CW24 v4 output already exists")
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

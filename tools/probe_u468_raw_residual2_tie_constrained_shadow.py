#!/usr/bin/env python3
"""RAM-only constrained repair probe for raw U468 policy ties.

The probe changes only ``actor_residual.2.weight`` in memory.  It builds a
minimum-norm direction whose native-BF16 first-order margin is positive for
all seven train rows that are wrong only because an expert action ties the
best competitor, while also making every tied margin positive on all 17
zero-margin train rows that raw U468 currently predicts correctly.

The direction is solved once in CPU float64 from a canonical least-squares
system.  Every endpoint is reconstructed directly from raw FP32 weights at a
fixed displacement grid; endpoints are not a training trajectory.  The probe
never creates an optimizer, calls backward, saves a checkpoint, writes an
artifact, opens a validation member, or performs a submission.
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
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
AGGREGATE_RUNNER = TOOLS / "run_u468_raw_actor6_aggregate512_shadow.py"
AGGREGATE_RUNNER_SHA256 = (
    "db7d6ca06b83d9d035e42476a4282c0ab25b8d532a589f3be80bb2299748b641"
)
PROFILE = ROOT / "artifacts/u468_raw_full_train_margin_profile_v3_20260802.json"
PROFILE_SHA256 = (
    "17af98e175167398a80cd2e1f8930c3d1d85cfe6ceea3143dfe2ba44dc89f1f9"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(
    path: Path, expected_sha256: str | None, label: str
) -> dict[str, Any]:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise RuntimeError(f"{label} is not a single-link regular file")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


# Authenticate the only imported project runner before executing its code.
AGGREGATE_EVIDENCE = require_regular(
    AGGREGATE_RUNNER, AGGREGATE_RUNNER_SHA256, "frozen aggregate512 runner"
)
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_aggregate512_db7d6ca0", AGGREGATE_RUNNER
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot construct aggregate512 import spec")
aggregate: ModuleType = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aggregate)

torch = aggregate.torch
orjson = aggregate.orjson
frozen = aggregate.frozen
ppo = aggregate.ppo
bc = aggregate.bc
repair = aggregate.repair


SCHEMA = "ptcg-u468-raw-residual2-tie-constrained-shadow-v1"
EXECUTION_SEED = 202608123
EDITED_PARAMETER = "actor_residual.2.weight"
REFERENCE_STEP_L2 = 7.44866428525324e-6
ALPHAS = (1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256)
SOLVER_ABS_RESIDUAL_MAX = 1e-9
LOSS_ABS_TOLERANCE = 1e-7

EXPECTED_TARGET_HASHES = {
    "core5": (
        "243f4a21d6c0b178a6306bdb63ac738aeb5bb13f6be0ccfaa68eebe02eac2d99",
    ),
    "flg": (
        "3de5aa54c1f507c59f90983f5e0d4efb6a83fc12cb0cc227fb82fca2c4823b2e",
        "9951f9647d87b43b035d6360842639667224016c8b0796313dd32cd6b254df3f",
        "ef65da0c418f3ce48dc47a71458cc3971f037262ee47dc9229eaa46d24727213",
    ),
    "pokemonfan": (
        "0142a2a3d5bdf5a9a7edc19761db708549701625055a07a4e90013884c752f43",
        "4e507ccfd5a370567adc775a86786c1adb404c5c56a40158adfc82059090c055",
        "d456bafc089282ed4053d9ad1d0bb795be0366c3256f6d3992fea6937fadfb50",
    ),
}
EXPECTED_GUARD_HASHES = {
    "core5": (
        "002afbcdd4f248353808c4d743785d1a0aabbdee0627486352945749a57d81b2",
        "18513bbf7b92945f6569ce34056c18c7a5877d7fc814fc5eaaf576cf5841480b",
        "6ff1d75550627a6d22d31bbd6a3c4070a7b290f2c16cb33f0506e4bf2cbc91bb",
    ),
    "flg": (
        "3a77af4f4ece3751269ff413f710c39fe8ec2c0bc30e2e8e845534c62fc7f2f9",
        "9417c4667da3af128f684cf5b847952b47686a329af28a94cbf01615f3de8644",
        "9424f0c86779748c2bee8634b7cf16a9e1c51cec69aeacc45f0264882d24e1c9",
        "c36fac369f580567bc515a6454bd0c1669d585b79523b3a73921410dae78897a",
        "c3e89f8e57250c65f1ec51ec1bf99db07904431016900154e053ade741904a1c",
        "f5c29fe63df23a40aff98d58447af577ba7244af2c762a288555041dd7fdfac4",
    ),
    "pokemonfan": (
        "061627395d4bb484f2e21d1a647f6d69fad043016efc102fe093aa55815f4a7f",
        "10192657ae504eed3bc9e3fd59f80f7e9499d6022e38bdddd6c5da69e78afd81",
        "403eac5b6e720e81ce1fa978600a0865e9c88797e9336cf0a36439690b3d3af5",
        "51eb642eb4c5a82cad4ac22c82bca296841ee8e348a19028aa847b9b658b4305",
        "5b3af894ebb7d7447a730ce5dbbfe9bcfc12505698837b1f39d275ca54cc06d1",
        "7bf6fa80288ce4f38874d11da426030af291cdd3880a0242458cd111632e537d",
        "df73f21e6b4ff3a35a91fd32b64c89dd10e4c4645fbfe7fcad9e08af147aea66",
        "e9f2ca559bae15049b5bf044557de57dc8900908127021a540c0bebd53c25813",
    ),
}


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")


def self_evidence() -> dict[str, Any]:
    return require_regular(Path(__file__).resolve(), None, "probe tool")


def ast_audit() -> dict[str, Any]:
    source = Path(__file__).resolve().read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__).resolve()))

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [dotted(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden_suffixes = (
        ".backward",
        ".save",
        ".write_text",
        ".write_bytes",
    )
    forbidden = sorted(
        name
        for name in calls
        if any(name.endswith(suffix) for suffix in forbidden_suffixes)
        or name.startswith("torch.optim")
    )
    if forbidden:
        raise RuntimeError(f"zero-write AST gate failed: {forbidden}")
    observed = {
        "torch.autograd.grad": calls.count("torch.autograd.grad"),
        "torch.linalg.lstsq": calls.count("torch.linalg.lstsq"),
        "parameter.copy_": calls.count("parameter.copy_"),
        "torch.save": calls.count("torch.save"),
    }
    expected = {
        "torch.autograd.grad": 1,
        "torch.linalg.lstsq": 1,
        "parameter.copy_": 1,
        "torch.save": 0,
    }
    if observed != expected:
        raise RuntimeError(f"probe call-site count drift: {observed}")
    return {
        "status": "zero_write_ast_audit_passed",
        "call_sites": observed,
        "optimizer_instances": 0,
        "backward_calls": 0,
        "checkpoint_or_artifact_write_calls": 0,
        "ram_parameter_copy_call_sites": 1,
    }


def load_profile() -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = require_regular(PROFILE, PROFILE_SHA256, "raw-U468 margin profile")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if (
        profile.get("schema_version")
        != "ptcg-u468-raw-full-train-margin-profile-v3"
        or profile.get("status") != "completed_train_only"
        or profile.get("split") != "train"
        or profile.get("validation_opened") is not False
    ):
        raise RuntimeError("raw-U468 train profile contract drift")
    return profile, evidence


def select_constraint_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    observed_targets: dict[str, list[str]] = {}
    observed_guards: dict[str, list[str]] = {}
    for source in sorted(frozen.DATASETS):
        panel = profile["profiles"][source]
        targets = [
            {**record, "source": source, "role": "target"}
            for record in panel["near_wrong"]
            if record.get("ordered_correct") is False
            and float(record.get("decision_margin")) == 0.0
            and float(record.get("selection_margin")) == 0.0
            and isinstance(record.get("expert_order"), list)
            and len(record["expert_order"]) > 0
            and len(record.get("predicted_order", [])) == len(record["expert_order"])
        ]
        guards = [
            {**record, "source": source, "role": "guard"}
            for record in panel["fragile_correct"]
            if record.get("ordered_correct") is True
            and float(record.get("decision_margin")) == 0.0
            and float(record.get("selection_margin")) == 0.0
            and isinstance(record.get("expert_order"), list)
            and len(record["expert_order"]) > 0
        ]
        targets.sort(key=lambda row: str(row["line_sha256"]))
        guards.sort(key=lambda row: str(row["line_sha256"]))
        observed_targets[source] = [str(row["line_sha256"]) for row in targets]
        observed_guards[source] = [str(row["line_sha256"]) for row in guards]
        selected.extend(targets)
        selected.extend(guards)
    if observed_targets != {
        source: list(EXPECTED_TARGET_HASHES[source]) for source in sorted(EXPECTED_TARGET_HASHES)
    }:
        raise RuntimeError(f"target identity drift: {observed_targets}")
    if observed_guards != {
        source: list(EXPECTED_GUARD_HASHES[source]) for source in sorted(EXPECTED_GUARD_HASHES)
    }:
        raise RuntimeError(f"guard identity drift: {observed_guards}")
    if len(selected) != 24 or len({row["line_sha256"] for row in selected}) != 24:
        raise RuntimeError("constraint row count/uniqueness drift")
    return selected


def load_canonical_constraint_batches(
    rows: Sequence[dict[str, Any]],
    model_config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild the profiler's exact sequential 256-row batch boundaries.

    Native BF16 GEMM rounding can change when the physical batch shape changes.
    A compact 24-row collate is therefore not authoritative for ties profiled in
    the original 256-row stream.  We featurize every train row in the same order
    as ``profile_archive`` and retain only full canonical batches containing a
    selected constraint row.
    """

    wanted = {str(record["line_sha256"]): record for record in rows}
    if len(wanted) != 24:
        raise RuntimeError("constraint identity set is not unique")
    found: dict[str, tuple[str, int, int]] = {}
    retained: list[dict[str, Any]] = []
    opened_members: dict[str, list[str]] = {source: [] for source in frozen.DATASETS}
    source_rows: dict[str, int] = {}

    def collate_pending(
        source: str,
        batch_index: int,
        features: list[dict[str, Any]],
        identities: list[dict[str, Any]],
    ) -> None:
        selected = [
            {"local_index": index, "record": wanted[str(identity["line_sha256"])]}
            for index, identity in enumerate(identities)
            if str(identity["line_sha256"]) in wanted
        ]
        if not selected:
            return
        batch = bc.collate_decisions(
            features,
            max_state_entities=int(model_config["max_state_entities"]),
            entity_fields=int(model_config["entity_fields"]),
            option_fields=int(model_config["option_fields"]),
        )
        if int(batch["action_counts"].shape[0]) != len(features):
            raise RuntimeError("canonical constraint batch row-count drift")
        for item in selected:
            record = item["record"]
            line_sha = str(record["line_sha256"])
            identity = identities[int(item["local_index"])]
            if line_sha in found:
                raise RuntimeError("constraint row repeated in canonical stream")
            if (
                str(identity["member"]) != str(record["member"])
                or int(identity["line_index"]) != int(record["line_index"])
            ):
                raise RuntimeError("canonical constraint location drift")
            found[line_sha] = (source, batch_index, int(item["local_index"]))
        retained.append(
            {
                "source": source,
                "batch_index": batch_index,
                "batch": batch,
                "selected": selected,
                "rows": len(features),
            }
        )

    for source, archive_path in frozen.DATASETS.items():
        pending_features: list[dict[str, Any]] = []
        pending_identities: list[dict[str, Any]] = []
        total_rows = 0
        batch_index = 0
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                member
                for member in archive.namelist()
                if member.startswith("train/") and member.endswith(".jsonl")
            )
            if not members:
                raise RuntimeError(f"{source}: no train members")
            for member in members:
                opened_members[source].append(member)
                with archive.open(member) as handle:
                    for line_index, raw in enumerate(handle):
                        row = orjson.loads(raw)
                        if str(row.get("split", "")) != "train":
                            raise RuntimeError("canonical stream contains non-train row")
                        previous_max = bc.MAX_ACTION_COUNT
                        bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
                        try:
                            feature = bc.featurize_row(
                                row,
                                int(model_config["hash_size"]),
                                int(model_config["max_state_entities"]),
                            )
                        finally:
                            bc.MAX_ACTION_COUNT = previous_max
                        if feature is None or not isinstance(row.get("action", []), list):
                            continue
                        expert = [int(value) for value in row.get("action", [])]
                        feature["action_sequence"] = expert
                        identity = {
                            "member": member,
                            "line_index": line_index,
                            "line_sha256": hashlib.sha256(raw).hexdigest(),
                        }
                        line_sha = str(identity["line_sha256"])
                        if line_sha in wanted:
                            record = wanted[line_sha]
                            if (
                                str(record["source"]) != source
                                or str(record["member"]) != member
                                or int(record["line_index"]) != line_index
                                or expert != [int(value) for value in record["expert_order"]]
                                or int(feature["context"]) != int(record["context"])
                                or int(feature["min_count"]) != int(record["min_count"])
                                or int(feature["max_count"]) != int(record["max_count"])
                            ):
                                raise RuntimeError("canonical selected-row metadata drift")
                        pending_features.append(feature)
                        pending_identities.append(identity)
                        total_rows += 1
                        if len(pending_features) == 256:
                            collate_pending(
                                source,
                                batch_index,
                                pending_features,
                                pending_identities,
                            )
                            batch_index += 1
                            pending_features = []
                            pending_identities = []
            if pending_features:
                collate_pending(
                    source,
                    batch_index,
                    pending_features,
                    pending_identities,
                )
        source_rows[source] = total_rows

    expected_source_rows = {
        source: int(profile["profiles"][source]["rows"])
        for source in frozen.DATASETS
    }
    if expected_source_rows != {"flg": 9443, "pokemonfan": 9487, "core5": 5120}:
        raise RuntimeError(f"frozen profile row-count drift: {expected_source_rows}")
    if source_rows != expected_source_rows:
        raise RuntimeError(f"canonical train row-count drift: {source_rows}")
    if set(found) != set(wanted):
        raise RuntimeError(f"canonical constraint rows missing: {sorted(set(wanted) - set(found))}")
    if any(
        not member.startswith("train/")
        for values in opened_members.values()
        for member in values
    ):
        raise RuntimeError("canonical loader opened non-train member")
    batch_manifest = [
        {
            "source": item["source"],
            "batch_index": item["batch_index"],
            "rows": item["rows"],
            "selected_rows": len(item["selected"]),
            "option_width": int(item["batch"]["option_mask"].shape[1]),
        }
        for item in retained
    ]
    return retained, {
        "rows": 24,
        "targets": 7,
        "guards": 17,
        "canonical_batch_size": 256,
        "canonical_source_rows": source_rows,
        "retained_batches": batch_manifest,
        "retained_batch_count": len(retained),
        "opened_members": opened_members,
        "non_train_members_opened": False,
        "line_identity_sha256": hashlib.sha256(
            "\n".join(str(row["line_sha256"]) for row in rows).encode("ascii")
        ).hexdigest(),
    }


def constraint_predictions(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> list[list[int]]:
    predictions, _, _, _ = ppo.sample_ordered_actions(
        dict(outputs), dict(batch), deterministic=True, canonicalize_order=False
    )
    return [[int(value) for value in prediction] for prediction in predictions]


def build_margin_constraints(
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    selected: Sequence[Mapping[str, Any]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    if logits.dtype != torch.bfloat16:
        raise RuntimeError(f"native policy logits dtype drift: {logits.dtype}")
    scalars: list[torch.Tensor] = []
    identities: list[dict[str, Any]] = []
    for item in selected:
        row_index = int(item["local_index"])
        record = item["record"]
        remaining = batch["option_mask"][row_index].bool().clone()
        found_for_row = 0
        expert = [int(value) for value in record["expert_order"]]
        for stage, chosen in enumerate(expert):
            if chosen < 0 or chosen >= int(remaining.shape[0]) or not bool(remaining[chosen]):
                raise RuntimeError("illegal expert action in constraint batch")
            competitors = remaining.nonzero(as_tuple=False).squeeze(1)
            competitors = competitors[competitors != chosen]
            if competitors.numel() == 0:
                remaining[chosen] = False
                continue
            chosen_logit = logits[row_index, chosen].float()
            competitor_logits = logits[row_index, competitors].float()
            best = competitor_logits.max()
            margin = chosen_logit - best
            if float(margin.detach().cpu()) < 0.0 and record["role"] == "guard":
                raise RuntimeError("raw guard has a negative expert-stage margin")
            if float(margin.detach().cpu()) == 0.0:
                tied = competitors[competitor_logits == best]
                for competitor in tied.tolist():
                    scalar = (
                        logits[row_index, chosen].float()
                        - logits[row_index, int(competitor)].float()
                    )
                    if float(scalar.detach().cpu()) != 0.0:
                        raise RuntimeError("tie scalar is not exact zero")
                    scalars.append(scalar)
                    identities.append(
                        {
                            "role": str(record["role"]),
                            "source": str(record["source"]),
                            "line_sha256": str(record["line_sha256"]),
                            "member": str(record["member"]),
                            "line_index": int(record["line_index"]),
                            "stage": stage,
                            "expert_action": chosen,
                            "tied_competitor": int(competitor),
                        }
                    )
                    found_for_row += 1
            remaining[chosen] = False
        if found_for_row < 1:
            raise RuntimeError(f"zero-margin row has no exact policy tie: {record['line_sha256']}")
    return scalars, identities


def collect_constraint_gradients(
    model: torch.nn.Module,
    parameter: torch.nn.Parameter,
    canonical_batches: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[dict[str, Any]], dict[str, list[int]]]:
    collected: list[tuple[dict[str, Any], torch.Tensor]] = []
    predictions_by_line: dict[str, list[int]] = {}
    for canonical in canonical_batches:
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in canonical["batch"].items()
        }
        outputs = ppo.model_forward(model, batch, device)
        predictions = constraint_predictions(outputs, batch)
        for item in canonical["selected"]:
            record = item["record"]
            prediction = predictions[int(item["local_index"])]
            line_sha = str(record["line_sha256"])
            if line_sha in predictions_by_line:
                raise RuntimeError("canonical raw prediction repeated")
            if prediction != [int(value) for value in record["predicted_order"]]:
                raise RuntimeError(f"canonical raw prediction drift: {line_sha}")
            predictions_by_line[line_sha] = prediction
        scalars, identities = build_margin_constraints(
            outputs["policy_logits"], batch, canonical["selected"]
        )
        for index, (scalar, identity) in enumerate(zip(scalars, identities)):
            gradient = torch.autograd.grad(
                scalar,
                parameter,
                retain_graph=index + 1 < len(scalars),
                create_graph=False,
                allow_unused=False,
                materialize_grads=False,
            )[0]
            vector = gradient.detach().cpu().reshape(-1).to(torch.float64)
            if not bool(torch.isfinite(vector).all()) or float(vector.norm()) <= 0.0:
                raise RuntimeError(f"invalid constraint gradient: {identity}")
            collected.append((identity, vector))
    if parameter.grad is not None:
        raise RuntimeError("autograd.grad materialized a .grad buffer")
    collected.sort(
        key=lambda item: (
            str(item[0]["source"]),
            str(item[0]["line_sha256"]),
            int(item[0]["stage"]),
            int(item[0]["expert_action"]),
            int(item[0]["tied_competitor"]),
        )
    )
    return (
        [item[1] for item in collected],
        [item[0] for item in collected],
        predictions_by_line,
    )


def solve_direction(
    gradients: Sequence[torch.Tensor],
    identities: Sequence[Mapping[str, Any]],
    parameter: torch.nn.Parameter,
) -> tuple[torch.Tensor, dict[str, Any]]:
    matrix = torch.stack([gradient / gradient.norm() for gradient in gradients])
    rhs = torch.ones(matrix.shape[0], dtype=torch.float64)
    solution = torch.linalg.lstsq(matrix, rhs, driver="gelsd")
    raw_direction = solution.solution
    residual = matrix @ raw_direction - rhs
    max_abs_residual = float(residual.abs().max())
    if not bool(torch.isfinite(raw_direction).all()) or max_abs_residual > SOLVER_ABS_RESIDUAL_MAX:
        raise RuntimeError(
            f"constraint system is infeasible/unstable: max residual {max_abs_residual}"
        )
    raw_norm = float(raw_direction.norm())
    if raw_norm <= 0.0:
        raise RuntimeError("constraint solution has zero norm")
    direction = raw_direction * (REFERENCE_STEP_L2 / raw_norm)
    derivatives = matrix @ direction
    if float(derivatives.min()) <= 0.0:
        raise RuntimeError("normalized direction is not positive on every tie")
    singular_values = torch.linalg.svdvals(matrix)
    positive = singular_values[singular_values > torch.finfo(torch.float64).eps]
    condition = (
        float(positive.max() / positive.min()) if positive.numel() else float("inf")
    )
    payload = direction.contiguous().view(torch.uint8).numpy().tobytes()
    by_role = {
        role: sum(str(identity["role"]) == role for identity in identities)
        for role in ("target", "guard")
    }
    return direction.reshape(parameter.shape), {
        "method": "cpu_float64_minimum_norm_equality_lstsq_gelsd",
        "parameter": EDITED_PARAMETER,
        "parameter_elements": int(parameter.numel()),
        "constraint_count": len(gradients),
        "constraints_by_role": by_role,
        "matrix_rank": int(solution.rank),
        "singular_values_max": float(singular_values.max()),
        "singular_values_min": float(singular_values.min()),
        "condition_over_positive_singular_values": condition,
        "unscaled_solution_l2": raw_norm,
        "max_abs_equality_residual": max_abs_residual,
        "reference_step_l2": REFERENCE_STEP_L2,
        "min_normalized_constraint_derivative_at_reference_step": float(derivatives.min()),
        "max_normalized_constraint_derivative_at_reference_step": float(derivatives.max()),
        "direction_sha256_float64": hashlib.sha256(payload).hexdigest(),
        "all_constraint_derivatives_positive": True,
    }


def set_endpoint_weight(
    parameter: torch.nn.Parameter,
    raw_cpu: torch.Tensor,
    direction: torch.Tensor,
    alpha: int,
) -> dict[str, Any]:
    candidate_cpu = (raw_cpu.to(torch.float64) + float(alpha) * direction).to(raw_cpu.dtype)
    with torch.no_grad():
        parameter.copy_(candidate_cpu.to(parameter.device))
    delta = candidate_cpu.to(torch.float64) - raw_cpu.to(torch.float64)
    return {
        "alpha": alpha,
        "actual_fp32_displacement_l2": float(delta.norm()),
        "actual_fp32_displacement_max_abs": float(delta.abs().max()),
        "changed_elements": int(torch.count_nonzero(delta)),
    }


def output_identity(
    model: torch.nn.Module,
    cpu_batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
    batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
    with torch.no_grad():
        outputs = ppo.model_forward(model, batch, device)
        predictions = constraint_predictions(outputs, batch)
    return {
        "count_logits": outputs["count_logits"].detach().cpu(),
        "value_logits": outputs["value_logits"].detach().cpu(),
    }, predictions


def evaluate_canonical_constraint_predictions(
    model: torch.nn.Module,
    canonical_batches: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> list[list[int]]:
    predictions_by_line: dict[str, list[int]] = {}
    with torch.no_grad():
        for canonical in canonical_batches:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in canonical["batch"].items()
            }
            outputs = ppo.model_forward(model, batch, device)
            predictions = constraint_predictions(outputs, batch)
            for item in canonical["selected"]:
                line_sha = str(item["record"]["line_sha256"])
                if line_sha in predictions_by_line:
                    raise RuntimeError("canonical candidate prediction repeated")
                predictions_by_line[line_sha] = predictions[int(item["local_index"])]
    expected = {str(row["line_sha256"]) for row in rows}
    if set(predictions_by_line) != expected:
        raise RuntimeError("canonical candidate prediction coverage drift")
    return [predictions_by_line[str(row["line_sha256"])] for row in rows]


def summarize_constraint_predictions(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[int]]
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise RuntimeError("constraint prediction alignment drift")
    target_by_source = {source: {"rows": 0, "correct": 0} for source in frozen.DATASETS}
    guard_wrong: list[str] = []
    target_correct_hashes: list[str] = []
    for row, prediction in zip(rows, predictions):
        correct = list(prediction) == [int(value) for value in row["expert_order"]]
        if row["role"] == "target":
            source = str(row["source"])
            target_by_source[source]["rows"] += 1
            target_by_source[source]["correct"] += int(correct)
            if correct:
                target_correct_hashes.append(str(row["line_sha256"]))
        elif not correct:
            guard_wrong.append(str(row["line_sha256"]))
    return {
        "target_by_source": target_by_source,
        "target_correct_total": len(target_correct_hashes),
        "target_correct_hashes": sorted(target_correct_hashes),
        "guard_wrong_count": len(guard_wrong),
        "guard_wrong_hashes": sorted(guard_wrong),
    }


def loss_nonregression(
    raw: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[bool, dict[str, float]]:
    deltas = {"mixed": float(candidate["mixed_ordered_loss"] - raw["mixed_ordered_loss"])}
    for source in frozen.DATASETS:
        for bucket in ("all", "hard", "fragile", "c34"):
            deltas[f"{source}_{bucket}"] = float(
                candidate["by_source_and_bucket"][source][bucket]["ordered_loss"]
                - raw["by_source_and_bucket"][source][bucket]["ordered_loss"]
            )
    return all(delta <= LOSS_ABS_TOLERANCE for delta in deltas.values()), deltas


def run_actual(
    constraint_rows: Sequence[dict[str, Any]],
    canonical_constraint_batches: Sequence[Mapping[str, Any]],
    direct_selections: Sequence[Sequence[dict[str, Any]]],
    direct_union: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    random.seed(EXECUTION_SEED)
    torch.manual_seed(EXECUTION_SEED)
    torch.cuda.manual_seed_all(EXECUTION_SEED)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)

    model, _ = frozen.load_raw_u468(device)
    raw_model_sha256 = ppo.model_state_sha256(model)
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    parameter = named[EDITED_PARAMETER]
    parameter.requires_grad_(True)
    active = [name for name, value in model.named_parameters() if value.requires_grad]
    if active != [EDITED_PARAMETER] or tuple(parameter.shape) != (1, 128):
        raise RuntimeError(f"edit scope/shape drift: {active}, {tuple(parameter.shape)}")
    raw_weight_cpu = parameter.detach().cpu().clone()

    model.eval()
    gradients, constraint_identities, raw_predictions_by_line = collect_constraint_gradients(
        model, parameter, canonical_constraint_batches, device
    )
    direction, solver = solve_direction(gradients, constraint_identities, parameter)
    raw_constraint_predictions = [
        raw_predictions_by_line[str(row["line_sha256"])] for row in constraint_rows
    ]
    parameter.requires_grad_(False)

    raw_direct_report, raw_direct_correct = frozen.evaluate_selected_union(
        model, [dict(direct_union)], [[row for batch in direct_selections for row in batch]], device
    )
    raw_head_outputs, _ = output_identity(model, direct_union, device)
    raw_constraint_summary = summarize_constraint_predictions(
        constraint_rows, raw_constraint_predictions
    )
    if raw_constraint_summary["target_correct_total"] != 0 or raw_constraint_summary["guard_wrong_count"] != 0:
        raise RuntimeError("raw target/guard correctness contract drift")

    candidates: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        displacement = set_endpoint_weight(
            parameter, raw_weight_cpu, direction, alpha
        )
        direct_report, direct_correct = frozen.evaluate_selected_union(
            model,
            [dict(direct_union)],
            [[row for batch in direct_selections for row in batch]],
            device,
        )
        head_outputs, _ = output_identity(model, direct_union, device)
        candidate_constraint_predictions = evaluate_canonical_constraint_predictions(
            model, canonical_constraint_batches, constraint_rows, device
        )
        constraint_summary = summarize_constraint_predictions(
            constraint_rows, candidate_constraint_predictions
        )
        correct_to_wrong = sorted(
            line_sha
            for line_sha, raw_correct in raw_direct_correct.items()
            if raw_correct and not direct_correct[line_sha]
        )
        wrong_to_correct = sorted(
            line_sha
            for line_sha, raw_correct in raw_direct_correct.items()
            if not raw_correct and direct_correct[line_sha]
        )
        heads_exact = {
            name: torch.equal(raw_head_outputs[name], head_outputs[name])
            for name in ("count_logits", "value_logits")
        }
        loss_gate, loss_deltas = loss_nonregression(raw_direct_report, direct_report)
        gate_checks = {
            "direct512_zero_correct_to_wrong": len(correct_to_wrong) == 0,
            "all_17_zero_margin_guards_correct": constraint_summary["guard_wrong_count"] == 0,
            "all_7_tie_targets_correct": constraint_summary["target_correct_total"] == 7,
            "all_3_pokemonfan_targets_correct": (
                constraint_summary["target_by_source"]["pokemonfan"]
                == {"rows": 3, "correct": 3}
            ),
            "count_logits_tensor_exact_raw": heads_exact["count_logits"],
            "value_logits_tensor_exact_raw": heads_exact["value_logits"],
            "direct512_all_source_buckets_losses_nonworse": loss_gate,
        }
        candidates.append(
            {
                "label": f"alpha{alpha}",
                "displacement": displacement,
                "constraint_predictions": constraint_summary,
                "direct512_transitions": {
                    "correct_to_wrong_count": len(correct_to_wrong),
                    "correct_to_wrong_hashes": correct_to_wrong,
                    "wrong_to_correct_count": len(wrong_to_correct),
                    "wrong_to_correct_hashes": wrong_to_correct,
                },
                "count_value_exact": heads_exact,
                "direct512_loss_deltas_candidate_minus_raw": loss_deltas,
                "direct512_metrics": direct_report,
                "gate_checks": gate_checks,
                "eligible": all(gate_checks.values()),
            }
        )

    set_endpoint_weight(parameter, raw_weight_cpu, direction, 0)
    restored_sha256 = ppo.model_state_sha256(model)
    if restored_sha256 != raw_model_sha256:
        raise RuntimeError("raw model state was not exactly restored")
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = eligible[0]["label"] if eligible else None
    return {
        "status": "completed_ram_only_shadow",
        "decision": "SHADOW_GO" if selected is not None else "SHADOW_CLOSE",
        "selected_smallest_eligible_alpha": selected,
        "solver": solver,
        "constraint_identities": constraint_identities,
        "raw_constraint_predictions": raw_constraint_summary,
        "raw_direct512_metrics": raw_direct_report,
        "candidates": candidates,
        "integrity": {
            "raw_model_state_sha256_before": raw_model_sha256,
            "raw_model_state_sha256_after_restore": restored_sha256,
            "model_state_restored_exact": True,
            "edited_parameter": EDITED_PARAMETER,
            "other_parameters_changed": False,
            "optimizer_instances": 0,
            "backward_calls": 0,
            "checkpoint_writes": 0,
            "model_artifact_writes": 0,
            "result_artifact_writes": 0,
            "validation_member_payloads_opened": False,
            "submission_performed": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("static-audit", "cache-audit", "actual"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    tool = self_evidence()
    static = ast_audit()
    fixed_inputs = aggregate.verify_fixed_inputs()
    profile, profile_evidence = load_profile()
    rows = select_constraint_rows(profile)
    parent = torch.load(frozen.U468, map_location="cpu", weights_only=False)
    if int(parent.get("update", -1)) != 468:
        raise RuntimeError("raw U468 update drift")
    canonical_constraint_batches, constraint_cache = load_canonical_constraint_batches(
        rows, parent["model_config"], profile
    )
    cache_audit, direct_selections, direct_union = aggregate.build_cache_audit()
    common = {
        "schema_version": SCHEMA,
        "tool": tool,
        "frozen_aggregate_runner": AGGREGATE_EVIDENCE,
        "profile": profile_evidence,
        "fixed_inputs": fixed_inputs,
        "static_audit": static,
        "contract": {
            "base": "raw full U468",
            "edited_parameter": EDITED_PARAMETER,
            "constraint_rows": {"targets": 7, "guards": 17},
            "reference_step_l2": REFERENCE_STEP_L2,
            "alphas": list(ALPHAS),
            "endpoint_construction": "raw_fp32_plus_alpha_times_one_fixed_direction",
            "selection": "smallest endpoint passing every frozen shadow gate",
            "train_only": True,
            "validation": False,
            "optimizer": False,
            "backward": False,
            "checkpoint_write": False,
            "model_artifact_write": False,
            "result_artifact_write": False,
            "stdout_only": True,
        },
        "constraint_cache": constraint_cache,
        "direct512_cache": cache_audit,
    }
    if args.mode == "static-audit":
        if args.device != "cpu":
            raise ValueError("static-audit requires default CPU")
        print(
            json.dumps(
                {
                    **common,
                    "status": "zero_write_static_audit_passed",
                    "archive_payloads_opened_for_constraints": True,
                    "validation_member_payloads_opened": False,
                    "writes_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if args.mode == "cache-audit":
        if args.device != "cpu":
            raise ValueError("cache-audit requires default CPU")
        print(
            json.dumps(
                {
                    **common,
                    "status": "zero_write_cache_audit_passed",
                    "validation_member_payloads_opened": False,
                    "writes_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual mode requires CUDA")
    result = run_actual(
        rows,
        canonical_constraint_batches,
        direct_selections,
        direct_union,
        torch.device("cuda"),
    )
    print(
        json.dumps(
            {**common, **result},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

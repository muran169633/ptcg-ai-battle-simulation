#!/usr/bin/env python3
"""Zero-write P/E/T actor6 coefficient-cone geometry preprobe.

The three frozen coefficient directions are

* P = actor6(P - R),
* E = 256 * actor6(E - R), and
* T = actor6(T2 - T1).

Only those six actor tensors are read from the five SHA-bound checkpoints.  A
CPU-float64 SVD constructs an orthonormal basis for their span.  At unchanged
raw U468, the CUDA path projects the 26 canonical exact-tie margin gradients
and the 13 direct512 ordered-loss gradients into that three-dimensional basis.

Feasibility is a deterministic linear program over basis coefficients in
[-1, 1]: maximize the minimum normalized target slope, subject to nonnegative
guard slopes and nonpositive loss slopes.  A failed strict-feasibility result
is accompanied by a float64 Farkas certificate when HiGHS can establish one.

This tool has no endpoint construction, parameter copy, optimizer, backward,
save, or result-file path.  It emits one compact JSON object to stdout.
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
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

TIE_TOOL = TOOLS / "probe_u468_raw_residual2_tie_constrained_shadow.py"
TIE_TOOL_SHA256 = "37d44c06575c26c82fe6563eb013b15598499aca40868ba3f04db313b35c0f30"

CHECKPOINT_SPECS: dict[str, tuple[Path, str]] = {
    "R": (
        ROOT
        / "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090"
        / "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt",
        "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f",
    ),
    "P": (
        ROOT
        / "artifacts/ppo_u468_p12delta_direction_beta050_075_100_design202608092"
        / "transport-beta-100.pt",
        "53284b1d4e94f09bff5b92a7fb0d24efd26ee67a2a332014cdc97572b00c3beb",
    ),
    "E": (
        ROOT
        / "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_materialize_design202608116"
        / "equalblend-sgd512-eval-only.pt",
        "1684493b48b9c6a77696150d925150c21d8028c5fdd45f80e9e838a94d472be2",
    ),
    "T1": (
        ROOT
        / "artifacts/ppo_u468_beta1157_trainhard_actor6_mix102_77_77_p1p2_design202608101"
        / "special-bc-trainhard-actor6-0001.pt",
        "7157bacb249d361a8bef986edc8e0e8a107ef97d34b847a963bd551b3cc00441",
    ),
    "T2": (
        ROOT
        / "artifacts/ppo_u468_beta1157_trainhard_actor6_mix102_77_77_p1p2_design202608101"
        / "special-bc-trainhard-actor6-0002.pt",
        "e8755aa309446d6afb71655794553532016a3a36cceb7b4ab1d6760c9e3f32a9",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_evidence(path: Path, expected_sha256: str | None, label: str) -> dict[str, Any]:
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
        "mode": oct(observed.st_mode & 0o777),
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


# Authenticate every executable dependency and every coefficient source before
# importing code or deserializing a checkpoint.
TIE_EVIDENCE = regular_evidence(TIE_TOOL, TIE_TOOL_SHA256, "frozen tie tool")
CHECKPOINT_EVIDENCE = {
    name: regular_evidence(path, digest, f"{name} checkpoint")
    for name, (path, digest) in CHECKPOINT_SPECS.items()
}
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_u468_tie_geometry_37d44c06", TIE_TOOL
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot construct frozen tie-tool import spec")
tie: ModuleType = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tie)

torch = tie.torch
ppo = tie.ppo
frozen = tie.frozen
aggregate = tie.aggregate


SCHEMA = "ptcg-u468-actor6-pet-cone-geometry-v1"
EXECUTION_SEED = 202608124
ACTOR_NAMES = tuple(frozen.ACTOR_NAMES)
EXPECTED_ACTOR_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
EXPECTED_ACTOR_ELEMENTS = 65793
DIRECTION_NAMES = ("P", "E", "T")
DIRECTION_FORMULAS = {
    "P": "actor6(P-R)",
    "E": "256*actor6(E-R)",
    "T": "actor6(T2-T1)",
}
E_DIRECTION_SCALE = 256.0
SOURCES = ("flg", "pokemonfan", "core5")
BUCKETS = ("all", "hard", "fragile", "c34")
LOSS_NAMES = ("mixed",) + tuple(
    f"{source}_{bucket}" for source in SOURCES for bucket in BUCKETS
)
EXPECTED_DIRECT_CACHE_SHA256 = (
    "ef5ab1b8f7e7162316e8ae80a6621f902e9a8e8cf73bb54daaf99e1355d0114a"
)
EXPECTED_DIRECT_BATCH_SHA256 = (
    "1c0bd23912b85dcbc64318d42ae41ba1fd7217f80e9a741e8970d91adeb1db05"
)
EXPECTED_DIRECT_IDENTITY_SHA256 = (
    "d0c148e1c992030a9c5442b7a407b50816a30fdec27488f4f79b26b9e5916ce8"
)
DIRECT_IDENTITY_FIELDS = (
    "source",
    "member",
    "line_index",
    "line_sha256",
    "episode_id",
    "team_name",
    "category",
    "context",
    "min_count",
    "max_count",
    "expert_order",
)
EXPECTED_GROUP_ROWS = {
    "mixed": 512,
    "flg_all": 128,
    "flg_hard": 96,
    "flg_fragile": 30,
    "flg_c34": 2,
    "pokemonfan_all": 192,
    "pokemonfan_hard": 64,
    "pokemonfan_fragile": 126,
    "pokemonfan_c34": 2,
    "core5_all": 192,
    "core5_hard": 64,
    "core5_fragile": 126,
    "core5_c34": 2,
}
SVD_RELATIVE_RANK_TOLERANCE = 1e-12
ORTHONORMAL_ABS_TOLERANCE = 1e-11
LP_PRIMAL_ABS_TOLERANCE = 1e-9
STRICT_TARGET_SLOPE_TOLERANCE = 1e-10
FARKAS_ABS_TOLERANCE = 1e-8


def canonical_json_bytes(value: Any) -> bytes:
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


def emit_json(value: Mapping[str, Any]) -> None:
    """Emit exactly one compact JSON document for the selected CLI branch."""
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


def vector_sha256_float64(vector: torch.Tensor) -> str:
    array = np.ascontiguousarray(vector.detach().cpu().numpy().astype("<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")
    if ACTOR_NAMES != EXPECTED_ACTOR_NAMES:
        raise RuntimeError(f"actor6 name/order drift: {ACTOR_NAMES}")
    if Path(frozen.U468).resolve() != CHECKPOINT_SPECS["R"][0].resolve():
        raise RuntimeError("frozen raw-U468 path is not coefficient anchor R")
    if frozen.U468_SHA256 != CHECKPOINT_SPECS["R"][1]:
        raise RuntimeError("frozen raw-U468 SHA is not coefficient anchor R")


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
    forbidden = sorted(
        name
        for name in calls
        if name.endswith(
            (
                ".backward",
                ".step",
                ".zero_grad",
                ".save",
                ".write_text",
                ".write_bytes",
                ".copy_",
            )
        )
        or name.startswith("torch.optim")
    )
    if forbidden:
        raise RuntimeError(f"zero-write/zero-update AST gate failed: {forbidden}")
    expected = {
        "torch.autograd.grad": 1,
        "torch.linalg.svd": 1,
        "optimize.linprog": 2,
        "torch.save": 0,
        "print": 1,
    }
    observed = {name: calls.count(name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"geometry call-site drift: {observed}")
    inherited = tie.ast_audit()
    if inherited.get("status") != "zero_write_ast_audit_passed":
        raise RuntimeError("frozen tie-tool AST audit failed")
    return {
        "status": "zero_write_geometry_ast_audit_passed",
        "call_sites": observed,
        "forbidden_calls": [],
        "endpoint_construction_calls": 0,
        "optimizer_calls": 0,
        "backward_calls": 0,
        "parameter_copy_calls": 0,
        "checkpoint_or_result_write_calls": 0,
        "frozen_tie_tool": inherited,
    }


def load_checkpoint(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    regular_evidence(path, expected_sha256, label)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("update", -1)) != 468:
        raise RuntimeError(f"{label} payload/update drift")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"{label} has no model_state_dict")
    return payload


def actor_vector(payload: Mapping[str, Any], label: str) -> torch.Tensor:
    state = payload["model_state_dict"]
    pieces: list[torch.Tensor] = []
    for name in ACTOR_NAMES:
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise RuntimeError(f"{label}/{name} is not an FP32 tensor")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{label}/{name} is nonfinite")
        pieces.append(value.detach().cpu().reshape(-1).to(torch.float64))
    result = torch.cat(pieces)
    if int(result.numel()) != EXPECTED_ACTOR_ELEMENTS:
        raise RuntimeError(f"{label} actor6 flattened length drift")
    return result


def build_basis_geometry() -> dict[str, Any]:
    payloads = {
        name: load_checkpoint(path, digest, f"{name} checkpoint")
        for name, (path, digest) in CHECKPOINT_SPECS.items()
    }
    reference_keys = tuple(payloads["R"]["model_state_dict"])
    if any(tuple(payload["model_state_dict"]) != reference_keys for payload in payloads.values()):
        raise RuntimeError("checkpoint model-state key/order drift")
    vectors = {name: actor_vector(payload, name) for name, payload in payloads.items()}
    directions = {
        "P": vectors["P"] - vectors["R"],
        "E": E_DIRECTION_SCALE * (vectors["E"] - vectors["R"]),
        "T": vectors["T2"] - vectors["T1"],
    }
    matrix = torch.stack([directions[name] for name in DIRECTION_NAMES], dim=1)
    if matrix.dtype != torch.float64 or tuple(matrix.shape) != (EXPECTED_ACTOR_ELEMENTS, 3):
        raise RuntimeError("P/E/T basis matrix shape/dtype drift")
    gram = matrix.T @ matrix
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = int(
        torch.count_nonzero(
            singular_values > singular_values[0] * SVD_RELATIVE_RANK_TOLERANCE
        )
    )
    if rank != 3:
        raise RuntimeError(f"P/E/T actor6 basis rank is {rank}, expected 3")
    q = u[:, :3].contiguous()
    # Canonicalize each otherwise sign-ambiguous SVD column.
    pivots: list[int] = []
    for column in range(3):
        pivot = int(torch.argmax(q[:, column].abs()))
        pivots.append(pivot)
        if float(q[pivot, column]) < 0.0:
            q[:, column].mul_(-1.0)
    identity_residual = q.T @ q - torch.eye(3, dtype=torch.float64)
    coordinate_matrix = q.T @ matrix
    reconstruction = q @ coordinate_matrix
    reconstruction_relative = float((reconstruction - matrix).norm() / matrix.norm())
    if (
        float(identity_residual.abs().max()) > ORTHONORMAL_ABS_TOLERANCE
        or reconstruction_relative > ORTHONORMAL_ABS_TOLERANCE
    ):
        raise RuntimeError("CPU-float64 orthonormal basis audit failed")

    discarded: dict[str, list[str]] = {}
    reference_state = payloads["R"]["model_state_dict"]
    actor_set = set(ACTOR_NAMES)
    for name, payload in payloads.items():
        if name == "R":
            discarded[name] = []
            continue
        state = payload["model_state_dict"]
        discarded[name] = sorted(
            key
            for key in reference_state
            if key not in actor_set and not torch.equal(reference_state[key], state[key])
        )

    norms = torch.sqrt(torch.diag(gram))
    cosines = gram / torch.outer(norms, norms)
    report = {
        "construction": dict(DIRECTION_FORMULAS),
        "actor_parameter_names": list(ACTOR_NAMES),
        "actor_elements": EXPECTED_ACTOR_ELEMENTS,
        "input_actor6_sha256_float64": {
            name: vector_sha256_float64(vector) for name, vector in vectors.items()
        },
        "direction_sha256_float64": {
            name: vector_sha256_float64(directions[name]) for name in DIRECTION_NAMES
        },
        "direction_l2": {
            name: float(norms[index]) for index, name in enumerate(DIRECTION_NAMES)
        },
        "gram": [[float(value) for value in row] for row in gram.tolist()],
        "cosine": [[float(value) for value in row] for row in cosines.tolist()],
        "svd": {
            "implementation": "torch.linalg.svd CPU float64 full_matrices=False",
            "relative_rank_tolerance": SVD_RELATIVE_RANK_TOLERANCE,
            "rank": rank,
            "singular_values": [float(value) for value in singular_values],
            "condition_number": float(singular_values[0] / singular_values[-1]),
            "canonical_sign_pivots": pivots,
            "right_singular_vectors_diagnostic": [
                [float(value) for value in row] for row in vh.tolist()
            ],
        },
        "orthonormal_basis_sha256_float64": vector_sha256_float64(q.reshape(-1)),
        "orthonormality_max_abs_residual": float(identity_residual.abs().max()),
        "span_reconstruction_relative_l2": reconstruction_relative,
        "orthobasis_coordinates_of_pet_columns": [
            [float(value) for value in row] for row in coordinate_matrix.tolist()
        ],
        "non_actor_tensors_ignored_by_design": discarded,
        "only_actor6_used_for_geometry": True,
    }
    return {
        "payloads": payloads,
        "vectors": vectors,
        "directions": directions,
        "matrix": matrix,
        "q": q,
        "coordinate_matrix": coordinate_matrix,
        "report": report,
    }


def flat_direct_identities(
    selections: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    flat = [row for batch in selections for row in batch]
    if len(flat) != 512:
        raise RuntimeError("direct512 identity row-count drift")
    payload = [{key: row[key] for key in DIRECT_IDENTITY_FIELDS} for row in flat]
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if digest != EXPECTED_DIRECT_IDENTITY_SHA256:
        raise RuntimeError(f"direct512 identity SHA drift: {digest}")
    if len({(row["source"], row["member"], row["line_index"]) for row in payload}) != 512:
        raise RuntimeError("direct512 identities are not unique")
    return payload


def direct_loss_masks(identities: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {"mixed": torch.ones(512, dtype=torch.bool)}
    for source in SOURCES:
        masks[f"{source}_all"] = torch.tensor(
            [str(row["source"]) == source for row in identities], dtype=torch.bool
        )
        for bucket in ("hard", "fragile", "c34"):
            masks[f"{source}_{bucket}"] = torch.tensor(
                [
                    str(row["source"]) == source
                    and str(row["category"]) == bucket
                    for row in identities
                ],
                dtype=torch.bool,
            )
    if tuple(masks) != LOSS_NAMES:
        raise RuntimeError(f"direct loss mask order drift: {tuple(masks)}")
    observed = {name: int(mask.sum()) for name, mask in masks.items()}
    if observed != EXPECTED_GROUP_ROWS:
        raise RuntimeError(f"direct loss group row-count drift: {observed}")
    return masks


def build_cache_context(raw_payload: Mapping[str, Any]) -> dict[str, Any]:
    fixed_inputs = aggregate.verify_fixed_inputs()
    profile, profile_evidence = tie.load_profile()
    rows = tie.select_constraint_rows(profile)
    canonical_batches, constraint_cache = tie.load_canonical_constraint_batches(
        rows, raw_payload["model_config"], profile
    )
    cache_audit, selections, direct_union = aggregate.build_cache_audit()
    identities = flat_direct_identities(selections)
    masks = direct_loss_masks(identities)
    if (
        constraint_cache["rows"] != 24
        or constraint_cache["targets"] != 7
        or constraint_cache["guards"] != 17
        or constraint_cache["retained_batch_count"] != 21
    ):
        raise RuntimeError("canonical constraint-cache contract drift")
    if (
        cache_audit["aggregate512"]["cache_sha256"]
        != EXPECTED_DIRECT_CACHE_SHA256
        or cache_audit["aggregate512"]["batch_sha256"]
        != EXPECTED_DIRECT_BATCH_SHA256
    ):
        raise RuntimeError("direct512 cache identity drift")
    report = {
        "fixed_input_keys": sorted(fixed_inputs),
        "profile": profile_evidence,
        "canonical_constraints": {
            "rows": constraint_cache["rows"],
            "targets": constraint_cache["targets"],
            "guards": constraint_cache["guards"],
            "tie_count_expected_at_actual": 26,
            "canonical_batch_size": constraint_cache["canonical_batch_size"],
            "retained_batch_count": constraint_cache["retained_batch_count"],
            "identity_sha256": constraint_cache["line_identity_sha256"],
        },
        "direct512": {
            "rows": 512,
            "cache_sha256": cache_audit["aggregate512"]["cache_sha256"],
            "batch_sha256": cache_audit["aggregate512"]["batch_sha256"],
            "identity_sha256": EXPECTED_DIRECT_IDENTITY_SHA256,
            "loss_group_rows": dict(EXPECTED_GROUP_ROWS),
        },
        "validation_member_payloads_opened": False,
        "writes_performed": False,
    }
    return {
        "rows": rows,
        "canonical_batches": canonical_batches,
        "direct_selections": selections,
        "direct_union": direct_union,
        "direct_identities": identities,
        "loss_masks": masks,
        "report": report,
    }


def configure_actor6(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    parameters: dict[str, torch.nn.Parameter] = {}
    for name in ACTOR_NAMES:
        parameter = named.get(name)
        if parameter is None or parameter.dtype != torch.float32:
            raise RuntimeError(f"runtime actor6 parameter drift: {name}")
        parameter.requires_grad_(True)
        parameters[name] = parameter
    active = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if active != ACTOR_NAMES:
        raise RuntimeError(f"runtime actor6 trainable scope drift: {active}")
    return parameters


def flat_actor6_gradient(
    scalar: torch.Tensor,
    parameters: Mapping[str, torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    values = torch.autograd.grad(
        scalar,
        tuple(parameters[name] for name in ACTOR_NAMES),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
        materialize_grads=False,
    )
    pieces: list[torch.Tensor] = []
    for name, value in zip(ACTOR_NAMES, values):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"nonfinite actor6 gradient: {name}")
        pieces.append(value.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
    flat = torch.cat(pieces)
    if int(flat.numel()) != EXPECTED_ACTOR_ELEMENTS or float(flat.norm()) <= 0.0:
        raise RuntimeError("invalid flattened actor6 gradient")
    return flat


def collect_tie_gradients(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.nn.Parameter],
    canonical_batches: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    raw_predictions: dict[str, list[int]] = {}
    for canonical in canonical_batches:
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in canonical["batch"].items()
        }
        outputs = ppo.model_forward(model, batch, device)
        predictions = tie.constraint_predictions(outputs, batch)
        for item in canonical["selected"]:
            record = item["record"]
            line_sha = str(record["line_sha256"])
            prediction = predictions[int(item["local_index"])]
            if line_sha in raw_predictions:
                raise RuntimeError("canonical raw prediction repeated")
            if prediction != [int(value) for value in record["predicted_order"]]:
                raise RuntimeError(f"canonical raw prediction drift: {line_sha}")
            raw_predictions[line_sha] = prediction
        scalars, identities = tie.build_margin_constraints(
            outputs["policy_logits"], batch, canonical["selected"]
        )
        if not scalars or len(scalars) != len(identities):
            raise RuntimeError("canonical tie scalar/identity drift")
        for index, (scalar, identity) in enumerate(zip(scalars, identities)):
            collected.append(
                {
                    "identity": dict(identity),
                    "gradient": flat_actor6_gradient(
                        scalar,
                        parameters,
                        retain_graph=index + 1 < len(scalars),
                    ),
                }
            )
        del outputs, batch, scalars
    collected.sort(
        key=lambda item: (
            str(item["identity"]["source"]),
            str(item["identity"]["line_sha256"]),
            int(item["identity"]["stage"]),
            int(item["identity"]["expert_action"]),
            int(item["identity"]["tied_competitor"]),
        )
    )
    roles = [str(item["identity"]["role"]) for item in collected]
    if len(collected) != 26 or roles.count("target") != 7 or roles.count("guard") != 19:
        raise RuntimeError(f"canonical tie-count/role drift: {roles}")
    return collected


def collect_direct_loss_gradients(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.nn.Parameter],
    direct_union: Mapping[str, torch.Tensor],
    masks_cpu: Mapping[str, torch.Tensor],
    device: torch.device,
) -> list[dict[str, Any]]:
    batch = {key: value.to(device, non_blocking=True) for key, value in direct_union.items()}
    outputs = ppo.model_forward(model, batch, device)
    if outputs["policy_logits"].dtype != torch.bfloat16:
        raise RuntimeError("direct512 native policy logits are not BF16")
    per_row = frozen.ordered_nll_per_row(outputs, batch)
    effective_weights = batch["sample_weights"].float() * torch.where(
        batch["contexts"] == ppo.SKILL_ORDER_CONTEXT,
        torch.full_like(batch["sample_weights"].float(), aggregate.ORDER_CONTEXT_WEIGHT),
        torch.ones_like(batch["sample_weights"].float()),
    )
    scalars: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for name in LOSS_NAMES:
        mask = masks_cpu[name].to(device=device, dtype=torch.bool)
        denominator = effective_weights[mask].sum()
        if int(mask.sum()) != EXPECTED_GROUP_ROWS[name] or float(denominator) <= 0.0:
            raise RuntimeError(f"invalid direct512 loss group: {name}")
        scalar = (per_row[mask] * effective_weights[mask]).sum() / denominator
        if scalar.ndim != 0 or not bool(torch.isfinite(scalar)):
            raise FloatingPointError(f"nonfinite direct512 loss: {name}")
        scalars.append(scalar)
        records.append(
            {
                "name": name,
                "rows": int(mask.sum()),
                "effective_weight": float(denominator.detach().cpu()),
                "loss": float(scalar.detach().cpu()),
            }
        )
    for index, (scalar, record) in enumerate(zip(scalars, records)):
        record["gradient"] = flat_actor6_gradient(
            scalar, parameters, retain_graph=index + 1 < len(scalars)
        )
    del outputs, batch, per_row, scalars
    return records


def projection_record(vector: torch.Tensor, q: torch.Tensor) -> dict[str, Any]:
    norm = float(vector.norm())
    projected = q.T @ vector
    projected_norm = float(projected.norm())
    normalized = projected / norm
    return {
        "full_gradient_l2": norm,
        "projected_l2": projected_norm,
        "captured_l2_fraction": projected_norm / norm,
        "orthobasis_components": [float(value) for value in projected],
        "full_norm_normalized_components": [float(value) for value in normalized],
        "_normalized_tensor": normalized,
        "_raw_tensor": projected,
    }


def max_min_lp(
    target_matrix: np.ndarray,
    guard_matrix: np.ndarray,
    loss_matrix: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    rows: list[np.ndarray] = []
    for row in target_matrix:
        rows.append(np.r_[-row, 1.0])
    for row in guard_matrix:
        rows.append(np.r_[-row, 0.0])
    for row in loss_matrix:
        rows.append(np.r_[row, 0.0])
    a_ub = np.asarray(rows, dtype=np.float64)
    b_ub = np.zeros(a_ub.shape[0], dtype=np.float64)
    objective = np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    solution = optimize.linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (None, None)),
        method="highs",
    )
    if not bool(solution.success) or solution.x is None:
        raise RuntimeError(f"HiGHS max-min LP failed: {solution.status} {solution.message}")
    x = np.asarray(solution.x, dtype=np.float64)
    y = x[:3]
    t = float(x[3])
    target_slopes = target_matrix @ y
    guard_slopes = guard_matrix @ y
    loss_slopes = loss_matrix @ y
    violations = np.r_[
        t - target_slopes,
        -guard_slopes,
        loss_slopes,
        np.abs(y) - 1.0,
    ]
    maximum_violation = float(max(0.0, float(violations.max(initial=0.0))))
    report = {
        "implementation": "scipy.optimize.linprog(method='highs') float64",
        "variables": "orthobasis_c0,c1,c2,t",
        "coefficient_bounds": [-1.0, 1.0],
        "objective": "maximize minimum full-gradient-normalized target slope",
        "status": int(solution.status),
        "message": str(solution.message),
        "success": bool(solution.success),
        "iterations": int(solution.nit),
        "objective_t": t,
        "orthobasis_coefficients": [float(value) for value in y],
        "primal_max_inequality_violation": maximum_violation,
        "target_min_normalized_slope": float(target_slopes.min()),
        "guard_min_normalized_slope": float(guard_slopes.min()),
        "loss_max_normalized_slope": float(loss_slopes.max()),
        "direct_residual_recomputation_is_authority": True,
    }
    return solution, report


def farkas_certificate(
    target_matrix: np.ndarray,
    guard_matrix: np.ndarray,
    loss_matrix: np.ndarray,
) -> dict[str, Any]:
    desired = np.vstack([target_matrix, guard_matrix, -loss_matrix])
    count = desired.shape[0]
    objective = np.linspace(1.0, 2.0, count, dtype=np.float64)
    a_eq = np.vstack(
        [desired.T, np.r_[np.ones(target_matrix.shape[0]), np.zeros(count - target_matrix.shape[0])]]
    )
    b_eq = np.r_[np.zeros(3, dtype=np.float64), 1.0]
    solution = optimize.linprog(
        objective,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=tuple((0.0, None) for _ in range(count)),
        method="highs",
    )
    if not bool(solution.success) or solution.x is None:
        return {
            "available": False,
            "status": int(solution.status),
            "message": str(solution.message),
        }
    multipliers = np.asarray(solution.x, dtype=np.float64)
    stationarity = desired.T @ multipliers
    target_mass = float(multipliers[: target_matrix.shape[0]].sum())
    minimum_multiplier = float(multipliers.min())
    verified = bool(
        float(np.max(np.abs(stationarity))) <= FARKAS_ABS_TOLERANCE
        and abs(target_mass - 1.0) <= FARKAS_ABS_TOLERANCE
        and minimum_multiplier >= -FARKAS_ABS_TOLERANCE
    )
    nonzero = [
        {"constraint_index": index, "multiplier": float(value)}
        for index, value in enumerate(multipliers)
        if float(value) > FARKAS_ABS_TOLERANCE
    ]
    return {
        "available": True,
        "verified": verified,
        "theorem": (
            "nonnegative lambda with H^T lambda=0 and target-lambda mass=1 "
            "certifies infeasibility of target>=1, guard>=0, -loss>=0"
        ),
        "status": int(solution.status),
        "message": str(solution.message),
        "stationarity_max_abs": float(np.max(np.abs(stationarity))),
        "target_multiplier_mass": target_mass,
        "minimum_multiplier": minimum_multiplier,
        "nonzero_multipliers": nonzero,
        "constraint_index_ranges": {
            "targets": [0, int(target_matrix.shape[0])],
            "guards": [
                int(target_matrix.shape[0]),
                int(target_matrix.shape[0] + guard_matrix.shape[0]),
            ],
            "loss_descent": [
                int(target_matrix.shape[0] + guard_matrix.shape[0]),
                count,
            ],
        },
    }


def solve_geometry(
    q: torch.Tensor,
    coordinate_matrix: torch.Tensor,
    ties: Sequence[Mapping[str, Any]],
    losses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tie_reports: list[dict[str, Any]] = []
    target_rows: list[list[float]] = []
    guard_rows: list[list[float]] = []
    for item in ties:
        projection = projection_record(item["gradient"], q)
        normalized = projection.pop("_normalized_tensor")
        projection.pop("_raw_tensor")
        identity = item["identity"]
        role = str(identity["role"])
        row = [float(value) for value in normalized]
        if role == "target":
            target_rows.append(row)
        elif role == "guard":
            guard_rows.append(row)
        else:
            raise RuntimeError(f"unknown tie role: {role}")
        tie_reports.append(
            {
                "role": role,
                "source": str(identity["source"]),
                "line_sha256": str(identity["line_sha256"]),
                "stage": int(identity["stage"]),
                "expert_action": int(identity["expert_action"]),
                "tied_competitor": int(identity["tied_competitor"]),
                **projection,
            }
        )

    loss_reports: list[dict[str, Any]] = []
    loss_rows: list[list[float]] = []
    for item in losses:
        projection = projection_record(item["gradient"], q)
        normalized = projection.pop("_normalized_tensor")
        projection.pop("_raw_tensor")
        loss_rows.append([float(value) for value in normalized])
        loss_reports.append(
            {
                "name": item["name"],
                "rows": item["rows"],
                "effective_weight": item["effective_weight"],
                "raw_loss": item["loss"],
                **projection,
            }
        )

    target_matrix = np.asarray(target_rows, dtype=np.float64)
    guard_matrix = np.asarray(guard_rows, dtype=np.float64)
    loss_matrix = np.asarray(loss_rows, dtype=np.float64)
    if target_matrix.shape != (7, 3) or guard_matrix.shape != (19, 3):
        raise RuntimeError("projected tie matrix shape drift")
    if loss_matrix.shape != (13, 3):
        raise RuntimeError("projected loss matrix shape drift")

    solution, lp_report = max_min_lp(target_matrix, guard_matrix, loss_matrix)
    y = np.asarray(solution.x[:3], dtype=np.float64)
    bounded_signal = bool(
        lp_report["primal_max_inequality_violation"] <= LP_PRIMAL_ABS_TOLERANCE
        and lp_report["objective_t"] > STRICT_TARGET_SLOPE_TOLERANCE
    )
    norm = float(np.linalg.norm(y))
    if not math.isfinite(norm):
        raise RuntimeError("max-min coefficient vector has nonfinite norm")
    if not bounded_signal:
        certificate = farkas_certificate(target_matrix, guard_matrix, loss_matrix)
        if not (
            certificate.get("available") is True
            and certificate.get("verified") is True
        ):
            raise RuntimeError(
                "no strict PET span signal, but no verified Farkas certificate "
                "was available"
            )
        return {
            "decision": "NO_SIGNAL",
            "feasible": False,
            "normalization": (
                "each projected row divided by its full actor6 gradient L2; signs "
                "remain exact and target max-min is a projected cosine margin"
            ),
            "lp": lp_report,
            "selected_unit_direction": None,
            "infeasibility_certificate": certificate,
            "tie_projections": tie_reports,
            "loss_projections": loss_reports,
        }
    if norm <= 0.0:
        raise RuntimeError("positive max-min objective has zero coefficient norm")
    unit_y = y / norm
    unit_target = target_matrix @ unit_y
    unit_guard = guard_matrix @ unit_y
    unit_loss = loss_matrix @ unit_y
    feasible = bool(
        lp_report["primal_max_inequality_violation"] <= LP_PRIMAL_ABS_TOLERANCE
        and float(unit_target.min()) > STRICT_TARGET_SLOPE_TOLERANCE
        and float(unit_guard.min()) >= -LP_PRIMAL_ABS_TOLERANCE
        and float(unit_loss.max()) <= LP_PRIMAL_ABS_TOLERANCE
    )
    pet_coefficients = torch.linalg.solve(
        coordinate_matrix,
        torch.from_numpy(unit_y).to(torch.float64),
    )
    for report in tie_reports:
        normalized = np.asarray(report["full_norm_normalized_components"], dtype=np.float64)
        raw = np.asarray(report["orthobasis_components"], dtype=np.float64)
        report["selected_unit_normalized_slope"] = float(normalized @ unit_y)
        report["selected_unit_raw_margin_slope"] = float(raw @ unit_y)
    for report in loss_reports:
        normalized = np.asarray(report["full_norm_normalized_components"], dtype=np.float64)
        raw = np.asarray(report["orthobasis_components"], dtype=np.float64)
        report["selected_unit_normalized_loss_slope"] = float(normalized @ unit_y)
        report["selected_unit_raw_loss_slope"] = float(raw @ unit_y)

    if not feasible:
        raise RuntimeError(
            "bounded LP reported a strict signal, but direct unit-direction "
            "residual verification failed"
        )
    return {
        "decision": "SPAN_SIGNAL",
        "feasible": True,
        "normalization": (
            "each projected row divided by its full actor6 gradient L2; signs "
            "remain exact and target max-min is a projected cosine margin"
        ),
        "lp": lp_report,
        "selected_unit_direction": {
            "orthobasis_coefficients": [float(value) for value in unit_y],
            "pet_column_coefficients": {
                name: float(pet_coefficients[index])
                for index, name in enumerate(DIRECTION_NAMES)
            },
            "actor6_l2": 1.0,
            "target_min_normalized_slope": float(unit_target.min()),
            "guard_min_normalized_slope": float(unit_guard.min()),
            "loss_max_normalized_slope": float(unit_loss.max()),
            "target_strict_positive": bool(
                float(unit_target.min()) > STRICT_TARGET_SLOPE_TOLERANCE
            ),
            "guards_nonnegative_with_tolerance": bool(
                float(unit_guard.min()) >= -LP_PRIMAL_ABS_TOLERANCE
            ),
            "losses_nonpositive_with_tolerance": bool(
                float(unit_loss.max()) <= LP_PRIMAL_ABS_TOLERANCE
            ),
        },
        "infeasibility_certificate": None,
        "tie_projections": tie_reports,
        "loss_projections": loss_reports,
    }


def common_report(tool: Mapping[str, Any], static: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "tool": dict(tool),
        "frozen_tie_tool": dict(TIE_EVIDENCE),
        "checkpoint_inputs": dict(CHECKPOINT_EVIDENCE),
        "static_audit": dict(static),
        "contract": {
            "base": "raw full U468 R",
            "directions": dict(DIRECTION_FORMULAS),
            "actor6_only": True,
            "basis_arithmetic": "CPU float64 Gram/SVD",
            "tie_gradients": "26 canonical native-BF16 exact margins at raw R",
            "tie_requirements": {"targets": "slope>0", "guards": "slope>=0"},
            "loss_gradients": (
                "direct512 exact ordered NLL mixed plus each source "
                "all/hard/fragile/c34 at raw R"
            ),
            "loss_requirement": "slope<=0",
            "solver": "deterministic scipy.optimize.linprog(method='highs')",
            "geometry_only": True,
            "endpoint_construction": False,
            "parameter_update": False,
            "optimizer": False,
            "backward": False,
            "checkpoint_or_result_write": False,
            "validation": False,
            "promotion": False,
            "submission": False,
            "stdout_only": True,
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
    self_path = Path(__file__).resolve()
    if self_path.parent != TOOLS:
        raise RuntimeError("geometry tool must remain in repository tools directory")
    tool = regular_evidence(self_path, None, "geometry tool")
    static = ast_audit()
    common = common_report(tool, static)
    if args.mode == "static-audit":
        if args.device != "cpu":
            raise ValueError("static-audit requires --device cpu")
        emit_json(
            {**common, "status": "zero_write_static_audit_passed"}
        )
        return

    basis = build_basis_geometry()
    caches = build_cache_context(basis["payloads"]["R"])
    audited = {
        **common,
        "basis": basis["report"],
        "cache": caches["report"],
    }
    if args.mode == "cache-audit":
        if args.device != "cpu":
            raise ValueError("cache-audit requires --device cpu")
        emit_json(
            {**audited, "status": "zero_write_cache_audit_passed"}
        )
        return

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual geometry probe requires CUDA")
    random.seed(EXECUTION_SEED)
    torch.manual_seed(EXECUTION_SEED)
    torch.cuda.manual_seed_all(EXECUTION_SEED)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    model, _ = frozen.load_raw_u468(device)
    model.eval()
    parameters = configure_actor6(model)
    model_hash_before = ppo.model_state_sha256(model)
    if model_hash_before != frozen.BASE_MODEL_SHA256:
        raise RuntimeError("raw U468 runtime model-state SHA drift")
    runtime_actor = torch.cat(
        [parameters[name].detach().cpu().reshape(-1).to(torch.float64) for name in ACTOR_NAMES]
    )
    if not torch.equal(runtime_actor, basis["vectors"]["R"]):
        raise RuntimeError("runtime raw actor6 is not tensor-exact to R")

    ties = collect_tie_gradients(
        model, parameters, caches["canonical_batches"], device
    )
    losses = collect_direct_loss_gradients(
        model, parameters, caches["direct_union"], caches["loss_masks"], device
    )
    geometry = solve_geometry(
        basis["q"], basis["coordinate_matrix"], ties, losses
    )
    model_hash_after = ppo.model_state_sha256(model)
    if model_hash_after != model_hash_before:
        raise RuntimeError("geometry probe changed raw model state")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("autograd.grad materialized parameter .grad buffers")

    post = {
        "tool_sha256_unchanged": sha256_file(self_path) == tool["sha256"],
        "tie_tool_sha256_unchanged": sha256_file(TIE_TOOL) == TIE_TOOL_SHA256,
        "checkpoint_sha256_unchanged": {
            name: sha256_file(path) == digest
            for name, (path, digest) in CHECKPOINT_SPECS.items()
        },
    }
    if not (
        post["tool_sha256_unchanged"]
        and post["tie_tool_sha256_unchanged"]
        and all(post["checkpoint_sha256_unchanged"].values())
    ):
        raise RuntimeError(f"postrun SHA binding drift: {post}")
    final = {
        **audited,
        "status": "completed_zero_write_pet_cone_geometry",
        "device": "cuda",
        "seed": EXECUTION_SEED,
        "geometry": geometry,
        "integrity": {
            "runtime_model_state_sha256_before": model_hash_before,
            "runtime_model_state_sha256_after": model_hash_after,
            "model_weights_unchanged": True,
            "actor6_scope_exact": True,
            "parameter_grad_buffers_materialized": 0,
            "optimizer_instances": 0,
            "backward_calls": 0,
            "endpoint_constructions": 0,
            "checkpoint_writes": 0,
            "result_writes": 0,
            "validation_member_payloads_opened": False,
            "submission_performed": False,
        },
        "postrun_reverification": post,
    }
    emit_json(final)


if __name__ == "__main__":
    main()

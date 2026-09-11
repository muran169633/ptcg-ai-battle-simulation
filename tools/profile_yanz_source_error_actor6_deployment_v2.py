#!/usr/bin/env python3
"""Build a train-only row profile under the real packaged deployment semantics.

This v2 profiler intentionally does not reuse the CUDA/BF16 batched inference
path from v1.  It binds the Alakazam hybrid-order submission entrypoint and its
vendored policy runtime, expands the frozen raw17 BC source into that runtime's
real 61-class model, and evaluates exactly one observation at a time on CPU in
FP32.  Only ``train/*.jsonl`` members of the two fixed archives are opened.

Full profiling is recoverable through immutable hash-bound JSON cache shards.
No model artifact, evaluation split, external upload, or submission is part of
this program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import orjson
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
SOURCE = ROOT / "artifacts/gold8_recent7_20260808/alakazam_control/specialist_bc/best.pt"
YANZ = ROOT / (
    "data/yanz_alakazam_live_20260810_v1/snapshot_20260810T0433Z/"
    "policies/yanzhou06_alakazam_live_55394520.zip"
)
OLD = ROOT / "data/gold8_recent7_20260808/archives/alakazam_control.zip"
HYBRID_TEMPLATE = ROOT / "submission_templates/ptcg_ppo_alakazam_hybrid_order_v1"
HYBRID_MAIN = HYBRID_TEMPLATE / "main.py"
CONTRACT = HYBRID_TEMPLATE / "CONTRACT.json"
RUNTIME = ROOT / "submission_templates/ptcg_ppo_alakazam_standard_pl_v1/policy_runtime.py"
MAIN_RUNTIME_ANCHOR = ROOT / (
    "submissions/ptcg_ppo_alakazam_bc12p5_u10_87p5_20260810/policy_runtime.py"
)
DECK_CSV = ROOT / (
    "data/gold8_recent7_20260808/decks/"
    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf.csv"
)

SCHEMA = "ptcg-yanz-source-error-actor6-deployment-profile-v2"
CACHE_SCHEMA = "ptcg-yanz-deployment-profile-cache-shard-v2"
SOURCE_SHA256 = "f5500086c16a02c19f3f2abce5e144446bd079248fd9fc3f4a620f3d079c7626"
SOURCE_BITWISE_STATE_SHA256 = (
    "b0bfe317017a9d4f3cbd367cd21ff96c7fdc60abd1924a6d8edbdf7b32917f91"
)
SOURCE_EXPANDED61_BITWISE_STATE_SHA256 = (
    "efefd3d5c2260e771c7ce7a67a67dbb585acce3ebb43742d247594922a734a62"
)
YANZ_SHA256 = "bbbb6f69b809c60e900f7d2992333b645d117b2a6c836642fecca0e1f4226720"
OLD_SHA256 = "4bd0de193cfb88b060435f84bbb5dc83a9eff3ab1aa4a498ef0859d2346f3c6c"
HYBRID_MAIN_SHA256 = (
    "b35c7b0976a9a377b7f3c61c69aa15ef477db96b1b6bbdeb4ceb0305312b3a2a"
)
RUNTIME_SHA256 = "fe7182a588962fd9c5e07e9e62433beaa1c5b3fa047bb8170e22435e4f93e997"
CONTRACT_SHA256 = "8cb9c929e21f5b74efa84ccc7cca9dc3797cded3a35f8e37f9c8ebef522cf205"
DECK_CSV_SHA256 = "0598646548d081832ec311c15fdc369b32c6f5e63175b0cfd1904d21fd082451"
DECK_HASH = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
EXPECTED_YANZ_RAW_ROWS = 1725
EXPECTED_OLD_RAW_ROWS = 59875
COUNT_CLASSES = 61
SKILL_ORDER_CONTEXT = 34
CACHE_CHUNK_ROWS = 512
BOUNDARY_STEP_SIZES = (123, 124, 124)
BROAD_PER_STEP = 32
MIN_CONTEXT34_BOUNDARY_PER_STEP = 4
SELECTION_SALT = "ptcg-yanz-actor6-deployment-profile-selection-v2"


class ProtocolError(RuntimeError):
    """A fail-closed deployment-profile contract violation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_script_sha(expected: str) -> None:
    actual = file_sha256(SCRIPT)
    if actual != expected:
        raise ProtocolError(f"profiler script changed during run: {actual} != {expected}")


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


def load_json_strict(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ProtocolError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON root is not an object: {path}")
    return value


def bitwise_model_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ptcg-model-state-bitwise-v1\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("model state must map string names to tensors")
        tensor = value.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(4, "big"))
        digest.update(dtype_bytes)
        digest.update(len(tensor.shape).to_bytes(4, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=True))
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} is not an object")
    return value


def validate_static_bindings() -> dict[str, Any]:
    files = {
        "source_checkpoint": (SOURCE, SOURCE_SHA256),
        "yanz_train_archive": (YANZ, YANZ_SHA256),
        "old_train_archive": (OLD, OLD_SHA256),
        "hybrid_main": (HYBRID_MAIN, HYBRID_MAIN_SHA256),
        "policy_runtime": (RUNTIME, RUNTIME_SHA256),
        "template_contract": (CONTRACT, CONTRACT_SHA256),
        "deck_csv": (DECK_CSV, DECK_CSV_SHA256),
    }
    bindings: dict[str, Any] = {}
    for name, (path, expected) in files.items():
        actual = file_sha256(path)
        if actual != expected:
            raise ProtocolError(f"static binding drift {name}: {actual} != {expected}")
        bindings[name] = {"path": str(path.resolve()), "sha256": actual}

    contract = load_json_strict(CONTRACT)
    if contract.get("template_version") != "ptcg-ppo-alakazam-hybrid-order-v1":
        raise ProtocolError("hybrid template contract version drift")
    template_files = _require_mapping(contract.get("template_files"), "template files")
    expected_template_files = {
        "main.py": HYBRID_MAIN_SHA256,
        "policy_runtime.py": RUNTIME_SHA256,
        "deck.csv": DECK_CSV_SHA256,
    }
    for name, expected in expected_template_files.items():
        if template_files.get(name) != expected:
            raise ProtocolError(f"hybrid template file binding drift: {name}")
    decode = _require_mapping(contract.get("decode"), "hybrid decode contract")
    expected_decode = {
        "order_mode": "hybrid",
        "cardinality": "legal_masked_greedy",
        "selection": "greedy_plackett_luce_without_replacement",
        "canonicalize_order": False,
        "sort_selected_indices": "all contexts except 34",
        "preserve_order_contexts": [34],
    }
    for key, expected in expected_decode.items():
        if decode.get(key) != expected:
            raise ProtocolError(f"hybrid decode contract drift: {key}")
    bindings["runtime_semantics"] = {
        "device": "cpu",
        "dtype": "torch.float32",
        "batch_size": 1,
        "autocast": False,
        "inference_mode": True,
        "torch_num_threads": 1,
        "torch_version": torch.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "count_classes": COUNT_CLASSES,
        "feature_input": (
            "observation plus current.yourIndex, fixed deck hash, empty team, "
            "empty action, terminal_reward=0, sample_weight=1"
        ),
        "decoder": dict(decode),
    }
    return bindings


def load_bound_runtime() -> ModuleType:
    if file_sha256(RUNTIME) != RUNTIME_SHA256:
        raise ProtocolError("policy runtime changed before import")
    module_name = "ptcg_bound_alakazam_policy_runtime_deployment_v2"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, RUNTIME)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot construct bound policy runtime import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if int(module.MAX_ACTION_COUNT) != 60:
        raise ProtocolError("bound policy runtime MAX_ACTION_COUNT drift")
    return module


def instantiate_deployment_source(
    runtime: ModuleType,
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any], dict[str, Any]]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ProtocolError("source checkpoint root drift")
    source_state = checkpoint.get("model_state_dict")
    if not isinstance(source_state, Mapping) or len(source_state) != 80:
        raise ProtocolError("source raw17 state drift")
    if bitwise_model_state_sha256(source_state) != SOURCE_BITWISE_STATE_SHA256:
        raise ProtocolError("source raw17 bitwise state drift")
    raw_config = _require_mapping(checkpoint.get("config"), "source config")
    required_config = (
        "hash_size",
        "categorical_dim",
        "model_dim",
        "layers",
        "heads",
        "dropout",
        "max_state_entities",
        "entity_fields",
        "option_fields",
    )
    if any(name not in raw_config for name in required_config):
        raise ProtocolError("source config is missing deployment fields")
    config = {
        name: (
            float(raw_config[name]) if name == "dropout" else int(raw_config[name])
        )
        for name in required_config
    }
    model = runtime.EntityOptionPolicy(
        hash_size=config["hash_size"],
        categorical_dim=config["categorical_dim"],
        model_dim=config["model_dim"],
        layers=config["layers"],
        heads=config["heads"],
        dropout=config["dropout"],
        max_state_entities=config["max_state_entities"],
    )
    runtime_state = model.state_dict()
    if len(runtime_state) != 80 or runtime_state["count_head.2.weight"].shape[0] != 61:
        raise ProtocolError("bound runtime is not the expected 80-tensor expanded61 model")
    expanded: dict[str, torch.Tensor] = {}
    for name, template in runtime_state.items():
        source_value = source_state.get(name)
        if not isinstance(source_value, torch.Tensor):
            raise ProtocolError(f"source tensor missing: {name}")
        if name == "count_head.2.weight":
            if source_value.shape[0] != 17 or template.shape[0] != 61:
                raise ProtocolError("source/runtime count weight shape drift")
            value = torch.zeros_like(template)
            value[:17].copy_(source_value)
        elif name == "count_head.2.bias":
            if source_value.shape[0] != 17 or template.shape[0] != 61:
                raise ProtocolError("source/runtime count bias shape drift")
            value = torch.full_like(template, -10.0)
            value[:17].copy_(source_value)
        else:
            if source_value.shape != template.shape:
                raise ProtocolError(f"source/runtime tensor shape drift: {name}")
            value = source_value.detach().cpu().clone()
        expanded[name] = value
    model.load_state_dict(expanded, strict=True)
    model.to(torch.device("cpu")).eval().requires_grad_(False)
    if model.count_head[-1].out_features != COUNT_CLASSES:
        raise ProtocolError("deployment source count head is not 61-class")
    if any(value.device.type != "cpu" for value in model.parameters()):
        raise ProtocolError("deployment source escaped CPU")
    if any(value.dtype != torch.float32 for value in model.parameters()):
        raise ProtocolError("deployment source is not FP32")
    if torch.get_num_threads() != 1:
        raise ProtocolError("deployment profiler did not lock one CPU thread")
    expanded_state_sha256 = bitwise_model_state_sha256(model.state_dict())
    if expanded_state_sha256 != SOURCE_EXPANDED61_BITWISE_STATE_SHA256:
        raise ProtocolError("expanded61 deployment source state drift")
    audit = {
        "raw_tensor_count": len(source_state),
        "tensor_count": len(model.state_dict()),
        "raw_count_classes": 17,
        "expanded_count_classes": 61,
        "raw_model_state_sha256_algorithm": "ptcg-model-state-bitwise-v1",
        "raw_model_state_sha256": bitwise_model_state_sha256(source_state),
        "expanded_model_state_sha256_algorithm": "ptcg-model-state-bitwise-v1",
        "expanded_model_state_sha256": expanded_state_sha256,
        "all_parameters_cpu": True,
        "all_parameters_float32": True,
        "training": False,
        "requires_grad": False,
    }
    return checkpoint, model, config, audit


def decode_hybrid(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    policy_logits = outputs.get("policy_logits")
    count_logits = outputs.get("count_logits")
    if not isinstance(policy_logits, torch.Tensor) or not isinstance(
        count_logits, torch.Tensor
    ):
        raise ProtocolError("decoder outputs missing")
    if policy_logits.shape[0] != 1 or count_logits.shape[0] != 1:
        raise ProtocolError("deployment decoder requires batch=1")
    if count_logits.ndim != 2 or count_logits.shape[1] != COUNT_CLASSES:
        raise ProtocolError("deployment decoder requires a real 61-class count head")
    option_mask = batch.get("option_mask")
    if not isinstance(option_mask, torch.Tensor) or option_mask.shape[0] != 1:
        raise ProtocolError("deployment option mask batch drift")
    option_count = int(option_mask[0].sum())
    minimum = int(batch["min_counts"][0])
    maximum = int(batch["max_counts"][0])
    context = int(batch["contexts"][0])
    if not (0 <= minimum <= maximum <= option_count <= policy_logits.shape[1]):
        raise ProtocolError("deployment legal bounds drift")
    if minimum == maximum:
        count = minimum
    else:
        values = torch.arange(COUNT_CLASSES, device=count_logits.device)
        allowed = (values >= minimum) & (values <= maximum)
        count = int(count_logits[0].float().masked_fill(~allowed, -1e9).argmax())
    logits = policy_logits[0, :option_count].float()
    remaining = torch.ones(option_count, dtype=torch.bool, device=logits.device)
    raw_action: list[int] = []
    for _ in range(count):
        chosen = int(logits.masked_fill(~remaining, -1e9).argmax())
        raw_action.append(chosen)
        remaining[chosen] = False
    hybrid_action = raw_action if context == SKILL_ORDER_CONTEXT else sorted(raw_action)
    if (
        len(hybrid_action) < minimum
        or len(hybrid_action) > maximum
        or len(hybrid_action) != len(set(hybrid_action))
        or any(index < 0 or index >= option_count for index in hybrid_action)
    ):
        raise ProtocolError("deployment decoder produced an illegal action")
    result = {
        "count": count,
        "raw_action": raw_action,
        "hybrid_action": hybrid_action,
        "context": context,
        "option_count": option_count,
        "minimum": minimum,
        "maximum": maximum,
    }
    return result


def actual_hybrid_main_action(
    observation: Mapping[str, Any],
    runtime: ModuleType,
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> list[int]:
    """Call the hash-bound main.py itself on one train-derived observation."""

    if file_sha256(HYBRID_MAIN) != HYBRID_MAIN_SHA256:
        raise ProtocolError("hybrid main changed before fixture execution")
    if file_sha256(RUNTIME) != RUNTIME_SHA256:
        raise ProtocolError("policy runtime changed before fixture execution")
    if file_sha256(MAIN_RUNTIME_ANCHOR) != RUNTIME_SHA256:
        raise ProtocolError("fixture runtime anchor is not the bound runtime")
    anchor_deck = MAIN_RUNTIME_ANCHOR.parent / "deck.csv"
    if file_sha256(anchor_deck) != DECK_CSV_SHA256:
        raise ProtocolError("fixture deck is not the bound Alakazam deck")
    module_name = (
        "ptcg_bound_alakazam_hybrid_main_fixture_"
        + hashlib.sha256(canonical_json_bytes(observation)).hexdigest()[:16]
    )
    spec = importlib.util.spec_from_file_location(module_name, HYBRID_MAIN)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot construct bound hybrid main import")
    module = importlib.util.module_from_spec(spec)
    previous_alias = sys.modules.get("policy_runtime")
    previous_runtime_file = runtime.__file__
    try:
        runtime.__file__ = str(MAIN_RUNTIME_ANCHOR)
        sys.modules["policy_runtime"] = runtime
        spec.loader.exec_module(module)
    finally:
        runtime.__file__ = previous_runtime_file
        if previous_alias is None:
            sys.modules.pop("policy_runtime", None)
        else:
            sys.modules["policy_runtime"] = previous_alias
    module._MODEL = model
    module._MODEL_CONFIG = dict(config)
    with torch.inference_mode():
        action = module.agent(dict(observation))
    if not isinstance(action, list) or not all(isinstance(value, int) for value in action):
        raise ProtocolError("bound hybrid main fixture returned a non-integer action")
    return action


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def decision_margins(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    expert_order: Sequence[int],
) -> dict[str, Any]:
    policy_logits = outputs["policy_logits"][0].float()
    count_logits = outputs["count_logits"][0].float()
    option_count = int(batch["option_mask"][0].sum())
    minimum = int(batch["min_counts"][0])
    maximum = int(batch["max_counts"][0])
    context = int(batch["contexts"][0])
    expert_count = len(expert_order)
    if not (minimum <= expert_count <= maximum <= option_count):
        raise ProtocolError("expert action violates deployment bounds")

    if minimum == maximum:
        count_margin = math.inf
    else:
        allowed = torch.arange(COUNT_CLASSES) >= minimum
        allowed &= torch.arange(COUNT_CLASSES) <= maximum
        alternatives = allowed.clone()
        alternatives[expert_count] = False
        count_margin = (
            float(count_logits[expert_count] - count_logits[alternatives].max())
            if alternatives.any()
            else math.inf
        )

    logits = policy_logits[:option_count]
    if context == SKILL_ORDER_CONTEXT:
        if not expert_order:
            policy_margin = math.inf
        else:
            remaining = torch.ones(option_count, dtype=torch.bool)
            values: list[float] = []
            for chosen in expert_order:
                chosen_index = int(chosen)
                if not 0 <= chosen_index < option_count or not remaining[chosen_index]:
                    raise ProtocolError("expert ordered action is illegal")
                alternatives = remaining.clone()
                alternatives[chosen_index] = False
                if alternatives.any():
                    values.append(
                        float(logits[chosen_index] - logits[alternatives].max())
                    )
                remaining[chosen_index] = False
            policy_margin = min(values, default=math.inf)
    elif not expert_order or len(expert_order) == option_count:
        policy_margin = math.inf
    else:
        target = torch.zeros(option_count, dtype=torch.bool)
        target[list(expert_order)] = True
        if not target.any() or target.all():
            policy_margin = math.inf
        else:
            policy_margin = float(logits[target].min() - logits[~target].max())
    deployment_margin = min(count_margin, policy_margin)
    if deployment_margin == count_margin and deployment_margin == policy_margin:
        limiting = "tie"
    elif deployment_margin == count_margin:
        limiting = "count"
    else:
        limiting = "policy"
    return {
        "policy_hybrid_margin": _finite_or_none(policy_margin),
        "count_margin": _finite_or_none(count_margin),
        "deployment_margin": _finite_or_none(deployment_margin),
        "deployment_margin_limiting_head": limiting,
    }


def _chunk_digest(member: str, items: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ptcg-train-raw-chunk-v2\0")
    digest.update(member.encode("utf-8"))
    digest.update(b"\0")
    for _, identity in items:
        digest.update(str(identity["member_line_number"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(identity["raw_line_sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(identity["line_sha256"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def iter_train_chunks(
    path: Path, source: str, chunk_size: int = CACHE_CHUNK_ROWS
) -> Iterable[dict[str, Any]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not members:
            raise ProtocolError(f"no train JSONL members: {path}")
        for member in members:
            pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
            with archive.open(member) as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = orjson.loads(line)
                    if not isinstance(row, dict):
                        raise ProtocolError(f"row is not an object: {path}:{member}")
                    if str(row.get("split", "")) != "train":
                        raise ProtocolError(f"non-train row in {path}:{member}")
                    if str(row.get("deck_hash", "")) != DECK_HASH:
                        raise ProtocolError(f"deck drift in {path}:{member}")
                    identity = {
                        "archive_member": member,
                        "member_line_number": line_number,
                        "raw_line_sha256": hashlib.sha256(line).hexdigest(),
                        "line_sha256": hashlib.sha256(
                            line.rstrip(b"\r\n")
                        ).hexdigest(),
                        "line_sha256_algorithm": "sha256(raw_line_rstrip_CR_LF)",
                    }
                    pending.append((row, identity))
                    if len(pending) == chunk_size:
                        yield {
                            "source": source,
                            "archive_member": member,
                            "first_line": int(pending[0][1]["member_line_number"]),
                            "last_line": int(pending[-1][1]["member_line_number"]),
                            "raw_chunk_sha256": _chunk_digest(member, pending),
                            "items": pending,
                        }
                        pending = []
            if pending:
                yield {
                    "source": source,
                    "archive_member": member,
                    "first_line": int(pending[0][1]["member_line_number"]),
                    "last_line": int(pending[-1][1]["member_line_number"]),
                    "raw_chunk_sha256": _chunk_digest(member, pending),
                    "items": pending,
                }


def feature_visible_signature(feature: Mapping[str, Any]) -> str:
    visible = {
        "global_fields": feature["global_fields"],
        "global_numeric": feature["global_numeric"],
        "entity_fields": feature["entity_fields"],
        "entity_numeric": feature["entity_numeric"],
        "option_fields": feature["option_fields"],
        "option_numeric": feature["option_numeric"],
        "min_count": feature["min_count"],
        "max_count": feature["max_count"],
        "context": feature["context"],
    }
    return hashlib.sha256(
        orjson.dumps(visible, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def evaluate_row(
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    source_name: str,
    runtime: ModuleType,
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    observation = row.get("observation")
    if not isinstance(observation, dict):
        raise ProtocolError("train row observation missing")
    select = observation.get("select")
    if not isinstance(select, dict):
        raise ProtocolError("train decision has no select object")
    expert_order = [int(value) for value in row.get("action", [])]
    current = observation.get("current") or {}
    inference_row = {
        "observation": observation,
        "seat": int(current.get("yourIndex", 0) or 0),
        "deck_hash": DECK_HASH,
        "team_name": "",
        "action": [],
        "terminal_reward": 0.0,
        "sample_weight": 1.0,
    }
    feature = runtime.featurize_row(
        inference_row,
        hash_size=int(config["hash_size"]),
        max_state_entities=int(config["max_state_entities"]),
    )
    if feature is None:
        raise ProtocolError(f"deployment main featurization failed: {identity['line_sha256']}")
    batch = runtime.collate_decisions(
        [feature],
        max_state_entities=int(config["max_state_entities"]),
        entity_fields=int(config["entity_fields"]),
        option_fields=int(config["option_fields"]),
    )
    if any(value.shape[0] != 1 for value in batch.values()):
        raise ProtocolError("deployment collator did not produce batch=1")
    if any(value.device.type != "cpu" for value in batch.values()):
        raise ProtocolError("deployment batch escaped CPU")
    with torch.inference_mode():
        outputs = model(batch)
    if outputs["policy_logits"].dtype != torch.float32:
        raise ProtocolError("deployment policy logits are not FP32")
    decoded = decode_hybrid(outputs, batch)
    if not (
        decoded["minimum"] <= len(expert_order) <= decoded["maximum"]
        and all(0 <= value < decoded["option_count"] for value in expert_order)
        and len(expert_order) == len(set(expert_order))
    ):
        raise ProtocolError(f"expert action is illegal: {identity['line_sha256']}")
    margins = decision_margins(outputs, batch, expert_order)
    raw_action = decoded["raw_action"]
    hybrid_action = decoded["hybrid_action"]
    return {
        **dict(identity),
        "source": source_name,
        "decision_key": (
            f"{row.get('episode_id')}:"
            f"{row.get('observation_step_index')}->"
            f"{row.get('action_step_index')}"
        ),
        "episode_id": str(row.get("episode_id", "")),
        "episode_uuid": str(row.get("episode_uuid", "")),
        "visible_signature": feature_visible_signature(feature),
        "context": int(feature["context"]),
        "team_name": str(row.get("team_name", "")),
        "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
        "expert_action_order": expert_order,
        "source_policy_action": [int(value) for value in raw_action],
        "source_hybrid_action": [int(value) for value in hybrid_action],
        "source_set_correct": sorted(raw_action) == sorted(expert_order),
        "source_hybrid_correct": hybrid_action == expert_order,
        "source_count_correct": int(decoded["count"]) == len(expert_order),
        **margins,
    }


def make_cache_payload(
    *,
    source: str,
    archive_sha256: str,
    archive_member: str,
    first_line: int,
    last_line: int,
    raw_chunk_sha256: str,
    records: Sequence[Mapping[str, Any]],
    cache_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    compact_records = [dict(record) for record in records]
    records_sha256 = hashlib.sha256(canonical_json_bytes(compact_records)).hexdigest()
    return {
        "schema_version": CACHE_SCHEMA,
        "status": "complete_immutable_train_chunk",
        "source": source,
        "archive_sha256": archive_sha256,
        "archive_member": archive_member,
        "first_line": first_line,
        "last_line": last_line,
        "raw_chunk_sha256": raw_chunk_sha256,
        "cache_bindings": dict(cache_bindings),
        "record_count": len(compact_records),
        "records_sha256_algorithm": "sha256(canonical-json-v1)",
        "records_sha256": records_sha256,
        "records": compact_records,
    }


def validate_cache_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
    archive_sha256: str,
    archive_member: str,
    first_line: int,
    last_line: int,
    raw_chunk_sha256: str,
    cache_bindings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if payload.get("schema_version") != CACHE_SCHEMA:
        raise ProtocolError("cache schema drift")
    if payload.get("status") != "complete_immutable_train_chunk":
        raise ProtocolError("cache status drift")
    expected = {
        "source": source,
        "archive_sha256": archive_sha256,
        "archive_member": archive_member,
        "first_line": first_line,
        "last_line": last_line,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ProtocolError(f"cache {key.replace('_', ' ')} drift")
    if payload.get("raw_chunk_sha256") != raw_chunk_sha256:
        raise ProtocolError("cache raw chunk SHA drift")
    if payload.get("cache_bindings") != dict(cache_bindings):
        raise ProtocolError("cache binding drift")
    records = payload.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ProtocolError("cache records drift")
    if int(payload.get("record_count", -1)) != len(records):
        raise ProtocolError("cache record count drift")
    actual_records_sha = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    if payload.get("records_sha256") != actual_records_sha:
        raise ProtocolError("cache records SHA drift")
    return [dict(record) for record in records]


def cache_shard_path(cache_dir: Path, chunk: Mapping[str, Any]) -> Path:
    member = str(chunk["archive_member"])
    member_slug = hashlib.sha256(member.encode("utf-8")).hexdigest()[:12]
    return (
        cache_dir
        / str(chunk["source"])
        / (
            f"{member_slug}_{int(chunk['first_line']):09d}_"
            f"{int(chunk['last_line']):09d}.json"
        )
    )


def write_new_json(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(payload))
        if path.exists():
            raise FileExistsError(f"JSON artifact appeared during write: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cache_contract_bindings(
    static_bindings: Mapping[str, Any], profiler_sha256: str
) -> dict[str, Any]:
    contract = {
        "source_checkpoint_sha256": SOURCE_SHA256,
        "hybrid_main_sha256": HYBRID_MAIN_SHA256,
        "policy_runtime_sha256": RUNTIME_SHA256,
        "template_contract_sha256": CONTRACT_SHA256,
        "profiler_sha256": profiler_sha256,
        "runtime_semantics": static_bindings["runtime_semantics"],
        "cache_chunk_rows": CACHE_CHUNK_ROWS,
        "record_schema": "deployment-decision-record-v2",
    }
    return {
        **contract,
        "cache_contract_sha256": hashlib.sha256(
            canonical_json_bytes(contract)
        ).hexdigest(),
    }


def profile_archive_cached(
    *,
    path: Path,
    archive_sha256: str,
    source_name: str,
    cache_dir: Path,
    cache_bindings: Mapping[str, Any],
    runtime: ModuleType,
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    raw_rows = 0
    cache_hits = 0
    cache_writes = 0
    for chunk in iter_train_chunks(path, source_name, CACHE_CHUNK_ROWS):
        items = chunk["items"]
        raw_rows += len(items)
        shard = cache_shard_path(cache_dir, chunk)
        if shard.exists():
            payload = load_json_strict(shard)
            chunk_records = validate_cache_payload(
                payload,
                source=source_name,
                archive_sha256=archive_sha256,
                archive_member=str(chunk["archive_member"]),
                first_line=int(chunk["first_line"]),
                last_line=int(chunk["last_line"]),
                raw_chunk_sha256=str(chunk["raw_chunk_sha256"]),
                cache_bindings=cache_bindings,
            )
            cache_hits += 1
        else:
            chunk_records = [
                evaluate_row(
                    row,
                    identity,
                    source_name,
                    runtime,
                    model,
                    config,
                )
                for row, identity in items
            ]
            payload = make_cache_payload(
                source=source_name,
                archive_sha256=archive_sha256,
                archive_member=str(chunk["archive_member"]),
                first_line=int(chunk["first_line"]),
                last_line=int(chunk["last_line"]),
                raw_chunk_sha256=str(chunk["raw_chunk_sha256"]),
                records=chunk_records,
                cache_bindings=cache_bindings,
            )
            write_new_json(payload, shard)
            cache_writes += 1
        expected_ids = [str(identity["line_sha256"]) for _, identity in items]
        actual_ids = [str(record.get("line_sha256")) for record in chunk_records]
        if actual_ids != expected_ids:
            raise ProtocolError(f"cache row identity drift: {shard}")
        records.extend(chunk_records)
        shards.append(
            {
                "path": str(shard.resolve()),
                "sha256": file_sha256(shard),
                "source": source_name,
                "archive_member": str(chunk["archive_member"]),
                "first_line": int(chunk["first_line"]),
                "last_line": int(chunk["last_line"]),
                "record_count": len(chunk_records),
                "raw_chunk_sha256": str(chunk["raw_chunk_sha256"]),
            }
        )
    line_hashes = [str(record["line_sha256"]) for record in records]
    if len(line_hashes) != len(set(line_hashes)):
        raise ProtocolError(f"duplicate accepted line SHA in {source_name}")
    return records, {
        "raw_rows": raw_rows,
        "accepted_rows": len(records),
        "cache_hits": cache_hits,
        "cache_writes": cache_writes,
        "cache_shards": len(shards),
    }, shards


def stable_hash(label: str, row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        f"{SELECTION_SALT}:{label}:{row['line_sha256']}".encode("utf-8")
    ).hexdigest()


def balanced_treatment_sizes(total: int) -> tuple[int, int, int]:
    if total < 3:
        raise ProtocolError("fewer than three deployment treatment rows")
    quotient, remainder = divmod(total, 3)
    return tuple(quotient + int(index < remainder) for index in range(3))  # type: ignore[return-value]


def assign_fixed_sizes(
    rows: Sequence[dict[str, Any]], sizes: Sequence[int], label: str
) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (stable_hash(label, row), row["line_sha256"]))
    if len(ordered) != sum(sizes):
        raise ProtocolError(f"{label} size drift")
    output: list[list[dict[str, Any]]] = []
    start = 0
    for size in sizes:
        output.append(ordered[start : start + size])
        start += size
    return output


def positive_actor6_policy_margin(row: Mapping[str, Any]) -> float | None:
    """Return only the actor6-controllable policy margin when strictly positive."""

    value = row.get("policy_hybrid_margin")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number > 0.0:
            return number
    return None


def choose_boundary_rows(
    eligible: Sequence[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    ordered = sorted(
        eligible,
        key=lambda row: (
            float(row["policy_hybrid_margin"]),
            stable_hash("boundary-tie", row),
            row["line_sha256"],
        ),
    )
    total = sum(BOUNDARY_STEP_SIZES)
    if len(ordered) < total:
        raise ProtocolError("insufficient positive actor6-policy-margin old-train rows")
    pool = ordered[:total]
    context34 = [row for row in pool if int(row["context"]) == 34]
    required = MIN_CONTEXT34_BOUNDARY_PER_STEP * 3
    replacements: list[dict[str, Any]] = []
    if len(context34) < required:
        outside34 = [row for row in ordered[total:] if int(row["context"]) == 34]
        need = required - len(context34)
        if len(outside34) < need:
            raise ProtocolError("insufficient context34 deployment boundary rows")
        replaceable = sorted(
            [row for row in pool if int(row["context"]) != 34],
            key=lambda row: (float(row["policy_hybrid_margin"]), row["line_sha256"]),
            reverse=True,
        )[:need]
        removed = {row["line_sha256"] for row in replaceable}
        replacements = outside34[:need]
        pool = [row for row in pool if row["line_sha256"] not in removed] + replacements

    steps: list[list[dict[str, Any]]] = [[] for _ in range(3)]
    capacities = list(BOUNDARY_STEP_SIZES)
    c34_rows = sorted(
        [row for row in pool if int(row["context"]) == 34],
        key=lambda row: (float(row["policy_hybrid_margin"]), row["line_sha256"]),
    )
    used: set[str] = set()
    for index in range(3):
        for row in c34_rows[
            index * MIN_CONTEXT34_BOUNDARY_PER_STEP :
            (index + 1) * MIN_CONTEXT34_BOUNDARY_PER_STEP
        ]:
            steps[index].append(row)
            used.add(str(row["line_sha256"]))
    remaining = sorted(
        [row for row in pool if row["line_sha256"] not in used],
        key=lambda row: (
            float(row["policy_hybrid_margin"]),
            stable_hash("boundary-assign", row),
        ),
    )
    for row in remaining:
        candidates = [
            index for index in range(3) if len(steps[index]) < capacities[index]
        ]
        if not candidates:
            raise ProtocolError("deployment boundary assignment overflow")
        chosen = min(
            candidates,
            key=lambda index: (
                len(steps[index]) / capacities[index],
                stable_hash(f"boundary-step-{index + 1}", row),
            ),
        )
        steps[chosen].append(row)
    if [len(rows) for rows in steps] != capacities:
        raise ProtocolError("deployment boundary quota drift")
    if any(
        sum(int(row["context"]) == 34 for row in rows)
        < MIN_CONTEXT34_BOUNDARY_PER_STEP
        for rows in steps
    ):
        raise ProtocolError("deployment context34 boundary quota drift")
    return steps, {
        "eligible_positive_actor6_policy_margin_rows": len(eligible),
        "selected_rows": len(pool),
        "context34_rows": sum(int(row["context"]) == 34 for row in pool),
        "quota_replacements": len(replacements),
        "selection_margin": "exact CPU-FP32 policy_hybrid_margin only",
        "count_margin_role": (
            "audit only; actor6 cannot change the count head, so count-limited "
            "rows do not receive boundary priority"
        ),
        "selection": (
            "lowest positive actor6 policy margins globally; if required, replace "
            "largest non-context34 rows by lowest outside context34 until the "
            "fixed four-per-step context34 minimum is feasible"
        ),
    }


def choose_broad_rows(
    eligible: Sequence[dict[str, Any]], excluded: set[str]
) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in eligible:
        if row["line_sha256"] in excluded:
            continue
        groups[(int(row["context"]), str(row["team_name"]))].append(row)
    for key, values in list(groups.items()):
        groups[key] = deque(
            sorted(values, key=lambda row: stable_hash(f"broad:{key}", row))
        )
    output: list[list[dict[str, Any]]] = []
    for step_index in range(3):
        chosen: list[dict[str, Any]] = []
        round_index = 0
        while len(chosen) < BROAD_PER_STEP:
            active = [key for key, values in groups.items() if values]
            if not active:
                raise ProtocolError("insufficient deployment broad retention rows")
            active.sort(
                key=lambda key: hashlib.sha256(
                    f"{SELECTION_SALT}:broad-step-{step_index + 1}:"
                    f"round-{round_index}:{key[0]}:{key[1]}".encode("utf-8")
                ).hexdigest()
            )
            for key in active:
                if len(chosen) >= BROAD_PER_STEP:
                    break
                chosen.append(groups[key].popleft())
            round_index += 1
        output.append(chosen)
    return output


def descriptor(row: Mapping[str, Any], category: str) -> dict[str, Any]:
    keys = (
        "source",
        "archive_member",
        "member_line_number",
        "raw_line_sha256",
        "line_sha256",
        "line_sha256_algorithm",
        "decision_key",
        "episode_id",
        "episode_uuid",
        "visible_signature",
        "context",
        "team_name",
        "sample_weight",
        "expert_action_order",
        "source_policy_action",
        "source_hybrid_action",
        "source_set_correct",
        "source_hybrid_correct",
        "source_count_correct",
        "policy_hybrid_margin",
        "count_margin",
        "deployment_margin",
        "deployment_margin_limiting_head",
    )
    result = {key: row[key] for key in keys}
    result["category"] = category
    return result


def _counter_by_context(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(int(row["context"])) for row in rows)
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def build_profile_from_records(
    *,
    yanz: Sequence[dict[str, Any]],
    old: Sequence[dict[str, Any]],
    static_bindings: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    cache_bindings: Mapping[str, Any],
    yanz_counts: Mapping[str, Any],
    old_counts: Mapping[str, Any],
    cache_shards: Sequence[Mapping[str, Any]],
    profiler_sha256: str,
) -> dict[str, Any]:
    if len(yanz) != EXPECTED_YANZ_RAW_ROWS or len(old) != EXPECTED_OLD_RAW_ROWS:
        raise ProtocolError("deployment full-train row count drift")
    hybrid_errors = [row for row in yanz if not bool(row["source_hybrid_correct"])]
    set_errors = [row for row in yanz if not bool(row["source_set_correct"])]
    treatment = [row for row in hybrid_errors if bool(row["source_count_correct"])]
    count_errors = [row for row in hybrid_errors if not bool(row["source_count_correct"])]
    treatment_sizes = balanced_treatment_sizes(len(treatment))
    treatment_steps = assign_fixed_sizes(
        treatment, treatment_sizes, "treatment-partition"
    )

    yanz_episode_ids = {str(row["episode_id"]) for row in yanz}
    yanz_episode_uuids = {str(row["episode_uuid"]) for row in yanz}
    yanz_signatures = {str(row["visible_signature"]) for row in yanz}
    eligible_old = [
        row
        for row in old
        if bool(row["source_hybrid_correct"])
        and bool(row["source_count_correct"])
        and str(row["episode_id"]) not in yanz_episode_ids
        and str(row["episode_uuid"]) not in yanz_episode_uuids
        and str(row["visible_signature"]) not in yanz_signatures
        and positive_actor6_policy_margin(row) is not None
    ]
    boundary_steps, boundary_audit = choose_boundary_rows(eligible_old)
    boundary_ids = {
        str(row["line_sha256"]) for rows in boundary_steps for row in rows
    }
    broad_steps = choose_broad_rows(eligible_old, boundary_ids)
    retention = [
        row
        for index in range(3)
        for row in boundary_steps[index] + broad_steps[index]
    ]
    retention_ids = [str(row["line_sha256"]) for row in retention]
    if len(retention_ids) != 467 or len(set(retention_ids)) != 467:
        raise ProtocolError("deployment retention union is not 467 unique rows")
    treatment_ids = [str(row["line_sha256"]) for row in treatment]
    if len(treatment_ids) != len(set(treatment_ids)):
        raise ProtocolError("deployment treatment rows repeat")
    if set(treatment_ids) & set(retention_ids):
        raise ProtocolError("deployment treatment/retention line overlap")
    if any(not bool(row["source_hybrid_correct"]) for row in retention):
        raise ProtocolError("deployment retention contains a source-wrong row")
    if any(bool(row["source_hybrid_correct"]) for row in treatment):
        raise ProtocolError("deployment treatment contains a source-correct row")

    steps: list[dict[str, Any]] = []
    for index in range(3):
        treatment_rows = [descriptor(row, "treatment") for row in treatment_steps[index]]
        boundary_rows = [
            descriptor(row, "retention_boundary") for row in boundary_steps[index]
        ]
        broad_rows = [descriptor(row, "retention_broad") for row in broad_steps[index]]
        steps.append(
            {
                "step": index + 1,
                "treatment": treatment_rows,
                "retention_boundary": boundary_rows,
                "retention_broad": broad_rows,
                "retention": boundary_rows + broad_rows,
                "counts": {
                    "treatment": len(treatment_rows),
                    "retention_boundary": len(boundary_rows),
                    "retention_broad": len(broad_rows),
                    "retention": len(boundary_rows) + len(broad_rows),
                    "retention_context34": sum(
                        int(row["context"]) == 34 for row in boundary_rows
                    ),
                },
            }
        )

    return {
        "schema_version": SCHEMA,
        "status": "frozen_train_only_deployment_semantics_row_selection",
        "bindings": {
            **dict(static_bindings),
            "profiler": {"path": str(SCRIPT), "sha256": profiler_sha256},
            "cache_contract": dict(cache_bindings),
            "expanded_source": dict(model_audit),
        },
        "selection_contract": {
            "opened_members": "train/*.jsonl only",
            "runtime": (
                "Alakazam hybrid-order submission main semantics: real61, CPU "
                "FP32, batch=1, no autocast"
            ),
            "treatment": (
                "all deployment-hybrid-wrong yanz rows with deployment-correct "
                "cardinality; count-involved errors excluded"
            ),
            "treatment_partition": {
                "method": "ascending salted SHA-256 then consecutive balanced thirds",
                "sizes": list(treatment_sizes),
            },
            "retention_eligibility": (
                "old-train deployment-hybrid-correct and count-correct with "
                "positive finite actor6 policy margin and no yanz episode id, "
                "episode UUID, or exact deployment-visible-signature overlap"
            ),
            "boundary": boundary_audit,
            "broad": (
                "32 per step by deterministic context/team round-robin with "
                "salted SHA order inside each stratum"
            ),
            "cross_step_overlap": 0,
        },
        "source_counts": {
            "yanz": dict(yanz_counts),
            "yanz_set_correct": sum(bool(row["source_set_correct"]) for row in yanz),
            "yanz_hybrid_correct": sum(
                bool(row["source_hybrid_correct"]) for row in yanz
            ),
            "yanz_count_correct": sum(
                bool(row["source_count_correct"]) for row in yanz
            ),
            "yanz_set_errors": len(set_errors),
            "yanz_hybrid_errors": len(hybrid_errors),
            "treatment_selection_only": len(treatment),
            "excluded_count_errors": len(count_errors),
            "treatment_by_context": _counter_by_context(treatment),
            "old": dict(old_counts),
            "old_hybrid_correct": sum(
                bool(row["source_hybrid_correct"]) for row in old
            ),
            "old_count_correct": sum(bool(row["source_count_correct"]) for row in old),
            "eligible_old_retention": len(eligible_old),
            "retention_selected": len(retention_ids),
            "retention_by_context": _counter_by_context(retention),
        },
        "excluded_count_error_rows": [
            descriptor(row, "excluded_count_error") for row in count_errors
        ],
        "cache": {
            "immutable_shards": list(cache_shards),
            "shard_count": len(cache_shards),
            "cache_artifacts_are_not_training_candidates": True,
        },
        "steps": steps,
        "scope": {
            "train_members_only": True,
            "model_artifact_written": False,
            "submission_performed": False,
        },
    }


def smoke(rows_per_source: int) -> dict[str, Any]:
    if not 1 <= rows_per_source <= 32:
        raise ValueError("--rows-per-source must be in [1, 32]")
    startup_script_sha256 = file_sha256(SCRIPT)
    static = validate_static_bindings()
    runtime = load_bound_runtime()
    _, model, config, model_audit = instantiate_deployment_source(runtime)
    sources = (("yanz", YANZ), ("old", OLD))
    reports: dict[str, Any] = {}
    for source_name, path in sources:
        records: list[dict[str, Any]] = []
        main_fixture_matches = 0
        for chunk in iter_train_chunks(path, source_name, CACHE_CHUNK_ROWS):
            for row, identity in chunk["items"]:
                record = evaluate_row(
                    row, identity, source_name, runtime, model, config
                )
                records.append(record)
                if actual_hybrid_main_action(
                    row["observation"], runtime, model, config
                ) == record["source_hybrid_action"]:
                    main_fixture_matches += 1
                if len(records) == rows_per_source:
                    break
            if len(records) == rows_per_source:
                break
        if len(records) != rows_per_source:
            raise ProtocolError(f"smoke row shortage: {source_name}")
        reports[source_name] = {
            "rows": len(records),
            "line_sha256": [row["line_sha256"] for row in records],
            "source_hybrid_correct": sum(
                bool(row["source_hybrid_correct"]) for row in records
            ),
            "source_count_correct": sum(
                bool(row["source_count_correct"]) for row in records
            ),
            "contexts": [int(row["context"]) for row in records],
            "actual_hybrid_main_fixture_matches": main_fixture_matches,
        }
    result = {
        "schema_version": f"{SCHEMA}-smoke",
        "status": "deployment_semantics_smoke_passed",
        "bindings": {
            **static,
            "profiler": {"path": str(SCRIPT), "sha256": startup_script_sha256},
        },
        "expanded_source": model_audit,
        "sources": reports,
        "scope": {
            "train_members_only": True,
            "cache_written": False,
            "profile_written": False,
            "model_artifact_written": False,
            "submission_performed": False,
        },
    }
    assert_script_sha(startup_script_sha256)
    return result


def build(cache_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v2 profile: {output}")
    startup_script_sha256 = file_sha256(SCRIPT)
    static = validate_static_bindings()
    runtime = load_bound_runtime()
    _, model, config, model_audit = instantiate_deployment_source(runtime)
    cache_bindings = cache_contract_bindings(static, startup_script_sha256)
    yanz, yanz_counts, yanz_shards = profile_archive_cached(
        path=YANZ,
        archive_sha256=YANZ_SHA256,
        source_name="yanz",
        cache_dir=cache_dir,
        cache_bindings=cache_bindings,
        runtime=runtime,
        model=model,
        config=config,
    )
    if yanz_counts["raw_rows"] != EXPECTED_YANZ_RAW_ROWS:
        raise ProtocolError("yanz full train raw-row count drift")
    old, old_counts, old_shards = profile_archive_cached(
        path=OLD,
        archive_sha256=OLD_SHA256,
        source_name="old",
        cache_dir=cache_dir,
        cache_bindings=cache_bindings,
        runtime=runtime,
        model=model,
        config=config,
    )
    if old_counts["raw_rows"] != EXPECTED_OLD_RAW_ROWS:
        raise ProtocolError("old full train raw-row count drift")
    profile = build_profile_from_records(
        yanz=yanz,
        old=old,
        static_bindings=static,
        model_audit=model_audit,
        cache_bindings=cache_bindings,
        yanz_counts=yanz_counts,
        old_counts=old_counts,
        cache_shards=yanz_shards + old_shards,
        profiler_sha256=startup_script_sha256,
    )
    if file_sha256(SOURCE) != SOURCE_SHA256:
        raise ProtocolError("source changed during v2 profiling")
    if file_sha256(YANZ) != YANZ_SHA256 or file_sha256(OLD) != OLD_SHA256:
        raise ProtocolError("train archive changed during v2 profiling")
    if file_sha256(HYBRID_MAIN) != HYBRID_MAIN_SHA256:
        raise ProtocolError("hybrid main changed during v2 profiling")
    if file_sha256(RUNTIME) != RUNTIME_SHA256:
        raise ProtocolError("policy runtime changed during v2 profiling")
    assert_script_sha(startup_script_sha256)
    write_new_json(profile, output)
    assert_script_sha(startup_script_sha256)
    return {
        "status": profile["status"],
        "output": str(output.resolve()),
        "sha256": file_sha256(output),
        "cache_dir": str(cache_dir.resolve()),
        "cache_shards": len(yanz_shards) + len(old_shards),
        "yanz_rows": len(yanz),
        "old_rows": len(old),
        "treatment_rows": int(
            profile["source_counts"]["treatment_selection_only"]
        ),
        "retention_rows": int(profile["source_counts"]["retention_selected"]),
        "model_artifact_written": False,
        "submission_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--rows-per-source", type=int, default=4)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--cache-dir", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "smoke":
        result = smoke(args.rows_per_source)
    else:
        result = build(args.cache_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

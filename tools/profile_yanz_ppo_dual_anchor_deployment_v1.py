#!/usr/bin/env python3
"""Build an immutable train-only PPO/pure-BC deployment profile.

The PPO checkpoint is the policy intended to be protected.  The frozen pure-BC
expanded61 deployment state is a teacher used to identify PPO regressions and
PPO-only repairs.  Every accepted train row is evaluated once per anchor under
the real Alakazam hybrid-order decoder on CPU, FP32, batch size one, and one
PyTorch CPU thread.

Only ``train/*.jsonl`` members in the two hash-bound yanz/old archives are ever
opened.  This program writes JSON profile/cache artifacts only; it cannot write
a model, read a holdout split, upload data, or submit a competition entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import profile_yanz_source_error_actor6_deployment_v2 as frozen  # noqa: E402


SCRIPT = Path(__file__).resolve()
FROZEN_PROFILER = TOOLS / "profile_yanz_source_error_actor6_deployment_v2.py"
PPO_PARENT = ROOT / (
    "artifacts/top3_bc77_20260810/alakazam_control/"
    "ppo_terminal01_corrected_v2_soup_u0_u10_v1/bc12p5_u10_87p5.pt"
)
PURE_BC = ROOT / "artifacts/gold8_recent7_20260808/alakazam_control/specialist_bc/best.pt"
YANZ = ROOT / (
    "data/yanz_alakazam_live_20260810_v1/snapshot_20260810T0433Z/"
    "policies/yanzhou06_alakazam_live_55394520.zip"
)
OLD = ROOT / "data/gold8_recent7_20260808/archives/alakazam_control.zip"
HYBRID_MAIN = ROOT / "submission_templates/ptcg_ppo_alakazam_hybrid_order_v1/main.py"
CONTRACT = ROOT / "submission_templates/ptcg_ppo_alakazam_hybrid_order_v1/CONTRACT.json"
RUNTIME = ROOT / "submission_templates/ptcg_ppo_alakazam_standard_pl_v1/policy_runtime.py"
MAIN_RUNTIME_ANCHOR = ROOT / (
    "submissions/ptcg_ppo_alakazam_bc12p5_u10_87p5_20260810/policy_runtime.py"
)
DECK_CSV = ROOT / (
    "data/gold8_recent7_20260808/decks/"
    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf.csv"
)

SCHEMA = "ptcg-yanz-ppo-dual-anchor-deployment-profile-v1"
CACHE_SCHEMA = "ptcg-yanz-ppo-dual-anchor-cache-shard-v1"
RECORD_SCHEMA = "ptcg-yanz-ppo-dual-anchor-decision-record-v1"
PPO_PARENT_SHA256 = "bc3bbf185e4f61e4d3adc5dd11241f0f6001be274e90ef42e3f341f22b052393"
PPO_PARENT_BITWISE_STATE_SHA256 = (
    "fa2b090892f4fc0f347061f8b365b35701f516bb9be621643040750f6fcad43e"
)
PURE_BC_SHA256 = "f5500086c16a02c19f3f2abce5e144446bd079248fd9fc3f4a620f3d079c7626"
PURE_BC_RAW_BITWISE_STATE_SHA256 = (
    "b0bfe317017a9d4f3cbd367cd21ff96c7fdc60abd1924a6d8edbdf7b32917f91"
)
PURE_BC_EXPANDED61_BITWISE_STATE_SHA256 = (
    "efefd3d5c2260e771c7ce7a67a67dbb585acce3ebb43742d247594922a734a62"
)
YANZ_SHA256 = "bbbb6f69b809c60e900f7d2992333b645d117b2a6c836642fecca0e1f4226720"
OLD_SHA256 = "4bd0de193cfb88b060435f84bbb5dc83a9eff3ab1aa4a498ef0859d2346f3c6c"
HYBRID_MAIN_SHA256 = "b35c7b0976a9a377b7f3c61c69aa15ef477db96b1b6bbdeb4ceb0305312b3a2a"
RUNTIME_SHA256 = "fe7182a588962fd9c5e07e9e62433beaa1c5b3fa047bb8170e22435e4f93e997"
CONTRACT_SHA256 = "8cb9c929e21f5b74efa84ccc7cca9dc3797cded3a35f8e37f9c8ebef522cf205"
DECK_CSV_SHA256 = "0598646548d081832ec311c15fdc369b32c6f5e63175b0cfd1904d21fd082451"
FROZEN_PROFILER_SHA256 = "386d23e280f42225b1752e987b1eb92df8b30b41a1be5ebbd9260b8f6388927e"
DECK_HASH = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
EXPECTED_YANZ_ROWS = 1725
EXPECTED_OLD_ROWS = 59875
CACHE_CHUNK_ROWS = 512
COUNT_CLASSES = 61

CATEGORY_ORDER = (
    "ppo_correct_protection",
    "bc_correct_ppo_wrong_recovery_target",
    "ppo_only_repair_protection",
    "both_wrong",
)


class ProtocolError(RuntimeError):
    """A fail-closed dual-anchor profiling contract violation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def assert_script_sha(expected: str) -> None:
    actual = file_sha256(SCRIPT)
    if actual != expected:
        raise ProtocolError(f"dual-anchor profiler changed during run: {actual} != {expected}")


def _file_binding(path: Path, expected: str, label: str) -> dict[str, str]:
    actual = file_sha256(path)
    if actual != expected:
        raise ProtocolError(f"static binding drift {label}: {actual} != {expected}")
    return {"path": str(path.resolve()), "sha256": actual}


def validate_static_bindings() -> dict[str, Any]:
    """Validate every executable, model, data, deck, and decode dependency."""

    bindings = {
        "ppo_parent_checkpoint": _file_binding(
            PPO_PARENT, PPO_PARENT_SHA256, "ppo parent checkpoint"
        ),
        "pure_bc_checkpoint": _file_binding(PURE_BC, PURE_BC_SHA256, "pure BC checkpoint"),
        "yanz_train_archive": _file_binding(YANZ, YANZ_SHA256, "yanz train archive"),
        "old_train_archive": _file_binding(OLD, OLD_SHA256, "old train archive"),
        "hybrid_main": _file_binding(HYBRID_MAIN, HYBRID_MAIN_SHA256, "hybrid main"),
        "policy_runtime": _file_binding(RUNTIME, RUNTIME_SHA256, "policy runtime"),
        "main_runtime_anchor": _file_binding(
            MAIN_RUNTIME_ANCHOR, RUNTIME_SHA256, "main runtime anchor"
        ),
        "template_contract": _file_binding(CONTRACT, CONTRACT_SHA256, "template contract"),
        "deck_csv": _file_binding(DECK_CSV, DECK_CSV_SHA256, "deck CSV"),
        "frozen_profiler_dependency": _file_binding(
            FROZEN_PROFILER, FROZEN_PROFILER_SHA256, "frozen profiler dependency"
        ),
    }
    try:
        frozen_bindings = frozen.validate_static_bindings()
    except frozen.ProtocolError as exc:
        raise ProtocolError(f"frozen deployment contract failed: {exc}") from exc
    expected_frozen = {
        "source_checkpoint": PURE_BC_SHA256,
        "yanz_train_archive": YANZ_SHA256,
        "old_train_archive": OLD_SHA256,
        "hybrid_main": HYBRID_MAIN_SHA256,
        "policy_runtime": RUNTIME_SHA256,
        "template_contract": CONTRACT_SHA256,
        "deck_csv": DECK_CSV_SHA256,
    }
    for name, expected in expected_frozen.items():
        if frozen_bindings[name]["sha256"] != expected:
            raise ProtocolError(f"frozen binding mismatch: {name}")
    runtime_semantics = dict(frozen_bindings["runtime_semantics"])
    required_semantics = {
        "device": "cpu",
        "dtype": "torch.float32",
        "batch_size": 1,
        "autocast": False,
        "inference_mode": True,
        "torch_num_threads": 1,
        "count_classes": COUNT_CLASSES,
    }
    for name, expected in required_semantics.items():
        if runtime_semantics.get(name) != expected:
            raise ProtocolError(f"runtime semantics drift: {name}")
    bindings["runtime_semantics"] = {
        **runtime_semantics,
        "models_per_row": 2,
        "model_evaluation_order": ["pure_bc_teacher", "ppo_parent"],
        "autocast_context_entered": False,
        "opened_archive_members": "train/*.jsonl only",
    }
    return bindings


def _normalise_model_config(value: Any, label: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} model config is not an object")
    names = (
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
    if any(name not in value for name in names):
        raise ProtocolError(f"{label} model config is incomplete")
    return {
        name: float(value[name]) if name == "dropout" else int(value[name])
        for name in names
    }


def load_bound_models(
    runtime: ModuleType,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, int | float], dict[str, Any]]:
    """Load the expanded61 pure-BC teacher and exact PPO parent on CPU."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        _, bc_model, bc_config, bc_audit = frozen.instantiate_deployment_source(runtime)
    except frozen.ProtocolError as exc:
        raise ProtocolError(f"pure BC teacher load failed: {exc}") from exc
    if bc_audit["expanded_model_state_sha256"] != PURE_BC_EXPANDED61_BITWISE_STATE_SHA256:
        raise ProtocolError("pure BC expanded61 teacher state drift")
    if bc_audit["raw_model_state_sha256"] != PURE_BC_RAW_BITWISE_STATE_SHA256:
        raise ProtocolError("pure BC raw state drift")

    checkpoint = torch.load(PPO_PARENT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ProtocolError("PPO parent checkpoint root drift")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or len(state) != 80:
        raise ProtocolError("PPO parent state drift")
    parent_state_sha = frozen.bitwise_model_state_sha256(state)
    if parent_state_sha != PPO_PARENT_BITWISE_STATE_SHA256:
        raise ProtocolError("PPO parent bitwise state drift")
    ppo_config = _normalise_model_config(checkpoint.get("model_config"), "PPO parent")
    if ppo_config != dict(bc_config):
        raise ProtocolError("PPO parent and pure BC deployment model configs differ")
    if checkpoint.get("learner_deck_hash") != DECK_HASH:
        raise ProtocolError("PPO parent learner deck drift")
    if int(checkpoint.get("update", -1)) != 9:
        raise ProtocolError("PPO parent update drift")
    if checkpoint.get("reward") != {"win": 1.0, "loss": 0.0, "draw": 0.0}:
        raise ProtocolError("PPO parent terminal reward contract drift")
    if checkpoint.get("action_distribution") != (
        "masked cardinality categorical + ordered Plackett-Luce without replacement"
    ):
        raise ProtocolError("PPO parent action distribution drift")

    ppo_model = runtime.EntityOptionPolicy(
        hash_size=int(ppo_config["hash_size"]),
        categorical_dim=int(ppo_config["categorical_dim"]),
        model_dim=int(ppo_config["model_dim"]),
        layers=int(ppo_config["layers"]),
        heads=int(ppo_config["heads"]),
        dropout=float(ppo_config["dropout"]),
        max_state_entities=int(ppo_config["max_state_entities"]),
    )
    template_state = ppo_model.state_dict()
    if len(template_state) != 80 or template_state["count_head.2.weight"].shape[0] != 61:
        raise ProtocolError("PPO runtime model is not the real expanded61 architecture")
    for name, template in template_state.items():
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != template.shape:
            raise ProtocolError(f"PPO parent tensor drift: {name}")
    ppo_model.load_state_dict(state, strict=True)
    ppo_model.to(torch.device("cpu")).eval().requires_grad_(False)
    if any(value.device.type != "cpu" for value in ppo_model.parameters()):
        raise ProtocolError("PPO parent escaped CPU")
    if any(value.dtype != torch.float32 for value in ppo_model.parameters()):
        raise ProtocolError("PPO parent is not FP32")
    if torch.get_num_threads() != 1:
        raise ProtocolError("dual-anchor profiler did not lock one CPU thread")
    loaded_state_sha = frozen.bitwise_model_state_sha256(ppo_model.state_dict())
    if loaded_state_sha != PPO_PARENT_BITWISE_STATE_SHA256:
        raise ProtocolError("loaded PPO parent bitwise state drift")
    ppo_audit = {
        "checkpoint_sha256": PPO_PARENT_SHA256,
        "model_state_sha256_algorithm": "ptcg-model-state-bitwise-v1",
        "model_state_sha256": loaded_state_sha,
        "tensor_count": len(ppo_model.state_dict()),
        "count_classes": int(ppo_model.count_head[-1].out_features),
        "all_parameters_cpu": True,
        "all_parameters_float32": True,
        "training": False,
        "requires_grad": False,
        "update": 9,
        "reward": dict(checkpoint["reward"]),
        "learner_deck_hash": str(checkpoint["learner_deck_hash"]),
        "feature_version": str(checkpoint.get("feature_version", "")),
        "bc_feature_version": str(checkpoint.get("bc_feature_version", "")),
    }
    return bc_model, ppo_model, ppo_config, {
        "pure_bc_teacher": dict(bc_audit),
        "ppo_parent": ppo_audit,
    }


def classify_transition(bc_correct: bool, ppo_correct: bool) -> str:
    """Return one of four exhaustive, mutually exclusive anchor categories."""

    if bc_correct and ppo_correct:
        return "ppo_correct_protection"
    if bc_correct and not ppo_correct:
        return "bc_correct_ppo_wrong_recovery_target"
    if not bc_correct and ppo_correct:
        return "ppo_only_repair_protection"
    return "both_wrong"


def _model_decision_fields(
    prefix: str,
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    expert_order: Sequence[int],
) -> dict[str, Any]:
    if outputs["policy_logits"].device.type != "cpu":
        raise ProtocolError(f"{prefix} policy logits escaped CPU")
    if outputs["policy_logits"].dtype != torch.float32:
        raise ProtocolError(f"{prefix} policy logits are not FP32")
    if outputs["count_logits"].dtype != torch.float32:
        raise ProtocolError(f"{prefix} count logits are not FP32")
    try:
        decoded = frozen.decode_hybrid(outputs, batch)
        margins = frozen.decision_margins(outputs, batch, expert_order)
    except frozen.ProtocolError as exc:
        raise ProtocolError(f"{prefix} deployment decode failed: {exc}") from exc
    raw_action = [int(value) for value in decoded["raw_action"]]
    hybrid_action = [int(value) for value in decoded["hybrid_action"]]
    expert = [int(value) for value in expert_order]
    return {
        f"{prefix}_predicted_count": int(decoded["count"]),
        f"{prefix}_policy_action": raw_action,
        f"{prefix}_hybrid_action": hybrid_action,
        f"{prefix}_set_correct": sorted(raw_action) == sorted(expert),
        f"{prefix}_hybrid_correct": hybrid_action == expert,
        f"{prefix}_count_correct": int(decoded["count"]) == len(expert),
        f"{prefix}_policy_hybrid_margin": margins["policy_hybrid_margin"],
        f"{prefix}_count_margin": margins["count_margin"],
        f"{prefix}_deployment_margin": margins["deployment_margin"],
        f"{prefix}_deployment_margin_limiting_head": margins[
            "deployment_margin_limiting_head"
        ],
    }


def _finite_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    value = float(left) - float(right)
    return value if math.isfinite(value) else None


def evaluate_dual_row(
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    source_name: str,
    runtime: ModuleType,
    bc_model: torch.nn.Module,
    ppo_model: torch.nn.Module,
    config: Mapping[str, int | float],
) -> dict[str, Any]:
    """Evaluate one train row under both anchors with one-row CPU tensors."""

    observation = row.get("observation")
    if not isinstance(observation, dict):
        raise ProtocolError("train row observation missing")
    if not isinstance(observation.get("select"), dict):
        raise ProtocolError("train decision has no select object")
    raw_expert = row.get("action", [])
    if not isinstance(raw_expert, list):
        raise ProtocolError("expert action is not a list")
    expert_order = [int(value) for value in raw_expert]
    current = observation.get("current") or {}
    if not isinstance(current, Mapping):
        raise ProtocolError("observation current field is not an object")
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
        raise ProtocolError(f"deployment featurization failed: {identity['line_sha256']}")
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
    option_count = int(batch["option_mask"][0].sum())
    minimum = int(batch["min_counts"][0])
    maximum = int(batch["max_counts"][0])
    if not (
        minimum <= len(expert_order) <= maximum <= option_count
        and len(expert_order) == len(set(expert_order))
        and all(0 <= value < option_count for value in expert_order)
    ):
        raise ProtocolError(f"expert action is illegal: {identity['line_sha256']}")
    with torch.inference_mode():
        bc_outputs = bc_model(batch)
        ppo_outputs = ppo_model(batch)
    bc = _model_decision_fields("bc", bc_outputs, batch, expert_order)
    ppo = _model_decision_fields("ppo", ppo_outputs, batch, expert_order)
    hybrid_category = classify_transition(
        bool(bc["bc_hybrid_correct"]), bool(ppo["ppo_hybrid_correct"])
    )
    set_category = classify_transition(
        bool(bc["bc_set_correct"]), bool(ppo["ppo_set_correct"])
    )
    count_category = classify_transition(
        bool(bc["bc_count_correct"]), bool(ppo["ppo_count_correct"])
    )
    return {
        "record_schema": RECORD_SCHEMA,
        **dict(identity),
        "source": source_name,
        "decision_key": (
            f"{row.get('episode_id')}:"
            f"{row.get('observation_step_index')}->"
            f"{row.get('action_step_index')}"
        ),
        "episode_id": str(row.get("episode_id", "")),
        "episode_uuid": str(row.get("episode_uuid", "")),
        "visible_signature": frozen.feature_visible_signature(feature),
        "context": int(feature["context"]),
        "team_name": str(row.get("team_name", "")),
        "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
        "expert_action_order": expert_order,
        **bc,
        **ppo,
        "hybrid_category": hybrid_category,
        "set_category": set_category,
        "count_category": count_category,
        "bc_ppo_policy_action_equal": bc["bc_policy_action"] == ppo["ppo_policy_action"],
        "bc_ppo_hybrid_action_equal": bc["bc_hybrid_action"] == ppo["ppo_hybrid_action"],
        "bc_ppo_predicted_count_equal": (
            int(bc["bc_predicted_count"]) == int(ppo["ppo_predicted_count"])
        ),
        "ppo_minus_bc_policy_hybrid_margin": _finite_delta(
            ppo["ppo_policy_hybrid_margin"], bc["bc_policy_hybrid_margin"]
        ),
        "ppo_minus_bc_count_margin": _finite_delta(
            ppo["ppo_count_margin"], bc["bc_count_margin"]
        ),
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
    compact = [dict(record) for record in records]
    return {
        "schema_version": CACHE_SCHEMA,
        "status": "complete_immutable_train_only_dual_anchor_chunk",
        "source": source,
        "archive_sha256": archive_sha256,
        "archive_member": archive_member,
        "first_line": int(first_line),
        "last_line": int(last_line),
        "raw_chunk_sha256": raw_chunk_sha256,
        "cache_bindings": dict(cache_bindings),
        "record_count": len(compact),
        "records_sha256_algorithm": "sha256(canonical-json-v1)",
        "records_sha256": hashlib.sha256(canonical_json_bytes(compact)).hexdigest(),
        "records": compact,
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
    if payload.get("status") != "complete_immutable_train_only_dual_anchor_chunk":
        raise ProtocolError("cache status drift")
    expected = {
        "source": source,
        "archive_sha256": archive_sha256,
        "archive_member": archive_member,
        "first_line": int(first_line),
        "last_line": int(last_line),
        "raw_chunk_sha256": raw_chunk_sha256,
        "cache_bindings": dict(cache_bindings),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ProtocolError(f"cache {name.replace('_', ' ')} drift")
    records = payload.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ProtocolError("cache records drift")
    if int(payload.get("record_count", -1)) != len(records):
        raise ProtocolError("cache record count drift")
    actual_sha = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    if payload.get("records_sha256") != actual_sha:
        raise ProtocolError("cache records SHA drift")
    for record in records:
        if record.get("record_schema") != RECORD_SCHEMA:
            raise ProtocolError("cache decision record schema drift")
        if record.get("hybrid_category") not in CATEGORY_ORDER:
            raise ProtocolError("cache hybrid category drift")
    return [dict(record) for record in records]


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
        temporary.write_bytes(canonical_json_bytes(payload))
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
        "ppo_parent_checkpoint_sha256": PPO_PARENT_SHA256,
        "ppo_parent_bitwise_state_sha256": PPO_PARENT_BITWISE_STATE_SHA256,
        "pure_bc_checkpoint_sha256": PURE_BC_SHA256,
        "pure_bc_expanded61_bitwise_state_sha256": (
            PURE_BC_EXPANDED61_BITWISE_STATE_SHA256
        ),
        "yanz_train_archive_sha256": YANZ_SHA256,
        "old_train_archive_sha256": OLD_SHA256,
        "hybrid_main_sha256": HYBRID_MAIN_SHA256,
        "policy_runtime_sha256": RUNTIME_SHA256,
        "template_contract_sha256": CONTRACT_SHA256,
        "deck_csv_sha256": DECK_CSV_SHA256,
        "frozen_profiler_dependency_sha256": FROZEN_PROFILER_SHA256,
        "profiler_sha256": profiler_sha256,
        "runtime_semantics": dict(static_bindings["runtime_semantics"]),
        "cache_chunk_rows": CACHE_CHUNK_ROWS,
        "record_schema": RECORD_SCHEMA,
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
    bc_model: torch.nn.Module,
    ppo_model: torch.nn.Module,
    config: Mapping[str, int | float],
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    raw_rows = cache_hits = cache_writes = 0
    try:
        chunks = frozen.iter_train_chunks(path, source_name, CACHE_CHUNK_ROWS)
        for chunk in chunks:
            items = chunk["items"]
            raw_rows += len(items)
            shard = frozen.cache_shard_path(cache_dir, chunk)
            if shard.exists():
                payload = frozen.load_json_strict(shard)
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
                    evaluate_dual_row(
                        row,
                        identity,
                        source_name,
                        runtime,
                        bc_model,
                        ppo_model,
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
                    "records_sha256": str(payload["records_sha256"]),
                }
            )
    except frozen.ProtocolError as exc:
        raise ProtocolError(f"train-only archive protocol failed: {exc}") from exc
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


def _category_counts(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(str(row[field]) for row in records)
    result = {category: int(counter.get(category, 0)) for category in CATEGORY_ORDER}
    if sum(result.values()) != len(records):
        raise ProtocolError(f"non-exhaustive category field: {field}")
    return result


def _category_by_context(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        counters[str(int(row["context"]))][str(row[field])] += 1
    return {
        context: {category: int(counter.get(category, 0)) for category in CATEGORY_ORDER}
        for context, counter in sorted(counters.items(), key=lambda item: int(item[0]))
    }


def _record_stream_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ptcg-yanz-ppo-dual-anchor-record-stream-v1\0")
    for row in records:
        digest.update(canonical_json_bytes(dict(row)))
    return digest.hexdigest()


def summarize_source(
    records: Sequence[Mapping[str, Any]], archive_counts: Mapping[str, int]
) -> dict[str, Any]:
    hybrid_categories = _category_counts(records, "hybrid_category")
    set_categories = _category_counts(records, "set_category")
    count_categories = _category_counts(records, "count_category")
    return {
        **dict(archive_counts),
        "bc": {
            "hybrid_correct": sum(bool(row["bc_hybrid_correct"]) for row in records),
            "set_correct": sum(bool(row["bc_set_correct"]) for row in records),
            "count_correct": sum(bool(row["bc_count_correct"]) for row in records),
        },
        "ppo": {
            "hybrid_correct": sum(bool(row["ppo_hybrid_correct"]) for row in records),
            "set_correct": sum(bool(row["ppo_set_correct"]) for row in records),
            "count_correct": sum(bool(row["ppo_count_correct"]) for row in records),
        },
        "hybrid_categories": hybrid_categories,
        "set_categories": set_categories,
        "count_categories": count_categories,
        "hybrid_categories_by_context": _category_by_context(records, "hybrid_category"),
        "bc_ppo_policy_action_equal": sum(
            bool(row["bc_ppo_policy_action_equal"]) for row in records
        ),
        "bc_ppo_hybrid_action_equal": sum(
            bool(row["bc_ppo_hybrid_action_equal"]) for row in records
        ),
        "bc_ppo_predicted_count_equal": sum(
            bool(row["bc_ppo_predicted_count_equal"]) for row in records
        ),
        "both_wrong_different_hybrid_action": sum(
            row["hybrid_category"] == "both_wrong"
            and not bool(row["bc_ppo_hybrid_action_equal"])
            for row in records
        ),
        "record_stream_sha256_algorithm": "ptcg-yanz-ppo-dual-anchor-record-stream-v1",
        "record_stream_sha256": _record_stream_sha256(records),
    }


def category_index(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result = {category: [] for category in CATEGORY_ORDER}
    for row in records:
        result[str(row["hybrid_category"])].append(str(row["line_sha256"]))
    if sum(len(values) for values in result.values()) != len(records):
        raise ProtocolError("hybrid category index is not exhaustive")
    return result


def build_profile_from_records(
    *,
    yanz: Sequence[dict[str, Any]],
    old: Sequence[dict[str, Any]],
    yanz_counts: Mapping[str, int],
    old_counts: Mapping[str, int],
    shards: Sequence[Mapping[str, Any]],
    static_bindings: Mapping[str, Any],
    cache_bindings: Mapping[str, Any],
    anchor_audit: Mapping[str, Any],
    profiler_sha256: str,
) -> dict[str, Any]:
    if len(yanz) != EXPECTED_YANZ_ROWS or len(old) != EXPECTED_OLD_ROWS:
        raise ProtocolError("full train row count drift")
    yanz_index = category_index(yanz)
    old_index = category_index(old)
    if set(yanz_index["bc_correct_ppo_wrong_recovery_target"]) & set(
        yanz_index["ppo_only_repair_protection"]
    ):
        raise ProtocolError("yanz recovery/protection categories overlap")
    index_payload = {"yanz": yanz_index, "old": old_index}
    return {
        "schema_version": SCHEMA,
        "status": "frozen_complete_train_only_dual_anchor_deployment_profile",
        "bindings": {
            **dict(static_bindings),
            "profiler": {"path": str(SCRIPT.resolve()), "sha256": profiler_sha256},
            "cache_contract": dict(cache_bindings),
            "anchors": dict(anchor_audit),
        },
        "profile_contract": {
            "opened_members": "train/*.jsonl only from the two fixed archives",
            "deployment_semantics": (
                "hybrid order; CPU; FP32; batch=1; one CPU thread; no autocast; "
                "pure BC teacher forward then PPO parent forward"
            ),
            "policy_under_protection": "ppo_parent",
            "teacher": "pure_bc_expanded61",
            "category_basis": "exact hybrid action versus expert action",
            "categories": {
                "ppo_correct_protection": "BC correct and PPO correct",
                "bc_correct_ppo_wrong_recovery_target": "BC correct and PPO wrong",
                "ppo_only_repair_protection": "BC wrong and PPO correct",
                "both_wrong": "BC wrong and PPO wrong",
            },
            "category_assignment": "exhaustive and mutually exclusive per row",
            "category_index_values": "line_sha256 in deterministic archive/member/line order",
            "per_row_payload_location": "immutable cache shard records",
            "profile_is_diagnostic_only": True,
        },
        "source_counts": {
            "yanz": summarize_source(yanz, yanz_counts),
            "old": summarize_source(old, old_counts),
        },
        "hybrid_category_index": {
            **index_payload,
            "sha256_algorithm": "sha256(canonical-json-v1)",
            "sha256": hashlib.sha256(canonical_json_bytes(index_payload)).hexdigest(),
        },
        "cache": {
            "immutable_shards": [dict(shard) for shard in shards],
            "shard_count": len(shards),
            "cache_artifacts_are_not_model_candidates": True,
        },
        "scope": {
            "train_members_only": True,
            "holdout_members_opened": False,
            "model_artifact_written": False,
            "submission_performed": False,
            "external_upload_performed": False,
        },
    }


def smoke(rows_per_source: int) -> dict[str, Any]:
    if not 1 <= rows_per_source <= 16:
        raise ValueError("--rows-per-source must be in [1, 16]")
    startup_sha = file_sha256(SCRIPT)
    static = validate_static_bindings()
    runtime = frozen.load_bound_runtime()
    bc_model, ppo_model, config, anchor_audit = load_bound_models(runtime)
    reports: dict[str, Any] = {}
    for source_name, path in (("yanz", YANZ), ("old", OLD)):
        records: list[dict[str, Any]] = []
        bc_main_matches = ppo_main_matches = 0
        try:
            chunks = frozen.iter_train_chunks(path, source_name, CACHE_CHUNK_ROWS)
            for chunk in chunks:
                for row, identity in chunk["items"]:
                    record = evaluate_dual_row(
                        row,
                        identity,
                        source_name,
                        runtime,
                        bc_model,
                        ppo_model,
                        config,
                    )
                    records.append(record)
                    if frozen.actual_hybrid_main_action(
                        row["observation"], runtime, bc_model, config
                    ) == record["bc_hybrid_action"]:
                        bc_main_matches += 1
                    if frozen.actual_hybrid_main_action(
                        row["observation"], runtime, ppo_model, config
                    ) == record["ppo_hybrid_action"]:
                        ppo_main_matches += 1
                    if len(records) == rows_per_source:
                        break
                if len(records) == rows_per_source:
                    break
        except frozen.ProtocolError as exc:
            raise ProtocolError(f"smoke train-only protocol failed: {exc}") from exc
        if len(records) != rows_per_source:
            raise ProtocolError(f"smoke row shortage: {source_name}")
        reports[source_name] = {
            "rows": len(records),
            "line_sha256": [str(row["line_sha256"]) for row in records],
            "hybrid_categories": _category_counts(records, "hybrid_category"),
            "bc_actual_hybrid_main_matches": bc_main_matches,
            "ppo_actual_hybrid_main_matches": ppo_main_matches,
            "contexts": [int(row["context"]) for row in records],
        }
    result = {
        "schema_version": f"{SCHEMA}-smoke",
        "status": "dual_anchor_deployment_semantics_smoke_passed",
        "bindings": {
            **static,
            "profiler": {"path": str(SCRIPT.resolve()), "sha256": startup_sha},
            "anchors": anchor_audit,
        },
        "sources": reports,
        "scope": {
            "train_members_only": True,
            "cache_written": False,
            "profile_written": False,
            "model_artifact_written": False,
            "submission_performed": False,
        },
    }
    assert_script_sha(startup_sha)
    return result


def build(cache_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dual-anchor profile: {output}")
    if cache_dir.resolve() == output.parent.resolve():
        raise ProtocolError("cache directory must be a dedicated child, not the output directory")
    startup_sha = file_sha256(SCRIPT)
    static = validate_static_bindings()
    runtime = frozen.load_bound_runtime()
    bc_model, ppo_model, config, anchor_audit = load_bound_models(runtime)
    cache_bindings = cache_contract_bindings(static, startup_sha)
    yanz, yanz_counts, yanz_shards = profile_archive_cached(
        path=YANZ,
        archive_sha256=YANZ_SHA256,
        source_name="yanz",
        cache_dir=cache_dir,
        cache_bindings=cache_bindings,
        runtime=runtime,
        bc_model=bc_model,
        ppo_model=ppo_model,
        config=config,
    )
    if yanz_counts["raw_rows"] != EXPECTED_YANZ_ROWS:
        raise ProtocolError("yanz full train row count drift")
    old, old_counts, old_shards = profile_archive_cached(
        path=OLD,
        archive_sha256=OLD_SHA256,
        source_name="old",
        cache_dir=cache_dir,
        cache_bindings=cache_bindings,
        runtime=runtime,
        bc_model=bc_model,
        ppo_model=ppo_model,
        config=config,
    )
    if old_counts["raw_rows"] != EXPECTED_OLD_ROWS:
        raise ProtocolError("old full train row count drift")
    profile = build_profile_from_records(
        yanz=yanz,
        old=old,
        yanz_counts=yanz_counts,
        old_counts=old_counts,
        shards=yanz_shards + old_shards,
        static_bindings=static,
        cache_bindings=cache_bindings,
        anchor_audit=anchor_audit,
        profiler_sha256=startup_sha,
    )
    if frozen.bitwise_model_state_sha256(bc_model.state_dict()) != (
        PURE_BC_EXPANDED61_BITWISE_STATE_SHA256
    ):
        raise ProtocolError("pure BC teacher changed during profiling")
    if frozen.bitwise_model_state_sha256(ppo_model.state_dict()) != (
        PPO_PARENT_BITWISE_STATE_SHA256
    ):
        raise ProtocolError("PPO parent changed during profiling")
    validate_static_bindings()
    assert_script_sha(startup_sha)
    write_new_json(profile, output)
    assert_script_sha(startup_sha)
    return {
        "status": str(profile["status"]),
        "output": str(output.resolve()),
        "sha256": file_sha256(output),
        "cache_dir": str(cache_dir.resolve()),
        "cache_shards": len(yanz_shards) + len(old_shards),
        "yanz_rows": len(yanz),
        "old_rows": len(old),
        "yanz_hybrid_categories": profile["source_counts"]["yanz"][
            "hybrid_categories"
        ],
        "old_hybrid_categories": profile["source_counts"]["old"][
            "hybrid_categories"
        ],
        "model_artifact_written": False,
        "submission_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke", help="read a few train rows, write nothing")
    smoke_parser.add_argument("--rows-per-source", type=int, default=2)
    build_parser = subparsers.add_parser("build", help="build immutable caches/profile")
    build_parser.add_argument("--cache-dir", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.command == "smoke":
        result = smoke(args.rows_per_source)
    else:
        result = build(args.cache_dir, args.output)
    sys.stdout.buffer.write(canonical_json_bytes(result))


if __name__ == "__main__":
    main()

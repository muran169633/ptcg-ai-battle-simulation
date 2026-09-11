#!/usr/bin/env python3
"""Armed one-shot full-train promotion gate for the sole equal-blend endpoint.

``contract`` emits the deterministic payload that must be preregistered before
``formal`` is allowed.  ``static-audit`` authenticates and deserializes the
sealed materialization chain, the raw parent, and every fixed train input, but
does not instantiate either policy or evaluate a row.  ``formal`` is present
for independent review; it refuses to start without the exact preregistration
path plus its caller-supplied SHA-256 and then consumes a one-shot O_EXCL marker.

The formal path is evaluation-only: raw U468 and the sole endpoint are each
evaluated on all train rows of FLG, PokemonFan, and core5.  All six evaluations
finish before the promotion decision.  No validation, broad, Gold, network,
package, upload, submission, training, optimizer, backward, or model-write path
exists here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_u468_raw_equalblend_train_only_promotion_gate_armed.py"

UNARMED_TOOL = TOOLS / "run_u468_raw_equalblend_train_only_promotion_gate.py"
UNARMED_TOOL_SHA256 = (
    "4f36bd0f86cbb811d8142f155027445b4bfb43a1b3e55b2fbf3b5787981aeae0"
)
UNARMED_CONTRACT = ROOT / (
    "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_contract_"
    "design202608116.json"
)
UNARMED_CONTRACT_SHA256 = (
    "e0d80b72aed237c3de4b643157b058acf2bff513d11420dd7579db0308b970c2"
)
UNARMED_CONTRACT_CANONICAL_SHA256 = (
    "21003ce1c1c21c0ee8cdc77a78aa7fb252b8d961c685de92adf90d6c7065c75e"
)

MATERIALIZER = TOOLS / "materialize_u468_raw_actor6_equalblend_sgd512_endpoint.py"
MATERIALIZER_SHA256 = (
    "fdd259ba846234db8358a630c392c637773645105116d3be7fe2089cbdf3ccd3"
)
MATERIALIZATION_BRANCH = (
    "ppo_u468_raw_actor6_equalblend_sgd512_materialize_design202608116"
)
MATERIALIZATION_PREREGISTRATION = ROOT / (
    f"artifacts/{MATERIALIZATION_BRANCH}.preregistration.json"
)
MATERIALIZATION_PREREGISTRATION_SHA256 = (
    "1d076b31e82d189ef829bc77bd89fe3808fd212bc3113298ae29aed437392882"
)
MATERIALIZATION_ATTEMPT_MARKER = ROOT / (
    f".ptcg-{MATERIALIZATION_BRANCH}-attempt.json"
)
MATERIALIZATION_ATTEMPT_MARKER_SHA256 = (
    "0a131e37a27f33c14a1e4f91ed7dba79ac2b340407c94f9e3704b624be20c460"
)
MATERIALIZATION_ROOT = ROOT / f"artifacts/{MATERIALIZATION_BRANCH}"
ENDPOINT = MATERIALIZATION_ROOT / "equalblend-sgd512-eval-only.pt"
ENDPOINT_FILE_SHA256 = (
    "1684493b48b9c6a77696150d925150c21d8028c5fdd45f80e9e838a94d472be2"
)
ENDPOINT_MODEL_STATE_SHA256 = (
    "5fd532338caf54abab95943d482dc35228559257e07d325bd1cacb20db62648d"
)
MATERIALIZATION_MANIFEST = MATERIALIZATION_ROOT / "materialization_manifest.json"
MATERIALIZATION_MANIFEST_SHA256 = (
    "51a133ec362ee489b05018728d43fb17f8ab12eb4adf1b223426afa6fafdccb1"
)
MATERIALIZATION_RESULT = MATERIALIZATION_ROOT / "materialization_result.json"
MATERIALIZATION_RESULT_SHA256 = (
    "9b0395c63478434ed6b577566ebd7663c6c4dfb20742ba747f374aeb618896e3"
)
MATERIALIZATION_ROOT_CONTENTS = {
    ENDPOINT.name,
    MATERIALIZATION_MANIFEST.name,
    MATERIALIZATION_RESULT.name,
}

SHADOW_RUNNER = TOOLS / "run_u468_raw_actor6_equalblend_sgd512_shadow.py"
SHADOW_RUNNER_SHA256 = (
    "e04f7b7579ef42d0c6f643db833779fb837ae86e3205d948b5564d08f8b30f8e"
)
SHADOW_PREREGISTRATION = ROOT / (
    "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_shadow_design202608115."
    "preregistration.json"
)
SHADOW_PREREGISTRATION_SHA256 = (
    "961ed51abcc5e502f603e5fc73385c64f9bc7fd0408154825920ed48002cb6da"
)
SHADOW_ATTEMPT_MARKER = ROOT / (
    ".ptcg-ppo_u468_raw_actor6_equalblend_sgd512_shadow_design202608115-"
    "attempt.json"
)
SHADOW_ATTEMPT_MARKER_SHA256 = (
    "42a14c657366cc51d104954dd00f1774af859850ef75ce304c629438ab22cbde"
)
SHADOW_RESULT = ROOT / (
    "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_shadow_design202608115."
    "result.json"
)
SHADOW_RESULT_SHA256 = (
    "a35d2050e92a9f0035847d3438241555573c053c4f99c0c4269a39bcd80bfe45"
)

EXPECTED_GRADIENT_SHA256 = (
    "47521ab0de80e6ec25f88a5704a7d9c185edf45197b25031e8b0c3e08be59e36"
)
EXPECTED_DIRECT_CACHE_SHA256 = (
    "ef5ab1b8f7e7162316e8ae80a6621f902e9a8e8cf73bb54daaf99e1355d0114a"
)
EXPECTED_DIRECT_BATCH_SHA256 = (
    "1c0bd23912b85dcbc64318d42ae41ba1fd7217f80e9a741e8970d91adeb1db05"
)
EXPECTED_FLAT_IDENTITY_SHA256 = (
    "d0c148e1c992030a9c5442b7a407b50816a30fdec27488f4f79b26b9e5916ce8"
)
EXPECTED_REFERENCE_HASHES = {
    "candidate_canonical_sha256": (
        "8eeecfa549ab52357eefbaa7e15f659cd721397e1eb36c58bcbe25e38ca36714"
    ),
    "displacement_canonical_sha256": (
        "57d636fab2a2561ebdbfe1795cf9708efdaeb5b987d49332b22469c12bf65456"
    ),
    "evaluations_canonical_sha256": (
        "b68278566bbb214886c731b06563c1260dbf7e4399f64b17ac191d841871aa32"
    ),
    "improvements_canonical_sha256": (
        "08fcfce87113816b1aedbcbabddee382b94c216951bb1da02715b7bf8bacd4b8"
    ),
    "training_canonical_sha256": (
        "5d75d665fefa99d99408ea638eb766aac0c735f12f1300326a938ab61d4ca46b"
    ),
}
EXPECTED_OBJECTIVE_FORMULA = (
    "0.5*L_union + (L_flg_hard+L_pokemonfan_hard+L_core5_hard)/6"
)
ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
FORBIDDEN_RESUME_KEYS = {
    "optimizer_state_dict",
    "bc_replay_optimizer_state_dict",
    "fresh_special_optimizer_state_dict",
    "opponent_quota_state",
}
EXPECTED_ENDPOINT_ROOT_KEYS = {
    "feature_version",
    "bc_feature_version",
    "config",
    "model_config",
    "learner_deck_hash",
    "reward",
    "action_distribution",
    "model_state_dict",
    "update",
    "evaluation_only",
    "resume_forbidden",
    "optimizer_states_omitted",
    "equalblend_sgd512_materialization",
}
PARENT_EXACT_METADATA_KEYS = (
    "feature_version",
    "bc_feature_version",
    "config",
    "model_config",
    "learner_deck_hash",
    "reward",
    "action_distribution",
)

BRANCH = "ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117"
FORMAL_PREREGISTRATION = ROOT / f"artifacts/{BRANCH}.preregistration.json"
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"
OUTPUT_ROOT = ROOT / f"artifacts/{BRANCH}"
SCHEMA = "ptcg-u468-raw-equalblend-train-only-promotion-gate-armed-v1"
PREREGISTRATION_SCHEMA = (
    "ptcg-u468-raw-equalblend-train-only-promotion-gate-armed-preregistration-v1"
)
FORMAL_SEED = 202608117
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    value = json.loads(
        payload,
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def read_bound(
    path: Path,
    expected_sha256: str,
    label: str,
    expected_mode: int,
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != expected_mode
        ):
            raise RuntimeError(f"{label} mode/type/link contract mismatch")
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
    identity = (after.st_dev, after.st_ino, after.st_size)
    if (
        (before.st_dev, before.st_ino, before.st_size) != identity
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or visible.st_nlink != 1
        or stat.S_IMODE(visible.st_mode) != expected_mode
        or (visible.st_dev, visible.st_ino, visible.st_size) != identity
    ):
        raise RuntimeError(f"{label} changed during authenticated read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    if digest != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 drift: expected {expected_sha256}, observed {digest}"
        )
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode": oct(expected_mode),
        "device": after.st_dev,
        "inode": after.st_ino,
        "nlink": after.st_nlink,
    }


# Authenticate the frozen policy contract before importing its executable code.
UNARMED_TOOL_PAYLOAD, UNARMED_TOOL_EVIDENCE = read_bound(
    UNARMED_TOOL, UNARMED_TOOL_SHA256, "frozen unarmed gate", 0o555
)
UNARMED_CONTRACT_PAYLOAD, UNARMED_CONTRACT_EVIDENCE = read_bound(
    UNARMED_CONTRACT,
    UNARMED_CONTRACT_SHA256,
    "frozen unarmed gate contract",
    0o444,
)
FROZEN_UNARMED_CONTRACT = strict_json_bytes(
    UNARMED_CONTRACT_PAYLOAD, "frozen unarmed gate contract"
)
if sha256_bytes(canonical_json_bytes(FROZEN_UNARMED_CONTRACT)) != (
    UNARMED_CONTRACT_CANONICAL_SHA256
):
    raise RuntimeError("frozen unarmed contract canonical SHA-256 mismatch")
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_UNARMED_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_equalblend_promotion_gate_4f36bd0f", UNARMED_TOOL
)
if _UNARMED_SPEC is None or _UNARMED_SPEC.loader is None:
    raise RuntimeError("cannot construct frozen unarmed gate import spec")
unarmed: ModuleType = importlib.util.module_from_spec(_UNARMED_SPEC)
_UNARMED_SPEC.loader.exec_module(unarmed)
helper = unarmed.helper
torch = helper.torch


def validate_runtime() -> None:
    if Path.cwd().resolve(strict=True) != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve(strict=True) != EXPECTED_PYTHON.resolve(strict=True):
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B flags")
    if Path(unarmed.__file__).resolve(strict=True) != UNARMED_TOOL.resolve(strict=True):
        raise RuntimeError("wrong frozen unarmed gate imported")
    if sha256_file(UNARMED_TOOL) != UNARMED_TOOL_SHA256:
        raise RuntimeError("frozen unarmed gate changed after import")


def verify_record(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    label: str,
    *,
    model_state_sha256: str | None = None,
) -> None:
    for key in ("path", "sha256", "bytes", "mode", "nlink"):
        if record.get(key) != evidence.get(key):
            raise RuntimeError(f"{label} record mismatch at {key}")
    if model_state_sha256 is not None and record.get("model_state_sha256") != (
        model_state_sha256
    ):
        raise RuntimeError(f"{label} model-state SHA record mismatch")


def verify_sealed_materialization_root() -> dict[str, Any]:
    observed = os.lstat(MATERIALIZATION_ROOT)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o555
    ):
        raise RuntimeError("materialization root is not a sealed mode-0555 directory")
    names = set(os.listdir(MATERIALIZATION_ROOT))
    if names != MATERIALIZATION_ROOT_CONTENTS:
        raise RuntimeError(f"materialization root contents drift: {sorted(names)}")
    for name in sorted(names):
        child = os.lstat(MATERIALIZATION_ROOT / name)
        if (
            stat.S_ISLNK(child.st_mode)
            or not stat.S_ISREG(child.st_mode)
            or child.st_nlink != 1
            or stat.S_IMODE(child.st_mode) != 0o444
        ):
            raise RuntimeError(f"unsealed materialization child: {name}")
    return {
        "path": str(MATERIALIZATION_ROOT.relative_to(ROOT)),
        "mode": "0o555",
        "exact_files": sorted(names),
        "exact_file_count": len(names),
        "all_files_mode_0o444_single_link_regular": True,
    }


def verify_shadow_recipe(
    preregistration: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    contract = preregistration.get("shadow_contract")
    if (
        preregistration.get("status") != "locked_before_actual"
        or not isinstance(contract, Mapping)
    ):
        raise RuntimeError("shadow preregistration contract mismatch")
    objective = contract.get("objective")
    optimizer = contract.get("optimizer")
    direct = contract.get("direct512")
    trajectory = contract.get("trajectory")
    if not all(isinstance(item, Mapping) for item in (objective, optimizer, direct, trajectory)):
        raise RuntimeError("shadow recipe sections are missing")
    if (
        contract.get("execution_seed") != 202608115
        or objective.get("formula") != EXPECTED_OBJECTIVE_FORMULA
        or objective.get("loss_mode") != "ordered"
        or objective.get("order_context_weight") != 8.0
        or objective.get("non_context34_fixed_multi_action_order_weight") != 1.0
        or objective.get("native_bf16_gradient_forward_calls") != 1
        or objective.get("complete_ordered_composite_scalar_calls") != 4
        or objective.get("combined_scalar_backward_calls") != 1
        or objective.get("no_extracted_gradient_reconstruction") is not True
        or objective.get("no_probe_gradient_reuse") is not True
        or optimizer.get("name") != "SGD"
        or optimizer.get("lr") != 5e-5
        or optimizer.get("momentum") != 0.0
        or optimizer.get("weight_decay") != 0.0
        or optimizer.get("max_grad_norm") != 0.5
        or optimizer.get("fresh") is not True
        or optimizer.get("plain") is not True
        or optimizer.get("state_entries_before_step") != 0
        or optimizer.get("state_entries_after_step") != 0
        or direct.get("rows") != 512
        or direct.get("cache_sha256") != EXPECTED_DIRECT_CACHE_SHA256
        or direct.get("batch_sha256") != EXPECTED_DIRECT_BATCH_SHA256
        or direct.get("flat_identity_sha256") != EXPECTED_FLAT_IDENTITY_SHA256
        or trajectory.get("actor_parameter_names") != list(ACTOR6_NAMES)
        or trajectory.get("candidate_count") != 1
        or trajectory.get("optimizer_step_calls") != 1
        or trajectory.get("model_mutation") != "RAM_only"
    ):
        raise RuntimeError("shadow equal-blend recipe drift")
    candidate = result.get("candidate")
    if (
        result.get("status") != "completed_actual_ram_only_equalblend_sgd512_step"
        or result.get("decision") != "GO"
        or result.get("single_candidate_only") is not True
        or result.get("checkpoint_writes") != 0
        or result.get("model_artifact_writes") != 0
        or result.get("training_artifact_writes") != 0
        or result.get("validation_member_payloads_opened") is not False
        or not isinstance(candidate, Mapping)
        or candidate.get("candidate")
        != "raw_u468_actor6_direct512_equalblend_plain_sgd_lr5e-5"
        or candidate.get("candidate_count") != 1
        or candidate.get("fully_passes") is not True
        or not all(candidate.get("gate_checks", {}).values())
    ):
        raise RuntimeError("shadow GO authorization drift")
    return {
        "execution_seed": 202608115,
        "formula": EXPECTED_OBJECTIVE_FORMULA,
        "optimizer": {
            "name": "SGD",
            "learning_rate": 5e-5,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "max_grad_norm": 0.5,
        },
        "direct512": {
            "rows": 512,
            "cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
            "batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
            "flat_identity_sha256": EXPECTED_FLAT_IDENTITY_SHA256,
        },
        "actor_parameter_names": list(ACTOR6_NAMES),
        "decision": "GO",
        "single_candidate_only": True,
    }


def verify_materialization_recipe(
    preregistration: Mapping[str, Any],
    marker: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    contract = preregistration.get("materialization_contract")
    if (
        preregistration.get("schema_version")
        != "ptcg-u468-raw-actor6-equalblend-sgd512-materialization-preregistration-v1"
        or preregistration.get("status") != "locked_before_materialization"
        or not isinstance(contract, Mapping)
        or marker.get("materialization_contract") != contract
        or manifest.get("materialization_contract") != contract
    ):
        raise RuntimeError("materialization contract is not exact across the chain")
    reproduction = contract.get("reproduction")
    direct = contract.get("direct512")
    outputs = contract.get("outputs")
    checkpoint = contract.get("checkpoint")
    if not all(
        isinstance(item, Mapping)
        for item in (reproduction, direct, outputs, checkpoint)
    ):
        raise RuntimeError("materialization recipe sections are missing")
    if (
        contract.get("branch") != MATERIALIZATION_BRANCH
        or contract.get("execution_seed") != 202608115
        or reproduction.get("actor_parameter_names") != list(ACTOR6_NAMES)
        or reproduction.get("combined_scalar_backward_calls") != 1
        or reproduction.get("complete_ordered_composite_scalar_calls") != 4
        or reproduction.get("gradient_clip_calls") != 1
        or reproduction.get("native_bf16_gradient_forward_calls") != 1
        or reproduction.get("plain_sgd_step_calls") != 1
        or reproduction.get("learning_rate") != 5e-5
        or reproduction.get("max_grad_norm") != 0.5
        or reproduction.get("post_step_retained_gradient_vector_sha256_float64")
        != EXPECTED_GRADIENT_SHA256
        or direct.get("rows") != 512
        or direct.get("cache_sha256") != EXPECTED_DIRECT_CACHE_SHA256
        or direct.get("batch_sha256") != EXPECTED_DIRECT_BATCH_SHA256
        or direct.get("flat_identity_sha256") != EXPECTED_FLAT_IDENTITY_SHA256
        or checkpoint.get("count") != 1
        or checkpoint.get("evaluation_only") is not True
        or checkpoint.get("resume_forbidden") is not True
        or set(checkpoint.get("forbidden_resume_keys", [])) != FORBIDDEN_RESUME_KEYS
        or outputs.get("fixed_contents") != sorted(MATERIALIZATION_ROOT_CONTENTS)
    ):
        raise RuntimeError("materialization equal-blend recipe drift")
    authorization = contract.get("shadow_authorization_chain", {}).get(
        "authorization"
    )
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("decision") != "GO"
        or authorization.get("single_candidate_only") is not True
        or authorization.get("equalblend_gradient_vector_sha256_float64")
        != EXPECTED_GRADIENT_SHA256
        or authorization.get("reference_hashes") != EXPECTED_REFERENCE_HASHES
    ):
        raise RuntimeError("materialization shadow authorization drift")
    return {
        "contract_exact_in_preregistration_marker_manifest": True,
        "single_eval_only_endpoint": True,
        "full_shadow_recipe_bound": True,
        "gradient_vector_sha256_float64": EXPECTED_GRADIENT_SHA256,
        "reference_hashes": dict(EXPECTED_REFERENCE_HASHES),
    }


def verify_endpoint_checkpoint(
    payload: bytes,
    parent_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = helper.checkpoint_from_bytes(payload, "sole equal-blend endpoint")
    if set(checkpoint) != EXPECTED_ENDPOINT_ROOT_KEYS:
        raise RuntimeError(f"endpoint root-key set drift: {sorted(checkpoint)}")
    if (
        helper.evaluator.checkpoint_kind(checkpoint) != "ppo"
        or checkpoint.get("update") != 468
        or checkpoint.get("evaluation_only") is not True
        or checkpoint.get("resume_forbidden") is not True
        or set(checkpoint.get("optimizer_states_omitted", [])) != FORBIDDEN_RESUME_KEYS
        or any(key in checkpoint for key in FORBIDDEN_RESUME_KEYS)
    ):
        raise RuntimeError("endpoint is not a non-resumable PPO eval-only checkpoint")
    for key in PARENT_EXACT_METADATA_KEYS:
        if checkpoint.get(key) != parent_checkpoint.get(key):
            raise RuntimeError(f"endpoint inference metadata drift: {key}")
    model_state = checkpoint["model_state_dict"]
    observed_model_sha256 = helper.model_state_sha256(model_state)
    if observed_model_sha256 != ENDPOINT_MODEL_STATE_SHA256:
        raise RuntimeError("endpoint model-state SHA-256 drift")
    actor6 = helper.verify_actor6_only(
        parent_checkpoint["model_state_dict"], model_state, "sole equal-blend endpoint"
    )
    if any(not bool(torch.isfinite(tensor).all()) for tensor in model_state.values()):
        raise RuntimeError("endpoint contains a nonfinite model tensor")
    parent_count = parent_checkpoint["model_state_dict"]["count_head.2.weight"]
    endpoint_count = model_state["count_head.2.weight"]
    if parent_count.shape[0] != endpoint_count.shape[0]:
        raise RuntimeError("endpoint count-head output class count drift")
    metadata = checkpoint.get("equalblend_sgd512_materialization")
    expected_metadata = {
        "schema_version": "ptcg-u468-raw-actor6-equalblend-sgd512-materializer-v1",
        "materializer_runner_sha256": MATERIALIZER_SHA256,
        "materialization_preregistration_sha256": (
            MATERIALIZATION_PREREGISTRATION_SHA256
        ),
        "promotion_gate_tool_sha256": UNARMED_TOOL_SHA256,
        "promotion_gate_contract_sha256": UNARMED_CONTRACT_SHA256,
        "shadow_runner_sha256": SHADOW_RUNNER_SHA256,
        "shadow_preregistration_sha256": SHADOW_PREREGISTRATION_SHA256,
        "shadow_attempt_marker_sha256": SHADOW_ATTEMPT_MARKER_SHA256,
        "shadow_result_sha256": SHADOW_RESULT_SHA256,
        "shadow_reference_hashes": EXPECTED_REFERENCE_HASHES,
        "direct512_cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
        "direct512_batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
        "equalblend_gradient_vector_sha256_float64": EXPECTED_GRADIENT_SHA256,
        "actor_parameter_names": list(ACTOR6_NAMES),
        "base_model_state_sha256": unarmed.PARENT_MODEL_STATE_SHA256,
        "endpoint_model_state_sha256": ENDPOINT_MODEL_STATE_SHA256,
        "evaluation_only": True,
        "resume_forbidden": True,
        "validation_member_payloads_opened": False,
    }
    if metadata != expected_metadata:
        raise RuntimeError("endpoint embedded materialization metadata drift")
    return checkpoint, {
        "checkpoint_kind": "ppo",
        "root_key_set_exact": True,
        "evaluation_only": True,
        "resume_forbidden": True,
        "optimizer_and_quota_states_absent": True,
        "model_state_sha256": observed_model_sha256,
        "actor6_audit": actor6,
        "count_head_output_classes_exact_parent": True,
        "all_model_tensors_finite": True,
        "embedded_materialization_metadata_exact": True,
        "authorization_uses_top_level_eval_only_and_resume_forbidden_only": True,
        "nested_config_resume_text_is_inert_and_not_authorization": True,
    }


def verify_materialization_chain(
    parent_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sealed_root = verify_sealed_materialization_root()
    bindings: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for name, path, digest, mode in (
        ("materializer", MATERIALIZER, MATERIALIZER_SHA256, 0o555),
        (
            "materialization_preregistration",
            MATERIALIZATION_PREREGISTRATION,
            MATERIALIZATION_PREREGISTRATION_SHA256,
            0o444,
        ),
        (
            "materialization_attempt_marker",
            MATERIALIZATION_ATTEMPT_MARKER,
            MATERIALIZATION_ATTEMPT_MARKER_SHA256,
            0o444,
        ),
        ("endpoint", ENDPOINT, ENDPOINT_FILE_SHA256, 0o444),
        (
            "materialization_manifest",
            MATERIALIZATION_MANIFEST,
            MATERIALIZATION_MANIFEST_SHA256,
            0o444,
        ),
        (
            "materialization_result",
            MATERIALIZATION_RESULT,
            MATERIALIZATION_RESULT_SHA256,
            0o444,
        ),
        ("shadow_runner", SHADOW_RUNNER, SHADOW_RUNNER_SHA256, 0o555),
        (
            "shadow_preregistration",
            SHADOW_PREREGISTRATION,
            SHADOW_PREREGISTRATION_SHA256,
            0o444,
        ),
        (
            "shadow_attempt_marker",
            SHADOW_ATTEMPT_MARKER,
            SHADOW_ATTEMPT_MARKER_SHA256,
            0o444,
        ),
        ("shadow_result", SHADOW_RESULT, SHADOW_RESULT_SHA256, 0o444),
    ):
        payload, evidence = read_bound(path, digest, name, mode)
        payloads[name] = payload
        bindings[name] = evidence

    preregistration = strict_json_bytes(
        payloads["materialization_preregistration"], "materialization preregistration"
    )
    marker = strict_json_bytes(
        payloads["materialization_attempt_marker"], "materialization attempt marker"
    )
    manifest = strict_json_bytes(
        payloads["materialization_manifest"], "materialization manifest"
    )
    result = strict_json_bytes(
        payloads["materialization_result"], "materialization completion result"
    )
    shadow_preregistration = strict_json_bytes(
        payloads["shadow_preregistration"], "shadow preregistration"
    )
    shadow_result = strict_json_bytes(payloads["shadow_result"], "shadow result")

    if (
        marker.get("schema_version")
        != "ptcg-u468-raw-actor6-equalblend-sgd512-materialization-attempt-v1"
        or marker.get("status")
        != "materialization_attempt_consumed_before_model_or_optimizer"
        or marker.get("branch") != MATERIALIZATION_BRANCH
        or marker.get("checkpoint_writes_before_lock") != 0
        or marker.get("model_instances_created_before_lock") != 0
        or marker.get("optimizer_instances_created_before_lock") != 0
        or marker.get("validation_member_payloads_opened") is not False
    ):
        raise RuntimeError("materialization attempt-marker contract drift")
    if (
        manifest.get("schema_version")
        != "ptcg-u468-raw-actor6-equalblend-sgd512-materialization-manifest-v1"
        or manifest.get("status")
        != "endpoint_materialized_and_verified_before_completion_result"
        or manifest.get("branch") != MATERIALIZATION_BRANCH
    ):
        raise RuntimeError("materialization manifest contract drift")
    if (
        result.get("schema_version")
        != "ptcg-u468-raw-actor6-equalblend-sgd512-materialization-result-v1"
        or result.get("status") != "completed_one_eval_only_endpoint_materialization"
        or result.get("decision") != "MATERIALIZED"
        or result.get("branch") != MATERIALIZATION_BRANCH
        or result.get("checkpoint_writes") != 1
        or result.get("manifest_writes") != 1
        or result.get("result_writes") != 1
        or result.get("submission_performed") is not False
        or result.get("validation_member_payloads_opened") is not False
    ):
        raise RuntimeError("materialization completion-result contract drift")

    verify_record(
        result.get("manifest", {}),
        bindings["materialization_manifest"],
        "result->manifest",
    )
    verify_record(
        result.get("endpoint", {}),
        bindings["endpoint"],
        "result->endpoint",
        model_state_sha256=ENDPOINT_MODEL_STATE_SHA256,
    )
    verify_record(
        result.get("preregistration", {}),
        bindings["materialization_preregistration"],
        "result->preregistration",
    )
    verify_record(
        result.get("attempt_marker", {}),
        bindings["materialization_attempt_marker"],
        "result->attempt-marker",
    )
    verify_record(
        manifest.get("endpoint", {}),
        bindings["endpoint"],
        "manifest->endpoint",
        model_state_sha256=ENDPOINT_MODEL_STATE_SHA256,
    )
    verify_record(
        manifest.get("preregistration", {}),
        bindings["materialization_preregistration"],
        "manifest->preregistration",
    )
    verify_record(
        manifest.get("attempt_marker", {}),
        bindings["materialization_attempt_marker"],
        "manifest->attempt-marker",
    )
    verify_record(manifest.get("runner", {}), bindings["materializer"], "manifest->runner")

    precommit = manifest.get("promotion_gate_precommit")
    if (
        not isinstance(precommit, Mapping)
        or precommit.get("tool", {}).get("sha256") != UNARMED_TOOL_SHA256
        or precommit.get("frozen_contract_artifact", {}).get("sha256")
        != UNARMED_CONTRACT_SHA256
        or precommit.get("contract_payload_canonical_sha256")
        != UNARMED_CONTRACT_CANONICAL_SHA256
        or precommit.get("endpoint_placeholders_unarmed") is not True
        or precommit.get("frozen_before_endpoint_materialization") is not True
        or precommit.get("validation_member_payloads_opened") is not False
        or precommit.get("validation_results_read") is not False
    ):
        raise RuntimeError("materialization promotion-gate precommit drift")
    if result.get("promotion_gate_contract", {}).get("sha256") != (
        UNARMED_CONTRACT_SHA256
    ) or result.get("promotion_gate_tool", {}).get("sha256") != UNARMED_TOOL_SHA256:
        raise RuntimeError("completion result lost frozen promotion-gate binding")

    shadow_recipe = verify_shadow_recipe(shadow_preregistration, shadow_result)
    materialization_recipe = verify_materialization_recipe(
        preregistration, marker, manifest
    )
    reproduction = manifest.get("reproduction")
    integrity = manifest.get("integrity")
    authorization = manifest.get("shadow_authorization")
    if (
        not isinstance(reproduction, Mapping)
        or reproduction.get("status") != "exact_shadow_record_reproduced"
        or reproduction.get("deterministic_full_record_matches_shadow_exactly")
        is not True
        or reproduction.get("frozen_tensors_unchanged") is not True
        or reproduction.get("endpoint_model_state_sha256")
        != ENDPOINT_MODEL_STATE_SHA256
        or reproduction.get("validation_member_payloads_opened") is not False
        or reproduction.get("retained_actor6_gradient", {}).get("exact_match")
        is not True
        or reproduction.get("retained_actor6_gradient", {}).get(
            "expected_vector_sha256_float64"
        )
        != EXPECTED_GRADIENT_SHA256
        or reproduction.get("retained_actor6_gradient", {}).get(
            "vector_sha256_float64"
        )
        != EXPECTED_GRADIENT_SHA256
        or reproduction.get("retained_actor6_gradient", {}).get("parameter_names")
        != list(ACTOR6_NAMES)
        or not isinstance(integrity, Mapping)
        or not all(
            integrity.get(key) is True
            for key in (
                "evaluation_only",
                "optimizer_and_quota_states_omitted",
                "resume_forbidden",
                "shadow_displacement_record_exact",
                "shadow_full_evaluation_record_exact",
                "shadow_full_training_record_exact",
            )
        )
        or integrity.get("endpoint_checkpoint_count_exact") != 1
        or integrity.get("submission_performed") is not False
        or integrity.get("validation_member_payloads_opened") is not False
        or not isinstance(authorization, Mapping)
        or authorization.get("decision") != "GO"
        or authorization.get("equalblend_gradient_vector_sha256_float64")
        != EXPECTED_GRADIENT_SHA256
        or authorization.get("reference_hashes") != EXPECTED_REFERENCE_HASHES
    ):
        raise RuntimeError("materialization reproduction/integrity evidence drift")

    endpoint_checkpoint, endpoint_audit = verify_endpoint_checkpoint(
        payloads["endpoint"], parent_checkpoint
    )
    return endpoint_checkpoint, {
        "sealed_root": sealed_root,
        "bindings": bindings,
        "result_to_manifest_to_endpoint_exact": True,
        "unarmed_gate_precommit_exact": True,
        "shadow_recipe": shadow_recipe,
        "materialization_recipe": materialization_recipe,
        "endpoint": endpoint_audit,
    }


def source_ast_audit() -> dict[str, Any]:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))

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
        if name.endswith(".backward")
        or name.endswith(".step")
        or name == "torch.save"
        or name.startswith("torch.optim")
    )
    if forbidden:
        raise RuntimeError(f"armed gate contains a training/update call: {forbidden}")
    expected = {
        "helper.evaluate_panel": 1,
        "unarmed.gate_panel": 1,
        "helper.instantiate_checkpoint": 1,
        "helper.publish_o_excl": 1,
        "helper.seal_tree": 1,
    }
    observed = {name: calls.count(name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"armed gate formal call-site drift: {observed}")
    marker_publish_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and dotted(node.func) == "publish_json"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ATTEMPT_MARKER"
    ]
    instantiate_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and dotted(node.func) == "helper.instantiate_checkpoint"
    ]
    evaluate_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and dotted(node.func) == "helper.evaluate_panel"
    ]
    if (
        len(marker_publish_lines) != 1
        or len(instantiate_lines) != 1
        or len(evaluate_lines) != 1
        or not marker_publish_lines[0] < instantiate_lines[0] < evaluate_lines[0]
    ):
        raise RuntimeError("formal marker is not before model instantiate/evaluate")
    return {
        "status": "armed_train_only_static_ast_audit_passed",
        "call_sites": observed,
        "optimizer_backward_step_save_calls": 0,
        "training_calls": 0,
        "candidate_count": 1,
        "formal_requires_preregistration_sha": True,
        "formal_marker_before_model_instantiate_and_evaluate": True,
        "formal_line_order": {
            "marker_publish": marker_publish_lines[0],
            "model_instantiate": instantiate_lines[0],
            "evaluation": evaluate_lines[0],
        },
    }


def armed_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "armed_contract_frozen_before_formal_not_executed",
        "runner": {
            "path": str(SCRIPT.relative_to(ROOT)),
            "sha256": sha256_file(SCRIPT),
            "python": str(EXPECTED_PYTHON),
            "flags": ["-I", "-B"],
        },
        "frozen_unarmed_gate": {
            "tool": {
                "path": str(UNARMED_TOOL.relative_to(ROOT)),
                "sha256": UNARMED_TOOL_SHA256,
            },
            "contract_artifact": {
                "path": str(UNARMED_CONTRACT.relative_to(ROOT)),
                "sha256": UNARMED_CONTRACT_SHA256,
                "canonical_sha256": UNARMED_CONTRACT_CANONICAL_SHA256,
            },
            "promotion_gate_byte_semantics": FROZEN_UNARMED_CONTRACT[
                "promotion_gate"
            ],
            "data": FROZEN_UNARMED_CONTRACT["data"],
            "evaluation": FROZEN_UNARMED_CONTRACT["evaluation"],
            "checkpoint_format": FROZEN_UNARMED_CONTRACT["checkpoint_format"],
        },
        "materialization_chain": {
            "runner": {
                "path": str(MATERIALIZER.relative_to(ROOT)),
                "sha256": MATERIALIZER_SHA256,
            },
            "preregistration": {
                "path": str(MATERIALIZATION_PREREGISTRATION.relative_to(ROOT)),
                "sha256": MATERIALIZATION_PREREGISTRATION_SHA256,
            },
            "attempt_marker": {
                "path": str(MATERIALIZATION_ATTEMPT_MARKER.relative_to(ROOT)),
                "sha256": MATERIALIZATION_ATTEMPT_MARKER_SHA256,
            },
            "endpoint": {
                "path": str(ENDPOINT.relative_to(ROOT)),
                "file_sha256": ENDPOINT_FILE_SHA256,
                "model_state_sha256": ENDPOINT_MODEL_STATE_SHA256,
            },
            "manifest": {
                "path": str(MATERIALIZATION_MANIFEST.relative_to(ROOT)),
                "sha256": MATERIALIZATION_MANIFEST_SHA256,
            },
            "completion_result": {
                "path": str(MATERIALIZATION_RESULT.relative_to(ROOT)),
                "sha256": MATERIALIZATION_RESULT_SHA256,
            },
            "sealed_root": {
                "path": str(MATERIALIZATION_ROOT.relative_to(ROOT)),
                "mode": "0o555",
                "exact_files": sorted(MATERIALIZATION_ROOT_CONTENTS),
                "file_mode": "0o444",
            },
            "required_chain": "completion result -> manifest -> sole endpoint",
        },
        "recipe": {
            "execution_seed": 202608115,
            "formula": EXPECTED_OBJECTIVE_FORMULA,
            "direct512": {
                "rows": 512,
                "cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
                "batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
                "flat_identity_sha256": EXPECTED_FLAT_IDENTITY_SHA256,
            },
            "optimizer": {
                "name": "SGD",
                "learning_rate": 5e-5,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "max_grad_norm": 0.5,
                "steps": 1,
            },
            "actor_parameter_names": list(ACTOR6_NAMES),
            "gradient_vector_sha256_float64": EXPECTED_GRADIENT_SHA256,
            "shadow_reference_hashes": dict(EXPECTED_REFERENCE_HASHES),
        },
        "formal_evaluation": {
            "models": ["raw_full_u468", "sole_equalblend_endpoint"],
            "panels": ["flg", "pokemonfan", "core5"],
            "evaluation_count": 6,
            "all_six_complete_before_decision": True,
            "batch_size": unarmed.BATCH_SIZE,
            "workers": unarmed.WORKERS,
            "max_rows": unarmed.MAX_ROWS,
            "device": unarmed.DEVICE,
            "prediction_order": "policy_greedy",
            "canonicalize_order": False,
            "promotion_logic": FROZEN_UNARMED_CONTRACT["promotion_gate"],
            "maximum_promoted_endpoints": 1,
        },
        "formal_authorization": {
            "preregistration_path": str(FORMAL_PREREGISTRATION.relative_to(ROOT)),
            "caller_must_supply_exact_preregistration_sha256": True,
            "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
            "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
            "attempts": 1,
            "retry": False,
            "publication": "O_EXCL then read-only tree seal",
        },
        "scope": {
            "train_only": True,
            "training": False,
            "validation": False,
            "broad": False,
            "gold": False,
            "network": False,
            "package": False,
            "upload": False,
            "submission": False,
            "model_write": False,
        },
        "resume_authorization_semantics": {
            "top_level_evaluation_only_must_be_true": True,
            "top_level_resume_forbidden_must_be_true": True,
            "forbidden_optimizer_and_quota_state_keys_must_be_absent": sorted(
                FORBIDDEN_RESUME_KEYS
            ),
            "nested_config_resume_checkpoint_or_reset_text_grants_authority": False,
        },
    }


def preregistration_payload() -> dict[str, Any]:
    return {
        "schema_version": PREREGISTRATION_SCHEMA,
        "status": "locked_before_formal_train_only_evaluation",
        "branch": BRANCH,
        "armed_contract": armed_contract(),
    }


def verify_fixed_train_inputs() -> tuple[
    dict[str, Any],
    dict[str, bytes],
    dict[str, Any],
    dict[str, Any],
]:
    evidence, payloads = helper.verify_fixed_inputs()
    profile = helper.verify_profile(payloads["profile"])
    parent = helper.verify_parent_checkpoint(payloads["parent"])
    if FROZEN_UNARMED_CONTRACT != unarmed.promotion_contract():
        raise RuntimeError("frozen unarmed artifact no longer equals frozen tool contract")
    if (
        unarmed.DATA_SHA256 != helper.DATA_SHA256
        or unarmed.EXPECTED_TRAIN_SHAPE != helper.EXPECTED_TRAIN_SHAPE
        or unarmed.BATCH_SIZE != helper.BATCH_SIZE
        or unarmed.ACTOR6_NAMES != ACTOR6_NAMES
    ):
        raise RuntimeError("frozen full-train helper/unarmed contract drift")
    return evidence, payloads, profile, parent


def static_audit() -> dict[str, Any]:
    fixed_evidence, _, profile, parent = verify_fixed_train_inputs()
    _, materialization = verify_materialization_chain(parent)
    preregistration_status: dict[str, Any]
    if FORMAL_PREREGISTRATION.exists() or FORMAL_PREREGISTRATION.is_symlink():
        payload, evidence = helper.read_regular_file(
            FORMAL_PREREGISTRATION, "armed formal preregistration"
        )
        if helper.load_json_bytes(payload, "armed formal preregistration") != (
            preregistration_payload()
        ):
            raise RuntimeError("present armed formal preregistration content drift")
        preregistration_status = {
            "present": True,
            "binding": evidence,
            "formal_still_requires_caller_supplied_sha256": True,
        }
    else:
        preregistration_status = {
            "present": False,
            "formal_blocked_until_frozen": True,
        }
    return {
        "schema_version": SCHEMA,
        "status": "armed_static_audit_passed_formal_not_executed",
        "runner": armed_contract()["runner"],
        "ast_audit": source_ast_audit(),
        "fixed_train_inputs": fixed_evidence,
        "raw_profile": profile,
        "materialization_chain": materialization,
        "formal_preregistration": preregistration_status,
        "formal_attempt_marker_absent": not ATTEMPT_MARKER.exists(),
        "formal_output_root_absent": not OUTPUT_ROOT.exists(),
        "endpoint_checkpoint_deserialized_cpu": True,
        "candidate_model_instantiated": False,
        "formal_evaluation_executed": False,
        "validation_member_payloads_opened": False,
        "validation_results_read": False,
        "writes_performed": False,
    }


def publish_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    evidence = helper.publish_o_excl(path, canonical_json_bytes(value))
    os.chmod(path, 0o444)
    _, sealed_evidence = helper.read_regular_file(
        path,
        f"sealed formal output {path.relative_to(ROOT)}",
        evidence["sha256"],
    )
    if sealed_evidence["mode"] != "0o444":
        raise RuntimeError(f"formal output did not seal read-only: {path}")
    return sealed_evidence


def verify_exact_sealed_output_tree(
    file_bindings: list[Mapping[str, Any]],
) -> dict[str, Any]:
    raw_dir = OUTPUT_ROOT / "raw_full_u468"
    endpoint_dir = OUTPUT_ROOT / "sole_equalblend_endpoint"
    expected_root_entries = {
        raw_dir.name,
        endpoint_dir.name,
        "gate_decision.json",
        "execution_manifest.json",
        "COMPLETED.json",
    }
    expected_panel_entries = {f"{panel}.json" for panel in unarmed.DATASETS}
    expected_files = {
        OUTPUT_ROOT / "gate_decision.json",
        OUTPUT_ROOT / "execution_manifest.json",
        OUTPUT_ROOT / "COMPLETED.json",
        *(raw_dir / name for name in expected_panel_entries),
        *(endpoint_dir / name for name in expected_panel_entries),
    }
    if len(expected_files) != 9:
        raise RuntimeError("sealed formal tree contract must contain exactly nine files")

    expected_entries = {
        OUTPUT_ROOT: expected_root_entries,
        raw_dir: expected_panel_entries,
        endpoint_dir: expected_panel_entries,
    }
    directory_evidence: list[dict[str, Any]] = []
    for directory, expected in expected_entries.items():
        visible = os.lstat(directory)
        if (
            stat.S_ISLNK(visible.st_mode)
            or not stat.S_ISDIR(visible.st_mode)
            or stat.S_IMODE(visible.st_mode) != 0o555
        ):
            raise RuntimeError(f"sealed formal directory contract mismatch: {directory}")
        observed = {entry.name for entry in os.scandir(directory)}
        if observed != expected:
            raise RuntimeError(
                f"sealed formal directory contents mismatch for {directory}: "
                f"expected {sorted(expected)}, observed {sorted(observed)}"
            )
        directory_evidence.append(
            {
                "path": str(directory.relative_to(ROOT)),
                "mode": "0o555",
                "device": visible.st_dev,
                "inode": visible.st_ino,
                "exact_entries": sorted(observed),
            }
        )

    bindings_by_path: dict[Path, Mapping[str, Any]] = {}
    for binding in file_bindings:
        path_value = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise RuntimeError("sealed formal output binding is malformed")
        path = ROOT / path_value
        if path in bindings_by_path:
            raise RuntimeError(f"duplicate sealed formal output binding: {path}")
        bindings_by_path[path] = binding
    if set(bindings_by_path) != expected_files:
        raise RuntimeError("sealed formal output bindings do not cover exact nine files")

    sealed_files: list[dict[str, Any]] = []
    for path in sorted(expected_files):
        binding = bindings_by_path[path]
        _, evidence = read_bound(
            path,
            str(binding["sha256"]),
            f"sealed formal output {path.relative_to(ROOT)}",
            0o444,
        )
        if evidence != dict(binding):
            raise RuntimeError(f"sealed formal output binding drift: {path}")
        sealed_files.append(evidence)
    return {
        "status": "exact_sealed_output_tree_verified",
        "directory_count": 3,
        "file_count": 9,
        "directories": directory_evidence,
        "files": sealed_files,
    }


def verify_formal_preregistration(expected_sha256: str | None) -> dict[str, Any]:
    if expected_sha256 is None or SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("formal requires --expected-preregistration-sha256")
    payload, evidence = read_bound(
        FORMAL_PREREGISTRATION,
        expected_sha256,
        "armed formal preregistration",
        0o444,
    )
    observed = strict_json_bytes(payload, "armed formal preregistration")
    if observed != preregistration_payload():
        raise RuntimeError("armed formal preregistration does not match this runner")
    return evidence


def formal(expected_preregistration_sha256: str | None) -> dict[str, Any]:
    preflight_contract = armed_contract()
    preregistration = verify_formal_preregistration(
        expected_preregistration_sha256
    )
    if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
        raise FileExistsError(ATTEMPT_MARKER)
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)

    fixed_evidence, payloads, profile, parent = verify_fixed_train_inputs()
    endpoint_checkpoint, materialization = verify_materialization_chain(parent)
    if not torch.cuda.is_available():
        raise RuntimeError("formal full-train promotion gate requires CUDA")

    # Re-authenticate every mutable input immediately before consuming the
    # one-shot marker.  The second pass must be byte/evidence-equivalent to the
    # preflight pass; only the second-pass payloads are used below.
    preregistration_at_lock = verify_formal_preregistration(
        expected_preregistration_sha256
    )
    (
        fixed_evidence_at_lock,
        payloads_at_lock,
        profile_at_lock,
        parent_at_lock,
    ) = verify_fixed_train_inputs()
    endpoint_checkpoint_at_lock, materialization_at_lock = (
        verify_materialization_chain(parent_at_lock)
    )
    contract_at_lock = armed_contract()
    if (
        contract_at_lock != preflight_contract
        or preregistration_at_lock != preregistration
        or fixed_evidence_at_lock != fixed_evidence
        or payloads_at_lock != payloads
        or profile_at_lock != profile
        or materialization_at_lock != materialization
    ):
        raise RuntimeError("formal inputs or runner drifted between preflight and lock")
    if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
        raise FileExistsError(ATTEMPT_MARKER)
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)

    preregistration = preregistration_at_lock
    fixed_evidence = fixed_evidence_at_lock
    payloads = payloads_at_lock
    profile = profile_at_lock
    endpoint_checkpoint = endpoint_checkpoint_at_lock
    materialization = materialization_at_lock
    marker = {
        "schema_version": SCHEMA,
        "status": "formal_attempt_consumed_before_any_train_evaluation",
        "branch": BRANCH,
        "runner": contract_at_lock["runner"],
        "preregistration": preregistration,
        "materialization_chain": contract_at_lock["materialization_chain"],
        "scope": contract_at_lock["scope"],
        "at_lock_reverification": {
            "runner_contract_exact_to_preflight": True,
            "preregistration_evidence_exact_to_preflight": True,
            "fixed_input_evidence_and_bytes_exact_to_preflight": True,
            "materialization_evidence_exact_to_preflight": True,
            "formal_preregistration": preregistration_at_lock,
            "fixed_train_inputs": fixed_evidence_at_lock,
            "materialization": materialization_at_lock,
        },
        "formal_evaluations_started_before_lock": 0,
        "validation_member_payloads_opened": False,
        "validation_results_read": False,
    }
    marker_evidence = publish_json(ATTEMPT_MARKER, marker)
    os.mkdir(OUTPUT_ROOT, mode=0o700)
    raw_dir = OUTPUT_ROOT / "raw_full_u468"
    endpoint_dir = OUTPUT_ROOT / "sole_equalblend_endpoint"
    os.mkdir(raw_dir, mode=0o700)
    os.mkdir(endpoint_dir, mode=0o700)

    started = time.time()
    torch.manual_seed(FORMAL_SEED)
    torch.cuda.manual_seed_all(FORMAL_SEED)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    models = (
        (
            "raw_full_u468",
            helper.checkpoint_from_bytes(payloads["parent"], "raw full U468"),
            raw_dir,
            {
                "path": str(unarmed.PARENT.relative_to(ROOT)),
                "file_sha256": unarmed.PARENT_FILE_SHA256,
                "model_state_sha256": unarmed.PARENT_MODEL_STATE_SHA256,
            },
        ),
        (
            "sole_equalblend_endpoint",
            endpoint_checkpoint,
            endpoint_dir,
            {
                "path": str(ENDPOINT.relative_to(ROOT)),
                "file_sha256": ENDPOINT_FILE_SHA256,
                "model_state_sha256": ENDPOINT_MODEL_STATE_SHA256,
            },
        ),
    )
    evaluations: dict[str, dict[str, dict[str, Any]]] = {}
    result_bindings: list[dict[str, Any]] = []
    for model_name, checkpoint, model_dir, binding in models:
        model, model_config, kind = helper.instantiate_checkpoint(
            checkpoint, device
        )
        model_results: dict[str, dict[str, Any]] = {}
        for panel in unarmed.DATASETS:
            panel_result = {
                "schema_version": SCHEMA,
                "status": "full_train_evaluation_completed",
                "model_name": model_name,
                "model": binding,
                "checkpoint_kind": kind,
                "checkpoint_update": checkpoint.get("update"),
                "device": str(device),
                "validation_results_read": False,
                **helper.evaluate_panel(
                    model,
                    model_config,
                    payloads[f"dataset:{panel}"],
                    panel,
                    device,
                ),
            }
            if model_name == "raw_full_u468":
                expected = unarmed.EXPECTED_TRAIN_SHAPE[panel]
                metrics = panel_result["metrics"]
                if (
                    metrics["set_exact_correct"]
                    != expected["profile_set_exact_correct"]
                    or metrics["ordered_exact_correct"]
                    != expected["profile_ordered_exact_correct"]
                ):
                    raise RuntimeError(f"{panel} raw metrics disagree with profile")
            model_results[panel] = panel_result
            result_bindings.append(
                publish_json(model_dir / f"{panel}.json", panel_result)
            )
        evaluations[model_name] = model_results
        del model
        torch.cuda.empty_cache()

    if len(result_bindings) != 6 or any(
        set(evaluations[name]) != set(unarmed.DATASETS) for name in evaluations
    ):
        raise RuntimeError("all six full-train evaluations did not complete")
    raw_metrics = {
        panel: evaluations["raw_full_u468"][panel]["metrics"]
        for panel in unarmed.DATASETS
    }
    endpoint_metrics = {
        panel: evaluations["sole_equalblend_endpoint"][panel]["metrics"]
        for panel in unarmed.DATASETS
    }
    panel_gates = {
        panel: unarmed.gate_panel(
            panel, raw_metrics[panel], endpoint_metrics[panel]
        )
        for panel in unarmed.DATASETS
    }
    promoted = all(record["pass"] for record in panel_gates.values())
    decision = {
        "schema_version": SCHEMA,
        "status": (
            "promoted_one_endpoint_for_separate_valid_preregistration"
            if promoted
            else "not_promoted_branch_closed_before_valid"
        ),
        "branch": BRANCH,
        "all_six_full_train_evaluations_completed_before_decision": True,
        "evaluation_count": 6,
        "panel_gates": panel_gates,
        "promoted": promoted,
        "promoted_endpoint": "sole_equalblend_endpoint" if promoted else None,
        "promoted_endpoint_count": int(promoted),
        "maximum_promoted_endpoints": 1,
        "frozen_promotion_gate": FROZEN_UNARMED_CONTRACT["promotion_gate"],
        "raw_profile": profile,
        "validation_member_payloads_opened": False,
        "validation_results_read": False,
        "downstream_executed": {
            "valid": False,
            "broad": False,
            "gold": False,
            "network": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }
    decision_binding = publish_json(OUTPUT_ROOT / "gate_decision.json", decision)
    execution_manifest = {
        "schema_version": SCHEMA,
        "status": "six_full_train_evaluations_and_decision_completed",
        "branch": BRANCH,
        "runner": armed_contract()["runner"],
        "preregistration": preregistration,
        "attempt_marker": marker_evidence,
        "fixed_train_inputs": fixed_evidence,
        "materialization": materialization,
        "ordered_evaluation_results": result_bindings,
        "evaluation_count": 6,
        "decision": decision_binding,
        "seconds": time.time() - started,
        "validation_member_payloads_opened": False,
        "validation_results_read": False,
    }
    execution_binding = publish_json(
        OUTPUT_ROOT / "execution_manifest.json", execution_manifest
    )
    completion = {
        "schema_version": SCHEMA,
        "status": "completed",
        "branch": BRANCH,
        "promoted": promoted,
        "promoted_endpoint": "sole_equalblend_endpoint" if promoted else None,
        "decision": decision_binding,
        "execution_manifest": execution_binding,
        "validation_member_payloads_opened": False,
        "validation_results_read": False,
    }
    completion_binding = publish_json(OUTPUT_ROOT / "COMPLETED.json", completion)
    helper.seal_tree(OUTPUT_ROOT)
    sealed_output_tree = verify_exact_sealed_output_tree(
        [
            *result_bindings,
            decision_binding,
            execution_binding,
            completion_binding,
        ]
    )
    return {
        "status": "completed",
        "promoted": promoted,
        "promoted_endpoint": completion["promoted_endpoint"],
        "decision": decision_binding,
        "execution_manifest": execution_binding,
        "completion": completion_binding,
        "sealed_output_tree": sealed_output_tree,
        "output_root": str(OUTPUT_ROOT),
        "validation_member_payloads_opened": False,
        "validation_results_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "static-audit", "formal"), required=True
    )
    parser.add_argument("--expected-preregistration-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    if args.mode == "contract":
        if args.expected_preregistration_sha256 is not None:
            raise ValueError("contract mode does not accept preregistration SHA")
        result = preregistration_payload()
    elif args.mode == "static-audit":
        if args.expected_preregistration_sha256 is not None:
            raise ValueError("static-audit mode does not accept preregistration SHA")
        result = static_audit()
    else:
        result = formal(args.expected_preregistration_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

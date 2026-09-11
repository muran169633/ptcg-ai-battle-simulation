#!/usr/bin/env python3
"""One-shot train-only endpoint gate for the raw-U468 balanced actor6 sweep.

This launcher deliberately does *not* inspect held-out result directories and
never opens a zip member outside ``train/``.  It evaluates raw U468 and both
P1/P2 endpoints on every evaluable train row of the three frozen archives,
then selects at most the smallest fully eligible step.  It cannot run any
held-out specialist, broad, Gold, package, upload, or submission action.

The training design must embed the exact object printed by ``--mode contract``
before formal training starts.  The completed training manifest binds that
design and the generated endpoint hashes.  A formal invocation consumes a
single O_EXCL attempt marker before creating any evaluation result.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import sys
import time
import zipfile
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import orjson
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate_policy_bc as evaluator  # noqa: E402
import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-raw-balanced-train-only-endpoint-gate-v1"
CONTRACT_SCHEMA = "ptcg-u468-raw-balanced-train-only-gate-contract-v1"
BRANCH = "ppo_u468_raw_trainhard_actor6_balanced64_96_96_p1p2_design202608111"
SCRIPT = ROOT / "tools/run_u468_raw_balanced_train_only_endpoint_gate.py"
TRAINER = ROOT / "tools/run_u468_raw_trainhard_actor6_balanced_mix_sweep.py"
DESIGN = ROOT / f"artifacts/{BRANCH}.design_preregistration.json"
TRAINING_ROOT = ROOT / f"artifacts/{BRANCH}"
TRAINING_MANIFEST = TRAINING_ROOT / "training_manifest.json"
TRAINING_SEAL = TRAINING_ROOT / "COMPLETED.json"
OUTPUT_ROOT = ROOT / f"artifacts/{BRANCH}.train_only_endpoint_gate"
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-train-only-gate-attempt.json"

PARENT = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
PARENT_FILE_SHA256 = (
    "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
)
PARENT_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
PROFILE = ROOT / "artifacts/u468_raw_full_train_margin_profile_v3_20260802.json"
PROFILE_SHA256 = (
    "17af98e175167398a80cd2e1f8930c3d1d85cfe6ceea3143dfe2ba44dc89f1f9"
)

DATASETS = {
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
    "pokemonfan": ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "core5": ROOT
    / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
}
DATA_SHA256 = {
    "flg": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    "pokemonfan": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}
EXPECTED_TRAIN_SHAPE = {
    "flg": {
        "rows": 9443,
        "nonempty_rows": 9426,
        "context34_rows": 42,
        "context34_nonempty_rows": 42,
        "profile_set_exact_correct": 7239,
        "profile_ordered_exact_correct": 7095,
        "train_member_count": 1,
    },
    "pokemonfan": {
        "rows": 9487,
        "nonempty_rows": 9450,
        "context34_rows": 38,
        "context34_nonempty_rows": 38,
        "profile_set_exact_correct": 8282,
        "profile_ordered_exact_correct": 8203,
        "train_member_count": 1,
    },
    "core5": {
        "rows": 5120,
        "nonempty_rows": 5106,
        "context34_rows": 20,
        "context34_nonempty_rows": 20,
        "profile_set_exact_correct": 4073,
        "profile_ordered_exact_correct": 4025,
        "train_member_count": 20,
    },
}

DEPENDENCY_SHA256 = {
    ROOT / "tools/evaluate_policy_bc.py": (
        "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4"
    ),
    ROOT / "tools/train_ppo.py": (
        "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
    ),
    ROOT / "tools/train_bc_orbit.py": (
        "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
    ),
}

EXPECTED_ENDPOINT_FILENAMES = {
    1: "special-bc-rawu468-balanced-actor6-0001.pt",
    2: "special-bc-rawu468-balanced-actor6-0002.pt",
}
ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)

BATCH_SIZE = 256
DEVICE = "cuda"
SKILL_ORDER_CONTEXT = 34
STRUCTURAL_METRICS = (
    "rows",
    "nonempty_rows",
    "by_context.34.rows",
    "by_context.34.nonempty_rows",
)
IMMUTABLE_METRICS = (
    "count_correct",
    "value_correct",
)
NON_REGRESSION_METRICS = (
    "set_exact_correct",
    "hybrid_order_exact_correct",
    "ordered_exact_correct",
    "top1_correct",
    "by_context.34.hybrid_order_exact_correct",
    "by_context.34.ordered_exact_correct",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
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


def load_hash_bound_profile_json(payload: bytes) -> dict[str, Any]:
    """Parse the frozen margin profile, whose JSON contains +/-Infinity.

    The profiler intentionally serializes infinite margins for decisions with
    no legal competitor.  This exception is limited to the one byte-for-byte
    hash-bound profile; NaN remains forbidden.  All control-plane JSON keeps
    using ``load_json_bytes`` above and therefore rejects every non-finite
    constant.
    """

    label = "hash-bound raw U468 train-only profile"
    observed_sha256 = sha256_bytes(payload)
    if observed_sha256 != PROFILE_SHA256:
        raise ValueError(
            f"{label} hash drift: expected {PROFILE_SHA256}, "
            f"observed {observed_sha256}"
        )

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def profile_constant(value: str) -> float:
        if value == "Infinity":
            return float("inf")
        if value == "-Infinity":
            return float("-inf")
        raise ValueError(f"{label} contains forbidden JSON constant {value}")

    parsed = json.loads(
        payload,
        object_pairs_hook=pairs,
        parse_constant=profile_constant,
    )
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return parsed


def read_regular_file(
    path: Path,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
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
        or (visible.st_dev, visible.st_ino, visible.st_size) != identity
    ):
        raise RuntimeError(f"{label} changed during read")
    payload = b"".join(chunks)
    observed_sha256 = sha256_bytes(payload)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            f"{label} hash drift: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed_sha256,
        "bytes": len(payload),
        "mode": oct(after.st_mode & 0o777),
        "inode": after.st_ino,
        "device": after.st_dev,
        "nlink": after.st_nlink,
    }


def publish_o_excl(path: Path, payload: bytes, mode: int = 0o600) -> dict[str, Any]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise OSError(f"short write while publishing {path}")
            written += count
        os.fsync(fd)
        before = os.fstat(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        reloaded = b"".join(chunks)
        after = os.fstat(fd)
        visible = os.lstat(path)
        identity = (after.st_dev, after.st_ino, after.st_size)
        if (
            payload != reloaded
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size) != identity
            or stat.S_ISLNK(visible.st_mode)
            or (visible.st_dev, visible.st_ino, visible.st_size) != identity
        ):
            raise RuntimeError(f"unsafe O_EXCL publication: {path}")
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_bytes(reloaded),
            "bytes": len(reloaded),
            "mode": oct(after.st_mode & 0o777),
            "inode": after.st_ino,
            "device": after.st_dev,
            "nlink": after.st_nlink,
        }
    finally:
        os.close(fd)


def model_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"non-tensor model state entry: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_runtime() -> None:
    if Path.cwd().resolve(strict=True) != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve(strict=True) != EXPECTED_PYTHON.resolve(strict=True):
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B flags")


def expected_design_contract() -> dict[str, Any]:
    script_sha256 = sha256_bytes(read_regular_file(SCRIPT, "gate launcher")[0])
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": "frozen_before_formal_training",
        "evaluator": {
            "path": str(SCRIPT.relative_to(ROOT)),
            "sha256": script_sha256,
            "python": str(EXPECTED_PYTHON),
            "flags": ["-I", "-B"],
        },
        "scope": {
            "data_split": "train",
            "all_train_rows": True,
            "validation_members_opened": False,
            "validation_results_read": False,
            "training": False,
            "specialist_valid": False,
            "broad": False,
            "gold": False,
            "package": False,
            "upload": False,
            "submission": False,
            "network": False,
        },
        "models": {
            "parent": "raw_full_u468",
            "candidate_steps": [1, 2],
            "run_parent_and_both_candidates_before_decision": True,
            "candidate_mutable_parameters_exact": list(ACTOR6_NAMES),
        },
        "panels": ["flg", "pokemonfan", "core5"],
        "evaluation": {
            "split": "train",
            "split_mode": "archive_member_prefix_and_row_field_both_train",
            "prediction_order": "policy_greedy",
            "batch_size": BATCH_SIZE,
            "workers": 0,
            "max_rows": None,
            "device": DEVICE,
            "archive_open_rule": "open_every_train_jsonl_member_once_and_no_other_member",
        },
        "gate": {
            "structural_metrics_exact_parent": list(STRUCTURAL_METRICS),
            "immutable_metrics_exact_parent": list(IMMUTABLE_METRICS),
            "non_regression_metrics": list(NON_REGRESSION_METRICS),
            "pokemonfan": "every_non_regression_metric_gte_parent",
            "core5": "every_non_regression_metric_gte_parent",
            "flg": (
                "every_non_regression_metric_gte_parent_and_at_least_one_"
                "non_regression_metric_improves_by_one_or_more"
            ),
        },
        "selection": {
            "eligible_endpoint_requires_all_three_panels": True,
            "select": "smallest_step_among_fully_eligible_endpoints",
            "maximum_selected_endpoints": 1,
            "none_eligible": "close_branch_before_any_valid_evaluation",
            "selected_only": "may_build_separate_six_panel_valid_preregistration",
        },
        "publication": {
            "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
            "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
            "formal_attempts_authorized": 1,
            "retry_authorized": False,
            "all_outputs": "O_EXCL",
        },
    }


def verify_fixed_inputs() -> tuple[dict[str, Any], dict[str, bytes]]:
    evidence: dict[str, Any] = {}
    payloads: dict[str, bytes] = {}
    payloads["parent"], evidence["parent"] = read_regular_file(
        PARENT, "raw U468 parent", PARENT_FILE_SHA256
    )
    payloads["profile"], evidence["profile"] = read_regular_file(
        PROFILE, "raw U468 train-only profile", PROFILE_SHA256
    )
    evidence["dependencies"] = {}
    for path, expected_hash in DEPENDENCY_SHA256.items():
        _, item = read_regular_file(path, str(path.relative_to(ROOT)), expected_hash)
        evidence["dependencies"][str(path.relative_to(ROOT))] = item
    evidence["datasets"] = {}
    for name, path in DATASETS.items():
        payload, item = read_regular_file(path, f"{name} archive", DATA_SHA256[name])
        payloads[f"dataset:{name}"] = payload
        evidence["datasets"][name] = item
    return evidence, payloads


def verify_profile(payload: bytes) -> dict[str, Any]:
    profile = load_hash_bound_profile_json(payload)
    if (
        profile.get("schema_version")
        != "ptcg-u468-raw-full-train-margin-profile-v3"
        or profile.get("status") != "completed_train_only"
        or profile.get("split") != "train"
        or profile.get("validation_opened") is not False
    ):
        raise ValueError("raw U468 train-only profile contract mismatch")
    base = profile.get("base")
    if not isinstance(base, dict) or (
        base.get("checkpoint_file_sha256") != PARENT_FILE_SHA256
        or base.get("model_state_sha256") != PARENT_MODEL_STATE_SHA256
    ):
        raise ValueError("train-only profile parent binding mismatch")
    profiles = profile.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(DATASETS):
        raise ValueError("train-only profile panel set mismatch")
    summary: dict[str, Any] = {}
    for name in DATASETS:
        panel = profiles[name]
        expected = EXPECTED_TRAIN_SHAPE[name]
        if (
            panel.get("archive_sha256") != DATA_SHA256[name]
            or panel.get("rows") != expected["rows"]
            or panel.get("set_correct") != expected["profile_set_exact_correct"]
            or panel.get("ordered_correct")
            != expected["profile_ordered_exact_correct"]
            or panel.get("non_train_members_opened") is not False
        ):
            raise ValueError(f"{name} train-only profile metrics/binding mismatch")
        opened = panel.get("opened_members")
        if (
            not isinstance(opened, list)
            or len(opened) != expected["train_member_count"]
            or any(
                not isinstance(member, str)
                or not member.startswith("train/")
                or not member.endswith(".jsonl")
                for member in opened
            )
        ):
            raise ValueError(f"{name} profile member audit mismatch")
        summary[name] = {
            "rows": panel["rows"],
            "set_exact_correct": panel["set_correct"],
            "ordered_exact_correct": panel["ordered_correct"],
            "opened_members": opened,
            "validation_members_opened": False,
        }
    return summary


def verify_design(
    payload: bytes,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed_sha256 = sha256_bytes(payload)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("training manifest design SHA-256 mismatch")
    design = load_json_bytes(payload, "training design")
    if design.get("branch") != BRANCH:
        raise ValueError("training design branch mismatch")
    executor = design.get("executor")
    if (
        not isinstance(executor, dict)
        or manifest_record_path(executor, "training design executor") != TRAINER
        or executor.get("python") != str(EXPECTED_PYTHON)
        or executor.get("flags") != ["-I", "-B"]
    ):
        raise ValueError("training design executor binding mismatch")
    _, trainer_evidence = read_regular_file(
        TRAINER,
        "training design executor",
        str(executor.get("sha256", "")),
    )
    contract = design.get("train_only_endpoint_gate")
    expected_contract = expected_design_contract()
    if contract != expected_contract:
        raise ValueError("training design does not contain the exact frozen gate contract")
    return design, {
        "path": str(DESIGN.relative_to(ROOT)),
        "sha256": observed_sha256,
        "executor": trainer_evidence,
        "frozen_gate_contract_sha256": sha256_bytes(
            canonical_json_bytes(expected_contract)
        ),
    }


def checkpoint_from_bytes(payload: bytes, label: str) -> dict[str, Any]:
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"{label} checkpoint root must be a dictionary")
    state = value.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError(f"{label} checkpoint is missing model_state_dict")
    return value


def verify_parent_checkpoint(payload: bytes) -> dict[str, Any]:
    checkpoint = checkpoint_from_bytes(payload, "raw U468 parent")
    if (
        checkpoint.get("update") != 468
        or evaluator.checkpoint_kind(checkpoint) != "ppo"
        or not isinstance(checkpoint.get("model_config"), dict)
    ):
        raise ValueError("parent checkpoint is not the expected PPO update U468")
    observed = model_state_sha256(checkpoint["model_state_dict"])
    if observed != PARENT_MODEL_STATE_SHA256:
        raise ValueError("raw U468 parent model-state hash mismatch")
    return checkpoint


def verify_actor6_only(
    parent_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if set(candidate_state) != set(parent_state):
        raise ValueError(f"{label} model-state key set differs from parent")
    changed: list[str] = []
    for name in sorted(parent_state):
        parent_tensor = parent_state[name]
        candidate_tensor = candidate_state[name]
        if not isinstance(parent_tensor, torch.Tensor) or not isinstance(
            candidate_tensor, torch.Tensor
        ):
            raise TypeError(f"{label} non-tensor model state entry {name}")
        if (
            parent_tensor.shape != candidate_tensor.shape
            or parent_tensor.dtype != candidate_tensor.dtype
        ):
            raise ValueError(f"{label} tensor schema differs at {name}")
        if not torch.equal(parent_tensor, candidate_tensor):
            changed.append(name)
    if set(changed) != set(ACTOR6_NAMES):
        raise ValueError(
            f"{label} changed tensor set is not exactly actor6: {changed}"
        )
    return {
        "changed_parameter_names": changed,
        "changed_parameter_count": len(changed),
        "allowed_parameter_names": list(ACTOR6_NAMES),
        "changed_parameter_set_exact_actor6": True,
        "all_six_actor_parameters_changed": True,
    }


def manifest_record_path(record: Mapping[str, Any], label: str) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} has no path")
    path = ROOT / raw
    try:
        relative = path.resolve(strict=True).relative_to(ROOT)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{label} path escapes/misses repository root") from error
    if str(relative) != raw:
        raise ValueError(f"{label} path is not canonical")
    return path


def verify_training_outputs(
    parent_checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bytes]]:
    manifest_payload, manifest_evidence = read_regular_file(
        TRAINING_MANIFEST, "training manifest"
    )
    manifest = load_json_bytes(manifest_payload, "training manifest")
    if (
        manifest.get("branch") != BRANCH
        or manifest.get("status") != "training_completed_all_endpoints_published"
        or manifest.get("parent_unchanged") is not True
        or manifest.get("validation_opened_during_training") is not False
    ):
        raise ValueError("completed training manifest contract mismatch")
    training = manifest.get("training")
    if not isinstance(training, dict) or (
        training.get("data_split") != "train"
        or training.get("validation_opened") is not False
        or training.get("endpoint_steps") != [1, 2]
        or training.get("trainable_parameter_names") != list(ACTOR6_NAMES)
        or training.get("count_head_frozen") is not True
        or training.get("value_head_frozen") is not True
    ):
        raise ValueError("training scope/endpoint contract mismatch")

    design_record = manifest.get("design")
    if not isinstance(design_record, dict):
        raise ValueError("training manifest is missing design binding")
    if manifest_record_path(design_record, "design binding") != DESIGN:
        raise ValueError("training manifest points to an unexpected design")
    design_payload, _ = read_regular_file(
        DESIGN,
        "training design",
        str(design_record.get("sha256", "")),
    )
    _, design_evidence = verify_design(
        design_payload,
        str(design_record.get("sha256", "")),
    )

    seal_payload, seal_evidence = read_regular_file(TRAINING_SEAL, "training seal")
    seal = load_json_bytes(seal_payload, "training seal")
    seal_manifest = seal.get("training_manifest")
    if (
        seal.get("branch") != BRANCH
        or seal.get("status") != "completed"
        or seal.get("endpoint_steps") != [1, 2]
        or not isinstance(seal_manifest, dict)
        or manifest_record_path(seal_manifest, "sealed training manifest")
        != TRAINING_MANIFEST
        or seal_manifest.get("sha256") != manifest_evidence["sha256"]
    ):
        raise ValueError("training completion seal mismatch")

    manifest_inputs = manifest.get("inputs")
    if not isinstance(manifest_inputs, dict):
        raise ValueError("training manifest inputs are missing")
    for name in DATASETS:
        record = manifest_inputs.get(name)
        if (
            not isinstance(record, dict)
            or manifest_record_path(record, f"{name} training input") != DATASETS[name]
            or record.get("sha256") != DATA_SHA256[name]
        ):
            raise ValueError(f"training manifest {name} input mismatch")

    endpoints_raw = manifest.get("endpoints")
    if not isinstance(endpoints_raw, list) or len(endpoints_raw) != 2:
        raise ValueError("training manifest must contain exactly P1 and P2")
    endpoints: list[dict[str, Any]] = []
    checkpoint_payloads: dict[str, bytes] = {}
    parent_state = parent_checkpoint["model_state_dict"]
    for record, step in zip(endpoints_raw, (1, 2), strict=True):
        if not isinstance(record, dict) or record.get("step") != step:
            raise ValueError("training manifest endpoint order mismatch")
        path = manifest_record_path(record, f"P{step} endpoint")
        if (
            path.parent != TRAINING_ROOT
            or path.name != EXPECTED_ENDPOINT_FILENAMES[step]
        ):
            raise ValueError(f"P{step} endpoint path mismatch")
        payload, evidence = read_regular_file(
            path,
            f"P{step} endpoint",
            str(record.get("sha256", "")),
        )
        checkpoint = checkpoint_from_bytes(payload, f"P{step}")
        if (
            evaluator.checkpoint_kind(checkpoint) != "ppo"
            or checkpoint.get("feature_version")
            != parent_checkpoint.get("feature_version")
            or checkpoint.get("bc_feature_version")
            != parent_checkpoint.get("bc_feature_version")
            or checkpoint.get("model_config")
            != parent_checkpoint.get("model_config")
        ):
            raise ValueError(f"P{step} checkpoint inference schema differs from parent")
        observed_model_hash = model_state_sha256(checkpoint["model_state_dict"])
        if observed_model_hash != record.get("model_state_sha256"):
            raise ValueError(f"P{step} model-state hash mismatch")
        manifest_changed = record.get("changed_parameter_names")
        if (
            not isinstance(manifest_changed, list)
            or set(manifest_changed) != set(ACTOR6_NAMES)
            or len(manifest_changed) != len(ACTOR6_NAMES)
        ):
            raise ValueError(f"P{step} manifest changed-parameter scope mismatch")
        actor_audit = verify_actor6_only(parent_state, checkpoint["model_state_dict"], f"P{step}")
        if sorted(manifest_changed) != sorted(actor_audit["changed_parameter_names"]):
            raise ValueError(f"P{step} changed-parameter evidence mismatch")
        name = f"p{step:03d}"
        checkpoint_payloads[name] = payload
        endpoints.append(
            {
                "name": name,
                "step": step,
                "path": str(path.relative_to(ROOT)),
                "sha256": evidence["sha256"],
                "model_state_sha256": observed_model_hash,
                "actor6_audit": actor_audit,
            }
        )
    return (
        {
            "manifest": manifest_evidence,
            "design": design_evidence,
            "completion_seal": seal_evidence,
        },
        endpoints,
        checkpoint_payloads,
    )


class StrictTrainDataset(IterableDataset):
    """Evaluator-compatible dataset that raises instead of skipping any row."""

    def __init__(
        self,
        archive_payload: bytes,
        hash_size: int,
        max_state_entities: int,
    ) -> None:
        super().__init__()
        self.archive_payload = archive_payload
        self.hash_size = hash_size
        self.max_state_entities = max_state_entities
        self.opened_members: list[str] = []
        self.inventory_train_members: list[str] = []
        self.raw_rows = 0
        self.emitted_rows = 0
        self.validation_members_opened = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        if worker is not None:
            raise RuntimeError("strict train-only gate requires workers=0")
        with zipfile.ZipFile(io.BytesIO(self.archive_payload)) as archive:
            names = archive.namelist()
            members = sorted(
                name
                for name in names
                if name.startswith("train/") and name.endswith(".jsonl")
            )
            if not members:
                raise RuntimeError("archive has no train JSONL members")
            self.inventory_train_members = members
            for member in members:
                if not member.startswith("train/"):
                    self.validation_members_opened = True
                    raise RuntimeError(f"refusing non-train archive member {member}")
                self.opened_members.append(member)
                with archive.open(member) as handle:
                    for line_index, line in enumerate(handle, start=1):
                        self.raw_rows += 1
                        try:
                            row = orjson.loads(line)
                        except Exception as error:
                            raise ValueError(
                                f"invalid JSON in {member}:{line_index}"
                            ) from error
                        if not isinstance(row, dict) or row.get("split") != "train":
                            raise ValueError(
                                f"non-train row inside {member}:{line_index}"
                            )
                        raw_action = row.get("action")
                        if not isinstance(raw_action, list):
                            raise ValueError(
                                f"invalid action in {member}:{line_index}"
                            )
                        try:
                            expert_order = [int(index) for index in raw_action]
                        except (TypeError, ValueError) as error:
                            raise ValueError(
                                f"invalid action index in {member}:{line_index}"
                            ) from error
                        previous_max = bc.MAX_ACTION_COUNT
                        bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
                        try:
                            features = bc.featurize_row(
                                row,
                                self.hash_size,
                                self.max_state_entities,
                            )
                        finally:
                            bc.MAX_ACTION_COUNT = previous_max
                        options = (
                            ((row.get("observation") or {}).get("select") or {}).get(
                                "option"
                            )
                            or []
                        )
                        if (
                            features is None
                            or len(expert_order) > ppo.MAX_ACTION_COUNT
                            or len(set(expert_order)) != len(expert_order)
                            or any(
                                index < 0 or index >= len(options)
                                for index in expert_order
                            )
                        ):
                            raise ValueError(
                                f"non-evaluable train row in {member}:{line_index}"
                            )
                        features["expert_action_order"] = expert_order
                        self.emitted_rows += 1
                        yield features


def instantiate_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], str]:
    kind = evaluator.checkpoint_kind(checkpoint)
    model_config = evaluator.model_config_from_checkpoint(checkpoint, kind)
    if kind == "bc":
        model = ppo.instantiate_model_from_bc(checkpoint, device)
    else:
        model = evaluator.instantiate_ppo_checkpoint(checkpoint, model_config, device)
    model.eval()
    return model, model_config, kind


def evaluate_panel(
    model: nn.Module,
    model_config: dict[str, Any],
    archive_payload: bytes,
    panel: str,
    device: torch.device,
) -> dict[str, Any]:
    dataset = StrictTrainDataset(
        archive_payload=archive_payload,
        hash_size=model_config["hash_size"],
        max_state_entities=model_config["max_state_entities"],
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=partial(
            evaluator.collate_ordered,
            max_state_entities=model_config["max_state_entities"],
            entity_fields=model_config["entity_fields"],
            option_fields=model_config["option_fields"],
        ),
        pin_memory=True,
        persistent_workers=False,
    )
    metrics, seconds = evaluator.evaluate(
        model,
        loader,
        device,
        canonicalize_order=False,
        max_rows=None,
        progress_interval=100,
    )
    expected = EXPECTED_TRAIN_SHAPE[panel]
    context34 = metrics.get("by_context", {}).get("34")
    if not isinstance(context34, dict):
        raise ValueError(f"{panel} metrics are missing context 34")
    if (
        dataset.validation_members_opened
        or dataset.opened_members != dataset.inventory_train_members
        or len(dataset.opened_members) != expected["train_member_count"]
        or dataset.raw_rows != expected["rows"]
        or dataset.emitted_rows != expected["rows"]
        or metrics.get("rows") != expected["rows"]
        or metrics.get("nonempty_rows") != expected["nonempty_rows"]
        or context34.get("rows") != expected["context34_rows"]
        or context34.get("nonempty_rows") != expected["context34_nonempty_rows"]
    ):
        raise ValueError(f"{panel} full-train structural/member audit failed")
    return {
        "panel": panel,
        "split": "train",
        "split_mode": "archive_member_prefix_and_row_field_both_train",
        "prediction_order": "policy_greedy",
        "batch_size": BATCH_SIZE,
        "workers": 0,
        "max_rows": None,
        "archive": str(DATASETS[panel].relative_to(ROOT)),
        "archive_sha256": DATA_SHA256[panel],
        "archive_inventory_train_members": dataset.inventory_train_members,
        "opened_members": dataset.opened_members,
        "validation_members_opened": False,
        "raw_train_rows": dataset.raw_rows,
        "emitted_train_rows": dataset.emitted_rows,
        "metrics": metrics,
        "seconds": seconds,
        "rows_per_second": metrics["rows"] / max(seconds, 1e-9),
    }


def metric_value(metrics: Mapping[str, Any], path: str) -> int:
    value: Any = metrics
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"missing metric path {path}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"metric {path} must be an integer count")
    return value


def gate_panel(
    panel: str,
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    passed = True
    for metric in STRUCTURAL_METRICS:
        parent_value = metric_value(parent, metric)
        actual = metric_value(candidate, metric)
        item_pass = actual == parent_value
        passed &= item_pass
        checks[f"structural_exact:{metric}"] = {
            "parent": parent_value,
            "actual": actual,
            "delta": actual - parent_value,
            "pass": item_pass,
        }
    for metric in IMMUTABLE_METRICS:
        parent_value = metric_value(parent, metric)
        actual = metric_value(candidate, metric)
        item_pass = actual == parent_value
        passed &= item_pass
        checks[f"immutable_exact:{metric}"] = {
            "parent": parent_value,
            "actual": actual,
            "delta": actual - parent_value,
            "pass": item_pass,
        }
    improvements: list[str] = []
    for metric in NON_REGRESSION_METRICS:
        parent_value = metric_value(parent, metric)
        actual = metric_value(candidate, metric)
        item_pass = actual >= parent_value
        passed &= item_pass
        delta = actual - parent_value
        if delta >= 1:
            improvements.append(metric)
        checks[f"non_regression:{metric}"] = {
            "parent": parent_value,
            "actual": actual,
            "delta": delta,
            "pass": item_pass,
        }
    improvement_required = panel == "flg"
    improvement_pass = not improvement_required or bool(improvements)
    passed &= improvement_pass
    checks["flg_at_least_one_quality_count_plus_one"] = {
        "required": improvement_required,
        "improved_metrics": improvements,
        "pass": improvement_pass,
    }
    return {"panel": panel, "pass": passed, "checks": checks}


def seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"refusing to seal symlink {path}")
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def static_audit() -> dict[str, Any]:
    evidence, payloads = verify_fixed_inputs()
    profile_summary = verify_profile(payloads["profile"])
    parent_checkpoint = verify_parent_checkpoint(payloads["parent"])
    design_status: dict[str, Any]
    if DESIGN.exists() or DESIGN.is_symlink():
        design_payload, design_file = read_regular_file(DESIGN, "training design")
        _, design_binding = verify_design(design_payload, design_file["sha256"])
        design_status = {"present": True, "binding": design_binding}
    else:
        design_status = {
            "present": False,
            "required_before_formal_training": True,
        }
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_passed",
        "formal_evaluation_executed": False,
        "validation_members_opened": False,
        "validation_results_read": False,
        "fixed_inputs": evidence,
        "profile_summary": profile_summary,
        "parent_update": parent_checkpoint.get("update"),
        "parent_model_state_sha256": model_state_sha256(
            parent_checkpoint["model_state_dict"]
        ),
        "design": design_status,
        "expected_design_contract": expected_design_contract(),
        "formal_absence_gate": {
            "attempt_marker_absent": not ATTEMPT_MARKER.exists(),
            "output_root_absent": not OUTPUT_ROOT.exists(),
        },
    }


def formal() -> dict[str, Any]:
    if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
        raise FileExistsError(ATTEMPT_MARKER)
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)

    evidence, payloads = verify_fixed_inputs()
    profile_summary = verify_profile(payloads["profile"])
    parent_checkpoint = verify_parent_checkpoint(payloads["parent"])
    training_evidence, endpoints, endpoint_payloads = verify_training_outputs(
        parent_checkpoint
    )

    attempt_payload = canonical_json_bytes(
        {
            "schema_version": SCHEMA,
            "status": "formal_attempt_consumed",
            "created_at_utc": utc_now(),
            "branch": BRANCH,
            "launcher_sha256": expected_design_contract()["evaluator"]["sha256"],
            "training_manifest_sha256": training_evidence["manifest"]["sha256"],
            "training_completion_seal_sha256": training_evidence[
                "completion_seal"
            ]["sha256"],
            "formal_attempts_authorized": 1,
            "retry_authorized": False,
            "scope": expected_design_contract()["scope"],
        }
    )
    marker_evidence = publish_o_excl(ATTEMPT_MARKER, attempt_payload)
    os.chmod(ATTEMPT_MARKER, 0o444)
    os.mkdir(OUTPUT_ROOT, mode=0o700)

    started = time.time()
    result_bindings: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, dict[str, Any]]] = {}
    model_specs = [
        {
            "name": "parent",
            "step": 0,
            "path": str(PARENT.relative_to(ROOT)),
            "sha256": PARENT_FILE_SHA256,
            "model_state_sha256": PARENT_MODEL_STATE_SHA256,
            "payload": payloads["parent"],
        }
    ] + [
        endpoint | {"payload": endpoint_payloads[endpoint["name"]]}
        for endpoint in endpoints
    ]

    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("formal train-only gate requires CUDA")
    device = torch.device(DEVICE)
    for spec in model_specs:
        name = spec["name"]
        checkpoint = checkpoint_from_bytes(spec.pop("payload"), name)
        observed_model_sha256 = model_state_sha256(checkpoint["model_state_dict"])
        if observed_model_sha256 != spec["model_state_sha256"]:
            raise ValueError(f"{name} effective model hash mismatch")
        model, model_config, kind = instantiate_checkpoint(checkpoint, device)
        model_results: dict[str, dict[str, Any]] = {}
        model_dir = OUTPUT_ROOT / name
        os.mkdir(model_dir, mode=0o700)
        for panel in DATASETS:
            result = {
                "schema_version": SCHEMA,
                "status": "train_only_evaluation_completed",
                "model": {key: value for key, value in spec.items()},
                "checkpoint_kind": kind,
                "checkpoint_update": checkpoint.get("update"),
                "feature_version": checkpoint.get("feature_version"),
                "device": str(device),
                "validation_results_read": False,
                **evaluate_panel(
                    model,
                    model_config,
                    payloads[f"dataset:{panel}"],
                    panel,
                    device,
                ),
            }
            if name == "parent":
                expected = EXPECTED_TRAIN_SHAPE[panel]
                metrics = result["metrics"]
                if (
                    metrics["set_exact_correct"]
                    != expected["profile_set_exact_correct"]
                    or metrics["ordered_exact_correct"]
                    != expected["profile_ordered_exact_correct"]
                ):
                    raise ValueError(
                        f"{panel} parent metrics disagree with the frozen train profile"
                    )
            model_results[panel] = result
            path = model_dir / f"{panel}.json"
            result_bindings.append(
                publish_o_excl(path, canonical_json_bytes(result))
            )
        evaluations[name] = model_results
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    endpoint_decisions: list[dict[str, Any]] = []
    eligible_steps: list[int] = []
    parent_metrics = {
        panel: evaluations["parent"][panel]["metrics"] for panel in DATASETS
    }
    for endpoint in endpoints:
        name = endpoint["name"]
        panel_gates = {
            panel: gate_panel(
                panel,
                parent_metrics[panel],
                evaluations[name][panel]["metrics"],
            )
            for panel in DATASETS
        }
        eligible = all(item["pass"] for item in panel_gates.values())
        if eligible:
            eligible_steps.append(endpoint["step"])
        endpoint_decisions.append(
            {
                "endpoint": name,
                "step": endpoint["step"],
                "eligible": eligible,
                "panel_gates": panel_gates,
            }
        )
    selected_step = min(eligible_steps) if eligible_steps else None
    selected_endpoint = f"p{selected_step:03d}" if selected_step is not None else None
    decision = {
        "schema_version": SCHEMA,
        "status": (
            "selected_one_endpoint_for_separate_six_panel_preregistration"
            if selected_step is not None
            else "no_endpoint_eligible_branch_closed_before_valid"
        ),
        "created_at_utc": utc_now(),
        "branch": BRANCH,
        "gate_contract": expected_design_contract(),
        "parent_profile_evidence": profile_summary,
        "all_parent_and_candidate_train_evaluations_completed_before_decision": True,
        "endpoint_decisions": endpoint_decisions,
        "eligible_steps": eligible_steps,
        "selected_step": selected_step,
        "selected_endpoint": selected_endpoint,
        "selected_endpoint_count": int(selected_endpoint is not None),
        "maximum_selected_endpoints": 1,
        "validation_members_opened": False,
        "validation_results_read": False,
        "downstream": {
            "valid_executed_by_this_tool": False,
            "broad_executed_by_this_tool": False,
            "gold_executed_by_this_tool": False,
            "network_used": False,
            "package_upload_submission_authorized": False,
        },
    }
    decision_binding = publish_o_excl(
        OUTPUT_ROOT / "gate_decision.json",
        canonical_json_bytes(decision),
    )
    result_bindings.append(decision_binding)
    execution_manifest = {
        "schema_version": SCHEMA,
        "status": "all_nine_train_only_evaluations_and_decision_completed",
        "created_at_utc": utc_now(),
        "branch": BRANCH,
        "runtime": {
            "python": sys.executable,
            "isolated": sys.flags.isolated,
            "dont_write_bytecode": sys.flags.dont_write_bytecode,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "fixed_inputs": evidence,
        "training_evidence": training_evidence,
        "endpoints": endpoints,
        "attempt_marker": marker_evidence,
        "ordered_result_bindings": result_bindings,
        "evaluation_count": len(result_bindings) - 1,
        "expected_evaluation_count": 9,
        "decision": decision_binding,
        "seconds": time.time() - started,
        "validation_members_opened": False,
        "validation_results_read": False,
    }
    if execution_manifest["evaluation_count"] != 9:
        raise RuntimeError("train-only evaluation count drift")
    manifest_binding = publish_o_excl(
        OUTPUT_ROOT / "execution_manifest.json",
        canonical_json_bytes(execution_manifest),
    )
    completion = {
        "schema_version": SCHEMA,
        "status": "completed",
        "branch": BRANCH,
        "selected_endpoint": selected_endpoint,
        "selected_step": selected_step,
        "validation_members_opened": False,
        "validation_results_read": False,
        "execution_manifest": manifest_binding,
        "decision": decision_binding,
    }
    completion_binding = publish_o_excl(
        OUTPUT_ROOT / "COMPLETED.json",
        canonical_json_bytes(completion),
    )
    seal_tree(OUTPUT_ROOT)
    return {
        "status": completion["status"],
        "selected_endpoint": selected_endpoint,
        "selected_step": selected_step,
        "decision": decision_binding,
        "execution_manifest": manifest_binding,
        "completion": completion_binding,
        "output_root": str(OUTPUT_ROOT),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("contract", "static-audit", "formal"),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    if args.mode == "contract":
        result = expected_design_contract()
    elif args.mode == "static-audit":
        result = static_audit()
    else:
        result = formal()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

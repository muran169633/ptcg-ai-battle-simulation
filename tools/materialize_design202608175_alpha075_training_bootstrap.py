#!/usr/bin/env python3
"""Materialize the promoted alpha0.75 policy as a fresh-optimizer training seed."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SOURCE = ROOT / "artifacts/design202608170_parent_plus_u476_actor_delta/parent-plus-u476-actor-alpha075.pt"
SOURCE_SHA256 = "e01d9161245e3559f4ce21a3f14f6e09a1a4ba2b84c0f2ba1a59f15eda7c85eb"
DECISION = ROOT / "artifacts/design202608174_iteration_decision.json"
DECISION_SHA256 = "b62aacccca3da1c45285ec50f3407f4feadb71614bbfb96e3762dff91f216db1"
OUTPUT_ROOT = ROOT / "artifacts/design202608175_alpha075_fresh_training_bootstrap"
OUTPUT = OUTPUT_ROOT / "alpha075-fresh-bootstrap-u476.pt"
MANIFEST = OUTPUT_ROOT / "manifest.json"
REQUIRED_KEYS = (
    "feature_version", "bc_feature_version", "config", "model_config",
    "learner_deck_hash", "reward", "action_distribution", "model_state_dict",
    "update",
)
FORBIDDEN_STATE_KEYS = (
    "optimizer_state_dict", "bc_replay_optimizer_state_dict",
    "opponent_quota_state", "ppo_objective_state",
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape)).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes() + b"\0")
    return digest.hexdigest()


def publish(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_regular(path: Path, expected_sha: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"not a regular frozen input: {path}")
    if file_sha(path) != expected_sha:
        raise RuntimeError(f"frozen input digest mismatch: {path}")


def main() -> int:
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("materializer requires repository cwd and my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("materializer requires Python -I -B")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)
    validate_regular(SOURCE, SOURCE_SHA256)
    validate_regular(DECISION, DECISION_SHA256)
    decision = json.loads(DECISION.read_text())
    promoted = decision.get("promoted_candidate", {})
    if (
        decision.get("status") != "completed_local_promotion"
        or promoted.get("checkpoint_sha256") != SOURCE_SHA256
        or decision.get("best_record_update", {}).get("local_incumbent_changed") is not True
    ):
        raise RuntimeError("decision does not authorize this exact local training bootstrap")
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if source.get("evaluation_only") is not True or source.get("resume_forbidden") is not True:
        raise RuntimeError("source is not the expected evaluation-only endpoint")
    if any(key not in source for key in REQUIRED_KEYS):
        raise RuntimeError("source is missing required checkpoint fields")
    if int(source["update"]) != 476:
        raise RuntimeError("source update mismatch")
    source_state_hash = state_sha(source["model_state_dict"])
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {key: copy.deepcopy(source[key]) for key in REQUIRED_KEYS}
    payload["config"] = copy.deepcopy(payload["config"])
    payload["config"]["trainable_scope"] = "heads"
    payload["fresh_training_bootstrap"] = {
        "schema_version": "ptcg-design202608175-alpha075-fresh-training-bootstrap-v1",
        "created_at_utc": created_at,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "source_model_state_sha256": source_state_hash,
        "promotion_decision": str(DECISION.relative_to(ROOT)),
        "promotion_decision_sha256": DECISION_SHA256,
        "model_weights_bit_exact_source": True,
        "optimizer_state_source": "absent_fresh_required",
        "quota_state_source": "absent_fresh_required",
    }
    if "evaluation_only" in payload or "resume_forbidden" in payload:
        raise RuntimeError("training bootstrap retained evaluation-only flags")
    if any(key in payload for key in FORBIDDEN_STATE_KEYS):
        raise RuntimeError("training bootstrap retained optimizer or quota state")
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    check = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if state_sha(check["model_state_dict"]) != source_state_hash:
        raise RuntimeError("serialized bootstrap changed model weights")
    if any(key in check for key in (*FORBIDDEN_STATE_KEYS, "evaluation_only", "resume_forbidden")):
        raise RuntimeError("serialized bootstrap retained forbidden state")
    os.mkdir(OUTPUT_ROOT, mode=0o700)
    publish(OUTPUT, raw)
    manifest = {
        "schema_version": "ptcg-design202608175-alpha075-fresh-training-bootstrap-manifest-v1",
        "created_at_utc": created_at,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "source_model_state_sha256": source_state_hash,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "output_model_state_sha256": state_sha(check["model_state_dict"]),
        "model_weights_bit_exact_source": True,
        "optimizer_states_present": False,
        "quota_state_present": False,
        "fresh_optimizer_required": True,
    }
    publish(MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

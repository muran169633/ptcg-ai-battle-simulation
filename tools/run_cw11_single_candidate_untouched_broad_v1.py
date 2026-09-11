#!/usr/bin/env python3
"""One-shot, hash-bound untouched broad evaluation for materialized CW11.

The default action is a read-only CPU preflight.  Formal CUDA evaluation is
possible only with ``--mode execute`` and the exact SHA-256 of this runner.
The runner is deliberately self-contained and binds the checkpoint, its
manifest/completion provenance, the materializer, and the formal gate.

CW11 consumed four specialist-valid guards during optimization.  The separate
specialist six-panel run is therefore dev consistency only and cannot be
promotion evidence.  These two broad archives are untouched by CW11; a broad
result is independent promotion evidence only if both panels and all 20 gates
pass in the single authorized attempt.

Scope: local behavior evaluation only.  This file does not train, run Gold,
access the network, package, upload, or submit anything.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
SCRIPT = ROOT / "tools/run_cw11_single_candidate_untouched_broad_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

CANDIDATE = ROOT / (
    "artifacts/ppo_u468_raw_actor6_metricguard_specialist_valid_"
    "cw11_materialized_v1_20260802/u468-cw11-formal-pass-eval-only.pt"
)
CANDIDATE_SHA256 = (
    "bea774c30cd3113d984ba8252324c330d293a8ce5042d7d17f357f8132e86775"
)
CANDIDATE_MODEL_STATE_SHA256 = (
    "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
)
EXPECTED_CHECKPOINT_UPDATE = 468
EXPECTED_CHECKPOINT_KEY_COUNT = 13
EXPECTED_MODEL_TENSOR_COUNT = 80
EXPECTED_NONACTOR_TENSOR_COUNT = 74
EXPECTED_RAW_NONACTOR_SHA256 = (
    "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
)
ACTOR_PARAMETER_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
MATERIALIZATION_AUDIT_KEY = "metricguard_specialist_valid_cw11_materialization"
MATERIALIZER_SCHEMA = (
    "ptcg-u468-raw-actor6-metricguard-specialist-valid-"
    "cw11-materializer-v1"
)
MATERIALIZATION_BRANCH = (
    "ppo_u468_raw_actor6_metricguard_specialist_valid_"
    "cw11_materialized_v1_20260802"
)
MATERIALIZATION_MANIFEST_SCHEMA = f"{MATERIALIZER_SCHEMA}-manifest"
MATERIALIZATION_COMPLETION_SCHEMA = f"{MATERIALIZER_SCHEMA}-completion"
EXPECTED_FORMAL_PF_WC_NET = {
    "set_exact": 6,
    "hybrid_order_exact": 6,
    "ordered_exact": 6,
    "top1_correct": 7,
}

MATERIALIZATION_MANIFEST = CANDIDATE.parent / "materialization_manifest.json"
MATERIALIZATION_MANIFEST_SHA256 = (
    "6356f08505b5777fdb30e4f8cfce8a776ca2918af540ebc324227678aeb02016"
)
MATERIALIZATION_COMPLETED = CANDIDATE.parent / "COMPLETED.json"
MATERIALIZATION_COMPLETED_SHA256 = (
    "786281007347ee40a0ac3dd0ad65202c26aed9d5901b3866720980f8cd77a646"
)
MATERIALIZER = ROOT / (
    "tools/materialize_u468_raw_actor6_metricguard_"
    "specialist_valid_cw11_v1.py"
)
MATERIALIZER_SHA256 = (
    "b7c356ab78085f44cbbdbfcf1a46691e126820dc9a6ca0aa2199312ba8a3bbc1"
)
FORMAL_GATE = ROOT / (
    "tools/run_u468_raw_actor6_metricguard_specialist_valid_"
    "cw11_fulltrain_gate_v1.py"
)
FORMAL_GATE_SHA256 = (
    "3ef943434bc2699b125a7bd907892c877d3645fe8bed92ba576ef746c91fe695"
)

EVALUATOR = ROOT / "tools/evaluate_policy_bc.py"
EVALUATOR_SHA256 = (
    "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4"
)
DEPENDENCIES: tuple[tuple[str, Path, str], ...] = (
    (
        "train_bc_orbit",
        ROOT / "tools/train_bc_orbit.py",
        "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
    ),
    (
        "train_ppo",
        ROOT / "tools/train_ppo.py",
        "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3",
    ),
)

OUTPUT_ROOT = ROOT / (
    "artifacts/cw11_4317_single_candidate_untouched_broad_v1_20260802"
)
ATTEMPT_MARKER = ROOT / (
    "artifacts/.ptcg-cw11-4317-single-candidate-"
    "untouched-broad-v1-20260802-attempt.json"
)
TERMINAL_FAILURE = ROOT / (
    "artifacts/.ptcg-cw11-4317-single-candidate-untouched-broad-v1-20260802-"
    "terminal-failure.json"
)
MANIFEST = OUTPUT_ROOT / "broad_execution_manifest.json"
DECISION = OUTPUT_ROOT / "broad_decision.json"
EXECUTION_PREREQUISITE_CONFIRMATION = (
    "cw11-dev-consistency-pass-and-broad-runner-independent-audit-go"
)

PROTOCOL = {
    "split": "valid",
    "split_mode": "archive",
    "split_seed": 20260723,
    "batch_size": 256,
    "workers": 8,
    "prediction_order_argument": "policy",
    "expected_prediction_order": "policy_greedy",
    "device": "cuda",
    "compact": True,
    "progress_interval": 0,
}

PANELS: tuple[dict[str, Any], ...] = (
    {
        "name": "old_retention",
        "data": ROOT / "data/bc_marnie_top50_rollforward_test25_hash80_seed20260921.zip",
        "data_sha256": (
            "a3b9d572bfc0a784b5b140d3dbe09314252c46dd0d9c1afa3423236cda54543c"
        ),
        "gates": {
            "rows": {"exact": 34204},
            "set": {"minimum": 28160},
            "hybrid": {"minimum": 27970},
            "ordered": {"minimum": 27751},
            "value": {"exact": 25025},
            "count": {"minimum": 34007},
            "top1": {"minimum": 28384},
            "context34_rows": {"exact": 250},
            "context34_hybrid": {"minimum": 189},
            "context34_ordered": {"minimum": 189},
        },
    },
    {
        "name": "valid29",
        "data": ROOT / (
            "data/bc_marnie_top50plus_gold21_timeforward_train28_"
            "valid29_v2_20260731.zip"
        ),
        "data_sha256": (
            "95638471e0b842c6366231e95b2e98a5d806f612a0085ef39110ba4cb6f0ad0d"
        ),
        "gates": {
            "rows": {"exact": 159829},
            "set": {"minimum": 127040},
            "hybrid": {"minimum": 126618},
            "ordered": {"minimum": 125785},
            "value": {"exact": 118593},
            "count": {"minimum": 158197},
            "top1": {"minimum": 128523},
            "context34_rows": {"exact": 639},
            "context34_hybrid": {"minimum": 529},
            "context34_ordered": {"minimum": 529},
        },
    },
)
PANEL_CONTRACT_SHA256 = (
    "604bc38da9a9436540c5f2d1cfd2240a90b201d5461edb974d6c93a0bf8a326e"
)

METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "rows": ("metrics", "rows"),
    "set": ("metrics", "set_exact_correct"),
    "hybrid": ("metrics", "hybrid_order_exact_correct"),
    "ordered": ("metrics", "ordered_exact_correct"),
    "value": ("metrics", "value_correct"),
    "count": ("metrics", "count_correct"),
    "top1": ("metrics", "top1_correct"),
    "context34_rows": ("metrics", "by_context", "34", "rows"),
    "context34_hybrid": (
        "metrics",
        "by_context",
        "34",
        "hybrid_order_exact_correct",
    ),
    "context34_ordered": (
        "metrics",
        "by_context",
        "34",
        "ordered_exact_correct",
    ),
}

SCOPE = {
    "local_only": True,
    "network": False,
    "training": False,
    "specialist_behavior": False,
    "specialist_valid_consumed_for_optimization": True,
    "dev_consistency_only": True,
    "dev_consistency_promotion_evidence": False,
    "broad_behavior": True,
    "broad_untouched": True,
    "broad_independent_promotion_evidence_only_if_pass": True,
    "gold": False,
    "package": False,
    "upload": False,
    "submission": False,
}


class ProtocolError(RuntimeError):
    """A frozen input, output, runtime, or command violated the protocol."""


@dataclass
class HeldInput:
    label: str
    path: Path
    expected_sha256: str
    expected_mode: int | None = None
    descriptor: int = -1
    device: int = -1
    inode: int = -1
    size: int = -1

    def acquire(self) -> dict[str, Any]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self.descriptor = os.open(self.path, flags)
        except OSError as error:
            raise ProtocolError(f"cannot open {self.label}: {self.path}") from error
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            file_stat = os.fstat(self.descriptor)
            path_stat = os.stat(self.path, follow_symlinks=False)
            if not stat.S_ISREG(file_stat.st_mode) or int(file_stat.st_nlink) != 1:
                raise ProtocolError(
                    f"{self.label} must be a single-link regular file"
                )
            observed_mode = stat.S_IMODE(file_stat.st_mode)
            if self.expected_mode is not None and observed_mode != self.expected_mode:
                raise ProtocolError(
                    f"{self.label} mode mismatch: {oct(observed_mode)}"
                )
            if stat.S_ISLNK(path_stat.st_mode) or (
                int(file_stat.st_dev),
                int(file_stat.st_ino),
            ) != (int(path_stat.st_dev), int(path_stat.st_ino)):
                raise ProtocolError(f"{self.label} path does not name the held inode")
            self.device = int(file_stat.st_dev)
            self.inode = int(file_stat.st_ino)
            self.size = int(file_stat.st_size)
            evidence = self.rehash("acquire")
            if not evidence["pass"]:
                raise ProtocolError(f"{self.label} SHA-256 mismatch")
            return evidence
        except BaseException:
            self.close()
            raise

    def rehash(self, phase: str) -> dict[str, Any]:
        if self.descriptor < 0:
            raise ProtocolError(f"{self.label} is not held")
        before = os.fstat(self.descriptor)
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            block = os.read(self.descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(self.descriptor)
        try:
            path_stat = os.stat(self.path, follow_symlinks=False)
            path_identity = (
                not stat.S_ISLNK(path_stat.st_mode)
                and (int(path_stat.st_dev), int(path_stat.st_ino))
                == (self.device, self.inode)
            )
        except OSError:
            path_identity = False
        stable = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
            int(before.st_nlink),
        ) == (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
            int(after.st_nlink),
        )
        observed = digest.hexdigest()
        observed_mode = stat.S_IMODE(after.st_mode)
        mode_exact = (
            self.expected_mode is None or observed_mode == self.expected_mode
        )
        passed = (
            stable
            and path_identity
            and byte_count == self.size
            and observed == self.expected_sha256
            and mode_exact
        )
        return {
            "phase": phase,
            "label": self.label,
            "path": root_relative(self.path),
            "expected_sha256": self.expected_sha256,
            "observed_sha256": observed,
            "bytes": byte_count,
            "device": self.device,
            "inode": self.inode,
            "mode_octal": format(observed_mode, "04o"),
            "expected_mode_octal": (
                None
                if self.expected_mode is None
                else format(self.expected_mode, "04o")
            ),
            "mode_exact": mode_exact,
            "shared_flock_held": True,
            "stable_while_read": stable,
            "path_identity_unchanged": path_identity,
            "pass": passed,
        }

    def close(self) -> None:
        if self.descriptor >= 0:
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self.descriptor)
                self.descriptor = -1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"{label} must be a lowercase SHA-256 digest")
    return value


def root_relative(path: Path) -> str:
    normalized = Path(os.path.normpath(os.fspath(path.absolute())))
    try:
        return str(normalized.relative_to(ROOT.absolute()))
    except ValueError as error:
        raise ProtocolError(f"path is outside the repository: {path}") from error


def panel_contract_audit() -> dict[str, Any]:
    contract = [
        {
            "name": panel["name"],
            "data": root_relative(Path(panel["data"])),
            "data_sha256": panel["data_sha256"],
            "gates": panel["gates"],
        }
        for panel in PANELS
    ]
    observed_sha256 = sha256_bytes(canonical_json_bytes(contract))
    checks = {
        "panel_order_exact": [panel["name"] for panel in PANELS]
        == ["old_retention", "valid29"],
        "old_rows_exact_34204": PANELS[0]["gates"]["rows"]
        == {"exact": 34204},
        "valid29_rows_exact_159829": PANELS[1]["gates"]["rows"]
        == {"exact": 159829},
        "ten_gates_each_twenty_total": (
            [len(panel["gates"]) for panel in PANELS] == [10, 10]
        ),
        "all_thresholds_and_archives_hash_exact": observed_sha256
        == PANEL_CONTRACT_SHA256,
    }
    if not all(checks.values()):
        raise ProtocolError(f"untouched broad panel contract drift: {checks}")
    return {
        "sha256": observed_sha256,
        "checks": checks,
        "contract": contract,
    }


def strict_json_loads(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ProtocolError(f"{label} has non-finite value {value}")

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProtocolError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return parsed


def validate_materialization_provenance(
    manifest_payload: bytes,
    completed_payload: bytes,
) -> dict[str, Any]:
    manifest = strict_json_loads(
        manifest_payload, "CW11 materialization manifest"
    )
    completed = strict_json_loads(
        completed_payload, "CW11 materialization completion"
    )
    manifest_checkpoint = manifest.get("checkpoint")
    manifest_modules = manifest.get("frozen_modules")
    manifest_gate = manifest.get("fulltrain_gate_audit")
    completion_checkpoint = completed.get("checkpoint")
    completion_manifest = completed.get("manifest")
    if not all(
        isinstance(value, Mapping)
        for value in (
            manifest_checkpoint,
            manifest_modules,
            manifest_gate,
            completion_checkpoint,
            completion_manifest,
        )
    ):
        raise ProtocolError("CW11 materialization provenance schema drift")
    verification = manifest_checkpoint.get("verification")
    formal_binding = manifest_modules.get("cw11_formal_fulltrain_gate")
    module_bindings = manifest_modules.get("binding_checks")
    if not all(
        isinstance(value, Mapping)
        for value in (verification, formal_binding, module_bindings)
    ):
        raise ProtocolError("CW11 manifest nested provenance drift")
    verification_checks = verification.get("checks")
    formal_checks = manifest_gate.get("checks")
    if not isinstance(verification_checks, Mapping) or not isinstance(
        formal_checks, Mapping
    ):
        raise ProtocolError("CW11 manifest verification checks drift")
    expected_ledger = [
        (ordinal, model, panel, rows)
        for ordinal, (model, panel, rows) in enumerate(
            (
                (model, panel, rows)
                for model in ("raw", "candidate")
                for panel, rows in (
                    ("flg", 9443),
                    ("pokemonfan", 9487),
                    ("core5", 5120),
                )
            ),
            start=1,
        )
    ]
    observed_ledger = [
        (
            int(item.get("ordinal", -1)),
            item.get("model"),
            item.get("panel"),
            int(item.get("rows", -1)),
        )
        for item in manifest.get("fulltrain_completion_ledger", [])
        if isinstance(item, Mapping) and item.get("completed") is True
    ]
    expected_checkpoint = {
        "path": root_relative(CANDIDATE),
        "sha256": CANDIDATE_SHA256,
        "model_state_sha256": CANDIDATE_MODEL_STATE_SHA256,
    }
    checks = {
        "manifest_schema_status_branch_exact": (
            manifest.get("schema_version") == MATERIALIZATION_MANIFEST_SCHEMA
            and manifest.get("status")
            == "staged_complete_pending_atomic_directory_publish"
            and manifest.get("branch") == MATERIALIZATION_BRANCH
        ),
        "manifest_checkpoint_exact": all(
            manifest_checkpoint.get(key) == value
            for key, value in expected_checkpoint.items()
        ),
        "manifest_checkpoint_verification_exact": (
            verification.get("candidate_model_state_sha256")
            == CANDIDATE_MODEL_STATE_SHA256
            and verification.get("candidate_nonactor_sha256")
            == EXPECTED_RAW_NONACTOR_SHA256
            and int(verification.get("state_tensor_count", -1))
            == EXPECTED_MODEL_TENSOR_COUNT
            and all(value is True for value in verification_checks.values())
        ),
        "formal_gate_binding_exact": (
            formal_binding.get("path") == root_relative(FORMAL_GATE)
            and formal_binding.get("sha256") == FORMAL_GATE_SHA256
            and formal_binding.get("mode_octal") == "0555"
            and all(value is True for value in module_bindings.values())
        ),
        "formal_gate_pass_exact": (
            all(value is True for value in formal_checks.values())
            and manifest.get("pokemonfan_observed_wc")
            == EXPECTED_FORMAL_PF_WC_NET
            and manifest.get("pokemonfan_observed_net_gains")
            == EXPECTED_FORMAL_PF_WC_NET
            and observed_ledger == expected_ledger
        ),
        "truthful_consumed_nonpromotion_scope": (
            manifest.get("specialist_valid_consumed_for_optimization") is True
            and manifest.get("promotion_evidence") is False
            and manifest.get(
                "additional_validation_broad_gold_opened_by_materializer"
            )
            is False
            and manifest.get("network_upload_submission") is False
        ),
        "publication_contract_exact": (
            manifest.get("publication", {}).get("output_root")
            == root_relative(CANDIDATE.parent)
            and manifest.get("publication", {}).get("method")
            == "renameat2(RENAME_NOREPLACE)"
            and manifest.get("publication", {}).get("no_overwrite") is True
            and manifest.get("publication", {}).get("files_mode") == "0444"
            and manifest.get("publication", {}).get("directory_mode") == "0555"
        ),
        "completion_schema_status_branch_exact": (
            completed.get("schema_version")
            == MATERIALIZATION_COMPLETION_SCHEMA
            and completed.get("status")
            == "complete_only_after_atomic_directory_publish"
            and completed.get("branch") == MATERIALIZATION_BRANCH
        ),
        "completion_checkpoint_and_manifest_exact": (
            all(
                completion_checkpoint.get(key) == value
                for key, value in expected_checkpoint.items()
            )
            and completion_manifest.get("path")
            == root_relative(MATERIALIZATION_MANIFEST)
            and completion_manifest.get("sha256")
            == MATERIALIZATION_MANIFEST_SHA256
        ),
        "completion_eval_only_nonpromotion_exact": (
            completed.get("evaluation_only") is True
            and completed.get("resume_forbidden") is True
            and completed.get("specialist_valid_consumed_for_optimization")
            is True
            and completed.get("promotion_evidence") is False
            and completed.get("submission_performed") is False
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW11 materialization provenance drift: {checks}")
    return {
        "manifest_sha256": MATERIALIZATION_MANIFEST_SHA256,
        "completed_sha256": MATERIALIZATION_COMPLETED_SHA256,
        "formal_gate_sha256": FORMAL_GATE_SHA256,
        "materializer_sha256": MATERIALIZER_SHA256,
        "checks": checks,
    }


def read_regular_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ProtocolError(f"{label} must be a single-link regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ProtocolError(f"{label} changed while it was read")
        payload = b"".join(chunks)
        return payload, {
            "path": root_relative(path),
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
        }
    finally:
        os.close(descriptor)


def model_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ProtocolError("model_state_dict must map names to tensors")
        value = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ProtocolError(f"non-finite checkpoint tensor: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_candidate(payload: bytes) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise ProtocolError("candidate cannot be loaded on CPU") from error
    if not isinstance(checkpoint, dict):
        raise ProtocolError("candidate checkpoint must be a dictionary")
    expected_keys = [
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
        MATERIALIZATION_AUDIT_KEY,
    ]
    if (
        len(expected_keys) != EXPECTED_CHECKPOINT_KEY_COUNT
        or list(checkpoint) != expected_keys
    ):
        raise ProtocolError("candidate top-level 13-key eval-only schema drift")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ProtocolError("candidate has no model_state_dict mapping")
    observed = model_state_sha256(state)
    if observed != CANDIDATE_MODEL_STATE_SHA256:
        raise ProtocolError("candidate runtime model-state SHA-256 mismatch")
    if checkpoint.get("update") != EXPECTED_CHECKPOINT_UPDATE:
        raise ProtocolError("candidate update must remain exactly 468")
    if checkpoint.get("evaluation_only") is not True:
        raise ProtocolError("candidate must remain evaluation-only")
    if checkpoint.get("resume_forbidden") is not True:
        raise ProtocolError("candidate must forbid training resume")
    nonactor = {
        name: tensor
        for name, tensor in state.items()
        if name not in ACTOR_PARAMETER_NAMES
    }
    nonactor_sha256 = model_state_sha256(nonactor)
    if (
        len(state) != EXPECTED_MODEL_TENSOR_COUNT
        or len(nonactor) != EXPECTED_NONACTOR_TENSOR_COUNT
        or EXPECTED_NONACTOR_TENSOR_COUNT
        != EXPECTED_MODEL_TENSOR_COUNT - len(ACTOR_PARAMETER_NAMES)
        or bool(set(ACTOR_PARAMETER_NAMES) - set(state))
        or nonactor_sha256 != EXPECTED_RAW_NONACTOR_SHA256
    ):
        raise ProtocolError("candidate actor-six/nonactor identity drift")
    forbidden_resume_keys = {
        "optimizer",
        "optimizer_state",
        "optimizer_states",
        "optimizer_state_dict",
        "optimizer_parameter_names",
        "scheduler_state_dict",
        "lr_scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
        "sampler_state",
        "bc_replay_optimizer_state_dict",
        "replay_optimizer_state_dict",
        "opponent_quota_state",
        "fresh_special_optimizer_state_dict",
        "metrics",
        "value_trunk_gradient",
        "actor_value_gradient",
    }
    present_forbidden = sorted(forbidden_resume_keys.intersection(checkpoint))
    if present_forbidden:
        raise ProtocolError(
            f"candidate unexpectedly contains resume state: {present_forbidden}"
        )
    audit = checkpoint.get(MATERIALIZATION_AUDIT_KEY)
    if not isinstance(audit, Mapping):
        raise ProtocolError("candidate lacks CW11 materialization audit")
    materializer = audit.get("materializer")
    fixed_inputs = audit.get("fixed_inputs")
    formal_input = (
        fixed_inputs.get("cw11_formal_fulltrain_gate")
        if isinstance(fixed_inputs, Mapping)
        else None
    )
    audit_checks = {
        "schema_exact": audit.get("schema_version") == MATERIALIZER_SCHEMA,
        "candidate_model_exact": audit.get("candidate_model_state_sha256")
        == CANDIDATE_MODEL_STATE_SHA256,
        "candidate_nonactor_exact": audit.get("candidate_nonactor_sha256")
        == EXPECTED_RAW_NONACTOR_SHA256,
        "evaluation_only": audit.get("evaluation_only") is True,
        "resume_forbidden": audit.get("resume_forbidden") is True,
        "specialist_consumed": audit.get(
            "specialist_valid_consumed_for_optimization"
        )
        is True,
        "not_promotion_evidence": audit.get("promotion_evidence") is False,
        "materializer_opened_no_additional_data": audit.get(
            "additional_validation_broad_gold_opened_by_materializer"
        )
        is False,
        "materializer_binding_exact": (
            isinstance(materializer, Mapping)
            and materializer.get("path") == root_relative(MATERIALIZER)
            and materializer.get("sha256") == MATERIALIZER_SHA256
            and materializer.get("mode_octal") == "0555"
        ),
        "formal_binding_exact": (
            isinstance(formal_input, Mapping)
            and formal_input.get("path") == root_relative(FORMAL_GATE)
            and formal_input.get("sha256") == FORMAL_GATE_SHA256
            and formal_input.get("mode_octal") == "0555"
        ),
    }
    if not all(audit_checks.values()):
        raise ProtocolError(f"candidate materialization audit drift: {audit_checks}")
    return {
        "checkpoint_update": EXPECTED_CHECKPOINT_UPDATE,
        "model_state_sha256": observed,
        "model_tensor_count": len(state),
        "top_level_key_count": len(checkpoint),
        "candidate_nonactor_sha256": nonactor_sha256,
        "evaluation_only": True,
        "resume_forbidden": True,
        "forbidden_resume_keys_present": [],
        "materialization_audit_checks": audit_checks,
    }


def atomic_create(path: Path, payload: bytes, mode: int = 0o444) -> None:
    if not path.parent.is_dir():
        raise ProtocolError(f"output parent is absent: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def require_absent(path: Path, label: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise ProtocolError(f"{label} already exists; no retry is permitted: {path}")


def command_for(panel: Mapping[str, Any], output: Path) -> list[str]:
    return [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        root_relative(EVALUATOR),
        "--checkpoint",
        root_relative(CANDIDATE),
        "--data",
        root_relative(Path(panel["data"])),
        "--split",
        "valid",
        "--split-mode",
        "archive",
        "--split-seed",
        "20260723",
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--prediction-order",
        "policy",
        "--device",
        "cuda",
        "--compact",
        "--progress-interval",
        "0",
        "--json-output",
        root_relative(output),
    ]


def metric_value(document: Mapping[str, Any], metric: str) -> int:
    current: Any = document
    for key in METRIC_PATHS[metric]:
        if not isinstance(current, Mapping) or key not in current:
            raise ProtocolError(
                f"output lacks metric path {'.'.join(METRIC_PATHS[metric])}"
            )
        current = current[key]
    if type(current) is not int:
        raise ProtocolError(f"output metric {metric} must be an integer")
    return current


def observed_metrics(document: Mapping[str, Any]) -> dict[str, int]:
    return {name: metric_value(document, name) for name in METRIC_PATHS}


def evaluate_gates(
    panel: Mapping[str, Any], metrics: Mapping[str, int]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for metric, comparison in panel["gates"].items():
        observed = metrics[metric]
        if "exact" in comparison:
            operator = "=="
            threshold = comparison["exact"]
            passed = observed == threshold
        else:
            operator = ">="
            threshold = comparison["minimum"]
            passed = observed >= threshold
        results.append(
            {
                "gate": f"{panel['name']}.{metric}",
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "pass": passed,
            }
        )
    return results


def validate_output_identity(
    document: Mapping[str, Any], panel: Mapping[str, Any]
) -> None:
    if Path(str(document.get("checkpoint"))).resolve() != CANDIDATE.resolve():
        raise ProtocolError("output checkpoint path mismatch")
    if document.get("checkpoint_sha256") != CANDIDATE_SHA256:
        raise ProtocolError("output checkpoint SHA-256 mismatch")
    if document.get("checkpoint_update") != EXPECTED_CHECKPOINT_UPDATE:
        raise ProtocolError("output checkpoint update mismatch")
    if Path(str(document.get("data"))).resolve() != Path(panel["data"]).resolve():
        raise ProtocolError("output data path mismatch")
    expected = {
        "split": "valid",
        "split_mode": "archive",
        "split_seed": 20260723,
        "prediction_order": "policy_greedy",
        "device": "cuda",
        "max_rows": None,
        "evaluator": "tools/evaluate_policy_bc.py",
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ProtocolError(f"output field {key} mismatch")
    filters = document.get("filters")
    if not isinstance(filters, Mapping):
        raise ProtocolError("output filters must be an object")
    if filters.get("deck_hashes") != [] or filters.get("team_names") != []:
        raise ProtocolError("output filters mismatch")


def cuda_preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is unavailable")
    device = torch.device("cuda")
    left = torch.tensor([2.0, 3.0], device=device)
    right = torch.tensor([5.0, 7.0], device=device)
    observed = float((left * right).sum().item())
    torch.cuda.synchronize(device)
    if observed != 31.0:
        raise ProtocolError("CUDA kernel preflight produced the wrong result")
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_index": torch.cuda.current_device(),
        "capability": list(torch.cuda.get_device_capability(device)),
        "cuda_runtime": torch.version.cuda,
        "kernel_result": observed,
    }


def load_candidate_and_provenance() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    candidate_payload, candidate_file = read_regular_file(CANDIDATE, "candidate")
    if candidate_file["sha256"] != CANDIDATE_SHA256:
        raise ProtocolError("candidate file SHA-256 mismatch")
    candidate_identity = validate_candidate(candidate_payload)
    manifest_payload, manifest_file = read_regular_file(
        MATERIALIZATION_MANIFEST, "CW11 materialization manifest"
    )
    completed_payload, completed_file = read_regular_file(
        MATERIALIZATION_COMPLETED, "CW11 materialization completion"
    )
    if (
        manifest_file["sha256"] != MATERIALIZATION_MANIFEST_SHA256
        or completed_file["sha256"] != MATERIALIZATION_COMPLETED_SHA256
    ):
        raise ProtocolError("materialization provenance file SHA-256 mismatch")
    parent_stat = os.stat(CANDIDATE.parent, follow_symlinks=False)
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_IMODE(parent_stat.st_mode) != 0o555
    ):
        raise ProtocolError("materialized checkpoint directory must remain mode 0555")
    provenance = validate_materialization_provenance(
        manifest_payload, completed_payload
    )
    provenance["files"] = {
        "manifest": manifest_file,
        "completed": completed_file,
        "directory_mode_octal": format(stat.S_IMODE(parent_stat.st_mode), "04o"),
    }
    return candidate_file, candidate_identity, provenance


def make_bindings(runner_sha256: str) -> list[HeldInput]:
    return [
        HeldInput("runner", SCRIPT, runner_sha256, expected_mode=0o555),
        HeldInput(
            "candidate", CANDIDATE, CANDIDATE_SHA256, expected_mode=0o444
        ),
        HeldInput(
            "materialization_manifest",
            MATERIALIZATION_MANIFEST,
            MATERIALIZATION_MANIFEST_SHA256,
            expected_mode=0o444,
        ),
        HeldInput(
            "materialization_completed",
            MATERIALIZATION_COMPLETED,
            MATERIALIZATION_COMPLETED_SHA256,
            expected_mode=0o444,
        ),
        HeldInput(
            "materializer",
            MATERIALIZER,
            MATERIALIZER_SHA256,
            expected_mode=0o555,
        ),
        HeldInput(
            "formal_gate",
            FORMAL_GATE,
            FORMAL_GATE_SHA256,
            expected_mode=0o555,
        ),
        HeldInput("old_retention_data", Path(PANELS[0]["data"]), PANELS[0]["data_sha256"]),
        HeldInput("valid29_data", Path(PANELS[1]["data"]), PANELS[1]["data_sha256"]),
        HeldInput("evaluator", EVALUATOR, EVALUATOR_SHA256),
        *(
            HeldInput(name, path, digest)
            for name, path, digest in DEPENDENCIES
        ),
    ]


def acquire_all(bindings: Sequence[HeldInput]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    try:
        for binding in bindings:
            evidence.append(binding.acquire())
    except BaseException:
        close_all(bindings)
        raise
    return evidence


def rehash_all(
    bindings: Sequence[HeldInput], phase: str
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for binding in bindings:
        try:
            records.append(binding.rehash(phase))
        except BaseException as error:
            records.append(
                {
                    "phase": phase,
                    "label": binding.label,
                    "path": root_relative(binding.path),
                    "expected_sha256": binding.expected_sha256,
                    "observed_sha256": None,
                    "pass": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "phase": phase,
        "binding_count": len(records),
        "pass": len(records) == len(bindings)
        and all(bool(record["pass"]) for record in records),
        "records": records,
    }


def close_all(bindings: Sequence[HeldInput]) -> None:
    for binding in reversed(bindings):
        binding.close()


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise ProtocolError("run from the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError(
            f"wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )
    if sys.flags.isolated != 1 or sys.dont_write_bytecode is not True:
        raise ProtocolError("requires Python flags -I -B")


def output_paths() -> list[Path]:
    return [OUTPUT_ROOT / f"{panel['name']}.json" for panel in PANELS]


def assert_fresh_outputs() -> None:
    require_absent(OUTPUT_ROOT, "fixed broad output root")
    require_absent(ATTEMPT_MARKER, "formal-attempt marker")
    require_absent(TERMINAL_FAILURE, "terminal-failure record")
    for path in (MANIFEST, DECISION, *output_paths()):
        require_absent(path, "formal broad output")


def static_preflight(runner_sha256: str) -> dict[str, Any]:
    panel_audit = panel_contract_audit()
    bindings = make_bindings(runner_sha256)
    try:
        acquired = acquire_all(bindings)
        candidate_file, candidate_identity, provenance = (
            load_candidate_and_provenance()
        )
        assert_fresh_outputs()
        commands = [
            {
                "order": order,
                "panel": panel["name"],
                "command": command_for(panel, output),
                "command_sha256": sha256_bytes(
                    canonical_json_bytes(command_for(panel, output))
                ),
                "gate_count": len(panel["gates"]),
            }
            for order, (panel, output) in enumerate(
                zip(PANELS, output_paths(), strict=True), start=1
            )
        ]
        final_rehash = rehash_all(bindings, "preflight_end")
        if not final_rehash["pass"]:
            raise ProtocolError("static-preflight final immutable-input rehash failed")
        if torch.cuda.is_initialized():
            raise ProtocolError("read-only preflight unexpectedly initialized CUDA")
        return {
            "schema_version": (
                "ptcg-cw11-single-candidate-untouched-broad-preflight-v1"
            ),
            "status": "preflight_passed_execute_not_started",
            "cuda_initialized": torch.cuda.is_initialized(),
            "formal_attempt_consumed": False,
            "runner": acquired[0],
            "candidate": candidate_file | candidate_identity,
            "materialization_provenance": provenance,
            "input_locks": {
                "count": len(bindings),
                "acquired": acquired,
                "final_rehash": final_rehash,
            },
            "protocol": PROTOCOL,
            "panels": commands,
            "authoritative_gates": {
                panel["name"]: panel["gates"] for panel in PANELS
            },
            "panel_contract_audit": panel_audit,
            "total_gate_count": sum(len(panel["gates"]) for panel in PANELS),
            "output_root": root_relative(OUTPUT_ROOT),
            "attempt_marker": root_relative(ATTEMPT_MARKER),
            "execution_prerequisite_confirmation": (
                EXECUTION_PREREQUISITE_CONFIRMATION
            ),
            "scope": SCOPE,
        }
    finally:
        close_all(bindings)


def evaluate_one(
    panel: Mapping[str, Any], output: Path, order: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    command = command_for(panel, output)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = utc_now()
    returncode: int | None = None
    stdout = b""
    stderr = b""
    launch_error: str | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
        returncode = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except BaseException as error:
        launch_error = f"{type(error).__name__}: {error}"

    record: dict[str, Any] = {
        "order": order,
        "panel": panel["name"],
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "attempt_count": 1,
        "retry_authorized": False,
        "command": command,
        "command_sha256": sha256_bytes(canonical_json_bytes(command)),
        "returncode": returncode,
        "launch_error": launch_error,
        "stdout": {"sha256": sha256_bytes(stdout), "bytes": len(stdout)},
        "stderr": {"sha256": sha256_bytes(stderr), "bytes": len(stderr)},
        "output": root_relative(output),
        "output_present": output.is_file() and not output.is_symlink(),
    }
    gates: list[dict[str, Any]] = []
    output_error: str | None = None
    if record["output_present"]:
        try:
            payload, file_evidence = read_regular_file(
                output, f"{panel['name']} evaluator output"
            )
            document = strict_json_loads(payload, f"{panel['name']} evaluator output")
            validate_output_identity(document, panel)
            metrics = observed_metrics(document)
            gates = evaluate_gates(panel, metrics)
            record["output_evidence"] = file_evidence
            record["metrics"] = metrics
        except BaseException as error:
            output_error = f"{type(error).__name__}: {error}"
        finally:
            try:
                os.chmod(output, 0o444, follow_symlinks=False)
            except OSError as error:
                if output_error is None:
                    output_error = f"failed to freeze evaluator output: {error}"
    else:
        output_error = "evaluator output is absent"
    record["output_error"] = output_error
    record["gates"] = gates
    record["evaluation_pass"] = (
        returncode == 0
        and launch_error is None
        and output_error is None
        and len(gates) == 10
        and all(bool(gate["pass"]) for gate in gates)
    )
    return record, gates


def write_terminal_failure(
    runner_sha256: str, marker_created: bool, error: BaseException
) -> None:
    if not marker_created:
        return
    payload = canonical_json_bytes(
        {
            "schema_version": (
                "ptcg-cw11-single-candidate-untouched-broad-"
                "terminal-failure-v1"
            ),
            "status": "terminal_failure_no_retry",
            "created_at_utc": utc_now(),
            "runner": {"path": root_relative(SCRIPT), "sha256": runner_sha256},
            "attempt_marker": root_relative(ATTEMPT_MARKER),
            "error": f"{type(error).__name__}: {error}",
            "scope": SCOPE,
        }
    )
    try:
        atomic_create(TERMINAL_FAILURE, payload, mode=0o444)
    except BaseException:
        pass


def formal_execute(runner_sha256: str) -> dict[str, Any]:
    panel_audit = panel_contract_audit()
    bindings = make_bindings(runner_sha256)
    marker_created = False
    try:
        acquired = acquire_all(bindings)
        candidate_file, candidate_identity, provenance = (
            load_candidate_and_provenance()
        )
        assert_fresh_outputs()
        before = rehash_all(bindings, "before_cuda_and_evaluation")
        if not before["pass"]:
            raise ProtocolError("pre-execution immutable-input rehash failed")
        cuda = cuda_preflight()
        after_cuda = rehash_all(bindings, "after_cuda_before_attempt_marker")
        if not after_cuda["pass"]:
            raise ProtocolError("post-CUDA immutable-input rehash failed")

        marker_payload = canonical_json_bytes(
            {
                "schema_version": (
                    "ptcg-cw11-single-candidate-untouched-broad-attempt-v1"
                ),
                "status": "formal_attempt_consumed_no_retry",
                "created_at_utc": utc_now(),
                "pid": os.getpid(),
                "runner": {"path": root_relative(SCRIPT), "sha256": runner_sha256},
                "candidate": {
                    "path": root_relative(CANDIDATE),
                    "sha256": CANDIDATE_SHA256,
                    "model_state_sha256": CANDIDATE_MODEL_STATE_SHA256,
                },
                "materialization_provenance": provenance,
                "panel_contract_audit": panel_audit,
                "cuda_preflight": cuda,
                "panel_count": 2,
                "attempts_per_panel": 1,
                "retry_authorized": False,
                "scope": SCOPE,
            }
        )
        atomic_create(ATTEMPT_MARKER, marker_payload, mode=0o444)
        marker_created = True
        OUTPUT_ROOT.mkdir(mode=0o700)

        records: list[dict[str, Any]] = []
        gates: list[dict[str, Any]] = []
        for order, (panel, output) in enumerate(
            zip(PANELS, output_paths(), strict=True), start=1
        ):
            record, panel_gates = evaluate_one(panel, output, order)
            records.append(record)
            gates.extend(panel_gates)

        after = rehash_all(bindings, "after_both_panels_before_decision")
        complete = (
            len(records) == 2
            and all(record["attempt_count"] == 1 for record in records)
        )
        passed = (
            complete
            and len(gates) == 20
            and all(bool(record["evaluation_pass"]) for record in records)
            and after["pass"]
        )
        manifest_document = {
            "schema_version": (
                "ptcg-cw11-single-candidate-untouched-broad-manifest-v1"
            ),
            "status": "two_panels_completed" if complete else "execution_incomplete",
            "created_at_utc": utc_now(),
            "runner": acquired[0],
            "candidate": candidate_file | candidate_identity,
            "materialization_provenance": provenance,
            "panel_contract_audit": panel_audit,
            "protocol": PROTOCOL,
            "cuda_preflight": cuda,
            "immutable_inputs": {
                "acquired": acquired,
                "before": before,
                "after_cuda": after_cuda,
                "after_both_panels": after,
            },
            "ordered_evaluations": records,
            "both_panels_attempted_once": complete,
            "gate_count": len(gates),
            "scope": SCOPE,
        }
        manifest_payload = canonical_json_bytes(manifest_document)
        atomic_create(MANIFEST, manifest_payload, mode=0o444)
        decision_document = {
            "schema_version": (
                "ptcg-cw11-single-candidate-untouched-broad-decision-v1"
            ),
            "status": "passed_broad_behavior" if passed else "failed_broad_behavior",
            "created_at_utc": utc_now(),
            "pass": passed,
            "candidate": {
                "path": root_relative(CANDIDATE),
                "sha256": CANDIDATE_SHA256,
                "model_state_sha256": CANDIDATE_MODEL_STATE_SHA256,
                "checkpoint_update": EXPECTED_CHECKPOINT_UPDATE,
            },
            "manifest": {
                "path": root_relative(MANIFEST),
                "sha256": sha256_bytes(manifest_payload),
            },
            "both_panels_attempted_once": complete,
            "all_20_gates_required": True,
            "gate_count": len(gates),
            "passed_gate_count": sum(bool(gate["pass"]) for gate in gates),
            "failed_gates": [gate for gate in gates if not gate["pass"]],
            "panel_pass": {
                record["panel"]: record["evaluation_pass"] for record in records
            },
            "post_execution_input_lock_pass": after["pass"],
            "evidence_classification": {
                "specialist_valid_consumed_for_optimization": True,
                "dev_consistency_only": True,
                "dev_consistency_promotion_evidence": False,
                "broad_untouched": True,
                "broad_independent_promotion_evidence": passed,
                "broad_independent_promotion_evidence_only_if_pass": True,
            },
            "authorization": {
                "gold_local_evaluation_authorized": False,
                "package_upload_or_submission_authorized": False,
            },
        }
        decision_payload = canonical_json_bytes(decision_document)
        atomic_create(DECISION, decision_payload, mode=0o444)
        os.chmod(OUTPUT_ROOT, 0o500)
        return {
            "status": decision_document["status"],
            "pass": passed,
            "decision": root_relative(DECISION),
            "decision_sha256": sha256_bytes(decision_payload),
            "both_panels_attempted_once": complete,
            "gate_count": len(gates),
            "passed_gate_count": decision_document["passed_gate_count"],
            "broad_independent_promotion_evidence": passed,
            "gold_local_evaluation_authorized": False,
        }
    except BaseException as error:
        write_terminal_failure(runner_sha256, marker_created, error)
        raise
    finally:
        close_all(bindings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or explicitly execute the frozen CW11 untouched broad gate."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "execute"),
        default="preflight",
        help="read-only CPU preflight by default; execute consumes the one-shot marker",
    )
    parser.add_argument(
        "--expected-runner-sha256",
        help="required in execute mode; exact SHA-256 of this runner",
    )
    parser.add_argument(
        "--confirm-dev-consistency-and-audit-go",
        help="required exact prerequisite confirmation in execute mode",
    )
    parser.add_argument(
        "--confirm-output-root",
        help="required exact unique output root in execute mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    runner_payload, runner_evidence = read_regular_file(SCRIPT, "runner")
    observed_runner_sha256 = runner_evidence["sha256"]
    if args.expected_runner_sha256 is not None:
        expected = require_sha256(
            args.expected_runner_sha256, "expected runner SHA-256"
        )
        if observed_runner_sha256 != expected:
            raise ProtocolError("runner SHA-256 differs from the command-line lock")
    if args.mode == "execute" and args.expected_runner_sha256 is None:
        raise ProtocolError("execute mode requires --expected-runner-sha256")
    if args.mode == "preflight" and (
        args.confirm_dev_consistency_and_audit_go is not None
        or args.confirm_output_root is not None
    ):
        raise ProtocolError("preflight mode rejects execute confirmations")
    if args.mode == "execute" and (
        args.confirm_dev_consistency_and_audit_go
        != EXECUTION_PREREQUISITE_CONFIRMATION
    ):
        raise ProtocolError(
            "execute requires confirmed dev consistency PASS and independent audit GO"
        )
    if args.mode == "execute" and args.confirm_output_root != root_relative(
        OUTPUT_ROOT
    ):
        raise ProtocolError("execute requires exact unique output-root confirmation")
    del runner_payload

    if args.mode == "preflight":
        result = static_preflight(observed_runner_sha256)
    else:
        result = formal_execute(observed_runner_sha256)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.mode == "execute" and not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

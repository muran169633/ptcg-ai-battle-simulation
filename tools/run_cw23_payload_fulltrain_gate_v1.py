#!/usr/bin/env python3
"""Payload-bound full-train gate for the sole CW23 boundary-QP candidate."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import lzma
import math
import os
import random
import stat
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_cw23_payload_fulltrain_gate_v1.py"
SCHEMA = "ptcg-cw23-payload-fulltrain-gate-v1"

CW23 = ROOT / "artifacts/cw23_cw11_boundary_qp_specialbc_trainonly_20260803_v1.json"
CW23_SHA256 = "b9960b8b48791c88d099dbe23930df9293c4e15facecdc4498272e4108956f45"
RAW = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
RAW_FILE_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
RAW_NONACTOR_SHA256 = "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
CW23_MODEL_SHA256 = "bf551805d807cb6a77f500a51a12204727ab5cbd371fb1d36a947c63299228cb"
CW23_PAYLOAD_CANONICAL_SHA256 = (
    "69488bd9e22832131dd9d69803228049847775a6ba730094ea946254151b597a"
)
ACTOR_LAYOUT_SHA256 = (
    "af606d32ec4d47b4fe7870a9076308f6faa50da9f90aa3ebd77517942bd2a14e"
)
ACTOR_RAW_SHA256 = (
    "be40ae847a973ec43c6d90a07e2a6fa60b084473b45db164166447843b53e24f"
)
ACTOR_COMPRESSED_SHA256 = (
    "9e6d01c0fa37695df96c44b06af49a309b42084b2044f3d1a043b6f6561188cd"
)

FULLTRAIN = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v3.py"
FULLTRAIN_SHA256 = "f32c077c0d577bcc7c0ad2e42bdfda1ec1641244004b29efd5c82008cfe92338"
FORMAL = TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v3.py"
FORMAL_SHA256 = "2d12ce9e76f6672911948dcc6458573899ec45fc39e748510544001025ede939"
CW20 = TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py"
CW20_SHA256 = "8f5b64c8a04b3de9a6d2890aac582967a50f4d708f3a11a64cbd395423d907d2"

ATTEMPT = ROOT / "artifacts/.ptcg-cw23_payload_fulltrain_gate_20260803_v1-attempt.json"
OUTPUT = ROOT / "artifacts/cw23_payload_fulltrain_gate_20260803_v1.json"
FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444
ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
ACTOR6_DIMENSION = 65793
ACTOR_RAW_BYTES = ACTOR6_DIMENSION * 4
ACTOR_COMPRESSED_BYTES = 242960
ACTOR_BASE64_CHUNKS = 4263
EXPECTED_ROWS = {"flg": 9443, "pokemonfan": 9487, "core5": 5120}
EXECUTION_BINDING = globals().get("_PTCG_HASH_BOUND_EXECUTION")


class ProtocolError(RuntimeError):
    """Fail-closed full-train protocol error."""


def canonical_json(value: Any) -> bytes:
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


def lexists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def read_regular(
    path: Path,
    expected_sha256: str | None,
    expected_mode: int,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise ProtocolError(f"{label}: unsafe type/link/mode")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    chunks: list[bytes] = []
    try:
        opened = os.fstat(fd)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
        after_path = path.lstat()
    finally:
        os.close(fd)
    identity = lambda value: (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )
    if not identity(before) == identity(opened) == identity(after_fd) == identity(after_path):
        raise ProtocolError(f"{label}: identity changed during read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    if (expected_sha256 is not None and digest != expected_sha256) or len(
        payload
    ) != int(after_fd.st_size):
        raise ProtocolError(f"{label}: SHA/size drift")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(stat.S_IMODE(after_fd.st_mode), "04o"),
        "device": int(after_fd.st_dev),
        "inode": int(after_fd.st_ino),
        "nlink": int(after_fd.st_nlink),
    }


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ProtocolError(f"{label}: duplicate key {key}")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ProtocolError(f"{label}: nonfinite token {token}")
        ),
    )
    if not isinstance(value, dict) or canonical_json(value) != payload:
        raise ProtocolError(f"{label}: noncanonical object")
    return value


def publish(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    observed_fd: os.stat_result | None = None
    try:
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise ProtocolError(f"short publication: {path}")
            written += count
        os.fsync(fd)
        os.fchmod(fd, EVIDENCE_MODE)
        os.fsync(fd)
        observed_fd = os.fstat(fd)
    finally:
        os.close(fd)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_fd = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    observed_path = path.lstat()
    if (
        observed_fd is None
        or not stat.S_ISREG(observed_path.st_mode)
        or stat.S_ISLNK(observed_path.st_mode)
        or int(observed_path.st_nlink) != 1
        or stat.S_IMODE(observed_path.st_mode) != EVIDENCE_MODE
        or (observed_fd.st_dev, observed_fd.st_ino, observed_fd.st_size)
        != (observed_path.st_dev, observed_path.st_ino, observed_path.st_size)
        or int(observed_path.st_size) != len(payload)
    ):
        raise ProtocolError(f"publication identity drift: {path}")
    reloaded, record = read_regular(
        path, sha256_bytes(payload), EVIDENCE_MODE, f"published {path.name}"
    )
    if reloaded != payload:
        raise ProtocolError(f"publication payload drift: {path}")
    return record


def import_frozen(
    path: Path,
    digest: str,
    name: str,
    expected_mode: int = FROZEN_MODE,
) -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = read_regular(path, digest, expected_mode, name)
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    code = compile(source, str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module, evidence


def exact_nested_module_loader(
    path: Path, digest: str, name: str, mode: int
) -> ModuleType:
    module, _ = import_frozen(path, digest, name, expected_mode=mode)
    return module


def validate_runtime(require_cuda: bool) -> dict[str, Any]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
    }
    if require_cuda:
        import torch

        checks.update(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_count_positive": torch.cuda.device_count() > 0,
                "cuda0_bf16": torch.cuda.is_bf16_supported(),
            }
        )
    if not all(checks.values()):
        raise ProtocolError(f"runtime drift: {checks}")
    return checks


def validate_execution_binding(
    source_record: Mapping[str, Any], audit_only_mode: bool
) -> dict[str, Any]:
    binding = EXECUTION_BINDING
    checks = {
        "mapping": isinstance(binding, Mapping),
        "schema": isinstance(binding, Mapping)
        and binding.get("schema_version")
        == "ptcg-cw23-fulltrain-hash-bound-launch-v1",
        "adapter_path": isinstance(binding, Mapping)
        and binding.get("adapter_path") == str(SCRIPT),
        "adapter_sha": isinstance(binding, Mapping)
        and binding.get("adapter_sha256") == source_record.get("sha256"),
        "adapter_bytes": isinstance(binding, Mapping)
        and binding.get("adapter_bytes") == source_record.get("bytes"),
        "adapter_mode": isinstance(binding, Mapping)
        and binding.get("adapter_mode_octal") == "0555",
        "held_fd_identity": isinstance(binding, Mapping)
        and binding.get("held_fd_identity_exact") is True,
        "compile_exec_same_bytes": isinstance(binding, Mapping)
        and binding.get("compile_exec_same_verified_bytes") is True,
        "audit_mode_exact": isinstance(binding, Mapping)
        and binding.get("audit_only") is audit_only_mode,
        "launcher_regular_frozen": isinstance(binding, Mapping)
        and isinstance(binding.get("launcher"), Mapping)
        and binding["launcher"].get("mode_octal") == "0555"
        and binding["launcher"].get("nlink") == 1
        and binding["launcher"].get("path")
        == "tools/launch_cw23_payload_fulltrain_gate_v1.py",
    }
    if not all(checks.values()) or not isinstance(binding, Mapping):
        raise ProtocolError(f"hash-bound execution contract failed: {checks}")
    copied = json.loads(canonical_json(binding).decode("utf-8"))
    if not isinstance(copied, dict):
        raise ProtocolError("execution binding is not JSON-native")
    return {"checks": checks, "binding": copied}


def source_audit(source: bytes, require_frozen: bool) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    string_literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = sorted(calls.intersection({"save", "backward", "step"}))
    checks = {
        "regular_single_link": SCRIPT.is_file()
        and not SCRIPT.is_symlink()
        and SCRIPT.stat().st_nlink == 1,
        "mode_exact": (not require_frozen)
        or stat.S_IMODE(SCRIPT.stat().st_mode) == FROZEN_MODE,
        "no_save_backward_step": not forbidden,
        "one_output_path": string_literals.count(str(OUTPUT.relative_to(ROOT))) == 1,
        "one_attempt_path": string_literals.count(str(ATTEMPT.relative_to(ROOT)))
        == 1,
        "raw_plus_absolute_payload_reconstruction": b"candidate_actor_float32_le"
        in source,
        "frozen_fulltrain_api": all(
            token in source
            for token in (
                b"load_fulltrain_inputs",
                b"evaluate_raw_candidate",
                b"decide_fulltrain",
            )
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"source audit failed: {checks}; forbidden={forbidden}")
    return {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": sha256_bytes(source),
        "bytes": len(source),
        "mode_octal": format(stat.S_IMODE(SCRIPT.stat().st_mode), "04o"),
        "checks": checks,
    }


def decode_actor(value: Mapping[str, Any]) -> bytes:
    expected_keys = {
        "base64_chunks_76",
        "compressed_bytes",
        "compressed_sha256",
        "compression",
        "dtype",
        "raw_bytes",
        "raw_sha256",
    }
    chunks = value.get("base64_chunks_76")
    if (
        set(value) != expected_keys
        or value.get("dtype") != "<f4"
        or value.get("compression") != "XZ_preset9_extreme_CRC64"
        or type(value.get("compressed_bytes")) is not int
        or value.get("compressed_bytes") != ACTOR_COMPRESSED_BYTES
        or value.get("compressed_sha256") != ACTOR_COMPRESSED_SHA256
        or type(value.get("raw_bytes")) is not int
        or value.get("raw_bytes") != ACTOR_RAW_BYTES
        or value.get("raw_sha256") != ACTOR_RAW_SHA256
        or not isinstance(chunks, list)
        or len(chunks) != ACTOR_BASE64_CHUNKS
        or any(not isinstance(part, str) or not part.isascii() for part in chunks)
        or any(len(part) != 76 for part in chunks[:-1])
        or len(chunks[-1]) != 36
    ):
        raise ProtocolError("actor payload format drift")
    encoded = "".join(chunks)
    compressed = base64.b64decode(encoded, validate=True)
    if (
        base64.standard_b64encode(compressed).decode("ascii") != encoded
        or len(compressed) != ACTOR_COMPRESSED_BYTES
        or sha256_bytes(compressed) != ACTOR_COMPRESSED_SHA256
    ):
        raise ProtocolError("compressed actor payload drift")
    if not lzma.is_check_supported(lzma.CHECK_CRC64):
        raise ProtocolError("XZ CRC64 is unsupported")
    raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    if (
        len(raw) != ACTOR_RAW_BYTES
        or sha256_bytes(raw) != ACTOR_RAW_SHA256
    ):
        raise ProtocolError("decoded actor payload drift")
    return raw


def validate_cw23(document: Mapping[str, Any]) -> Mapping[str, Any]:
    endpoint = document.get("endpoint", {})
    trial = endpoint.get("trial", {})
    payload = endpoint.get("candidate_payload")
    reconstruction = endpoint.get("pure_payload_reconstruction", {})
    checks = {
        "decision_GO": document.get("decision")
        == document.get("status")
        == endpoint.get("decision")
        == "GO_CW23_BOUNDARY_QP_PREFLIGHT",
        "trial_pass": trial.get("pass") is True,
        "candidate_hash_exact": trial.get("candidate_model_state_sha256")
        == reconstruction.get("candidate_model_state_sha256")
        == CW23_MODEL_SHA256,
        "payload_present": isinstance(payload, Mapping),
        "final_checks_true": isinstance(endpoint.get("final_checks"), Mapping)
        and bool(endpoint.get("final_checks"))
        and all(value is True for value in endpoint["final_checks"].values()),
        "reconstruction_pass": reconstruction.get("pass") is True
        and isinstance(reconstruction.get("checks"), Mapping)
        and all(value is True for value in reconstruction["checks"].values()),
        "base_exact_CW11": document.get("base", {}).get("model_state_sha256")
        == CW11_MODEL_SHA256,
        "outer_restore_pass": endpoint.get("outer_restore_pass") is True
        and isinstance(endpoint.get("outer_restore_checks"), Mapping)
        and bool(endpoint.get("outer_restore_checks"))
        and all(value is True for value in endpoint["outer_restore_checks"].values()),
        "CW11_eval_only_not_opened": document.get("base", {}).get(
            "materialized_eval_only_CW11_opened"
        )
        is False
        and document.get("base", {}).get("resume_forbidden_respected") is True,
        "zero_budget": type(
            document.get("official_unique_changed_candidate_count_consumed")
        )
        is int
        and document.get("official_unique_changed_candidate_count_consumed") == 0,
        "no_submission_package": document.get("submission_performed") is False
        and document.get("package_upload_performed") is False,
    }
    if not all(checks.values()) or not isinstance(payload, Mapping):
        raise ProtocolError(f"CW23 candidate contract drift: {checks}")
    anchor = payload.get("anchor", {})
    payload_checks = {
        "payload_canonical_sha": sha256_bytes(canonical_json(payload))
        == CW23_PAYLOAD_CANONICAL_SHA256,
        "raw_file": anchor.get("raw_checkpoint_file_sha256") == RAW_FILE_SHA256,
        "raw_model": anchor.get("raw_model_state_sha256") == RAW_MODEL_SHA256,
        "CW11_provenance": anchor.get("CW11_provenance_model_state_sha256")
        == CW11_MODEL_SHA256,
        "CW11_eval_only_false": anchor.get(
            "CW11_materialized_eval_only_checkpoint_used"
        )
        is False,
        "terminal": anchor.get("terminal_model_state_sha256")
        == CW23_MODEL_SHA256,
        "actor_names": payload.get("actor_names") == list(ACTOR6_NAMES),
        "changed_actor_names": payload.get("changed_actor_names")
        == list(ACTOR6_NAMES),
        "layout_list": isinstance(payload.get("actor_layout"), list),
        "layout_sha": payload.get("actor_layout_sha256") == ACTOR_LAYOUT_SHA256
        and sha256_bytes(
            json.dumps(
                payload.get("actor_layout"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        == ACTOR_LAYOUT_SHA256,
        "actor_payload_mapping": isinstance(
            payload.get("candidate_actor_float32_le"), Mapping
        ),
        "additional_l2": math.isclose(
            float(payload.get("actual_additional_from_CW11_l2", math.inf)),
            0.0009999992194425359,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
    }
    if not all(payload_checks.values()):
        raise ProtocolError(f"CW23 payload anchor drift: {payload_checks}")
    return payload


def reconstruct_states(
    candidate_payload: Mapping[str, Any],
    raw_payload: bytes,
    formal: ModuleType,
    cw20: ModuleType,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    import numpy as np

    helper = formal.load_helper()
    torch = helper.torch
    checkpoint = helper.verify_parent_checkpoint(raw_payload)
    raw_state = checkpoint.get("model_state_dict")
    if (
        checkpoint.get("update") != 468
        or helper.evaluator.checkpoint_kind(checkpoint) != "ppo"
        or not isinstance(checkpoint.get("model_config"), dict)
        or not isinstance(raw_state, Mapping)
        or helper.model_state_sha256(raw_state) != RAW_MODEL_SHA256
    ):
        raise ProtocolError("raw checkpoint contract drift")
    raw = {name: tensor.detach().cpu().clone() for name, tensor in raw_state.items()}
    candidate = {name: tensor.detach().cpu().clone() for name, tensor in raw.items()}
    layout = candidate_payload.get("actor_layout")
    if not isinstance(layout, list) or len(layout) != len(ACTOR6_NAMES):
        raise ProtocolError("actor layout is absent")
    if any(
        not isinstance(record, dict)
        or set(record) != {"dtype", "name", "numel", "shape", "start", "stop"}
        for record in layout
    ):
        raise ProtocolError("actor layout record drift")
    actor_value = candidate_payload.get("candidate_actor_float32_le")
    if not isinstance(actor_value, Mapping):
        raise ProtocolError("actor payload is absent")
    decoded = decode_actor(actor_value)
    decoded_array = np.frombuffer(decoded, dtype="<f4")
    if decoded_array.shape != (ACTOR6_DIMENSION,) or not bool(
        np.isfinite(decoded_array).all()
    ):
        raise ProtocolError("decoded actor vector is nonfinite or malformed")
    cw20.copy_actor_bytes(candidate, ACTOR6_NAMES, layout, decoded, np, torch)
    reconstructed_actor = cw20.actor_bytes(candidate, ACTOR6_NAMES, np)
    changed = sorted(
        name for name in raw if not torch.equal(raw[name], candidate[name])
    )
    nonactor_names = sorted(set(raw).difference(ACTOR6_NAMES))
    raw_nonactor = helper.model_state_sha256(
        {name: raw[name] for name in nonactor_names}
    )
    candidate_nonactor = helper.model_state_sha256(
        {name: candidate[name] for name in nonactor_names}
    )
    candidate_hash = helper.model_state_sha256(candidate)
    helper_actor_audit = helper.verify_actor6_only(raw, candidate, "CW23 candidate")
    checks = {
        "raw_hash_exact": helper.model_state_sha256(raw) == RAW_MODEL_SHA256,
        "candidate_hash_exact": candidate_hash == CW23_MODEL_SHA256,
        "actor_payload_bytes_exact": reconstructed_actor == decoded,
        "changed_scope_exact_actor6": changed == sorted(ACTOR6_NAMES),
        "raw_nonactor_exact": raw_nonactor == RAW_NONACTOR_SHA256,
        "candidate_nonactor_exact": candidate_nonactor == RAW_NONACTOR_SHA256,
        "state_key_count_80": len(raw) == len(candidate) == 80,
        "all_tensors_finite": all(
            bool(torch.isfinite(value).all()) for value in candidate.values()
        ),
        "helper_actor6_audit_pass": helper_actor_audit.get(
            "changed_parameter_set_exact_actor6"
        )
        is True
        and helper_actor_audit.get("all_six_actor_parameters_changed") is True,
    }
    if not all(checks.values()):
        raise ProtocolError(f"absolute payload reconstruction failed: {checks}")
    return {"raw": raw, "candidate": candidate}, checkpoint, {
        "checks": checks,
        "candidate_model_state_sha256": candidate_hash,
        "changed_parameter_names": changed,
        "actor_float32_le_sha256": sha256_bytes(reconstructed_actor),
        "raw_nonactor_sha256": raw_nonactor,
        "candidate_nonactor_sha256": candidate_nonactor,
        "helper_actor6_audit": helper_actor_audit,
        "CW11_materialized_eval_only_checkpoint_opened": False,
    }


def execution_checks(
    panel_results: Mapping[str, Any], execution: Mapping[str, Any], restore: Mapping[str, Any]
) -> dict[str, bool]:
    expected_ledger = [
        {"ordinal": ordinal, "model": model, "panel": panel}
        for ordinal, (model, panel) in enumerate(
            (
                (model, panel)
                for model in ("raw", "candidate")
                for panel in ("flg", "pokemonfan", "core5")
            ),
            start=1,
        )
    ]
    observed_ledger = [
        {key: item[key] for key in ("ordinal", "model", "panel")}
        for item in execution.get("completion_ledger", [])
    ]
    dataset_audits = execution.get("dataset_audits", {})
    return {
        "device_exact_cuda0": execution.get("device") == "cuda:0",
        "workers_exact0": type(execution.get("workers")) is int
        and execution.get("workers") == 0,
        "model_kind_ppo": execution.get("model_kind") == "ppo",
        "one_model_instance": type(execution.get("model_instance_count")) is int
        and execution.get("model_instance_count") == 1,
        "all_live_actor_overlays_verified": execution.get(
            "all_ten_live_actor_overlays_verified_after_copy"
        )
        is True,
        "same_batch_raw_and_candidate": execution.get("same_batch_all_ten_states")
        is True,
        "batch_size_exact256": int(execution.get("batch_size", -1)) == 256,
        "evaluation_count_exact6": int(execution.get("evaluation_count_exact", -1))
        == 6,
        "ledger_raw_then_candidate_three_panels": observed_ledger
        == expected_ledger,
        "all_panel_rows_exact": all(
            int(panel_results[model][panel]["rows"]) == EXPECTED_ROWS[panel]
            for model in ("raw", "candidate")
            for panel in ("flg", "pokemonfan", "core5")
        ),
        "total_unique_train_rows24050": sum(EXPECTED_ROWS.values()) == 24050,
        "validation_members_not_opened": execution.get(
            "validation_member_payloads_opened"
        )
        is False,
        "dataset_audits_exact_train_only": isinstance(dataset_audits, Mapping)
        and set(dataset_audits) == set(EXPECTED_ROWS)
        and all(
            isinstance(dataset_audits[panel], Mapping)
            and int(dataset_audits[panel].get("rows", -1)) == EXPECTED_ROWS[panel]
            and dataset_audits[panel].get("workers") == 0
            and dataset_audits[panel].get("validation_member_payloads_opened")
            is False
            and dataset_audits[panel].get(
                "identity_feature_synchronization_exact"
            )
            is True
            and all(
                isinstance(name, str) and name.startswith("train/")
                for name in dataset_audits[panel].get("train_members", [])
            )
            for panel in EXPECTED_ROWS
        ),
        "evaluator_finally_raw_restore": restore.get("finally_raw_restore_pass")
        is True,
        "evaluator_final_raw_hash_exact": restore.get(
            "final_raw_model_state_sha256"
        )
        == RAW_MODEL_SHA256,
    }


def production_run(
    expected_self_sha256: str, execution_binding: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = validate_runtime(require_cuda=True)
    source, self_record = read_regular(
        SCRIPT, expected_self_sha256, FROZEN_MODE, "self"
    )
    source_record = source_audit(source, require_frozen=True)
    cw23_bytes, cw23_record = read_regular(CW23, CW23_SHA256, EVIDENCE_MODE, "CW23 result")
    raw_bytes, raw_record = read_regular(RAW, RAW_FILE_SHA256, 0o664, "raw U468")
    fulltrain, fulltrain_record = import_frozen(FULLTRAIN, FULLTRAIN_SHA256, "fulltrain_v3")
    formal, formal_record = import_frozen(FORMAL, FORMAL_SHA256, "formal_v3")
    cw20, cw20_record = import_frozen(CW20, CW20_SHA256, "cw20_payload_codec")
    formal.load_module = exact_nested_module_loader
    candidate_document = strict_json(cw23_bytes, "CW23 result")
    candidate_payload = validate_cw23(candidate_document)
    states, checkpoint, reconstruction = reconstruct_states(
        candidate_payload, raw_bytes, formal, cw20
    )
    state_hash_helper = formal.load_helper()
    if int(formal.SEED) != 202608121 or not callable(formal.cuda_runtime):
        raise ProtocolError("formal evaluator seed/runtime contract drift")
    formal_runtime = formal.cuda_runtime(state_hash_helper)
    formal_runtime_checks = {
        "seed_exact": formal_runtime.get("seed") == formal.SEED == 202608121,
        "device_exact": formal_runtime.get("device") == "cuda:0",
        "bf16_supported": formal_runtime.get("bf16_supported") is True,
        "deterministic_algorithms": formal_runtime.get(
            "deterministic_algorithms"
        )
        is True,
        "cudnn_benchmark_false": formal_runtime.get("cudnn_benchmark") is False,
        "cudnn_deterministic_true": formal_runtime.get("cudnn_deterministic")
        is True,
        "float32_matmul_precision_high": formal_runtime.get(
            "float32_matmul_precision"
        )
        == "high",
        "cublas_workspace_exact": formal_runtime.get("cublas_workspace_config")
        == ":4096:8",
    }
    if not all(formal_runtime_checks.values()):
        raise ProtocolError(f"formal CUDA runtime drift: {formal_runtime_checks}")
    random.seed(formal.SEED)
    state_hash_helper.torch.manual_seed(formal.SEED)
    state_hash_helper.torch.cuda.manual_seed_all(formal.SEED)
    before_hashes = {
        name: state_hash_helper.model_state_sha256(state)
        for name, state in states.items()
    }
    design, archives, fulltrain_inputs, frozen_summaries = fulltrain.load_fulltrain_inputs(
        formal
    )
    formal_globals = {
        "MODEL_ORDER": formal.MODEL_ORDER,
        "ALPHAS": formal.ALPHAS,
        "copy_actor_state_to_model": formal.copy_actor_state_to_model,
    }
    panel_results, evaluator_execution, evaluator_restore = fulltrain.evaluate_raw_candidate(
        formal,
        design,
        states,
        checkpoint,
        archives,
        frozen_summaries,
    )
    helper_after = formal.load_helper()
    after_hashes = {
        name: helper_after.model_state_sha256(state) for name, state in states.items()
    }
    checks = execution_checks(panel_results, evaluator_execution, evaluator_restore)
    checks.update(
        {
            "states_unchanged_during_evaluation": before_hashes == after_hashes,
            "raw_state_exact_after": after_hashes.get("raw") == RAW_MODEL_SHA256,
            "candidate_state_exact_after": after_hashes.get("candidate")
            == CW23_MODEL_SHA256,
            "formal_globals_restored": formal.MODEL_ORDER
            == formal_globals["MODEL_ORDER"]
            and formal.ALPHAS is formal_globals["ALPHAS"]
            and formal.copy_actor_state_to_model
            is formal_globals["copy_actor_state_to_model"],
            "deterministic_runtime_still_active": state_hash_helper.torch.are_deterministic_algorithms_enabled()
            and state_hash_helper.torch.backends.cudnn.benchmark is False
            and state_hash_helper.torch.backends.cudnn.deterministic is True
            and state_hash_helper.torch.get_float32_matmul_precision() == "high",
        }
    )
    if not all(checks.values()):
        raise ProtocolError(f"fulltrain execution integrity failed: {checks}")
    decision = fulltrain.decide_fulltrain(panel_results)
    _, self_after = read_regular(
        SCRIPT, expected_self_sha256, FROZEN_MODE, "self after fulltrain"
    )
    if self_after != self_record:
        raise ProtocolError("adapter source identity changed during fulltrain")
    outer_status = (
        "GO_CW23_FULLTRAIN_GATE"
        if decision.get("pass") is True
        else "NO_GO_CW23_FULLTRAIN_GATE"
    )
    result = {
        "schema_version": SCHEMA,
        "status": outer_status,
        "candidate_model_state_sha256": CW23_MODEL_SHA256,
        "inputs": {
            "self": {
                **self_record,
                "source_audit": source_record,
                "post_evaluation": self_after,
                "stable_during_fulltrain": True,
            },
            "hash_bound_execution": dict(execution_binding),
            "CW23": cw23_record,
            "raw_U468": raw_record,
            "fulltrain_v3": fulltrain_record,
            "formal_v3": formal_record,
            "CW20_payload_codec": cw20_record,
            "fulltrain_inputs": fulltrain_inputs,
        },
        "reconstruction": reconstruction,
        "execution": {
            "device": evaluator_execution.get("device"),
            "batch_size": evaluator_execution.get("batch_size"),
            "workers": evaluator_execution.get("workers"),
            "evaluation_count_exact": evaluator_execution.get(
                "evaluation_count_exact"
            ),
            "completion_ledger": evaluator_execution.get("completion_ledger"),
            "dataset_audits": evaluator_execution.get("dataset_audits"),
            "checks": checks,
            "fulltrain_evaluator_finally_raw_restore": evaluator_restore,
        },
        "evaluations": panel_results,
        "decision": decision,
        "scope_audit": {
            "all_24050_train_rows": True,
            "validation_members_opened": False,
            "validation_results_read": False,
            "test_rows_opened": False,
            "training_optimizer_backward": False,
            "hyperparameter_sweep": False,
            "candidate_RAM_only": True,
            "CW11_materialized_eval_only_checkpoint_opened": False,
            "checkpoint_writes": 0,
            "model_artifact_writes": 0,
            "network_upload_submission": False,
        },
        "runtime": {
            "adapter": runtime,
            "formal": formal_runtime,
            "formal_checks": formal_runtime_checks,
            "python_torch_cuda_seed": int(formal.SEED),
        },
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
        "package_upload_performed": False,
    }
    canonical_json(result)
    return result


def audit_only() -> dict[str, Any]:
    runtime = validate_runtime(require_cuda=False)
    source, source_record = read_regular(SCRIPT, None, FROZEN_MODE, "self")
    execution_binding = validate_execution_binding(source_record, True)
    cw23_bytes, cw23_record = read_regular(CW23, CW23_SHA256, EVIDENCE_MODE, "CW23 result")
    raw_bytes, raw_record = read_regular(RAW, RAW_FILE_SHA256, 0o664, "raw U468")
    del raw_bytes
    fulltrain, fulltrain_record = import_frozen(FULLTRAIN, FULLTRAIN_SHA256, "fulltrain_v3_static")
    formal, formal_record = import_frozen(FORMAL, FORMAL_SHA256, "formal_v3_static")
    cw20, cw20_record = import_frozen(CW20, CW20_SHA256, "cw20_static")
    formal.load_module = exact_nested_module_loader
    payload = validate_cw23(strict_json(cw23_bytes, "CW23 result"))
    interface_checks = {
        "fulltrain_load_inputs": callable(fulltrain.load_fulltrain_inputs),
        "fulltrain_evaluate": callable(fulltrain.evaluate_raw_candidate),
        "fulltrain_decide": callable(fulltrain.decide_fulltrain),
        "formal_helper": callable(formal.load_helper),
        "CW20_copy_actor_bytes": callable(cw20.copy_actor_bytes),
        "actor_names_exact": payload.get("actor_names") == list(ACTOR6_NAMES),
    }
    if not all(interface_checks.values()) or lexists(ATTEMPT) or lexists(OUTPUT):
        raise ProtocolError(
            f"static contract failed: {interface_checks}, "
            f"attempt={lexists(ATTEMPT)}, output={lexists(OUTPUT)}"
        )
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_pass",
        "runtime": runtime,
        "source": {**source_record, "audit": source_audit(source, True)},
        "hash_bound_execution": execution_binding,
        "inputs": {
            "CW23": cw23_record,
            "raw_U468": raw_record,
            "fulltrain_v3": fulltrain_record,
            "formal_v3": formal_record,
            "CW20": cw20_record,
        },
        "interfaces": interface_checks,
        "attempt_absent": True,
        "output_absent": True,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(canonical_json(audit_only()).decode("utf-8"), end="")
        return
    validate_runtime(require_cuda=False)
    if lexists(ATTEMPT) or lexists(OUTPUT):
        raise ProtocolError("fulltrain one-shot target already exists")
    source, source_record = read_regular(SCRIPT, None, FROZEN_MODE, "self")
    source_audit(source, require_frozen=True)
    execution_binding = validate_execution_binding(source_record, False)
    marker = {
        "schema_version": SCHEMA,
        "status": "attempt_committed_before_CUDA",
        "source_sha256": sha256_bytes(source),
        "source": source_record,
        "hash_bound_execution": execution_binding,
        "CW23_sha256": CW23_SHA256,
        "candidate_model_state_sha256": CW23_MODEL_SHA256,
        "raw_checkpoint_file_sha256": RAW_FILE_SHA256,
        "fulltrain_v3_sha256": FULLTRAIN_SHA256,
        "formal_v3_sha256": FORMAL_SHA256,
        "scope": "raw_plus_candidate_all_24050_train_rows_only",
        "official_unique_changed_candidate_count_consumed_before": 0,
        "submission_authorized": False,
    }
    attempt_record = publish(ATTEMPT, canonical_json(marker))
    try:
        result = production_run(sha256_bytes(source), execution_binding)
    except BaseException as error:
        failure = {
            "schema_version": SCHEMA,
            "status": "ERROR_CW23_FULLTRAIN_GATE",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "attempt": attempt_record,
            "candidate_model_state_sha256": CW23_MODEL_SHA256,
            "official_unique_changed_candidate_count_consumed": 0,
            "submission_performed": False,
            "package_upload_performed": False,
        }
        publication = publish(OUTPUT, canonical_json(failure))
        print(
            canonical_json(
                {
                    "schema_version": SCHEMA,
                    "status": failure["status"],
                    "output": publication,
                }
            ).decode("utf-8"),
            end="",
        )
        raise
    result["attempt"] = attempt_record
    payload = canonical_json(result)
    publication = publish(OUTPUT, payload)
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "pass": result["decision"]["pass"],
                "candidate_model_state_sha256": CW23_MODEL_SHA256,
                "output": publication,
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
            }
        ).decode("utf-8"),
        end="",
    )


if __name__ == "__main__":
    main()

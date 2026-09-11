#!/usr/bin/env python3
"""Run one hash-bound E904 endpoint on the six frozen specialist panels.

This is a local, one-shot evaluation executor.  It does not train, run broad
behavior, run Gold, access the network, package a model, upload, or submit.
Every evaluator command is supplied by and checked against one byte-hash-bound
preregistration.  A formal attempt marker is consumed before the first panel;
all six panels are then attempted exactly once in their declared order, even if
an earlier panel fails.

The decision is deliberately narrow: value and action-count correctness must
remain exactly equal to the frozen raw-U468 parent on every panel; the five
non-Pokemon-Fan panels may not regress on any other frozen metric; and the
Pokemon-Fan panel must gain at least +3/+3/+5/+3 on set/hybrid/ordered/top1
while retaining both context-34 ordering counts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
SCRIPT = ROOT / "tools/run_e904_single_endpoint_specialist.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
PREREGISTRATION_SCHEMA = "ptcg-e904-single-endpoint-specialist-preregistration-v2"
MANIFEST_SCHEMA = "ptcg-e904-single-endpoint-specialist-manifest-v2"
DECISION_SCHEMA = "ptcg-e904-single-endpoint-specialist-decision-v2"

E904_MODEL_STATE_SHA256 = (
    "e904efb322c15ca4bee62573f4f439309d1d4ecb3ae62620d13fd70e8acaa91f"
)
EXPECTED_CHECKPOINT_UPDATE = 468
RAW_U468_CHECKPOINT = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
RAW_U468_CHECKPOINT_SHA256 = (
    "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
)
RAW_U468_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
EXPECTED_EVALUATOR = ROOT / "tools/evaluate_policy_bc.py"
EXPECTED_EVALUATOR_SHA256 = (
    "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4"
)
EXPECTED_DEPENDENCIES = {
    "train_bc_orbit": (
        ROOT / "tools/train_bc_orbit.py",
        "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
    ),
    "train_ppo": (
        ROOT / "tools/train_ppo.py",
        "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3",
    ),
}

SHARED_PROTOCOL = {
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

PANEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "pokemonfan",
        "data_name": "pokemonfan",
        "data_path": "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
        "data_sha256": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
        "team_name": None,
        "baseline_path": "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/specialist_behavior/parent/pokemonfan.json",
        "baseline_sha256": "525881ac4ca154c0814ceacbb9eac666e1a853a89d0107d5d29a7fb934bb798c",
    },
    {
        "name": "flg",
        "data_name": "flg",
        "data_path": "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
        "data_sha256": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
        "team_name": None,
        "baseline_path": "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/specialist_behavior/parent/flg.json",
        "baseline_sha256": "93c86e6e8a0c7adbc908b8ce92249b8920812163899b28914e4cd9cbc76186a8",
    },
    {
        "name": "core5",
        "data_name": "core5",
        "data_path": "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
        "data_sha256": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
        "team_name": None,
        "baseline_path": "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/specialist_behavior/parent/core5.json",
        "baseline_sha256": "f31c70ef1cb15f3a34adf5267bed975d460f49b513e182dc24b679947880208e",
    },
    {
        "name": "dominic",
        "data_name": "core5",
        "data_path": "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
        "data_sha256": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
        "team_name": "Dominic Peel",
        "baseline_path": "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/specialist_behavior/parent/dominic.json",
        "baseline_sha256": "3584fc418fcbbd5eb0d84b73840622bb77219b9bb53dce0b7c8b3f71c37d191a",
    },
    {
        "name": "luca",
        "data_name": "core5",
        "data_path": "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
        "data_sha256": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
        "team_name": "Luca",
        "baseline_path": "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/specialist_behavior/parent/luca.json",
        "baseline_sha256": "8174b814f337e7c0a4741b9f6a38865f2dc2c42cbba7bdffede3699aa909d7f2",
    },
    {
        "name": "szlach",
        "data_name": "core5",
        "data_path": "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
        "data_sha256": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
        "team_name": "szlachetny snieg",
        "baseline_path": "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/specialist_behavior/parent/szlach.json",
        "baseline_sha256": "251f7494e7c9ba193a90f341c55af330c9e087b4e7a99a9ecf936b24fd5d075e",
    },
)

BASELINE_METRICS: dict[str, dict[str, int]] = {
    "pokemonfan": {
        "rows": 15152,
        "context34_rows": 58,
        "set": 13017,
        "hybrid": 13016,
        "ordered": 12864,
        "value": 11033,
        "count": 15022,
        "top1": 13116,
        "context34_hybrid": 57,
        "context34_ordered": 57,
    },
    "flg": {
        "rows": 2312,
        "context34_rows": 3,
        "set": 1761,
        "hybrid": 1748,
        "ordered": 1731,
        "value": 1807,
        "count": 2299,
        "top1": 1775,
        "context34_hybrid": 3,
        "context34_ordered": 3,
    },
    "core5": {
        "rows": 8319,
        "context34_rows": 43,
        "set": 6550,
        "hybrid": 6534,
        "ordered": 6505,
        "value": 6127,
        "count": 8210,
        "top1": 6653,
        "context34_hybrid": 41,
        "context34_ordered": 41,
    },
    "dominic": {
        "rows": 2623,
        "context34_rows": 15,
        "set": 1981,
        "hybrid": 1974,
        "ordered": 1973,
        "value": 1940,
        "count": 2579,
        "top1": 2019,
        "context34_hybrid": 13,
        "context34_ordered": 13,
    },
    "luca": {
        "rows": 1378,
        "context34_rows": 7,
        "set": 1069,
        "hybrid": 1069,
        "ordered": 1059,
        "value": 960,
        "count": 1377,
        "top1": 1086,
        "context34_hybrid": 7,
        "context34_ordered": 7,
    },
    "szlach": {
        "rows": 2567,
        "context34_rows": 7,
        "set": 2071,
        "hybrid": 2071,
        "ordered": 2062,
        "value": 1906,
        "count": 2510,
        "top1": 2109,
        "context34_hybrid": 7,
        "context34_ordered": 7,
    },
}

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


class ProtocolError(RuntimeError):
    """The preregistration or a bound artifact violates the frozen protocol."""


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
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProtocolError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def root_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{label} must be a non-empty root-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError(f"{label} must be a safe root-relative path")
    resolved = Path(os.path.normpath(os.fspath((ROOT / relative).absolute())))
    try:
        resolved.relative_to(ROOT.absolute())
    except ValueError as error:
        raise ProtocolError(f"{label} resolves outside the repository") from error
    return resolved


def root_relative_text(path: Path) -> str:
    normalized = Path(os.path.normpath(os.fspath(path.absolute())))
    try:
        return str(normalized.relative_to(ROOT.absolute()))
    except ValueError as error:
        raise ProtocolError(f"path is outside the repository: {path}") from error


def read_plain_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProtocolError(f"cannot open {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ProtocolError(f"{label} must be a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if not stable or len(payload) != after.st_size:
            raise ProtocolError(f"{label} changed while it was read")
        path_stat = os.stat(path, follow_symlinks=False)
        if (path_stat.st_dev, path_stat.st_ino) != (after.st_dev, after.st_ino):
            raise ProtocolError(f"{label} path changed while it was read")
        return payload, {
            "path": root_relative_text(path),
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
            "nlink": int(after.st_nlink),
        }
    finally:
        os.close(descriptor)


def verify_binding(
    value: Any,
    label: str,
    *,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[Path, bytes, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} binding must be an object")
    path = root_relative_path(value.get("path"), f"{label}.path")
    digest = require_sha256(value.get("sha256"), f"{label}.sha256")
    if expected_path is not None and path != Path(
        os.path.normpath(os.fspath(expected_path.absolute()))
    ):
        raise ProtocolError(f"{label} path differs from the frozen path")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProtocolError(f"{label} digest differs from the frozen digest")
    payload, evidence = read_plain_file(path, label)
    if evidence["sha256"] != digest:
        raise ProtocolError(f"{label} SHA-256 mismatch")
    return path, payload, evidence


def model_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ProtocolError("checkpoint model_state_dict must map names to tensors")
        value = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ProtocolError(f"checkpoint tensor is non-finite: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_checkpoint_identity(
    payload: bytes,
    binding: Mapping[str, Any],
    *,
    expected_model_state_sha256: str = E904_MODEL_STATE_SHA256,
    label: str = "candidate",
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise ProtocolError(f"{label} checkpoint cannot be loaded on CPU") from error
    if not isinstance(checkpoint, dict):
        raise ProtocolError(f"{label} checkpoint must contain a dict")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ProtocolError(f"{label} checkpoint has no model_state_dict mapping")
    observed = model_state_sha256(state)
    declared = require_sha256(
        binding.get("model_state_sha256"),
        f"{label}.model_state_sha256",
    )
    if (
        declared != expected_model_state_sha256
        or observed != expected_model_state_sha256
    ):
        raise ProtocolError(
            f"{label} is not the frozen expected runtime model state"
        )
    update = checkpoint.get("update")
    if type(update) is not int or update != EXPECTED_CHECKPOINT_UPDATE:
        raise ProtocolError(f"{label} checkpoint update must remain exactly 468")
    if binding.get("checkpoint_update") != EXPECTED_CHECKPOINT_UPDATE:
        raise ProtocolError(f"{label} preregistration update must be exactly 468")
    return {
        "model_state_sha256": observed,
        "checkpoint_update": update,
        "model_tensor_count": len(state),
    }


def metric_value(document: Mapping[str, Any], metric: str) -> int:
    current: Any = document
    for key in METRIC_PATHS[metric]:
        if not isinstance(current, Mapping) or key not in current:
            raise ProtocolError(f"output lacks metric path {'.'.join(METRIC_PATHS[metric])}")
        current = current[key]
    if type(current) is not int:
        raise ProtocolError(f"output metric {metric} must be an integer")
    return current


def observed_metrics(document: Mapping[str, Any]) -> dict[str, int]:
    return {name: metric_value(document, name) for name in METRIC_PATHS}


def expected_thresholds(panel_name: str) -> dict[str, dict[str, int]]:
    parent = BASELINE_METRICS[panel_name]
    thresholds: dict[str, dict[str, int]] = {
        "rows": {"exact": parent["rows"]},
        "context34_rows": {"exact": parent["context34_rows"]},
        "value": {"exact": parent["value"]},
        "count": {"exact": parent["count"]},
    }
    gains = {
        "set": 3,
        "hybrid": 3,
        "ordered": 5,
        "top1": 3,
    }
    for metric in ("set", "hybrid", "ordered", "top1"):
        increment = gains[metric] if panel_name == "pokemonfan" else 0
        thresholds[metric] = {"minimum": parent[metric] + increment}
    for metric in ("context34_hybrid", "context34_ordered"):
        thresholds[metric] = {"minimum": parent[metric]}
    return thresholds


def expected_command(
    checkpoint: Path,
    panel: Mapping[str, Any],
    output: Path,
) -> list[str]:
    command = [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        root_relative_text(EXPECTED_EVALUATOR),
        "--checkpoint",
        root_relative_text(checkpoint),
        "--data",
        str(panel["data_path"]),
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
        root_relative_text(output),
    ]
    team_name = panel.get("team_name")
    if team_name is not None:
        command.extend(["--team-name", str(team_name)])
    return command


def atomic_create(path: Path, payload: bytes, mode: int = 0o444) -> None:
    if path.parent != ROOT / "artifacts" and not path.parent.is_dir():
        raise ProtocolError(f"output parent does not exist: {path.parent}")
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
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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


def rehash_bound_inputs(
    bindings: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    raise_on_failure: bool,
) -> dict[str, Any]:
    """Reopen every immutable input and compare it with its frozen digest."""
    records: list[dict[str, Any]] = []
    for binding in bindings:
        label = str(binding["label"])
        path = binding["path"]
        expected = str(binding["sha256"])
        record: dict[str, Any] = {
            "label": label,
            "path": root_relative_text(path),
            "expected_sha256": expected,
        }
        try:
            _, observed = read_plain_file(path, f"{phase} {label}")
            record["observed_sha256"] = observed["sha256"]
            record["bytes"] = observed["bytes"]
            record["pass"] = observed["sha256"] == expected
            record["error"] = None
        except BaseException as error:
            record["observed_sha256"] = None
            record["bytes"] = None
            record["pass"] = False
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    result = {
        "phase": phase,
        "binding_count": len(records),
        "pass": len(records) == len(bindings)
        and all(bool(record["pass"]) for record in records),
        "records": records,
    }
    if raise_on_failure and not result["pass"]:
        failures = [record["label"] for record in records if not record["pass"]]
        raise ProtocolError(
            f"{phase} immutable-input rehash failed: {', '.join(failures)}"
        )
    return result


def validate_output_identity(
    document: Mapping[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    panel: Mapping[str, Any],
) -> None:
    if Path(str(document.get("checkpoint"))).resolve() != checkpoint:
        raise ProtocolError("evaluator output checkpoint path mismatch")
    if document.get("checkpoint_sha256") != checkpoint_sha256:
        raise ProtocolError("evaluator output checkpoint SHA-256 mismatch")
    if document.get("checkpoint_update") != EXPECTED_CHECKPOINT_UPDATE:
        raise ProtocolError("evaluator output checkpoint update mismatch")
    if Path(str(document.get("data"))).resolve() != (ROOT / str(panel["data_path"])).resolve():
        raise ProtocolError("evaluator output data path mismatch")
    expected_scalars = {
        "split": "valid",
        "split_mode": "archive",
        "split_seed": 20260723,
        "device": "cuda",
        "prediction_order": "policy_greedy",
        "max_rows": None,
        "evaluator": "tools/evaluate_policy_bc.py",
    }
    for key, expected in expected_scalars.items():
        if document.get(key) != expected:
            raise ProtocolError(f"evaluator output field {key} mismatch")
    expected_teams = [] if panel.get("team_name") is None else [panel["team_name"]]
    filters = document.get("filters")
    if not isinstance(filters, Mapping):
        raise ProtocolError("evaluator output filters must be an object")
    if filters.get("deck_hashes") != [] or filters.get("team_names") != expected_teams:
        raise ProtocolError("evaluator output filters mismatch")


def validate_preregistration(
    preregistration_path: Path,
    expected_preregistration_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg_payload, prereg_evidence = read_plain_file(
        preregistration_path,
        "preregistration",
    )
    expected_digest = require_sha256(
        expected_preregistration_sha256,
        "expected preregistration SHA-256",
    )
    if prereg_evidence["sha256"] != expected_digest:
        raise ProtocolError("preregistration SHA-256 mismatch")
    prereg = strict_json_loads(prereg_payload, "preregistration")
    if prereg.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise ProtocolError("unsupported preregistration schema")
    if prereg.get("status") != "locked_before_specialist_evaluation":
        raise ProtocolError("preregistration is not formally locked")

    executor = prereg.get("executor")
    if not isinstance(executor, Mapping):
        raise ProtocolError("executor binding must be an object")
    _, _, executor_evidence = verify_binding(
        executor,
        "executor",
        expected_path=SCRIPT,
    )
    if executor.get("python") != str(EXPECTED_PYTHON):
        raise ProtocolError("executor Python binding mismatch")
    if executor.get("flags") != ["-I", "-B"]:
        raise ProtocolError("executor flags must be exactly [-I, -B]")

    evaluator = prereg.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ProtocolError("evaluator binding must be an object")
    evaluator_path, _, evaluator_evidence = verify_binding(
        evaluator,
        "evaluator",
        expected_path=EXPECTED_EVALUATOR,
        expected_sha256=EXPECTED_EVALUATOR_SHA256,
    )
    if evaluator.get("python") != str(EXPECTED_PYTHON):
        raise ProtocolError("evaluator Python binding mismatch")
    if evaluator.get("flags") != ["-I", "-B"]:
        raise ProtocolError("evaluator flags must be exactly [-I, -B]")

    dependencies = prereg.get("evaluator_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(EXPECTED_DEPENDENCIES):
        raise ProtocolError("evaluator dependency set mismatch")
    dependency_evidence: dict[str, Any] = {}
    dependency_paths: dict[str, Path] = {}
    for name, (path, digest) in EXPECTED_DEPENDENCIES.items():
        dependency_path, _, observed_dependency = verify_binding(
            dependencies[name],
            f"evaluator dependency {name}",
            expected_path=path,
            expected_sha256=digest,
        )
        dependency_paths[name] = dependency_path
        dependency_evidence[name] = observed_dependency

    candidate = prereg.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ProtocolError("candidate binding must be an object")
    if set(candidate) != {
        "path",
        "sha256",
        "model_state_sha256",
        "checkpoint_update",
    }:
        raise ProtocolError("candidate binding has an unexpected field set")
    checkpoint, checkpoint_payload, checkpoint_evidence = verify_binding(
        candidate,
        "candidate checkpoint",
    )
    checkpoint_identity = load_checkpoint_identity(checkpoint_payload, candidate)
    checkpoint_evidence.update(checkpoint_identity)

    raw_parent = prereg.get("raw_u468_parent")
    expected_raw_parent = {
        "path": root_relative_text(RAW_U468_CHECKPOINT),
        "sha256": RAW_U468_CHECKPOINT_SHA256,
        "model_state_sha256": RAW_U468_MODEL_STATE_SHA256,
        "checkpoint_update": EXPECTED_CHECKPOINT_UPDATE,
    }
    if not isinstance(raw_parent, Mapping) or dict(raw_parent) != expected_raw_parent:
        raise ProtocolError("raw U468 parent binding differs from the frozen parent")
    raw_parent_path, raw_parent_payload, raw_parent_evidence = verify_binding(
        raw_parent,
        "raw U468 parent checkpoint",
        expected_path=RAW_U468_CHECKPOINT,
        expected_sha256=RAW_U468_CHECKPOINT_SHA256,
    )
    raw_parent_evidence.update(
        load_checkpoint_identity(
            raw_parent_payload,
            raw_parent,
            expected_model_state_sha256=RAW_U468_MODEL_STATE_SHA256,
            label="raw U468 parent",
        )
    )

    if prereg.get("shared_protocol") != SHARED_PROTOCOL:
        raise ProtocolError("shared specialist protocol mismatch")
    if prereg.get("authoritative_panel_gates") != {
        name: expected_thresholds(name) for name in BASELINE_METRICS
    }:
        raise ProtocolError("authoritative panel gates differ from code-level gates")

    data_bindings = prereg.get("data_bindings")
    baseline_bindings = prereg.get("baseline_bindings")
    if not isinstance(data_bindings, Mapping) or set(data_bindings) != {
        "pokemonfan",
        "flg",
        "core5",
    }:
        raise ProtocolError("specialist data binding set mismatch")
    if not isinstance(baseline_bindings, Mapping) or set(baseline_bindings) != {
        panel["name"] for panel in PANEL_SPECS
    }:
        raise ProtocolError("specialist baseline binding set mismatch")

    data_evidence: dict[str, Any] = {}
    baseline_evidence: dict[str, Any] = {}
    for panel in PANEL_SPECS:
        data_name = str(panel["data_name"])
        if data_name not in data_evidence:
            _, _, evidence = verify_binding(
                data_bindings[data_name],
                f"data {data_name}",
                expected_path=ROOT / str(panel["data_path"]),
                expected_sha256=str(panel["data_sha256"]),
            )
            data_evidence[data_name] = evidence
        baseline_path, baseline_payload, evidence = verify_binding(
            baseline_bindings[str(panel["name"])],
            f"baseline {panel['name']}",
            expected_path=ROOT / str(panel["baseline_path"]),
            expected_sha256=str(panel["baseline_sha256"]),
        )
        baseline = strict_json_loads(
            baseline_payload,
            f"baseline {panel['name']}",
        )
        validate_output_identity(
            baseline,
            checkpoint=raw_parent_path,
            checkpoint_sha256=RAW_U468_CHECKPOINT_SHA256,
            panel=panel,
        )
        if observed_metrics(baseline) != BASELINE_METRICS[str(panel["name"])]:
            raise ProtocolError(f"baseline metric drift for {panel['name']}")
        evidence["metrics"] = observed_metrics(baseline)
        evidence["resolved_path"] = str(baseline_path)
        baseline_evidence[str(panel["name"])] = evidence

    output_rule = prereg.get("output_rule")
    if not isinstance(output_rule, Mapping):
        raise ProtocolError("output_rule must be an object")
    output_root = root_relative_path(output_rule.get("root"), "output_rule.root")
    if output_root.parent != ROOT / "artifacts":
        raise ProtocolError("specialist output root must be directly under artifacts")
    marker = root_relative_path(
        output_rule.get("attempt_marker"),
        "output_rule.attempt_marker",
    )
    if marker.parent != ROOT / "artifacts":
        raise ProtocolError("attempt marker must be directly under artifacts")
    manifest_path = output_root / "specialist_execution_manifest.json"
    decision_path = output_root / "specialist_decision.json"
    expected_output_rule = {
        "root": root_relative_text(output_root),
        "root_absent_at_lock": True,
        "attempt_marker": root_relative_text(marker),
        "attempt_marker_absent_at_lock": True,
        "evaluation_count_exact": 6,
        "attempts_per_evaluation": 1,
        "retry_authorized": False,
        "run_all_before_decision": True,
        "manifest": "specialist_execution_manifest.json",
        "decision": "specialist_decision.json",
        "final_directory_mode": "0o500",
    }
    if output_rule != expected_output_rule:
        raise ProtocolError("output_rule differs from the one-shot frozen rule")

    evaluations = prereg.get("ordered_evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 6:
        raise ProtocolError("ordered_evaluations must contain exactly six panels")
    output_paths: list[Path] = []
    for order, (evaluation, panel) in enumerate(
        zip(evaluations, PANEL_SPECS, strict=True),
        start=1,
    ):
        if not isinstance(evaluation, Mapping):
            raise ProtocolError(f"evaluation {order} must be an object")
        output = output_root / f"{panel['name']}.json"
        command = expected_command(checkpoint, panel, output)
        expected = {
            "order": order,
            "panel": panel["name"],
            "team_name": panel["team_name"],
            "output": root_relative_text(output),
            "output_absent_at_lock": True,
            "command": command,
            "command_sha256": sha256_bytes(canonical_json_bytes(command)),
            "attempts_authorized": 1,
        }
        if dict(evaluation) != expected:
            raise ProtocolError(f"evaluation {order} differs from the frozen command")
        output_paths.append(output)

    scope = prereg.get("scope")
    expected_scope = {
        "local_only": True,
        "network": False,
        "training": False,
        "specialist_behavior": True,
        "broad": False,
        "gold": False,
        "package": False,
        "upload": False,
        "submission": False,
    }
    if scope != expected_scope:
        raise ProtocolError("scope differs from the local specialist-only boundary")

    immutable_bindings: list[dict[str, Any]] = [
        {
            "label": "preregistration",
            "path": preregistration_path,
            "sha256": prereg_evidence["sha256"],
        },
        {
            "label": "executor",
            "path": SCRIPT,
            "sha256": executor_evidence["sha256"],
        },
        {
            "label": "evaluator",
            "path": evaluator_path,
            "sha256": evaluator_evidence["sha256"],
        },
        {
            "label": "candidate checkpoint",
            "path": checkpoint,
            "sha256": checkpoint_evidence["sha256"],
        },
        {
            "label": "raw U468 parent checkpoint",
            "path": raw_parent_path,
            "sha256": raw_parent_evidence["sha256"],
        },
    ]
    immutable_bindings.extend(
        {
            "label": f"evaluator dependency {name}",
            "path": dependency_paths[name],
            "sha256": dependency_evidence[name]["sha256"],
        }
        for name in sorted(dependency_paths)
    )
    immutable_bindings.extend(
        {
            "label": f"data {name}",
            "path": ROOT / str(data_bindings[name]["path"]),
            "sha256": data_evidence[name]["sha256"],
        }
        for name in sorted(data_evidence)
    )
    immutable_bindings.extend(
        {
            "label": f"baseline {name}",
            "path": ROOT / str(baseline_bindings[name]["path"]),
            "sha256": baseline_evidence[name]["sha256"],
        }
        for name in sorted(baseline_evidence)
    )

    return prereg, {
        "preregistration": prereg_evidence,
        "executor": executor_evidence,
        "evaluator": evaluator_evidence,
        "evaluator_dependencies": dependency_evidence,
        "candidate": checkpoint_evidence,
        "raw_u468_parent": raw_parent_evidence,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": checkpoint_evidence["sha256"],
        "data": data_evidence,
        "baselines": baseline_evidence,
        "output_root": output_root,
        "marker": marker,
        "manifest_path": manifest_path,
        "decision_path": decision_path,
        "output_paths": output_paths,
        "immutable_bindings": immutable_bindings,
    }


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
        "kernel_result": observed,
        "cuda_runtime": torch.version.cuda,
    }


def evaluate_gates(panel_name: str, metrics: Mapping[str, int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for metric, comparison in expected_thresholds(panel_name).items():
        observed = metrics[metric]
        if "exact" in comparison:
            threshold = comparison["exact"]
            passed = observed == threshold
            operator = "=="
        else:
            threshold = comparison["minimum"]
            passed = observed >= threshold
            operator = ">="
        results.append(
            {
                "gate": f"{panel_name}.{metric}",
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "pass": passed,
            }
        )
    return results


def run_all_panels(
    prereg: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    all_gates: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    evaluations = prereg["ordered_evaluations"]
    checkpoint = evidence["checkpoint_path"]
    checkpoint_sha256 = evidence["checkpoint_sha256"]
    for evaluation, panel, output in zip(
        evaluations,
        PANEL_SPECS,
        evidence["output_paths"],
        strict=True,
    ):
        started = utc_now()
        command = list(evaluation["command"])
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
            "order": evaluation["order"],
            "panel": panel["name"],
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "command_sha256": evaluation["command_sha256"],
            "attempt_count": 1,
            "returncode": returncode,
            "launch_error": launch_error,
            "stdout": {"sha256": sha256_bytes(stdout), "bytes": len(stdout)},
            "stderr": {"sha256": sha256_bytes(stderr), "bytes": len(stderr)},
            "output": root_relative_text(output),
            "output_present": output.is_file() and not output.is_symlink(),
        }
        panel_gates: list[dict[str, Any]] = []
        output_error: str | None = None
        if record["output_present"]:
            try:
                payload, output_evidence = read_plain_file(
                    output,
                    f"candidate output {panel['name']}",
                )
                document = strict_json_loads(payload, f"candidate output {panel['name']}")
                validate_output_identity(
                    document,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    panel=panel,
                )
                metrics = observed_metrics(document)
                panel_gates = evaluate_gates(str(panel["name"]), metrics)
                record["output_evidence"] = output_evidence
                record["metrics"] = metrics
            except BaseException as error:
                output_error = f"{type(error).__name__}: {error}"
            finally:
                try:
                    os.chmod(output, 0o444, follow_symlinks=False)
                except OSError as error:
                    if output_error is None:
                        output_error = f"could not freeze output: {error}"
        else:
            output_error = "candidate output is absent"
        record["output_error"] = output_error
        record["gates"] = panel_gates
        record["evaluation_pass"] = (
            returncode == 0
            and launch_error is None
            and output_error is None
            and len(panel_gates) == 10
            and all(gate["pass"] for gate in panel_gates)
        )
        records.append(record)
        all_gates.extend(panel_gates)
    return records, all_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the six preregistered E904 specialist panels exactly once."
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    return parser.parse_args()


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise ProtocolError("executor must run from the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError(
            f"wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )
    if sys.flags.isolated != 1 or sys.dont_write_bytecode is not True:
        raise ProtocolError("executor requires Python flags -I -B")


def main() -> None:
    args = parse_args()
    validate_runtime()
    preregistration_argument = args.preregistration
    preregistration_path = Path(
        os.path.normpath(
            os.fspath(
                preregistration_argument
                if preregistration_argument.is_absolute()
                else Path.cwd() / preregistration_argument
            )
        )
    )
    try:
        preregistration_path.relative_to(ROOT.absolute())
    except ValueError as error:
        raise ProtocolError("preregistration must remain inside the repository") from error
    try:
        preregistration_stat = os.lstat(preregistration_path)
    except OSError as error:
        raise ProtocolError("preregistration cannot be inspected") from error
    if stat.S_ISLNK(preregistration_stat.st_mode):
        raise ProtocolError("preregistration must not be a symlink")

    prereg, evidence = validate_preregistration(
        preregistration_path,
        args.expected_preregistration_sha256,
    )
    output_root: Path = evidence["output_root"]
    marker: Path = evidence["marker"]
    require_absent(output_root, "specialist output root")
    require_absent(marker, "formal attempt marker")
    for path in (evidence["manifest_path"], evidence["decision_path"], *evidence["output_paths"]):
        require_absent(path, "formal specialist artifact")

    pre_execution_rehash = rehash_bound_inputs(
        evidence["immutable_bindings"],
        phase="pre_execution",
        raise_on_failure=True,
    )
    cuda = cuda_preflight()
    marker_payload = canonical_json_bytes(
        {
            "schema_version": "ptcg-e904-single-endpoint-specialist-attempt-v2",
            "status": "formal_attempt_consumed",
            "created_at_utc": utc_now(),
            "pid": os.getpid(),
            "preregistration": {
                "path": root_relative_text(preregistration_path),
                "sha256": evidence["preregistration"]["sha256"],
            },
            "candidate": {
                "path": root_relative_text(evidence["checkpoint_path"]),
                "sha256": evidence["checkpoint_sha256"],
                "model_state_sha256": E904_MODEL_STATE_SHA256,
            },
            "raw_u468_parent": {
                "path": root_relative_text(RAW_U468_CHECKPOINT),
                "sha256": RAW_U468_CHECKPOINT_SHA256,
                "model_state_sha256": RAW_U468_MODEL_STATE_SHA256,
            },
            "pre_execution_rehash": pre_execution_rehash,
            "cuda_preflight": cuda,
        }
    )
    atomic_create(marker, marker_payload, mode=0o444)
    output_root.mkdir(mode=0o700)

    records, gates = run_all_panels(prereg, evidence)
    post_execution_rehash = rehash_bound_inputs(
        evidence["immutable_bindings"],
        phase="post_execution",
        raise_on_failure=False,
    )
    complete = len(records) == 6 and all(
        record["attempt_count"] == 1 for record in records
    )
    passed = (
        complete
        and post_execution_rehash["pass"]
        and len(gates) == 60
        and all(record["evaluation_pass"] for record in records)
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "six_panels_completed" if complete else "execution_incomplete",
        "created_at_utc": utc_now(),
        "preregistration": evidence["preregistration"],
        "executor": evidence["executor"],
        "evaluator": evidence["evaluator"],
        "evaluator_dependencies": evidence["evaluator_dependencies"],
        "candidate": evidence["candidate"],
        "raw_u468_parent": evidence["raw_u468_parent"],
        "data": evidence["data"],
        "baselines": evidence["baselines"],
        "immutable_input_rechecks": {
            "pre_execution": pre_execution_rehash,
            "post_execution": post_execution_rehash,
        },
        "cuda_preflight": cuda,
        "ordered_evaluations": records,
        "gate_count": len(gates),
        "all_six_attempted_once": complete,
        "scope": prereg["scope"],
    }
    manifest_payload = canonical_json_bytes(manifest)
    atomic_create(evidence["manifest_path"], manifest_payload, mode=0o444)
    manifest_sha256 = sha256_bytes(manifest_payload)
    decision = {
        "schema_version": DECISION_SCHEMA,
        "status": "passed_specialist_retention" if passed else "failed_specialist_retention",
        "created_at_utc": utc_now(),
        "pass": passed,
        "manifest": {
            "path": root_relative_text(evidence["manifest_path"]),
            "sha256": manifest_sha256,
        },
        "preregistration": {
            "path": root_relative_text(preregistration_path),
            "sha256": evidence["preregistration"]["sha256"],
        },
        "candidate": {
            "path": root_relative_text(evidence["checkpoint_path"]),
            "sha256": evidence["checkpoint_sha256"],
            "model_state_sha256": E904_MODEL_STATE_SHA256,
            "checkpoint_update": EXPECTED_CHECKPOINT_UPDATE,
        },
        "raw_u468_parent": {
            "path": root_relative_text(RAW_U468_CHECKPOINT),
            "sha256": RAW_U468_CHECKPOINT_SHA256,
            "model_state_sha256": RAW_U468_MODEL_STATE_SHA256,
            "checkpoint_update": EXPECTED_CHECKPOINT_UPDATE,
        },
        "immutable_input_rechecks_pass": (
            pre_execution_rehash["pass"] and post_execution_rehash["pass"]
        ),
        "immutable_input_rechecks": {
            "pre_execution": pre_execution_rehash,
            "post_execution": post_execution_rehash,
        },
        "all_six_attempted_once": complete,
        "all_60_gates_required": True,
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(gate["pass"]) for gate in gates),
        "failed_gates": [gate for gate in gates if not gate["pass"]],
        "panel_pass": {
            record["panel"]: record["evaluation_pass"] for record in records
        },
        "authorization": {
            "broad_behavior_authorized": passed,
            "gold_authorized": False,
            "package_upload_or_submission_authorized": False,
        },
    }
    decision_payload = canonical_json_bytes(decision)
    atomic_create(evidence["decision_path"], decision_payload, mode=0o444)
    os.chmod(output_root, 0o500)
    print(
        json.dumps(
            {
                "decision": root_relative_text(evidence["decision_path"]),
                "decision_sha256": sha256_bytes(decision_payload),
                "pass": passed,
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

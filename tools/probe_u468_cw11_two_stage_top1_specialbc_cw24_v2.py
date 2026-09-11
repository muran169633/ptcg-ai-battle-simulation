#!/usr/bin/env python3
"""Frozen-constant CW24 v2 train-only wrapper around the audited v1 engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v2.py"
ENGINE = TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py"
ENGINE_SHA256 = "a3c96eed95a6b558c61303e2190eaa4f353970add4509e8770c3dfcc1621db39"
ENGINE_MODE = 0o555
V1_RESULT = ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v1.json"
V1_RESULT_SHA256 = "9e3be911bfba95338538f315d47d3f7ce9f598e830ace607332d46697cb0cda5"
OUTPUT = ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v2.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-two-stage-top1-specialbc-cw24-v2"
SEED = 202608042
PF_RADIUS_CAP = 9.8e-4
TOTAL_RADIUS_CAP = 9.9999e-4
AUX_WEIGHTS = {
    "dominic32_nll_descent": 0.25,
    "dominic_closest_margin_ascent": 0.35,
    "special9_nll_descent": 0.10,
    "szlach_top1_ce_descent": 0.10,
    "core_other_top1_ce_descent": 0.20,
}


class ProtocolError(RuntimeError):
    """Fail-closed v2 wrapper error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    def check(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ProtocolError("nonfinite JSON value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ProtocolError("non-string JSON key")
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def regular_source(path: Path, expected_sha: str, expected_mode: int, label: str) -> tuple[bytes, dict[str, Any]]:
    before = path.lstat()
    source = path.read_bytes()
    after = path.lstat()
    digest = hashlib.sha256(source).hexdigest()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "identity_stable": (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": digest == expected_sha,
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} drift: {checks}")
    return source, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(source),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def self_evidence(require_frozen: bool) -> dict[str, Any]:
    before = SCRIPT.lstat()
    source = SCRIPT.read_bytes()
    after = SCRIPT.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "identity_stable": (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "mode_0555": stat.S_IMODE(after.st_mode) == 0o555,
        "one_frozen_variant": source.count(b"\nPF_RADIUS" + b"_CAP =") == 1
        and source.count(b"\nTOTAL_RADIUS" + b"_CAP =") == 1
        and source.count(b"\nAUX_" + b"WEIGHTS =") == 1,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_optimizer_step": (b"optimizer" + b".step") not in source,
    }
    if require_frozen and not all(checks.values()):
        raise ProtocolError(f"v2 source audit failed: {checks}")
    return {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ProtocolError("short v2 result write")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    observed = path.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and int(observed.st_nlink) == 1,
        "mode_0444": stat.S_IMODE(observed.st_mode) == 0o444,
        "size_exact": int(observed.st_size) == len(payload),
        "sha_exact": sha256_file(path) == hashlib.sha256(payload).hexdigest(),
    }
    if not all(checks.values()):
        raise ProtocolError(f"v2 publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def load_engine() -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = regular_source(ENGINE, ENGINE_SHA256, ENGINE_MODE, "frozen CW24 v1 engine")
    name = "cw24_v2_locked_v1_engine"
    if name in sys.modules:
        raise ProtocolError("v1 engine module name occupied")
    module = types.ModuleType(name)
    module.__file__ = str(ENGINE)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, str(ENGINE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, evidence


def production_run() -> dict[str, Any]:
    source = self_evidence(require_frozen=True)
    _, v1_result_evidence = regular_source(V1_RESULT, V1_RESULT_SHA256, 0o444, "frozen v1 result")
    engine, engine_evidence = load_engine()
    originals = {
        "PF_RADIUS_CAP": engine.PF_RADIUS_CAP,
        "TOTAL_RADIUS_CAP": engine.TOTAL_RADIUS_CAP,
        "AUX_WEIGHTS": engine.AUX_WEIGHTS,
        "SEED": engine.SEED,
    }
    engine.PF_RADIUS_CAP = PF_RADIUS_CAP
    engine.TOTAL_RADIUS_CAP = TOTAL_RADIUS_CAP
    engine.AUX_WEIGHTS = dict(AUX_WEIGHTS)
    engine.SEED = SEED
    try:
        result = engine.production_run()
    finally:
        for key, value in originals.items():
            setattr(engine, key, value)
    observed = result["endpoint"].get("two_stage_geometry", {})
    variant_checks = {
        "PF_radius_exact": observed.get("PF_planned_l2") == PF_RADIUS_CAP,
        "total_radius_exact": math.isclose(
            float(observed.get("planned_total_l2", -1.0)),
            TOTAL_RADIUS_CAP,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "weights_exact": observed.get("aux_weights") == AUX_WEIGHTS,
        "single_candidate": observed.get("single_candidate_only") is True,
    }
    if not all(variant_checks.values()):
        raise ProtocolError(f"v2 variant application drift: {variant_checks}")
    result["schema_version"] = SCHEMA
    result["seed"] = SEED
    result["inputs"]["CW24_v2_wrapper"] = source
    result["inputs"]["CW24_v2_frozen_v1_engine"] = engine_evidence
    result["inputs"]["CW24_v1_NO_GO_result"] = v1_result_evidence
    result["audit"]["CW24_v2_variant"] = {
        "PF_radius_cap": PF_RADIUS_CAP,
        "total_radius_cap": TOTAL_RADIUS_CAP,
        "aux_weights": AUX_WEIGHTS,
        "checks": variant_checks,
        "reason": "v1 PF-only repaired one of three targets; v2 reallocates radius without adding a candidate grid",
        "single_changed_train_shadow": True,
        "official_unique_changed_candidate_count_consumed": 0,
    }
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 2
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    canonical_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        _, engine_evidence = regular_source(ENGINE, ENGINE_SHA256, ENGINE_MODE, "frozen v1 engine")
        _, result_evidence = regular_source(V1_RESULT, V1_RESULT_SHA256, 0o444, "frozen v1 result")
        print(canonical_json({
            "schema_version": SCHEMA,
            "status": "static_audit_pass",
            "source": self_evidence(require_frozen=True),
            "engine": engine_evidence,
            "v1_result": result_evidence,
            "variant": {
                "PF_radius_cap": PF_RADIUS_CAP,
                "total_radius_cap": TOTAL_RADIUS_CAP,
                "aux_weights": AUX_WEIGHTS,
            },
            "output_absent": not OUTPUT.exists(),
            "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
            "CUDA_initialized": False,
            "writes_performed": 0,
        }).decode(), end="")
        return
    if OUTPUT.exists():
        raise ProtocolError("frozen CW24 v2 output already exists")
    result = production_run()
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    print(canonical_json({
        "schema_version": SCHEMA,
        "status": result["status"],
        "decision": result["decision"],
        "candidate_model_state_sha256": result["endpoint"].get("trial", {}).get(
            "candidate_model_state_sha256"
        ),
        "candidate_payload_present": result["endpoint"].get("candidate_payload") is not None,
        "changed_candidate_train_shadow_count": result["endpoint"].get(
            "changed_candidate_train_shadow_count"
        ),
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
        "output": publication,
    }).decode(), end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal v14 repair wrapper derived from the frozen v13 two-stage probe.

The exact frozen v13 source is authenticated and transformed only in memory.
Stage 1 and the stage-2 target remain exact, while stage 2 restores standard
projected SGD: backward, the original clip, SGD, then only v12's endpoint ball
projection.  Its actor hash set becomes a hash-to-step ledger (CW11 is step 0),
and an exact repeated stage-2 actor becomes structured ``NO_GO_V14_STALLED``;
stage-1 repeats remain fail-closed protocol errors.  This draft never invokes
frozen v13's consumed attempt marker and never materializes actor bytes or delta
arrays on NO_GO.
"""

from __future__ import annotations

import argparse
import ast
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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_stall_diagnostic_specialbc_cw24_v14.py"
OUTPUT = ROOT / "artifacts/cw24_cw11_stall_diagnostic_specialbc_trainonly_v14.json"
ATTEMPT_MARKER = ROOT / "artifacts/.ptcg-cw24-cw11-stall-diagnostic-specialbc-v14-attempt.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-stall-diagnostic-specialbc-cw24-v14"
SEED = 202608052
STAGE1_REFERENCE_STEP = 29
STAGE2_MAX_STEPS = 32
TOTAL_MAX_STEPS = STAGE1_REFERENCE_STEP + STAGE2_MAX_STEPS

V13_SOURCE = TOOLS / "probe_u468_cw11_two_stage_pf7_specialbc_cw24_v13.py"
V13_SOURCE_SHA256 = "a99762d1a2c98b0521199d845d3f041fff40b3787ff3ceda0fd8207e1c6d0ff7"
V13_ATTEMPT = ROOT / "artifacts/.ptcg-cw24-cw11-two-stage-pf7-specialbc-v13-attempt.json"
V13_ATTEMPT_SHA256 = "924e4741a407351246e25c19e476e74e7cd504886403114c540638f72c00316d"
V13_OUTPUT = ROOT / "artifacts/cw24_cw11_two_stage_pf7_specialbc_trainonly_v13.json"

DEPENDENCIES = (
    ("frozen_v13_source", V13_SOURCE, V13_SOURCE_SHA256, 0o555),
    ("frozen_v13_consumed_attempt", V13_ATTEMPT, V13_ATTEMPT_SHA256, 0o444),
    (
        "frozen_v12_source",
        TOOLS / "probe_u468_cw11_projected_nonlinear_specialbc_cw24_v12.py",
        "003e476325a08ff5be400167fccbd43705fac9a4071d806101ddb02969ea2301",
        0o555,
    ),
    (
        "frozen_v12_result",
        ROOT / "artifacts/cw24_cw11_projected_nonlinear_specialbc_trainonly_v12.json",
        "effee98dc03c6d505f7bc2007dae88e28190c0fbba80f22c7902363592b554dc",
        0o444,
    ),
    (
        "frozen_CW24_v1",
        TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py",
        "a3c96eed95a6b558c61303e2190eaa4f353970add4509e8770c3dfcc1621db39",
        0o555,
    ),
    (
        "frozen_CW22",
        TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py",
        "f307f3f26355125f97c857f14b97a73d9a7a4f2ea39ca8b0a578e801055e82b4",
        0o555,
    ),
    (
        "frozen_CW23",
        TOOLS / "probe_u468_cw11_boundary_qp_specialbc_cw23_v1.py",
        "4a2d209af71a7ea5d955bd373f1fbada936688fc5bc597615c41192dc360b4b8",
        0o555,
    ),
    (
        "plain_SGD_reference",
        TOOLS / "run_u468_raw_actor6_equalblend_sgd512_shadow.py",
        "e04f7b7579ef42d0c6f643db833779fb837ae86e3205d948b5564d08f8b30f8e",
        0o555,
    ),
    (
        "dynamic_pair_reference",
        TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v1.py",
        "9c5a7377d82b18e437ca89d064ce8be60554646573e3aca27f4e40f4989ab77c",
        0o555,
    ),
    (
        "frozen_B352_selection",
        ROOT / "artifacts/cw24_top1_b352_selection_v1.json",
        "1f72b26eccd436ca0d341f837ec42949f5823bf7f0be73461aaeab28166cc222",
        0o444,
    ),
)

INIT_OLD = '''        actor_hashes = {
            hashlib.sha256(
                cw20.actor_bytes(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
            ).hexdigest()
        }
'''
INIT_NEW = '''        actor_hashes = {
            hashlib.sha256(
                cw20.actor_bytes(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
            ).hexdigest(): 0
        }
'''
RADIAL_OLD = '''            if global_step > STAGE1_REFERENCE_STEP:
                radial_gradient_projection = project_non_outward_sgd_direction(
                    parameters, cw22.EXPECTED_ACTOR_NAMES, cw11_actor
                )
            else:
                radial_gradient_projection = {
                    "evaluated": False,
                    "placement": "stage1_exact_v12_replay_no_gradient_projection",
                    "outward_component_removed": False,
                }
'''
RADIAL_NEW = '''            if global_step > STAGE1_REFERENCE_STEP:
                radial_gradient_projection = {
                    "evaluated": False,
                    "placement": "v14_standard_projected_SGD_no_preclip_radial_projection",
                    "outward_component_removed": False,
                    "sequence": "backward_then_original_clip_then_SGD_then_endpoint_ball_projection",
                }
            else:
                radial_gradient_projection = {
                    "evaluated": False,
                    "placement": "stage1_exact_v12_replay_no_gradient_projection",
                    "outward_component_removed": False,
                }
'''
FINAL_RADIAL_OLD = '''            "stage2_radial_gradient_projection_every_step": all(
                record["radial_gradient_projection"]["evaluated"] is True
                and all(record["radial_gradient_projection"]["checks"].values())
                for record in stage2_trajectory
            ),
'''
FINAL_RADIAL_NEW = '''            "v14_standard_projected_SGD_every_stage2_step": all(
                record["radial_gradient_projection"]["evaluated"] is False
                and record["radial_gradient_projection"]["placement"]
                == "v14_standard_projected_SGD_no_preclip_radial_projection"
                for record in stage2_trajectory
            ),
'''
COMMON_RADIAL_OLD = '''            "radial_gradient_projection_contract": {
                "stage1_never_evaluated": all(
                    record["radial_gradient_projection"]["evaluated"] is False
                    for record in stage1_trajectory
                ),
                "stage2_every_step_evaluated": all(
                    record["radial_gradient_projection"]["evaluated"] is True
                    for record in stage2_trajectory
                ),
                "stage2_every_step_non_outward": all(
                    all(record["radial_gradient_projection"]["checks"].values())
                    for record in stage2_trajectory
                ),
                "placement": "after_backward_before_clip_then_SGD_step_then_CW11_ball_projection",
            },
'''
COMMON_RADIAL_NEW = '''            "v14_standard_projected_SGD_contract": {
                "preclip_radial_projection_absent": all(
                    record["radial_gradient_projection"]["evaluated"] is False
                    for record in stage1_trajectory + stage2_trajectory
                ),
                "original_gradient_clip_then_plain_SGD": True,
                "endpoint_CW11_ball_projection": "v12.project_actor6_after_every_step",
            },
'''
AUDIT_RADIAL_OLD = '''        "stage2_radial_gradient_projection_every_step": all(
            record["radial_gradient_projection"]["evaluated"] is True
            and all(record["radial_gradient_projection"]["checks"].values())
            for record in endpoint["stage2_trajectory"]
        ),
'''
AUDIT_RADIAL_NEW = '''        "v14_stage2_standard_projected_SGD_every_step": all(
            record["radial_gradient_projection"]["evaluated"] is False
            and record["radial_gradient_projection"]["placement"]
            == "v14_standard_projected_SGD_no_preclip_radial_projection"
            for record in endpoint["stage2_trajectory"]
        ),
'''
DUPLICATE_OLD = '''            if actor_hash in actor_hashes:
                raise ProtocolError("repeated actor6 state in two-stage trajectory")
            actor_hashes.add(actor_hash)
'''
DUPLICATE_NEW = '''            duplicate_of_step = actor_hashes.get(actor_hash)
            if duplicate_of_step is not None:
                if global_step <= STAGE1_REFERENCE_STEP:
                    raise ProtocolError("repeated actor6 state in exact stage1 trajectory")
                if last_trial is None:
                    raise ProtocolError("v14 stalled state lacks a prior native trial")
                previous_gate = dict(last_trial["native_terminal_gate"])
                safe_projection = {
                    "raw_step_l2": float(projection["raw_step_l2"]),
                    "projection_applied": bool(projection["projection_applied"]),
                    "projected_actor_l2": float(projection["actual_l2"]),
                    "projected_delta_float64_le_sha256": str(
                        projection["actual_delta_float64_le_sha256"]
                    ),
                }
                return {
                    "decision": "NO_GO_V14_STALLED",
                    "reason": "PROJECTED_SGD_REPEATED_AN_EARLIER_ACTOR_STATE",
                    "context_checks": context_checks,
                    "dependency_checks": dependency_checks,
                    "cache": cache_audit,
                    "baseline_checks": contract["baseline_checks"],
                    "objective_mask_counts": contract["mask_counts"],
                    "baseline_loss": baseline_loss_audit,
                    "baseline_dynamic_pairs_added": [
                        dict(value) for value in baseline_added
                    ],
                    "fixed_pair_contract": [
                        dict(value) for value in contract["fixed_pairs"]
                    ],
                    "zero_margin_contract": [
                        dict(value) for value in contract["zero_contract"]
                    ],
                    "frozen_v12_reference": dict(reference_audit),
                    "stage1_transition": transition if stage2_started else None,
                    "stage1_trajectory": stage1_trajectory,
                    "stage2_trajectory": stage2_trajectory,
                    "changed_candidate_train_shadow_count": (
                        len(stage1_trajectory) + len(stage2_trajectory)
                    ),
                    "optimizer_step_calls_attempted": int(global_step),
                    "max_steps": TOTAL_MAX_STEPS,
                    "trial": last_trial,
                    "stalled_diagnostic": {
                        "global_step": int(global_step),
                        "duplicate_of_step": int(duplicate_of_step),
                        "CW11_step_identifier": 0,
                        "duplicate_actor_float32_le_sha256": actor_hash,
                        "radial_gradient_projection": dict(
                            radial_gradient_projection
                        ),
                        "gradient_norm_before_clip": preclip_norm_value,
                        "gradient_clip_max_norm": MAX_GRAD_NORM,
                        "projection": safe_projection,
                        "prior_trial_step": int(last_trial["step"]),
                        "prior_PF_native": dict(previous_gate["PF_native"]),
                        "prior_native_gate": previous_gate,
                        "actor_bytes_present": False,
                        "delta_array_present": False,
                    },
                    "candidate_payload": None,
                }
            actor_hashes[actor_hash] = int(global_step)
'''


class ProtocolError(RuntimeError):
    """Fail-closed v14 diagnostic protocol error."""


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


def regular_source(
    path: Path, expected_sha: str, expected_mode: int, label: str
) -> tuple[bytes, dict[str, Any]]:
    before = path.lstat()
    source = path.read_bytes()
    after = path.lstat()
    digest = hashlib.sha256(source).hexdigest()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
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


def path_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    return False


def binding_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(record["path"]),
        "sha256": str(record["sha256"]),
        "bytes": int(record["bytes"]),
        "mode_octal": str(record["mode_octal"]),
    }


def dependency_evidence() -> dict[str, Any]:
    return {
        label: regular_source(path, digest, mode, label)[1]
        for label, path, digest, mode in DEPENDENCIES
    }


def validate_pre_cuda_runtime() -> dict[str, Any]:
    checks = {
        "repo_root": Path.cwd().resolve() == ROOT,
        "my_project_env": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1
        and sys.dont_write_bytecode is True,
        "pycache_prefix_dev_null": sys.pycache_prefix == "/dev/null",
        "torch_not_imported_before_claim": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        raise ProtocolError(f"pre-CUDA runtime drift: {checks}")
    return {
        "python": str(Path(sys.executable).resolve()),
        "checks": checks,
        "pass": True,
    }


def transform_v13_source() -> tuple[bytes, dict[str, Any]]:
    raw, evidence = regular_source(
        V13_SOURCE, V13_SOURCE_SHA256, 0o555, "frozen v13 source"
    )
    source = raw.decode("utf-8")
    transforms = (
        ("actor_hash_ledger", INIT_OLD, INIT_NEW),
        ("remove_preclip_radial_projection", RADIAL_OLD, RADIAL_NEW),
        ("stage2_duplicate_to_stall", DUPLICATE_OLD, DUPLICATE_NEW),
        ("common_standard_projected_SGD_contract", COMMON_RADIAL_OLD, COMMON_RADIAL_NEW),
        ("GO_standard_projected_SGD_gate", FINAL_RADIAL_OLD, FINAL_RADIAL_NEW),
        ("result_standard_projected_SGD_audit", AUDIT_RADIAL_OLD, AUDIT_RADIAL_NEW),
    )
    records = []
    for label, old, new in transforms:
        occurrences = source.count(old)
        if occurrences != 1:
            raise ProtocolError(
                f"{label}: expected one exact frozen-v13 transform site, got {occurrences}"
            )
        source = source.replace(old, new, 1)
        records.append(
            {
                "label": label,
                "occurrences": occurrences,
                "old_sha256": hashlib.sha256(old.encode()).hexdigest(),
                "new_sha256": hashlib.sha256(new.encode()).hexdigest(),
            }
        )
    transformed = source.encode("utf-8")
    compile(transformed, str(V13_SOURCE), "exec", dont_inherit=True)
    tree = ast.parse(transformed, filename=str(V13_SOURCE))
    call_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.append(node.func.attr)
    checks = {
        "six_unique_exact_transforms": len(records) == 6,
        "old_duplicate_exception_removed": DUPLICATE_OLD.encode() not in transformed,
        "structured_stall_present": transformed.count(b'"NO_GO_V14_STALLED"') == 1,
        "stage1_repeat_remains_protocol_error": transformed.count(
            b'if global_step <= STAGE1_REFERENCE_STEP:\n'
            b'                    raise ProtocolError('
            b'"repeated actor6 state in exact stage1 trajectory")'
        )
        == 1,
        "stage1_repeat_guard_precedes_structured_stall": transformed.find(
            b'if global_step <= STAGE1_REFERENCE_STEP:'
        )
        < transformed.find(b'"NO_GO_V14_STALLED"'),
        "stage1_projection_diagnostic_preserved": transformed.count(
            b'"placement": "stage1_exact_v12_replay_no_gradient_projection"'
        )
        == 1,
        "preclip_radial_projection_call_removed": call_names.count(
            "project_non_outward_sgd_direction"
        )
        == 0,
        "original_clip_one_site": call_names.count("clip_grad_norm_") == 1,
        "plain_SGD_step_one_site": call_names.count("step") == 1,
        "endpoint_ball_projection_one_site": call_names.count("project_actor6") == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v14 transformed-v13 audit failed: {checks}")
    return transformed, {
        "frozen_source": evidence,
        "transformed_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_bytes": len(transformed),
        "transforms": records,
        "checks": checks,
    }


def transformed_v13_module() -> tuple[ModuleType, dict[str, Any]]:
    source, audit = transform_v13_source()
    module_name = "cw24_v14_transformed_frozen_v13"
    if module_name in sys.modules:
        raise ProtocolError("transformed v13 module name already occupied")
    module = types.ModuleType(module_name)
    module.__file__ = str(V13_SOURCE)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(V13_SOURCE), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, audit


def load_json_no_duplicates(raw: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"{label}: duplicate key {key}")
            result[key] = value
        return result

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ProtocolError(f"{label}: nonfinite constant {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ProtocolError(f"{label}: root is not an object")
    return value


def validate_v13_terminal_state() -> dict[str, Any]:
    raw, evidence = regular_source(
        V13_ATTEMPT, V13_ATTEMPT_SHA256, 0o444, "consumed v13 attempt"
    )
    payload = load_json_no_duplicates(raw, "consumed v13 attempt")
    checks = {
        "schema_exact": payload.get("schema_version")
        == "ptcg-u468-cw11-two-stage-pf7-specialbc-cw24-v13-attempt",
        "status_claimed": payload.get("status")
        == "claimed_before_module_exec_or_CUDA",
        "source_exact": payload.get("source", {}).get("sha256")
        == V13_SOURCE_SHA256,
        "total61": payload.get("total_maximum_changed_train_shadows") == 61,
        "stage1_29_stage2_32": payload.get("stage1_reference_steps") == 29
        and payload.get("stage2_maximum_steps") == 32,
        "retry_forbidden": payload.get("retry_authorized") is False,
        "v13_output_absent_now": path_absent(V13_OUTPUT),
    }
    if not all(checks.values()):
        raise ProtocolError(f"frozen v13 terminal-state drift: {checks}")
    return {"attempt": evidence, "checks": checks, "payload": payload}


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    _, transform = transform_v13_source()
    checks = {
        "syntax_valid": True,
        "transform_static_audit": all(transform["checks"].values()),
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_network_import": not any(
            name.split(".")[0] in {"requests", "urllib", "httpx", "socket"}
            for name in imports
        ),
        "single_result_publication_site": source.count(
            b"publish_o_excl(" + b"OUTPUT"
        )
        == 1,
        "single_attempt_publication_site": source.count(
            b"publish_o_excl(" + b"ATTEMPT_MARKER"
        )
        == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v14 source audit failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "transform": transform,
    }


def self_evidence(require_frozen: bool) -> dict[str, Any]:
    before = SCRIPT.lstat()
    source = SCRIPT.read_bytes()
    after = SCRIPT.lstat()
    mode = stat.S_IMODE(after.st_mode)
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
        "mode_allowed": mode == 0o555 if require_frozen else mode in {0o644, 0o664, 0o555},
        "source_audit": source_audit()["pass"],
    }
    if not all(checks.values()):
        raise ProtocolError(f"v14 self evidence failed: {checks}")
    return {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "mode_octal": format(mode, "04o"),
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
                raise ProtocolError("short v14 publication write")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory_fd = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
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
        raise ProtocolError(f"v14 publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def v13_terminal_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "consumed_attempt": binding_summary(state["attempt"]),
        "checks": dict(state["checks"]),
        "v13_output": str(V13_OUTPUT.relative_to(ROOT)),
        "v13_output_absent": bool(state["checks"]["v13_output_absent_now"]),
    }


def claim_attempt() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v14 output already exists before attempt claim")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v14 one-shot attempt was already consumed")
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    terminal = validate_v13_terminal_state()
    transform = transform_v13_source()[1]
    payload = {
        "schema_version": f"{SCHEMA}-attempt",
        "status": "claimed_before_module_exec_or_CUDA",
        "source": binding_summary(source),
        "dependencies": {
            name: binding_summary(record) for name, record in dependencies.items()
        },
        "frozen_v13_terminal_state": v13_terminal_summary(terminal),
        "in_memory_v13_transform": {
            "transformed_sha256": transform["transformed_sha256"],
            "transformed_bytes": transform["transformed_bytes"],
            "checks": transform["checks"],
        },
        "runtime": runtime,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent_by_lstat_at_claim": True,
        "v13_output_absent_by_lstat_at_claim": True,
        "CUDA_preflight_state": "not_imported_pending_production_validation",
        "single_seed": SEED,
        "stage1_reference_steps": STAGE1_REFERENCE_STEP,
        "stage2_maximum_steps": STAGE2_MAX_STEPS,
        "total_maximum_changed_train_shadows": TOTAL_MAX_STEPS,
        "stage2_optimizer_sequence": [
            "backward",
            "original_clip_grad_norm",
            "plain_SGD_step",
            "v12_project_actor6_endpoint_ball_projection",
        ],
        "preclip_radial_gradient_projection": False,
        "stage2_exact_repeat_outcome": "NO_GO_V14_STALLED",
        "stage1_exact_repeat_outcome": "ProtocolError",
        "retry_authorized": False,
        "network_package_upload_submission": False,
    }
    publication = publish_o_excl(ATTEMPT_MARKER, canonical_json(payload))
    return {"payload": payload, "publication": publication}


def no_go_material_audit(value: Any) -> dict[str, Any]:
    violations: list[str] = []
    counters = {
        "objects": 0,
        "arrays": 0,
        "candidate_payload_keys": 0,
        "binary_values": 0,
    }
    forbidden_exact = {
        "actor_bytes",
        "candidate_actor_float32_le",
        "actual_delta",
    }

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            counters["objects"] += 1
            for key, child in item.items():
                child_path = f"{path}.{key}"
                if key == "candidate_payload":
                    counters["candidate_payload_keys"] += 1
                    if child is not None:
                        violations.append(f"{child_path}:non_null")
                if key in forbidden_exact:
                    violations.append(f"{child_path}:forbidden_material_key")
                if "delta" in key.lower() and isinstance(child, (list, tuple)):
                    violations.append(f"{child_path}:delta_array")
                walk(child, child_path)
            return
        if isinstance(item, (list, tuple)):
            counters["arrays"] += 1
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
            return
        if isinstance(item, (bytes, bytearray, memoryview)):
            counters["binary_values"] += 1
            violations.append(f"{path}:binary_value")

    walk(value, "$<result>")
    checks = {
        "candidate_payloads_all_null": not any(
            value.endswith(":non_null") for value in violations
        ),
        "no_actor_or_actual_delta_material_keys": not any(
            value.endswith(":forbidden_material_key") for value in violations
        ),
        "no_delta_arrays": not any(
            value.endswith(":delta_array") for value in violations
        ),
        "no_binary_values": counters["binary_values"] == 0,
    }
    if violations or not all(checks.values()):
        raise ProtocolError(
            f"v14 NO_GO material disclosure rejected: {violations[:8]}"
        )
    return {"checks": checks, "counts": counters, "pass": True}


def validate_stalled_endpoint(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic = endpoint.get("stalled_diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ProtocolError("v14 stalled endpoint lacks diagnostic object")
    global_step = int(diagnostic.get("global_step", -1))
    duplicate_of_step = int(diagnostic.get("duplicate_of_step", -1))
    stage1 = endpoint.get("stage1_trajectory")
    stage2 = endpoint.get("stage2_trajectory")
    radial = diagnostic.get("radial_gradient_projection")
    projection = diagnostic.get("projection")
    actor_hash = str(diagnostic.get("duplicate_actor_float32_le_sha256", ""))
    checks = {
        "decision_exact": endpoint.get("decision") == "NO_GO_V14_STALLED",
        "payload_null": endpoint.get("candidate_payload") is None,
        "stage2_only": STAGE1_REFERENCE_STEP < global_step <= TOTAL_MAX_STEPS,
        "duplicate_precedes_attempt": 0 <= duplicate_of_step < global_step,
        "CW11_is_ledger_step_zero": diagnostic.get("CW11_step_identifier") == 0,
        "stage1_exact_count": isinstance(stage1, list)
        and len(stage1) == STAGE1_REFERENCE_STEP,
        "stage2_unique_count_before_repeat": isinstance(stage2, list)
        and len(stage2) == global_step - STAGE1_REFERENCE_STEP - 1,
        "changed_count_excludes_repeated_endpoint": endpoint.get(
            "changed_candidate_train_shadow_count"
        )
        == STAGE1_REFERENCE_STEP
        + (len(stage2) if isinstance(stage2, list) else -1),
        "optimizer_attempt_count_includes_repeat": endpoint.get(
            "optimizer_step_calls_attempted"
        )
        == global_step,
        "prior_trial_is_previous_unique_step": diagnostic.get("prior_trial_step")
        == global_step - 1,
        "hash_only_actor_identity": len(actor_hash) == 64
        and all(character in "0123456789abcdef" for character in actor_hash),
        "no_actor_bytes": diagnostic.get("actor_bytes_present") is False,
        "no_delta_array": diagnostic.get("delta_array_present") is False,
        "preclip_radial_projection_absent": isinstance(radial, Mapping)
        and radial.get("evaluated") is False
        and radial.get("placement")
        == "v14_standard_projected_SGD_no_preclip_radial_projection",
        "safe_projection_scalars_and_hash_only": isinstance(projection, Mapping)
        and set(projection)
        == {
            "raw_step_l2",
            "projection_applied",
            "projected_actor_l2",
            "projected_delta_float64_le_sha256",
        },
        "prior_native_gate_present": isinstance(
            diagnostic.get("prior_native_gate"), Mapping
        ),
        "prior_PF_native_present": isinstance(
            diagnostic.get("prior_PF_native"), Mapping
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"v14 stalled endpoint contract failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "attempted_global_step": global_step,
        "attempted_stage2_step": global_step - STAGE1_REFERENCE_STEP,
        "duplicate_of_step": duplicate_of_step,
        "cycle_length": global_step - duplicate_of_step,
    }


def production_run() -> dict[str, Any]:
    pre_cuda_runtime = validate_pre_cuda_runtime()
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    terminal = validate_v13_terminal_state()
    v13, transform = transformed_v13_module()
    result = v13.production_run()

    post_source = self_evidence(require_frozen=True)
    post_dependencies = dependency_evidence()
    post_terminal = validate_v13_terminal_state()
    rehash_checks = {
        "v14_source_byte_identity": post_source == source,
        "all_dependencies_byte_identity": post_dependencies == dependencies,
        "frozen_v13_terminal_state_identity": post_terminal == terminal,
        "v13_output_absent_after_training": path_absent(V13_OUTPUT),
    }
    if not all(rehash_checks.values()):
        raise ProtocolError(f"v14 post-run immutable input drift: {rehash_checks}")
    if not isinstance(result, dict) or not isinstance(result.get("endpoint"), Mapping):
        raise ProtocolError("transformed v13 returned malformed result")
    endpoint = result["endpoint"]
    decision = str(endpoint.get("decision", ""))
    allowed_decisions = {
        "GO_CW24_TWO_STAGE_PF7_TRAIN_GATE",
        "NO_GO_CW24_TWO_STAGE_PF7_TRAIN_GATE",
        "NO_GO_V14_STALLED",
    }
    if decision not in allowed_decisions:
        raise ProtocolError(f"unexpected transformed-v13 decision: {decision}")
    payload_present = endpoint.get("candidate_payload") is not None
    if payload_present != decision.startswith("GO_"):
        raise ProtocolError("v14 GO/payload equivalence failed")
    changed = int(endpoint.get("changed_candidate_train_shadow_count", -1))
    if changed < STAGE1_REFERENCE_STEP or changed > TOTAL_MAX_STEPS:
        raise ProtocolError("v14 changed train-shadow budget drift")
    stalled_audit = (
        validate_stalled_endpoint(endpoint)
        if decision == "NO_GO_V14_STALLED"
        else None
    )

    result["schema_version"] = SCHEMA
    result["status"] = decision
    result["decision"] = decision
    result["seed"] = SEED
    base = result.setdefault("base", {})
    base["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_v12_replay_then_"
        "PF7_directed_special_BC_with_standard_projected_SGD"
    )
    two_stage = endpoint.setdefault("two_stage_contract", {})
    two_stage.update(
        {
            "stage1": (
                "exact frozen-v12 trajectory through buffered step29 after "
                "first PF0 A/B pass at step28"
            ),
            "stage1_steps": STAGE1_REFERENCE_STEP,
            "stage2": (
                "exact v13 PF7 target and retention objective with standard "
                "projected SGD"
            ),
            "stage2_max_steps": STAGE2_MAX_STEPS,
            "stage2_gradient_projection": "none_before_original_clip",
            "optimizer_sequence": [
                "backward",
                "original_clip_grad_norm",
                "plain_SGD_step",
                "v12_project_actor6_endpoint_ball_projection",
            ],
            "exact_stage2_actor_repeat": "structured_NO_GO_V14_STALLED",
            "exact_stage1_actor_repeat": "fail_closed_ProtocolError",
        }
    )

    selection = result.get("selection")
    if not isinstance(selection, Mapping):
        raise ProtocolError("v14 result lacks selection object")
    optimization = selection.get("optimization_contract")
    if not isinstance(optimization, dict):
        raise ProtocolError("v14 result lacks mutable optimization contract")
    optimization.pop("stage2_non_outward_gradient_projection_before_clip", None)
    optimization.update(
        {
            "preclip_radial_gradient_projection": False,
            "standard_projected_SGD": True,
            "stage2_optimizer_sequence": [
                "backward",
                "original_clip_grad_norm",
                "plain_SGD_step",
                "v12_project_actor6_endpoint_ball_projection",
            ],
            "only_constraint_projection": "v12.project_actor6_endpoint_ball",
            "stage2_exact_repeat_to_structured_NO_GO": True,
            "stage1_repeat_remains_protocol_error": True,
        }
    )
    historical = result.get("historical_exact_CW11_replay")
    if not isinstance(historical, dict):
        raise ProtocolError("v14 result lacks historical replay audit")
    historical.pop(
        "new_validation_rows_opened_for_CW24_v13_selection_or_candidate", None
    )
    historical["new_validation_rows_opened_for_CW24_v14_selection_or_candidate"] = 0

    inputs = result.setdefault("inputs", {})
    inputs["CW24_v14_source"] = source
    inputs["CW24_v14_source_post_run"] = post_source
    inputs["CW24_v14_dependencies"] = dependencies
    inputs["CW24_v14_dependencies_post_run"] = post_dependencies
    inputs["CW24_v14_frozen_v13_terminal_state"] = terminal
    inputs["CW24_v14_frozen_v13_terminal_state_post_run"] = post_terminal
    inputs["CW24_v14_in_memory_v13_transform"] = transform
    audit = result.setdefault("audit", {})
    stage1_trajectory = endpoint.get("stage1_trajectory", [])
    stage2_trajectory = endpoint.get("stage2_trajectory", [])
    v14_contract = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "pre_cuda_runtime": pre_cuda_runtime,
        "post_run_rehash_checks": rehash_checks,
        "train_only": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 1,
        "optimizer_step_calls_attempted": int(
            endpoint.get("optimizer_step_calls_attempted", changed)
        ),
        "unique_changed_train_shadow_count": changed,
        "within_total61_budget": changed <= TOTAL_MAX_STEPS,
        "stage1_reference_exact": len(stage1_trajectory)
        == STAGE1_REFERENCE_STEP
        and all(
            all(record["v12_reference_exact_checks"].values())
            for record in stage1_trajectory
        ),
        "preclip_radial_projection_absent_every_record": all(
            record["radial_gradient_projection"]["evaluated"] is False
            for record in stage1_trajectory + stage2_trajectory
        ),
        "standard_projected_SGD_sequence": [
            "backward",
            "original_clip_grad_norm",
            "plain_SGD_step",
            "v12_project_actor6_endpoint_ball_projection",
        ],
        "candidate_payload_present_iff_GO": payload_present
        == decision.startswith("GO_"),
        "structured_stall_audit": stalled_audit,
        "frozen_v13_output_absent_before_and_after": terminal["checks"][
            "v13_output_absent_now"
        ]
        and post_terminal["checks"]["v13_output_absent_now"],
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    if not all(
        value
        for key, value in v14_contract.items()
        if key
        not in {
            "pre_cuda_runtime",
            "optimizer_step_calls_attempted",
            "unique_changed_train_shadow_count",
            "standard_projected_SGD_sequence",
            "structured_stall_audit",
        }
    ):
        raise ProtocolError(f"v14 production contract failed: {v14_contract}")
    audit["CW24_v14_contract"] = v14_contract
    result["exploratory_disclosure"] = {
        "train_only_result_is_not_promotion_evidence": True,
        "stage1_is_exact_frozen_v12_reproduction_not_new_selection": True,
        "stage2_target_loss_weights_and_budgets_are_exact_v13": True,
        "only_algorithmic_repair_is_removal_of_preclip_radial_projection": True,
        "PPO_term_is_policy_KL_not_new_rollout": True,
        "fulltrain_exact_CW11_comparator_required_after_GO": True,
        "specialist_then_broad_then_Gold_required_after_fulltrain_GO": True,
    }
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 2
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    if decision.startswith("NO_GO"):
        audit["NO_GO_material_audit"] = no_go_material_audit(result)
        no_go_material_audit(result)
    canonical_json(result)
    return result


def audit_only() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v14 output target must be absent during static audit")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v14 attempt marker must be absent during static audit")
    source = self_evidence(require_frozen=False)
    dependencies = dependency_evidence()
    terminal = validate_v13_terminal_state()
    transform = transform_v13_source()[1]
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_pass_unfrozen_draft",
        "source": source,
        "source_audit": source_audit(),
        "dependencies": dependencies,
        "frozen_v13_terminal_state": terminal,
        "in_memory_v13_transform": transform,
        "stage_design": {
            "stage1_steps": STAGE1_REFERENCE_STEP,
            "stage1_source": "exact frozen v13 and v12 replay",
            "stage2_max_steps": STAGE2_MAX_STEPS,
            "stage2_target_loss_weights_and_gates": "exact frozen v13",
            "stage2_optimizer_sequence": [
                "backward",
                "original_clip_grad_norm",
                "plain_SGD_step",
                "v12_project_actor6_endpoint_ball_projection",
            ],
            "preclip_radial_gradient_projection": False,
            "stage2_exact_actor_repeat": "structured_NO_GO_V14_STALLED",
            "stage1_exact_actor_repeat": "fail_closed_ProtocolError",
            "total_changed_train_shadow_cap": TOTAL_MAX_STEPS,
            "NO_GO_payload": None,
            "NO_GO_actor_bytes_or_delta_arrays": False,
        },
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent": True,
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
        "attempt_marker_absent": True,
        "frozen_v13_output": str(V13_OUTPUT.relative_to(ROOT)),
        "frozen_v13_output_absent": True,
        "pre_cuda_runtime": runtime,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "CUDA_initialized": False,
        "production_modules_executed": False,
        "writes_performed": 0,
        "draft_freeze_required_before_production": source["mode_octal"] != "0555",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        raise ProtocolError("output path is not the exact v14 target")
    if args.audit_only:
        print(canonical_json(audit_only()).decode("utf-8"), end="")
        return
    if not path_absent(OUTPUT):
        raise ProtocolError("v14 O_EXCL output already exists")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v14 one-shot attempt marker already exists")
    attempt = claim_attempt()
    result = production_run()
    attempt_post = regular_source(
        ATTEMPT_MARKER,
        attempt["publication"]["sha256"],
        0o444,
        "v14 attempt marker post-run",
    )[1]
    final_source = self_evidence(require_frozen=True)
    final_dependencies = dependency_evidence()
    final_terminal = validate_v13_terminal_state()
    inputs = result["inputs"]
    attempt_source = attempt["payload"]["source"]
    attempt_dependencies = attempt["payload"]["dependencies"]
    attempt_terminal = attempt["payload"]["frozen_v13_terminal_state"]
    binding_checks = {
        "attempt_source_equals_production_pre": attempt_source
        == binding_summary(inputs["CW24_v14_source"]),
        "attempt_source_equals_production_post": attempt_source
        == binding_summary(inputs["CW24_v14_source_post_run"]),
        "attempt_source_equals_final_rehash": attempt_source
        == binding_summary(final_source),
        "attempt_dependencies_equal_production_pre": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in inputs["CW24_v14_dependencies"].items()
        },
        "attempt_dependencies_equal_production_post": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in inputs["CW24_v14_dependencies_post_run"].items()
        },
        "attempt_dependencies_equal_final_rehash": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in final_dependencies.items()
        },
        "attempt_v13_terminal_equals_production_pre": attempt_terminal
        == v13_terminal_summary(inputs["CW24_v14_frozen_v13_terminal_state"]),
        "attempt_v13_terminal_equals_production_post": attempt_terminal
        == v13_terminal_summary(
            inputs["CW24_v14_frozen_v13_terminal_state_post_run"]
        ),
        "attempt_v13_terminal_equals_final": attempt_terminal
        == v13_terminal_summary(final_terminal),
        "frozen_v13_output_still_absent": path_absent(V13_OUTPUT),
    }
    if not all(binding_checks.values()):
        raise ProtocolError(f"v14 attempt-to-publication drift: {binding_checks}")
    inputs["CW24_v14_source_final_before_publication"] = final_source
    inputs["CW24_v14_dependencies_final_before_publication"] = final_dependencies
    inputs["CW24_v14_frozen_v13_terminal_state_final_before_publication"] = (
        final_terminal
    )
    result.setdefault("audit", {})["one_shot_attempt"] = {
        "claim": attempt,
        "post_run": attempt_post,
        "unchanged": attempt_post["sha256"] == attempt["publication"]["sha256"],
        "claim_to_publication_binding_checks": binding_checks,
    }
    if str(result["decision"]).startswith("NO_GO"):
        result["audit"]["NO_GO_material_audit_final"] = no_go_material_audit(
            result
        )
        no_go_material_audit(result)
    canonical_json(result)
    if not path_absent(V13_OUTPUT):
        raise ProtocolError("frozen v13 output appeared before v14 publication")
    publication = publish_o_excl(OUTPUT, canonical_json(result))
    endpoint = result["endpoint"]
    print(
        canonical_json(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "decision": result["decision"],
                "candidate_model_state_sha256": endpoint.get("trial", {}).get(
                    "candidate_model_state_sha256"
                ),
                "candidate_payload_present": endpoint.get("candidate_payload")
                is not None,
                "changed_candidate_train_shadow_count": endpoint.get(
                    "changed_candidate_train_shadow_count"
                ),
                "optimizer_step_calls_attempted": endpoint.get(
                    "optimizer_step_calls_attempted",
                    endpoint.get("changed_candidate_train_shadow_count"),
                ),
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
                "output": publication,
            }
        ).decode("utf-8"),
        end="",
    )


if __name__ == "__main__":
    main()

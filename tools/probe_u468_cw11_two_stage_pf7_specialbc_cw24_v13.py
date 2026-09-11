#!/usr/bin/env python3
"""Two-stage PF7-directed special-BC probe from exact CW11 and frozen B352.

Stage 1 deterministically replays frozen CW24 v12 through step 29: step 28 is
the first native-BF16 endpoint where both PF0 rows pass, while step 29 gives
each row one extra native tick of margin.  Stage 2 keeps the same actor6 SGD
optimizer and CW11-centered projection, but changes the training objective to
a PF7 target plus explicit PF0/zero and dynamic retention/top1 preservation
penalties.  Before clipping each stage-2 gradient, the SGD descent direction is
projected to remove any component pointing outward from the CW11 radius.
The first endpoint satisfying the complete v12 native hard gate is selected.

This is an unfrozen audit-first train-only draft.  It writes no checkpoint,
opens no new validation/test row, exposes no actor payload on NO_GO, and uses a
pre-CUDA O_EXCL attempt claim plus immutable-input rehashes before publication.
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
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_two_stage_pf7_specialbc_cw24_v13.py"
OUTPUT = ROOT / "artifacts/cw24_cw11_two_stage_pf7_specialbc_trainonly_v13.json"
ATTEMPT_MARKER = ROOT / "artifacts/.ptcg-cw24-cw11-two-stage-pf7-specialbc-v13-attempt.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-two-stage-pf7-specialbc-cw24-v13"
SEED = 202608052

V12_SOURCE = TOOLS / "probe_u468_cw11_projected_nonlinear_specialbc_cw24_v12.py"
V12_SOURCE_SHA256 = "003e476325a08ff5be400167fccbd43705fac9a4071d806101ddb02969ea2301"
V12_SOURCE_MODE = 0o555
V12_RESULT = ROOT / "artifacts/cw24_cw11_projected_nonlinear_specialbc_trainonly_v12.json"
V12_RESULT_SHA256 = "effee98dc03c6d505f7bc2007dae88e28190c0fbba80f22c7902363592b554dc"
V12_RESULT_MODE = 0o444
V1_SOURCE = TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py"
V1_SOURCE_SHA256 = "a3c96eed95a6b558c61303e2190eaa4f353970add4509e8770c3dfcc1621db39"
CW22_SOURCE = TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py"
CW22_SOURCE_SHA256 = "f307f3f26355125f97c857f14b97a73d9a7a4f2ea39ca8b0a578e801055e82b4"
CW23_SOURCE = TOOLS / "probe_u468_cw11_boundary_qp_specialbc_cw23_v1.py"
CW23_SOURCE_SHA256 = "4a2d209af71a7ea5d955bd373f1fbada936688fc5bc597615c41192dc360b4b8"
EQUALBLEND_SOURCE = TOOLS / "run_u468_raw_actor6_equalblend_sgd512_shadow.py"
EQUALBLEND_SOURCE_SHA256 = "e04f7b7579ef42d0c6f643db833779fb837ae86e3205d948b5564d08f8b30f8e"
RAWCUT_SOURCE = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v1.py"
RAWCUT_SOURCE_SHA256 = "9c5a7377d82b18e437ca89d064ce8be60554646573e3aca27f4e40f4989ab77c"
B352_SELECTION = ROOT / "artifacts/cw24_top1_b352_selection_v1.json"
B352_SELECTION_SHA256 = "1f72b26eccd436ca0d341f837ec42949f5823bf7f0be73461aaeab28166cc222"

TOTAL_MAX_STEPS = 61
PF0_FIRST_NATIVE_PASS_STEP = 28
STAGE1_REFERENCE_STEP = 29
STAGE2_MAX_STEPS = TOTAL_MAX_STEPS - STAGE1_REFERENCE_STEP
LEARNING_RATE = 5.0e-5
MAX_GRAD_NORM = 0.5
PROJECT_RADIUS = 9.99975e-4
HARD_RADIUS = 9.9998e-4
MAX_DYNAMIC_PAIRS = 64
ORDER_CONTEXT_WEIGHT = 8.0
PF_TERMINAL_MARGIN_TICK = 1.0 / 512.0
STAGE2_UNION_WEIGHT = 0.15
STAGE2_PF0_BC_WEIGHT = 0.10
STAGE2_PF7_BC_WEIGHT = 0.35
STAGE2_DOMINIC_BC_WEIGHT = 0.05
STAGE2_PF7_PAIR_WEIGHT = 1.50
STAGE2_PF0_PAIR_WEIGHT = 1.00
STAGE2_DYNAMIC_PAIR_WEIGHT = 0.75
STAGE2_ZERO_PAIR_WEIGHT = 0.25
STAGE2_KL_WEIGHT = 0.05

STAGE1_REFERENCE_ACTOR_SHA256 = (
    "29829209b76dd90a374a3bd7824738225b92e5e84fcf76f8735693680cc61cdf"
)
STAGE1_REFERENCE_MODEL_SHA256 = (
    "c7404a0432cdc3b81f9fbb2b7470f404e6856a7454a07e92be2421793c85e52a"
)
STAGE1_REFERENCE_DELTA_SHA256 = (
    "8ea62f63756548c55b73e042ec349c77e1aa205c3967727b5ffed3d48a86f380"
)


class ProtocolError(RuntimeError):
    """Fail-closed CW24 v13 protocol error."""


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


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    calls: list[tuple[str, int]] = []
    imports: list[str] = []
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append((node.func.id, int(node.lineno)))
            elif isinstance(node.func, ast.Attribute):
                calls.append((node.func.attr, int(node.lineno)))
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    counts = Counter(name for name, _ in calls)
    checks = {
        "syntax_valid": True,
        "exact_one_SGD_constructor": counts["SGD"] == 1,
        "exact_one_optimizer_step_site": counts["step"] == 1,
        "exact_one_backward_site": counts["backward"] == 1,
        "exact_one_gradient_clip_site": counts["clip_grad_norm_"] == 1,
        "exact_one_stage2_radial_projection_call": counts[
            "project_non_outward_sgd_direction"
        ]
        == 1,
        "exact_one_step29_fixed_pair_q_builder_call": counts[
            "build_stage2_pair_contract"
        ]
        == 1,
        "fixed_pair_q_uses_v12_native_helper": counts[
            "local_positive_bf16_q"
        ]
        == 1,
        "tempered_penalty_uses_per_pair_temperature": (
            "pair" + "[\"temperature\"]"
        ).encode() in source,
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_network_import": not any(
            name.split(".")[0] in {"requests", "urllib", "httpx", "socket"}
            for name in imports
        ),
        "single_result_publication_site": source.count(
            b"publish_o_excl(" + b"OUTPUT"
        )
        == 1,
        "single_attempt_claim_site": source.count(
            b"publish_o_excl(" + b"ATTEMPT_MARKER"
        )
        == 1,
        "two_stage_budget_exact": constants.get("TOTAL_MAX_STEPS") == 61
        and constants.get("PF0_FIRST_NATIVE_PASS_STEP") == 28
        and constants.get("STAGE1_REFERENCE_STEP") == 29,
        "stage2_budget_derived": STAGE2_MAX_STEPS == 32
        and STAGE2_MAX_STEPS
        == TOTAL_MAX_STEPS - STAGE1_REFERENCE_STEP,
        "projection_radii_exact": constants.get("PROJECT_RADIUS") == 9.99975e-4
        and constants.get("HARD_RADIUS") == 9.9998e-4,
        "stage2_weight_formula_exact": constants.get("STAGE2_UNION_WEIGHT")
        == 0.15
        and constants.get("STAGE2_PF0_BC_WEIGHT") == 0.10
        and constants.get("STAGE2_PF7_BC_WEIGHT") == 0.35
        and constants.get("STAGE2_DOMINIC_BC_WEIGHT") == 0.05
        and constants.get("STAGE2_PF7_PAIR_WEIGHT") == 1.50
        and constants.get("STAGE2_PF0_PAIR_WEIGHT") == 1.00
        and constants.get("STAGE2_DYNAMIC_PAIR_WEIGHT") == 0.75
        and constants.get("STAGE2_ZERO_PAIR_WEIGHT") == 0.25
        and constants.get("STAGE2_KL_WEIGHT") == 0.05,
        "dynamic_pair_cap": constants.get("MAX_DYNAMIC_PAIRS") == 64,
        "no_dynamic_pair_terminal_gate": (
            "all_active_pair_" + "thresholds"
        ).encode() not in source,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v13 source audit failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "call_counts": {
            name: counts[name]
            for name in (
                "SGD",
                "step",
                "backward",
                "clip_grad_norm_",
                "project_non_outward_sgd_direction",
                "build_stage2_pair_contract",
                "local_positive_bf16_q",
            )
        },
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
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
        "static_source_audit": source_audit()["pass"],
    }
    if not all(checks.values()):
        raise ProtocolError(f"v13 self evidence failed: {checks}")
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


def dependency_evidence() -> dict[str, Any]:
    specifications = {
        "frozen_v12_source": regular_source(
            V12_SOURCE, V12_SOURCE_SHA256, V12_SOURCE_MODE, "frozen v12 source"
        )[1],
        "frozen_v12_result": regular_source(
            V12_RESULT, V12_RESULT_SHA256, V12_RESULT_MODE, "frozen v12 result"
        )[1],
        "frozen_CW24_v1": regular_source(
            V1_SOURCE, V1_SOURCE_SHA256, 0o555, "frozen CW24 v1"
        )[1],
        "frozen_CW22": regular_source(
            CW22_SOURCE, CW22_SOURCE_SHA256, 0o555, "frozen CW22"
        )[1],
        "frozen_CW23": regular_source(
            CW23_SOURCE, CW23_SOURCE_SHA256, 0o555, "frozen CW23"
        )[1],
        "plain_SGD_reference": regular_source(
            EQUALBLEND_SOURCE,
            EQUALBLEND_SOURCE_SHA256,
            0o555,
            "plain SGD reference",
        )[1],
        "dynamic_pair_reference": regular_source(
            RAWCUT_SOURCE,
            RAWCUT_SOURCE_SHA256,
            0o555,
            "dynamic pair reference",
        )[1],
        "frozen_B352_selection": regular_source(
            B352_SELECTION,
            B352_SELECTION_SHA256,
            0o444,
            "frozen B352 selection",
        )[1],
    }
    return specifications


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


def load_locked_module(
    path: Path, expected_sha: str, expected_mode: int, module_name: str
) -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = regular_source(path, expected_sha, expected_mode, module_name)
    if module_name in sys.modules:
        raise ProtocolError(f"module name already occupied: {module_name}")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, evidence


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
                raise ProtocolError("short v13 publication write")
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
        raise ProtocolError(f"v13 publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def claim_attempt() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v13 output already exists before attempt claim")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v13 one-shot attempt was already consumed")
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    payload = {
        "schema_version": "ptcg-u468-cw11-two-stage-pf7-specialbc-cw24-v13-attempt",
        "status": "claimed_before_module_exec_or_CUDA",
        "source": binding_summary(source),
        "dependencies": {
            name: binding_summary(record) for name, record in dependencies.items()
        },
        "runtime": runtime,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent_by_lstat_at_claim": True,
        "CUDA_preflight_state": "not_imported_pending_production_validation",
        "single_seed": SEED,
        "PF0_first_simultaneous_native_pass_step": PF0_FIRST_NATIVE_PASS_STEP,
        "stage1_reference_steps": STAGE1_REFERENCE_STEP,
        "stage2_maximum_steps": STAGE2_MAX_STEPS,
        "total_maximum_changed_train_shadows": TOTAL_MAX_STEPS,
        "retry_authorized": False,
        "network_package_upload_submission": False,
    }
    publication = publish_o_excl(ATTEMPT_MARKER, canonical_json(payload))
    return {"payload": payload, "publication": publication}


def load_json_no_duplicates(raw: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ProtocolError(f"{label}: nonfinite JSON constant {value}")

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ProtocolError(f"{label}: root is not an object")
    return value


def pf0_ab_pass(gate: Mapping[str, Any]) -> bool:
    native = gate["PF_native"]
    return bool(native["pf0_boundary_a"]["pass"]) and bool(
        native["pf0_boundary_b"]["pass"]
    )


def load_v12_reference() -> tuple[dict[str, Any], dict[str, Any]]:
    raw, evidence = regular_source(
        V12_RESULT, V12_RESULT_SHA256, V12_RESULT_MODE, "frozen v12 result"
    )
    payload = load_json_no_duplicates(raw, "frozen v12 result")
    endpoint = payload.get("endpoint")
    trajectory = endpoint.get("trajectory") if isinstance(endpoint, Mapping) else None
    first_pf0 = None
    if isinstance(trajectory, list):
        for record in trajectory:
            if pf0_ab_pass(record["native_terminal_gate"]):
                first_pf0 = int(record["step"])
                break
    reference = (
        trajectory[STAGE1_REFERENCE_STEP - 1]
        if isinstance(trajectory, list) and len(trajectory) >= STAGE1_REFERENCE_STEP
        else {}
    )
    checks = {
        "schema_exact": payload.get("schema_version")
        == "ptcg-u468-cw11-projected-nonlinear-specialbc-cw24-v12",
        "NO_GO_exact": payload.get("status")
        == "NO_GO_CW24_PROJECTED_NONLINEAR_TRAIN_GATE"
        and isinstance(endpoint, Mapping)
        and endpoint.get("decision")
        == "NO_GO_CW24_PROJECTED_NONLINEAR_TRAIN_GATE",
        "no_payload": isinstance(endpoint, Mapping)
        and endpoint.get("candidate_payload") is None,
        "trajectory48": isinstance(trajectory, list) and len(trajectory) == 48,
        "unique_actor48": isinstance(trajectory, list)
        and len(
            {
                str(record["candidate_actor_float32_le_sha256"])
                for record in trajectory
            }
        )
        == 48,
        "first_PF0_AB_native_pass_step28": first_pf0
        == PF0_FIRST_NATIVE_PASS_STEP,
        "reference_actor_exact": reference.get(
            "candidate_actor_float32_le_sha256"
        )
        == STAGE1_REFERENCE_ACTOR_SHA256,
        "reference_model_exact": reference.get("candidate_model_state_sha256")
        == STAGE1_REFERENCE_MODEL_SHA256,
        "reference_delta_exact": reference.get("projection", {}).get(
            "actual_delta_float64_le_sha256"
        )
        == STAGE1_REFERENCE_DELTA_SHA256,
        "reference_PF7_still_false": reference.get("native_terminal_gate", {})
        .get("PF_native", {})
        .get("pf7_boundary", {})
        .get("pass")
        is False,
        "reference_PF0_each_has_one_q_cushion": all(
            float(reference.get("native_terminal_gate", {})
            .get("PF_native", {})
            .get(name, {})
            .get("native_fixed_pair_margin", float("-inf")))
            - float(reference.get("native_terminal_gate", {})
            .get("PF_native", {})
            .get(name, {})
            .get("threshold", float("inf")))
            >= PF_TERMINAL_MARGIN_TICK
            for name in ("pf0_boundary_a", "pf0_boundary_b")
        ),
        "reference_other_native_checks_true": all(
            value
            for name, value in reference.get("native_terminal_gate", {})
            .get("checks", {})
            .items()
            if name != "three_PF_native_order_set_and_margin"
        ),
        "source_binding_exact": payload.get("inputs", {})
        .get("CW24_v12_source", {})
        .get("sha256")
        == V12_SOURCE_SHA256,
    }
    if not all(checks.values()):
        raise ProtocolError(f"frozen v12 reference drift: {checks}")
    return payload, {
        "evidence": evidence,
        "checks": checks,
        "first_PF0_AB_native_pass_step": first_pf0,
        "stage1_buffered_transition_step": STAGE1_REFERENCE_STEP,
        "reference_actor_sha256": STAGE1_REFERENCE_ACTOR_SHA256,
        "reference_model_sha256": STAGE1_REFERENCE_MODEL_SHA256,
        "reference_delta_sha256": STAGE1_REFERENCE_DELTA_SHA256,
    }


def masked_policy_kl(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    teacher_policy_logits: Any,
    protected_mask: Any,
) -> Any:
    import torch

    option_mask = batch["option_mask"].bool()
    teacher = teacher_policy_logits.float().masked_fill(~option_mask, -1e9)
    current = outputs["policy_logits"].float().masked_fill(~option_mask, -1e9)
    teacher_log_probs = torch.log_softmax(teacher, dim=1)
    teacher_probs = teacher_log_probs.exp()
    current_log_probs = torch.log_softmax(current, dim=1)
    per_row = (teacher_probs * (teacher_log_probs - current_log_probs)).sum(dim=1)
    weights = batch["sample_weights"].float() * protected_mask.float()
    return (per_row * weights).sum() / weights.sum()


def tempered_pair_penalty(
    outputs: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]
) -> Any:
    import torch
    import torch.nn.functional as functional

    if not pairs:
        raise ProtocolError("empty stage2 pair penalty")
    logits = outputs["policy_logits"].float()
    terms = []
    for pair in pairs:
        row, positive, negative = (
            int(pair["row_index"]),
            int(pair["positive_option"]),
            int(pair["negative_option"]),
        )
        margin = logits[row, positive] - logits[row, negative]
        threshold = logits.new_tensor(float(pair["threshold"]))
        temperature_value = float(pair["temperature"])
        if not math.isfinite(temperature_value) or temperature_value <= 0.0:
            raise ProtocolError("stage2 pair has invalid frozen local BF16 q")
        temperature = logits.new_tensor(temperature_value)
        terms.append(
            temperature * functional.softplus((threshold - margin) / temperature)
        )
    result = torch.stack(terms).mean()
    if result.ndim != 0 or not bool(torch.isfinite(result)):
        raise ProtocolError("nonfinite stage2 tempered pair penalty")
    return result


def attach_dynamic_temperatures(pairs: Sequence[dict[str, Any]]) -> None:
    """Bind every dynamic pair's stored local-q threshold as its temperature."""

    for pair in pairs:
        temperature = float(pair["threshold"])
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ProtocolError("dynamic pair stored local q is invalid")
        pair["temperature"] = temperature
        pair["temperature_source"] = "stored_dynamic_local_BF16_q"


def build_stage2_pair_contract(
    outputs: Mapping[str, Any],
    contract: Mapping[str, Any],
    v12: ModuleType,
) -> dict[str, Any]:
    """Freeze actual step29 local BF16 q for every fixed stage-2 pair."""

    def with_q(pair: Mapping[str, Any], threshold: float, source: str) -> dict[str, Any]:
        row, positive, negative = (
            int(pair["row_index"]),
            int(pair["positive_option"]),
            int(pair["negative_option"]),
        )
        q = float(
            v12.local_positive_bf16_q(outputs, row, positive, negative)
        )
        if not math.isfinite(q) or q <= 0.0:
            raise ProtocolError("fixed stage2 pair local BF16 q is invalid")
        return {
            **dict(pair),
            "threshold": float(threshold),
            "threshold_source": source,
            "temperature": q,
            "temperature_source": "actual_step29_local_BF16_q",
        }

    pf7_source = next(
        pair
        for pair in contract["fixed_pairs"]
        if "pf7_boundary" in pair["origins"]
    )
    pf7_q_probe = with_q(
        pf7_source, 0.0, "temporary_probe_before_positive_q_target"
    )
    pf7_pair = {
        **pf7_q_probe,
        "threshold": float(pf7_q_probe["temperature"]),
        "threshold_source": "stage2_positive_one_actual_local_q_cushion",
    }
    pf0_pairs = []
    for pair in contract["fixed_pairs"]:
        if not any(
            str(origin).startswith("pf0_boundary") for origin in pair["origins"]
        ):
            continue
        q_probe = with_q(
            pair,
            float(pair["threshold"]),
            "temporary_probe_before_terminal_plus_q_target",
        )
        q_probe["threshold"] = float(pair["threshold"]) + float(
            q_probe["temperature"]
        )
        q_probe["threshold_source"] = "stage2_terminal_threshold_plus_actual_local_q"
        pf0_pairs.append(q_probe)
    zero_pairs = [
        with_q(pair, 0.0, "stage2_explicit_zero_margin_floor")
        for pair in contract["zero_contract"]
    ]
    if len(pf0_pairs) != 2 or len(zero_pairs) != 6:
        raise ProtocolError("stage2 fixed pair count drift")
    return {
        "frozen_at_global_step": STAGE1_REFERENCE_STEP,
        "PF7": pf7_pair,
        "PF0": pf0_pairs,
        "zero": zero_pairs,
        "local_q_values": {
            "PF7": float(pf7_pair["temperature"]),
            "PF0": [float(pair["temperature"]) for pair in pf0_pairs],
            "zero": [float(pair["temperature"]) for pair in zero_pairs],
        },
        "checks": {
            "PF7_one": True,
            "PF0_two": len(pf0_pairs) == 2,
            "zero_six": len(zero_pairs) == 6,
            "all_q_finite_positive": all(
                math.isfinite(float(pair["temperature"]))
                and float(pair["temperature"]) > 0.0
                for pair in [pf7_pair] + pf0_pairs + zero_pairs
            ),
        },
    }


def stage2_objective(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    teacher_policy_logits: Any,
    contract: Mapping[str, Any],
    stage2_pair_contract: Mapping[str, Any],
    preservation_pairs: Sequence[Mapping[str, Any]],
    ppo: ModuleType,
    v12: ModuleType,
) -> tuple[Any, dict[str, float]]:
    import torch

    union = v12.complete_ordered_scalar(
        outputs, v12.masked_batch(batch, contract["masks"]["union"]), ppo
    )
    pf0 = v12.complete_ordered_scalar(
        outputs, v12.masked_batch(batch, contract["masks"]["pf0"]), ppo
    )
    pf7 = v12.complete_ordered_scalar(
        outputs, v12.masked_batch(batch, contract["masks"]["pf7"]), ppo
    )
    dominic = v12.complete_ordered_scalar(
        outputs, v12.masked_batch(batch, contract["masks"]["dominic"]), ppo
    )
    base_soft = 0.5 * union + (pf0 + pf7 + dominic) / 6.0
    pf7_pair = dict(stage2_pair_contract["PF7"])
    pf0_pairs = [dict(pair) for pair in stage2_pair_contract["PF0"]]
    if len(pf0_pairs) != 2:
        raise ProtocolError("stage2 fixed PF0 preservation pair drift")
    dynamic_pairs = [dict(pair) for pair in preservation_pairs]
    if not dynamic_pairs:
        raise ProtocolError("stage2 exact-v12 transition lacks dynamic preservation pairs")
    zero_pairs = [dict(pair) for pair in stage2_pair_contract["zero"]]
    if len(zero_pairs) != 6:
        raise ProtocolError("stage2 explicit six-zero pair contract drift")
    pf7_penalty = tempered_pair_penalty(outputs, [pf7_pair])
    pf0_penalty = tempered_pair_penalty(outputs, pf0_pairs)
    dynamic_penalty = tempered_pair_penalty(outputs, dynamic_pairs)
    zero_penalty = tempered_pair_penalty(outputs, zero_pairs)
    kl = masked_policy_kl(
        outputs,
        batch,
        teacher_policy_logits,
        contract["masks"]["protected"],
    )
    total = (
        STAGE2_UNION_WEIGHT * union
        + STAGE2_PF0_BC_WEIGHT * pf0
        + STAGE2_PF7_BC_WEIGHT * pf7
        + STAGE2_DOMINIC_BC_WEIGHT * dominic
        + STAGE2_PF7_PAIR_WEIGHT * pf7_penalty
        + STAGE2_PF0_PAIR_WEIGHT * pf0_penalty
        + STAGE2_DYNAMIC_PAIR_WEIGHT * dynamic_penalty
        + STAGE2_ZERO_PAIR_WEIGHT * zero_penalty
        + STAGE2_KL_WEIGHT * kl
    )
    scalars = (
        union,
        pf0,
        pf7,
        dominic,
        base_soft,
        pf7_penalty,
        pf0_penalty,
        dynamic_penalty,
        zero_penalty,
        kl,
        total,
    )
    if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in scalars):
        raise ProtocolError("nonfinite/non-scalar stage2 objective")
    return total, {
        "L_union": float(union.detach().cpu()),
        "L_pf0": float(pf0.detach().cpu()),
        "L_pf7": float(pf7.detach().cpu()),
        "L_dominic": float(dominic.detach().cpu()),
        "base_soft_loss": float(base_soft.detach().cpu()),
        "L_pf7_tempered_pair": float(pf7_penalty.detach().cpu()),
        "L_pf0_tempered_pairs": float(pf0_penalty.detach().cpu()),
        "L_dynamic_tempered_pairs": float(dynamic_penalty.detach().cpu()),
        "L_six_zero_tempered_pairs": float(zero_penalty.detach().cpu()),
        "L_protected_KL": float(kl.detach().cpu()),
        "L_total": float(total.detach().cpu()),
        "PF7_pair_count": 1,
        "fixed_PF0_plus_one_q_pair_count": len(pf0_pairs),
        "dynamic_stored_threshold_pair_count": len(dynamic_pairs),
        "explicit_zero_threshold_pair_count": len(zero_pairs),
    }


def stage2_obligations(
    outputs: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    contract: Mapping[str, Any],
    v1: ModuleType,
    v12: ModuleType,
) -> list[dict[str, Any]]:
    candidates = v12.dynamic_obligations(
        outputs, rows, baseline, current, contract, v1
    )
    result = []
    for pair in candidates:
        origins = {str(value) for value in pair["origins"]}
        if "PF_false:pf7_boundary" in origins:
            continue
        result.append(dict(pair))
    return result


def project_non_outward_sgd_direction(
    parameters: Mapping[str, Any],
    names: Sequence[str],
    cw11_actor: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove an outward radial component from d=-grad before clipping."""

    import torch

    first = parameters[names[0]]
    radial_dot = torch.zeros((), dtype=torch.float64, device=first.device)
    delta_square = torch.zeros_like(radial_dot)
    direction_square_before = torch.zeros_like(radial_dot)
    deltas: dict[str, Any] = {}
    for name in names:
        gradient = parameters[name].grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise ProtocolError("stage2 radial projection saw missing/nonfinite gradient")
        delta = parameters[name].detach() - cw11_actor[name]
        direction = -gradient.detach()
        deltas[name] = delta
        radial_dot += (delta.double() * direction.double()).sum()
        delta_square += delta.double().square().sum()
        direction_square_before += direction.double().square().sum()
    radial_before = float(radial_dot.detach().cpu())
    delta_square_value = float(delta_square.detach().cpu())
    direction_square_before_value = float(direction_square_before.detach().cpu())
    if not all(
        math.isfinite(value)
        for value in (
            radial_before,
            delta_square_value,
            direction_square_before_value,
        )
    ) or delta_square_value <= 0.0:
        raise ProtocolError("stage2 radial projection geometry is invalid")
    removed = radial_before > 0.0
    coefficient = radial_before / delta_square_value if removed else 0.0
    if removed:
        with torch.no_grad():
            for name in names:
                parameters[name].grad.add_(deltas[name], alpha=coefficient)

    radial_after_tensor = torch.zeros_like(radial_dot)
    direction_square_after = torch.zeros_like(radial_dot)
    for name in names:
        adjusted_direction = -parameters[name].grad.detach()
        radial_after_tensor += (
            deltas[name].double() * adjusted_direction.double()
        ).sum()
        direction_square_after += adjusted_direction.double().square().sum()
    radial_after = float(radial_after_tensor.detach().cpu())
    direction_square_after_value = float(direction_square_after.detach().cpu())
    tolerance = max(1.0e-12, abs(radial_before) * 1.0e-5)
    checks = {
        "stage2_only_call_contract": True,
        "finite_geometry": all(
            math.isfinite(value)
            for value in (
                radial_after,
                coefficient,
                direction_square_after_value,
            )
        ),
        "non_outward_after_projection": radial_after <= tolerance,
        "gradient_finite_after_projection": all(
            bool(torch.isfinite(parameters[name].grad).all()) for name in names
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"stage2 non-outward gradient projection failed: {checks}")
    return {
        "evaluated": True,
        "placement": "after_backward_before_clip",
        "descent_direction": "negative_gradient",
        "radial_dot_before": radial_before,
        "outward_component_removed": removed,
        "removed_coefficient": coefficient,
        "radial_dot_after": radial_after,
        "non_outward_tolerance": tolerance,
        "CW11_delta_l2": math.sqrt(delta_square_value),
        "descent_l2_before": math.sqrt(direction_square_before_value),
        "descent_l2_after": math.sqrt(direction_square_after_value),
        "checks": checks,
    }


def run_two_stage_core(
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cw20: ModuleType,
    cw19: ModuleType,
    cw15: ModuleType,
    modules: Mapping[str, ModuleType],
    *,
    cw22: ModuleType,
    cw23: ModuleType,
    v1: ModuleType,
    v12: ModuleType,
    reference_payload: Mapping[str, Any],
    reference_audit: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    context_checks = cw22.validate_exact_context(context, cw20, cw15)
    helper = context["helper"]
    model = context["model"]
    ppo = helper.ppo
    dependency_checks = {
        "BC_source_exact_CW15": v12.sha256_file(Path(helper.bc.__file__))
        == cw15.MODULE_SHAS[cw15.BC],
        "PPO_source_exact_CW15": v12.sha256_file(Path(ppo.__file__))
        == cw15.MODULE_SHAS[cw15.PPO],
        "v12_stage_constants_exact": v12.SEED == SEED
        and v12.LEARNING_RATE == LEARNING_RATE
        and v12.MAX_GRAD_NORM == MAX_GRAD_NORM
        and v12.PROJECT_RADIUS == PROJECT_RADIUS
        and v12.HARD_RADIUS == HARD_RADIUS
        and v12.MAX_STEPS == 48
        and TOTAL_MAX_STEPS == 61
        and v12.MAX_DYNAMIC_PAIRS == MAX_DYNAMIC_PAIRS,
    }
    if not all(dependency_checks.values()):
        raise ProtocolError(f"v13 runtime dependency drift: {dependency_checks}")
    reference_trajectory = reference_payload["endpoint"]["trajectory"]
    batch_cpu, cache_audit = v1.load_train_b352(
        rows, context["model_config"], helper.bc, ppo, cw22
    )
    device = next(model.parameters()).device
    named = dict(model.named_parameters())
    requires_grad_before = {
        name: bool(parameter.requires_grad) for name, parameter in named.items()
    }
    for parameter in named.values():
        parameter.requires_grad_(False)
        parameter.grad = None
    for name in cw22.EXPECTED_ACTOR_NAMES:
        named[name].requires_grad_(True)
    parameters = {name: named[name] for name in cw22.EXPECTED_ACTOR_NAMES}
    parameter_sequence = modules["geometry"].configure_actor6(model)
    if len(parameter_sequence) != len(cw22.EXPECTED_ACTOR_NAMES) or any(
        parameters[name] is not value
        for name, value in zip(cw22.EXPECTED_ACTOR_NAMES, parameter_sequence)
    ):
        raise ProtocolError("actor6 implementation tensor identity drift")
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        raise ProtocolError("actor6 storage is not float32")
    cw11_actor = cw20.clone_actor(parameters, cw22.EXPECTED_ACTOR_NAMES)
    cw11_flat = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
    try:
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch_cpu.items()
        }
        outputs = ppo.model_forward(model, batch, device)
        with torch.no_grad():
            baseline = v12.evaluate_snapshot(outputs, batch, rows, ppo, cw22)
        baseline_native = {
            "count_logits": outputs["count_logits"].detach().clone(),
            "value_logits": outputs["value_logits"].detach().clone(),
        }
        teacher_policy_logits = outputs["policy_logits"].detach().clone()
        contract = v12.build_row_contract(
            rows, batch, outputs, baseline, cw23, v1
        )
        dynamic_pairs: list[dict[str, Any]] = []
        baseline_obligations = v12.dynamic_obligations(
            outputs, rows, baseline, baseline, contract, v1
        )
        baseline_added = v12.merge_dynamic_pairs(
            dynamic_pairs, baseline_obligations, 0
        )
        total_loss, loss_audit = v12.objective_components(
            outputs,
            batch,
            teacher_policy_logits,
            contract["fixed_pairs"],
            dynamic_pairs,
            contract["masks"],
            ppo,
        )
        baseline_loss_audit = dict(loss_audit)
        baseline_base_soft_loss = float(loss_audit["base_soft_loss"])
        if not math.isfinite(baseline_base_soft_loss):
            raise ProtocolError("baseline base soft loss is nonfinite")

        optimizer = torch.optim.SGD(
            list(parameters.values()),
            lr=LEARNING_RATE,
            momentum=0.0,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            maximize=False,
            foreach=False,
            differentiable=False,
            fused=False,
        )
        if optimizer.state or optimizer.state_dict()["state"]:
            raise ProtocolError("fresh plain SGD unexpectedly has state")

        actor_hashes = {
            hashlib.sha256(
                cw20.actor_bytes(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
            ).hexdigest()
        }
        stage1_trajectory: list[dict[str, Any]] = []
        stage2_trajectory: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        last_trial: dict[str, Any] | None = None
        stage2_started = False
        stage2_pair_contract: dict[str, Any] | None = None
        for global_step in range(1, TOTAL_MAX_STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradients = [parameters[name].grad for name in cw22.EXPECTED_ACTOR_NAMES]
            if any(gradient is None for gradient in gradients) or any(
                not bool(torch.isfinite(gradient).all())
                for gradient in gradients
                if gradient is not None
            ):
                raise ProtocolError("actor6 backward produced missing/nonfinite gradient")
            if global_step > STAGE1_REFERENCE_STEP:
                radial_gradient_projection = project_non_outward_sgd_direction(
                    parameters, cw22.EXPECTED_ACTOR_NAMES, cw11_actor
                )
            else:
                radial_gradient_projection = {
                    "evaluated": False,
                    "placement": "stage1_exact_v12_replay_no_gradient_projection",
                    "outward_component_removed": False,
                }
            preclip_norm = torch.nn.utils.clip_grad_norm_(
                list(parameters.values()), MAX_GRAD_NORM
            )
            preclip_norm_value = float(preclip_norm.detach().cpu())
            if not math.isfinite(preclip_norm_value):
                raise ProtocolError("nonfinite actor6 gradient norm")
            optimizer.step()
            if optimizer.state or optimizer.state_dict()["state"]:
                raise ProtocolError("plain SGD acquired forbidden optimizer state")

            projection = v12.project_actor6(
                parameters,
                cw22.EXPECTED_ACTOR_NAMES,
                cw11_flat,
                cw20,
                np,
                torch,
            )
            actor_bytes = cw20.actor_bytes(
                parameters, cw22.EXPECTED_ACTOR_NAMES, np
            )
            actor_hash = hashlib.sha256(actor_bytes).hexdigest()
            if actor_hash in actor_hashes:
                raise ProtocolError("repeated actor6 state in two-stage trajectory")
            actor_hashes.add(actor_hash)
            candidate_hash = helper.model_state_sha256(model.state_dict())
            outputs = ppo.model_forward(model, batch, device)
            with torch.no_grad():
                current = v12.evaluate_snapshot(outputs, batch, rows, ppo, cw22)

            if global_step <= STAGE1_REFERENCE_STEP:
                obligations = v12.dynamic_obligations(
                    outputs, rows, baseline, current, contract, v1
                )
                added = v12.merge_dynamic_pairs(
                    dynamic_pairs, obligations, global_step
                )
                total_loss, loss_audit = v12.objective_components(
                    outputs,
                    batch,
                    teacher_policy_logits,
                    contract["fixed_pairs"],
                    dynamic_pairs,
                    contract["masks"],
                    ppo,
                )
            else:
                if not stage2_started:
                    raise ProtocolError("stage2 loop entered before objective switch")
                obligations = stage2_obligations(
                    outputs, rows, baseline, current, contract, v1, v12
                )
                added = v12.merge_dynamic_pairs(
                    dynamic_pairs, obligations, global_step
                )
                attach_dynamic_temperatures(dynamic_pairs)
                if stage2_pair_contract is None:
                    raise ProtocolError("stage2 fixed pair contract is absent")
                total_loss, loss_audit = stage2_objective(
                    outputs,
                    batch,
                    teacher_policy_logits,
                    contract,
                    stage2_pair_contract,
                    dynamic_pairs,
                    ppo,
                    v12,
                )

            changed_names = [
                name
                for name in cw22.EXPECTED_ACTOR_NAMES
                if not torch.equal(parameters[name], cw11_actor[name])
            ]
            gate = v12.terminal_gate(
                outputs=outputs,
                rows=rows,
                current=current,
                baseline=baseline,
                baseline_native=baseline_native,
                contract=contract,
                base_soft_loss=float(loss_audit["base_soft_loss"]),
                baseline_base_soft_loss=baseline_base_soft_loss,
                actual_delta=projection["actual_delta"],
                candidate_hash=candidate_hash,
                changed_names=changed_names,
                model=model,
                parameters=parameters,
                helper=helper,
                cw20=cw20,
                cw22=cw22,
            )
            diagnostics = v12.soft_floor_diagnostics(rows, baseline, current, cw22)
            common_trial = {
                "step": global_step,
                "candidate_model_state_sha256": candidate_hash,
                "candidate_actor_float32_le_sha256": actor_hash,
                "gradient_norm_before_clip": preclip_norm_value,
                "gradient_clip_max_norm": MAX_GRAD_NORM,
                "radial_gradient_projection": radial_gradient_projection,
                "loss": loss_audit,
                "projection": {
                    key: value
                    for key, value in projection.items()
                    if key != "actual_delta"
                },
                "dynamic_pairs_added": [dict(value) for value in added],
                "dynamic_pair_count": len(dynamic_pairs),
                "native_terminal_gate": gate,
                "soft_floor_diagnostics": diagnostics,
                "pass": gate["pass"],
            }
            last_trial = common_trial

            if global_step <= STAGE1_REFERENCE_STEP:
                reference = reference_trajectory[global_step - 1]
                exact_checks = {
                    "stage1_radial_projection_not_evaluated": common_trial[
                        "radial_gradient_projection"
                    ]["evaluated"]
                    is False,
                    "actor_sha": common_trial[
                        "candidate_actor_float32_le_sha256"
                    ]
                    == reference["candidate_actor_float32_le_sha256"],
                    "model_sha": common_trial["candidate_model_state_sha256"]
                    == reference["candidate_model_state_sha256"],
                    "gradient_norm": common_trial["gradient_norm_before_clip"]
                    == reference["gradient_norm_before_clip"],
                    "loss": common_trial["loss"] == reference["loss"],
                    "projection": common_trial["projection"]
                    == reference["projection"],
                    "dynamic_pairs_added": common_trial["dynamic_pairs_added"]
                    == reference["dynamic_pairs_added"],
                    "dynamic_pair_count": common_trial["dynamic_pair_count"]
                    == reference["dynamic_pair_count"],
                    "native_terminal_gate": common_trial["native_terminal_gate"]
                    == reference["native_terminal_gate"],
                    "soft_floor_diagnostics": common_trial[
                        "soft_floor_diagnostics"
                    ]
                    == reference["soft_floor_diagnostics"],
                    "pass": common_trial["pass"] == reference["pass"],
                }
                if not all(exact_checks.values()):
                    raise ProtocolError(
                        f"stage1 diverged from frozen v12 at step {global_step}: {exact_checks}"
                    )
                observed_pf0 = pf0_ab_pass(gate)
                expected_pf0 = global_step >= PF0_FIRST_NATIVE_PASS_STEP
                if observed_pf0 != expected_pf0:
                    raise ProtocolError("stage1 first PF0 A/B native pass drift")
                if gate["pass"]:
                    raise ProtocolError("stage1 unexpectedly reached complete hard pass")
                stage1_trajectory.append(
                    {
                        **common_trial,
                        "stage": 1,
                        "v12_reference_exact_checks": exact_checks,
                    }
                )
                if global_step == STAGE1_REFERENCE_STEP:
                    stage2_pair_contract = build_stage2_pair_contract(
                        outputs, contract, v12
                    )
                    transition_checks = {
                        "actor_sha_exact": actor_hash
                        == STAGE1_REFERENCE_ACTOR_SHA256,
                        "model_sha_exact": candidate_hash
                        == STAGE1_REFERENCE_MODEL_SHA256,
                        "delta_sha_exact": projection[
                            "actual_delta_float64_le_sha256"
                        ]
                        == STAGE1_REFERENCE_DELTA_SHA256,
                        "PF0_A_B_both_native_pass": pf0_ab_pass(gate),
                        "PF0_A_B_exact_buffered_native_cells": all(
                            float(
                                gate["PF_native"][name][
                                    "native_fixed_pair_margin"
                                ]
                            )
                            == 2.0 / 512.0
                            and float(gate["PF_native"][name]["threshold"])
                            == 1.0 / 512.0
                            for name in ("pf0_boundary_a", "pf0_boundary_b")
                        ),
                        "stage2_pair_q_contract_pass": all(
                            stage2_pair_contract["checks"].values()
                        ),
                        "PF7_native_not_yet_pass": gate["PF_native"][
                            "pf7_boundary"
                        ]["pass"]
                        is False,
                        "all_other_native_checks_pass": all(
                            value
                            for name, value in gate["checks"].items()
                            if name != "three_PF_native_order_set_and_margin"
                        ),
                    }
                    stage2_candidates = stage2_obligations(
                        outputs, rows, baseline, current, contract, v1, v12
                    )
                    transition_added = v12.merge_dynamic_pairs(
                        dynamic_pairs,
                        stage2_candidates,
                        STAGE1_REFERENCE_STEP,
                    )
                    attach_dynamic_temperatures(dynamic_pairs)
                    transition_checks.update(
                        {
                            "exact_two_frozen_v12_dynamic_pairs": len(dynamic_pairs)
                            == 2,
                            "dynamic_local_q_values_exact": sorted(
                                float(pair["temperature"])
                                for pair in dynamic_pairs
                            )
                            == [0.0078125, 0.03125],
                        }
                    )
                    if not all(transition_checks.values()):
                        raise ProtocolError(
                            f"stage1 transition gate failed: {transition_checks}"
                        )
                    total_loss, loss_audit = stage2_objective(
                        outputs,
                        batch,
                        teacher_policy_logits,
                        contract,
                        stage2_pair_contract,
                        dynamic_pairs,
                        ppo,
                        v12,
                    )
                    stage2_started = True
                    transition = {
                        "global_step": global_step,
                        "checks": transition_checks,
                        "preservation_pairs_added_at_switch": [
                            dict(value) for value in transition_added
                        ],
                        "dynamic_preservation_pair_count": len(dynamic_pairs),
                        "frozen_stage2_pair_contract": stage2_pair_contract,
                        "stage2_initial_loss": loss_audit,
                    }
            else:
                if global_step != STAGE1_REFERENCE_STEP + len(stage2_trajectory) + 1:
                    raise ProtocolError("stage2 global/local step accounting drift")
                stage2_trial = {
                    **common_trial,
                    "stage": 2,
                    "stage2_step": global_step - STAGE1_REFERENCE_STEP,
                    "PF7_strong_direction": {
                        "target_margin": float(
                            stage2_pair_contract["PF7"]["threshold"]
                        ),
                        "frozen_local_q": float(
                            stage2_pair_contract["PF7"]["temperature"]
                        ),
                        "pair_weight": STAGE2_PF7_PAIR_WEIGHT,
                        "PF7_BC_weight": STAGE2_PF7_BC_WEIGHT,
                    },
                }
                stage2_trajectory.append(stage2_trial)
                if gate["pass"]:
                    selected = {
                        "global_step": global_step,
                        "stage2_step": global_step - STAGE1_REFERENCE_STEP,
                        "candidate_model_state_sha256": candidate_hash,
                        "actor_bytes": actor_bytes,
                        "actor_hash": actor_hash,
                        "actual_delta": projection["actual_delta"].copy(),
                        "actual_delta_sha256": projection[
                            "actual_delta_float64_le_sha256"
                        ],
                        "changed_names": changed_names,
                        "loss": dict(loss_audit),
                        "gate": gate,
                        "diagnostics": diagnostics,
                        "dynamic_pairs": [dict(value) for value in dynamic_pairs],
                    }
                    break

        if not stage2_started or len(stage1_trajectory) != STAGE1_REFERENCE_STEP:
            raise ProtocolError("stage1 did not reach exact deterministic transition")
        changed_shadows = len(stage1_trajectory) + len(stage2_trajectory)
        if changed_shadows > TOTAL_MAX_STEPS:
            raise ProtocolError("total changed train-shadow budget exceeded")
        common = {
            "context_checks": context_checks,
            "dependency_checks": dependency_checks,
            "cache": cache_audit,
            "baseline_checks": contract["baseline_checks"],
            "objective_mask_counts": contract["mask_counts"],
            "baseline_loss": baseline_loss_audit,
            "baseline_dynamic_pairs_added": [dict(value) for value in baseline_added],
            "fixed_pair_contract": [dict(value) for value in contract["fixed_pairs"]],
            "zero_margin_contract": [dict(value) for value in contract["zero_contract"]],
            "frozen_v12_reference": dict(reference_audit),
            "stage1_transition": transition,
            "stage1_trajectory": stage1_trajectory,
            "stage2_trajectory": stage2_trajectory,
            "radial_gradient_projection_contract": {
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
            "optimizer_contract": {
                "name": "SGD",
                "plain": True,
                "single_optimizer_across_both_stages": True,
                "state_entries_terminal": len(optimizer.state),
                "learning_rate": LEARNING_RATE,
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
                "maximize": False,
                "foreach": False,
                "differentiable": False,
                "fused": False,
                "max_grad_norm": MAX_GRAD_NORM,
            },
            "projection_contract": {
                "center": "exact_CW11_actor6_float32",
                "projection_radius": PROJECT_RADIUS,
                "hard_actual_radius": HARD_RADIUS,
                "project_after_every_step": True,
            },
            "two_stage_contract": {
                "stage1": "exact frozen-v12 trajectory through buffered step29 after first PF0 A/B pass at step28",
                "stage1_steps": STAGE1_REFERENCE_STEP,
                "PF0_first_native_pass_step": PF0_FIRST_NATIVE_PASS_STEP,
                "stage2": "PF7 BC plus positive-one-actual-local-q pair target and non-outward SGD direction",
                "stage2_max_steps": STAGE2_MAX_STEPS,
                "stage2_formula": (
                    "0.15*L_union + 0.10*L_pf0 + 0.35*L_pf7 + 0.05*L_dominic "
                    "+ 1.50*H_pf7 + 1.00*mean(H_pf0_A_B) + 0.75*mean(H_dynamic) "
                    "+ 0.25*mean(H_six_zero) + 0.05*KL_protected"
                ),
                "H_definition": "pair.temperature*softplus((pair.threshold-margin)/pair.temperature)",
                "pair_temperature": "actual step29 local BF16 q for PF and zero; stored local q for dynamic",
                "PF0_preservation": "two terminal-threshold-plus-local-q pair penalties plus native hard gate",
                "zero_preservation": "six always-explicit threshold-zero pair penalties plus native hard gate",
                "retention_top1_preservation": "sticky dynamic stored-threshold penalties plus native hard gates",
                "stage2_gradient_projection": "after backward before clip remove outward radial component of d=-g",
                "dynamic_pairs_terminal_gate": False,
                "first_all_hard_pass_stop": True,
            },
            "changed_candidate_train_shadow_count": changed_shadows,
            "max_steps": TOTAL_MAX_STEPS,
        }
        if selected is None:
            return {
                "decision": "NO_GO_CW24_TWO_STAGE_PF7_TRAIN_GATE",
                "reason": "NO_ALL_NATIVE_HARD_PASS_WITHIN_32_PF7_DIRECTED_STEPS",
                **common,
                "trial": last_trial,
                "candidate_payload": None,
            }

        if any(record["pass"] for record in stage2_trajectory[:-1]):
            raise ProtocolError("selected endpoint is not first stage2 all-hard pass")
        if stage2_pair_contract is None:
            raise ProtocolError("selected endpoint lacks frozen stage2 pair contract")
        layout = cw20.actor_layout(parameters, cw22.EXPECTED_ACTOR_NAMES)
        payload = {
            "anchor": {
                "reconstruction_base": "original_raw_U468",
                "raw_checkpoint": str(cw23.RAW.relative_to(ROOT)),
                "raw_checkpoint_file_sha256": cw23.RAW_SHA256,
                "raw_model_state_sha256": cw22.RAW_MODEL_SHA256,
                "CW11_provenance_model_state_sha256": cw22.CW11_MODEL_SHA256,
                "CW11_provenance_vector_float64_le_sha256": cw22.CW11_VECTOR_SHA256,
                "CW11_provenance_active_pair_ledger_sha256": cw22.CW11_LEDGER_SHA256,
                "frozen_v12_source_sha256": V12_SOURCE_SHA256,
                "frozen_v12_result_sha256": V12_RESULT_SHA256,
                "stage1_reference_actor_sha256": STAGE1_REFERENCE_ACTOR_SHA256,
                "terminal_model_state_sha256": selected[
                    "candidate_model_state_sha256"
                ],
            },
            "formula": "load_original_raw_U468_then_replace_all_six_absolute_actor_float32_tensors_from_v13_payload",
            "actor_names": list(cw22.EXPECTED_ACTOR_NAMES),
            "changed_actor_names": selected["changed_names"],
            "actor_layout": layout,
            "actor_layout_sha256": hashlib.sha256(
                json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "candidate_actor_float32_le": cw20.xz_payload(selected["actor_bytes"]),
            "actual_additional_from_CW11_l2": selected["gate"][
                "actual_additional_from_CW11_l2"
            ],
            "actual_delta_float64_le_sha256": selected["actual_delta_sha256"],
            "selected_global_step": selected["global_step"],
            "selected_stage2_step": selected["stage2_step"],
            "stage2_pair_contract_sha256": hashlib.sha256(
                canonical_json(stage2_pair_contract)
            ).hexdigest(),
            "dynamic_pair_ledger_sha256": hashlib.sha256(
                canonical_json(selected["dynamic_pairs"])
            ).hexdigest(),
            "selection_sha256": v1.ROWS_CANONICAL_SHA256,
            "cache_sha256": cache_audit["cache_sha256"],
        }

        modules["cutting"].restore_raw_actor(
            modules["ram"], parameter_sequence, context["raw_actor"], torch
        )
        raw_restored_hash = helper.model_state_sha256(model.state_dict())
        decoded = cw20.decode_xz(payload["candidate_actor_float32_le"])
        cw20.copy_actor_bytes(
            parameters,
            cw22.EXPECTED_ACTOR_NAMES,
            layout,
            decoded,
            np,
            torch,
        )
        reconstructed_hash = helper.model_state_sha256(model.state_dict())
        replay_flat = cw20.flat_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, np)
        replay_delta = replay_flat - cw11_flat
        replay_outputs = ppo.model_forward(model, batch, device)
        with torch.no_grad():
            replay_snapshot = v12.evaluate_snapshot(
                replay_outputs, batch, rows, ppo, cw22
            )
        _, replay_loss = stage2_objective(
            replay_outputs,
            batch,
            teacher_policy_logits,
            contract,
            stage2_pair_contract,
            selected["dynamic_pairs"],
            ppo,
            v12,
        )
        replay_changed_names = [
            name
            for name in cw22.EXPECTED_ACTOR_NAMES
            if not torch.equal(parameters[name], cw11_actor[name])
        ]
        replay_gate = v12.terminal_gate(
            outputs=replay_outputs,
            rows=rows,
            current=replay_snapshot,
            baseline=baseline,
            baseline_native=baseline_native,
            contract=contract,
            base_soft_loss=float(replay_loss["base_soft_loss"]),
            baseline_base_soft_loss=baseline_base_soft_loss,
            actual_delta=replay_delta,
            candidate_hash=reconstructed_hash,
            changed_names=replay_changed_names,
            model=model,
            parameters=parameters,
            helper=helper,
            cw20=cw20,
            cw22=cw22,
        )
        replay_diagnostics = v12.soft_floor_diagnostics(
            rows, baseline, replay_snapshot, cw22
        )
        reconstruction_checks = {
            "raw_U468_restore_exact": raw_restored_hash == cw22.RAW_MODEL_SHA256,
            "candidate_hash_exact": reconstructed_hash
            == selected["candidate_model_state_sha256"],
            "candidate_actor_bytes_exact": cw20.actor_bytes(
                parameters, cw22.EXPECTED_ACTOR_NAMES, np
            )
            == selected["actor_bytes"],
            "actual_delta_sha_exact": v12.float64_sha(replay_delta)
            == selected["actual_delta_sha256"],
            "nonactor_exact_raw": cw20.nonactor_sha(
                model, cw22.EXPECTED_ACTOR_NAMES, helper
            )
            == cw22.RAW_NONACTOR_SHA256,
            "repeat_B352_native_terminal_gate": replay_gate["pass"],
            "repeat_stage2_loss_exact": replay_loss == selected["loss"],
            "repeat_soft_diagnostics_exact": replay_diagnostics
            == selected["diagnostics"],
        }
        reconstruction_pass = all(reconstruction_checks.values())
        final_checks = {
            "exact_CW11_context": all(context_checks.values()),
            "dependency_sources": all(dependency_checks.values()),
            "B352_exact_train_only_cache": all(cache_audit["checks"].values()),
            "stage1_exact_v12_through_buffered_step29": all(
                all(record["v12_reference_exact_checks"].values())
                for record in stage1_trajectory
            ),
            "stage2_radial_gradient_projection_every_step": all(
                record["radial_gradient_projection"]["evaluated"] is True
                and all(record["radial_gradient_projection"]["checks"].values())
                for record in stage2_trajectory
            ),
            "first_stage2_all_native_hard_pass": selected["gate"]["pass"],
            "absolute_payload_reconstruction_and_repeat_gate": reconstruction_pass,
        }
        final_pass = all(final_checks.values())
        return {
            "decision": "GO_CW24_TWO_STAGE_PF7_TRAIN_GATE"
            if final_pass
            else "NO_GO_CW24_TWO_STAGE_PF7_TRAIN_GATE",
            "reason": "FIRST_PF7_DIRECTED_ALL_HARD_ENDPOINT_PASSED_AND_REPLAYED"
            if final_pass
            else "ABSOLUTE_PAYLOAD_RECONSTRUCTION_OR_REPEAT_GATE_FAILED",
            **common,
            "trial": stage2_trajectory[selected["stage2_step"] - 1],
            "pure_payload_reconstruction": {
                "checks": reconstruction_checks,
                "pass": reconstruction_pass,
                "raw_model_state_sha256": raw_restored_hash,
                "candidate_model_state_sha256": reconstructed_hash,
                "repeat_native_terminal_gate": replay_gate,
                "repeat_stage2_loss": replay_loss,
                "repeat_soft_floor_diagnostics": replay_diagnostics,
            },
            "final_checks": final_checks,
            "candidate_payload": payload if final_pass else None,
        }
    finally:
        cw20.restore_actor(parameters, cw22.EXPECTED_ACTOR_NAMES, cw11_actor, torch)
        for name, parameter in named.items():
            parameter.requires_grad_(requires_grad_before[name])
            parameter.grad = None


def production_run() -> dict[str, Any]:
    pre_cuda_runtime = validate_pre_cuda_runtime()
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    reference_payload, reference_audit = load_v12_reference()
    v12, v12_evidence = load_locked_module(
        V12_SOURCE, V12_SOURCE_SHA256, V12_SOURCE_MODE, "cw24_v13_frozen_v12"
    )
    v1, v1_evidence = v12.load_locked_module(
        v12.V1, v12.V1_SHA256, v12.SOURCE_MODE, "cw24_v13_v1"
    )
    rows, selection_audit = v1.load_selection()
    cw23, cw23_evidence = v1.load_locked_module(
        v1.CW23, v1.CW23_SHA256, v1.CW23_MODE, "cw24_v13_cw23"
    )
    cw22, cw22_evidence = cw23.load_cw22()
    original_core = cw22.run_targeted_core
    original_seed = cw22.SEED

    def bound_core(
        context: Mapping[str, Any],
        ignored_rows: Sequence[Mapping[str, Any]],
        cw20: ModuleType,
        cw19: ModuleType,
        cw15: ModuleType,
        modules: Mapping[str, ModuleType],
    ) -> dict[str, Any]:
        if len(ignored_rows) != 256:
            raise ProtocolError("frozen CW22 reconstruction callback row drift")
        return run_two_stage_core(
            context,
            rows,
            cw20,
            cw19,
            cw15,
            modules,
            cw22=cw22,
            cw23=cw23,
            v1=v1,
            v12=v12,
            reference_payload=reference_payload,
            reference_audit=reference_audit,
        )

    cw22.run_targeted_core = bound_core
    cw22.SEED = SEED
    try:
        result = cw22.production_run()
    finally:
        cw22.run_targeted_core = original_core
        cw22.SEED = original_seed

    post_source = self_evidence(require_frozen=True)
    post_dependencies = dependency_evidence()
    rehash_checks = {
        "v13_source_byte_identity": post_source == source,
        "v12_source_and_result_byte_identity": post_dependencies == dependencies,
    }
    if not all(rehash_checks.values()):
        raise ProtocolError(f"post-run immutable input drift: {rehash_checks}")
    endpoint = result["endpoint"]
    decision = str(endpoint["decision"])
    payload_present = endpoint.get("candidate_payload") is not None
    if (decision.startswith("GO_")) != payload_present:
        raise ProtocolError("v13 GO/payload equivalence failed")
    if decision.startswith("NO_GO") and payload_present:
        raise ProtocolError("v13 NO_GO exposed forbidden candidate payload")
    changed = int(endpoint.get("changed_candidate_train_shadow_count", -1))
    if changed < STAGE1_REFERENCE_STEP or changed > TOTAL_MAX_STEPS:
        raise ProtocolError("v13 changed train-shadow budget drift")

    frozen_engine_selection = result["selection"]
    result["schema_version"] = SCHEMA
    result["status"] = decision
    result["decision"] = decision
    result["seed"] = SEED
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_v12_replay_then_PF7_directed_special_BC"
    )
    result["selection"] = {
        "rows": 352,
        "sha256": v1.ROWS_CANONICAL_SHA256,
        "checks": selection_audit["checks"],
        "counts": selection_audit["counts"],
        "scope": selection_audit["scope"],
        "optimization_contract": {
            "single_seed": SEED,
            "single_plain_actor6_SGD_optimizer": True,
            "stage1_exact_v12_steps": STAGE1_REFERENCE_STEP,
            "stage1_first_PF0_A_B_native_pass_step": PF0_FIRST_NATIVE_PASS_STEP,
            "stage1_buffered_transition_step": STAGE1_REFERENCE_STEP,
            "stage2_maximum_steps": STAGE2_MAX_STEPS,
            "total_maximum_changed_train_shadows": TOTAL_MAX_STEPS,
            "PF7_target": "positive_one_actual_step29_local_BF16_q",
            "PF7_pair_weight": STAGE2_PF7_PAIR_WEIGHT,
            "stage2_full_weight_vector": {
                "union": STAGE2_UNION_WEIGHT,
                "PF0_BC": STAGE2_PF0_BC_WEIGHT,
                "PF7_BC": STAGE2_PF7_BC_WEIGHT,
                "Dominic_BC": STAGE2_DOMINIC_BC_WEIGHT,
                "PF7_H": STAGE2_PF7_PAIR_WEIGHT,
                "PF0_H": STAGE2_PF0_PAIR_WEIGHT,
                "dynamic_H": STAGE2_DYNAMIC_PAIR_WEIGHT,
                "zero_H": STAGE2_ZERO_PAIR_WEIGHT,
                "protected_KL": STAGE2_KL_WEIGHT,
            },
            "PF0_zero_retention_top1_native_hard_gates": True,
            "preservation_dynamic_pairs_training_only": True,
            "six_zero_pairs_always_explicit_in_stage2_loss": True,
            "per_pair_frozen_local_BF16_q_temperature": True,
            "stage2_non_outward_gradient_projection_before_clip": True,
            "projection_radius": PROJECT_RADIUS,
            "hard_actual_radius": HARD_RADIUS,
            "first_all_native_hard_pass_stop": True,
        },
        "frozen_v12_selection": reference_payload["selection"],
        "frozen_engine_B256_reconstruction_selection": frozen_engine_selection,
    }
    historical = result["historical_exact_CW11_replay"]
    historical.pop("new_validation_rows_opened_for_CW22_selection_or_candidate", None)
    historical.pop("new_validation_rows_opened_for_CW24_v12_selection_or_candidate", None)
    historical["new_validation_rows_opened_for_CW24_v13_selection_or_candidate"] = 0
    inputs = result.setdefault("inputs", {})
    inputs["CW24_v13_source"] = source
    inputs["CW24_v13_source_post_run"] = post_source
    inputs["CW24_v13_dependencies"] = dependencies
    inputs["CW24_v13_dependencies_post_run"] = post_dependencies
    inputs["CW24_v13_frozen_v12_module"] = v12_evidence
    inputs["CW24_v13_frozen_v12_reference"] = reference_audit
    inputs["CW24_v13_v1_loader"] = v1_evidence
    inputs["CW24_v13_CW23_loader"] = cw23_evidence
    inputs["CW24_v13_CW22_engine"] = cw22_evidence
    audit = result.setdefault("audit", {})
    audit["source"] = source
    audit["CW24_v13_contract"] = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "pre_cuda_runtime": pre_cuda_runtime,
        "post_run_rehash_checks": rehash_checks,
        "train_only": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 1,
        "optimizer_step_calls": changed,
        "optimizer_step_calls_within_total61": changed <= TOTAL_MAX_STEPS,
        "stage1_reference_exact": all(
            all(record["v12_reference_exact_checks"].values())
            for record in endpoint["stage1_trajectory"]
        ),
        "stage1_no_radial_gradient_projection": all(
            record["radial_gradient_projection"]["evaluated"] is False
            for record in endpoint["stage1_trajectory"]
        ),
        "stage2_radial_gradient_projection_every_step": all(
            record["radial_gradient_projection"]["evaluated"] is True
            and all(record["radial_gradient_projection"]["checks"].values())
            for record in endpoint["stage2_trajectory"]
        ),
        "candidate_payload_present_iff_GO": payload_present
        == decision.startswith("GO_"),
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "train_only_result_is_not_promotion_evidence": True,
        "stage1_is_exact_frozen_v12_reproduction_not_new_selection": True,
        "PPO_term_is_policy_KL_not_new_rollout": True,
        "fulltrain_exact_CW11_comparator_required_after_GO": True,
        "specialist_then_broad_then_Gold_required_after_fulltrain_GO": True,
    }
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 2
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    canonical_json(result)
    return result


def audit_only() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v13 output target must be absent during static audit")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v13 attempt marker must be absent during static audit")
    source = self_evidence(require_frozen=False)
    dependencies = dependency_evidence()
    _, reference = load_v12_reference()
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_pass_unfrozen_draft",
        "source": source,
        "source_audit": source_audit(),
        "dependencies": dependencies,
        "frozen_v12_reference": reference,
        "stage_design": {
            "stage1_steps": STAGE1_REFERENCE_STEP,
            "PF0_first_simultaneous_native_pass_step": PF0_FIRST_NATIVE_PASS_STEP,
            "stage1_stop": "buffered frozen-v12 step29 with both PF0 margins exactly 2over512",
            "stage2_max_steps": STAGE2_MAX_STEPS,
            "stage2_target": "PF7 positive one actual-step29 local BF16 q cushion",
            "stage2_pair_temperature": "per-pair frozen local BF16 q",
            "six_zero_pairs_always_explicit": True,
            "stage2_non_outward_SGD_direction_projection": "after backward before clip",
            "total_changed_train_shadow_cap": TOTAL_MAX_STEPS,
            "first_all_native_hard_pass_stop": True,
            "NO_GO_payload": None,
        },
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent": True,
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
        "attempt_marker_absent": True,
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
        raise ProtocolError("output path is not the exact v13 target")
    if args.audit_only:
        print(canonical_json(audit_only()).decode("utf-8"), end="")
        return
    if not path_absent(OUTPUT):
        raise ProtocolError("v13 O_EXCL output already exists")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v13 one-shot attempt marker already exists")
    attempt = claim_attempt()
    result = production_run()
    attempt_post = regular_source(
        ATTEMPT_MARKER,
        attempt["publication"]["sha256"],
        0o444,
        "v13 attempt marker post-run",
    )[1]
    final_source = self_evidence(require_frozen=True)
    final_dependencies = dependency_evidence()
    attempt_source = attempt["payload"]["source"]
    attempt_dependencies = attempt["payload"]["dependencies"]
    production_source_pre = result["inputs"]["CW24_v13_source"]
    production_source_post = result["inputs"]["CW24_v13_source_post_run"]
    production_dependencies_pre = result["inputs"]["CW24_v13_dependencies"]
    production_dependencies_post = result["inputs"][
        "CW24_v13_dependencies_post_run"
    ]
    binding_checks = {
        "attempt_source_equals_production_pre": attempt_source
        == binding_summary(production_source_pre),
        "attempt_source_equals_production_post": attempt_source
        == binding_summary(production_source_post),
        "attempt_source_equals_final_rehash": attempt_source
        == binding_summary(final_source),
        "attempt_dependencies_equal_production_pre": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in production_dependencies_pre.items()
        },
        "attempt_dependencies_equal_production_post": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in production_dependencies_post.items()
        },
        "attempt_dependencies_equal_final_rehash": attempt_dependencies
        == {
            name: binding_summary(record)
            for name, record in final_dependencies.items()
        },
    }
    if not all(binding_checks.values()):
        raise ProtocolError(f"v13 attempt-to-publication drift: {binding_checks}")
    result["inputs"]["CW24_v13_source_final_before_publication"] = final_source
    result["inputs"][
        "CW24_v13_dependencies_final_before_publication"
    ] = final_dependencies
    result.setdefault("audit", {})["one_shot_attempt"] = {
        "claim": attempt,
        "post_run": attempt_post,
        "unchanged": attempt_post["sha256"] == attempt["publication"]["sha256"],
        "claim_to_publication_binding_checks": binding_checks,
    }
    canonical_json(result)
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
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
                "output": publication,
            }
        ).decode("utf-8"),
        end="",
    )


if __name__ == "__main__":
    main()

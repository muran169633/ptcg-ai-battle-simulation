#!/usr/bin/env python3
"""One-shot 24,050-row train-only gate for the current-parent boundary-QP child.

UNAVAILABLE CLOSED-LINEAGE DRAFT: the sole B256 attempt returned NO_GO because
its nominal L2 exceeded the frozen 0.001 cap; the capped endpoint fixed 0/5
targets and regressed guards.  This file was never frozen or run and formal mode
is permanently disabled.  It is retained only as an unexecuted design record.
"""

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
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
ARTIFACTS = ROOT / "artifacts"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_u456_g8_guarded_u468_currentparent_boundaryqp_fulltrain24050_gate_v1.py"
SCHEMA = "ptcg-u456-g8-guarded-u468-currentparent-boundaryqp-fulltrain24050-gate-v1"
PREREG_SCHEMA = f"{SCHEMA}-execution-preregistration-v1"

# These are the only intentionally unresolved draft bindings.  Replace them
# exactly once after the sole B256 run publishes a GO result, then freeze this
# runner and create the execution preregistration before formal execution.
SOLVER_SHA256 = "d19c5338329365cc4bb4ba8586df29aa1c616588c6ae5640265e0391e73b4c85"
CANDIDATE_RESULT = ARTIFACTS / (
    "ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_design202608143.result.json"
)
CANDIDATE_RESULT_SHA256 = "BIND_AFTER_B256_GO_CANDIDATE_RESULT_SHA256"

SOLVER = TOOLS / "run_u456_g8_guarded_u468_currentparent_boundaryqp_specialbc_v1.py"
SOLVER_SCHEMA = "ptcg-u456-g8-guarded-u468-currentparent-boundaryqp-specialbc-v1"
SOLVER_GO_STATUS = "GO_CURRENT_PARENT_BOUNDARYQP_B256"

MASTER = ARTIFACTS / (
    "ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.master_preregistration.json"
)
MASTER_SHA256 = "0efc5a08ba1a83ed26b0b518c826c5744d67e98895d2da21449e56291116bfa8"
CORRECTION_V2 = ARTIFACTS / (
    "ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.master_correction_v2.json"
)
CORRECTION_V2_SHA256 = "6accc12201900aba1a50d64d4ffd21796d36180cdb44990694b768a5a351263a"
CORRECTION_V3 = ARTIFACTS / (
    "ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.master_correction_v3.json"
)
CORRECTION_V3_SHA256 = "709972d0774536fd79ab5627d7b970e0b6ab4b5f2987d3e0e1f5f762b200461d"
CORRECTION_V4 = ARTIFACTS / (
    "ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.master_correction_v4.json"
)
CORRECTION_V4_SHA256 = "060fd238feba40908cb90fee81c81d4e4a981dfade274faf416233deae886857"

PARENT = ARTIFACTS / (
    "ppo_u456_g8_guarded3x4x96_p12_design202608141/ppo_stage/block3/"
    "B_gold_league/seed-202608141/checkpoints/update-0468.pt"
)
PARENT_FILE_SHA256 = "a9290745b8eb58704c4617594ac5ab49500c6e65ecd9f273a8285c12547fbb83"
PARENT_MODEL_SHA256 = "832c724277d6c2263d76ddbf41622f4ba06621ec3155fa8b7ae2591203db1c5e"
PARENT_ACTOR_FLOAT32_LE_SHA256 = (
    "03822d2e9dfd09843493894a22bc4ed22b608dbd7c4a05183ff8e507f773dc8a"
)
PARENT_NONACTOR_SHA256 = (
    "ce84a183fbe82d40861938ba3cc8f1fa93b19ff53aa14ecbc8eee18ccd00db87"
)

FORMAL = TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v3.py"
FORMAL_SHA256 = "2d12ce9e76f6672911948dcc6458573899ec45fc39e748510544001025ede939"
C758 = TOOLS / "run_u456_g8_guarded3x4_u468_exactp12_fulltrain24050_gate_v1.py"
C758_SHA256 = "c75829fe5bf7d5b2bcf4069fac8bd505a86e531c795f63caf3438a9171eaefc2"
CW20 = TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py"
CW20_SHA256 = "8f5b64c8a04b3de9a6d2890aac582967a50f4d708f3a11a64cbd395423d907d2"

PREREGISTRATION = ARTIFACTS / (
    "ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.fulltrain24050_execution_preregistration.json"
)
ATTEMPT = ARTIFACTS / (
    ".ptcg-u456-g8-guarded-u468-currentparent-boundaryqp-"
    "fulltrain24050-attempt-v1.json"
)
OUTPUT = ARTIFACTS / (
    "ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_"
    "design202608143.fulltrain24050_result.json"
)

PANEL_ORDER = ("flg", "pokemonfan", "core5")
EXPECTED_ROWS = {"flg": 9443, "pokemonfan": 9487, "core5": 5120}
EXPECTED_CONTEXT34_ROWS = {"flg": 42, "pokemonfan": 38, "core5": 20}
EXPECTED_TOTAL_ROWS = 24050
DATASETS = {
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
    "pokemonfan": ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "core5": ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
}
DATA_SHA256 = {
    "flg": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    "pokemonfan": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}
DATA_MODE = {"flg": 0o664, "pokemonfan": 0o664, "core5": 0o600}

ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
EFFECTIVE_CHANGED_ACTOR5 = ACTOR6_NAMES[:-1]
ACTOR6_DIMENSION = 65793
ACTOR6_RAW_BYTES = ACTOR6_DIMENSION * 4
ACTOR_L2_HARD_CAP = 1.0e-3
POLICY_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
PF_MINIMUM_NET_AND_WC = {
    "set_exact": 3,
    "hybrid_order_exact": 3,
    "ordered_exact": 5,
    "top1_correct": 3,
}
PLACEHOLDER_PREFIX = "BIND_AFTER_B256_GO_"
LINEAGE_CLOSED_NO_GO = True
CLOSURE_REASON = (
    "sole B256 boundary-QP candidate failed frozen cap/target/guard gates; "
    "correction_v2 forbids relaxation, retry, a second candidate, or fulltrain"
)


class ProtocolError(RuntimeError):
    """Fail-closed protocol error."""


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


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        raise ProtocolError(f"{label}: identity changed during held-fd read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    if len(payload) != int(after_fd.st_size) or (
        expected_sha256 is not None and digest != expected_sha256
    ):
        raise ProtocolError(f"{label}: SHA/size drift")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(stat.S_IMODE(after_fd.st_mode), "04o"),
        "single_link_regular_held_fd_identity_exact": True,
    }


def strict_json(
    payload: bytes,
    label: str,
    *,
    require_canonical: bool = True,
) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ProtocolError(f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ProtocolError(f"{label}: nonfinite JSON token {token}")
        ),
    )
    if not isinstance(value, dict) or (
        require_canonical and canonical_json(value) != payload
    ):
        raise ProtocolError(f"{label}: JSON object/canonical encoding drift")
    return value


def publish(path: Path, payload: bytes) -> dict[str, Any]:
    if path.parent != ARTIFACTS or not path.parent.is_dir():
        raise ProtocolError(f"publication parent drift: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    observed = None
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(fd, payload[offset:])
            if count <= 0:
                raise ProtocolError(f"short publication: {path}")
            offset += count
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        observed = os.fstat(fd)
    finally:
        os.close(fd)
    reloaded, record = read_regular(path, sha256_bytes(payload), 0o444, path.name)
    if observed is None or int(observed.st_size) != len(payload) or reloaded != payload:
        raise ProtocolError(f"publication verification failed: {path}")
    return record


def import_frozen(path: Path, digest: str, name: str) -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = read_regular(path, digest, 0o555, name)
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    code = compile(source, str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module, evidence


def validate_runtime() -> dict[str, bool]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"runtime drift: {checks}")
    return checks


def binding_state() -> dict[str, Any]:
    pending = {
        "solver_sha256": SOLVER_SHA256.startswith(PLACEHOLDER_PREFIX),
        "candidate_result_path": CANDIDATE_RESULT.name.startswith(PLACEHOLDER_PREFIX),
        "candidate_result_sha256": CANDIDATE_RESULT_SHA256.startswith(
            PLACEHOLDER_PREFIX
        ),
    }
    return {
        "pending": pending,
        "lineage_closed_no_go": LINEAGE_CLOSED_NO_GO,
        "closure_reason": CLOSURE_REASON,
        "all_bound": not LINEAGE_CLOSED_NO_GO
        and not any(pending.values())
        and is_sha256(SOLVER_SHA256)
        and is_sha256(CANDIDATE_RESULT_SHA256),
    }


def static_source_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    forbidden_imports = {"requests", "urllib", "http", "socket", "subprocess", "kaggle"}
    forbidden_calls = {"backward", "step", "save", "savez", "unlink", "rename", "replace", "rmtree"}
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            imports.extend(
                {"line": node.lineno, "name": name}
                for name in names
                if name.split(".", 1)[0] in forbidden_imports
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden_calls:
                calls.append({"line": node.lineno, "name": node.func.attr})
    checks = {
        "ast_parse": True,
        "no_network_subprocess_submission_import": not imports,
        "no_training_checkpoint_or_destructive_call": not calls,
        "c758_reference_exact": C758_SHA256
        == "c75829fe5bf7d5b2bcf4069fac8bd505a86e531c795f63caf3438a9171eaefc2",
        "actor6_dimension_exact": ACTOR6_DIMENSION == 65793,
        "actor6_scope_exact": len(ACTOR6_NAMES) == 6
        and len(set(ACTOR6_NAMES)) == 6
        and EFFECTIVE_CHANGED_ACTOR5 == ACTOR6_NAMES[:-1],
        "rows_exact_24050": sum(EXPECTED_ROWS.values()) == EXPECTED_TOTAL_ROWS,
        "pf_thresholds_exact": PF_MINIMUM_NET_AND_WC
        == {
            "set_exact": 3,
            "hybrid_order_exact": 3,
            "ordered_exact": 5,
            "top1_correct": 3,
        },
        "absolute_payload_reconstruction_present": b"candidate_actor_float32_le"
        in source,
        "count_value_bit_exact_gate_present": all(
            token in source
            for token in (
                b"native_count_logits_mismatch_count",
                b"native_value_logits_mismatch_count",
            )
        ),
        # The sole occurrence is this audit literal, not an argparse option.
        "candidate_not_a_cli_sweep_axis": source.count(b"--candidate-result") == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(
            f"static source audit failed: {checks}; imports={imports}; calls={calls}"
        )
    return {"checks": checks, "forbidden_imports": imports, "forbidden_calls": calls}


def expected_gate_contract() -> dict[str, Any]:
    return {
        "data": {
            "split": "train_only",
            "panel_order": list(PANEL_ORDER),
            "rows": dict(EXPECTED_ROWS),
            "context34_rows": dict(EXPECTED_CONTEXT34_ROWS),
            "total_unique_rows": EXPECTED_TOTAL_ROWS,
            "validation_test_broad_gold_opened": False,
        },
        "candidate": {
            "same_single_B256_candidate": True,
            "reconstruction": "guarded_U468_parent_plus_absolute_actor6_payload",
            "authorized_scope": list(ACTOR6_NAMES),
            "expected_changed_scope": list(EFFECTIVE_CHANGED_ACTOR5),
            "common_logit_bias_bit_exact_parent": True,
            "nonactor74_bit_exact_parent": True,
            "radial_actor6_l2_at_most": ACTOR_L2_HARD_CAP,
            "second_candidate_constructed": False,
        },
        "evaluation": {
            "model_order_inside_every_batch": ["parent", "candidate"],
            "same_model_instance": True,
            "batch_size": 256,
            "workers": 0,
            "device": "cuda:0",
            "native_output_dtype": "torch.bfloat16",
            "deterministic_algorithms": True,
            "logical_evaluations": 6,
        },
        "policy_and_count_gate": {
            "pokemonfan_minimum_net_and_wc": dict(PF_MINIMUM_NET_AND_WC),
            "pokemonfan_context34_policy_and_count_minimum_net": 0,
            "flg_all_six_policy_and_count_minimum_net": 0,
            "core5_all_six_policy_and_count_minimum_net": 0,
            "flg_core5_policy_cw": "reported_not_independent_zero_CW_gate",
        },
        "immutable_output_gate": {
            "count_logits_native_bit_exact_every_row": True,
            "value_logits_native_bit_exact_every_row": True,
            "count_value_fingerprints_and_correctness_exact": True,
        },
        "decision": {
            "pass": "GO_CURRENT_PARENT_BOUNDARYQP_FULLTRAIN",
            "fail": "NO_GO_CURRENT_PARENT_BOUNDARYQP_FULLTRAIN_CLOSE_LINEAGE",
            "retry_or_threshold_adjustment": False,
            "specialist_preregistration_authorized_only_on_pass": True,
            "official_unique_changed_candidate_count_consumed": 0,
            "package_upload_submission_authorized": False,
        },
    }


def expected_bindings(self_sha256: str) -> dict[str, Any]:
    return {
        "runner": {
            "path": str(SCRIPT.relative_to(ROOT)),
            "sha256": self_sha256,
            "mode_octal": "0555",
        },
        "solver": {
            "path": str(SOLVER.relative_to(ROOT)),
            "sha256": SOLVER_SHA256,
            "mode_octal": "0555",
        },
        "candidate_B256_result": {
            "path": str(CANDIDATE_RESULT.relative_to(ROOT)),
            "sha256": CANDIDATE_RESULT_SHA256,
            "mode_octal": "0444",
            "required_status": SOLVER_GO_STATUS,
        },
        "master": {"path": str(MASTER.relative_to(ROOT)), "sha256": MASTER_SHA256},
        "corrections": [
            {"path": str(CORRECTION_V2.relative_to(ROOT)), "sha256": CORRECTION_V2_SHA256},
            {"path": str(CORRECTION_V3.relative_to(ROOT)), "sha256": CORRECTION_V3_SHA256},
            {"path": str(CORRECTION_V4.relative_to(ROOT)), "sha256": CORRECTION_V4_SHA256},
        ],
        "parent": {
            "path": str(PARENT.relative_to(ROOT)),
            "sha256": PARENT_FILE_SHA256,
            "runtime_model_state_sha256": PARENT_MODEL_SHA256,
        },
        "formal_v3": {
            "path": str(FORMAL.relative_to(ROOT)),
            "sha256": FORMAL_SHA256,
            "mode_octal": "0555",
        },
        "c758_fulltrain_reference": {
            "path": str(C758.relative_to(ROOT)),
            "sha256": C758_SHA256,
            "mode_octal": "0555",
        },
        "actor_payload_codec": {
            "path": str(CW20.relative_to(ROOT)),
            "sha256": CW20_SHA256,
            "mode_octal": "0555",
        },
        "train_archives": {
            panel: {
                "path": str(DATASETS[panel].relative_to(ROOT)),
                "sha256": DATA_SHA256[panel],
                "mode_octal": format(DATA_MODE[panel], "04o"),
            }
            for panel in PANEL_ORDER
        },
    }


def formal_command_template(self_sha256: str) -> list[str]:
    return [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        str(SCRIPT),
        "--mode",
        "formal",
        "--expected-tool-sha256",
        self_sha256,
        "--expected-preregistration-sha256",
        "<LOCKED_PREREGISTRATION_SHA256>",
    ]


def read_control_plane() -> tuple[dict[str, Any], dict[str, Any]]:
    specifications = (
        ("master", MASTER, MASTER_SHA256),
        ("correction_v2", CORRECTION_V2, CORRECTION_V2_SHA256),
        ("correction_v3", CORRECTION_V3, CORRECTION_V3_SHA256),
        ("correction_v4", CORRECTION_V4, CORRECTION_V4_SHA256),
    )
    values: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for name, path, digest in specifications:
        payload, record = read_regular(path, digest, 0o444, name)
        values[name] = strict_json(payload, name, require_canonical=False)
        evidence[name] = record
    master = values["master"]
    correction_v2 = values["correction_v2"]
    correction_v3 = values["correction_v3"]
    correction_v4 = values["correction_v4"]
    fulltrain = master.get("fulltrain_gate_if_b256_passes", {})
    checks = {
        "master_schema": master.get("schema_version")
        == "ptcg-u456-g8-guarded-u468-currentparent-boundaryqp-master-preregistration-v1",
        "master_locked_before_candidate": master.get("status")
        == "locked_before_parent_profile_or_changed_candidate_construction",
        "master_parent_exact": master.get("parent", {}).get("file_sha256")
        == PARENT_FILE_SHA256
        and master.get("parent", {}).get("runtime_model_state_sha256")
        == PARENT_MODEL_SHA256,
        "master_same_candidate_once": fulltrain.get("same_candidate_exactly_once") is True,
        "master_pf_thresholds": fulltrain.get("pokemonfan_minimum_net_and_wc")
        == PF_MINIMUM_NET_AND_WC,
        "master_context_count_nonnegative": fulltrain.get(
            "pokemonfan_context34_and_count_minimum_net"
        )
        == 0,
        "master_flg_core_nonnegative": fulltrain.get(
            "flg_and_core5_all_policy_and_count_minimum_net"
        )
        == 0,
        "master_count_value_logits_exact": fulltrain.get(
            "count_and_value_logits_bit_exact_every_row"
        )
        is True,
        "v2_cap_0p001": math.isclose(
            float(correction_v2.get("corrected_value", math.inf)),
            ACTOR_L2_HARD_CAP,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "v3_net_not_zero_cw": correction_v3.get("corrected_value")
        == "reported_but_not_a_zero_CW_hard_gate",
        "v4_zero_candidate_selection_correction": correction_v4.get(
            "candidate_accounting", {}
        ).get("changed_candidates_constructed_before_v4")
        == 0,
    }
    if not all(checks.values()):
        raise ProtocolError(f"control-plane drift: {checks}")
    return values, {"inputs": evidence, "checks": checks}


def verify_preregistration(
    expected_sha256: str,
    self_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, evidence = read_regular(
        PREREGISTRATION,
        expected_sha256,
        0o444,
        "fulltrain execution preregistration",
    )
    value = strict_json(payload, "fulltrain execution preregistration")
    template = formal_command_template(self_sha256)
    checks = {
        "schema": value.get("schema_version") == PREREG_SCHEMA,
        "status": value.get("status")
        == "locked_after_B256_GO_before_only_fulltrain_attempt",
        "bindings_exact": value.get("bindings") == expected_bindings(self_sha256),
        "gate_contract_exact": value.get("gate_contract") == expected_gate_contract(),
        "formal_command_template": value.get("formal_command_template") == template,
        "formal_command_template_sha256": value.get("formal_command_template_sha256")
        == sha256_bytes(canonical_json(template)),
        "attempt_path": value.get("output_contract", {}).get("attempt")
        == str(ATTEMPT.relative_to(ROOT)),
        "output_path": value.get("output_contract", {}).get("result")
        == str(OUTPUT.relative_to(ROOT)),
        "one_attempt": value.get("output_contract", {}).get("formal_attempts_authorized")
        == 1,
        "outputs_absent_at_lock": value.get("output_contract", {}).get(
            "attempt_and_result_absent_at_lock"
        )
        is True,
        "retry_forbidden": value.get("output_contract", {}).get("retry_authorized")
        is False,
    }
    if not all(checks.values()):
        raise ProtocolError(f"execution preregistration drift: {checks}")
    return value, {**evidence, "checks": checks}


def all_true(mapping: Any) -> bool:
    return isinstance(mapping, Mapping) and bool(mapping) and all(
        value is True for value in mapping.values()
    )


def validate_candidate_document(
    document: Mapping[str, Any],
    result_evidence: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    payload = document.get("candidate_payload")
    self_record = document.get("self", {})
    parent_record = document.get("input_lock", {}).get("parent", {})
    accounting = document.get("candidate_accounting", {})
    scope = document.get("scope", {})
    candidate_integrity = document.get("candidate_integrity", {})
    b256 = document.get("b256", {})
    restore = document.get("restore", {})
    checks = {
        "candidate_result_hash_bound": result_evidence.get("sha256")
        == CANDIDATE_RESULT_SHA256,
        "solver_schema": document.get("schema_version") == SOLVER_SCHEMA,
        "solver_GO": document.get("status") == SOLVER_GO_STATUS,
        "solver_self_path": self_record.get("path") == str(SOLVER.relative_to(ROOT)),
        "solver_self_sha": self_record.get("sha256") == SOLVER_SHA256,
        "solver_self_mode": self_record.get("mode_octal") == "0555",
        "parent_file_exact": parent_record.get("sha256") == PARENT_FILE_SHA256,
        "parent_checks_true": all_true(document.get("parent_checkpoint_checks")),
        "parent_actor_nonactor_checks_true": all_true(
            document.get("parent_actor_nonactor_checks")
        ),
        "B256_checks_true": all_true(b256.get("checks")),
        "candidate_integrity_checks_true": all_true(candidate_integrity.get("checks")),
        "payload_present": isinstance(payload, Mapping),
        "restore_pass": restore.get("pass") is True,
        "one_candidate": accounting.get("changed_candidate_train_shadow_count") == 1
        and accounting.get("parent_only_profile_changed_candidates") == 0
        and accounting.get("official_unique_changed_candidate_count_consumed") == 0
        and accounting.get("endpoint_sweep") is False
        and accounting.get("retry") is False,
        "train_only_scope": scope.get("train_only") is True
        and scope.get("validation_or_test_members_opened") == 0
        and scope.get("changed_candidates_constructed") == 1
        and scope.get("optimizer_instances") == 0
        and scope.get("optimizer_steps") == 0
        and scope.get("checkpoint_writes") == 0
        and scope.get("model_artifact_writes") == 0
        and scope.get("network_package_upload_submission") is False
        and scope.get("fulltrain_evaluation_performed") is False,
    }
    if not all(checks.values()) or not isinstance(payload, Mapping):
        raise ProtocolError(f"B256 candidate result drift: {checks}")
    payload_checks = {
        "parent_file": payload.get("parent_checkpoint_file_sha256")
        == PARENT_FILE_SHA256,
        "parent_model": payload.get("parent_model_state_sha256")
        == PARENT_MODEL_SHA256,
        "actor_names": payload.get("actor_names") == list(ACTOR6_NAMES),
        "changed_names": payload.get("changed_parameter_names")
        == sorted(EFFECTIVE_CHANGED_ACTOR5),
        "layout_list": isinstance(payload.get("actor_layout"), list),
        "actor_blob_mapping": isinstance(
            payload.get("candidate_actor_float32_le"), Mapping
        ),
        "candidate_model_sha": is_sha256(payload.get("candidate_model_state_sha256"))
        and payload.get("candidate_model_state_sha256")
        == candidate_integrity.get("candidate_model_state_sha256"),
        "candidate_nonactor": payload.get("candidate_nonactor_sha256")
        == candidate_integrity.get("candidate_nonactor_sha256")
        == PARENT_NONACTOR_SHA256,
        "actor_payload_sha": is_sha256(
            payload.get("candidate_actor_float32_le_sha256")
        ),
        "actor_l2_cap": math.isfinite(
            float(payload.get("actual_float32_quantized_delta_l2", math.inf))
        )
        and 0.0
        < float(payload.get("actual_float32_quantized_delta_l2", math.inf))
        <= ACTOR_L2_HARD_CAP + 1.0e-12,
        "no_optimizer": payload.get("optimizer_instances") == 0
        and payload.get("optimizer_steps") == 0
        and payload.get("learning_rate") is None,
        "single_candidate": payload.get("single_candidate_only") is True,
    }
    if not all(payload_checks.values()):
        raise ProtocolError(f"candidate payload contract drift: {payload_checks}")
    return payload, {"result_checks": checks, "payload_checks": payload_checks}


def decode_actor_payload(value: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
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
    structural = {
        "keys_exact": set(value) == expected_keys,
        "dtype_little_endian_float32": value.get("dtype") == "<f4",
        "compression_exact": value.get("compression")
        == "XZ_preset9_extreme_CRC64",
        "raw_bytes_exact": value.get("raw_bytes") == ACTOR6_RAW_BYTES,
        "raw_sha_valid": is_sha256(value.get("raw_sha256")),
        "compressed_bytes_positive": type(value.get("compressed_bytes")) is int
        and int(value.get("compressed_bytes", 0)) > 0,
        "compressed_sha_valid": is_sha256(value.get("compressed_sha256")),
        "chunks_nonempty_ascii": isinstance(chunks, list)
        and bool(chunks)
        and all(isinstance(part, str) and part.isascii() for part in chunks),
        "chunks_canonical_width": isinstance(chunks, list)
        and bool(chunks)
        and all(len(part) == 76 for part in chunks[:-1])
        and 0 < len(chunks[-1]) <= 76,
    }
    if not all(structural.values()) or not isinstance(chunks, list):
        raise ProtocolError(f"actor payload structural drift: {structural}")
    encoded = "".join(chunks)
    compressed = base64.b64decode(encoded, validate=True)
    compressed_checks = {
        "canonical_base64": base64.standard_b64encode(compressed).decode("ascii")
        == encoded,
        "compressed_bytes": len(compressed) == int(value["compressed_bytes"]),
        "compressed_sha": sha256_bytes(compressed) == value["compressed_sha256"],
        "crc64_supported": lzma.is_check_supported(lzma.CHECK_CRC64),
    }
    if not all(compressed_checks.values()):
        raise ProtocolError(f"compressed actor payload drift: {compressed_checks}")
    raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    raw_checks = {
        "raw_bytes": len(raw) == ACTOR6_RAW_BYTES == int(value["raw_bytes"]),
        "raw_sha": sha256_bytes(raw) == value["raw_sha256"],
    }
    if not all(raw_checks.values()):
        raise ProtocolError(f"raw actor payload drift: {raw_checks}")
    return raw, {
        "structural": structural,
        "compressed": compressed_checks,
        "raw": raw_checks,
    }


def reconstruct_candidate(
    helper: ModuleType,
    cw20: ModuleType,
    parent_payload: bytes,
    candidate_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np

    torch = helper.torch
    parent = helper.checkpoint_from_bytes(parent_payload, "guarded U468 parent")
    parent_state = parent.get("model_state_dict")
    if not isinstance(parent_state, Mapping):
        raise ProtocolError("parent state is absent")
    raw = {name: value.detach().cpu().clone() for name, value in parent_state.items()}
    candidate = {name: value.detach().cpu().clone() for name, value in raw.items()}
    actor_value = candidate_payload.get("candidate_actor_float32_le")
    layout = candidate_payload.get("actor_layout")
    if not isinstance(actor_value, Mapping) or not isinstance(layout, list):
        raise ProtocolError("candidate actor payload/layout absent")
    actor_bytes, decode_audit = decode_actor_payload(actor_value)
    if sha256_bytes(actor_bytes) != candidate_payload.get(
        "candidate_actor_float32_le_sha256"
    ):
        raise ProtocolError("candidate actor outer/inner SHA drift")
    if sha256_bytes(canonical_json(layout)) != candidate_payload.get(
        "actor_layout_sha256"
    ):
        raise ProtocolError("candidate actor layout SHA drift")
    vector = np.frombuffer(actor_bytes, dtype="<f4")
    if vector.shape != (ACTOR6_DIMENSION,) or not bool(np.isfinite(vector).all()):
        raise ProtocolError("candidate actor vector malformed/nonfinite")
    cw20.copy_actor_bytes(candidate, ACTOR6_NAMES, layout, actor_bytes, np, torch)
    reconstructed = cw20.actor_bytes(candidate, ACTOR6_NAMES, np)
    parent_actor = cw20.actor_bytes(raw, ACTOR6_NAMES, np)
    changed = sorted(
        name for name in raw if not torch.equal(raw[name], candidate[name])
    )
    nonactor_names = sorted(set(raw).difference(ACTOR6_NAMES))
    parent_nonactor = helper.model_state_sha256(
        {name: raw[name] for name in nonactor_names}
    )
    candidate_nonactor = helper.model_state_sha256(
        {name: candidate[name] for name in nonactor_names}
    )
    candidate_model_sha = helper.model_state_sha256(candidate)
    delta = np.frombuffer(reconstructed, dtype="<f4").astype("<f8") - np.frombuffer(
        parent_actor, dtype="<f4"
    ).astype("<f8")
    delta_l2 = float(np.linalg.norm(delta))
    expected_l2 = float(candidate_payload["actual_float32_quantized_delta_l2"])
    checks = {
        "parent_update468": parent.get("update") == 468,
        "parent_kind_ppo": helper.evaluator.checkpoint_kind(parent) == "ppo",
        "parent_model_config": isinstance(parent.get("model_config"), dict),
        "parent_model_sha": helper.model_state_sha256(raw) == PARENT_MODEL_SHA256,
        "parent_actor_sha": sha256_bytes(parent_actor)
        == PARENT_ACTOR_FLOAT32_LE_SHA256,
        "actor_bytes_exact_payload": reconstructed == actor_bytes,
        "candidate_model_sha": candidate_model_sha
        == candidate_payload.get("candidate_model_state_sha256"),
        "changed_exact_effective_actor5": changed
        == sorted(EFFECTIVE_CHANGED_ACTOR5),
        "common_actor_bias_bit_exact": torch.equal(
            raw[ACTOR6_NAMES[-1]], candidate[ACTOR6_NAMES[-1]]
        ),
        "nonactor_count74": len(nonactor_names) == 74,
        "parent_nonactor_sha": parent_nonactor == PARENT_NONACTOR_SHA256,
        "candidate_nonactor_sha": candidate_nonactor == PARENT_NONACTOR_SHA256,
        "state_key_count80": len(raw) == len(candidate) == 80,
        "candidate_differs_parent": candidate_model_sha != PARENT_MODEL_SHA256,
        "all_candidate_tensors_finite": all(
            not (value.is_floating_point() or value.is_complex())
            or bool(torch.isfinite(value).all())
            for value in candidate.values()
        ),
        "recomputed_l2_matches_payload": math.isclose(
            delta_l2, expected_l2, rel_tol=0.0, abs_tol=1.0e-15
        ),
        "recomputed_l2_within_cap": 0.0 < delta_l2 <= ACTOR_L2_HARD_CAP + 1.0e-12,
    }
    if not all(checks.values()):
        raise ProtocolError(f"candidate reconstruction failed: {checks}")
    candidate_checkpoint = dict(parent)
    candidate_checkpoint["model_state_dict"] = candidate
    return parent, candidate_checkpoint, {
        "checks": checks,
        "decode": decode_audit,
        "changed_parameter_names": changed,
        "candidate_model_state_sha256": candidate_model_sha,
        "parent_nonactor_sha256": parent_nonactor,
        "candidate_nonactor_sha256": candidate_nonactor,
        "actor_float32_le_sha256": sha256_bytes(reconstructed),
        "actual_actor6_l2": delta_l2,
    }


def load_train_inputs(
    formal: ModuleType,
) -> tuple[ModuleType, dict[str, bytes], dict[str, Any]]:
    design = formal.load_module(
        formal.DESIGN_TOOL,
        formal.DESIGN_TOOL_SHA256,
        "currentparent_fulltrain_design121",
        0o555,
    )
    design_checks = {
        "panels_exact": tuple(design.DATASETS) == PANEL_ORDER,
        "dataset_paths_exact": all(
            Path(design.DATASETS[panel]).resolve() == DATASETS[panel].resolve()
            for panel in PANEL_ORDER
        ),
        "dataset_hashes_exact": all(
            design.DATA_SHA256[panel] == DATA_SHA256[panel] for panel in PANEL_ORDER
        ),
        "row_counts_exact": all(
            int(design.EXPECTED_TRAIN[panel]["rows"]) == EXPECTED_ROWS[panel]
            for panel in PANEL_ORDER
        ),
        "context34_counts_exact": all(
            int(design.EXPECTED_TRAIN[panel]["context34_rows"])
            == EXPECTED_CONTEXT34_ROWS[panel]
            for panel in PANEL_ORDER
        ),
    }
    if not all(design_checks.values()):
        raise ProtocolError(f"frozen train design drift: {design_checks}")
    archives: dict[str, bytes] = {}
    evidence: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        payload, record = read_regular(
            DATASETS[panel], DATA_SHA256[panel], DATA_MODE[panel], f"{panel} train archive"
        )
        archives[panel] = payload
        evidence[panel] = record
    return design, archives, {"design_checks": design_checks, "train_archives": evidence}


def evaluate_parent_candidate_actor6_only(
    formal: ModuleType,
    c758: ModuleType,
    helper: ModuleType,
    design: ModuleType,
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    archives: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reuse the c758/formal two-state protocol with an actor6-only live overlay."""

    torch = helper.torch
    states = {
        "raw": {
            name: value.detach().cpu().clone()
            for name, value in parent["model_state_dict"].items()
        },
        "candidate": {
            name: value.detach().cpu().clone()
            for name, value in candidate["model_state_dict"].items()
        },
    }
    original_order = formal.MODEL_ORDER
    original_alphas = formal.ALPHAS
    original_copy = formal.copy_actor_state_to_model
    original_verify = formal.verify_live_actor_state
    original_summary = formal.assert_frozen_summary_exact
    original_parent_hash = design.PARENT_MODEL_STATE_SHA256
    live: dict[str, Any] = {}
    restore = {
        "live_model_seen": False,
        "finally_parent_restore_attempted": False,
        "finally_parent_restore_pass": False,
    }

    def copy_actor6(model: Any, state: Mapping[str, Any], module: ModuleType) -> None:
        del module
        live["model"] = model
        current = model.state_dict()
        if set(current) != set(state):
            raise ProtocolError("live/state key drift during actor6 overlay")
        with torch.no_grad():
            for name in ACTOR6_NAMES:
                target = current[name]
                target.copy_(state[name].to(device=target.device, dtype=target.dtype))

    def verify_full_live_state(
        model: Any, state: Mapping[str, Any], module: ModuleType
    ) -> None:
        observed = module.model_state_sha256(model.state_dict())
        expected = module.model_state_sha256(state)
        if observed != expected:
            raise ProtocolError(
                f"full live actor6 overlay hash mismatch: {observed} != {expected}"
            )

    def same_run_summary(
        summary: Mapping[str, Any],
        frozen: Mapping[str, Any],
        panel: str,
        model_name: str,
    ) -> dict[str, Any]:
        return c758.same_run_parent_summary_hook(
            summary, frozen, panel, model_name, formal
        )

    formal.MODEL_ORDER = ("raw", "candidate")
    formal.ALPHAS = {"raw": 0, "candidate": 1}
    formal.copy_actor_state_to_model = copy_actor6
    formal.verify_live_actor_state = verify_full_live_state
    formal.assert_frozen_summary_exact = same_run_summary
    design.PARENT_MODEL_STATE_SHA256 = PARENT_MODEL_SHA256
    try:
        panels, advisory, execution = formal.evaluate_all_states(
            helper,
            design,
            states,
            parent,
            archives,
            {"raw": {panel: {} for panel in PANEL_ORDER}},
        )
    finally:
        try:
            model = live.get("model")
            if model is not None:
                restore["live_model_seen"] = True
                restore["finally_parent_restore_attempted"] = True
                copy_actor6(model, states["raw"], helper)
                verify_full_live_state(model, states["raw"], helper)
                restored_sha = helper.model_state_sha256(model.state_dict())
                restore["final_parent_model_state_sha256"] = restored_sha
                restore["finally_parent_restore_pass"] = (
                    restored_sha == PARENT_MODEL_SHA256
                )
        finally:
            formal.MODEL_ORDER = original_order
            formal.ALPHAS = original_alphas
            formal.copy_actor_state_to_model = original_copy
            formal.verify_live_actor_state = original_verify
            formal.assert_frozen_summary_exact = original_summary
            design.PARENT_MODEL_STATE_SHA256 = original_parent_hash
    if not restore["finally_parent_restore_pass"]:
        raise ProtocolError("fulltrain evaluator final parent restoration failed")
    execution["all_two_live_actor6_overlays_verified_after_copy"] = execution.pop(
        "all_ten_live_actor_overlays_verified_after_copy"
    )
    execution["same_batch_parent_then_candidate"] = execution.pop(
        "same_batch_all_ten_states"
    )
    execution["all_six_completed_before_decision"] = (
        int(execution["evaluation_count_exact"]) == 6
    )
    execution.pop("all_30_completed_before_decision", None)
    return panels, execution, {"advisory": advisory, "restore": restore}


def immutable_output_gate(panel_results: Mapping[str, Any]) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        parent = panel_results["raw"][panel]
        candidate = panel_results["candidate"][panel]
        integrity = candidate["count_value_integrity"]
        fingerprints = integrity["fingerprint_exact_raw"]
        transitions = integrity["correctness_transitions"]
        count_transition = transitions["count_correct"]
        value_transition = transitions["value_correct"]
        checks = {
            "parent_rows_exact": int(parent["rows"]) == EXPECTED_ROWS[panel],
            "candidate_rows_exact": int(candidate["rows"]) == EXPECTED_ROWS[panel],
            "native_count_logits_bit_exact_every_row": int(
                integrity["native_count_logits_mismatch_count"]
            )
            == 0,
            "native_value_logits_bit_exact_every_row": int(
                integrity["native_value_logits_mismatch_count"]
            )
            == 0,
            "count_logits_fingerprint_exact": fingerprints[
                "count_logits_float32_le"
            ]
            is True,
            "value_logits_fingerprint_exact": fingerprints[
                "value_logits_float32_le"
            ]
            is True,
            "all_count_value_derived_fingerprints_exact": all(
                fingerprints[name] is True
                for name in (
                    "predicted_count",
                    "greedy_length",
                    "count_correct",
                    "value_sign",
                    "value_correct",
                )
            ),
            "count_correct_no_transition": all(
                int(count_transition["cells"][cell]["count"]) == 0
                for cell in ("cw", "wc")
            ),
            "value_correct_no_transition": all(
                int(value_transition["cells"][cell]["count"]) == 0
                for cell in ("cw", "wc")
            ),
            "count_aggregate_exact": int(parent["official_metrics"]["count_correct"])
            == int(candidate["official_metrics"]["count_correct"]),
            "value_aggregate_exact": int(parent["official_metrics"]["value_correct"])
            == int(candidate["official_metrics"]["value_correct"]),
        }
        panels[panel] = {"checks": checks, "pass": all(checks.values())}
    return {"panels": panels, "pass": all(item["pass"] for item in panels.values())}


def execution_gate(
    panel_results: Mapping[str, Any],
    execution: Mapping[str, Any],
    runtime_integrity: Mapping[str, Any],
) -> dict[str, bool]:
    dataset_audits = execution.get("dataset_audits", {})
    checks = {
        "device_cuda0": execution.get("device") == "cuda:0",
        "model_kind_ppo": execution.get("model_kind") == "ppo",
        "one_model_instance": int(execution.get("model_instance_count", -1)) == 1,
        "workers_zero": int(execution.get("workers", -1)) == 0,
        "batch_size256": int(execution.get("batch_size", -1)) == 256,
        "six_logical_evaluations": int(execution.get("evaluation_count_exact", -1))
        == 6,
        "same_batch_parent_then_candidate": execution.get(
            "same_batch_parent_then_candidate"
        )
        is True,
        "both_live_overlays_verified": execution.get(
            "all_two_live_mutable10_overlays_verified_after_copy"
        )
        is True,
        "all_panels_rows_exact": all(
            int(panel_results[model][panel]["rows"]) == EXPECTED_ROWS[panel]
            for model in ("raw", "candidate")
            for panel in PANEL_ORDER
        ),
        "dataset_audits_train_only": isinstance(dataset_audits, Mapping)
        and set(dataset_audits) == set(PANEL_ORDER)
        and all(
            int(dataset_audits[panel].get("rows", -1)) == EXPECTED_ROWS[panel]
            and dataset_audits[panel].get("workers") == 0
            and dataset_audits[panel].get("validation_member_payloads_opened")
            is False
            and all(
                isinstance(member, str) and member.startswith("train/")
                for member in dataset_audits[panel].get("train_members", [])
            )
            for panel in PANEL_ORDER
        ),
        "validation_members_not_opened": execution.get(
            "validation_member_payloads_opened"
        )
        is False,
        "all_six_before_decision": execution.get("all_six_completed_before_decision")
        is True,
        "finally_parent_restored": runtime_integrity.get("restore", {}).get(
            "finally_parent_restore_pass"
        )
        is True,
        "finally_parent_hash_exact": runtime_integrity.get("restore", {}).get(
            "final_parent_model_state_sha256"
        )
        == PARENT_MODEL_SHA256,
    }
    return checks


def decide(
    c758: ModuleType,
    panel_results: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        tuple(c758.POLICY_METRICS) != POLICY_METRICS
        or dict(c758.PF_MINIMUM_NET)
        != {
            **PF_MINIMUM_NET_AND_WC,
            "context34_hybrid_order_exact": 0,
            "context34_ordered_exact": 0,
            "count_correct": 0,
        }
    ):
        raise ProtocolError("c758 gate constants drift")
    established = c758.decide(panel_results)
    immutable = immutable_output_gate(panel_results)
    passed = (
        established.get("all_three_panels_pass") is True
        and immutable.get("pass") is True
    )
    return {
        "status": (
            "GO_CURRENT_PARENT_BOUNDARYQP_FULLTRAIN"
            if passed
            else "NO_GO_CURRENT_PARENT_BOUNDARYQP_FULLTRAIN_CLOSE_LINEAGE"
        ),
        "pass": passed,
        "established_c758_policy_and_count_net_gate": established,
        "count_value_native_bit_exact_gate": immutable,
        "pokemonfan_minimum_net_and_wc": dict(PF_MINIMUM_NET_AND_WC),
        "flg_core5_policy_CW_reported_not_independent_zero_CW_gate": True,
        "specialist_execution_preregistration_authorized": passed,
        "specialist_execution_authorized_now": False,
        "retry_or_threshold_adjustment_authorized": False,
        "official_unique_changed_candidate_count_consumed": 0,
        "broad_gold_package_upload_submission_authorized": False,
    }


def output_absence() -> dict[str, bool]:
    return {
        "preregistration_present": PREREGISTRATION.is_file()
        and not PREREGISTRATION.is_symlink(),
        "attempt_absent": not lexists(ATTEMPT),
        "result_absent": not lexists(OUTPUT),
    }


def run_formal(
    source_record: Mapping[str, Any],
    expected_preregistration_sha256: str,
) -> dict[str, Any]:
    if LINEAGE_CLOSED_NO_GO:
        raise ProtocolError(f"closed lineage: {CLOSURE_REASON}")
    if not binding_state()["all_bound"]:
        raise ProtocolError(f"draft bindings unresolved: {binding_state()}")
    control_values, control_evidence = read_control_plane()
    del control_values
    candidate_bytes, candidate_record = read_regular(
        CANDIDATE_RESULT,
        CANDIDATE_RESULT_SHA256,
        0o444,
        "sole B256 candidate result",
    )
    candidate_document = strict_json(candidate_bytes, "sole B256 candidate result")
    candidate_payload, candidate_validation = validate_candidate_document(
        candidate_document, candidate_record
    )
    _, solver_record = read_regular(SOLVER, SOLVER_SHA256, 0o555, "frozen B256 solver")
    prereg, prereg_record = verify_preregistration(
        expected_preregistration_sha256,
        str(source_record["sha256"]),
    )
    absence = output_absence()
    if not all(absence.values()):
        raise ProtocolError(f"one-shot output absence failed: {absence}")
    marker = {
        "schema_version": f"{SCHEMA}-attempt-v1",
        "status": "attempt_committed_before_checkpoint_or_train_archive_or_CUDA",
        "runner_sha256": source_record["sha256"],
        "preregistration_sha256": expected_preregistration_sha256,
        "candidate_result_sha256": CANDIDATE_RESULT_SHA256,
        "candidate_model_state_sha256": candidate_payload[
            "candidate_model_state_sha256"
        ],
        "attempts_authorized": 1,
        "retry_authorized": False,
        "official_unique_changed_candidate_count_consumed": 0,
        "package_upload_submission_authorized": False,
    }
    attempt_record = publish(ATTEMPT, canonical_json(marker))
    try:
        formal, formal_record = import_frozen(
            FORMAL, FORMAL_SHA256, "currentparent_fulltrain_formal_v3"
        )
        c758, c758_record = import_frozen(
            C758, C758_SHA256, "currentparent_fulltrain_c758"
        )
        cw20, cw20_record = import_frozen(
            CW20, CW20_SHA256, "currentparent_fulltrain_actor_codec"
        )
        helper = formal.load_helper()
        cuda_runtime = formal.cuda_runtime(helper)
        random.seed(formal.SEED)
        helper.torch.manual_seed(formal.SEED)
        helper.torch.cuda.manual_seed_all(formal.SEED)
        parent_bytes, parent_record = read_regular(
            PARENT, PARENT_FILE_SHA256, 0o444, "guarded U468 parent"
        )
        parent, candidate, reconstruction = reconstruct_candidate(
            helper, cw20, parent_bytes, candidate_payload
        )
        design, archives, train_evidence = load_train_inputs(formal)
        before_hashes = {
            "parent": helper.model_state_sha256(parent["model_state_dict"]),
            "candidate": helper.model_state_sha256(candidate["model_state_dict"]),
        }
        panel_results, execution, runtime_integrity = c758.evaluate_two_states(
            formal,
            helper,
            design,
            parent,
            candidate,
            archives,
        )
        after_hashes = {
            "parent": helper.model_state_sha256(parent["model_state_dict"]),
            "candidate": helper.model_state_sha256(candidate["model_state_dict"]),
        }
        execution_checks = execution_gate(
            panel_results, execution, runtime_integrity
        )
        execution_checks["CPU_states_unchanged"] = before_hashes == after_hashes
        execution_checks["candidate_hash_exact_after"] = after_hashes["candidate"] \
            == candidate_payload["candidate_model_state_sha256"]
        if not all(execution_checks.values()):
            raise ProtocolError(f"fulltrain execution integrity failed: {execution_checks}")
        decision = decide(c758, panel_results)
        result = {
            "schema_version": SCHEMA,
            "status": decision["status"],
            "attempt": attempt_record,
            "preregistration": prereg_record,
            "input_lock": {
                "runner": dict(source_record),
                "solver": solver_record,
                "candidate_B256_result": candidate_record,
                "control_plane": control_evidence,
                "parent": parent_record,
                "formal_v3": formal_record,
                "c758_fulltrain_reference": c758_record,
                "actor_payload_codec": cw20_record,
                "train": train_evidence,
            },
            "candidate_B256_validation": candidate_validation,
            "candidate_reconstruction": reconstruction,
            "runtime": cuda_runtime,
            "execution": {**execution, "checks": execution_checks},
            "runtime_integrity": runtime_integrity,
            "evaluations": panel_results,
            "decision": decision,
            "scope_audit": {
                "all_24050_train_rows": True,
                "validation_test_broad_gold_opened": False,
                "same_existing_candidate_only": True,
                "second_candidate_constructed": False,
                "training_optimizer_backward": False,
                "checkpoint_or_model_artifact_writes": 0,
                "evidence_result_writes": 1,
                "network_package_upload_submission": False,
                "official_unique_changed_candidate_count_consumed": 0,
            },
            "preregistered_contract": prereg["gate_contract"],
        }
        publication = publish(OUTPUT, canonical_json(result))
        return {
            "schema_version": SCHEMA,
            "status": result["status"],
            "pass": decision["pass"],
            "candidate_model_state_sha256": candidate_payload[
                "candidate_model_state_sha256"
            ],
            "output": publication,
            "official_unique_changed_candidate_count_consumed": 0,
            "submission_performed": False,
        }
    except BaseException as error:
        failure = {
            "schema_version": SCHEMA,
            "status": "ERROR_CURRENT_PARENT_BOUNDARYQP_FULLTRAIN_CLOSE_LINEAGE",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "attempt": attempt_record,
            "candidate_result_sha256": CANDIDATE_RESULT_SHA256,
            "official_unique_changed_candidate_count_consumed": 0,
            "package_upload_submission_performed": False,
            "retry_authorized": False,
        }
        if not lexists(OUTPUT):
            publish(OUTPUT, canonical_json(failure))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "formal"), default="static")
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--expected-preregistration-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = validate_runtime()
    source, source_record = read_regular(
        SCRIPT,
        args.expected_tool_sha256,
        0o555,
        "current-parent fulltrain adapter",
    )
    static = static_source_audit(source)
    if args.mode == "static":
        if args.expected_preregistration_sha256 is not None:
            raise ProtocolError("static mode forbids preregistration input")
        state = binding_state()
        print(
            canonical_json(
                {
                    "schema_version": f"{SCHEMA}-static-v1",
                    "status": "UNAVAILABLE_CLOSED_LINEAGE_DRAFT_NOT_RUN",
                    "runner": source_record,
                    "runtime": runtime,
                    "static": static,
                    "binding_state": state,
                    "bindings": expected_bindings(source_record["sha256"]),
                    "gate_contract": expected_gate_contract(),
                    "formal_command_template": formal_command_template(
                        source_record["sha256"]
                    ),
                    "publication": {
                        "preregistration": str(PREREGISTRATION.relative_to(ROOT)),
                        "attempt": str(ATTEMPT.relative_to(ROOT)),
                        "result": str(OUTPUT.relative_to(ROOT)),
                    },
                    "candidate_result_opened": False,
                    "checkpoint_or_archive_opened": False,
                    "cuda_accessed": False,
                    "writes_performed": False,
                    "validation_test_network_submission": False,
                }
            ).decode("utf-8"),
            end="",
        )
        return
    if not is_sha256(args.expected_preregistration_sha256):
        raise ProtocolError("formal mode requires preregistration SHA-256")
    expected_argv = [
        str(SCRIPT),
        "--mode",
        "formal",
        "--expected-tool-sha256",
        source_record["sha256"],
        "--expected-preregistration-sha256",
        args.expected_preregistration_sha256,
    ]
    observed_argv = [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    if observed_argv != expected_argv:
        raise ProtocolError(f"formal argv/order drift: {observed_argv} != {expected_argv}")
    receipt = run_formal(source_record, args.expected_preregistration_sha256)
    print(canonical_json(receipt).decode("utf-8"), end="")


if __name__ == "__main__":
    main()

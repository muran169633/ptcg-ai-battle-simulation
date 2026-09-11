#!/usr/bin/env python3
"""CW13 consumed-valid official-B256 cutting-plane probe.

The authoritative acceptance oracle is all six consumed specialist-valid
views in their frozen B256/8-worker stream shapes.  B33 is retained only as a
legacy compatibility gate.  This probe is stdout-only: it never writes a
checkpoint/result, accesses broad/Gold, or performs network/upload/submission
actions.  A successful terminal candidate is emitted as a pure reconstruction
payload; downstream consumers must not reopen the consumed specialist data or
rerun this solver.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import lzma
import math
import os
import random
import stat
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "probe_u468_cw11_consumed_valid_official6_cw13_cuttingplane_v1.py"
SCHEMA = "ptcg-cw13-consumed-valid-official-b256-cuttingplane-v1"
SEED = 202608207
FROZEN_MODE = 0o555

CW12_PARENT_SOLVER_SHA256 = (
    "e16eb0aa6d3ea0905309210126abd9f1f4cbb2f093b7e9dfdac49271466377ec"
)
CW12_ATTEMPT_MARKER = ROOT / (
    "artifacts/.ptcg-cw12_consumed_valid_official6_cuttingplane_20260802_v1-"
    "attempt.json"
)
CW12_ATTEMPT_MARKER_SHA256 = (
    "e8285105d1c50c6608e020433bd82425c7c6debeb357cfc82993db2f90a4ab3a"
)
CW12_FAILURE_STDOUT = ROOT / (
    "artifacts/cw12_consumed_valid_official6_cuttingplane_20260802_v1.stdout.json"
)
CW12_FAILURE_STDOUT_SHA256 = (
    "dd7367f10112b70ae3264c005e52f839ec83f440dea66da827f9ec3306004b43"
)
CW12_FAILURE_STDERR_SHA256 = (
    "364a4d971f5bb39c92fbd719b7d0f70df1efef27b9862a4f5dc0831fbb8d8526"
)
CW12_FAILURE_STDERR_BYTES = 3768
CW12_EMPTY_STDOUT_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

RAW = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
RAW_FILE_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
RAW_NONACTOR_SHA256 = "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
E904 = ROOT / (
    "artifacts/ppo_u468_raw_actor6_metricguard_fulltrain_cw10_materialized_v1_20260802/"
    "u468-cw10-fulltrain-pass-eval-only.pt"
)
E904_FILE_SHA256 = "91b64ddf754b149297dd6177038de871fdc69bcea0d86787cff851e4fd79e0dd"
E904_MODEL_SHA256 = "e904efb322c15ca4bee62573f4f439309d1d4ecb3ae62620d13fd70e8acaa91f"
CW11_CHECKPOINT = ROOT / (
    "artifacts/ppo_u468_raw_actor6_metricguard_specialist_valid_cw11_"
    "materialized_v1_20260802/u468-cw11-formal-pass-eval-only.pt"
)
CW11_FILE_SHA256 = "bea774c30cd3113d984ba8252324c330d293a8ce5042d7d17f357f8132e86775"
CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
CW11_VECTOR_SHA256 = "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
CW11_TOTAL_FROM_RAW_L2 = 0.00792176975336988
CW11_ACTIVE_LEDGER_SHA256 = "6ae0c07a94627cb6fd0155f265f75d13bd7505b0e5f62cd62c3e26ceb7714436"

GUARDPLAN = ROOT / "artifacts/cw11_failure_attribution_cw12_guardplan_20260802_v1.json"
GUARDPLAN_SHA256 = "3f280363301fad9b87dff99c5eb152a853b9e850039d3106eccfd390e9dc3fa4"
FORENSIC = ROOT / "artifacts/cw11_specialist_dev_row_flips_readonly_20260802_v1.json"
FORENSIC_SHA256 = "b8ac6b08cfb7ee6f916216fbf0f70ec6d0c9e790595ed6dfc12f0a0c88be0a7b"
PREREG = ROOT / (
    "artifacts/cw11_4317_specialist_devconsistency_20260802_v1."
    "specialist_dev_consistency_preregistration.json"
)
PREREG_SHA256 = "b2231b5018e3ddba89649e7373726ed29bc5513b2d8cd8befb6230297ff70cc4"
FAILED_DECISION = ROOT / (
    "artifacts/cw11_4317_specialist_devconsistency_20260802_v1."
    "specialist_dev_consistency/dev_consistency_decision.json"
)
FAILED_DECISION_SHA256 = "9c925a66f116a10b0ac996a76fdcbe0c378f0e4d02d45c635a2d47228b189a35"
FAILED_MANIFEST = ROOT / (
    "artifacts/cw11_4317_specialist_devconsistency_20260802_v1."
    "specialist_dev_consistency/dev_consistency_execution_manifest.json"
)
FAILED_MANIFEST_SHA256 = "47ebd3683b73f397ffff5c6874136c9f12d23ee8eb619a9705445fabd4b73a27"
MATERIALIZATION_MANIFEST = ROOT / (
    "artifacts/ppo_u468_raw_actor6_metricguard_specialist_valid_cw11_"
    "materialized_v1_20260802/materialization_manifest.json"
)
MATERIALIZATION_MANIFEST_SHA256 = (
    "6356f08505b5777fdb30e4f8cfce8a776ca2918af540ebc324227678aeb02016"
)

PF = ROOT / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip"
FLG = ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip"
CORE5 = ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip"
DATA_SHAS = {
    PF: "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    FLG: "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    CORE5: "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}
VIEW_ORDER = ("pokemonfan", "flg", "core5", "dominic", "luca", "szlach")
VIEW_SPECS = {
    "flg": {"archive": FLG, "team_names": ()},
    "pokemonfan": {"archive": PF, "team_names": ()},
    "core5": {"archive": CORE5, "team_names": ()},
    "dominic": {"archive": CORE5, "team_names": ("Dominic Peel",)},
    "luca": {"archive": CORE5, "team_names": ("Luca",)},
    "szlach": {"archive": CORE5, "team_names": ("szlachetny snieg",)},
}

CW11_PROBE = TOOLS / "probe_u468_raw_actor6_metricguard_specialist_valid_cw11_cuttingplane_v1.py"
CW10 = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw10_cuttingplane_v1.py"
CW9 = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw9_cuttingplane_v1.py"
PRIMARY = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_cw4_cuttingplane_v1.py"
CUTTING = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v2.py"
GEOMETRY = TOOLS / "probe_u468_raw_actor6_metricguard_specialbc_v1.py"
RAM = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_ram_ray_v1.py"
FORENSIC_ANALYZER = TOOLS / "analyze_e904_specialist_row_flips_v1.py"
EVALUATOR = TOOLS / "evaluate_policy_bc.py"
BC = TOOLS / "train_bc_orbit.py"
PPO = TOOLS / "train_ppo.py"
MODULE_SHAS = {
    CW11_PROBE: "23bc74022210942115eaf339a38e710e88ad81ba75d62237e8eb9e2c11e074ee",
    CW10: "546e90c5b3ca35c84b8efc08109b2f310b45b2895d24aeee70561d13bffa146c",
    CW9: "f77a76f2ac91732e8bc8d9a436ca7a49058b5de50f7b3114db14cffa1fbf5292",
    PRIMARY: "40715927549457b161fc907757279e4d85466ded1d1e6bfed62f67c13b529e7c",
    CUTTING: "c2866ba8b00eba6b424197a520419a5717401335cc47202b4fcc711611f67503",
    GEOMETRY: "ddecd3a85bc2b43c28854afc56678c21943198ac3fde3613765d6a49e440eedb",
    RAM: "86b05d4f826576717907141aef2c534f140a1531b87c8e2d28aeb624a2657e4d",
    FORENSIC_ANALYZER: "c577dc1280cb36197f5125dbd1a0e6be3ddbd3a8dfb17d7aaf6af53a466d6634",
    EVALUATOR: "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4",
    BC: "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
    PPO: "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3",
}

ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
ACTOR6_LAYOUT_SHA256 = "b86476b9ccbbeeac7b46754f7f15e349e398aa627fc3dc6060d6c3d6e3bd26cb"
ACTOR6_FLAT_LENGTH = 65793
ACTOR6_FLOAT64_BYTES = 526344
POLICY_METRICS = ("set", "hybrid", "ordered", "top1")
MAX_OUTER_ITERATIONS = 12
STEP_L2_CAP = 0.001
ADDITIONAL_TOTAL_L2_CAP = 0.001
L2_CAP_ABS_TOL = 1e-12
FIRST_FEASIBLE = True
CLASSIFICATION = {
    "specialist_valid_consumed": True,
    "dev_tuning_only": True,
    "promotion_evidence": False,
    "broad_access": False,
    "gold_access": False,
    "package_upload_submission": False,
}
EXPECTED_CHANGED_PHYSICAL_ROWS = 47
EXPECTED_RAW_TO_CW11_TRANSITIONS = 22
EXPECTED_TRANSITION_OCCURRENCES = 26
EXPECTED_UNIQUE_OFFICIAL_CONTEXTS = 25
EXPECTED_INITIAL_REPAIR_PHYSICAL_ROWS = 3
EXPECTED_INITIAL_REPAIR_CONTEXT_GUARDS = 4
EXPECTED_LEGACY_ROWS = 33
EXPECTED_LEGACY_ACTIVE_PAIRS = 34
EXPECTED_PHYSICAL47_SHA256 = "f133f164d2ea5ba201dade867ab8eda35dddde084cb1561c8510f4a3c66f329a"
EXPECTED_PHYSICAL22_SHA256 = "4e18e4889dff5bc731dd7da6837c223f95c2f7047e1d70ace08534ad1ac1b899"
EXPECTED_OCCURRENCES26_SHA256 = "2aac78c6064cbf2c5dc0049cd1015a0d88d4e9e861d1e7c8dc27e649338db651"
EXPECTED_CONTEXTS25_SHA256 = "5c5d9cc1a575741e6f82b3088eaee84d80a7799ecf6aa5788fdf983920af7ff4"
EXPECTED_VIEW_ROWS = {
    "flg": 2312,
    "pokemonfan": 15152,
    "core5": 8319,
    "dominic": 2623,
    "luca": 1378,
    "szlach": 2567,
}

CW11_OFFICIAL_ROOT = ROOT / (
    "artifacts/cw11_4317_specialist_devconsistency_20260802_v1."
    "specialist_dev_consistency"
)
CW11_OFFICIAL_SHAS = {
    "pokemonfan": "eaab85cb0a495947b5039a98edef72f70432339ec1545d45a98f01e75bcae1c7",
    "flg": "9ffc78ea4a198e4b5af5d8207c1b06c9810301f69637f32986fc659a0689d17d",
    "core5": "058b6c55d2f5ceae16db61df271077ae6851149c82b6a7425fa94548b6607c59",
    "dominic": "ee5b3825c370376a572306a6611bf2e957969f799676033a3fa0f275067d56d7",
    "luca": "5ac88b3d745395a448193f5840e712d62fe28e35df619f2d07cb1848ac17e295",
    "szlach": "aabe4a970f0333685f4106b29222c1425bd03321ff95624d109aafdb04a88ff6",
}
EXPECTED_INPUT_MODES = {
    CW12_ATTEMPT_MARKER: 0o444,
    CW12_FAILURE_STDOUT: 0o444,
    RAW: 0o664,
    E904: 0o444,
    CW11_CHECKPOINT: 0o444,
    GUARDPLAN: 0o444,
    FORENSIC: 0o444,
    PREREG: 0o444,
    FAILED_DECISION: 0o444,
    FAILED_MANIFEST: 0o444,
    MATERIALIZATION_MANIFEST: 0o444,
    PF: 0o664,
    FLG: 0o664,
    CORE5: 0o600,
    CW11_PROBE: 0o555,
    CW10: 0o555,
    CW9: 0o555,
    PRIMARY: 0o555,
    CUTTING: 0o555,
    GEOMETRY: 0o555,
    RAM: 0o555,
    FORENSIC_ANALYZER: 0o664,
    EVALUATOR: 0o664,
    BC: 0o664,
    PPO: 0o664,
    **{
        CW11_OFFICIAL_ROOT / f"{panel}.json": 0o444
        for panel in CW11_OFFICIAL_SHAS
    },
}

FIXED_CONTEXT_EXPECTATIONS = (
    {
        "view": "pokemonfan",
        "batch_ordinal": 4,
        "batch_size": 256,
        "offset": 14,
        "dynamic_max_options": 34,
        "line_sha256": "b17cdedb548123fa2ee16c1a7dfbad579057c57769a2ad0baeda5b2b6cb70492",
    },
    {
        "view": "pokemonfan",
        "batch_ordinal": 33,
        "batch_size": 256,
        "offset": 147,
        "dynamic_max_options": 19,
        "line_sha256": "80bf3ff2feadb10265e3fba7568a88220f23a4522a0f3604404ec7936a1d5f93",
    },
    {
        "view": "core5",
        "batch_ordinal": 27,
        "batch_size": 256,
        "offset": 14,
        "dynamic_max_options": 25,
        "line_sha256": "89efa747488c6f09396eda370ae99b2b1f15c2106028232965d6f72428bb3d59",
    },
    {
        "view": "dominic",
        "batch_ordinal": 7,
        "batch_size": 256,
        "offset": 14,
        "dynamic_max_options": 25,
        "line_sha256": "89efa747488c6f09396eda370ae99b2b1f15c2106028232965d6f72428bb3d59",
    },
)

FIXED_REPAIR_IDENTITIES = (
    {
        "archive": str(CORE5.relative_to(ROOT)),
        "member": "valid/part-00000.jsonl",
        "line_index": 1550,
        "line_sha256": "89efa747488c6f09396eda370ae99b2b1f15c2106028232965d6f72428bb3d59",
        "decision_sha256": "87876a365beda9344732e4446533b8f5717ecd29b9a85d88d1c56ab2c38a2807",
        "positive_option": 0,
        "negative_option": 1,
        "raw_margin": 0.00390625,
        "required_views": ("core5", "dominic"),
    },
    {
        "archive": str(PF.relative_to(ROOT)),
        "member": "valid/part-00000.jsonl",
        "line_index": 782,
        "line_sha256": "b17cdedb548123fa2ee16c1a7dfbad579057c57769a2ad0baeda5b2b6cb70492",
        "decision_sha256": "037e21640da0bfa05adbccf2fd5d06600926c66b5796a261aa84ca7387c6a8ac",
        "positive_option": 5,
        "negative_option": 0,
        "raw_margin": 0.00390625,
        "required_views": ("pokemonfan",),
    },
    {
        "archive": str(PF.relative_to(ROOT)),
        "member": "valid/part-00000.jsonl",
        "line_index": 8339,
        "line_sha256": "80bf3ff2feadb10265e3fba7568a88220f23a4522a0f3604404ec7936a1d5f93",
        "decision_sha256": "afccaa89504fbaed73971be5ee5d71d426bb41f5eea7e0d01ee9bd34cdf3cec1",
        "positive_option": 1,
        "negative_option": 0,
        "raw_margin": 0.0009765625,
        "required_views": ("pokemonfan",),
        "strict_positive": True,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read_json_locked(path: Path, expected_sha: str, expected_mode: int | None = None) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha:
        raise RuntimeError(f"input hash drift: {path}: {observed} != {expected_sha}")
    if expected_mode is not None and stat.S_IMODE(path.stat().st_mode) != expected_mode:
        raise RuntimeError(f"input mode drift: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_cw12_failure_evidence() -> dict[str, Any]:
    """Bind CW13 to the exact consumed CW12 attempt and its sole failure."""

    marker = read_json_locked(
        CW12_ATTEMPT_MARKER, CW12_ATTEMPT_MARKER_SHA256, 0o444
    )
    failure = read_json_locked(
        CW12_FAILURE_STDOUT, CW12_FAILURE_STDOUT_SHA256, 0o444
    )
    child = failure.get("child")
    if not isinstance(child, dict):
        raise RuntimeError("CW12 failure child evidence is not an object")
    try:
        stderr = base64.b64decode(
            str(child["stderr_base64"]).encode("ascii"), validate=True
        )
        stdout = base64.b64decode(
            str(child["stdout_base64"]).encode("ascii"), validate=True
        )
        stderr_text = stderr.decode("utf-8")
    except Exception as exc:
        raise RuntimeError("CW12 failure child evidence is not exact base64 UTF-8") from exc

    failure_prefix = "RuntimeError: full stream separation oracle failed: "
    failure_lines = [
        line for line in stderr_text.splitlines() if line.startswith(failure_prefix)
    ]
    if len(failure_lines) != 1 or stderr_text.count(failure_prefix) != 1:
        raise RuntimeError("CW12 trace does not contain exactly one oracle hard map")
    parsed_hard = ast.literal_eval(failure_lines[0][len(failure_prefix):])
    expected_hard = {
        "all_six_row_totals_exact": True,
        "count_value_logits_exact_raw_all_batches": True,
        "monitor47_complete_all_view_occurrences": True,
        "transition22_occurrences_complete": True,
        "favorable_transitions_retained": False,
        "new_harm_cut_cardinality_exact": True,
        "new_harm_cuts_unique": True,
    }
    if not isinstance(parsed_hard, dict):
        raise RuntimeError("CW12 oracle hard evidence is not a mapping")
    false_keys = sorted(key for key, value in parsed_hard.items() if value is False)
    marker_solver = marker.get("solver", {})
    failure_solver = failure.get("solver", {})
    failure_marker = failure.get("attempt_marker", {})
    checks = {
        "marker_schema_exact": marker.get("schema_version")
        == "ptcg-cw12-consumed-valid-official6-one-shot-attempt-v1",
        "marker_status_exact": marker.get("status")
        == "one_shot_attempt_consumed_before_cuda_and_official6",
        "marker_one_attempt_no_retry": marker.get("attempts_authorized") == 1
        and marker.get("retry_authorized") is False,
        "marker_solver_parent_exact": marker_solver.get("sha256")
        == CW12_PARENT_SOLVER_SHA256
        and marker_solver.get("mode_octal") == "0555"
        and marker_solver.get("nlink") == 1,
        "failure_schema_exact": failure.get("schema_version")
        == "ptcg-cw12-consumed-valid-official6-one-shot-terminal-failure-v1",
        "failure_status_exact": failure.get("status")
        == "terminal_failure_attempt_consumed_no_retry",
        "failure_stage_exact": failure.get("stage") == "solver_subprocess_nonzero",
        "failure_consumed_no_retry": failure.get("attempt_remains_consumed") is True
        and failure.get("retry_authorized") is False,
        "failure_solver_parent_exact": failure_solver.get("sha256")
        == CW12_PARENT_SOLVER_SHA256,
        "failure_marker_crosslink_exact": failure_marker.get("sha256")
        == CW12_ATTEMPT_MARKER_SHA256
        and failure_marker.get("mode_octal") == "0444"
        and failure_marker.get("nlink") == 1,
        "child_rc1_exact": child.get("returncode") == 1,
        "child_empty_stdout_exact": stdout == b""
        and child.get("stdout_bytes") == 0
        and child.get("stdout_sha256") == CW12_EMPTY_STDOUT_SHA256,
        "child_stderr_bytes_exact": len(stderr) == CW12_FAILURE_STDERR_BYTES
        and child.get("stderr_bytes") == CW12_FAILURE_STDERR_BYTES,
        "child_stderr_sha_exact": hashlib.sha256(stderr).hexdigest()
        == CW12_FAILURE_STDERR_SHA256
        == child.get("stderr_sha256"),
        "trace_one_oracle_failure_exact": len(failure_lines) == 1
        and stderr_text.count(failure_prefix) == 1,
        "trace_hard_map_exact": parsed_hard == expected_hard,
        "trace_only_favorable_retention_false": false_keys
        == ["favorable_transitions_retained"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW12 consumed failure evidence drift: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "parent_solver_sha256": CW12_PARENT_SOLVER_SHA256,
        "attempt_marker": {
            "path": str(CW12_ATTEMPT_MARKER.relative_to(ROOT)),
            "sha256": CW12_ATTEMPT_MARKER_SHA256,
            "mode": "0444",
            "nlink": 1,
        },
        "failure_stdout": {
            "path": str(CW12_FAILURE_STDOUT.relative_to(ROOT)),
            "sha256": CW12_FAILURE_STDOUT_SHA256,
            "mode": "0444",
            "nlink": 1,
        },
        "child_returncode": int(child["returncode"]),
        "stderr_sha256": CW12_FAILURE_STDERR_SHA256,
        "oracle_hard_map": parsed_hard,
        "oracle_false_keys": false_keys,
    }


def identity_tuple(value: Mapping[str, Any]) -> tuple[str, str, int, str]:
    if "physical_key" in value:
        physical = value["physical_key"]
        if not isinstance(physical, list) or len(physical) != 4:
            raise RuntimeError("invalid physical_key identity")
        return (str(physical[0]), str(physical[1]), int(physical[2]), str(physical[3]))
    return (
        str(value["archive"]),
        str(value["member"]),
        int(value.get("line_index", value.get("line_index_zero_based"))),
        str(value["line_sha256"]),
    )


def validate_runtime(*, require_cuda: bool) -> dict[str, Any]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated_exact": sys.flags.isolated == 1,
        "dont_write_bytecode_exact": sys.flags.dont_write_bytecode == 1,
        "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
    }
    if not all(checks.values()):
        raise RuntimeError(f"runtime contract failed: {checks}")
    evidence: dict[str, Any] = {"checks": checks, "cuda_required": require_cuda}
    if require_cuda:
        import torch

        cuda_checks = {
            "available": torch.cuda.is_available(),
            "native_bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "device_count_positive": torch.cuda.device_count() > 0,
        }
        if not all(cuda_checks.values()):
            raise RuntimeError(f"CUDA contract failed: {cuda_checks}")
        evidence["cuda"] = {
            "checks": cuda_checks,
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return evidence


def validate_frozen_inputs() -> dict[str, Any]:
    bindings = {
        CW12_ATTEMPT_MARKER: CW12_ATTEMPT_MARKER_SHA256,
        CW12_FAILURE_STDOUT: CW12_FAILURE_STDOUT_SHA256,
        RAW: RAW_FILE_SHA256,
        E904: E904_FILE_SHA256,
        CW11_CHECKPOINT: CW11_FILE_SHA256,
        GUARDPLAN: GUARDPLAN_SHA256,
        FORENSIC: FORENSIC_SHA256,
        PREREG: PREREG_SHA256,
        FAILED_DECISION: FAILED_DECISION_SHA256,
        FAILED_MANIFEST: FAILED_MANIFEST_SHA256,
        MATERIALIZATION_MANIFEST: MATERIALIZATION_MANIFEST_SHA256,
        **DATA_SHAS,
        **MODULE_SHAS,
        **{
            CW11_OFFICIAL_ROOT / f"{panel}.json": digest
            for panel, digest in CW11_OFFICIAL_SHAS.items()
        },
    }
    if set(bindings) != set(EXPECTED_INPUT_MODES):
        raise RuntimeError("frozen SHA/mode binding key sets differ")
    records = []
    for path, expected in bindings.items():
        observed_stat = path.lstat()
        if not stat.S_ISREG(observed_stat.st_mode) or observed_stat.st_nlink != 1:
            raise RuntimeError(f"frozen input is not one-link regular file: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"frozen input drift: {path}: {observed} != {expected}")
        observed_mode = stat.S_IMODE(observed_stat.st_mode)
        expected_mode = EXPECTED_INPUT_MODES[path]
        if observed_mode != expected_mode:
            raise RuntimeError(
                f"frozen input mode drift: {path}: {observed_mode:04o} != "
                f"{expected_mode:04o}"
            )
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": observed,
                "mode": f"{observed_mode:04o}",
                "expected_mode": f"{expected_mode:04o}",
                "nlink": int(observed_stat.st_nlink),
                "regular": True,
            }
        )
    return {"all_exact": True, "binding_count": len(records), "records": records}


def validate_forensic_contract() -> dict[str, Any]:
    plan = read_json_locked(GUARDPLAN, GUARDPLAN_SHA256, 0o444)
    forensic = read_json_locked(FORENSIC, FORENSIC_SHA256, 0o444)
    prereg = read_json_locked(PREREG, PREREG_SHA256, 0o444)
    decision = read_json_locked(FAILED_DECISION, FAILED_DECISION_SHA256, 0o444)
    manifest = read_json_locked(FAILED_MANIFEST, FAILED_MANIFEST_SHA256, 0o444)
    materialization = read_json_locked(
        MATERIALIZATION_MANIFEST, MATERIALIZATION_MANIFEST_SHA256, 0o444
    )
    rows47 = forensic["unique_physical_changed_rows"]
    rows22 = plan["raw_to_cw11_metric_changed_retention_rows"]
    ids47 = sorted([list(identity_tuple(row)) for row in rows47])
    ids22 = sorted([list(identity_tuple(row)) for row in rows22])
    occurrences = [
        occurrence for row in rows22 for occurrence in row["panel_view_occurrences"]
    ]
    occurrence_identities = sorted(
        [
            *list(identity_tuple(row)),
            str(occurrence["panel"]),
            int(occurrence["official_stream_batch_ordinal_one_based"]),
            int(occurrence["official_stream_batch_size"]),
            int(occurrence["official_stream_offset_zero_based"]),
        ]
        for row in rows22
        for occurrence in row["panel_view_occurrences"]
    )
    context_keys = {
        (
            str(value["panel"]),
            int(value["official_stream_batch_ordinal_one_based"]),
            int(value["official_stream_batch_size"]),
        )
        for value in occurrences
    }
    fixed_by_identity = {
        identity_tuple(value): value
        for value in [
            plan["minimum_repair"]["mandatory_shared_core5_dominic_guard"],
            *plan["minimum_repair"]["pokemonfan_choose_any_two_pool"],
        ]
    }
    fixed_checks = []
    for expected in FIXED_REPAIR_IDENTITIES:
        key = identity_tuple(expected)
        observed = fixed_by_identity.get(key)
        check = {
            "identity": key,
            "present": observed is not None,
            "decision_exact": observed is not None
            and str(observed["decision_sha256"]) == str(expected["decision_sha256"]),
            "pair_exact": observed is not None
            and int(observed["positive_option"]) == int(expected["positive_option"])
            and int(observed["negative_option"]) == int(expected["negative_option"]),
            "raw_floor_exact": observed is not None
            and float(observed["raw_pair_margin"]) == float(expected["raw_margin"]),
            "views_exact": observed is not None
            and tuple(value["panel"] for value in observed["panel_view_occurrences"])
            == tuple(expected["required_views"]),
        }
        fixed_checks.append(check)
    fixed_keys = {identity_tuple(value) for value in FIXED_REPAIR_IDENTITIES}
    fixed_rows = {identity_tuple(value): value for value in rows22 if identity_tuple(value) in fixed_keys}
    fixed_transition_checks = []
    for expected in FIXED_REPAIR_IDENTITIES:
        row = fixed_rows.get(identity_tuple(expected))
        changed = [] if row is None else row["pairwise"]["raw_to_cw11"]["changed_metrics"]
        fixed_transition_checks.append(
            {
                "identity": identity_tuple(expected),
                "present_in_rows22": row is not None,
                "main_metrics_all_changed": set(("set", "hybrid", "ordered", "top1"))
                .issubset(set(changed)),
                "raw_all_true": row is not None
                and all(bool(row["states"]["raw"][metric]) for metric in changed),
                "e904_all_true": row is not None
                and all(bool(row["states"]["e904"][metric]) for metric in changed),
                "cw11_all_false": row is not None
                and all(not bool(row["states"]["cw11"][metric]) for metric in changed),
            }
        )
    minimum_solution_sets = {
        frozenset(tuple(value) for value in solution["physical_keys"])
        for solution in plan["minimum_repair"]["minimum_solutions"]
    }
    failed_deficits = {
        str(key): int(value)
        for key, value in plan["minimum_repair"]["failed_gate_deficits"].items()
    }
    fixed_coverage: dict[str, int] = defaultdict(int)
    for key in fixed_keys:
        observed = fixed_by_identity[key]
        for gate, count in observed["coverage"].items():
            fixed_coverage[str(gate)] += int(count)
    transition_classification = {"favorable_cells": 0, "harmful_cells": 0}
    for row in rows22:
        for metric in row["pairwise"]["raw_to_cw11"]["changed_metrics"]:
            before = bool(row["states"]["raw"][metric])
            after = bool(row["states"]["cw11"][metric])
            if not before and after:
                transition_classification["favorable_cells"] += 1
            elif before and not after:
                transition_classification["harmful_cells"] += 1
            else:
                raise RuntimeError("changed metric has non-transition flags")
    official_commands = prereg["ordered_evaluations"]
    manifest_executions = manifest["ordered_evaluations"]
    execution_bindings = []
    for command, execution in zip(
        official_commands, manifest_executions, strict=True
    ):
        panel = str(command["panel"])
        expected_output = str(
            (CW11_OFFICIAL_ROOT / f"{panel}.json").relative_to(ROOT)
        )
        expected_output_sha = CW11_OFFICIAL_SHAS[panel]
        binding_checks = {
            "order_exact": int(command["order"]) == int(execution["order"]),
            "panel_exact": panel == str(execution["panel"]),
            "command_sha_self_exact": canonical_sha(command["command"])
            == str(command["command_sha256"]),
            "command_sha_manifest_exact": str(command["command_sha256"])
            == str(execution["command_sha256"]),
            "output_path_exact": str(command["output"])
            == str(execution["output"])
            == str(execution["output_evidence"]["path"])
            == expected_output,
            "one_attempt_exact": int(command["attempts_authorized"]) == 1
            and bool(command["output_absent_at_lock"])
            and int(execution["attempt_count"]) == 1,
            "successful_process_exact": int(execution["returncode"]) == 0
            and bool(execution["output_present"])
            and execution["launch_error"] is None
            and execution["output_error"] is None,
            "stdout_output_SHA_exact": str(execution["stdout"]["sha256"])
            == str(execution["output_evidence"]["sha256"])
            == expected_output_sha,
        }
        execution_bindings.append(
            {
                "order": int(command["order"]),
                "panel": panel,
                "command_sha256": str(command["command_sha256"]),
                "output": expected_output,
                "output_sha256": expected_output_sha,
                "checks": binding_checks,
            }
        )
    checks = {
        "guardplan_schema_exact": plan["schema_version"]
        == "ptcg-cw11-failure-attribution-cw12-guardplan-v1",
        "forensic_status_exact": forensic["status"]
        == "completed_read_only_failure_attribution",
        "changed_rows_exact_47": len(rows47) == EXPECTED_CHANGED_PHYSICAL_ROWS,
        "changed_rows_unique_47": len({tuple(value) for value in ids47})
        == EXPECTED_CHANGED_PHYSICAL_ROWS,
        "changed_rows_identity_sha_exact": canonical_sha(ids47)
        == EXPECTED_PHYSICAL47_SHA256,
        "raw_to_cw11_rows_exact_22": len(rows22) == EXPECTED_RAW_TO_CW11_TRANSITIONS,
        "raw_to_cw11_rows_unique_22": len({tuple(value) for value in ids22})
        == EXPECTED_RAW_TO_CW11_TRANSITIONS,
        "raw_to_cw11_identity_sha_exact": canonical_sha(ids22)
        == EXPECTED_PHYSICAL22_SHA256,
        "rows22_subset_rows47": {tuple(value) for value in ids22}.issubset(
            {tuple(value) for value in ids47}
        ),
        "transition_occurrences_exact_26": len(occurrences)
        == EXPECTED_TRANSITION_OCCURRENCES,
        "transition_occurrences_unique_26": len(
            {tuple(value) for value in occurrence_identities}
        )
        == EXPECTED_TRANSITION_OCCURRENCES,
        "transition_occurrence_sha_exact": canonical_sha(occurrence_identities)
        == EXPECTED_OCCURRENCES26_SHA256,
        "unique_contexts_exact_25": len(context_keys)
        == EXPECTED_UNIQUE_OFFICIAL_CONTEXTS,
        "context_identity_sha_exact": canonical_sha(sorted(context_keys))
        == EXPECTED_CONTEXTS25_SHA256,
        "all_transition_contexts_full_B256": all(value[2] == 256 for value in context_keys),
        "fixed_repair_physical_rows_exact_3": len(FIXED_REPAIR_IDENTITIES)
        == EXPECTED_INITIAL_REPAIR_PHYSICAL_ROWS,
        "fixed_repair_context_guards_exact_4": sum(
            len(value["required_views"]) for value in FIXED_REPAIR_IDENTITIES
        )
        == EXPECTED_INITIAL_REPAIR_CONTEXT_GUARDS,
        "fixed_repair_bindings_all_exact": all(
            all(value[name] for name in ("present", "decision_exact", "pair_exact", "raw_floor_exact", "views_exact"))
            for value in fixed_checks
        ),
        "fixed_repair_transitions_raw_e904_true_cw11_false": all(
            all(
                value[name]
                for name in (
                    "present_in_rows22",
                    "main_metrics_all_changed",
                    "raw_all_true",
                    "e904_all_true",
                    "cw11_all_false",
                )
            )
            for value in fixed_transition_checks
        ),
        "fixed3_is_preregistered_minimum_solution": frozenset(fixed_keys)
        in minimum_solution_sets,
        "fixed3_coverage_meets_all_failed_deficits": all(
            fixed_coverage[gate] >= required for gate, required in failed_deficits.items()
        ),
        "failed_decision_exact": decision["status"] == "failed_dev_consistency"
        and decision["passed_gate_count"] == 51
        and decision["gate_count"] == 60,
        "failed_manifest_six_once_exact": bool(manifest["all_six_attempted_once"])
        and len(manifest["ordered_evaluations"]) == 6,
        "official_commands_exact_6": len(official_commands) == 6,
        "manifest_executions_exact_6": len(manifest_executions) == 6,
        "prereg_manifest_execution_bindings_exact": len(execution_bindings) == 6
        and all(all(value["checks"].values()) for value in execution_bindings),
        "official_command_order_exact": tuple(value["panel"] for value in official_commands)
        == ("pokemonfan", "flg", "core5", "dominic", "luca", "szlach"),
        "official_protocol_B256_W8_BF16_clone": prereg["shared_protocol"]["batch_size"] == 256
        and prereg["shared_protocol"]["workers"] == 8
        and prereg["shared_protocol"]["device"] == "cuda"
        and prereg["shared_protocol"]["prediction_order_argument"] == "policy",
        "official_protocol_split_seed_order_exact": prereg["shared_protocol"]["split"]
        == "valid"
        and prereg["shared_protocol"]["split_mode"] == "archive"
        and prereg["shared_protocol"]["split_seed"] == 20260723
        and prereg["shared_protocol"]["expected_prediction_order"] == "policy_greedy",
        "materialized_model_exact": materialization["checkpoint"]["model_state_sha256"]
        == CW11_MODEL_SHA256,
        "materialized_terminal_33_34_exact": materialization["terminal"]["selected_row_count"]
        == EXPECTED_LEGACY_ROWS
        and materialization["terminal"]["active_pair_count"]
        == EXPECTED_LEGACY_ACTIVE_PAIRS
        and materialization["terminal"]["active_pair_ledger_sha256"]
        == CW11_ACTIVE_LEDGER_SHA256,
        "no_train_exact_matches": plan["minimum_repair"]["train_exact_fingerprint_match_count"]
        == 0,
        "guardplan_all_declared_invariants_true": all(
            bool(value) for value in plan["invariants"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"forensic contract failed: {checks}")
    return {
        "checks": checks,
        "fixed_repair_checks": fixed_checks,
        "fixed_transition_checks": fixed_transition_checks,
        "identity_hashes": {
            "changed47": canonical_sha(ids47),
            "raw_to_cw11_22": canonical_sha(ids22),
            "occurrences26": canonical_sha(occurrence_identities),
            "contexts25": canonical_sha(sorted(context_keys)),
        },
        "transition_classification": transition_classification,
        "context_counts": {
            "physical_rows": len(rows22),
            "panel_occurrences": len(occurrences),
            "unique_panel_batch_size_contexts": len(context_keys),
        },
        "context_key_sha256": canonical_sha(sorted(context_keys)),
        "official_command_bindings": [
            {
                "panel": value["panel"],
                "order": value["order"],
                "command_sha256": value["command_sha256"],
            }
            for value in official_commands
        ],
        "prereg_manifest_execution_bindings": execution_bindings,
    }


def static_source_audit(source: bytes) -> dict[str, Any]:
    text = source.decode("utf-8")
    tree = ast.parse(text, filename=str(SCRIPT))
    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function_names = set(function_nodes)
    required_functions = {
        "dynamic_cut_for_harm",
        "dynamic_restoration_cut_for_favorable_transition",
        "fixed_repair_gates_and_cuts",
        "initial_official_repair_cuts",
        "load_official_six_streams",
        "snapshot_official_six_views",
        "full_stream_separation_oracle",
        "official_context_gradients",
        "apply_additional_from_cw11",
        "encode_terminal_payload",
        "reconstruct_terminal_candidate",
        "run_outer_cutting_plane",
        "run_cw13_consumer",
        "run_probe",
        "validate_cw12_failure_evidence",
    }

    def dotted_name(node: ast.AST) -> str | None:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        return None

    def short_call_name(node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def calls_in(node: ast.AST) -> set[str]:
        return {
            name
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            for name in (short_call_name(child),)
            if name is not None
        }

    call_graph = {
        name: sorted(calls_in(function_nodes[name]))
        for name in sorted(required_functions & function_names)
    }
    called_dotted = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in (dotted_name(node.func),)
        if name is not None
    }
    called_short = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in (short_call_name(node),)
        if name is not None
    }
    forbidden_call_names = {
        "backward",
        "makedirs",
        "mkdir",
        "remove",
        "rename",
        "rmtree",
        "save",
        "savez",
        "step",
        "unlink",
        "write",
        "write_bytes",
        "write_text",
    }
    forbidden_call_hits = sorted(forbidden_call_names & called_short)
    forbidden_import_roots = {
        "http",
        "kaggle",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "urllib",
    }
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_import_hits = sorted(forbidden_import_roots & imported_roots)
    output_arguments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and dotted_name(node.func) == "parser.add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--output"
    ]
    outer = function_nodes.get("run_outer_cutting_plane")
    proposal_loops = [] if outer is None else [
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.For)
        and {
            "snapshot_official_six_views",
            "full_stream_separation_oracle",
            "fixed_repair_gates_and_cuts",
        }.issubset(calls_in(node))
    ]
    segments = {
        name: ast.get_source_segment(text, function_nodes[name]) or ""
        for name in function_nodes
    }
    apply_compact = "".join(segments.get("apply_additional_from_cw11", "").split())
    outer_segment = segments.get("run_outer_cutting_plane", "")
    reconstruct_segment = segments.get("reconstruct_terminal_candidate", "")
    dynamic_segment = segments.get("dynamic_cut_for_harm", "")
    restoration_segment = segments.get(
        "dynamic_restoration_cut_for_favorable_transition", ""
    )
    fixed_segment = segments.get("fixed_repair_gates_and_cuts", "")
    terminal_acceptance_segment = segments.get("terminal_acceptance", "")
    main_segment = segments.get("main", "")
    oracle_hard_keys = set()
    oracle_node = function_nodes.get("full_stream_separation_oracle")
    if oracle_node is not None:
        for node in ast.walk(oracle_node):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name) or node.targets[0].id != "hard":
                continue
            if isinstance(node.value, ast.Dict):
                oracle_hard_keys = {
                    str(key.value)
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
    draft_sentinel = "static" + "_" + "draft"
    unfinished_sentinel = "run implementation" + " has not passed"
    clipped_rejection_sentinel = "QP_was_" + "not_clipped"
    checks = {
        "schema_literal_present": SCHEMA in text,
        "run_functions_declared": required_functions.issubset(function_names),
        "no_forbidden_side_effect_calls": not forbidden_call_hits,
        "no_network_or_submission_imports": not forbidden_import_hits,
        "stdout_only_no_output_argument": not output_arguments,
        "main_run_path_calls_run_probe": "run_probe" in calls_in(
            function_nodes["main"]
        ),
        "no_unfinished_run_sentinel": draft_sentinel not in text
        and unfinished_sentinel not in text,
        "outer_calls_exact_live_chain": {
            "load_official_six_streams",
            "snapshot_official_six_views",
            "full_stream_separation_oracle",
            "official_context_gradients",
            "reconstruct_terminal_candidate",
        }.issubset(calls_in(function_nodes["run_outer_cutting_plane"])),
        "every_proposal_runs_full_six_then_oracle": len(proposal_loops) == 1,
        "full_six_view_order_exact": VIEW_ORDER
        == ("pokemonfan", "flg", "core5", "dominic", "luca", "szlach"),
        "full_six_row_totals_exact": EXPECTED_VIEW_ROWS
        == {
            "flg": 2312,
            "pokemonfan": 15152,
            "core5": 8319,
            "dominic": 2623,
            "luca": 1378,
            "szlach": 2567,
        },
        "official_loader_B256_W8_exact": "batch_size=256" in "".join(
            segments.get("load_official_six_streams", "").split()
        )
        and "num_workers=8" in "".join(
            segments.get("load_official_six_streams", "").split()
        ),
        "general_dynamic_cut_CW11_only_no_raw_floor": all(
            token not in dynamic_segment
            for token in ("raw_snapshot", "raw_U468", "same_official_shape_raw")
        )
        and "frozen_cw11_same_shape" in dynamic_segment
        and "strict_positive_native_bf16_q" in dynamic_segment,
        "favorable_break_uses_explicit_CW11_restoration_cut": all(
            token in restoration_segment
            for token in (
                "dynamic_cut_for_harm",
                "favorable_transition_dynamic_restoration",
                "raw_incorrect",
                "cw11_correct",
                "candidate_incorrect",
            )
        ),
        "favorable_retention_not_an_oracle_hard_fail":
        "favorable_transitions_retained" not in oracle_hard_keys
        and "favorable_transition_restoration_cut_cardinality_exact"
        in oracle_hard_keys
        and "favorable_breaks_have_exact_restoration_cuts" in oracle_hard_keys,
        "preexisting_oracle_integrity_and_completeness_hard_checks_retained": {
            "all_six_row_totals_exact",
            "count_value_logits_exact_raw_all_batches",
            "monitor47_complete_all_view_occurrences",
            "transition22_occurrences_complete",
            "new_harm_cut_cardinality_exact",
            "new_harm_cuts_unique",
        }.issubset(oracle_hard_keys),
        "favorable_retention_remains_terminal_acceptance_gate":
        'oracle["transition_state_checks"]["favorable_transitions_retained"]'
        in terminal_acceptance_segment,
        "restoration_cuts_merge_before_next_outer_iteration":
        'oracle["deterministic_restoration_cuts"]' in outer_segment
        and "merge_active_cuts(active, additions)" in outer_segment
        and "favorable_transition_break_count" in outer_segment,
        "fixed_dynamic_cut_raw_anchored_after_iteration0": "iteration == 0"
        in fixed_segment
        and '"anchor_model": "raw_U468"' in fixed_segment
        and "strict_positive_native_bf16_q" in fixed_segment,
        "fixed_dynamic_cuts_merged_after_proposal": 'fixed["deterministic_repair_cuts"]'
        in outer_segment,
        "QP_cap_is_audit_not_rejection": clipped_rejection_sentinel not in text
        and '"capped_record_only"' in outer_segment,
        "total_application_is_float64_add_then_raw_apply": "np.add(cw11_total,additional,dtype=np.float64)"
        in apply_compact
        and apply_compact.index("restore_raw_actor")
        < apply_compact.index("apply_cumulative_from_raw"),
        "terminal_payload_is_immediately_reconstructed": "encode_terminal_payload"
        in outer_segment
        and "reconstruct_terminal_candidate" in outer_segment
        and outer_segment.index("encode_terminal_payload")
        < outer_segment.rindex("reconstruct_terminal_candidate"),
        "pure_reconstructor_has_no_specialist_access": all(
            token not in reconstruct_segment
            for token in (
                "FORENSIC",
                "GUARDPLAN",
                "PREREG",
                "load_official_six_streams",
                "snapshot_official_six_views",
            )
        ),
        "terminal_payload_contains_math_and_XOR_audits": all(
            token in reconstruct_segment
            for token in (
                "additional_plus_CW11_exact_total_bytes",
                "cw11_model_sha_exact",
                "model_sha_exact",
                "actor_XOR_backup_exact",
            )
        ),
        "downstream_candidate_consumer_hook_retained": "candidate_consumer"
        in segments.get("run_probe", "")
        and "user_candidate_consumer" in outer_segment,
        "official_matmul_precision_high_scoped_and_restored": all(
            token in segments.get("run_cw13_consumer", "")
            for token in (
                'set_float32_matmul_precision("high")',
                "previous_matmul_precision",
                "precision_restored",
            )
        ),
        "first_feasible_literal": "FIRST_FEASIBLE = True" in text,
        "caps_literal": "MAX_OUTER_ITERATIONS = 12" in text
        and "STEP_L2_CAP = 0.001" in text
        and "ADDITIONAL_TOTAL_L2_CAP = 0.001" in text,
        "main_has_static_and_run_branches": 'args.mode == "static"' in main_segment
        and "run_probe(" in main_segment,
        "cw12_consumed_failure_evidence_static_and_run_bound":
        "validate_cw12_failure_evidence" in calls_in(function_nodes["static_result"])
        and "validate_cw12_failure_evidence" in calls_in(function_nodes["run_probe"]),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "forbidden_call_hits": forbidden_call_hits,
        "forbidden_import_hits": forbidden_import_hits,
        "called_dotted": sorted(called_dotted),
        "call_graph": call_graph,
        "proposal_loop_count": len(proposal_loops),
        "oracle_hard_keys": sorted(oracle_hard_keys),
        "required_functions": sorted(required_functions),
        "declared_functions": sorted(function_names),
    }


def import_frozen(
    path: Path, expected_sha: str, expected_mode: int, name: str
) -> ModuleType:
    if EXPECTED_INPUT_MODES.get(path) != expected_mode:
        raise RuntimeError(f"inconsistent frozen import mode binding: {path}")
    observed_stat = path.lstat()
    if (
        not stat.S_ISREG(observed_stat.st_mode)
        or observed_stat.st_nlink != 1
        or sha256_file(path) != expected_sha
        or stat.S_IMODE(observed_stat.st_mode) != expected_mode
    ):
        raise RuntimeError(f"frozen module drift: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_modules() -> dict[str, ModuleType]:
    return {
        "cw11": import_frozen(
            CW11_PROBE,
            MODULE_SHAS[CW11_PROBE],
            EXPECTED_INPUT_MODES[CW11_PROBE],
            "cw13_frozen_cw11",
        ),
        "cw10": import_frozen(
            CW10, MODULE_SHAS[CW10], EXPECTED_INPUT_MODES[CW10], "cw13_frozen_cw10"
        ),
        "cw9": import_frozen(
            CW9, MODULE_SHAS[CW9], EXPECTED_INPUT_MODES[CW9], "cw13_frozen_cw9"
        ),
        "primary": import_frozen(
            PRIMARY,
            MODULE_SHAS[PRIMARY],
            EXPECTED_INPUT_MODES[PRIMARY],
            "cw13_frozen_primary",
        ),
        "cutting": import_frozen(
            CUTTING,
            MODULE_SHAS[CUTTING],
            EXPECTED_INPUT_MODES[CUTTING],
            "cw13_frozen_cutting",
        ),
        "geometry": import_frozen(
            GEOMETRY,
            MODULE_SHAS[GEOMETRY],
            EXPECTED_INPUT_MODES[GEOMETRY],
            "cw13_frozen_geometry",
        ),
        "ram": import_frozen(
            RAM, MODULE_SHAS[RAM], EXPECTED_INPUT_MODES[RAM], "cw13_frozen_ram"
        ),
        "legacy": import_frozen(
            FORENSIC_ANALYZER,
            MODULE_SHAS[FORENSIC_ANALYZER],
            EXPECTED_INPUT_MODES[FORENSIC_ANALYZER],
            "cw13_frozen_forensic_analyzer",
        ),
    }


def compact_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in identity.items()
        if key != "options"
    }


def batch_context_identity(
    view: str,
    archive: Path,
    team_names: Sequence[str],
    batch_ordinal: int,
    cpu_batch: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = {
        "view_name": view,
        "team_filter": list(team_names),
        "archive": str(archive.relative_to(ROOT)),
        "batch_ordinal_one_based": batch_ordinal,
        "actual_batch_size": len(identities),
        "dynamic_max_options": int(cpu_batch["option_mask"].shape[1]),
        "ordered_decision_ids": [str(identity["decision_sha256"]) for identity in identities],
        "ordered_physical_ids": [
            [
                str(identity["archive"]),
                str(identity["member"]),
                int(identity["line_index"]),
                str(identity["line_sha256"]),
            ]
            for identity in identities
        ],
        "native_output_dtype": "torch.bfloat16",
        "evaluator_sha256": MODULE_SHAS[EVALUATOR],
        "protocol": {
            "split": "valid",
            "split_mode": "archive",
            "split_seed": 20260723,
            "batch_size": 256,
            "workers": 8,
            "prediction_order": "policy_greedy",
        },
    }
    return {**value, "context_hash": canonical_sha(value)}


def load_official_six_streams(
    helper: ModuleType,
    legacy: ModuleType,
    model_config: Mapping[str, Any],
    forensic: Mapping[str, Any],
    guardplan: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    torch = helper.torch
    streams: dict[str, list[dict[str, Any]]] = {}
    context_lookup: dict[str, tuple[str, int]] = {}
    row_totals: dict[str, int] = {}
    for view in VIEW_ORDER:
        spec = VIEW_SPECS[view]
        archive = spec["archive"]
        team_names = tuple(spec["team_names"])
        dataset = legacy.IdentifiedValidDataset(archive, model_config, team_names)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            num_workers=8,
            collate_fn=partial(
                legacy.collate_identified,
                max_state_entities=int(model_config["max_state_entities"]),
                entity_fields=int(model_config["entity_fields"]),
                option_fields=int(model_config["option_fields"]),
            ),
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
        )
        batches = []
        for ordinal, (cpu_batch, identities) in enumerate(loader, start=1):
            compact = [compact_identity(value) for value in identities]
            context = batch_context_identity(
                view, archive, team_names, ordinal, cpu_batch, compact
            )
            if context["context_hash"] in context_lookup:
                raise RuntimeError("official batch context hash collision")
            context_lookup[context["context_hash"]] = (view, ordinal)
            batches.append(
                {
                    "cpu_batch": cpu_batch,
                    "identities": compact,
                    "context": context,
                    "raw_snapshot": None,
                    "reference_snapshot": None,
                    "candidate_snapshot": None,
                }
            )
        streams[view] = batches
        row_totals[view] = sum(len(value["identities"]) for value in batches)
        if row_totals[view] != EXPECTED_VIEW_ROWS[view]:
            raise RuntimeError(f"{view}: official stream row total drift")

    fixed_context_checks = []
    for expected in FIXED_CONTEXT_EXPECTATIONS:
        batch = streams[expected["view"]][int(expected["batch_ordinal"]) - 1]
        identity = batch["identities"][int(expected["offset"])]
        observed = batch["context"]
        checks = {
            "batch_size_exact": int(observed["actual_batch_size"])
            == int(expected["batch_size"]),
            "offset_in_range": int(expected["offset"]) < len(batch["identities"]),
            "dynamic_max_options_exact": int(observed["dynamic_max_options"])
            == int(expected["dynamic_max_options"]),
            "line_sha_exact": str(identity["line_sha256"])
            == str(expected["line_sha256"]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"fixed official context drift: {expected}: {checks}")
        fixed_context_checks.append(
            {
                "expected": dict(expected),
                "context_hash": observed["context_hash"],
                "decision_sha256": identity["decision_sha256"],
                "checks": checks,
            }
        )

    expected_occurrences = {
        (
            *identity_tuple(row),
            str(occurrence["panel"]),
            int(occurrence["official_stream_batch_ordinal_one_based"]),
            int(occurrence["official_stream_batch_size"]),
            int(occurrence["official_stream_offset_zero_based"]),
        )
        for row in guardplan["raw_to_cw11_metric_changed_retention_rows"]
        for occurrence in row["panel_view_occurrences"]
    }
    observed_occurrences = set()
    monitor_keys = {
        identity_tuple(row) for row in forensic["unique_physical_changed_rows"]
    }
    monitor_occurrence_count = 0
    for view, batches in streams.items():
        for batch in batches:
            context = batch["context"]
            for offset, identity in enumerate(batch["identities"]):
                key = identity_tuple(identity)
                if key in monitor_keys:
                    monitor_occurrence_count += 1
                occurrence = (
                    *key,
                    view,
                    int(context["batch_ordinal_one_based"]),
                    int(context["actual_batch_size"]),
                    offset,
                )
                if occurrence in expected_occurrences:
                    observed_occurrences.add(occurrence)
    if observed_occurrences != expected_occurrences:
        raise RuntimeError("22-row official occurrence identity mismatch")
    return streams, {
        "view_order": list(VIEW_ORDER),
        "row_totals": row_totals,
        "batch_counts": {view: len(value) for view, value in streams.items()},
        "all_batches_manifest_sha256": canonical_sha(
            [
                batch["context"]
                for view in VIEW_ORDER
                for batch in streams[view]
            ]
        ),
        "context_hashes_unique": len(context_lookup)
        == sum(len(value) for value in streams.values()),
        "fixed_context_checks": fixed_context_checks,
        "raw_to_cw11_occurrences_exact_26": len(observed_occurrences)
        == EXPECTED_TRANSITION_OCCURRENCES,
        "forensic_monitor_occurrence_count": monitor_occurrence_count,
    }


def empty_metric_counts() -> dict[str, int]:
    return {
        "rows": 0,
        "set": 0,
        "hybrid": 0,
        "ordered": 0,
        "top1": 0,
        "count": 0,
        "value": 0,
        "context34_rows": 0,
        "context34_hybrid": 0,
        "context34_ordered": 0,
    }


def add_row_metrics(counts: dict[str, int], row: Mapping[str, Any]) -> None:
    flags = row["flags"]
    counts["rows"] += 1
    counts["set"] += int(flags["set_exact"] is True)
    counts["hybrid"] += int(flags["hybrid_order_exact"] is True)
    counts["ordered"] += int(flags["ordered_exact"] is True)
    counts["top1"] += int(flags["top1_correct"] is True)
    counts["count"] += int(flags["count_correct"] is True)
    counts["value"] += int(flags["value_correct"] is True)
    if int(row["context"]) == 34:
        counts["context34_rows"] += 1
        counts["context34_hybrid"] += int(
            flags["context34_hybrid_order_exact"] is True
        )
        counts["context34_ordered"] += int(flags["context34_ordered_exact"] is True)


def frozen_file_rehashes(view: str) -> dict[str, str]:
    paths = {
        "cw11_checkpoint": CW11_CHECKPOINT,
        "data": VIEW_SPECS[view]["archive"],
        "evaluator": EVALUATOR,
        "bc": BC,
        "ppo": PPO,
        "guardplan": GUARDPLAN,
        "forensic": FORENSIC,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def expected_official_counts(view: str) -> dict[str, int]:
    payload = json.loads((CW11_OFFICIAL_ROOT / f"{view}.json").read_text())
    metrics = payload["metrics"]
    context34 = metrics["by_context"].get("34", {})
    return {
        "rows": int(metrics["rows"]),
        "set": int(metrics["set_exact_correct"]),
        "hybrid": int(metrics["hybrid_order_exact_correct"]),
        "ordered": int(metrics["ordered_exact_correct"]),
        "top1": int(metrics["top1_correct"]),
        "count": int(metrics["count_correct"]),
        "value": int(metrics["value_correct"]),
        "context34_rows": int(context34.get("rows", 0)),
        "context34_hybrid": int(context34.get("hybrid_order_exact_correct", 0)),
        "context34_ordered": int(context34.get("ordered_exact_correct", 0)),
    }


def snapshot_official_six_views(
    helper: ModuleType,
    ram: ModuleType,
    model: Any,
    streams: Mapping[str, Sequence[dict[str, Any]]],
    device: Any,
    *,
    phase: str,
    model_sha256: str,
) -> dict[str, Any]:
    if phase not in {"raw", "reference", "candidate"}:
        raise ValueError("invalid six-view snapshot phase")
    completion = []
    panels = {}
    all_complete = True
    for ordinal, view in enumerate(VIEW_ORDER, start=1):
        model_training_before = bool(model.training)
        model_sha_before = helper.model_state_sha256(model.state_dict())
        pre = frozen_file_rehashes(view)
        counts = empty_metric_counts()
        batch_records = []
        try:
            for batch in streams[view]:
                cpu_batch = batch["cpu_batch"]
                gpu_batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
                snapshot = ram.snapshot_forward(helper, model, gpu_batch, cpu_batch, device)
                if any(value.dtype != helper.torch.bfloat16 for value in snapshot["outputs_cpu"].values()):
                    raise RuntimeError("official clone output is not native BF16")
                batch[f"{phase}_snapshot"] = snapshot
                for row in snapshot["official_rows"]:
                    add_row_metrics(counts, row)
                batch_records.append(
                    {
                        "context_hash": batch["context"]["context_hash"],
                        "batch_ordinal_one_based": batch["context"]["batch_ordinal_one_based"],
                        "actual_batch_size": batch["context"]["actual_batch_size"],
                        "dynamic_max_options": batch["context"]["dynamic_max_options"],
                        "action_sha256": canonical_sha(snapshot["actions"]),
                        "tensor_fingerprints": dict(snapshot["fingerprints"]),
                    }
                )
            rc = 0
            error = None
        except Exception as exc:
            all_complete = False
            rc = 1
            error = f"{type(exc).__name__}: {exc}"
        post = frozen_file_rehashes(view)
        model_sha_after = helper.model_state_sha256(model.state_dict())
        model_training_after = bool(model.training)
        record = {
            "execution": "in_process_frozen_official_evaluator_clone",
            "ordinal": ordinal,
            "view": view,
            "expected_model_state_sha256": model_sha256,
            "model_state_sha256": model_sha_after,
            "model_state_sha256_before": model_sha_before,
            "model_state_sha256_after": model_sha_after,
            "model_state_unchanged_exact": model_sha_before
            == model_sha_after
            == model_sha256,
            "model_eval_before_after_exact": not model_training_before
            and not model_training_after,
            "float32_matmul_precision": helper.torch.get_float32_matmul_precision(),
            "rows": counts["rows"],
            "expected_rows": EXPECTED_VIEW_ROWS[view],
            "return_code": rc,
            "error": error,
            "input_pre_rehash": pre,
            "input_post_rehash": post,
            "input_rehash_exact": pre == post,
            "batch_manifest_sha256": canonical_sha(batch_records),
            "in_memory_output_sha256": canonical_sha(
                {"metrics": counts, "batches": batch_records}
            ),
        }
        completion.append(record)
        panels[view] = {"metrics": counts, "batches": batch_records}
    hard = {
        "six_views_completed": len(completion) == 6 and all_complete,
        "view_order_exact": tuple(value["view"] for value in completion) == VIEW_ORDER,
        "all_rc0": all(value["return_code"] == 0 for value in completion),
        "all_input_rehash_exact": all(value["input_rehash_exact"] for value in completion),
        "model_state_unchanged_every_view": all(
            value["model_state_unchanged_exact"] for value in completion
        ),
        "model_eval_every_view": all(
            value["model_eval_before_after_exact"] for value in completion
        ),
        "official_float32_matmul_precision_high": all(
            value["float32_matmul_precision"] == "high" for value in completion
        ),
        "all_row_totals_exact": all(
            panels[view]["metrics"]["rows"] == EXPECTED_VIEW_ROWS[view]
            for view in VIEW_ORDER
        ),
        "one_model_sha_across_six": {
            value["model_state_sha256"] for value in completion
        }
        == {model_sha256},
    }
    if not all(hard.values()):
        raise RuntimeError(f"six-view completion contract failed: {hard}")
    if phase == "reference":
        reproductions = {
            view: panels[view]["metrics"] == expected_official_counts(view)
            for view in VIEW_ORDER
        }
        if not all(reproductions.values()):
            raise RuntimeError(f"CW11 official clone reproduction failed: {reproductions}")
    else:
        reproductions = None
    return {
        "phase": phase,
        "model_state_sha256": model_sha256,
        "completion_ledger": completion,
        "panels": panels,
        "hard_checks": hard,
        "reference_reproduction": reproductions,
    }


def policy_flag(row: Mapping[str, Any], metric: str) -> bool:
    key = {
        "set": "set_exact",
        "hybrid": "hybrid_order_exact",
        "ordered": "ordered_exact",
        "top1": "top1_correct",
    }[metric]
    return bool(row["flags"][key])


def pair_margin(snapshot: Mapping[str, Any], offset: int, positive: int, negative: int) -> float:
    logits = snapshot["outputs_cpu"]["policy_logits"]
    return float(logits[offset, positive].float() - logits[offset, negative].float())


def positive_bf16_q(snapshot: Mapping[str, Any], offset: int, positive: int, negative: int, torch: Any) -> float:
    logits = snapshot["outputs_cpu"]["policy_logits"]
    spacings = []
    for value in (logits[offset, positive], logits[offset, negative]):
        spacings.extend(
            (
                torch.nextafter(value, torch.full_like(value, float("inf"))) - value,
                value - torch.nextafter(value, torch.full_like(value, float("-inf"))),
            )
        )
    result = max(float(value.float()) for value in spacings)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("invalid native-BF16 positive quantum")
    return result


def first_different_pair(expert: Sequence[int], candidate: Sequence[int]) -> tuple[int, int] | None:
    if len(expert) != len(candidate):
        return None
    for expected, observed in zip(expert, candidate):
        if int(expected) != int(observed):
            return int(expected), int(observed)
    return None


def set_missing_pair(expert: Sequence[int], candidate: Sequence[int]) -> tuple[int, int] | None:
    expert_set = {int(value) for value in expert}
    candidate_set = {int(value) for value in candidate}
    missing = [int(value) for value in expert if int(value) not in candidate_set]
    included = [int(value) for value in candidate if int(value) not in expert_set]
    if not missing or not included:
        return None
    return missing[0], included[0]


def dynamic_cut_for_harm(
    view: str,
    batch: Mapping[str, Any],
    offset: int,
    metric: str,
    iteration: int,
    torch: Any,
) -> dict[str, Any]:
    reference = batch["reference_snapshot"]
    candidate = batch["candidate_snapshot"]
    reference_row = reference["official_rows"][offset]
    candidate_row = candidate["official_rows"][offset]
    expert = [int(value) for value in reference_row["expert_order"]]
    observed = [int(value) for value in candidate_row["predicted_order"]]
    hybrid_observed = [int(value) for value in candidate_row["hybrid_order"]]
    if metric == "top1":
        option_pair = (
            int(reference_row["top1_index_full_policy_vector"]),
            int(candidate_row["top1_index_full_policy_vector"]),
        )
        construction = "reference_correct_top1_vs_candidate_wrong_top1"
    elif metric == "set":
        option_pair = set_missing_pair(expert, observed)
        construction = "reference_expert_missing_vs_candidate_included"
    elif metric == "hybrid" and int(reference_row["context"]) != 34:
        option_pair = set_missing_pair(expert, hybrid_observed)
        if option_pair is None:
            option_pair = first_different_pair(expert, hybrid_observed)
            construction = "reference_hybrid_first_order_difference"
        else:
            construction = "reference_hybrid_missing_vs_candidate_included"
    else:
        option_pair = first_different_pair(
            expert, hybrid_observed if metric == "hybrid" else observed
        )
        construction = "reference_expert_vs_candidate_first_order_difference"
    if option_pair is None or option_pair[0] == option_pair[1]:
        raise RuntimeError("new harm has no deterministic separating option pair")
    positive, negative = option_pair
    reference_floor = pair_margin(reference, offset, positive, negative)
    tie_floor = positive_bf16_q(reference, offset, positive, negative, torch)
    threshold = max(reference_floor, tie_floor)
    current_margin = pair_margin(candidate, offset, positive, negative)
    if not math.isfinite(threshold) or current_margin >= threshold:
        raise RuntimeError("derived new-harm cut does not separate candidate")
    identity = batch["identities"][offset]
    context = batch["context"]
    return {
        "context_type": "official_B256",
        "context_hash": context["context_hash"],
        "view": view,
        "batch_ordinal_one_based": int(context["batch_ordinal_one_based"]),
        "actual_batch_size": int(context["actual_batch_size"]),
        "dynamic_max_options": int(context["dynamic_max_options"]),
        "offset_zero_based": offset,
        "physical_identity": list(identity_tuple(identity)),
        "decision_sha256": str(identity["decision_sha256"]),
        "metric": metric,
        "positive_option": positive,
        "negative_option": negative,
        "threshold": threshold,
        "threshold_sources": {
            "frozen_cw11_same_shape": reference_floor,
            "strict_positive_native_bf16_q": tie_floor,
        },
        "anchor_model": "frozen_CW11_4317",
        "anchor_source": "same_official_shape_CW11_margin",
        "current_margin": current_margin,
        "construction": construction,
        "created_iteration": iteration,
        "origins": ["full_six_stream_new_harm"],
    }


def dynamic_restoration_cut_for_favorable_transition(
    view: str,
    batch: Mapping[str, Any],
    offset: int,
    metric: str,
    iteration: int,
    torch: Any,
) -> dict[str, Any]:
    """Separate a proposal that destroyed a frozen RAW->CW11 improvement."""

    raw_row = batch["raw_snapshot"]["official_rows"][offset]
    reference_row = batch["reference_snapshot"]["official_rows"][offset]
    candidate_row = batch["candidate_snapshot"]["official_rows"][offset]
    transition_checks = {
        "raw_incorrect": policy_flag(raw_row, metric) is False,
        "cw11_correct": policy_flag(reference_row, metric) is True,
        "candidate_incorrect": policy_flag(candidate_row, metric) is False,
    }
    if not all(transition_checks.values()):
        raise RuntimeError(
            "favorable transition restoration requested for non-broken transition"
        )
    cut = dynamic_cut_for_harm(view, batch, offset, metric, iteration, torch)
    cut["construction"] = (
        "favorable_transition_restoration__" + str(cut["construction"])
    )
    cut["origins"] = ["favorable_transition_dynamic_restoration"]
    cut["restoration_transition"] = {
        "raw_correct": False,
        "cw11_correct": True,
        "candidate_correct": False,
        "checks": transition_checks,
    }
    return cut


def fixed_repair_gates_and_cuts(
    helper: ModuleType,
    streams: Mapping[str, Sequence[dict[str, Any]]],
    *,
    iteration: int,
) -> dict[str, Any]:
    records = []
    cuts = []
    for context_expected in FIXED_CONTEXT_EXPECTATIONS:
        view = str(context_expected["view"])
        batch = streams[view][int(context_expected["batch_ordinal"]) - 1]
        offset = int(context_expected["offset"])
        identity = batch["identities"][offset]
        physical = next(
            value
            for value in FIXED_REPAIR_IDENTITIES
            if str(value["line_sha256"]) == str(identity["line_sha256"])
        )
        raw = batch["raw_snapshot"]
        candidate = batch["candidate_snapshot"]
        positive = int(physical["positive_option"])
        negative = int(physical["negative_option"])
        raw_margin = pair_margin(raw, offset, positive, negative)
        candidate_margin = pair_margin(candidate, offset, positive, negative)
        raw_floor = float(physical["raw_margin"])
        raw_metric_flags = {
            metric: policy_flag(raw["official_rows"][offset], metric)
            for metric in POLICY_METRICS
        }
        metric_flags = {
            metric: policy_flag(candidate["official_rows"][offset], metric)
            for metric in POLICY_METRICS
        }
        if not all(raw_metric_flags.values()):
            raise RuntimeError("fixed repair raw anchor is not policy-correct")
        strict_positive = bool(physical.get("strict_positive", False))
        raw_row = raw["official_rows"][offset]
        candidate_row = candidate["official_rows"][offset]
        raw_order = [int(value) for value in raw_row["predicted_order"]]
        raw_hybrid_order = [int(value) for value in raw_row["hybrid_order"]]
        candidate_order = [int(value) for value in candidate_row["predicted_order"]]
        candidate_hybrid_order = [
            int(value) for value in candidate_row["hybrid_order"]
        ]
        for metric in POLICY_METRICS:
            if iteration == 0 or metric_flags[metric]:
                continue
            if metric == "top1":
                option_pair = (
                    int(raw_row["top1_index_full_policy_vector"]),
                    int(candidate_row["top1_index_full_policy_vector"]),
                )
                construction = "fixed_raw_correct_top1_vs_candidate_wrong_top1"
            elif metric == "set":
                option_pair = set_missing_pair(raw_order, candidate_order)
                construction = "fixed_raw_missing_vs_candidate_included"
            elif metric == "hybrid" and int(raw_row["context"]) != 34:
                option_pair = set_missing_pair(
                    raw_hybrid_order, candidate_hybrid_order
                )
                if option_pair is None:
                    option_pair = first_different_pair(
                        raw_hybrid_order, candidate_hybrid_order
                    )
                    construction = "fixed_raw_hybrid_first_order_difference"
                else:
                    construction = "fixed_raw_hybrid_missing_vs_candidate_included"
            else:
                option_pair = first_different_pair(
                    raw_hybrid_order if metric == "hybrid" else raw_order,
                    candidate_hybrid_order if metric == "hybrid" else candidate_order,
                )
                construction = "fixed_raw_vs_candidate_first_order_difference"
            if option_pair is None or option_pair[0] == option_pair[1]:
                raise RuntimeError(
                    "fixed repair policy failure has no deterministic separating pair"
                )
            dynamic_positive, dynamic_negative = option_pair
            dynamic_raw_floor = pair_margin(
                raw, offset, dynamic_positive, dynamic_negative
            )
            tie_floor = positive_bf16_q(
                raw,
                offset,
                dynamic_positive,
                dynamic_negative,
                helper.torch,
            )
            threshold = (
                max(dynamic_raw_floor, tie_floor)
                if strict_positive
                else dynamic_raw_floor
            )
            current_margin = pair_margin(
                candidate, offset, dynamic_positive, dynamic_negative
            )
            if not math.isfinite(threshold) or current_margin >= threshold:
                raise RuntimeError(
                    "fixed raw-anchored dynamic cut does not separate candidate"
                )
            cuts.append(
                {
                    "context_type": "official_B256",
                    "context_hash": batch["context"]["context_hash"],
                    "view": view,
                    "batch_ordinal_one_based": batch["context"]
                    ["batch_ordinal_one_based"],
                    "actual_batch_size": batch["context"]["actual_batch_size"],
                    "dynamic_max_options": batch["context"]["dynamic_max_options"],
                    "offset_zero_based": offset,
                    "physical_identity": list(identity_tuple(identity)),
                    "decision_sha256": identity["decision_sha256"],
                    "metric": f"fixed_dynamic_{metric}",
                    "positive_option": dynamic_positive,
                    "negative_option": dynamic_negative,
                    "threshold": threshold,
                    "threshold_sources": {
                        "same_official_shape_raw": dynamic_raw_floor,
                        "strict_positive_native_bf16_q": (
                            tie_floor if strict_positive else None
                        ),
                    },
                    "anchor_model": "raw_U468",
                    "anchor_source": "same_official_shape_raw_margin",
                    "strict_positive_required": strict_positive,
                    "current_margin": current_margin,
                    "construction": construction,
                    "created_iteration": iteration,
                    "origins": ["fixed3_policy_flag_dynamic_repair"],
                }
            )
        records.append(
            {
                "view": view,
                "context_hash": batch["context"]["context_hash"],
                "batch_ordinal_one_based": batch["context"]["batch_ordinal_one_based"],
                "actual_batch_size": batch["context"]["actual_batch_size"],
                "dynamic_max_options": batch["context"]["dynamic_max_options"],
                "offset_zero_based": offset,
                "physical_identity": list(identity_tuple(identity)),
                "decision_sha256": identity["decision_sha256"],
                "positive_option": positive,
                "negative_option": negative,
                "raw_margin": raw_margin,
                "frozen_raw_floor": raw_floor,
                "candidate_margin": candidate_margin,
                "pair_floor_pass": candidate_margin >= raw_floor,
                "strict_positive_required": strict_positive,
                "strict_positive_pass": not strict_positive or candidate_margin > 0.0,
                "raw_policy_flags": raw_metric_flags,
                "policy_flags": metric_flags,
                "all_policy_flags_pass": all(metric_flags.values()),
            }
        )
    cuts.sort(
        key=lambda value: (
            VIEW_ORDER.index(value["view"]),
            value["batch_ordinal_one_based"],
            value["offset_zero_based"],
            value["metric"],
        )
    )
    return {
        "context_guard_count": len(records),
        "records": records,
        "pass": len(records) == EXPECTED_INITIAL_REPAIR_CONTEXT_GUARDS
        and all(
            value["pair_floor_pass"]
            and value["strict_positive_pass"]
            and value["all_policy_flags_pass"]
            for value in records
        ),
        "deterministic_repair_cuts": cuts,
        "deterministic_repair_cut_count": len(cuts),
    }


def cut_identity(cut: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(cut["context_hash"]),
        str(cut["decision_sha256"]),
        str(cut["metric"]),
        int(cut["positive_option"]),
        int(cut["negative_option"]),
    )


def official_gate_records(
    candidate_metrics: Mapping[str, Mapping[str, int]],
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    records = []
    for view in VIEW_ORDER:
        for metric, rule in prereg["authoritative_panel_gates"][view].items():
            observed = int(candidate_metrics[view][metric])
            if "exact" in rule:
                operator = "=="
                threshold = int(rule["exact"])
                passed = observed == threshold
            else:
                operator = ">="
                threshold = int(rule["minimum"])
                passed = observed >= threshold
            records.append(
                {
                    "gate": f"{view}.{metric}",
                    "observed": observed,
                    "operator": operator,
                    "threshold": threshold,
                    "pass": passed,
                }
            )
    if len(records) != 60:
        raise RuntimeError("authoritative gate cardinality drift")
    return {
        "gate_count": len(records),
        "passed_gate_count": sum(int(value["pass"]) for value in records),
        "pass": all(value["pass"] for value in records),
        "records": records,
        "failed": [value for value in records if not value["pass"]],
    }


def involved_pairs_from_forensic(row: Mapping[str, Any]) -> list[tuple[int, int]]:
    pairs = set()
    for transition in row["pairwise"].values():
        first = transition["first_prediction_difference"]
        if first is None:
            continue
        left = first["left_choice"]
        right = first["right_choice"]
        if left is not None and right is not None and int(left) != int(right):
            pairs.add((int(left), int(right)))
    return sorted(pairs)


def full_stream_separation_oracle(
    helper: ModuleType,
    streams: Mapping[str, Sequence[dict[str, Any]]],
    forensic: Mapping[str, Any],
    guardplan: Mapping[str, Any],
    prereg: Mapping[str, Any],
    *,
    iteration: int,
) -> dict[str, Any]:
    monitor_source = {
        identity_tuple(row): row for row in forensic["unique_physical_changed_rows"]
    }
    transition_source = {
        identity_tuple(row): row
        for row in guardplan["raw_to_cw11_metric_changed_retention_rows"]
    }
    monitor_records = []
    transition_records = []
    new_harms = []
    new_harm_cuts = []
    restoration_cuts = []
    new_decision_flips = []
    candidate_metrics = {view: empty_metric_counts() for view in VIEW_ORDER}
    count_value_exact = True
    for view in VIEW_ORDER:
        for batch in streams[view]:
            raw_snapshot = batch["raw_snapshot"]
            reference = batch["reference_snapshot"]
            candidate = batch["candidate_snapshot"]
            if raw_snapshot is None or reference is None or candidate is None:
                raise RuntimeError("oracle called without complete snapshots")
            count_value_exact = count_value_exact and helper.torch.equal(
                candidate["outputs_cpu"]["count_logits"],
                raw_snapshot["outputs_cpu"]["count_logits"],
            ) and helper.torch.equal(
                candidate["outputs_cpu"]["value_logits"],
                raw_snapshot["outputs_cpu"]["value_logits"],
            )
            context = batch["context"]
            for offset, identity in enumerate(batch["identities"]):
                raw_row = raw_snapshot["official_rows"][offset]
                reference_row = reference["official_rows"][offset]
                candidate_row = candidate["official_rows"][offset]
                add_row_metrics(candidate_metrics[view], candidate_row)
                key = identity_tuple(identity)
                transition_source_row = transition_source.get(key)
                favorable_metrics = set()
                if transition_source_row is not None:
                    favorable_metrics = {
                        metric
                        for metric in transition_source_row["pairwise"]["raw_to_cw11"]
                        ["changed_metrics"]
                        if not bool(transition_source_row["states"]["raw"][metric])
                        and bool(transition_source_row["states"]["cw11"][metric])
                    }
                row_new_harms = []
                for metric in POLICY_METRICS:
                    if policy_flag(reference_row, metric) and not policy_flag(candidate_row, metric):
                        harm = {
                            "view": view,
                            "context_hash": context["context_hash"],
                            "batch_ordinal_one_based": context["batch_ordinal_one_based"],
                            "actual_batch_size": context["actual_batch_size"],
                            "offset_zero_based": offset,
                            "physical_identity": list(identity_tuple(identity)),
                            "decision_sha256": identity["decision_sha256"],
                            "metric": metric,
                            "cw11_correct": True,
                            "candidate_correct": False,
                        }
                        row_new_harms.append(harm)
                        new_harms.append(harm)
                        if metric in favorable_metrics:
                            restoration_cuts.append(
                                dynamic_restoration_cut_for_favorable_transition(
                                    view, batch, offset, metric, iteration, helper.torch
                                )
                            )
                        else:
                            new_harm_cuts.append(
                                dynamic_cut_for_harm(
                                    view, batch, offset, metric, iteration, helper.torch
                                )
                            )
                if (
                    reference_row["predicted_order"] != candidate_row["predicted_order"]
                    or reference_row["top1_index_full_policy_vector"]
                    != candidate_row["top1_index_full_policy_vector"]
                ):
                    new_decision_flips.append(
                        {
                            "view": view,
                            "context_hash": context["context_hash"],
                            "batch_ordinal_one_based": context["batch_ordinal_one_based"],
                            "offset_zero_based": offset,
                            "physical_identity": list(identity_tuple(identity)),
                            "decision_sha256": identity["decision_sha256"],
                            "cw11_order": reference_row["predicted_order"],
                            "candidate_order": candidate_row["predicted_order"],
                            "cw11_top1": reference_row["top1_index_full_policy_vector"],
                            "candidate_top1": candidate_row["top1_index_full_policy_vector"],
                            "new_harm_metrics": [value["metric"] for value in row_new_harms],
                        }
                    )
                if key in monitor_source:
                    source = monitor_source[key]
                    pair_margins = []
                    for positive, negative in involved_pairs_from_forensic(source):
                        pair_margins.append(
                            {
                                "positive_option": positive,
                                "negative_option": negative,
                                "raw_margin": pair_margin(raw_snapshot, offset, positive, negative),
                                "cw11_margin": pair_margin(reference, offset, positive, negative),
                                "candidate_margin": pair_margin(candidate, offset, positive, negative),
                            }
                        )
                    monitor_records.append(
                        {
                            "physical_identity": list(key),
                            "decision_sha256": identity["decision_sha256"],
                            "view": view,
                            "context": {
                                **context,
                                "offset_zero_based": offset,
                            },
                            "raw_order": source["states"]["raw"]["order"],
                            "e904_order": source["states"]["e904"]["order"],
                            "cw11_order": source["states"]["cw11"]["order"],
                            "candidate_order": candidate_row["predicted_order"],
                            "candidate_policy_flags": {
                                metric: policy_flag(candidate_row, metric)
                                for metric in POLICY_METRICS
                            },
                            "pair_margins": pair_margins,
                        }
                    )
                if key in transition_source:
                    source = transition_source[key]
                    for metric in source["pairwise"]["raw_to_cw11"]["changed_metrics"]:
                        before = bool(source["states"]["raw"][metric])
                        after = bool(source["states"]["cw11"][metric])
                        classification = "favorable" if not before and after else "harmful"
                        candidate_flag = policy_flag(candidate_row, metric)
                        transition_records.append(
                            {
                                "physical_identity": list(key),
                                "decision_sha256": identity["decision_sha256"],
                                "view": view,
                                "context_hash": context["context_hash"],
                                "batch_ordinal_one_based": context[
                                    "batch_ordinal_one_based"
                                ],
                                "actual_batch_size": context["actual_batch_size"],
                                "offset_zero_based": offset,
                                "metric": metric,
                                "raw_correct": before,
                                "cw11_correct": after,
                                "candidate_correct": candidate_flag,
                                "classification": classification,
                                "retained": candidate_flag if classification == "favorable" else True,
                                "harmful_repaired": candidate_flag if classification == "harmful" else None,
                            }
                        )
    monitor_records.sort(
        key=lambda value: (
            VIEW_ORDER.index(value["view"]),
            value["context"]["batch_ordinal_one_based"],
            value["context"]["offset_zero_based"],
        )
    )
    transition_records.sort(
        key=lambda value: (
            VIEW_ORDER.index(value["view"]),
            tuple(value["physical_identity"]),
            POLICY_METRICS.index(value["metric"]),
        )
    )
    new_harms.sort(
        key=lambda value: (
            VIEW_ORDER.index(value["view"]),
            value["batch_ordinal_one_based"],
            value["offset_zero_based"],
            POLICY_METRICS.index(value["metric"]),
        )
    )
    new_harm_cuts.sort(
        key=lambda value: (
            VIEW_ORDER.index(value["view"]),
            value["batch_ordinal_one_based"],
            value["offset_zero_based"],
            POLICY_METRICS.index(value["metric"]),
            value["positive_option"],
            value["negative_option"],
        )
    )
    restoration_cuts.sort(
        key=lambda value: (
            VIEW_ORDER.index(value["view"]),
            value["batch_ordinal_one_based"],
            value["offset_zero_based"],
            POLICY_METRICS.index(value["metric"]),
            value["positive_option"],
            value["negative_option"],
        )
    )
    all_separating_cuts = sorted(
        [*new_harm_cuts, *restoration_cuts], key=cut_identity
    )
    expected_monitor_occurrence_identities = {
        (
            *identity_tuple(row),
            str(occurrence["panel"]),
            int(occurrence["official_stream_batch_ordinal_one_based"]),
            int(occurrence["official_stream_batch_size"]),
            int(occurrence["official_stream_offset_zero_based"]),
        )
        for row in forensic["unique_physical_changed_rows"]
        for occurrence in row["panel_view_occurrences"]
    }
    observed_monitor_occurrence_identities = {
        (
            *tuple(value["physical_identity"]),
            str(value["view"]),
            int(value["context"]["batch_ordinal_one_based"]),
            int(value["context"]["actual_batch_size"]),
            int(value["context"]["offset_zero_based"]),
        )
        for value in monitor_records
    }
    expected_transition_occurrence_identities = {
        (
            *identity_tuple(row),
            str(occurrence["panel"]),
            int(occurrence["official_stream_batch_ordinal_one_based"]),
            int(occurrence["official_stream_batch_size"]),
            int(occurrence["official_stream_offset_zero_based"]),
        )
        for row in guardplan["raw_to_cw11_metric_changed_retention_rows"]
        for occurrence in row["panel_view_occurrences"]
    }
    observed_transition_occurrence_identities = {
        (
            *tuple(value["physical_identity"]),
            str(value["view"]),
            int(value["batch_ordinal_one_based"]),
            int(value["actual_batch_size"]),
            int(value["offset_zero_based"]),
        )
        for value in transition_records
    }
    favorable_transition_breaks = [
        value
        for value in transition_records
        if value["classification"] == "favorable" and not value["retained"]
    ]
    favorable_break_identities = {
        (
            str(value["context_hash"]),
            str(value["decision_sha256"]),
            str(value["metric"]),
        )
        for value in favorable_transition_breaks
    }
    restoration_cut_identities = {
        (
            str(value["context_hash"]),
            str(value["decision_sha256"]),
            str(value["metric"]),
        )
        for value in restoration_cuts
    }
    transition_state_checks = {
        "favorable_transitions_retained": not favorable_transition_breaks,
        "favorable_breaks_have_exact_restoration_cuts":
        favorable_break_identities == restoration_cut_identities,
    }
    hard = {
        "all_six_row_totals_exact": candidate_metrics
        and all(candidate_metrics[view]["rows"] == EXPECTED_VIEW_ROWS[view] for view in VIEW_ORDER),
        "count_value_logits_exact_raw_all_batches": count_value_exact,
        "monitor47_complete_all_view_occurrences":
        observed_monitor_occurrence_identities
        == expected_monitor_occurrence_identities,
        "transition22_occurrences_complete":
        observed_transition_occurrence_identities
        == expected_transition_occurrence_identities,
        "new_harm_cut_cardinality_exact": len(all_separating_cuts)
        == len(new_harms),
        "new_harm_cuts_unique": len(
            {cut_identity(value) for value in all_separating_cuts}
        )
        == len(all_separating_cuts),
        "favorable_transition_restoration_cut_cardinality_exact":
        len(restoration_cuts) == len(favorable_transition_breaks),
        "favorable_transition_restoration_cuts_unique": len(
            {cut_identity(value) for value in restoration_cuts}
        )
        == len(restoration_cuts),
        "favorable_breaks_have_exact_restoration_cuts": transition_state_checks[
            "favorable_breaks_have_exact_restoration_cuts"
        ],
    }
    if not all(hard.values()):
        raise RuntimeError(f"full stream separation oracle failed: {hard}")
    gates = official_gate_records(candidate_metrics, prereg)
    return {
        "iteration": iteration,
        "candidate_metrics": candidate_metrics,
        "authoritative_60_gates": gates,
        "new_harm_count_vs_cw11": len(new_harms),
        "new_harms": new_harms,
        "deterministic_new_harm_cuts": new_harm_cuts,
        "deterministic_restoration_cuts": restoration_cuts,
        "deterministic_all_separating_cuts": all_separating_cuts,
        "favorable_transition_break_count": len(favorable_transition_breaks),
        "favorable_transition_breaks": favorable_transition_breaks,
        "new_decision_flip_count_vs_cw11": len(new_decision_flips),
        "new_decision_flips": new_decision_flips,
        "forensic47_monitor": monitor_records,
        "transition22_ledger": transition_records,
        "transition22_classification_counts": {
            "favorable_cells": sum(
                int(value["classification"] == "favorable")
                for value in transition_records
            ),
            "harmful_cells": sum(
                int(value["classification"] == "harmful")
                for value in transition_records
            ),
            "harmful_cells_repaired": sum(
                int(
                    value["classification"] == "harmful"
                    and value["harmful_repaired"] is True
                )
                for value in transition_records
            ),
        },
        "transition_state_checks": transition_state_checks,
        "hard_checks": hard,
    }


def canonical_ledger_to_pairs(
    ledger: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for record in ledger:
        row_index, positive, negative = [int(value) for value in record["key"]]
        pairs.append(
            {
                "row_index": row_index,
                "identity": dict(record["identity"]),
                "role": str(record["role"]),
                "positive_option": positive,
                "negative_option": negative,
                "threshold": float(record["threshold"]),
                "threshold_source": str(record["threshold_source"]),
                "threshold_sources": [str(value) for value in record["threshold_sources"]],
                "threshold_history": [dict(value) for value in record["threshold_history"]],
                "origins": [str(value) for value in record["origins"]],
                "constructions": [str(value) for value in record["construction"]],
                "created_iteration": int(record["created_iteration"]),
            }
        )
    return pairs


def load_legacy_B33(
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
) -> dict[str, Any]:
    cw11 = modules["cw11"]
    old, descriptor_audit = cw11.reconstruct_old_descriptors(
        modules["primary"], modules["geometry"], modules["cw9"], modules["cw10"]
    )
    descriptors = [
        *old,
        *[dict(value) for value in cw11.ADDED_VALID_GUARDS],
    ]
    rows, loading_audit = cw11.load_selected_train_and_exact_valid_rows(
        modules["geometry"],
        context["helper"],
        descriptors,
        context["model_config"],
    )
    helper = context["helper"]
    cpu_batch = helper.evaluator.collate_ordered(
        [item["features"] for item in rows],
        max_state_entities=int(context["model_config"]["max_state_entities"]),
        entity_fields=int(context["model_config"]["entity_fields"]),
        option_fields=int(context["model_config"]["option_fields"]),
    )
    device = next(context["model"].parameters()).device
    gpu_batch = {key: value.to(device) for key, value in cpu_batch.items()}
    pairs = canonical_ledger_to_pairs(context["active_pair_ledger"])
    ledger_audit = modules["cutting"].active_pair_ledger_audit(pairs)
    identities = [cw11.identity_record(value) for value in descriptors]
    context_value = {
        "context_type": "legacy_B33_native_BF16",
        "batch_size": len(descriptors),
        "ordered_descriptor_identities": identities,
        "native_output_dtype": "torch.bfloat16",
        "cw11_probe_sha256": MODULE_SHAS[CW11_PROBE],
    }
    checks = {
        "rows_exact_33": len(descriptors) == EXPECTED_LEGACY_ROWS,
        "pairs_exact_34": len(pairs) == EXPECTED_LEGACY_ACTIVE_PAIRS,
        "ledger_sha_exact": ledger_audit["canonical_json_sha256"]
        == CW11_ACTIVE_LEDGER_SHA256,
        "incoming_selected_gate_pass": bool(context["selected_row_gate"]["pass"]),
        "incoming_model_sha_exact": str(context["candidate_model_state_sha256"])
        == CW11_MODEL_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"legacy B33 reconstruction failed: {checks}")
    return {
        "descriptors": descriptors,
        "cpu_batch": cpu_batch,
        "gpu_batch": gpu_batch,
        "pairs": pairs,
        "context_hash": canonical_sha(context_value),
        "context_identity": context_value,
        "raw_snapshot": None,
        "reference_snapshot": None,
        "candidate_snapshot": None,
        "audit": {
            "checks": checks,
            "descriptor_audit": descriptor_audit,
            "loading_audit": loading_audit,
            "ledger_audit": ledger_audit,
        },
    }


def legacy_pair_cuts(legacy: Mapping[str, Any]) -> list[dict[str, Any]]:
    cuts = []
    for ordinal, pair in enumerate(legacy["pairs"]):
        identity = pair["identity"]
        cuts.append(
            {
                "context_type": "legacy_B33",
                "context_hash": legacy["context_hash"],
                "view": "legacy_B33",
                "batch_ordinal_one_based": 1,
                "actual_batch_size": 33,
                "dynamic_max_options": int(legacy["cpu_batch"]["option_mask"].shape[1]),
                "offset_zero_based": int(pair["row_index"]),
                "physical_identity": [
                    str(identity["panel"]),
                    str(identity["member"]),
                    int(identity["line_index_zero_based"]),
                    str(identity["line_sha256"]),
                ],
                "decision_sha256": f"legacy-line:{identity['line_sha256']}",
                "metric": f"legacy_pair_{ordinal:02d}",
                "positive_option": int(pair["positive_option"]),
                "negative_option": int(pair["negative_option"]),
                "threshold": float(pair["threshold"]),
                "threshold_sources": {
                    "frozen_CW11_active_ledger": float(pair["threshold"]),
                },
                "anchor_model": "frozen_CW11_legacy_ledger",
                "anchor_source": str(pair["threshold_source"]),
                "construction": "verbatim_CW11_active_pair",
                "created_iteration": 0,
                "origins": ["legacy_B33_compatibility"],
            }
        )
    if len(cuts) != EXPECTED_LEGACY_ACTIVE_PAIRS:
        raise RuntimeError("legacy cut cardinality drift")
    return cuts


def initial_official_repair_cuts(
    streams: Mapping[str, Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    cuts = []
    for expected_context in FIXED_CONTEXT_EXPECTATIONS:
        view = str(expected_context["view"])
        batch = streams[view][int(expected_context["batch_ordinal"]) - 1]
        offset = int(expected_context["offset"])
        identity = batch["identities"][offset]
        physical = next(
            value
            for value in FIXED_REPAIR_IDENTITIES
            if str(value["line_sha256"]) == str(identity["line_sha256"])
        )
        positive = int(physical["positive_option"])
        negative = int(physical["negative_option"])
        raw_margin = pair_margin(batch["raw_snapshot"], offset, positive, negative)
        cw11_margin = pair_margin(
            batch["reference_snapshot"], offset, positive, negative
        )
        threshold = float(physical["raw_margin"])
        if raw_margin != threshold or not threshold > 0.0 or not cw11_margin < threshold:
            raise RuntimeError("fixed repair official raw-floor binding failed")
        cuts.append(
            {
                "context_type": "official_B256",
                "context_hash": batch["context"]["context_hash"],
                "view": view,
                "batch_ordinal_one_based": batch["context"]["batch_ordinal_one_based"],
                "actual_batch_size": batch["context"]["actual_batch_size"],
                "dynamic_max_options": batch["context"]["dynamic_max_options"],
                "offset_zero_based": offset,
                "physical_identity": list(identity_tuple(identity)),
                "decision_sha256": identity["decision_sha256"],
                "metric": "fixed_initial_repair_floor",
                "positive_option": positive,
                "negative_option": negative,
                "threshold": threshold,
                "threshold_sources": {"same_official_shape_raw": raw_margin},
                "anchor_model": "raw_U468",
                "anchor_source": "same_official_shape_raw_margin",
                "strict_positive_required": bool(physical.get("strict_positive", False)),
                "cw11_start_margin": cw11_margin,
                "construction": "fixed_preregistered_raw_correct_vs_CW11_harmful_pair",
                "created_iteration": 0,
                "origins": ["fixed3_minimum_repair"],
            }
        )
    if len(cuts) != EXPECTED_INITIAL_REPAIR_CONTEXT_GUARDS:
        raise RuntimeError("initial official repair cut cardinality drift")
    return cuts


def merge_active_cuts(
    active: list[dict[str, Any]], additions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    existing = {cut_identity(value): value for value in active}
    added = []
    strengthened = []
    for value in additions:
        candidate = copy.deepcopy(dict(value))
        key = cut_identity(candidate)
        if key not in existing:
            active.append(candidate)
            existing[key] = candidate
            added.append(copy.deepcopy(candidate))
            continue
        record = existing[key]
        old = float(record["threshold"])
        observed = float(candidate["threshold"])
        if observed > old:
            record["threshold"] = observed
            record["threshold_sources"] = dict(candidate["threshold_sources"])
            record["anchor_model"] = candidate["anchor_model"]
            record["anchor_source"] = candidate["anchor_source"]
            strengthened.append(
                {
                    "identity": list(key),
                    "old_threshold": old,
                    "new_threshold": observed,
                }
            )
        for origin in candidate["origins"]:
            if origin not in record["origins"]:
                record["origins"].append(origin)
    active.sort(key=cut_identity)
    return {"added": added, "strengthened": strengthened, "active_count": len(active)}


def official_context_lookup(
    streams: Mapping[str, Sequence[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    return {
        batch["context"]["context_hash"]: batch
        for view in VIEW_ORDER
        for batch in streams[view]
    }


def active_cut_gates(
    active: Sequence[Mapping[str, Any]],
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    lookup = official_context_lookup(streams)
    records = []
    for cut in active:
        if cut["context_type"] == "legacy_B33":
            snapshot = legacy["candidate_snapshot"]
        else:
            snapshot = lookup[str(cut["context_hash"])]["candidate_snapshot"]
        margin = pair_margin(
            snapshot,
            int(cut["offset_zero_based"]),
            int(cut["positive_option"]),
            int(cut["negative_option"]),
        )
        threshold = float(cut["threshold"])
        records.append(
            {
                "identity": list(cut_identity(cut)),
                "context_type": cut["context_type"],
                "margin": margin,
                "threshold": threshold,
                "residual": margin - threshold,
                "pass": margin + tolerance >= threshold,
            }
        )
    return {
        "count": len(records),
        "violated_count": sum(int(not value["pass"]) for value in records),
        "residual_min": min(value["residual"] for value in records),
        "pass": all(value["pass"] for value in records),
        "records": records,
    }


def official_context_gradients(
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: Mapping[str, Any],
    active: Sequence[Mapping[str, Any]],
    parameters: Sequence[Any],
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    helper = context["helper"]
    model = context["model"]
    device = next(model.parameters()).device
    official_lookup = official_context_lookup(streams)
    ordered = sorted(active, key=cut_identity)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cut in ordered:
        groups[str(cut["context_hash"])].append(cut)
    gradients_by_key = {}
    observed_by_key = {}
    context_records = []
    for context_hash in sorted(groups):
        cuts = groups[context_hash]
        if cuts[0]["context_type"] == "legacy_B33":
            cpu_batch = legacy["cpu_batch"]
            gpu_batch = legacy["gpu_batch"]
            expected = legacy["candidate_snapshot"]
        else:
            batch = official_lookup[context_hash]
            cpu_batch = batch["cpu_batch"]
            gpu_batch = {
                key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()
            }
            expected = batch["candidate_snapshot"]
        outputs = helper.ppo.model_forward(model, dict(gpu_batch), device)
        if any(value.dtype != helper.torch.bfloat16 for value in outputs.values()):
            raise RuntimeError("active official-context graph is not native BF16")
        if any(
            not helper.torch.equal(outputs[key].detach().cpu(), expected["outputs_cpu"][key])
            for key in ("policy_logits", "count_logits", "value_logits")
        ):
            raise RuntimeError("active official-context graph/no-grad fingerprint mismatch")
        logits = outputs["policy_logits"].float()
        for index, cut in enumerate(cuts):
            margin = (
                logits[int(cut["offset_zero_based"]), int(cut["positive_option"])]
                - logits[int(cut["offset_zero_based"]), int(cut["negative_option"])]
            )
            flat, _ = modules["geometry"].gradient_for_margin(
                margin,
                parameters,
                helper.torch,
                retain_graph=index + 1 < len(cuts),
            )
            key = cut_identity(cut)
            gradients_by_key[key] = flat
            observed_by_key[key] = float(margin.detach().cpu())
        context_records.append(
            {
                "context_hash": context_hash,
                "context_type": cuts[0]["context_type"],
                "cut_count": len(cuts),
                "actual_batch_size": int(cpu_batch["targets"].shape[0]),
            }
        )
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("official-context autograd materialized .grad")
    matrix = np.stack([gradients_by_key[cut_identity(value)] for value in ordered], axis=0)
    observed = np.asarray(
        [observed_by_key[cut_identity(value)] for value in ordered], dtype=np.float64
    )
    thresholds = np.asarray([float(value["threshold"]) for value in ordered], dtype=np.float64)
    rhs = thresholds - observed
    return matrix, rhs, {
        "active_cut_count": len(ordered),
        "official_context_count": len(groups),
        "gradient_shape": list(matrix.shape),
        "current_margin_min": float(observed.min()),
        "rhs_min": float(rhs.min()),
        "rhs_max": float(rhs.max()),
        "violated_linear_constraints": int((rhs > 0.0).sum()),
        "contexts": context_records,
        "ordered_cut_identity_sha256": canonical_sha(
            [list(cut_identity(value)) for value in ordered]
        ),
    }


def apply_additional_from_cw11(
    modules: Mapping[str, ModuleType],
    parameters: Sequence[Any],
    raw_actor: Sequence[Any],
    cw11_total: Any,
    additional: Any,
    torch: Any,
) -> Any:
    import numpy as np

    total = np.add(cw11_total, additional, dtype=np.float64)
    modules["cutting"].restore_raw_actor(
        modules["ram"], parameters, raw_actor, torch
    )
    modules["cutting"].apply_cumulative_from_raw(
        modules["ram"], parameters, raw_actor, total, torch
    )
    return total


def actor6_layout(parameters: Sequence[Any]) -> list[dict[str, Any]]:
    layout = []
    offset = 0
    for name, parameter in zip(ACTOR6_NAMES, parameters, strict=True):
        count = int(parameter.numel())
        layout.append(
            {
                "name": name,
                "shape": [int(value) for value in parameter.shape],
                "numel": count,
                "start": offset,
                "stop": offset + count,
            }
        )
        offset += count
    if (
        offset != ACTOR6_FLAT_LENGTH
        or canonical_sha(layout) != ACTOR6_LAYOUT_SHA256
        or any(str(parameter.dtype) != "torch.float32" for parameter in parameters)
    ):
        raise RuntimeError("actor6 canonical layout/dtype drift")
    return layout


def xz_base64_payload(raw: bytes, payload_dtype: str) -> dict[str, Any]:
    compressed = lzma.compress(
        raw,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        preset=9 | lzma.PRESET_EXTREME,
    )
    encoded = base64.standard_b64encode(compressed).decode("ascii")
    return {
        "payload_dtype": payload_dtype,
        "raw_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "compression": "XZ_preset9_extreme_CRC64",
        "compressed_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "base64_variant": "standard_RFC4648",
        "base64_chunks_76": [
            encoded[index : index + 76] for index in range(0, len(encoded), 76)
        ],
    }


def actor_float32_le_bytes(parameters: Sequence[Any], np: Any) -> bytes:
    parts = []
    for parameter in parameters:
        array = parameter.detach().float().cpu().contiguous().numpy()
        parts.append(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes())
    value = b"".join(parts)
    if len(value) != ACTOR6_FLAT_LENGTH * 4:
        raise RuntimeError("actor6 float32 byte length drift")
    return value


def encode_terminal_payload(
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    parameters: Sequence[Any],
    cw11_actor_bytes: bytes,
    additional: Any,
    total: Any,
    terminal_model_sha256: str,
) -> dict[str, Any]:
    import numpy as np

    additional_raw = np.ascontiguousarray(additional.astype("<f8")).tobytes()
    total_raw = np.ascontiguousarray(total.astype("<f8")).tobytes()
    if len(additional_raw) != ACTOR6_FLOAT64_BYTES or len(total_raw) != ACTOR6_FLOAT64_BYTES:
        raise RuntimeError("terminal float64 vector byte length drift")
    final_actor_bytes = actor_float32_le_bytes(parameters, np)
    xor_raw = bytes(left ^ right for left, right in zip(cw11_actor_bytes, final_actor_bytes))
    layout = actor6_layout(parameters)
    return {
        "reconstruction_formula": (
            "total_f64=np.add(CW11_total_f64,additional_f64,dtype=float64); "
            "frozen restore_raw_actor; frozen apply_cumulative_from_raw(raw_actor,total_f64)"
        ),
        "anchor": {
            "raw_model_state_sha256": RAW_MODEL_SHA256,
            "cw11_model_state_sha256": CW11_MODEL_SHA256,
            "cw11_total_float64_le_sha256": CW11_VECTOR_SHA256,
            "terminal_model_state_sha256": terminal_model_sha256,
        },
        "actor_layout": layout,
        "actor_layout_sha256": ACTOR6_LAYOUT_SHA256,
        "actor_parameter_dtype": "torch.float32",
        "flat_length": ACTOR6_FLAT_LENGTH,
        "additional_from_CW11_l2": float(np.linalg.norm(additional)),
        "terminal_total_from_raw_l2": float(np.linalg.norm(total)),
        "additional_from_CW11_float64_le": xz_base64_payload(
            additional_raw, "<f8"
        ),
        "terminal_total_from_raw_float64_le": xz_base64_payload(total_raw, "<f8"),
        "cw11_actor_float32_le_sha256": hashlib.sha256(cw11_actor_bytes).hexdigest(),
        "terminal_actor_float32_le_sha256": hashlib.sha256(final_actor_bytes).hexdigest(),
        "cw11_to_terminal_actor_float32_xor_backup": xz_base64_payload(
            xor_raw, "uint8_xor"
        ),
        "xor_backup_policy": "audit_only_never_used_to_correct_mathematical_reconstruction",
        "application_dependency_shas": {
            "cutting": MODULE_SHAS[CUTTING],
            "ram": MODULE_SHAS[RAM],
            "geometry": MODULE_SHAS[GEOMETRY],
        },
        "raw_checkpoint_template_is_original_U468": context["checkpoint"] is not None,
    }


def decode_xz_base64_payload(value: Mapping[str, Any]) -> bytes:
    encoded = "".join(str(chunk) for chunk in value["base64_chunks_76"])
    compressed = base64.b64decode(encoded, altchars=None, validate=True)
    if hashlib.sha256(compressed).hexdigest() != str(value["compressed_sha256"]):
        raise RuntimeError("compressed terminal payload SHA mismatch")
    raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    if len(raw) != int(value["raw_bytes"]) or hashlib.sha256(raw).hexdigest() != str(
        value["raw_sha256"]
    ):
        raise RuntimeError("decoded terminal payload drift")
    return raw


def reconstruct_terminal_candidate(
    model: Any,
    helper: ModuleType,
    raw_actor: Sequence[Any],
    cw11_total_float64: Any,
    payload: Mapping[str, Any],
    *,
    cutting: ModuleType,
    geometry: ModuleType,
    ram: ModuleType,
) -> dict[str, Any]:
    """Pure reconstruction helper; it never opens specialist data."""
    import numpy as np

    if payload["actor_layout_sha256"] != ACTOR6_LAYOUT_SHA256:
        raise RuntimeError("terminal actor layout binding mismatch")
    anchor_checks = {
        "raw_model_anchor_exact": payload["anchor"]["raw_model_state_sha256"]
        == RAW_MODEL_SHA256,
        "cw11_model_anchor_exact": payload["anchor"]["cw11_model_state_sha256"]
        == CW11_MODEL_SHA256,
        "cw11_vector_anchor_exact": payload["anchor"]
        ["cw11_total_float64_le_sha256"]
        == CW11_VECTOR_SHA256,
    }
    if not all(anchor_checks.values()):
        raise RuntimeError(f"terminal payload anchor drift: {anchor_checks}")
    parameters = geometry.configure_actor6(model)
    if actor6_layout(parameters) != payload["actor_layout"]:
        raise RuntimeError("terminal actor layout values mismatch")
    additional_raw = decode_xz_base64_payload(
        payload["additional_from_CW11_float64_le"]
    )
    total_raw = decode_xz_base64_payload(payload["terminal_total_from_raw_float64_le"])
    additional = np.frombuffer(additional_raw, dtype="<f8").astype(np.float64, copy=True)
    frozen_cw11 = np.asarray(cw11_total_float64, dtype=np.float64)
    if additional.shape != (ACTOR6_FLAT_LENGTH,) or frozen_cw11.shape != (
        ACTOR6_FLAT_LENGTH,
    ):
        raise RuntimeError("pure reconstruction vector shape drift")
    if hashlib.sha256(np.ascontiguousarray(frozen_cw11.astype("<f8")).tobytes()).hexdigest() != CW11_VECTOR_SHA256:
        raise RuntimeError("pure reconstruction CW11 vector anchor drift")
    total = np.add(frozen_cw11, additional, dtype=np.float64)
    if np.ascontiguousarray(total.astype("<f8")).tobytes() != total_raw:
        raise RuntimeError("additional plus CW11 vector does not reproduce terminal total")
    if total.shape != (ACTOR6_FLAT_LENGTH,):
        raise RuntimeError("terminal total vector shape drift")
    additional_l2 = float(np.linalg.norm(additional))
    total_l2 = float(np.linalg.norm(total))
    vector_checks = {
        "additional_plus_CW11_exact_total_bytes": np.ascontiguousarray(
            total.astype("<f8")
        ).tobytes()
        == total_raw,
        "additional_l2_exact": math.isclose(
            additional_l2,
            float(payload["additional_from_CW11_l2"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "total_l2_exact": math.isclose(
            total_l2,
            float(payload["terminal_total_from_raw_l2"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "additional_l2_cap": additional_l2
        <= ADDITIONAL_TOTAL_L2_CAP + L2_CAP_ABS_TOL,
    }
    if not all(vector_checks.values()):
        raise RuntimeError(f"pure reconstruction vector audit failed: {vector_checks}")

    cutting.restore_raw_actor(ram, parameters, raw_actor, helper.torch)
    observed_raw_model_sha = helper.model_state_sha256(model.state_dict())
    cutting.apply_cumulative_from_raw(
        ram, parameters, raw_actor, frozen_cw11, helper.torch
    )
    observed_cw11_model_sha = helper.model_state_sha256(model.state_dict())
    reconstructed_cw11_actor_bytes = actor_float32_le_bytes(parameters, np)
    observed_cw11_actor_sha = hashlib.sha256(
        reconstructed_cw11_actor_bytes
    ).hexdigest()

    cutting.restore_raw_actor(ram, parameters, raw_actor, helper.torch)
    cutting.apply_cumulative_from_raw(ram, parameters, raw_actor, total, helper.torch)
    observed_model_sha = helper.model_state_sha256(model.state_dict())
    final_actor_bytes = actor_float32_le_bytes(parameters, np)
    observed_actor_sha = hashlib.sha256(final_actor_bytes).hexdigest()
    xor_raw = decode_xz_base64_payload(
        payload["cw11_to_terminal_actor_float32_xor_backup"]
    )
    if len(xor_raw) != len(reconstructed_cw11_actor_bytes):
        raise RuntimeError("terminal actor XOR byte length drift")
    xor_reconstructed_final = bytes(
        left ^ right
        for left, right in zip(reconstructed_cw11_actor_bytes, xor_raw, strict=True)
    )
    checks = {
        **anchor_checks,
        **vector_checks,
        "raw_model_sha_exact": observed_raw_model_sha == RAW_MODEL_SHA256,
        "cw11_model_sha_exact": observed_cw11_model_sha == CW11_MODEL_SHA256,
        "cw11_actor_bytes_sha_exact": observed_cw11_actor_sha
        == payload["cw11_actor_float32_le_sha256"],
        "model_sha_exact": observed_model_sha
        == payload["anchor"]["terminal_model_state_sha256"],
        "actor_bytes_sha_exact": observed_actor_sha
        == payload["terminal_actor_float32_le_sha256"],
        "actor_XOR_backup_exact": xor_reconstructed_final == final_actor_bytes,
        "actor_XOR_audit_only": payload["xor_backup_policy"]
        == "audit_only_never_used_to_correct_mathematical_reconstruction",
        "no_specialist_data_opened": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"pure terminal reconstruction failed: {checks}")
    return {
        "checks": checks,
        "model_state_sha256": observed_model_sha,
        "additional_from_CW11_l2": additional_l2,
        "terminal_total_from_raw_l2": total_l2,
        "additional_float64_le_sha256": hashlib.sha256(additional_raw).hexdigest(),
        "total_float64_le_sha256": hashlib.sha256(total_raw).hexdigest(),
        "terminal_actor_float32_le_sha256": observed_actor_sha,
    }


def snapshot_legacy_B33(
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    legacy: dict[str, Any],
    phase: str,
) -> Mapping[str, Any]:
    device = next(context["model"].parameters()).device
    snapshot = modules["ram"].snapshot_forward(
        context["helper"],
        context["model"],
        legacy["gpu_batch"],
        legacy["cpu_batch"],
        device,
    )
    legacy[f"{phase}_snapshot"] = snapshot
    return snapshot


def legacy_B33_gate(
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    legacy: Mapping[str, Any],
) -> dict[str, Any]:
    model = context["model"]
    helper = context["helper"]
    nonactor_names = sorted(set(model.state_dict()) - set(ACTOR6_NAMES))
    nonactor_sha = helper.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )
    patch_audit: dict[str, Any] = {
        "gate_call_count": 0,
        "global_restored_every_call": True,
    }
    gate = modules["cw11"].gate33(
        modules["cutting"],
        legacy["descriptors"],
        legacy["pairs"],
        legacy["raw_snapshot"],
        legacy["candidate_snapshot"],
        nonactor_sha == RAW_NONACTOR_SHA256,
        helper.torch,
        patch_audit,
    )
    return {
        "gate": gate,
        "patch_audit": patch_audit,
        "pass": bool(gate["pass"])
        and bool(patch_audit["global_restored_every_call"]),
    }


def legacy_dynamic_cuts(
    context: Mapping[str, Any],
    legacy: Mapping[str, Any],
    false_obligations: Sequence[Mapping[str, Any]],
    iteration: int,
) -> list[dict[str, Any]]:
    torch = context["helper"].torch
    reference = legacy["reference_snapshot"]
    candidate = legacy["candidate_snapshot"]
    cuts = []
    metric_map = {
        "set_exact": "set",
        "hybrid_order_exact": "hybrid",
        "ordered_exact": "ordered",
        "top1_correct": "top1",
    }
    for obligation in false_obligations:
        offset = int(obligation["row_index"])
        metric_original = str(obligation["metric"])
        metric = metric_map.get(metric_original, metric_original.replace("_exact", ""))
        reference_row = reference["official_rows"][offset]
        candidate_row = candidate["official_rows"][offset]
        expert = [int(value) for value in reference_row["expert_order"]]
        observed = [int(value) for value in candidate_row["predicted_order"]]
        hybrid_observed = [int(value) for value in candidate_row["hybrid_order"]]
        if metric == "top1":
            pair = (
                int(reference_row["top1_index_full_policy_vector"]),
                int(candidate_row["top1_index_full_policy_vector"]),
            )
            construction = "legacy_reference_top1_vs_candidate"
        elif metric == "set":
            pair = set_missing_pair(expert, observed)
            construction = "legacy_reference_missing_vs_candidate_included"
        elif metric == "hybrid" and int(reference_row["context"]) != 34:
            pair = set_missing_pair(expert, hybrid_observed)
            if pair is None:
                pair = first_different_pair(expert, hybrid_observed)
                construction = "legacy_reference_hybrid_first_order_difference"
            else:
                construction = "legacy_reference_hybrid_missing_vs_candidate_included"
        else:
            pair = first_different_pair(
                expert, hybrid_observed if metric == "hybrid" else observed
            )
            construction = "legacy_reference_first_order_difference"
        if pair is None or pair[0] == pair[1]:
            raise RuntimeError("legacy compatibility failure has no separating pair")
        positive, negative = pair
        reference_floor = pair_margin(reference, offset, positive, negative)
        tie_floor = positive_bf16_q(reference, offset, positive, negative, torch)
        threshold = max(reference_floor, tie_floor)
        current_margin = pair_margin(candidate, offset, positive, negative)
        if current_margin >= threshold:
            raise RuntimeError("legacy compatibility cut does not separate")
        descriptor = legacy["descriptors"][offset]
        cuts.append(
            {
                "context_type": "legacy_B33",
                "context_hash": legacy["context_hash"],
                "view": "legacy_B33",
                "batch_ordinal_one_based": 1,
                "actual_batch_size": 33,
                "dynamic_max_options": int(legacy["cpu_batch"]["option_mask"].shape[1]),
                "offset_zero_based": offset,
                "physical_identity": [
                    str(descriptor["panel"]),
                    str(descriptor["member"]),
                    int(descriptor["line_index_zero_based"]),
                    str(descriptor["line_sha256"]),
                ],
                "decision_sha256": f"legacy-line:{descriptor['line_sha256']}",
                "metric": f"legacy_dynamic_{metric_original}",
                "positive_option": positive,
                "negative_option": negative,
                "threshold": threshold,
                "threshold_sources": {
                    "frozen_CW11_B33_same_shape": reference_floor,
                    "strict_positive_native_bf16_q": tie_floor,
                },
                "anchor_model": "frozen_CW11_4317",
                "anchor_source": "same_B33_shape_CW11_margin",
                "current_margin": current_margin,
                "construction": construction,
                "created_iteration": iteration,
                "origins": ["legacy_B33_compatibility_new_harm"],
            }
        )
    cuts.sort(key=cut_identity)
    return cuts


def nonactor_sha256(context: Mapping[str, Any]) -> str:
    model = context["model"]
    names = sorted(set(model.state_dict()) - set(ACTOR6_NAMES))
    return context["helper"].model_state_sha256(
        {name: model.state_dict()[name] for name in names}
    )


def raw_metric_reproduction(
    raw_result: Mapping[str, Any], forensic: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        view: all(
            int(raw_result["panels"][view]["metrics"][metric])
            == int(forensic["panels"][view]["metrics"]["raw"][metric])
            for metric in ("rows", "set", "hybrid", "ordered", "top1", "count", "value")
        )
        for view in VIEW_ORDER
    }


def terminal_acceptance(
    *,
    snapshot: Mapping[str, Any],
    oracle: Mapping[str, Any],
    fixed: Mapping[str, Any],
    legacy_gate: Mapping[str, Any],
    active_gates: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "six_official_views_complete": all(snapshot["hard_checks"].values()),
        "original_60_of_60_gates": oracle["authoritative_60_gates"]["pass"]
        and oracle["authoritative_60_gates"]["passed_gate_count"] == 60,
        "fixed3_four_context_guards_pass": bool(fixed["pass"]),
        "legacy_old34_B33_pass": bool(legacy_gate["pass"]),
        "new_harm_count_zero": int(oracle["new_harm_count_vs_cw11"]) == 0,
        "forensic_favorable_retention_pass": bool(
            oracle["transition_state_checks"]["favorable_transitions_retained"]
        ),
        "active_cut_violations_zero": bool(active_gates["pass"]),
        "monitor47_complete": bool(
            oracle["hard_checks"]["monitor47_complete_all_view_occurrences"]
        ),
        "transition22_complete": bool(
            oracle["hard_checks"]["transition22_occurrences_complete"]
        ),
        "actor6_nonactor_count_value_L2_integrity": all(
            bool(value) for value in integrity.values()
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def run_outer_cutting_plane(
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    *,
    user_candidate_consumer: Any = None,
) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    helper = context["helper"]
    torch = helper.torch
    model = context["model"]
    device = next(model.parameters()).device
    forensic = read_json_locked(FORENSIC, FORENSIC_SHA256, 0o444)
    guardplan = read_json_locked(GUARDPLAN, GUARDPLAN_SHA256, 0o444)
    prereg = read_json_locked(PREREG, PREREG_SHA256, 0o444)
    required_context = {
        "model",
        "checkpoint",
        "helper",
        "model_config",
        "raw_actor",
        "raw_model_state_sha256",
        "raw_nonactor_sha256",
        "terminal_cumulative_float64",
        "terminal_cumulative_float64_le_sha256",
        "selected_row_gate",
        "active_pair_ledger",
        "expanded_row_count",
        "candidate_model_state_sha256",
    }
    if not required_context.issubset(context):
        raise RuntimeError("CW13 incoming CW11 consumer schema drift")
    incoming_checks = {
        "model_sha_exact": helper.model_state_sha256(model.state_dict())
        == CW11_MODEL_SHA256
        == str(context["candidate_model_state_sha256"]),
        "model_eval_exact": model.training is False,
        "raw_sha_exact": str(context["raw_model_state_sha256"])
        == RAW_MODEL_SHA256,
        "nonactor_sha_exact": str(context["raw_nonactor_sha256"])
        == RAW_NONACTOR_SHA256
        and nonactor_sha256(context) == RAW_NONACTOR_SHA256,
        "cw11_vector_sha_exact": str(context["terminal_cumulative_float64_le_sha256"])
        == CW11_VECTOR_SHA256,
        "cw11_vector_l2_exact": math.isclose(
            float(np.linalg.norm(context["terminal_cumulative_float64"])),
            CW11_TOTAL_FROM_RAW_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "legacy_33_34_exact": int(context["expanded_row_count"])
        == EXPECTED_LEGACY_ROWS
        and len(context["active_pair_ledger"]) == EXPECTED_LEGACY_ACTIVE_PAIRS,
        "legacy_gate_pass": bool(context["selected_row_gate"]["pass"]),
    }
    if not all(incoming_checks.values()):
        raise RuntimeError(f"incoming CW11 terminal binding failed: {incoming_checks}")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    parameters = modules["geometry"].configure_actor6(model)
    layout = actor6_layout(parameters)
    raw_actor = context["raw_actor"]
    if len(raw_actor) != len(parameters):
        raise RuntimeError("raw actor tensor cardinality drift")
    cw11_total = np.asarray(
        context["terminal_cumulative_float64"], dtype=np.float64
    ).copy()
    if cw11_total.shape != (ACTOR6_FLAT_LENGTH,):
        raise RuntimeError("CW11 total vector shape drift")

    # Bind the materialized checkpoint to the exact live reconstruction.
    materialized_model, materialized_config, _, materialized_kind = helper.evaluator.load_policy(
        CW11_CHECKPOINT, device
    )
    materialized_checks = {
        "kind_ppo": materialized_kind == "ppo",
        "model_eval_exact": materialized_model.training is False,
        "config_exact": materialized_config == context["model_config"],
        "model_sha_exact": helper.model_state_sha256(materialized_model.state_dict())
        == CW11_MODEL_SHA256,
        "state_tensors_exact_live": set(materialized_model.state_dict())
        == set(model.state_dict())
        and all(
            torch.equal(materialized_model.state_dict()[name], model.state_dict()[name])
            for name in model.state_dict()
        ),
    }
    del materialized_model
    torch.cuda.empty_cache()
    if not all(materialized_checks.values()):
        raise RuntimeError(f"materialized/live CW11 mismatch: {materialized_checks}")

    legacy = load_legacy_B33(context, modules)
    streams, stream_loading_audit = load_official_six_streams(
        helper, modules["legacy"], context["model_config"], forensic, guardplan
    )

    # Freeze exact raw and CW11 reference snapshots before any proposal.
    modules["cutting"].restore_raw_actor(
        modules["ram"], parameters, raw_actor, torch
    )
    if helper.model_state_sha256(model.state_dict()) != RAW_MODEL_SHA256:
        raise RuntimeError("temporary raw restoration failed")
    snapshot_legacy_B33(context, modules, legacy, "raw")
    raw_six = snapshot_official_six_views(
        helper,
        modules["ram"],
        model,
        streams,
        device,
        phase="raw",
        model_sha256=RAW_MODEL_SHA256,
    )
    raw_reproduction = raw_metric_reproduction(raw_six, forensic)
    if not all(raw_reproduction.values()):
        raise RuntimeError(f"raw official clone reproduction failed: {raw_reproduction}")

    modules["cutting"].apply_cumulative_from_raw(
        modules["ram"], parameters, raw_actor, cw11_total, torch
    )
    if helper.model_state_sha256(model.state_dict()) != CW11_MODEL_SHA256:
        raise RuntimeError("CW11 reference reapplication failed")
    cw11_actor_bytes = actor_float32_le_bytes(parameters, np)
    snapshot_legacy_B33(context, modules, legacy, "reference")
    legacy["candidate_snapshot"] = legacy["reference_snapshot"]
    reference_six = snapshot_official_six_views(
        helper,
        modules["ram"],
        model,
        streams,
        device,
        phase="reference",
        model_sha256=CW11_MODEL_SHA256,
    )
    for view in VIEW_ORDER:
        for batch in streams[view]:
            batch["candidate_snapshot"] = batch["reference_snapshot"]
    reference_legacy_gate = legacy_B33_gate(context, modules, legacy)
    if not reference_legacy_gate["pass"]:
        raise RuntimeError("CW11 reference no longer passes legacy B33")
    initial_oracle = full_stream_separation_oracle(
        helper, streams, forensic, guardplan, prereg, iteration=0
    )
    initial_fixed = fixed_repair_gates_and_cuts(helper, streams, iteration=0)
    initial_checks = {
        "cw11_failed_exact_51_of_60": initial_oracle["authoritative_60_gates"]
        ["passed_gate_count"]
        == 51,
        "cw11_has_no_self_new_harm": initial_oracle["new_harm_count_vs_cw11"] == 0,
        "cw11_favorable_transitions_retained": initial_oracle[
            "transition_state_checks"
        ]["favorable_transitions_retained"],
        "cw11_has_no_restoration_cuts": initial_oracle[
            "favorable_transition_break_count"
        ]
        == 0
        and not initial_oracle["deterministic_restoration_cuts"],
        "fixed_repairs_initially_fail": not initial_fixed["pass"],
        "iteration0_has_no_dynamic_fixed_cuts": initial_fixed[
            "deterministic_repair_cut_count"
        ]
        == 0,
        "legacy_reference_pass": reference_legacy_gate["pass"],
    }
    if not all(initial_checks.values()):
        raise RuntimeError(f"CW13 initial-state audit failed: {initial_checks}")

    active: list[dict[str, Any]] = []
    legacy_merge = merge_active_cuts(active, legacy_pair_cuts(legacy))
    fixed_merge = merge_active_cuts(active, initial_official_repair_cuts(streams))
    if (
        legacy_merge["active_count"] != 34
        or fixed_merge["active_count"] != 38
        or len({cut_identity(value) for value in active}) != 38
    ):
        raise RuntimeError("CW13 initial 34+4 active cut ledger drift")
    initial_active_gates = active_cut_gates(
        active, streams, legacy, modules["cutting"].PAIR_THRESHOLD_TOLERANCE
    )
    if initial_active_gates["violated_count"] != 4:
        raise RuntimeError("CW13 initial fixed repair violation count drift")

    additional = np.zeros(ACTOR6_FLAT_LENGTH, dtype=np.float64)
    current_total = cw11_total.copy()
    iterations = [
        {
            "iteration": 0,
            "kind": "exact_CW11_reference_before_fixed3_repair",
            "model_state_sha256": CW11_MODEL_SHA256,
            "additional_l2": 0.0,
            "total_from_raw_l2": float(np.linalg.norm(cw11_total)),
            "active_cut_count": len(active),
            "active_cut_identity_sha256": canonical_sha(
                [list(cut_identity(value)) for value in active]
            ),
            "legacy_gate": reference_legacy_gate,
            "six_view_snapshot": reference_six,
            "oracle": initial_oracle,
            "fixed_repair_gates": initial_fixed,
            "active_cut_gates": initial_active_gates,
        }
    ]
    terminal_iteration = None
    close_reason = "maximum_12_iterations_without_first_feasible"
    terminal_payload = None
    terminal_oracle = None
    terminal_active_gates = None
    terminal_legacy_gate = None
    terminal_reconstruction_audit = None
    for iteration in range(1, MAX_OUTER_ITERATIONS + 1):
        try:
            gradients, rhs, gradient_audit = official_context_gradients(
                context, modules, streams, legacy, active, parameters
            )
            correction, qp = modules["cutting"].solve_minimum_l2_correction(
                gradients, rhs, np, optimize
            )
        except Exception as exc:
            close_reason = f"fail_closed_gradient_or_QP:{type(exc).__name__}:{exc}"
            iterations.append(
                {
                    "iteration": iteration,
                    "kind": "fail_closed_before_proposal",
                    "close_reason": close_reason,
                    "active_cut_count": len(active),
                }
            )
            break
        step_l2 = float(np.linalg.norm(correction))
        proposal_additional = np.add(additional, correction, dtype=np.float64)
        proposal_additional_l2 = float(np.linalg.norm(proposal_additional))
        cap_checks = {
            "step_positive_finite": math.isfinite(step_l2) and step_l2 > 0.0,
            "step_at_most_cap": step_l2 <= STEP_L2_CAP + L2_CAP_ABS_TOL,
            "additional_total_finite": math.isfinite(proposal_additional_l2),
            "additional_total_at_most_cap": proposal_additional_l2
            <= ADDITIONAL_TOTAL_L2_CAP + L2_CAP_ABS_TOL,
        }
        if not all(cap_checks.values()):
            close_reason = "fail_closed_L2_cap_gate"
            iterations.append(
                {
                    "iteration": iteration,
                    "kind": "fail_closed_before_proposal",
                    "close_reason": close_reason,
                    "cap_checks": cap_checks,
                    "qp": qp,
                    "step_l2": step_l2,
                    "proposal_additional_l2": proposal_additional_l2,
                }
            )
            break
        proposal_total = apply_additional_from_cw11(
            modules,
            parameters,
            raw_actor,
            cw11_total,
            proposal_additional,
            torch,
        )
        proposal_model_sha = helper.model_state_sha256(model.state_dict())
        snapshot_legacy_B33(context, modules, legacy, "candidate")
        legacy_gate = legacy_B33_gate(context, modules, legacy)
        candidate_six = snapshot_official_six_views(
            helper,
            modules["ram"],
            model,
            streams,
            device,
            phase="candidate",
            model_sha256=proposal_model_sha,
        )
        oracle = full_stream_separation_oracle(
            helper, streams, forensic, guardplan, prereg, iteration=iteration
        )
        fixed = fixed_repair_gates_and_cuts(helper, streams, iteration=iteration)
        active_gates = active_cut_gates(
            active, streams, legacy, modules["cutting"].PAIR_THRESHOLD_TOLERANCE
        )
        integrity = {
            "actor6_trainable_scope_exact": tuple(
                name for name, value in model.named_parameters() if value.requires_grad
            )
            == ACTOR6_NAMES,
            "actor6_layout_exact": actor6_layout(parameters) == layout,
            "nonactor_74_exact_raw": nonactor_sha256(context) == RAW_NONACTOR_SHA256,
            "count_value_logits_exact_raw": bool(
                oracle["hard_checks"]["count_value_logits_exact_raw_all_batches"]
            ),
            "native_BF16_six_outputs": True,
            "step_L2_cap": cap_checks["step_at_most_cap"],
            "additional_total_L2_cap": cap_checks["additional_total_at_most_cap"],
            "model_sha_hex64": len(proposal_model_sha) == 64,
            "no_checkpoint_written_eval_only_13key_deferred": True,
        }
        acceptance = terminal_acceptance(
            snapshot=candidate_six,
            oracle=oracle,
            fixed=fixed,
            legacy_gate=legacy_gate,
            active_gates=active_gates,
            integrity=integrity,
        )
        legacy_additions = legacy_dynamic_cuts(
            context,
            legacy,
            legacy_gate["gate"]["false_obligations"],
            iteration,
        )
        additions = sorted(
            [
                *oracle["deterministic_new_harm_cuts"],
                *oracle["deterministic_restoration_cuts"],
                *fixed["deterministic_repair_cuts"],
                *legacy_additions,
            ],
            key=cut_identity,
        )
        iteration_record = {
            "iteration": iteration,
            "kind": "unique_fixed_order_official_context_proposal",
            "model_state_sha256": proposal_model_sha,
            "gradient_audit": gradient_audit,
            "qp": qp,
            "step_l2": step_l2,
            "additional_l2": proposal_additional_l2,
            "additional_float64_le_sha256": modules["geometry"].vector_sha256_float64_le(
                proposal_additional, np
            ),
            "total_from_raw_l2": float(np.linalg.norm(proposal_total)),
            "total_from_raw_float64_le_sha256": modules["geometry"].vector_sha256_float64_le(
                proposal_total, np
            ),
            "cap_checks": cap_checks,
            "qp_cap_audit": {
                "capped_record_only": bool(qp["capped"]),
                "trust_region_clipping_is_legal": True,
                "absolute_tolerance": L2_CAP_ABS_TOL,
            },
            "legacy_gate": legacy_gate,
            "six_view_snapshot": candidate_six,
            "oracle": oracle,
            "fixed_repair_gates": fixed,
            "active_cut_gates": active_gates,
            "integrity": integrity,
            "acceptance": acceptance,
            "deterministic_addition_count": len(additions),
            "deterministic_addition_identity_sha256": canonical_sha(
                [list(cut_identity(value)) for value in additions]
            ),
            "favorable_transition_restoration": {
                "break_count": oracle["favorable_transition_break_count"],
                "cut_count": len(oracle["deterministic_restoration_cuts"]),
                "retained": oracle["transition_state_checks"]
                ["favorable_transitions_retained"],
                "will_continue_if_nonterminal": bool(
                    not acceptance["pass"]
                    and oracle["favorable_transition_break_count"] > 0
                    and oracle["deterministic_restoration_cuts"]
                ),
            },
        }
        iterations.append(iteration_record)
        additional = proposal_additional
        current_total = proposal_total
        if acceptance["pass"]:
            if additions:
                raise RuntimeError(
                    "accepted terminal unexpectedly produced deterministic cuts"
                )
            terminal_iteration = iteration
            close_reason = "first_feasible_fixed_iteration_candidate"
            terminal_oracle = oracle
            terminal_active_gates = active_gates
            terminal_legacy_gate = legacy_gate
            terminal_payload = encode_terminal_payload(
                context,
                modules,
                parameters,
                cw11_actor_bytes,
                additional,
                current_total,
                proposal_model_sha,
            )
            terminal_reconstruction_audit = reconstruct_terminal_candidate(
                model,
                helper,
                raw_actor,
                cw11_total,
                terminal_payload,
                cutting=modules["cutting"],
                geometry=modules["geometry"],
                ram=modules["ram"],
            )
            if (
                not all(terminal_reconstruction_audit["checks"].values())
                or terminal_reconstruction_audit["model_state_sha256"]
                != proposal_model_sha
            ):
                raise RuntimeError("immediate terminal reconstruction audit failed")
            iteration_record["terminal_reconstruction_audit"] = copy.deepcopy(
                terminal_reconstruction_audit
            )
            break
        merge = merge_active_cuts(active, additions)
        iteration_record["post_oracle_cut_merge"] = merge
        if not additions and active_gates["violated_count"] == 0:
            close_reason = "fail_closed_nonfeasible_without_separating_cut"
            break

    success = terminal_iteration is not None
    terminal_active_cut_ledger = copy.deepcopy(active) if success else None
    terminal_active_cut_ledger_sha256 = (
        canonical_sha(terminal_active_cut_ledger) if success else None
    )
    candidate_consumer_called = False
    if success and user_candidate_consumer is not None:
        user_candidate_consumer(
            {
                "helper": helper,
                "model": model,
                "checkpoint": context["checkpoint"],
                "model_config": dict(context["model_config"]),
                "raw_actor": raw_actor,
                "raw_model_state_sha256": RAW_MODEL_SHA256,
                "raw_nonactor_sha256": RAW_NONACTOR_SHA256,
                "cw11_model_state_sha256": CW11_MODEL_SHA256,
                "cw11_total_float64": cw11_total.copy(),
                "cw11_total_float64_le_sha256": CW11_VECTOR_SHA256,
                "terminal_additional_float64": additional.copy(),
                "terminal_total_float64": current_total.copy(),
                "terminal_payload": copy.deepcopy(terminal_payload),
                "terminal_model_state_sha256": terminal_payload["anchor"]
                ["terminal_model_state_sha256"],
                "terminal_iteration": terminal_iteration,
                "active_cut_ledger": copy.deepcopy(terminal_active_cut_ledger),
                "active_cut_ledger_sha256": terminal_active_cut_ledger_sha256,
                "legacy_B33_gate": copy.deepcopy(terminal_legacy_gate),
                "terminal_consumed_valid_oracle": copy.deepcopy(terminal_oracle),
                "terminal_active_cut_gates": copy.deepcopy(terminal_active_gates),
                "reconstruction_helper": reconstruct_terminal_candidate,
                "consumed_specialist_data_must_not_be_reopened_downstream": True,
            }
        )
        candidate_consumer_called = True
    return {
        "status": (
            "consumed_valid_CW13_optimization_closure_first_feasible"
            if success
            else "closed_no_CW13_candidate"
        ),
        "classification": dict(CLASSIFICATION),
        "incoming_checks": incoming_checks,
        "materialized_live_binding": materialized_checks,
        "actor6_layout": layout,
        "stream_loading_audit": stream_loading_audit,
        "raw_six_view_reproduction": raw_reproduction,
        "initial_checks": initial_checks,
        "initial_active_cut_count": 38,
        "iterations": iterations,
        "decision": {
            "first_feasible_required": True,
            "terminal_iteration": terminal_iteration,
            "close_reason": close_reason,
            "terminal_model_state_sha256": (
                None
                if terminal_payload is None
                else terminal_payload["anchor"]["terminal_model_state_sha256"]
            ),
            "terminal_additional_l2": float(np.linalg.norm(additional)),
            "terminal_total_from_raw_l2": float(np.linalg.norm(current_total)),
            "terminal_active_cut_count": len(active),
            "candidate_consumer_called": candidate_consumer_called,
            "eligible_only_for_formal_fulltrain_revalidation": success,
            "eligible_as_promotion_evidence": False,
        },
        "terminal_reconstruction_payload": terminal_payload,
        "terminal_reconstruction_audit": terminal_reconstruction_audit,
        "terminal_active_cut_ledger": terminal_active_cut_ledger,
        "terminal_active_cut_ledger_sha256": terminal_active_cut_ledger_sha256,
        "downstream_contract": {
            "formal_must_not_call_official6_solver_again": True,
            "formal_reconstructs_from_payload_without_specialist_access": True,
            "formal_uses_original_raw_U468_checkpoint_template": True,
            "formal_then_runs_one_frozen_fulltrain_revalidation_only": True,
            "broad_requires_new_CW13_hash_bound_one_shot_runner": True,
        },
    }


def run_cw13_consumer(
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    *,
    user_candidate_consumer: Any = None,
) -> dict[str, Any]:
    helper = context["helper"]
    model = context["model"]
    parameters = modules["geometry"].configure_actor6(model)
    cw11_total = context["terminal_cumulative_float64"].copy()
    previous_matmul_precision = helper.torch.get_float32_matmul_precision()
    result = None
    restored = False
    precision_restored = False
    try:
        helper.torch.set_float32_matmul_precision("high")
        if helper.torch.get_float32_matmul_precision() != "high":
            raise RuntimeError("failed to enter frozen official matmul precision")
        result = run_outer_cutting_plane(
            context,
            modules,
            user_candidate_consumer=user_candidate_consumer,
        )
        return result
    finally:
        model_restore_error = None
        precision_restore_error = None
        try:
            modules["cutting"].restore_raw_actor(
                modules["ram"], parameters, context["raw_actor"], helper.torch
            )
            modules["cutting"].apply_cumulative_from_raw(
                modules["ram"],
                parameters,
                context["raw_actor"],
                cw11_total,
                helper.torch,
            )
            restored = (
                helper.model_state_sha256(model.state_dict()) == CW11_MODEL_SHA256
            )
        except Exception as exc:
            model_restore_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                helper.torch.set_float32_matmul_precision(
                    previous_matmul_precision
                )
                precision_restored = (
                    helper.torch.get_float32_matmul_precision()
                    == previous_matmul_precision
                )
            except Exception as exc:
                precision_restore_error = f"{type(exc).__name__}: {exc}"
        if result is not None:
            result["local_finally"] = {
                "restored_exact_CW11_before_outer_finally": restored,
                "outer_CW11_chain_will_restore_raw": True,
                "official_float32_matmul_precision_during_outer": "high",
                "previous_float32_matmul_precision": previous_matmul_precision,
                "matmul_precision_restored_exact": precision_restored,
                "model_restore_error": model_restore_error,
                "precision_restore_error": precision_restore_error,
            }
        if (
            not restored
            or not precision_restored
            or model_restore_error is not None
            or precision_restore_error is not None
        ):
            raise RuntimeError(
                "CW13 local finally failed to restore CW11 or matmul precision"
            )


def run_probe(
    primary: ModuleType,
    primary_source: bytes,
    primary_evidence: Mapping[str, Any],
    *,
    candidate_consumer: Any = None,
) -> dict[str, Any]:
    direct_frozen_inputs = validate_frozen_inputs()
    direct_cw12_failure_evidence = validate_cw12_failure_evidence()
    direct_forensic_contract = validate_forensic_contract()
    direct_runtime = validate_runtime(require_cuda=True)
    modules = frozen_modules()
    # Use the caller-supplied primary object throughout the live patch chain.
    modules["primary"] = primary
    holder: dict[str, Any] = {}

    def internal_consumer(context: Mapping[str, Any]) -> None:
        if holder:
            raise RuntimeError("CW13 live consumer called more than once")
        holder.update(
            run_cw13_consumer(
                context,
                modules,
                user_candidate_consumer=candidate_consumer,
            )
        )

    cw11_result = modules["cw11"].run_probe(
        primary,
        primary_source,
        primary_evidence,
        candidate_consumer=internal_consumer,
    )
    if not holder:
        raise RuntimeError("CW13 consumer was not reached by exact CW11 reconstruction")
    final_checks = {
        "cw11_reconstruction_status_exact": cw11_result["status"]
        == "exploratory_33row_specialist_valid_CW11_success",
        "cw11_model_sha_exact": cw11_result["second_stage"]["decision"]
        ["candidate_model_state_sha256_before_CW10_finally_restore"]
        == CW11_MODEL_SHA256,
        "cw11_vector_sha_exact": cw11_result["second_stage"]["decision"]
        ["terminal_cumulative_float64_le_sha256"]
        == CW11_VECTOR_SHA256,
        "cw11_ledger_sha_exact": cw11_result["second_stage"]["active_pair_contract"]
        ["final_canonical_ledger_sha256"]
        == CW11_ACTIVE_LEDGER_SHA256,
        "cw13_local_restore_CW11": bool(
            holder.get("local_finally", {}).get(
                "restored_exact_CW11_before_outer_finally", False
            )
        ),
        "cw13_local_restore_matmul_precision": bool(
            holder.get("local_finally", {}).get(
                "matmul_precision_restored_exact", False
            )
        ),
        "outer_chain_final_raw_restore": bool(cw11_result["final_integrity"]["pass"]),
    }
    if not all(final_checks.values()):
        raise RuntimeError(f"CW13 outer reconstruction/finally failed: {final_checks}")
    return {
        "schema_version": SCHEMA,
        "status": holder["status"],
        "classification": holder["classification"],
        "scope": {
            "exact_CW11_reconstruction": True,
            "CW13_favorable_transition_restoration": True,
            "authoritative_full_six_official_B256_oracle_each_proposal": True,
            "candidate_RAM_only": True,
            "optimizer_backward_training": False,
            "model_or_result_writes": 0,
            "network_upload_submission": False,
            "broad_or_gold_access": False,
            "specialist_valid_consumed_dev_only": True,
        },
        "input_lock": {
            "primary": dict(primary_evidence),
            "cw11_probe": {
                "path": str(CW11_PROBE.relative_to(ROOT)),
                "sha256": MODULE_SHAS[CW11_PROBE],
            },
            "guardplan": {
                "path": str(GUARDPLAN.relative_to(ROOT)),
                "sha256": GUARDPLAN_SHA256,
            },
            "forensic": {
                "path": str(FORENSIC.relative_to(ROOT)),
                "sha256": FORENSIC_SHA256,
            },
            "failed_decision": {
                "path": str(FAILED_DECISION.relative_to(ROOT)),
                "sha256": FAILED_DECISION_SHA256,
            },
            "consumed_CW12_failure": {
                "attempt_marker_path": str(CW12_ATTEMPT_MARKER.relative_to(ROOT)),
                "attempt_marker_sha256": CW12_ATTEMPT_MARKER_SHA256,
                "failure_stdout_path": str(CW12_FAILURE_STDOUT.relative_to(ROOT)),
                "failure_stdout_sha256": CW12_FAILURE_STDOUT_SHA256,
                "parent_solver_sha256": CW12_PARENT_SOLVER_SHA256,
            },
        },
        "frozen_inputs": direct_frozen_inputs,
        "cw12_consumed_failure_evidence": direct_cw12_failure_evidence,
        "forensic_contract": direct_forensic_contract,
        "runtime": direct_runtime,
        "cw11_reconstruction_summary": {
            "status": cw11_result["status"],
            "model_state_sha256": CW11_MODEL_SHA256,
            "total_float64_le_sha256": CW11_VECTOR_SHA256,
            "active_pair_count": EXPECTED_LEGACY_ACTIVE_PAIRS,
            "active_pair_ledger_sha256": CW11_ACTIVE_LEDGER_SHA256,
        },
        "second_stage": holder,
        "terminal_reconstruction_audit": holder[
            "terminal_reconstruction_audit"
        ],
        "terminal": {
            "status": holder["status"],
            "decision": holder["decision"],
            "reconstruction_payload": holder["terminal_reconstruction_payload"],
            "reconstruction_audit": holder["terminal_reconstruction_audit"],
            "active_cut_ledger": holder["terminal_active_cut_ledger"],
            "active_cut_ledger_sha256": holder[
                "terminal_active_cut_ledger_sha256"
            ],
            "downstream_contract": holder["downstream_contract"],
        },
        "final_integrity": {
            "checks": final_checks,
            "pass": all(final_checks.values()),
            "model_left_raw_after_outer_finally": True,
        },
        "run_executed": True,
        "cuda_accessed": True,
        "writes_performed": False,
    }


def static_result() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    if stat.S_IMODE(SCRIPT.stat().st_mode) != FROZEN_MODE:
        raise RuntimeError("CW13 script must be frozen mode 0555")
    source_audit = static_source_audit(source)
    if not source_audit["pass"]:
        raise RuntimeError(f"static source audit failed: {source_audit}")
    return {
        "schema_version": SCHEMA,
        "status": "static_ready_CW13_run_implemented",
        "classification": dict(CLASSIFICATION),
        "contract": {
            "starting_model_sha256": CW11_MODEL_SHA256,
            "parent_CW12_solver_sha256": CW12_PARENT_SOLVER_SHA256,
            "CW13_favorable_transition_restoration": True,
            "fixed_repair_physical_rows": EXPECTED_INITIAL_REPAIR_PHYSICAL_ROWS,
            "fixed_repair_context_guards": EXPECTED_INITIAL_REPAIR_CONTEXT_GUARDS,
            "legacy_B33_rows": EXPECTED_LEGACY_ROWS,
            "legacy_active_pairs": EXPECTED_LEGACY_ACTIVE_PAIRS,
            "authoritative_view_order": list(VIEW_ORDER),
            "official_batch_size": 256,
            "official_workers": 8,
            "changed_physical_monitor_rows": EXPECTED_CHANGED_PHYSICAL_ROWS,
            "raw_to_cw11_transition_rows": EXPECTED_RAW_TO_CW11_TRANSITIONS,
            "transition_panel_occurrences": EXPECTED_TRANSITION_OCCURRENCES,
            "unique_transition_batch_contexts": EXPECTED_UNIQUE_OFFICIAL_CONTEXTS,
            "max_outer_iterations": MAX_OUTER_ITERATIONS,
            "per_step_l2_cap": STEP_L2_CAP,
            "additional_total_l2_cap_from_cw11": ADDITIONAL_TOTAL_L2_CAP,
            "l2_cap_absolute_tolerance": L2_CAP_ABS_TOL,
            "actor_parameter_names": list(ACTOR6_NAMES),
            "first_feasible": FIRST_FEASIBLE,
            "candidate_ordering": "fixed_iteration_order_no_best_of_N",
            "terminal_semantics": "consumed_valid_optimization_closure_only",
            "attempt_semantics": (
                "stdout-only probe has no marker; a future audited one-shot launcher must own "
                "unique absent output and attempt-marker enforcement"
            ),
        },
        "runtime": validate_runtime(require_cuda=False),
        "frozen_inputs": validate_frozen_inputs(),
        "cw12_consumed_failure_evidence": validate_cw12_failure_evidence(),
        "forensic_contract": validate_forensic_contract(),
        "source_audit": source_audit,
        "run_executed": False,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="static")
    args = parser.parse_args()
    if args.mode == "static":
        result = static_result()
    else:
        validate_runtime(require_cuda=False)
        cw11 = import_frozen(
            CW11_PROBE,
            MODULE_SHAS[CW11_PROBE],
            EXPECTED_INPUT_MODES[CW11_PROBE],
            "cw13_main_frozen_cw11",
        )
        source, self_evidence = cw11.read_regular_bytes(
            SCRIPT,
            None,
            "CW13 consumed-valid solver",
            expected_mode=FROZEN_MODE,
        )
        source_audit = static_source_audit(source)
        if not source_audit["pass"]:
            raise RuntimeError(f"CW13 source audit failed: {source_audit}")
        primary_source, primary_evidence = cw11.read_regular_bytes(
            PRIMARY,
            MODULE_SHAS[PRIMARY],
            "CW13 frozen primary source",
            expected_mode=FROZEN_MODE,
        )
        primary = import_frozen(
            PRIMARY,
            MODULE_SHAS[PRIMARY],
            EXPECTED_INPUT_MODES[PRIMARY],
            "cw13_main_frozen_primary",
        )
        result = run_probe(
            primary,
            primary_source,
            primary_evidence,
        )
        result["input_lock"]["self"] = self_evidence
        result["source_audit"] = source_audit
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()

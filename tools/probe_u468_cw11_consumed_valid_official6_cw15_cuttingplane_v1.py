#!/usr/bin/env python3
"""CW15 sequential-affine consumed-valid official-B256 cutting-plane probe.

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
SCRIPT = TOOLS / "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py"
SCHEMA = "ptcg-cw15-consumed-valid-official-b256-sequential-affine-tangent-cuttingplane-v1"
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

CW13_PARENT_SOLVER_SHA256 = (
    "dbbdc12e7c2f571f82de92d30450d8ac794c7215ee86137b9aa97ebc42d28297"
)
CW13_LAUNCHER_SHA256 = (
    "d170a1a6aefcbdaac86d76bbe5ab64a5cf915ffa20cb2ab84b790ec5124636bf"
)
CW13_ATTEMPT_MARKER = ROOT / (
    "artifacts/.ptcg-cw13_consumed_valid_official6_cuttingplane_20260802_v1-"
    "attempt.json"
)
CW13_ATTEMPT_MARKER_SHA256 = (
    "d9e069424ce6ffa49d9617838f085ebdf2723951caeb605e2b9ad6406e3efa34"
)
CW13_CLOSED_STDOUT = ROOT / (
    "artifacts/cw13_consumed_valid_official6_cuttingplane_20260802_v1.stdout.json"
)
CW13_CLOSED_STDOUT_SHA256 = (
    "6e92ad787adb2ce9a074c3a4cebaf1ce8e5dd9d9174db26822f524c7e85e0c7b"
)
CW13_STDERR_AUDIT = ROOT / (
    "artifacts/cw13_consumed_valid_official6_cuttingplane_20260802_v1."
    "stderr-audit.json"
)
CW13_STDERR_AUDIT_SHA256 = (
    "f72537d134612242865c7fe689f20e740deb542959ac57e68ce426fec7255193"
)
CW13_CHILD_STDERR_SHA256 = (
    "5ceb096b8ac5376a3ee276bf70012f90cb2fd479e3fa2804f2c0a8145aff4512"
)
CW13_CHILD_STDERR_BYTES = 347
CW13_ITERATION1_ADDITIONAL_L2 = 0.0009424461480998953
CW13_ITERATION2_UNCAPPED_CORRECTION_L2 = 0.0011619656695555538
CW13_ITERATION2_CUMULATIVE_TOTAL_L2 = 0.0013625501502583153
CW13_THRESHOLD_FUNCTION_SHAS = {
    "positive_bf16_q": "c39f13a4cc48ebf04296506059d41ed4e14bfedffac7df89be4f2a87a1286242",
    "dynamic_cut_for_harm": "db6098d3d017c35db1f1a2a70b1116c4f38b32fd5181b8537220944cad44ea2c",
    "dynamic_restoration_cut_for_favorable_transition": "5155a7a2ed1d2863d0da612191bf3d75205170d8493855a70d8cf26dc8f401e6",
    "fixed_repair_gates_and_cuts": "ae72debe74b2409200cadc6b21e81542201c334cd0aa941c75425ec8e5501081",
    "legacy_pair_cuts": "123fda651f92498669f5a32a99a4cc5ecfe66f2e4b96a7a6c2700af64ed255b9",
    "initial_official_repair_cuts": "9aead99e06041cb1a30a17544d0a59a3b9b4163872534b5ed3f1baafa08a239e",
    "legacy_dynamic_cuts": "c52f9967d7e84ce46b78e2482e9cdfea213f3bf17af803acb34be09f182c9187",
    "merge_active_cuts": "1e45787ba545d5b0090d3758d0b9e2aa3483ee7853173807cb827776ab990b35",
}

CW14_PARENT_SOLVER_SHA256 = (
    "febfc16225cc0b915fead337cbf140ff9b4c94d1f5b1cf12921f51b3b340e51f"
)
CW14_LAUNCHER_SHA256 = (
    "11d64eb708cbd39e06b8ae2c8028c515c50ad9541ab88848afed0a388ad20af7"
)
CW14_ATTEMPT_MARKER = ROOT / (
    "artifacts/.ptcg-cw14_consumed_valid_official6_cuttingplane_20260802_v1-"
    "attempt.json"
)
CW14_ATTEMPT_MARKER_SHA256 = (
    "5a427b29aab2996472f4cad1344370bc4784d5ce40f611bdfd9c2565556c316d"
)
CW14_CLOSED_STDOUT = ROOT / (
    "artifacts/cw14_consumed_valid_official6_cuttingplane_20260802_v1.stdout.json"
)
CW14_CLOSED_STDOUT_SHA256 = (
    "3f903ba9320b61c70fdcf5794d1676e3aea44f576498f58e6148b69230576e69"
)
CW14_STDERR_AUDIT = ROOT / (
    "artifacts/cw14_consumed_valid_official6_cuttingplane_20260802_v1."
    "stderr-audit.json"
)
CW14_STDERR_AUDIT_SHA256 = (
    "4897afc99ec44b2e532ff24f16e8d993b46d761323072642e03dc95d4b61486d"
)
CW14_CHILD_STDERR_SHA256 = CW13_CHILD_STDERR_SHA256
CW14_CHILD_STDERR_BYTES = CW13_CHILD_STDERR_BYTES
CW14_ITERATION1_ADDITIONAL_L2 = 0.0009424461480998953
CW14_TERMINAL_ADDITIONAL_L2 = 0.0009447829261258565
CW14_TERMINAL_TOTAL_FROM_RAW_L2 = 0.007979438546138178
CW14_ITERATION1_MODEL_SHA256 = (
    "ba5098d4e5c52bb6a02c4e69fa2015428d018f85953edc4fb6bc14330634d769"
)
CW14_STALLED_MODEL_SHA256 = (
    "d3d4d3a573c2153a33de691b37a42a490d7f0d9a327f165dffa6bade5576a50d"
)
CW14_ITERATION1_ADDITIONAL_SHA256 = (
    "e3814600afa6f1c7843ca073d89a86cb1e32cf162d89f8cbfaf8db378ad3183f"
)
CW14_STALLED_ADDITIONAL_SHA256 = (
    "08712b376218bbe223ff7a9de1373413af8f2bf32e6cf797ac1959b12e342721"
)
CW14_ITERATION1_LINEARIZATION_SHA256 = (
    "a4cd72180e54b70bc28bdcb63dd2272dfdc175babe04ea88b82eff1d36dcc159"
)
CW14_STALLED_LINEARIZATION_SHA256 = (
    "5ca3b7ef897fbc76ea25c27a0ab89fd7a233bad90d36f4a26d43dd42b60eecc1"
)
CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256 = (
    "87b4dec5267ce5343f75acfae5807f1f91192a49192703bc2b3f70677b42b31b"
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
QP_DIRECT_RESIDUAL_ABS_TOL = 1e-8
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
    CW14_ATTEMPT_MARKER: 0o444,
    CW14_CLOSED_STDOUT: 0o444,
    CW14_STDERR_AUDIT: 0o444,
    CW13_ATTEMPT_MARKER: 0o444,
    CW13_CLOSED_STDOUT: 0o444,
    CW13_STDERR_AUDIT: 0o444,
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


def validate_cw13_closure_evidence() -> dict[str, Any]:
    """Bind the exact CW13 closure and isolate its cumulative-step geometry."""

    marker = read_json_locked(
        CW13_ATTEMPT_MARKER, CW13_ATTEMPT_MARKER_SHA256, 0o444
    )
    closed = read_json_locked(
        CW13_CLOSED_STDOUT, CW13_CLOSED_STDOUT_SHA256, 0o444
    )
    stderr_audit = read_json_locked(
        CW13_STDERR_AUDIT, CW13_STDERR_AUDIT_SHA256, 0o444
    )
    second = closed.get("second_stage")
    terminal = closed.get("terminal")
    if not isinstance(second, dict) or not isinstance(terminal, dict):
        raise RuntimeError("CW13 closure is missing second_stage or terminal")
    iterations = second.get("iterations")
    if not isinstance(iterations, list) or len(iterations) != 3:
        raise RuntimeError("CW13 closure iteration cardinality drift")
    iteration0, iteration1, iteration2 = iterations
    child_stderr = base64.b64decode(
        str(stderr_audit["base64"]).encode("ascii"), validate=True
    )
    failed_acceptance = sorted(
        key
        for key, value in iteration1["acceptance"]["checks"].items()
        if value is False
    )
    added = iteration1["post_oracle_cut_merge"]["added"]
    threshold_origin_ledger = sorted(
        (
            tuple(str(value) for value in cut["origins"]),
            float(cut["threshold"]),
        )
        for cut in added
    )
    expected_threshold_origin_ledger = sorted(
        [
            *[(('favorable_transition_dynamic_restoration',), 0.0009765625)] * 4,
            *[(('favorable_transition_dynamic_restoration',), 0.0028076171875)] * 4,
            *[(('full_six_stream_new_harm',), 0.0078125)] * 4,
        ]
    )
    unique_added_pairs: dict[tuple[str, int, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for cut in added:
        unique_added_pairs[
            (
                str(cut["context_hash"]),
                int(cut["positive_option"]),
                int(cut["negative_option"]),
            )
        ].append(cut)
    unique_added_pair_anchor_rhs = sorted(
        float(values[0]["threshold"])
        - float(values[0]["threshold_sources"]["frozen_cw11_same_shape"])
        for values in unique_added_pairs.values()
    )
    decision = second["decision"]
    post = stderr_audit["post_child_integrity"]
    checks = {
        "marker_schema_status_exact": marker.get("schema_version")
        == "ptcg-cw13-consumed-valid-official6-one-shot-attempt-v1"
        and marker.get("status")
        == "one_shot_attempt_consumed_before_cuda_and_official6",
        "marker_one_attempt_no_retry": marker.get("attempts_authorized") == 1
        and marker.get("retry_authorized") is False,
        "marker_solver_exact": marker.get("solver", {}).get("sha256")
        == CW13_PARENT_SOLVER_SHA256
        and marker.get("solver", {}).get("mode_octal") == "0555"
        and marker.get("solver", {}).get("nlink") == 1,
        "marker_launcher_exact": marker.get("launcher", {}).get("sha256")
        == CW13_LAUNCHER_SHA256
        and marker.get("launcher", {}).get("mode_octal") == "0555"
        and marker.get("launcher", {}).get("nlink") == 1,
        "marker_targets_exact": marker.get("stdout_output")
        == str(CW13_CLOSED_STDOUT.relative_to(ROOT))
        and marker.get("stderr_audit")
        == str(CW13_STDERR_AUDIT.relative_to(ROOT)),
        "marker_preflight_31_exact": marker.get(
            "independent_frozen_input_rehash", {}
        ).get("pass")
        is True
        and marker.get("independent_frozen_input_rehash", {}).get("binding_count")
        == 31,
        "closed_schema_status_exact": closed.get("schema_version")
        == "ptcg-cw13-consumed-valid-official-b256-cuttingplane-v1"
        and closed.get("status") == "closed_no_CW13_candidate"
        and second.get("status") == "closed_no_CW13_candidate"
        and terminal.get("status") == "closed_no_CW13_candidate",
        "closed_run_integrity_exact": closed.get("run_executed") is True
        and closed.get("cuda_accessed") is True
        and closed.get("writes_performed") is False
        and closed.get("classification") == CLASSIFICATION
        and closed.get("final_integrity", {}).get("pass") is True,
        "closed_source_and_input_audits_exact": closed.get("source_audit", {}).get(
            "pass"
        )
        is True
        and closed.get("frozen_inputs", {}).get("all_exact") is True
        and closed.get("frozen_inputs", {}).get("binding_count") == 31
        and closed.get("cw12_consumed_failure_evidence", {}).get("pass") is True,
        "closed_self_solver_exact": closed.get("input_lock", {}).get("self", {}).get(
            "sha256"
        )
        == CW13_PARENT_SOLVER_SHA256,
        "terminal_absent_closed_exact": decision.get("terminal_iteration") is None
        and decision.get("terminal_model_state_sha256") is None
        and terminal.get("reconstruction_payload") is None
        and terminal.get("reconstruction_audit") is None
        and terminal.get("active_cut_ledger") is None
        and terminal.get("active_cut_ledger_sha256") is None,
        "closure_reason_is_cumulative_total_cap": decision.get("close_reason")
        == "fail_closed_L2_cap_gate"
        and decision.get("terminal_active_cut_count") == 50,
        "iteration_order_exact": [value.get("iteration") for value in iterations]
        == [0, 1, 2]
        and iteration0.get("kind")
        == "exact_CW11_reference_before_fixed3_repair"
        and iteration1.get("kind")
        == "unique_fixed_order_official_context_proposal"
        and iteration2.get("kind") == "fail_closed_before_proposal",
        "iteration1_minimum_step_exact": math.isclose(
            float(iteration1["additional_l2"]),
            CW13_ITERATION1_ADDITIONAL_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        and math.isclose(
            float(iteration1["step_l2"]),
            CW13_ITERATION1_ADDITIONAL_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        and iteration1["qp"]["capped"] is False,
        "iteration1_all_original_gates_then_12_harms": iteration1["oracle"]
        ["authoritative_60_gates"]["passed_gate_count"]
        == 60
        and iteration1["oracle"]["authoritative_60_gates"]["pass"] is True
        and iteration1["fixed_repair_gates"]["pass"] is True
        and iteration1["active_cut_gates"]["pass"] is True
        and iteration1["oracle"]["new_harm_count_vs_cw11"] == 12
        and iteration1["oracle"]["favorable_transition_break_count"] == 8
        and len(iteration1["oracle"]["deterministic_restoration_cuts"]) == 8,
        "iteration1_only_expected_terminal_failures": iteration1["acceptance"]
        ["pass"]
        is False
        and failed_acceptance
        == ["forensic_favorable_retention_pass", "new_harm_count_zero"],
        "iteration1_hard_checks_and_restoration_exact": all(
            bool(value) for value in iteration1["oracle"]["hard_checks"].values()
        )
        and iteration1["oracle"]["transition_state_checks"]
        ["favorable_breaks_have_exact_restoration_cuts"]
        is True
        and iteration1["favorable_transition_restoration"]
        == {
            "break_count": 8,
            "cut_count": 8,
            "retained": False,
            "will_continue_if_nonterminal": True,
        },
        "iteration1_cut_merge_38_to_50_exact": iteration1[
            "deterministic_addition_count"
        ]
        == 12
        and iteration1["deterministic_addition_identity_sha256"]
        == "e91917703f6573f90016444c07eb7350a565b69a8a3362b48d1c7ec4710f5dc0"
        and len(added) == 12
        and not iteration1["post_oracle_cut_merge"]["strengthened"]
        and iteration1["post_oracle_cut_merge"]["active_count"] == 50,
        "iteration1_threshold_origin_ledger_exact": threshold_origin_ledger
        == expected_threshold_origin_ledger,
        "iteration1_three_unique_CW11_anchored_pairs_light_rhs": len(
            unique_added_pairs
        )
        == 3
        and all(len(values) == 4 for values in unique_added_pairs.values())
        and all(
            values[0]["anchor_model"] == "frozen_CW11_4317"
            and values[0]["anchor_source"]
            == "same_official_shape_CW11_margin"
            for values in unique_added_pairs.values()
        )
        and unique_added_pair_anchor_rhs == [0.0, 0.0, 0.0009765625],
        "iteration2_current_point_correction_exact": iteration2.get("close_reason")
        == "fail_closed_L2_cap_gate"
        and iteration2["qp"]["capped"] is True
        and math.isclose(
            float(iteration2["qp"]["uncapped_l2"]),
            CW13_ITERATION2_UNCAPPED_CORRECTION_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        and float(iteration2["qp"]["applied_l2"]) == STEP_L2_CAP,
        "iteration2_cumulative_total_exceeded_unchanged_cap": math.isclose(
            float(iteration2["proposal_additional_l2"]),
            CW13_ITERATION2_CUMULATIVE_TOTAL_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        and iteration2["proposal_additional_l2"] > ADDITIONAL_TOTAL_L2_CAP
        and iteration2["cap_checks"]
        == {
            "step_positive_finite": True,
            "step_at_most_cap": True,
            "additional_total_finite": True,
            "additional_total_at_most_cap": False,
        },
        "stderr_schema_status_exact": stderr_audit.get("schema_version")
        == "ptcg-cw13-consumed-valid-official6-one-shot-stderr-audit-v1"
        and stderr_audit.get("status")
        == "captured_losslessly_not_a_success_veto",
        "stderr_lossless_no_traceback": len(child_stderr) == CW13_CHILD_STDERR_BYTES
        and stderr_audit.get("bytes") == CW13_CHILD_STDERR_BYTES
        and hashlib.sha256(child_stderr).hexdigest()
        == CW13_CHILD_STDERR_SHA256
        == stderr_audit.get("sha256")
        and stderr_audit.get("checks")
        == {"python_traceback_absent": True, "utf8_exact": True},
        "stderr_post_child_integrity_exact": post.get("pass") is True
        and all(bool(value) for value in post.get("checks", {}).values())
        and post.get("solver", {}).get("sha256") == CW13_PARENT_SOLVER_SHA256
        and post.get("attempt_marker", {}).get("sha256")
        == CW13_ATTEMPT_MARKER_SHA256
        and post.get("independent_frozen_input_rehash", {}).get("binding_count")
        == 31
        and post.get("independent_frozen_input_rehash", {}).get("pass") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW13 anchored-total closure evidence drift: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "parent_solver_sha256": CW13_PARENT_SOLVER_SHA256,
        "attempt_marker_sha256": CW13_ATTEMPT_MARKER_SHA256,
        "closed_stdout_sha256": CW13_CLOSED_STDOUT_SHA256,
        "stderr_audit_sha256": CW13_STDERR_AUDIT_SHA256,
        "geometry_diagnosis": {
            "iteration1_additional_l2": CW13_ITERATION1_ADDITIONAL_L2,
            "iteration2_current_point_uncapped_correction_l2":
            CW13_ITERATION2_UNCAPPED_CORRECTION_L2,
            "iteration2_cumulative_total_l2":
            CW13_ITERATION2_CUMULATIVE_TOTAL_L2,
            "unchanged_total_cap": ADDITIONAL_TOTAL_L2_CAP,
            "failure_before_iteration2_oracle": True,
            "iteration1_unique_new_pair_count": len(unique_added_pairs),
            "iteration1_unique_new_pair_anchor_rhs":
            unique_added_pair_anchor_rhs,
            "threshold_change_supported_by_evidence": False,
            "single_variable_change": "minimum_total_additional_from_CW11_anchor",
        },
    }


def validate_cw14_closure_evidence() -> dict[str, Any]:
    """Bind the exact CW14 closure and prove its anchor-tangent fixed point."""

    marker = read_json_locked(
        CW14_ATTEMPT_MARKER, CW14_ATTEMPT_MARKER_SHA256, 0o444
    )
    closed = read_json_locked(
        CW14_CLOSED_STDOUT, CW14_CLOSED_STDOUT_SHA256, 0o444
    )
    stderr_audit = read_json_locked(
        CW14_STDERR_AUDIT, CW14_STDERR_AUDIT_SHA256, 0o444
    )
    second = closed.get("second_stage")
    terminal = closed.get("terminal")
    if not isinstance(second, dict) or not isinstance(terminal, dict):
        raise RuntimeError("CW14 closure is missing second_stage or terminal")
    iterations = second.get("iterations")
    if not isinstance(iterations, list) or len(iterations) != 13:
        raise RuntimeError("CW14 closure iteration cardinality drift")
    iteration0, iteration1, *stalled = iterations
    if len(stalled) != 11:
        raise RuntimeError("CW14 stalled iteration cardinality drift")
    child_stderr = base64.b64decode(
        str(stderr_audit["base64"]).encode("ascii"), validate=True
    )
    post = stderr_audit["post_child_integrity"]

    def violated_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
        identity = value["identity"]
        return (
            str(identity[0]),
            str(identity[1]),
            str(identity[2]),
            int(identity[3]),
            int(identity[4]),
            float(value["margin"]),
            float(value["threshold"]),
            float(value["residual"]),
        )

    expected_pair_specs = {
        (
            "4afd6b4112ce49bab88c5192c1c5425a395cd629ebf2d8fc81cbbcba6d0d2b0b",
            "ccd32395c70e834cc78a32d66fd182596a5977de86f2c5506c811878975e3f1c",
            1,
            8,
        ): (-0.00390625, 0.0009765625, -0.0048828125),
        (
            "861c6645f1764ac84ecbc9fe9448697ddf74b3e728d65dc28142c6d038ccea66",
            "3b8a466d01e49811024d8d01a187aab370672b50b64804ef889cc41a20ed4145",
            1,
            2,
        ): (0.002685546875, 0.0028076171875, -0.0001220703125),
        (
            "ffd866085ed5f2741e44a98d6c34bb33a2d421c1289c29c9b393a752d9b24a0a",
            "aad8534415cc11e23338dcd82ed3bbf1ec14c375f953fa67ff17d006955463f4",
            3,
            6,
        ): (-0.0078125, 0.0078125, -0.015625),
    }
    expected_violated = sorted(
        (
            context_hash,
            decision_sha,
            metric,
            positive,
            negative,
            margin,
            threshold,
            residual,
        )
        for (
            context_hash,
            decision_sha,
            positive,
            negative,
        ), (margin, threshold, residual) in expected_pair_specs.items()
        for metric in POLICY_METRICS
    )
    stalled_violated = [
        sorted(
            violated_signature(value)
            for value in iteration["active_cut_gates"]["records"]
            if value["pass"] is False
        )
        for iteration in stalled
    ]
    stalled_failed_acceptance = [
        sorted(
            key
            for key, value in iteration["acceptance"]["checks"].items()
            if value is False
        )
        for iteration in stalled
    ]
    decision = second["decision"]
    checks = {
        "marker_schema_status_exact": marker.get("schema_version")
        == "ptcg-cw14-consumed-valid-official6-one-shot-attempt-v1"
        and marker.get("status")
        == "one_shot_attempt_consumed_before_cuda_and_official6",
        "marker_one_attempt_no_retry": marker.get("attempts_authorized") == 1
        and marker.get("retry_authorized") is False,
        "marker_solver_exact": marker.get("solver", {}).get("sha256")
        == CW14_PARENT_SOLVER_SHA256
        and marker.get("solver", {}).get("mode_octal") == "0555"
        and marker.get("solver", {}).get("nlink") == 1,
        "marker_launcher_exact": marker.get("launcher", {}).get("sha256")
        == CW14_LAUNCHER_SHA256
        and marker.get("launcher", {}).get("mode_octal") == "0555"
        and marker.get("launcher", {}).get("nlink") == 1,
        "marker_targets_and_rehash_exact": marker.get("stdout_output")
        == str(CW14_CLOSED_STDOUT.relative_to(ROOT))
        and marker.get("stderr_audit")
        == str(CW14_STDERR_AUDIT.relative_to(ROOT))
        and marker.get("independent_frozen_input_rehash", {}).get("pass") is True
        and marker.get("independent_frozen_input_rehash", {}).get("binding_count")
        == 34,
        "closed_schema_status_exact": closed.get("schema_version")
        == "ptcg-cw14-consumed-valid-official-b256-anchored-total-cuttingplane-v1"
        and closed.get("status") == "closed_no_CW14_candidate"
        and second.get("status") == "closed_no_CW14_candidate"
        and terminal.get("status") == "closed_no_CW14_candidate",
        "closed_run_and_input_audits_exact": closed.get("run_executed") is True
        and closed.get("cuda_accessed") is True
        and closed.get("writes_performed") is False
        and closed.get("classification") == CLASSIFICATION
        and closed.get("final_integrity", {}).get("pass") is True
        and closed.get("source_audit", {}).get("pass") is True
        and closed.get("frozen_inputs", {}).get("all_exact") is True
        and closed.get("frozen_inputs", {}).get("binding_count") == 34
        and closed.get("cw12_consumed_failure_evidence", {}).get("pass") is True
        and closed.get("cw13_consumed_closure_evidence", {}).get("pass") is True,
        "closed_self_solver_exact": closed.get("input_lock", {}).get("self", {}).get(
            "sha256"
        )
        == CW14_PARENT_SOLVER_SHA256,
        "terminal_absent_closed_exact": decision.get("terminal_iteration") is None
        and decision.get("terminal_model_state_sha256") is None
        and decision.get("candidate_consumer_called") is False
        and terminal.get("reconstruction_payload") is None
        and terminal.get("reconstruction_audit") is None
        and terminal.get("active_cut_ledger") is None
        and terminal.get("active_cut_ledger_sha256") is None,
        "closure_reason_and_terminal_geometry_exact": decision.get("close_reason")
        == "maximum_12_iterations_without_first_feasible"
        and decision.get("terminal_active_cut_count") == 50
        and math.isclose(
            float(decision["terminal_additional_l2"]),
            CW14_TERMINAL_ADDITIONAL_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        and math.isclose(
            float(decision["terminal_total_from_raw_l2"]),
            CW14_TERMINAL_TOTAL_FROM_RAW_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        ),
        "iteration_order_exact": [value.get("iteration") for value in iterations]
        == list(range(13))
        and iteration0.get("kind")
        == "exact_CW11_reference_before_fixed3_repair"
        and all(
            value.get("kind") == "unique_fixed_order_official_context_proposal"
            for value in iterations[1:]
        ),
        "iteration1_anchor_solution_and_merge_exact": iteration1.get(
            "model_state_sha256"
        )
        == CW14_ITERATION1_MODEL_SHA256
        and iteration1.get("additional_float64_le_sha256")
        == CW14_ITERATION1_ADDITIONAL_SHA256
        and math.isclose(
            float(iteration1["additional_l2"]),
            CW14_ITERATION1_ADDITIONAL_L2,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        and iteration1["gradient_audit"]["cut_linearization_ledger_sha256"]
        == CW14_ITERATION1_LINEARIZATION_SHA256
        and iteration1["active_cut_gates"]["violated_count"] == 0
        and iteration1["oracle"]["new_harm_count_vs_cw11"] == 12
        and iteration1["oracle"]["favorable_transition_break_count"] == 8
        and len(iteration1["post_oracle_cut_merge"]["added"]) == 12
        and not iteration1["post_oracle_cut_merge"]["strengthened"]
        and iteration1["post_oracle_cut_merge"]["active_count"] == 50,
        "iterations2_through12_same_anchor_fixed_point_exact": all(
            value.get("model_state_sha256") == CW14_STALLED_MODEL_SHA256
            and value.get("additional_float64_le_sha256")
            == CW14_STALLED_ADDITIONAL_SHA256
            and math.isclose(
                float(value["additional_l2"]),
                CW14_TERMINAL_ADDITIONAL_L2,
                rel_tol=0.0,
                abs_tol=1e-18,
            )
            and value["gradient_audit"]["cut_linearization_ledger_sha256"]
            == CW14_STALLED_LINEARIZATION_SHA256
            and value["active_cut_gates"]["violated_count"] == 12
            and value["oracle"]["new_harm_count_vs_cw11"] == 8
            and value["oracle"]["favorable_transition_break_count"] == 4
            and value["deterministic_addition_count"] == 8
            and not value["post_oracle_cut_merge"]["added"]
            and not value["post_oracle_cut_merge"]["strengthened"]
            and value["post_oracle_cut_merge"]["active_count"] == 50
            for value in stalled
        ),
        "post_merge_all_12_actual_active_violations_exact": all(
            value == expected_violated for value in stalled_violated
        )
        and len(expected_violated) == 12,
        "positive_but_below_floor_pair_not_lost": any(
            signature[0]
            == "861c6645f1764ac84ecbc9fe9448697ddf74b3e728d65dc28142c6d038ccea66"
            and signature[5] > 0.0
            and signature[7] == -0.0001220703125
            for signature in expected_violated
        ),
        "stalled_acceptance_failures_exact": all(
            value
            == [
                "active_cut_violations_zero",
                "forensic_favorable_retention_pass",
                "new_harm_count_zero",
            ]
            for value in stalled_failed_acceptance
        ),
        "stderr_lossless_no_traceback": stderr_audit.get("schema_version")
        == "ptcg-cw14-consumed-valid-official6-one-shot-stderr-audit-v1"
        and stderr_audit.get("status")
        == "captured_losslessly_not_a_success_veto"
        and len(child_stderr) == CW14_CHILD_STDERR_BYTES
        and stderr_audit.get("bytes") == CW14_CHILD_STDERR_BYTES
        and hashlib.sha256(child_stderr).hexdigest()
        == CW14_CHILD_STDERR_SHA256
        == stderr_audit.get("sha256")
        and stderr_audit.get("checks")
        == {"python_traceback_absent": True, "utf8_exact": True},
        "stderr_post_child_integrity_exact": post.get("pass") is True
        and all(bool(value) for value in post.get("checks", {}).values())
        and post.get("solver", {}).get("sha256") == CW14_PARENT_SOLVER_SHA256
        and post.get("attempt_marker", {}).get("sha256")
        == CW14_ATTEMPT_MARKER_SHA256
        and post.get("independent_frozen_input_rehash", {}).get("binding_count")
        == 34
        and post.get("independent_frozen_input_rehash", {}).get("pass") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW14 sequential-affine diagnosis evidence drift: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "parent_solver_sha256": CW14_PARENT_SOLVER_SHA256,
        "attempt_marker_sha256": CW14_ATTEMPT_MARKER_SHA256,
        "closed_stdout_sha256": CW14_CLOSED_STDOUT_SHA256,
        "stderr_audit_sha256": CW14_STDERR_AUDIT_SHA256,
        "fixed_point_diagnosis": {
            "stalled_iterations": list(range(2, 13)),
            "stalled_model_state_sha256": CW14_STALLED_MODEL_SHA256,
            "stalled_additional_float64_le_sha256":
            CW14_STALLED_ADDITIONAL_SHA256,
            "stalled_anchor_linearization_ledger_sha256":
            CW14_STALLED_LINEARIZATION_SHA256,
            "actual_active_violation_count": len(expected_violated),
            "semantic_pair_count": len(expected_pair_specs),
            "includes_positive_but_below_floor_pair": True,
            "threshold_change_supported_by_evidence": False,
            "single_variable_change":
            "candidate_point_tangents_for_all_post_merge_active_violations",
        },
    }


def anchored_total_geometry_self_test() -> dict[str, Any]:
    """Exact 2-D witness for the cumulative-step false-cap failure mode."""

    old_threshold = 0.0008
    new_threshold = 0.0013
    previous = (0.0008, 0.0)
    minimum_current_step = (0.00025, 0.00025)
    cumulative = tuple(
        left + right
        for left, right in zip(previous, minimum_current_step, strict=True)
    )
    anchored_minimum_total = (0.0008, 0.0005)
    previous_l2 = math.hypot(*previous)
    step_l2 = math.hypot(*minimum_current_step)
    cumulative_l2 = math.hypot(*cumulative)
    anchored_l2 = math.hypot(*anchored_minimum_total)
    anchored_delta = tuple(
        left - right
        for left, right in zip(anchored_minimum_total, previous, strict=True)
    )
    anchored_delta_l2 = math.hypot(*anchored_delta)
    projection_dot = sum(
        left * right for left, right in zip(previous, anchored_delta, strict=True)
    )
    checks = {
        "same_old_cut_threshold": previous[0] >= old_threshold
        and cumulative[0] >= old_threshold
        and anchored_minimum_total[0] >= old_threshold,
        "same_new_cut_threshold": sum(cumulative) >= new_threshold
        and sum(anchored_minimum_total) >= new_threshold,
        "minimum_current_step_is_smaller_but_cumulative_total_exceeds_cap":
        step_l2 < ADDITIONAL_TOTAL_L2_CAP
        and cumulative_l2 > ADDITIONAL_TOTAL_L2_CAP,
        "anchored_minimum_total_satisfies_same_cuts_inside_cap": anchored_l2
        < ADDITIONAL_TOTAL_L2_CAP,
        "anchored_total_is_lower_norm_than_cumulative_step_result": anchored_l2
        < cumulative_l2,
        "nested_minimum_total_norm_monotone": anchored_l2 >= previous_l2,
        "nested_projection_dot_nonnegative": projection_dot
        >= -L2_CAP_ABS_TOL,
        "anchored_delta_preserves_original_step_cap": anchored_delta_l2
        <= STEP_L2_CAP,
        "cap_and_thresholds_unchanged": STEP_L2_CAP
        == ADDITIONAL_TOTAL_L2_CAP
        == 0.001
        and old_threshold == 0.0008
        and new_threshold == 0.0013,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW14 anchored-total geometry self-test failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "witness": {
            "cuts": ["x>=0.0008", "x+y>=0.0013"],
            "previous_l2": previous_l2,
            "minimum_current_step_l2": step_l2,
            "cumulative_total_l2": cumulative_l2,
            "anchored_minimum_total_l2": anchored_l2,
            "anchored_delta_l2": anchored_delta_l2,
            "projection_dot": projection_dot,
            "cap": ADDITIONAL_TOTAL_L2_CAP,
        },
    }


def sequential_affine_tangent_self_test() -> dict[str, Any]:
    """Pure-CPU 1-D witness that candidate tangents escape an anchor stall."""

    import numpy as np

    threshold = 0.0002
    cap = 0.001

    def margin(value: float) -> float:
        return value - 1000.0 * value * value

    def gradient(value: float) -> float:
        return 1.0 - 2000.0 * value

    def point_sha(value: float) -> str:
        vector = np.asarray([value], dtype="<f8")
        return hashlib.sha256(vector.tobytes()).hexdigest()

    x_value = 0.0
    affine_ledger = []
    records = []
    previous_norm = 0.0
    for iteration in range(8):
        observed = margin(x_value)
        slope = gradient(x_value)
        if point_sha(x_value) in {
            value["linearization_point_sha256"] for value in affine_ledger
        }:
            if abs(observed - threshold) <= 1e-15:
                break
            raise RuntimeError("CW15 self-test stalled before feasibility")
        rhs = threshold - observed + slope * x_value
        tangent_residual = slope * x_value - rhs
        record = {
            "semantic_identity": ["self_test", "pair", "top1", 0, 1],
            "linearization_point_sha256": point_sha(x_value),
            "threshold": threshold,
            "observed_margin": observed,
            "gradient": slope,
            "point_dot_gradient": slope * x_value,
            "rhs": rhs,
            "tangent_residual_at_point": tangent_residual,
        }
        affine_ledger.append(record)
        next_x = max(
            float(value["rhs"]) / float(value["gradient"])
            for value in affine_ledger
        )
        actual_residual = margin(next_x) - threshold
        records.append(
            {
                "iteration": iteration,
                "linearization_point": x_value,
                "linearization_point_sha256": point_sha(x_value),
                "observed_margin": observed,
                "gradient": slope,
                "rhs": rhs,
                "tangent_formula_exact": rhs
                == threshold - observed + slope * x_value,
                "tangent_residual_matches_actual": math.isclose(
                    tangent_residual,
                    observed - threshold,
                    rel_tol=0.0,
                    abs_tol=1e-18,
                ),
                "minimum_total_solution": next_x,
                "minimum_total_norm_monotone": next_x + 1e-18
                >= previous_norm,
                "actual_residual": actual_residual,
            }
        )
        previous_norm = next_x
        x_value = next_x
    anchor_only_x = threshold
    exact_small_root = (1.0 - math.sqrt(0.2)) / 2000.0
    checks = {
        "anchor_tangent_proposal_really_violates_nonlinear_margin": margin(
            anchor_only_x
        )
        < threshold,
        "every_tangent_rhs_formula_exact": all(
            value["tangent_formula_exact"] for value in records
        ),
        "every_tangent_point_residual_matches_actual": all(
            value["tangent_residual_matches_actual"] for value in records
        ),
        "affine_ledger_strictly_grows_by_unique_point_sha": len(affine_ledger)
        == len({value["linearization_point_sha256"] for value in affine_ledger})
        and len(affine_ledger) >= 4,
        "minimum_total_norm_monotone_under_nested_tangents": all(
            value["minimum_total_norm_monotone"] for value in records
        ),
        "sequential_tangents_escape_anchor_fixed_point": x_value
        > anchor_only_x
        and abs(x_value - exact_small_root) < 1e-12
        and abs(margin(x_value) - threshold) < 1e-12,
        "solution_stays_inside_unchanged_cap": x_value < cap
        and cap == STEP_L2_CAP == ADDITIONAL_TOTAL_L2_CAP == 0.001,
        "semantic_threshold_unchanged": all(
            value["threshold"] == threshold for value in affine_ledger
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"CW15 sequential-affine tangent self-test failed: {checks}"
        )
    return {
        "checks": checks,
        "pass": True,
        "witness": {
            "margin": "m(x)=x-1000*x^2",
            "threshold": threshold,
            "anchor_only_proposal": anchor_only_x,
            "anchor_only_actual_margin": margin(anchor_only_x),
            "sequential_solution": x_value,
            "sequential_actual_margin": margin(x_value),
            "exact_small_root": exact_small_root,
            "affine_tangent_count": len(affine_ledger),
            "cap": cap,
            "records": records,
        },
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
        CW14_ATTEMPT_MARKER: CW14_ATTEMPT_MARKER_SHA256,
        CW14_CLOSED_STDOUT: CW14_CLOSED_STDOUT_SHA256,
        CW14_STDERR_AUDIT: CW14_STDERR_AUDIT_SHA256,
        CW13_ATTEMPT_MARKER: CW13_ATTEMPT_MARKER_SHA256,
        CW13_CLOSED_STDOUT: CW13_CLOSED_STDOUT_SHA256,
        CW13_STDERR_AUDIT: CW13_STDERR_AUDIT_SHA256,
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
    top_level_literals = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    required_functions = {
        "dynamic_cut_for_harm",
        "dynamic_restoration_cut_for_favorable_transition",
        "fixed_repair_gates_and_cuts",
        "initial_official_repair_cuts",
        "load_official_six_streams",
        "snapshot_official_six_views",
        "full_stream_separation_oracle",
        "linearize_semantic_cuts_at_current_point",
        "candidate_output_state_fingerprint",
        "affine_tangent_identity",
        "validate_affine_tangent_record",
        "merge_affine_tangents",
        "affine_qp_arrays",
        "affine_ledger_integrity_self_test",
        "apply_additional_from_cw11",
        "anchored_total_geometry_self_test",
        "sequential_affine_tangent_self_test",
        "restore_cw11_anchor_for_total_qp",
        "encode_terminal_payload",
        "reconstruct_terminal_candidate",
        "run_outer_cutting_plane",
        "run_cw15_consumer",
        "run_probe",
        "validate_cw12_failure_evidence",
        "validate_cw13_closure_evidence",
        "validate_cw14_closure_evidence",
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
    tangent_segment = segments.get(
        "linearize_semantic_cuts_at_current_point", ""
    )
    affine_merge_segment = segments.get("merge_affine_tangents", "")
    anchor_restore_segment = segments.get("restore_cw11_anchor_for_total_qp", "")
    main_segment = segments.get("main", "")
    threshold_function_shas = {
        name: hashlib.sha256(segments.get(name, "").encode("utf-8")).hexdigest()
        for name in CW13_THRESHOLD_FUNCTION_SHAS
    }
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
    cap_rejection_nodes = [] if outer is None else [
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.If)
        and "fail_closed_anchored_minimum_total_exceeds_cap"
        in (ast.get_source_segment(text, node) or "")
        and any(isinstance(child, ast.Break) for child in ast.walk(node))
    ]
    proposal_additional_assignments = [] if outer is None else [
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "proposal_additional"
            for target in node.targets
        )
    ]
    outer_positions = {
        token: outer_segment.find(token)
        for token in (
            "restore_cw11_anchor_for_total_qp",
            "linearize_semantic_cuts_at_current_point",
            "affine_qp_arrays",
            "solve_minimum_l2_correction",
            "fail_closed_anchored_minimum_total_exceeds_cap",
            "proposal_additional = np.asarray",
        )
    }
    draft_sentinel = "static" + "_" + "draft"
    unfinished_sentinel = "run implementation" + " has not passed"
    checks = {
        "schema_literal_exact": top_level_literals.get("SCHEMA") == SCHEMA,
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
            "linearize_semantic_cuts_at_current_point",
            "merge_affine_tangents",
            "affine_qp_arrays",
            "restore_cw11_anchor_for_total_qp",
            "reconstruct_terminal_candidate",
        }.issubset(calls_in(function_nodes["run_outer_cutting_plane"])),
        "every_new_model_proposal_runs_full_six_then_oracle": len(
            proposal_loops
        )
        == 1,
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
        "threshold_functions_byte_exact_CW13": threshold_function_shas
        == CW13_THRESHOLD_FUNCTION_SHAS,
        "semantic_and_affine_ledgers_are_explicitly_separate": all(
            token in outer_segment
            for token in (
                "active: list[dict[str, Any]]",
                "affine_ledger: list[dict[str, Any]]",
                "semantic_merge = merge_active_cuts(active, additions)",
                "candidate_affine_merge = merge_affine_tangents",
            )
        ),
        "tangent_key_binds_semantic_point_model_threshold_bits": all(
            token in segments.get("affine_tangent_identity", "")
            for token in (
                "semantic_identity",
                "linearization_point_sha256",
                "linearization_model_state_sha256",
                "threshold_float64_le_sha256",
            )
        ),
        "tangent_formula_and_graph_snapshot_equality_hard_gated": all(
            token in tangent_segment
            for token in (
                "graph_outputs_bit_equal_no_grad_snapshot",
                "graph_margin_bit_equal_snapshot",
                "graph_margin_bit_equal_active_gate",
                "rhs_exact_t_minus_m_plus_Gx",
                "tangent_residual_matches_actual_residual",
                "candidate_tangent_point_is_actually_violated",
                "threshold - observed_margin + point_dot_gradient",
            )
        ),
        "initial_and_new_semantics_receive_CW11_anchor_tangent": all(
            token in outer_segment
            for token in (
                "missing_anchor_cuts",
                'linearization_kind="CW11_anchor"',
                "all_current_thresholds_have_CW11_anchor_tangent",
                "new_anchor_tangent_one_to_one",
            )
        )
        and all(
            token in anchor_restore_segment
            for token in (
                "zero_additional_exact",
                "anchor_total_sha_exact",
                "all_official_candidate_snapshots_alias_frozen_reference",
            )
        ),
        "new_semantic_thresholds_receive_anchor_tangent_same_iteration": all(
            token in outer_segment
            for token in (
                "semantic_changed",
                "missing_post_merge_anchor_cuts",
                "post_merge_anchor_affine_merge",
                "new_anchor_tangent_one_to_one",
                "all_current_semantics_now_have_anchor",
                "post_merge_new_semantic_anchor_audit",
            )
        ),
        "post_merge_full_active_false_rows_tangented_not_only_additions": all(
            token in outer_segment
            for token in (
                "semantic_merge = merge_active_cuts(active, additions)",
                "post_merge_active_gates = active_cut_gates",
                'if value["pass"] is False',
                "violated_semantic_cuts",
                'linearization_kind="post_merge_actual_active_violation"',
                "candidate_tangent_semantic_identity_bijection",
                "all_violations_not_only_oracle_additions_are_tangented",
            )
        ),
        "minimum_total_from_CW11_origin_not_cumulative_step": all(
            token in outer_segment
            for token in (
                "anchored_solution",
                "proposal_additional = np.asarray",
                "minimum total additional from exact CW11 origin",
            )
        )
        and "np.add(additional, correction" not in outer_segment
        and "np.add(additional,correction" not in "".join(outer_segment.split())
        and len(proposal_additional_assignments) == 1
        and "asarray" in calls_in(proposal_additional_assignments[0]),
        "affine_ledger_identity_unique_and_repeat_bytes_exact": all(
            token in affine_merge_segment
            for token in (
                "affine_tangent_identity",
                "duplicate identities",
                "repeated affine identity changed bytes",
                "linearization_record_sha256",
                "gradient_float64_le_sha256",
            )
        ),
        "nested_minimum_total_geometry_is_hard_gated": all(
            token in outer_segment
            for token in (
                "affine_cut_set_nested",
                "minimum_total_norm_monotone_nondecreasing",
                "nested_projection_dot_nonnegative",
                "projection_dot",
                "previous_to_proposal_delta_at_most_step_cap",
            )
        ),
        "uncapped_over_0p001_closes_and_clip_never_applies": all(
            token in outer_segment
            for token in (
                "fail_closed_anchored_minimum_total_exceeds_cap",
                "anchored_minimum_total_l2",
                'bool(qp["capped"])',
                '"clipped_vector_applied": False',
            )
        )
        and min(
            outer_positions["fail_closed_anchored_minimum_total_exceeds_cap"],
            outer_positions["proposal_additional = np.asarray"],
        )
        >= 0
        and outer_positions["fail_closed_anchored_minimum_total_exceeds_cap"]
        < outer_positions["proposal_additional = np.asarray"]
        and len(cap_rejection_nodes) == 1
        and "apply_additional_from_cw11" not in calls_in(cap_rejection_nodes[0]),
        "anchor_restore_tangent_ledger_QP_order_exact": min(outer_positions.values())
        >= 0
        and outer_positions["restore_cw11_anchor_for_total_qp"]
        < outer_positions["linearize_semantic_cuts_at_current_point"]
        < outer_positions["affine_qp_arrays"]
        < outer_positions["solve_minimum_l2_correction"]
        < outer_positions["proposal_additional = np.asarray"],
        "anchored_QP_explicit_numeric_and_residual_gates": all(
            token in outer_segment
            for token in (
                "solver_success",
                "solver_status_zero",
                "uncapped_l2_finite",
                "reported_applied_l2_finite",
                "direct_uncapped_residual_gate",
                "QP_DIRECT_RESIDUAL_ABS_TOL",
                "reported_applied_l2_equals_uncapped_l2",
                "anchored_solution_vector_l2_matches_reported_applied",
            )
        ),
        "anchor_candidate_and_combined_QP_residuals_separately_gated": all(
            token in outer_segment
            for token in (
                "combined_direct_residual_gate",
                "CW11_anchor_direct_residual_gate",
                "candidate_tangent_direct_residual_gate",
                "combined_min_matches_frozen_solver",
                "anchor_candidate_and_combined_direct_residuals_pass",
                "qp_group_residual_audit",
            )
        ),
        "stagnation_and_quantized_plateau_rules_explicit": all(
            token in outer_segment
            for token in (
                "exact_x_and_model_state_repeated",
                "x_changed_but_model_state_same",
                "fail_closed_repeated_or_noninjective_x_model_state",
                "BF16_output_plateau_with_changed_model_allowed",
                "semantic_state_repeated",
                "fail_closed_repeated_model_output_violation_state",
                "fail_closed_actual_violation_without_new_tangent",
            )
        ),
        "violations_force_strict_affine_growth": all(
            token in outer_segment
            for token in (
                "nonterminal_actual_violations_force_strict_affine_growth",
                'candidate_affine_merge["strict_growth"]',
                "fail_closed_actual_violation_without_new_tangent",
            )
        ),
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
            token in segments.get("run_cw15_consumer", "")
            for token in (
                'set_float32_matmul_precision("high")',
                "previous_matmul_precision",
                "precision_restored",
            )
        ),
        "first_feasible_literal": top_level_literals.get("FIRST_FEASIBLE") is True,
        "caps_literal": top_level_literals.get("MAX_OUTER_ITERATIONS") == 12
        and top_level_literals.get("STEP_L2_CAP") == 0.001
        and top_level_literals.get("ADDITIONAL_TOTAL_L2_CAP") == 0.001,
        "main_has_static_and_run_branches": 'args.mode == "static"' in main_segment
        and "run_probe(" in main_segment,
        "cw12_consumed_failure_evidence_static_and_run_bound":
        "validate_cw12_failure_evidence" in calls_in(function_nodes["static_result"])
        and "validate_cw12_failure_evidence" in calls_in(function_nodes["run_probe"]),
        "cw13_consumed_closure_evidence_static_and_run_bound":
        "validate_cw13_closure_evidence" in calls_in(function_nodes["static_result"])
        and "validate_cw13_closure_evidence" in calls_in(function_nodes["run_probe"]),
        "cw14_consumed_closure_evidence_static_and_run_bound":
        "validate_cw14_closure_evidence" in calls_in(function_nodes["static_result"])
        and "validate_cw14_closure_evidence" in calls_in(function_nodes["run_probe"]),
        "anchored_total_math_self_test_static_and_run_bound":
        "anchored_total_geometry_self_test" in calls_in(function_nodes["static_result"])
        and "anchored_total_geometry_self_test" in calls_in(function_nodes["run_probe"]),
        "sequential_affine_math_self_test_static_and_run_bound":
        "sequential_affine_tangent_self_test" in calls_in(
            function_nodes["static_result"]
        )
        and "sequential_affine_tangent_self_test" in calls_in(
            function_nodes["run_probe"]
        ),
        "affine_ledger_self_test_static_and_run_bound":
        "affine_ledger_integrity_self_test" in calls_in(
            function_nodes["static_result"]
        )
        and "affine_ledger_integrity_self_test" in calls_in(
            function_nodes["run_probe"]
        ),
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
        "threshold_function_shas": threshold_function_shas,
        "required_functions": sorted(required_functions),
        "declared_functions": sorted(function_names),
        "top_level_literals": top_level_literals,
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
            "cw15_frozen_cw11",
        ),
        "cw10": import_frozen(
            CW10, MODULE_SHAS[CW10], EXPECTED_INPUT_MODES[CW10], "cw15_frozen_cw10"
        ),
        "cw9": import_frozen(
            CW9, MODULE_SHAS[CW9], EXPECTED_INPUT_MODES[CW9], "cw15_frozen_cw9"
        ),
        "primary": import_frozen(
            PRIMARY,
            MODULE_SHAS[PRIMARY],
            EXPECTED_INPUT_MODES[PRIMARY],
            "cw15_frozen_primary",
        ),
        "cutting": import_frozen(
            CUTTING,
            MODULE_SHAS[CUTTING],
            EXPECTED_INPUT_MODES[CUTTING],
            "cw15_frozen_cutting",
        ),
        "geometry": import_frozen(
            GEOMETRY,
            MODULE_SHAS[GEOMETRY],
            EXPECTED_INPUT_MODES[GEOMETRY],
            "cw15_frozen_geometry",
        ),
        "ram": import_frozen(
            RAM, MODULE_SHAS[RAM], EXPECTED_INPUT_MODES[RAM], "cw15_frozen_ram"
        ),
        "legacy": import_frozen(
            FORENSIC_ANALYZER,
            MODULE_SHAS[FORENSIC_ANALYZER],
            EXPECTED_INPUT_MODES[FORENSIC_ANALYZER],
            "cw15_frozen_forensic_analyzer",
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


def float64_vector_sha256(value: Any) -> str:
    import numpy as np

    vector = np.asarray(value, dtype=np.float64)
    return hashlib.sha256(
        np.ascontiguousarray(vector.astype("<f8")).tobytes()
    ).hexdigest()


def float64_scalar_sha256(value: float) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray([value], dtype="<f8").tobytes()).hexdigest()


def candidate_output_state_fingerprint(
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    """Hash the exact current no-grad BF16 output platform, in stream order."""

    ordered_snapshots = [
        ("legacy_B33", str(legacy["context_hash"]), legacy["candidate_snapshot"])
    ]
    ordered_snapshots.extend(
        (
            str(view),
            str(batch["context"]["context_hash"]),
            batch["candidate_snapshot"],
        )
        for view in VIEW_ORDER
        for batch in streams[view]
    )
    digest = hashlib.sha256()
    tensor_identities = []
    total_bytes = 0
    for platform, context_hash, snapshot in ordered_snapshots:
        if snapshot is None:
            raise RuntimeError("CW15 output fingerprint lacks candidate snapshot")
        for output_name in ("policy_logits", "count_logits", "value_logits"):
            tensor = snapshot["outputs_cpu"][output_name].detach().cpu().contiguous()
            if tensor.dtype != torch.bfloat16:
                raise RuntimeError("CW15 output fingerprint is not native BF16")
            raw = tensor.view(torch.uint8).numpy().tobytes()
            tensor_identity = {
                "platform": platform,
                "context_hash": context_hash,
                "output_name": output_name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            header = canonical_json(tensor_identity)
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(len(raw).to_bytes(8, "little"))
            digest.update(raw)
            tensor_identities.append(tensor_identity)
            total_bytes += len(raw)
    return {
        "sha256": digest.hexdigest(),
        "snapshot_count": len(ordered_snapshots),
        "tensor_count": len(tensor_identities),
        "tensor_bytes": total_bytes,
        "ordered_tensor_identity_sha256": canonical_sha(tensor_identities),
        "native_bf16_exact": True,
    }


def affine_tangent_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        *tuple(record["semantic_identity"]),
        str(record["linearization_point_sha256"]),
        str(record["linearization_model_state_sha256"]),
        str(record["threshold_float64_le_sha256"]),
    )


def sanitized_affine_tangent(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key != "gradient_float64"
    }


def validate_affine_tangent_record(record: Mapping[str, Any]) -> dict[str, bool]:
    import numpy as np

    sanitized = sanitized_affine_tangent(record)
    recorded_sha = str(sanitized.pop("linearization_record_sha256"))
    gradient = np.asarray(record["gradient_float64"], dtype=np.float64)
    checks = {
        "gradient_shape_exact": gradient.shape == (ACTOR6_FLAT_LENGTH,),
        "gradient_finite": bool(np.isfinite(gradient).all()),
        "gradient_sha_exact": float64_vector_sha256(gradient)
        == str(record["gradient_float64_le_sha256"]),
        "threshold_bits_sha_exact": float64_scalar_sha256(
            float(record["threshold"])
        )
        == str(record["threshold_float64_le_sha256"]),
        "linearization_record_sha_exact": canonical_sha(sanitized)
        == recorded_sha,
        "identity_hash_fields_hex64": all(
            len(str(record[key])) == 64
            for key in (
                "linearization_point_sha256",
                "linearization_model_state_sha256",
                "threshold_float64_le_sha256",
                "gradient_float64_le_sha256",
                "linearization_record_sha256",
            )
        ),
        "semantic_identity_cardinality_exact": len(record["semantic_identity"])
        == 5,
        "formula_checks_all_true": all(
            bool(value) for value in record["formula_checks"].values()
        ),
        "point_tangent_residual_matches_actual": math.isclose(
            float(record["tangent_residual_at_point"]),
            float(record["actual_residual"]),
            rel_tol=0.0,
            abs_tol=L2_CAP_ABS_TOL,
        ),
    }
    return checks


def merge_affine_tangents(
    affine_ledger: list[dict[str, Any]],
    additions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if any(
        not all(validate_affine_tangent_record(value).values())
        for value in affine_ledger
    ):
        raise RuntimeError("CW15 existing affine ledger record integrity drift")
    if any(
        not all(validate_affine_tangent_record(value).values())
        for value in additions
    ):
        raise RuntimeError("CW15 incoming affine tangent record integrity drift")
    existing = {affine_tangent_identity(value): value for value in affine_ledger}
    if len(existing) != len(affine_ledger):
        raise RuntimeError("CW15 affine ledger already contains duplicate identities")
    added = []
    repeated = []
    batch_keys = [affine_tangent_identity(value) for value in additions]
    if len(batch_keys) != len(set(batch_keys)):
        raise RuntimeError("CW15 affine tangent addition batch is not unique")
    for incoming in additions:
        candidate = dict(incoming)
        key = affine_tangent_identity(candidate)
        if key in existing:
            old = existing[key]
            exact = (
                str(old["linearization_record_sha256"])
                == str(candidate["linearization_record_sha256"])
                and str(old["gradient_float64_le_sha256"])
                == str(candidate["gradient_float64_le_sha256"])
            )
            if not exact:
                raise RuntimeError("CW15 repeated affine identity changed bytes")
            repeated.append(sanitized_affine_tangent(candidate))
            continue
        affine_ledger.append(candidate)
        existing[key] = candidate
        added.append(sanitized_affine_tangent(candidate))
    affine_ledger.sort(key=affine_tangent_identity)
    sanitized = [sanitized_affine_tangent(value) for value in affine_ledger]
    return {
        "added": added,
        "repeated": repeated,
        "added_count": len(added),
        "repeated_count": len(repeated),
        "affine_count": len(affine_ledger),
        "affine_ledger_sha256": canonical_sha(sanitized),
        "strict_growth": len(added) > 0,
    }


def affine_qp_arrays(
    affine_ledger: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    ordered = sorted(affine_ledger, key=affine_tangent_identity)
    if not ordered or len({affine_tangent_identity(value) for value in ordered}) != len(
        ordered
    ):
        raise RuntimeError("CW15 affine QP ledger is empty or non-unique")
    if any(
        not all(validate_affine_tangent_record(value).values()) for value in ordered
    ):
        raise RuntimeError("CW15 affine QP record integrity drift")
    matrix = np.stack(
        [np.asarray(value["gradient_float64"], dtype=np.float64) for value in ordered],
        axis=0,
    )
    rhs = np.asarray([float(value["rhs"]) for value in ordered], dtype=np.float64)
    sanitized = [sanitized_affine_tangent(value) for value in ordered]
    checks = {
        "matrix_shape_exact": matrix.shape
        == (len(ordered), ACTOR6_FLAT_LENGTH),
        "rhs_shape_exact": rhs.shape == (len(ordered),),
        "matrix_and_rhs_finite": bool(
            np.isfinite(matrix).all() and np.isfinite(rhs).all()
        ),
        "identity_unique": len(
            {affine_tangent_identity(value) for value in ordered}
        )
        == len(ordered),
        "record_hashes_hex64": all(
            len(str(value["linearization_record_sha256"])) == 64
            and len(str(value["gradient_float64_le_sha256"])) == 64
            for value in ordered
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW15 affine QP array audit failed: {checks}")
    return matrix, rhs, {
        "checks": checks,
        "pass": True,
        "affine_tangent_count": len(ordered),
        "matrix_shape": list(matrix.shape),
        "rhs_min": float(rhs.min()),
        "rhs_max": float(rhs.max()),
        "ordered_affine_identity_sha256": canonical_sha(
            [list(affine_tangent_identity(value)) for value in ordered]
        ),
        "affine_ledger_sha256": canonical_sha(sanitized),
    }


def affine_ledger_integrity_self_test() -> dict[str, Any]:
    """Pure in-memory checks for tangent identity, deduplication, and QP arrays."""

    import numpy as np

    def make_record(
        *, point_sha: str, model_sha: str, threshold: float, coordinate: int
    ) -> dict[str, Any]:
        gradient = np.zeros(ACTOR6_FLAT_LENGTH, dtype=np.float64)
        gradient[coordinate] = 1.0
        core = {
            "semantic_identity": ["unit-context", "unit-decision", "top1", 1, 2],
            "linearization_point_sha256": point_sha,
            "linearization_model_state_sha256": model_sha,
            "linearization_output_state_sha256": "3" * 64,
            "linearization_kind": "pure_in_memory_self_test",
            "threshold": threshold,
            "threshold_float64_le_sha256": float64_scalar_sha256(threshold),
            "observed_margin": 0.0,
            "gradient_float64_le_sha256": float64_vector_sha256(gradient),
            "point_dot_gradient": 0.0,
            "rhs": threshold,
            "actual_residual": -threshold,
            "tangent_residual_at_point": -threshold,
            "formula_checks": {"self_test": True},
        }
        return {
            **core,
            "linearization_record_sha256": canonical_sha(core),
            "gradient_float64": gradient,
        }

    first = make_record(
        point_sha=CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
        model_sha=CW11_MODEL_SHA256,
        threshold=0.0002,
        coordinate=0,
    )
    second = make_record(
        point_sha="2" * 64,
        model_sha="4" * 64,
        threshold=0.0002,
        coordinate=1,
    )
    changed_threshold = make_record(
        point_sha=CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
        model_sha=CW11_MODEL_SHA256,
        threshold=0.0003,
        coordinate=0,
    )
    ledger: list[dict[str, Any]] = []
    first_merge = merge_affine_tangents(ledger, [first])
    exact_repeat = merge_affine_tangents(ledger, [first])
    nested_merge = merge_affine_tangents(ledger, [second, changed_threshold])
    matrix, rhs, qp_audit = affine_qp_arrays(ledger)
    mismatch_fail_closed = False
    changed_bytes = dict(first)
    changed_bytes["linearization_record_sha256"] = "0" * 64
    try:
        merge_affine_tangents(ledger, [changed_bytes])
    except RuntimeError:
        mismatch_fail_closed = True
    sanitized = [sanitized_affine_tangent(value) for value in ledger]
    checks = {
        "first_merge_adds_one": first_merge["added_count"] == 1
        and first_merge["repeated_count"] == 0,
        "exact_identity_repeat_is_deduplicated": exact_repeat["added_count"] == 0
        and exact_repeat["repeated_count"] == 1,
        "point_model_and_threshold_bits_change_identity": nested_merge[
            "added_count"
        ]
        == 2
        and len({affine_tangent_identity(value) for value in ledger}) == 3,
        "same_identity_changed_bytes_fail_closed": mismatch_fail_closed,
        "private_gradient_not_serialized": all(
            "gradient_float64" not in value for value in sanitized
        ),
        "affine_QP_shapes_exact": matrix.shape == (3, ACTOR6_FLAT_LENGTH)
        and rhs.shape == (3,)
        and qp_audit["pass"] is True,
        "affine_QP_arrays_finite": bool(
            np.isfinite(matrix).all() and np.isfinite(rhs).all()
        ),
        "ledger_sha_hex64": len(nested_merge["affine_ledger_sha256"]) == 64,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW15 affine ledger self-test failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "affine_tangent_count": len(ledger),
        "affine_ledger_sha256": canonical_sha(sanitized),
        "matrix_shape": list(matrix.shape),
        "rhs": rhs.tolist(),
    }


def linearize_semantic_cuts_at_current_point(
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: Mapping[str, Any],
    semantic_cuts: Sequence[Mapping[str, Any]],
    parameters: Sequence[Any],
    linearization_point: Any,
    *,
    linearization_model_state_sha256: str,
    linearization_output_state_sha256: str,
    linearization_kind: str,
    expected_gate_margins: Mapping[tuple[Any, ...], float],
    require_actual_violation: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build absolute affine tangents G_k x >= t-m_k+G_k x_k."""

    import numpy as np

    helper = context["helper"]
    model = context["model"]
    device = next(model.parameters()).device
    point = np.asarray(linearization_point, dtype=np.float64)
    point_sha = float64_vector_sha256(point)
    live_model_sha = helper.model_state_sha256(model.state_dict())
    live_output_fingerprint = candidate_output_state_fingerprint(
        streams, legacy, helper.torch
    )
    ordered = sorted(semantic_cuts, key=cut_identity)
    expected_keys = set(expected_gate_margins)
    point_checks = {
        "point_shape_exact": point.shape == (ACTOR6_FLAT_LENGTH,),
        "point_finite": bool(np.isfinite(point).all()),
        "model_state_sha_exact_live": live_model_sha
        == linearization_model_state_sha256,
        "output_state_sha_exact_snapshots": live_output_fingerprint["sha256"]
        == linearization_output_state_sha256,
        "model_eval_exact": model.training is False,
        "semantic_cut_identity_unique": len({cut_identity(value) for value in ordered})
        == len(ordered),
        "expected_gate_identity_exact": {cut_identity(value) for value in ordered}
        == expected_keys,
        "zero_point_sha_exact_if_CW11_anchor": linearization_kind
        != "CW11_anchor"
        or (
            point_sha == CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256
            and np.count_nonzero(point) == 0
            and linearization_model_state_sha256 == CW11_MODEL_SHA256
        ),
    }
    if not ordered or not all(point_checks.values()):
        raise RuntimeError(f"CW15 linearization point drift: {point_checks}")
    official_lookup = official_context_lookup(streams)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cut in ordered:
        groups[str(cut["context_hash"])].append(cut)
    records = []
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
                key: value.to(device, non_blocking=True)
                for key, value in cpu_batch.items()
            }
            expected = batch["candidate_snapshot"]
        outputs = helper.ppo.model_forward(model, dict(gpu_batch), device)
        graph_checks = {
            "native_BF16": all(
                value.dtype == helper.torch.bfloat16 for value in outputs.values()
            ),
            "graph_outputs_bit_equal_no_grad_snapshot": all(
                helper.torch.equal(
                    outputs[key].detach().cpu(), expected["outputs_cpu"][key]
                )
                for key in ("policy_logits", "count_logits", "value_logits")
            ),
        }
        if not all(graph_checks.values()):
            raise RuntimeError(f"CW15 graph/snapshot fingerprint drift: {graph_checks}")
        logits = outputs["policy_logits"].float()
        for index, cut in enumerate(cuts):
            semantic_identity = cut_identity(cut)
            margin = (
                logits[
                    int(cut["offset_zero_based"]),
                    int(cut["positive_option"]),
                ]
                - logits[
                    int(cut["offset_zero_based"]),
                    int(cut["negative_option"]),
                ]
            )
            flat, _ = modules["geometry"].gradient_for_margin(
                margin,
                parameters,
                helper.torch,
                retain_graph=index + 1 < len(cuts),
            )
            gradient = np.asarray(flat, dtype=np.float64)
            observed_margin = float(margin.detach().cpu())
            snapshot_margin = pair_margin(
                expected,
                int(cut["offset_zero_based"]),
                int(cut["positive_option"]),
                int(cut["negative_option"]),
            )
            threshold = float(cut["threshold"])
            point_dot_gradient = float(gradient @ point)
            rhs = float(threshold - observed_margin + point_dot_gradient)
            actual_residual = float(observed_margin - threshold)
            tangent_residual_at_point = float(point_dot_gradient - rhs)
            threshold_sha = float64_scalar_sha256(threshold)
            gradient_sha = hashlib.sha256(
                np.ascontiguousarray(gradient.astype("<f8")).tobytes()
            ).hexdigest()
            core = {
                "semantic_identity": list(semantic_identity),
                "linearization_point_sha256": point_sha,
                "linearization_model_state_sha256":
                linearization_model_state_sha256,
                "linearization_output_state_sha256":
                linearization_output_state_sha256,
                "linearization_kind": linearization_kind,
                "threshold": threshold,
                "threshold_float64_le_sha256": threshold_sha,
                "observed_margin": observed_margin,
                "gradient_float64_le_sha256": gradient_sha,
                "point_dot_gradient": point_dot_gradient,
                "rhs": rhs,
                "actual_residual": actual_residual,
                "tangent_residual_at_point": tangent_residual_at_point,
            }
            formula_checks = {
                "gradient_shape_exact": gradient.shape
                == (ACTOR6_FLAT_LENGTH,),
                "gradient_finite": bool(np.isfinite(gradient).all()),
                "observed_threshold_dot_rhs_finite": all(
                    math.isfinite(value)
                    for value in (
                        observed_margin,
                        threshold,
                        point_dot_gradient,
                        rhs,
                    )
                ),
                "graph_margin_bit_equal_snapshot": observed_margin
                == snapshot_margin,
                "graph_margin_bit_equal_active_gate": observed_margin
                == float(expected_gate_margins[semantic_identity]),
                "rhs_exact_t_minus_m_plus_Gx": rhs
                == threshold - observed_margin + point_dot_gradient,
                "tangent_residual_matches_actual_residual": math.isclose(
                    tangent_residual_at_point,
                    actual_residual,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "candidate_tangent_point_is_actually_violated":
                not require_actual_violation or actual_residual < 0.0,
            }
            if not all(formula_checks.values()):
                raise RuntimeError(
                    f"CW15 affine tangent formula drift: {formula_checks}"
                )
            record = {
                **core,
                "formula_checks": formula_checks,
                "linearization_record_sha256": canonical_sha(
                    {**core, "formula_checks": formula_checks}
                ),
                "gradient_float64": gradient.copy(),
            }
            records.append(record)
        context_records.append(
            {
                "context_hash": context_hash,
                "context_type": cuts[0]["context_type"],
                "cut_count": len(cuts),
                "actual_batch_size": int(cpu_batch["targets"].shape[0]),
                "graph_checks": graph_checks,
            }
        )
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("CW15 official-context autograd materialized .grad")
    sanitized = [sanitized_affine_tangent(value) for value in records]
    audit_checks = {
        "point_checks_pass": all(point_checks.values()),
        "record_count_matches_semantic_cuts": len(records) == len(ordered),
        "affine_identity_unique": len(
            {affine_tangent_identity(value) for value in records}
        )
        == len(records),
        "all_formula_checks_pass": all(
            all(value["formula_checks"].values()) for value in records
        ),
        "all_candidate_tangent_residuals_negative_if_required":
        not require_actual_violation
        or all(value["tangent_residual_at_point"] < 0.0 for value in records),
    }
    if not all(audit_checks.values()):
        raise RuntimeError(f"CW15 tangent ledger audit failed: {audit_checks}")
    return records, {
        "checks": audit_checks,
        "pass": True,
        "linearization_kind": linearization_kind,
        "linearization_point_sha256": point_sha,
        "linearization_model_state_sha256": linearization_model_state_sha256,
        "linearization_output_fingerprint": live_output_fingerprint,
        "semantic_cut_count": len(ordered),
        "affine_record_count": len(records),
        "semantic_identity_sha256": canonical_sha(
            [list(cut_identity(value)) for value in ordered]
        ),
        "sanitized_affine_record_sha256": canonical_sha(sanitized),
        "contexts": context_records,
        "records": sanitized,
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


def restore_cw11_anchor_for_total_qp(
    context: Mapping[str, Any],
    modules: Mapping[str, ModuleType],
    streams: Mapping[str, Sequence[dict[str, Any]]],
    legacy: dict[str, Any],
    parameters: Sequence[Any],
    raw_actor: Sequence[Any],
    cw11_total: Any,
) -> dict[str, Any]:
    """Restore the fixed CW11 origin before solving an absolute-total QP."""

    import numpy as np

    zero_additional = np.zeros(ACTOR6_FLAT_LENGTH, dtype=np.float64)
    anchor_total = apply_additional_from_cw11(
        modules,
        parameters,
        raw_actor,
        cw11_total,
        zero_additional,
        context["helper"].torch,
    )
    official_batch_count = 0
    for view in VIEW_ORDER:
        for batch in streams[view]:
            if batch["reference_snapshot"] is None:
                raise RuntimeError("CW15 anchor restore lacks frozen reference snapshot")
            batch["candidate_snapshot"] = batch["reference_snapshot"]
            official_batch_count += 1
    if legacy["reference_snapshot"] is None:
        raise RuntimeError("CW15 anchor restore lacks legacy reference snapshot")
    legacy["candidate_snapshot"] = legacy["reference_snapshot"]
    checks = {
        "model_state_exact_CW11": context["helper"].model_state_sha256(
            context["model"].state_dict()
        )
        == CW11_MODEL_SHA256,
        "model_eval_exact": context["model"].training is False,
        "zero_additional_exact": bool(np.count_nonzero(zero_additional) == 0),
        "anchor_total_float64_bytes_exact": np.ascontiguousarray(
            anchor_total.astype("<f8")
        ).tobytes()
        == np.ascontiguousarray(cw11_total.astype("<f8")).tobytes(),
        "anchor_total_sha_exact": modules["geometry"].vector_sha256_float64_le(
            anchor_total, np
        )
        == CW11_VECTOR_SHA256,
        "nonactor_exact_raw": nonactor_sha256(context) == RAW_NONACTOR_SHA256,
        "all_official_candidate_snapshots_alias_frozen_reference": all(
            batch["candidate_snapshot"] is batch["reference_snapshot"]
            for view in VIEW_ORDER
            for batch in streams[view]
        ),
        "legacy_candidate_snapshot_alias_frozen_reference": legacy[
            "candidate_snapshot"
        ]
        is legacy["reference_snapshot"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW15 exact CW11 anchor restore failed: {checks}")
    return {
        "anchor_model": "frozen_CW11_4317",
        "anchor_model_state_sha256": CW11_MODEL_SHA256,
        "anchor_total_float64_le_sha256": CW11_VECTOR_SHA256,
        "official_batch_count": official_batch_count,
        "legacy_context_count": 1,
        "checks": checks,
        "pass": True,
    }


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
        raise RuntimeError("CW15 incoming CW11 consumer schema drift")
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
        "CW15_semantic_caps_unchanged_equal_0p001": STEP_L2_CAP
        == ADDITIONAL_TOTAL_L2_CAP
        == 0.001,
        "fixed_repairs_initially_fail": not initial_fixed["pass"],
        "iteration0_has_no_dynamic_fixed_cuts": initial_fixed[
            "deterministic_repair_cut_count"
        ]
        == 0,
        "legacy_reference_pass": reference_legacy_gate["pass"],
    }
    if not all(initial_checks.values()):
        raise RuntimeError(f"CW15 initial-state audit failed: {initial_checks}")

    active: list[dict[str, Any]] = []
    legacy_merge = merge_active_cuts(active, legacy_pair_cuts(legacy))
    fixed_merge = merge_active_cuts(active, initial_official_repair_cuts(streams))
    if (
        legacy_merge["active_count"] != 34
        or fixed_merge["active_count"] != 38
        or len({cut_identity(value) for value in active}) != 38
    ):
        raise RuntimeError("CW15 initial 34+4 semantic cut ledger drift")
    initial_active_gates = active_cut_gates(
        active, streams, legacy, modules["cutting"].PAIR_THRESHOLD_TOLERANCE
    )
    if initial_active_gates["violated_count"] != 4:
        raise RuntimeError("CW15 initial fixed repair violation count drift")

    additional = np.zeros(ACTOR6_FLAT_LENGTH, dtype=np.float64)
    if float64_vector_sha256(additional) != CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256:
        raise RuntimeError("CW15 zero additional point SHA drift")
    current_total = cw11_total.copy()
    initial_output_fingerprint = candidate_output_state_fingerprint(
        streams, legacy, torch
    )
    iterations = [
        {
            "iteration": 0,
            "kind": "exact_CW11_reference_before_fixed3_repair",
            "model_state_sha256": CW11_MODEL_SHA256,
            "output_state_fingerprint": initial_output_fingerprint,
            "additional_l2": 0.0,
            "additional_float64_le_sha256":
            CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
            "total_from_raw_l2": float(np.linalg.norm(cw11_total)),
            "active_cut_count": len(active),
            "active_cut_identity_sha256": canonical_sha(
                [list(cut_identity(value)) for value in active]
            ),
            "optimization_affine_tangent_count": 0,
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
    affine_ledger: list[dict[str, Any]] = []
    prior_qp_affine_identities: set[tuple[Any, ...]] = set()
    seen_point_to_model: dict[str, str] = {}
    seen_model_to_point: dict[str, str] = {}
    seen_semantic_states: set[tuple[str, str, str]] = set()
    previous_proposal_model_sha = None
    previous_proposal_output_sha = None
    for iteration in range(1, MAX_OUTER_ITERATIONS + 1):
        anchor_audit = None
        anchor_linearization_audit = None
        anchor_affine_merge = None
        affine_qp_audit = None
        try:
            previous_additional = additional.copy()
            current_active_identities = {cut_identity(value) for value in active}
            anchor_audit = restore_cw11_anchor_for_total_qp(
                context,
                modules,
                streams,
                legacy,
                parameters,
                raw_actor,
                cw11_total,
            )
            anchor_output_fingerprint = candidate_output_state_fingerprint(
                streams, legacy, torch
            )
            anchor_gates = active_cut_gates(
                active,
                streams,
                legacy,
                modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
            )
            anchor_gate_margin_by_identity = {
                tuple(value["identity"]): float(value["margin"])
                for value in anchor_gates["records"]
            }
            existing_affine_keys = {
                affine_tangent_identity(value) for value in affine_ledger
            }
            missing_anchor_cuts = [
                value
                for value in active
                if (
                    *cut_identity(value),
                    CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
                    CW11_MODEL_SHA256,
                    float64_scalar_sha256(float(value["threshold"])),
                )
                not in existing_affine_keys
            ]
            if missing_anchor_cuts:
                missing_anchor_margins = {
                    cut_identity(value): anchor_gate_margin_by_identity[
                        cut_identity(value)
                    ]
                    for value in missing_anchor_cuts
                }
                anchor_records, anchor_linearization_audit = (
                    linearize_semantic_cuts_at_current_point(
                        context,
                        modules,
                        streams,
                        legacy,
                        missing_anchor_cuts,
                        parameters,
                        np.zeros(ACTOR6_FLAT_LENGTH, dtype=np.float64),
                        linearization_model_state_sha256=CW11_MODEL_SHA256,
                        linearization_output_state_sha256=anchor_output_fingerprint[
                            "sha256"
                        ],
                        linearization_kind="CW11_anchor",
                        expected_gate_margins=missing_anchor_margins,
                        require_actual_violation=False,
                    )
                )
                anchor_affine_merge = merge_affine_tangents(
                    affine_ledger, anchor_records
                )
            else:
                anchor_linearization_audit = {
                    "pass": True,
                    "semantic_cut_count": 0,
                    "reason": "all_current_semantic_thresholds_already_have_CW11_anchor_tangent",
                }
                anchor_affine_merge = {
                    "added": [],
                    "repeated": [],
                    "added_count": 0,
                    "repeated_count": 0,
                    "affine_count": len(affine_ledger),
                    "affine_ledger_sha256": canonical_sha(
                        [
                            sanitized_affine_tangent(value)
                            for value in sorted(
                                affine_ledger, key=affine_tangent_identity
                            )
                        ]
                    ),
                    "strict_growth": False,
                }
            anchor_coverage = {
                (
                    *cut_identity(value),
                    CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
                    CW11_MODEL_SHA256,
                    float64_scalar_sha256(float(value["threshold"])),
                )
                for value in active
            }.issubset({affine_tangent_identity(value) for value in affine_ledger})
            anchor_tangent_checks = {
                "missing_anchor_cut_identity_unique": len(
                    {cut_identity(value) for value in missing_anchor_cuts}
                )
                == len(missing_anchor_cuts),
                "new_anchor_tangent_one_to_one": anchor_affine_merge[
                    "added_count"
                ]
                == len(missing_anchor_cuts),
                "new_anchor_tangent_has_no_repeat": anchor_affine_merge[
                    "repeated_count"
                ]
                == 0,
                "all_current_thresholds_have_CW11_anchor_tangent":
                anchor_coverage,
                "anchor_output_matches_initial_CW11": anchor_output_fingerprint[
                    "sha256"
                ]
                == initial_output_fingerprint["sha256"],
            }
            if not all(anchor_tangent_checks.values()):
                raise RuntimeError(
                    f"CW15 anchor tangent coverage drift: {anchor_tangent_checks}"
                )
            gradients, rhs, affine_qp_audit = affine_qp_arrays(affine_ledger)
            ordered_affine_for_qp = sorted(
                affine_ledger, key=affine_tangent_identity
            )
            if len(ordered_affine_for_qp) != int(gradients.shape[0]):
                raise RuntimeError("CW15 affine QP row/ledger cardinality drift")
            current_qp_affine_identities = {
                affine_tangent_identity(value) for value in affine_ledger
            }
            affine_cut_set_nested = prior_qp_affine_identities.issubset(
                current_qp_affine_identities
            )
            prior_qp_affine_identities = set(current_qp_affine_identities)
            anchored_solution, qp = modules["cutting"].solve_minimum_l2_correction(
                gradients, rhs, np, optimize
            )
            qp = {
                **qp,
                "CW15_objective": (
                    "minimum total additional from exact CW11 origin over the "
                    "monotone sequential affine tangent ledger"
                ),
                "semantic_thresholds_unchanged_from_CW14": True,
                "affine_tangent_count": len(affine_ledger),
                "affine_ledger_sha256": affine_qp_audit[
                    "affine_ledger_sha256"
                ],
            }
            if not (
                float(qp["trust_region_l2_cap"])
                == STEP_L2_CAP
                == ADDITIONAL_TOTAL_L2_CAP
                == 0.001
            ):
                raise RuntimeError("CW15 frozen QP/total cap binding drift")
        except Exception as exc:
            close_reason = f"fail_closed_affine_or_QP:{type(exc).__name__}:{exc}"
            iterations.append(
                {
                    "iteration": iteration,
                    "kind": "fail_closed_before_proposal",
                    "close_reason": close_reason,
                    "active_cut_count": len(active),
                    "optimization_affine_tangent_count": len(affine_ledger),
                    "anchor_audit": anchor_audit,
                    "anchor_linearization_audit": anchor_linearization_audit,
                    "anchor_affine_merge": anchor_affine_merge,
                    "affine_qp_audit": affine_qp_audit,
                }
            )
            break
        anchored_minimum_total_l2 = float(qp["uncapped_l2"])
        reported_applied_l2 = float(qp["applied_l2"])
        anchored_solution_vector_l2 = float(np.linalg.norm(anchored_solution))
        direct_uncapped_residual_min = float(qp["direct_uncapped_residual_min"])
        if bool(qp["capped"]):
            qp_group_residual_audit = {
                "pass": True,
                "checked": False,
                "reason": "capped_uncapped_solution_not_returned_and_will_close_before_apply",
                "combined": None,
                "CW11_anchor": None,
                "candidate_tangent": None,
            }
        else:
            direct_residuals = np.asarray(
                gradients @ anchored_solution - rhs, dtype=np.float64
            )

            def residual_group(kind: str, selector: Any) -> dict[str, Any]:
                selected = direct_residuals[selector]
                residual_min = (
                    None if selected.size == 0 else float(selected.min())
                )
                return {
                    "kind": kind,
                    "row_count": int(selected.size),
                    "residual_min": residual_min,
                    "finite": bool(np.isfinite(selected).all()),
                    "direct_residual_gate": selected.size == 0
                    or residual_min >= -QP_DIRECT_RESIDUAL_ABS_TOL,
                }

            anchor_selector = np.asarray(
                [
                    value["linearization_kind"] == "CW11_anchor"
                    for value in ordered_affine_for_qp
                ],
                dtype=bool,
            )
            candidate_selector = np.logical_not(anchor_selector)
            combined_group = residual_group(
                "combined", np.ones(len(ordered_affine_for_qp), dtype=bool)
            )
            anchor_group = residual_group("CW11_anchor", anchor_selector)
            candidate_group = residual_group(
                "candidate_tangent", candidate_selector
            )
            group_checks = {
                "partition_exact": int(anchor_selector.sum())
                + int(candidate_selector.sum())
                == len(ordered_affine_for_qp),
                "anchor_rows_present": int(anchor_selector.sum()) > 0,
                "combined_min_matches_frozen_solver": math.isclose(
                    float(combined_group["residual_min"]),
                    direct_uncapped_residual_min,
                    rel_tol=0.0,
                    abs_tol=L2_CAP_ABS_TOL,
                ),
                "combined_direct_residual_gate": combined_group[
                    "direct_residual_gate"
                ],
                "CW11_anchor_direct_residual_gate": anchor_group[
                    "direct_residual_gate"
                ],
                "candidate_tangent_direct_residual_gate": candidate_group[
                    "direct_residual_gate"
                ],
                "all_group_residuals_finite": combined_group["finite"]
                and anchor_group["finite"]
                and candidate_group["finite"],
            }
            qp_group_residual_audit = {
                "pass": all(group_checks.values()),
                "checked": True,
                "absolute_tolerance": QP_DIRECT_RESIDUAL_ABS_TOL,
                "checks": group_checks,
                "combined": combined_group,
                "CW11_anchor": anchor_group,
                "candidate_tangent": candidate_group,
            }
        qp_checks = {
            "solver_success": qp["solver"]["success"] is True,
            "solver_status_zero": int(qp["solver"]["status"]) == 0,
            "uncapped_l2_finite": math.isfinite(anchored_minimum_total_l2),
            "reported_applied_l2_finite": math.isfinite(reported_applied_l2),
            "direct_uncapped_residual_finite": math.isfinite(
                direct_uncapped_residual_min
            ),
            "direct_uncapped_residual_gate": direct_uncapped_residual_min
            >= -QP_DIRECT_RESIDUAL_ABS_TOL,
            "anchored_solution_vector_l2_matches_reported_applied": math.isclose(
                anchored_solution_vector_l2,
                reported_applied_l2,
                rel_tol=0.0,
                abs_tol=L2_CAP_ABS_TOL,
            ),
            "reported_applied_l2_equals_uncapped_l2": math.isclose(
                reported_applied_l2,
                anchored_minimum_total_l2,
                rel_tol=0.0,
                abs_tol=L2_CAP_ABS_TOL,
            ),
            "affine_qp_audit_pass": bool(affine_qp_audit["pass"]),
            "anchor_candidate_and_combined_direct_residuals_pass": bool(
                qp_group_residual_audit["pass"]
            ),
        }
        if bool(qp["capped"]) or (
            anchored_minimum_total_l2
            > ADDITIONAL_TOTAL_L2_CAP + L2_CAP_ABS_TOL
        ):
            close_reason = "fail_closed_anchored_minimum_total_exceeds_cap"
            iterations.append(
                {
                    "iteration": iteration,
                    "kind": "fail_closed_before_proposal",
                    "close_reason": close_reason,
                    "active_cut_count": len(active),
                    "optimization_affine_tangent_count": len(affine_ledger),
                    "anchor_audit": anchor_audit,
                    "anchor_linearization_audit": anchor_linearization_audit,
                    "anchor_affine_merge": anchor_affine_merge,
                    "affine_qp_audit": affine_qp_audit,
                    "qp": qp,
                    "qp_group_residual_audit": qp_group_residual_audit,
                    "qp_checks": qp_checks,
                    "anchored_minimum_total_l2": anchored_minimum_total_l2,
                    "total_cap": ADDITIONAL_TOTAL_L2_CAP,
                    "clipped_vector_applied": False,
                }
            )
            break
        if not all(qp_checks.values()):
            close_reason = "fail_closed_anchored_QP_integrity_gate"
            iterations.append(
                {
                    "iteration": iteration,
                    "kind": "fail_closed_before_proposal",
                    "close_reason": close_reason,
                    "active_cut_count": len(active),
                    "optimization_affine_tangent_count": len(affine_ledger),
                    "anchor_audit": anchor_audit,
                    "anchor_linearization_audit": anchor_linearization_audit,
                    "anchor_affine_merge": anchor_affine_merge,
                    "affine_qp_audit": affine_qp_audit,
                    "qp": qp,
                    "qp_group_residual_audit": qp_group_residual_audit,
                    "qp_checks": qp_checks,
                    "clipped_vector_applied": False,
                }
            )
            break
        proposal_additional = np.asarray(
            anchored_solution, dtype=np.float64
        ).copy()
        proposal_point_sha = float64_vector_sha256(proposal_additional)
        proposal_additional_l2 = float(np.linalg.norm(proposal_additional))
        previous_additional_l2 = float(np.linalg.norm(previous_additional))
        previous_to_proposal_delta = np.subtract(
            proposal_additional, previous_additional, dtype=np.float64
        )
        previous_to_proposal_delta_l2 = float(
            np.linalg.norm(previous_to_proposal_delta)
        )
        projection_dot = float(previous_additional @ previous_to_proposal_delta)
        geometry_checks = {
            "affine_cut_set_nested": affine_cut_set_nested,
            "minimum_total_norm_monotone_nondecreasing":
            proposal_additional_l2 + L2_CAP_ABS_TOL >= previous_additional_l2,
            "nested_projection_dot_nonnegative": projection_dot
            >= -L2_CAP_ABS_TOL,
            "applied_equals_uncapped_minimum_total": not bool(qp["capped"])
            and math.isclose(
                proposal_additional_l2,
                anchored_minimum_total_l2,
                rel_tol=0.0,
                abs_tol=L2_CAP_ABS_TOL,
            ),
            "previous_to_proposal_delta_finite": math.isfinite(
                previous_to_proposal_delta_l2
            ),
            "previous_to_proposal_delta_at_most_step_cap":
            previous_to_proposal_delta_l2
            <= STEP_L2_CAP + L2_CAP_ABS_TOL,
            "proposal_point_sha_hex64": len(proposal_point_sha) == 64,
        }
        cap_checks = {
            "frozen_solver_cap_equals_total_cap": float(
                qp["trust_region_l2_cap"]
            )
            == STEP_L2_CAP
            == ADDITIONAL_TOTAL_L2_CAP,
            "anchored_minimum_total_positive_finite": math.isfinite(
                anchored_minimum_total_l2
            )
            and anchored_minimum_total_l2 > 0.0,
            "anchored_minimum_total_at_most_cap": anchored_minimum_total_l2
            <= ADDITIONAL_TOTAL_L2_CAP + L2_CAP_ABS_TOL,
            "additional_total_finite": math.isfinite(proposal_additional_l2),
            "additional_total_at_most_cap": proposal_additional_l2
            <= ADDITIONAL_TOTAL_L2_CAP + L2_CAP_ABS_TOL,
            "solver_did_not_clip": qp["capped"] is False,
            "previous_to_proposal_delta_at_most_step_cap": geometry_checks[
                "previous_to_proposal_delta_at_most_step_cap"
            ],
        }
        if not all(cap_checks.values()) or not all(geometry_checks.values()):
            close_reason = "fail_closed_anchored_total_geometry_or_cap_gate"
            iterations.append(
                {
                    "iteration": iteration,
                    "kind": "fail_closed_before_proposal",
                    "close_reason": close_reason,
                    "cap_checks": cap_checks,
                    "geometry_checks": geometry_checks,
                    "anchor_audit": anchor_audit,
                    "anchor_linearization_audit": anchor_linearization_audit,
                    "anchor_affine_merge": anchor_affine_merge,
                    "affine_qp_audit": affine_qp_audit,
                    "qp": qp,
                    "qp_group_residual_audit": qp_group_residual_audit,
                    "anchored_minimum_total_l2": anchored_minimum_total_l2,
                    "previous_additional_l2": previous_additional_l2,
                    "previous_to_proposal_delta_l2": previous_to_proposal_delta_l2,
                    "projection_dot": projection_dot,
                    "proposal_additional_l2": proposal_additional_l2,
                    "proposal_additional_float64_le_sha256": proposal_point_sha,
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
        prior_model_for_point = seen_point_to_model.get(proposal_point_sha)
        prior_point_for_model = seen_model_to_point.get(proposal_model_sha)
        pre_snapshot_stagnation = {
            "exact_x_and_model_state_repeated": prior_model_for_point
            == proposal_model_sha,
            "x_repeated_with_different_model_nondeterminism":
            prior_model_for_point is not None
            and prior_model_for_point != proposal_model_sha,
            "x_changed_but_model_state_same": prior_point_for_model is not None
            and prior_point_for_model != proposal_point_sha,
        }
        additional = proposal_additional
        current_total = proposal_total
        if any(pre_snapshot_stagnation.values()):
            close_reason = "fail_closed_repeated_or_noninjective_x_model_state"
            iterations.append(
                {
                    "iteration": iteration,
                    "kind": "fail_closed_stagnant_reapplication_before_oracle",
                    "close_reason": close_reason,
                    "model_state_sha256": proposal_model_sha,
                    "additional_l2": proposal_additional_l2,
                    "additional_float64_le_sha256": proposal_point_sha,
                    "pre_snapshot_stagnation": pre_snapshot_stagnation,
                    "prior_model_for_point": prior_model_for_point,
                    "prior_point_for_model": prior_point_for_model,
                    "optimization_affine_tangent_count": len(affine_ledger),
                    "affine_qp_audit": affine_qp_audit,
                    "qp": qp,
                    "clipped_vector_applied": False,
                }
            )
            break
        seen_point_to_model[proposal_point_sha] = proposal_model_sha
        seen_model_to_point[proposal_model_sha] = proposal_point_sha

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
        candidate_output_fingerprint = candidate_output_state_fingerprint(
            streams, legacy, torch
        )
        bf16_output_plateau_allowed = bool(
            previous_proposal_output_sha is not None
            and candidate_output_fingerprint["sha256"]
            == previous_proposal_output_sha
            and proposal_model_sha != previous_proposal_model_sha
        )
        oracle = full_stream_separation_oracle(
            helper, streams, forensic, guardplan, prereg, iteration=iteration
        )
        fixed = fixed_repair_gates_and_cuts(helper, streams, iteration=iteration)
        active_gates_before_merge = active_cut_gates(
            active, streams, legacy, modules["cutting"].PAIR_THRESHOLD_TOLERANCE
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
        semantic_merge = merge_active_cuts(active, additions)
        post_merge_active_gates = active_cut_gates(
            active, streams, legacy, modules["cutting"].PAIR_THRESHOLD_TOLERANCE
        )
        active_by_identity = {cut_identity(value): value for value in active}
        violated_gate_records = [
            value
            for value in post_merge_active_gates["records"]
            if value["pass"] is False
        ]
        violated_identities = {
            tuple(value["identity"]) for value in violated_gate_records
        }
        violated_semantic_cuts = [
            active_by_identity[value] for value in sorted(violated_identities)
        ]
        violated_margin_by_identity = {
            tuple(value["identity"]): float(value["margin"])
            for value in violated_gate_records
        }
        violated_set_sha = canonical_sha(
            [list(value) for value in sorted(violated_identities)]
        )
        semantic_state = (
            proposal_model_sha,
            candidate_output_fingerprint["sha256"],
            violated_set_sha,
        )
        semantic_state_repeated = semantic_state in seen_semantic_states
        if not semantic_state_repeated:
            seen_semantic_states.add(semantic_state)

        candidate_linearization_audit = None
        candidate_affine_merge = {
            "added": [],
            "repeated": [],
            "added_count": 0,
            "repeated_count": 0,
            "affine_count": len(affine_ledger),
            "affine_ledger_sha256": canonical_sha(
                [
                    sanitized_affine_tangent(value)
                    for value in sorted(affine_ledger, key=affine_tangent_identity)
                ]
            ),
            "strict_growth": False,
        }
        if violated_semantic_cuts and not semantic_state_repeated:
            candidate_records, candidate_linearization_audit = (
                linearize_semantic_cuts_at_current_point(
                    context,
                    modules,
                    streams,
                    legacy,
                    violated_semantic_cuts,
                    parameters,
                    proposal_additional,
                    linearization_model_state_sha256=proposal_model_sha,
                    linearization_output_state_sha256=
                    candidate_output_fingerprint["sha256"],
                    linearization_kind="post_merge_actual_active_violation",
                    expected_gate_margins=violated_margin_by_identity,
                    require_actual_violation=True,
                )
            )
            candidate_affine_merge = merge_affine_tangents(
                affine_ledger, candidate_records
            )
        elif not violated_semantic_cuts:
            candidate_linearization_audit = {
                "pass": True,
                "semantic_cut_count": 0,
                "reason": "post_merge_active_violation_count_zero",
            }
        candidate_added_semantic_identities = {
            tuple(value["semantic_identity"])
            for value in candidate_affine_merge["added"]
        }
        candidate_tangent_checks = {
            "post_merge_false_rows_identity_unique": len(violated_identities)
            == len(violated_gate_records),
            "false_rows_resolve_against_full_semantic_ledger": len(
                violated_semantic_cuts
            )
            == len(violated_gate_records),
            "candidate_tangent_semantic_identity_bijection":
            candidate_added_semantic_identities == violated_identities
            if violated_semantic_cuts and not semantic_state_repeated
            else not candidate_added_semantic_identities,
            "candidate_tangent_count_bijection": candidate_affine_merge[
                "added_count"
            ]
            == len(violated_gate_records)
            if violated_semantic_cuts and not semantic_state_repeated
            else candidate_affine_merge["added_count"] == 0,
            "candidate_tangent_no_repeat": candidate_affine_merge[
                "repeated_count"
            ]
            == 0,
            "nonterminal_actual_violations_force_strict_affine_growth":
            not violated_semantic_cuts
            or semantic_state_repeated
            or candidate_affine_merge["strict_growth"],
            "all_violations_not_only_oracle_additions_are_tangented":
            not violated_semantic_cuts
            or semantic_state_repeated
            or candidate_added_semantic_identities == violated_identities,
        }
        post_merge_anchor_audit = {
            "required": False,
            "pass": True,
            "missing_semantic_cut_count": 0,
            "reason": "no_new_or_strengthened_semantic_threshold",
        }
        post_merge_anchor_affine_merge = {
            "added": [],
            "repeated": [],
            "added_count": 0,
            "repeated_count": 0,
            "affine_count": len(affine_ledger),
            "affine_ledger_sha256": canonical_sha(
                [
                    sanitized_affine_tangent(value)
                    for value in sorted(affine_ledger, key=affine_tangent_identity)
                ]
            ),
            "strict_growth": False,
        }
        semantic_changed = bool(
            semantic_merge["added"] or semantic_merge["strengthened"]
        )
        if semantic_changed:
            semantic_change_nonterminal_checks = {
                "oracle_or_fixed_or_legacy_not_terminal":
                not oracle["authoritative_60_gates"]["pass"]
                or int(oracle["new_harm_count_vs_cw11"]) != 0
                or not oracle["transition_state_checks"]
                ["favorable_transitions_retained"]
                or not fixed["pass"]
                or not legacy_gate["pass"]
                or not post_merge_active_gates["pass"],
                "semantic_merge_really_changed": semantic_changed,
            }
            if not all(semantic_change_nonterminal_checks.values()):
                raise RuntimeError(
                    "CW15 semantic change unexpectedly compatible with terminal"
                )
            post_anchor_restore = restore_cw11_anchor_for_total_qp(
                context,
                modules,
                streams,
                legacy,
                parameters,
                raw_actor,
                cw11_total,
            )
            post_anchor_output_fingerprint = candidate_output_state_fingerprint(
                streams, legacy, torch
            )
            post_anchor_gates = active_cut_gates(
                active,
                streams,
                legacy,
                modules["cutting"].PAIR_THRESHOLD_TOLERANCE,
            )
            post_anchor_margin_by_identity = {
                tuple(value["identity"]): float(value["margin"])
                for value in post_anchor_gates["records"]
            }
            affine_keys_after_candidate = {
                affine_tangent_identity(value) for value in affine_ledger
            }
            missing_post_merge_anchor_cuts = [
                value
                for value in active
                if (
                    *cut_identity(value),
                    CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
                    CW11_MODEL_SHA256,
                    float64_scalar_sha256(float(value["threshold"])),
                )
                not in affine_keys_after_candidate
            ]
            missing_post_anchor_margins = {
                cut_identity(value): post_anchor_margin_by_identity[
                    cut_identity(value)
                ]
                for value in missing_post_merge_anchor_cuts
            }
            post_anchor_records, post_anchor_linearization = (
                linearize_semantic_cuts_at_current_point(
                    context,
                    modules,
                    streams,
                    legacy,
                    missing_post_merge_anchor_cuts,
                    parameters,
                    np.zeros(ACTOR6_FLAT_LENGTH, dtype=np.float64),
                    linearization_model_state_sha256=CW11_MODEL_SHA256,
                    linearization_output_state_sha256=
                    post_anchor_output_fingerprint["sha256"],
                    linearization_kind="CW11_anchor",
                    expected_gate_margins=missing_post_anchor_margins,
                    require_actual_violation=False,
                )
            )
            post_merge_anchor_affine_merge = merge_affine_tangents(
                affine_ledger, post_anchor_records
            )
            post_merge_anchor_checks = {
                "restore_exact_CW11": bool(post_anchor_restore["pass"]),
                "output_fingerprint_exact_initial_CW11":
                post_anchor_output_fingerprint["sha256"]
                == initial_output_fingerprint["sha256"],
                "missing_count_positive": len(missing_post_merge_anchor_cuts) > 0,
                "new_anchor_tangent_one_to_one": post_merge_anchor_affine_merge[
                    "added_count"
                ]
                == len(missing_post_merge_anchor_cuts),
                "new_anchor_tangent_no_repeat": post_merge_anchor_affine_merge[
                    "repeated_count"
                ]
                == 0,
                "all_current_semantics_now_have_anchor": {
                    (
                        *cut_identity(value),
                        CW11_ZERO_ADDITIONAL_FLOAT64_LE_SHA256,
                        CW11_MODEL_SHA256,
                        float64_scalar_sha256(float(value["threshold"])),
                    )
                    for value in active
                }.issubset(
                    {affine_tangent_identity(value) for value in affine_ledger}
                ),
            }
            if not all(post_merge_anchor_checks.values()):
                raise RuntimeError(
                    f"CW15 post-merge anchor tangent drift: {post_merge_anchor_checks}"
                )
            post_merge_anchor_audit = {
                "required": True,
                "pass": True,
                "missing_semantic_cut_count": len(
                    missing_post_merge_anchor_cuts
                ),
                "semantic_change_nonterminal_checks":
                semantic_change_nonterminal_checks,
                "restore": post_anchor_restore,
                "output_fingerprint": post_anchor_output_fingerprint,
                "linearization": post_anchor_linearization,
                "checks": post_merge_anchor_checks,
            }
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
            "native_BF16_six_outputs": candidate_output_fingerprint[
                "native_bf16_exact"
            ],
            "exact_CW11_anchor_before_total_QP": bool(anchor_audit["pass"]),
            "all_QP_semantics_have_current_threshold_CW11_anchor_tangent":
            anchor_tangent_checks[
                "all_current_thresholds_have_CW11_anchor_tangent"
            ],
            "optimization_affine_ledger_nested": affine_cut_set_nested,
            "post_merge_actual_violation_tangent_bijection": all(
                candidate_tangent_checks.values()
            ),
            "new_or_strengthened_semantics_have_immediate_CW11_anchor_tangent":
            bool(post_merge_anchor_audit["pass"]),
            "anchored_minimum_total_L2_cap": cap_checks[
                "anchored_minimum_total_at_most_cap"
            ],
            "additional_total_L2_cap": cap_checks["additional_total_at_most_cap"],
            "nested_minimum_total_geometry": all(geometry_checks.values()),
            "uncapped_solution_only_never_clip": cap_checks[
                "solver_did_not_clip"
            ],
            "anchored_QP_integrity": all(qp_checks.values()),
            "model_sha_hex64": len(proposal_model_sha) == 64,
            "output_state_sha_hex64": len(
                candidate_output_fingerprint["sha256"]
            )
            == 64,
            "no_checkpoint_written_eval_only_13key_deferred": True,
        }
        acceptance = terminal_acceptance(
            snapshot=candidate_six,
            oracle=oracle,
            fixed=fixed,
            legacy_gate=legacy_gate,
            active_gates=post_merge_active_gates,
            integrity=integrity,
        )
        iteration_record = {
            "iteration": iteration,
            "kind": "unique_fixed_order_official_context_proposal",
            "model_state_sha256": proposal_model_sha,
            "output_state_fingerprint": candidate_output_fingerprint,
            "anchor_audit": anchor_audit,
            "anchor_linearization_audit": anchor_linearization_audit,
            "anchor_affine_merge": anchor_affine_merge,
            "anchor_tangent_checks": anchor_tangent_checks,
            "affine_qp_audit": affine_qp_audit,
            "qp": qp,
            "qp_group_residual_audit": qp_group_residual_audit,
            "qp_checks": qp_checks,
            "anchored_minimum_total_l2": anchored_minimum_total_l2,
            "previous_additional_l2": previous_additional_l2,
            "previous_to_proposal_delta_l2": previous_to_proposal_delta_l2,
            "previous_to_proposal_projection_dot": projection_dot,
            "additional_l2": proposal_additional_l2,
            "additional_float64_le_sha256": proposal_point_sha,
            "total_from_raw_l2": float(np.linalg.norm(proposal_total)),
            "total_from_raw_float64_le_sha256": modules[
                "geometry"
            ].vector_sha256_float64_le(proposal_total, np),
            "cap_checks": cap_checks,
            "geometry_checks": geometry_checks,
            "qp_cap_audit": {
                "capped_record_only": bool(qp["capped"]),
                "clipped_vector_applied": False,
                "uncapped_over_cap_closes_before_proposal": True,
                "absolute_tolerance": L2_CAP_ABS_TOL,
            },
            "stagnation_audit": {
                "pre_snapshot": pre_snapshot_stagnation,
                "BF16_output_plateau_with_changed_model_allowed":
                bf16_output_plateau_allowed,
                "semantic_state_key": list(semantic_state),
                "semantic_state_repeated": semantic_state_repeated,
            },
            "legacy_gate": legacy_gate,
            "six_view_snapshot": candidate_six,
            "oracle": oracle,
            "fixed_repair_gates": fixed,
            "active_cut_gates_before_merge": active_gates_before_merge,
            "post_oracle_cut_merge": semantic_merge,
            "active_cut_gates": post_merge_active_gates,
            "candidate_linearization_audit": candidate_linearization_audit,
            "post_merge_candidate_affine_tangent_merge":
            candidate_affine_merge,
            "candidate_tangent_checks": candidate_tangent_checks,
            "post_merge_new_semantic_anchor_audit": post_merge_anchor_audit,
            "post_merge_new_semantic_anchor_affine_merge":
            post_merge_anchor_affine_merge,
            "optimization_affine_tangent_count_after_candidate_and_new_anchor": len(
                affine_ledger
            ),
            "optimization_affine_ledger_sha256_after_candidate_and_new_anchor":
            post_merge_anchor_affine_merge["affine_ledger_sha256"],
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
        previous_proposal_model_sha = proposal_model_sha
        previous_proposal_output_sha = candidate_output_fingerprint["sha256"]
        if acceptance["pass"]:
            if additions or violated_semantic_cuts:
                raise RuntimeError(
                    "accepted terminal unexpectedly produced semantic cuts or violations"
                )
            terminal_iteration = iteration
            close_reason = "first_feasible_fixed_iteration_candidate"
            terminal_oracle = oracle
            terminal_active_gates = post_merge_active_gates
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
        if semantic_state_repeated:
            close_reason = "fail_closed_repeated_model_output_violation_state"
            iteration_record["close_reason"] = close_reason
            break
        if violated_semantic_cuts and not candidate_affine_merge["strict_growth"]:
            close_reason = "fail_closed_actual_violation_without_new_tangent"
            iteration_record["close_reason"] = close_reason
            break
        if not violated_semantic_cuts and not (
            semantic_merge["added"] or semantic_merge["strengthened"]
        ):
            close_reason = "fail_closed_nonfeasible_without_separating_cut"
            iteration_record["close_reason"] = close_reason
            break

    success = terminal_iteration is not None
    terminal_active_cut_ledger = copy.deepcopy(active) if success else None
    terminal_active_cut_ledger_sha256 = (
        canonical_sha(terminal_active_cut_ledger) if success else None
    )
    optimization_affine_closure_ledger = [
        sanitized_affine_tangent(value)
        for value in sorted(affine_ledger, key=affine_tangent_identity)
    ]
    optimization_affine_closure_ledger_sha256 = canonical_sha(
        optimization_affine_closure_ledger
    )
    terminal_optimization_affine_ledger = (
        copy.deepcopy(optimization_affine_closure_ledger) if success else None
    )
    terminal_optimization_affine_ledger_sha256 = (
        optimization_affine_closure_ledger_sha256 if success else None
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
                "optimization_affine_tangent_ledger": copy.deepcopy(
                    terminal_optimization_affine_ledger
                ),
                "optimization_affine_tangent_ledger_sha256":
                terminal_optimization_affine_ledger_sha256,
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
            "consumed_valid_CW15_sequential_affine_tangent_closure_first_feasible"
            if success
            else "closed_no_CW15_candidate"
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
            "optimization_affine_tangent_count": len(affine_ledger),
            "optimization_affine_tangent_ledger_sha256":
            optimization_affine_closure_ledger_sha256,
            "candidate_consumer_called": candidate_consumer_called,
            "eligible_only_for_formal_fulltrain_revalidation": success,
            "eligible_as_promotion_evidence": False,
        },
        "terminal_reconstruction_payload": terminal_payload,
        "terminal_reconstruction_audit": terminal_reconstruction_audit,
        "terminal_active_cut_ledger": terminal_active_cut_ledger,
        "terminal_active_cut_ledger_sha256": terminal_active_cut_ledger_sha256,
        "optimization_affine_closure_ledger":
        optimization_affine_closure_ledger,
        "optimization_affine_closure_ledger_sha256":
        optimization_affine_closure_ledger_sha256,
        "terminal_optimization_affine_ledger":
        terminal_optimization_affine_ledger,
        "terminal_optimization_affine_ledger_sha256":
        terminal_optimization_affine_ledger_sha256,
        "downstream_contract": {
            "formal_must_not_call_official6_solver_again": True,
            "formal_reconstructs_from_payload_without_specialist_access": True,
            "formal_uses_original_raw_U468_checkpoint_template": True,
            "formal_then_runs_one_frozen_fulltrain_revalidation_only": True,
            "broad_requires_new_CW15_hash_bound_one_shot_runner": True,
        },
    }


def run_cw15_consumer(
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
                "CW15 local finally failed to restore CW11 or matmul precision"
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
    direct_cw13_closure_evidence = validate_cw13_closure_evidence()
    direct_cw14_closure_evidence = validate_cw14_closure_evidence()
    direct_anchored_total_math = anchored_total_geometry_self_test()
    direct_sequential_affine_math = sequential_affine_tangent_self_test()
    direct_affine_ledger_math = affine_ledger_integrity_self_test()
    direct_forensic_contract = validate_forensic_contract()
    direct_runtime = validate_runtime(require_cuda=True)
    modules = frozen_modules()
    # Use the caller-supplied primary object throughout the live patch chain.
    modules["primary"] = primary
    holder: dict[str, Any] = {}

    def internal_consumer(context: Mapping[str, Any]) -> None:
        if holder:
            raise RuntimeError("CW15 live consumer called more than once")
        holder.update(
            run_cw15_consumer(
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
        raise RuntimeError("CW15 consumer was not reached by exact CW11 reconstruction")
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
        "cw15_local_restore_CW11": bool(
            holder.get("local_finally", {}).get(
                "restored_exact_CW11_before_outer_finally", False
            )
        ),
        "cw15_local_restore_matmul_precision": bool(
            holder.get("local_finally", {}).get(
                "matmul_precision_restored_exact", False
            )
        ),
        "outer_chain_final_raw_restore": bool(cw11_result["final_integrity"]["pass"]),
    }
    if not all(final_checks.values()):
        raise RuntimeError(f"CW15 outer reconstruction/finally failed: {final_checks}")
    return {
        "schema_version": SCHEMA,
        "status": holder["status"],
        "classification": holder["classification"],
        "scope": {
            "exact_CW11_reconstruction": True,
            "CW15_minimum_total_from_CW11_origin": True,
            "semantic_active_cut_ledger": True,
            "sequential_affine_tangent_optimization_ledger": True,
            "post_merge_all_active_false_rows_tangented": True,
            "favorable_transition_restoration": True,
            "authoritative_full_six_official_B256_oracle_each_new_model_proposal":
            True,
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
            "consumed_CW13_closure": {
                "attempt_marker_path": str(CW13_ATTEMPT_MARKER.relative_to(ROOT)),
                "attempt_marker_sha256": CW13_ATTEMPT_MARKER_SHA256,
                "closed_stdout_path": str(CW13_CLOSED_STDOUT.relative_to(ROOT)),
                "closed_stdout_sha256": CW13_CLOSED_STDOUT_SHA256,
                "stderr_audit_path": str(CW13_STDERR_AUDIT.relative_to(ROOT)),
                "stderr_audit_sha256": CW13_STDERR_AUDIT_SHA256,
                "parent_solver_sha256": CW13_PARENT_SOLVER_SHA256,
            },
            "consumed_CW14_closure": {
                "attempt_marker_path": str(CW14_ATTEMPT_MARKER.relative_to(ROOT)),
                "attempt_marker_sha256": CW14_ATTEMPT_MARKER_SHA256,
                "closed_stdout_path": str(CW14_CLOSED_STDOUT.relative_to(ROOT)),
                "closed_stdout_sha256": CW14_CLOSED_STDOUT_SHA256,
                "stderr_audit_path": str(CW14_STDERR_AUDIT.relative_to(ROOT)),
                "stderr_audit_sha256": CW14_STDERR_AUDIT_SHA256,
                "parent_solver_sha256": CW14_PARENT_SOLVER_SHA256,
            },
        },
        "frozen_inputs": direct_frozen_inputs,
        "cw12_consumed_failure_evidence": direct_cw12_failure_evidence,
        "cw13_consumed_closure_evidence": direct_cw13_closure_evidence,
        "cw14_consumed_closure_evidence": direct_cw14_closure_evidence,
        "anchored_total_geometry_self_test": direct_anchored_total_math,
        "sequential_affine_tangent_self_test": direct_sequential_affine_math,
        "affine_ledger_integrity_self_test": direct_affine_ledger_math,
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
            "optimization_affine_tangent_ledger": holder[
                "terminal_optimization_affine_ledger"
            ],
            "optimization_affine_tangent_ledger_sha256": holder[
                "terminal_optimization_affine_ledger_sha256"
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
        raise RuntimeError("CW15 script must be frozen mode 0555")
    source_audit = static_source_audit(source)
    if not source_audit["pass"]:
        raise RuntimeError(f"static source audit failed: {source_audit}")
    return {
        "schema_version": SCHEMA,
        "status": "static_ready_CW15_sequential_affine_tangent_run_implemented",
        "classification": dict(CLASSIFICATION),
        "contract": {
            "starting_model_sha256": CW11_MODEL_SHA256,
            "parent_CW12_solver_sha256": CW12_PARENT_SOLVER_SHA256,
            "parent_CW13_solver_sha256": CW13_PARENT_SOLVER_SHA256,
            "parent_CW14_solver_sha256": CW14_PARENT_SOLVER_SHA256,
            "CW15_minimum_total_from_CW11_origin": True,
            "semantic_active_cut_ledger_separate": True,
            "optimization_affine_tangent_ledger_monotone": True,
            "initial_and_new_semantic_cuts_get_CW11_anchor_tangent": True,
            "post_merge_all_active_false_rows_get_candidate_tangent": True,
            "candidate_tangent_key_fields": [
                "semantic_cut_identity",
                "linearization_point_float64_le_sha256",
                "linearization_model_state_sha256",
                "threshold_float64_le_sha256",
            ],
            "candidate_tangent_rhs": "threshold-margin+gradient_dot_point",
            "favorable_transition_restoration": True,
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
            "frozen_solver_trust_region_cap_reinterpreted_as_minimum_total":
            STEP_L2_CAP,
            "previous_to_proposal_delta_l2_cap": STEP_L2_CAP,
            "additional_total_l2_cap_from_cw11": ADDITIONAL_TOTAL_L2_CAP,
            "direct_uncapped_linear_residual_tolerance":
            QP_DIRECT_RESIDUAL_ABS_TOL,
            "anchor_candidate_and_combined_QP_residuals_separately_gated": True,
            "uncapped_over_total_cap_closes_before_proposal": True,
            "clipped_vector_must_never_be_applied": True,
            "nested_affine_minimum_total_norm_monotone_gate": True,
            "nested_projection_dot_nonnegative_gate": True,
            "actual_violation_requires_strict_affine_ledger_growth": True,
            "x_model_and_semantic_state_stagnation_fail_closed": True,
            "single_BF16_output_plateau_with_changed_model_allowed": True,
            "threshold_change_from_CW14": False,
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
        "cw13_consumed_closure_evidence": validate_cw13_closure_evidence(),
        "cw14_consumed_closure_evidence": validate_cw14_closure_evidence(),
        "anchored_total_geometry_self_test": anchored_total_geometry_self_test(),
        "sequential_affine_tangent_self_test":
        sequential_affine_tangent_self_test(),
        "affine_ledger_integrity_self_test":
        affine_ledger_integrity_self_test(),
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
            "cw15_main_frozen_cw11",
        )
        source, self_evidence = cw11.read_regular_bytes(
            SCRIPT,
            None,
            "CW15 sequential-affine consumed-valid solver",
            expected_mode=FROZEN_MODE,
        )
        source_audit = static_source_audit(source)
        if not source_audit["pass"]:
            raise RuntimeError(f"CW15 source audit failed: {source_audit}")
        primary_source, primary_evidence = cw11.read_regular_bytes(
            PRIMARY,
            MODULE_SHAS[PRIMARY],
            "CW15 frozen primary source",
            expected_mode=FROZEN_MODE,
        )
        primary = import_frozen(
            PRIMARY,
            MODULE_SHAS[PRIMARY],
            EXPECTED_INPUT_MODES[PRIMARY],
            "cw15_main_frozen_primary",
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

#!/usr/bin/env python3
"""Failure-preserving one-shot launcher for frozen CW23 boundary-QP."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import math
import os
import signal
import stat
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "run_cw23_boundary_qp_specialbc_trainonly_one_shot_v1.py"
SOLVER = TOOLS / "probe_u468_cw11_boundary_qp_specialbc_cw23_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-cw23-boundary-qp-specialbc-one-shot-launcher-v1"
CHILD_SCHEMA = "ptcg-u468-cw11-boundary-qp-specialbc-cw23-v1"

ATTEMPT = ROOT / "artifacts/.ptcg-cw23_boundary_qp_specialbc_trainonly_20260803_v1-attempt.json"
CHILD_STDOUT = ROOT / "artifacts/cw23_boundary_qp_specialbc_trainonly_20260803_v1.stdout.json"
CHILD_STDERR = ROOT / "artifacts/cw23_boundary_qp_specialbc_trainonly_20260803_v1.stderr.txt"
EXECUTION = ROOT / "artifacts/cw23_boundary_qp_specialbc_trainonly_20260803_v1.execution.json"
OUTPUT = ROOT / "artifacts/cw23_cw11_boundary_qp_specialbc_trainonly_20260803_v1.json"

SOLVER_SHA256 = "4a2d209af71a7ea5d955bd373f1fbada936688fc5bc597615c41192dc360b4b8"
EXPECTED_CACHE_SHA256 = "a936e106c6ef0e02895046ab70fa9480a9426f2b371d11c6d25ed1447771519b"
CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
SELECTION_PAYLOAD_SHA256 = "c4a9b711d8f5636daea8a428b9fd5e506bffcfdaeb614c7faedf7c75e4165a21"
FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444
LINUX_MFD_CLOEXEC = 0x0001
LINUX_MFD_ALLOW_SEALING = 0x0002
LINUX_F_ADD_SEALS = 1033
LINUX_F_GET_SEALS = 1034
LINUX_F_SEAL_SEAL = 0x0001
LINUX_F_SEAL_SHRINK = 0x0002
LINUX_F_SEAL_GROW = 0x0004
LINUX_F_SEAL_WRITE = 0x0008

DIRECT_INPUTS: dict[Path, tuple[str, int]] = {
    SOLVER: (SOLVER_SHA256, 0o555),
    TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py": (
        "f307f3f26355125f97c857f14b97a73d9a7a4f2ea39ca8b0a578e801055e82b4",
        0o555,
    ),
    TOOLS / "select_cw22_targeted_b256_from_cw11_profile_v2.py": (
        "fe3ade2f38ea9e2972976654ffd584e85cc17183a7c1787c444261e515dd6c30",
        0o555,
    ),
    ROOT / "artifacts/cw22_targeted_b256_selection_v2_20260803.json": (
        "035f071902c3f43acd5cec328400d6172a990bd19864d2d794cb2cdff5f7bee7",
        0o444,
    ),
    ROOT / "artifacts/u468_cw11_full_train_margin_profile_v1_20260803.json": (
        "da684858d2c459416234cdc3cae1ffc32a5ec23a1ec376f66a0cf440776a846e",
        0o444,
    ),
    TOOLS / "probe_u468_cw11_fixed_pcgrad_specialbc_cw21_v1.py": (
        "9c18db25e16fe6b9feb1ca72d0682a1baa31ad6dc19e1a6283fd7db2da0f9cf4",
        0o555,
    ),
    TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py": (
        "8f5b64c8a04b3de9a6d2890aac582967a50f4d708f3a11a64cbd395423d907d2",
        0o555,
    ),
    TOOLS / "probe_u468_cw11_bootstrap_pcgrad_specialbc_cw19_v1.py": (
        "65f16009641481cd13538714520828640d00aed42aae920e24d9b70910423ae8",
        0o555,
    ),
    TOOLS / "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py": (
        "2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24",
        0o555,
    ),
    TOOLS / "probe_u468_raw_actor6_metricguard_specialist_valid_cw11_cuttingplane_v1.py": (
        "23bc74022210942115eaf339a38e710e88ad81ba75d62237e8eb9e2c11e074ee",
        0o555,
    ),
    TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw10_cuttingplane_v1.py": (
        "546e90c5b3ca35c84b8efc08109b2f310b45b2895d24aeee70561d13bffa146c",
        0o555,
    ),
    TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw9_cuttingplane_v1.py": (
        "f77a76f2ac91732e8bc8d9a436ca7a49058b5de50f7b3114db14cffa1fbf5292",
        0o555,
    ),
    TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_cw4_cuttingplane_v1.py": (
        "40715927549457b161fc907757279e4d85466ded1d1e6bfed62f67c13b529e7c",
        0o555,
    ),
    TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v2.py": (
        "c2866ba8b00eba6b424197a520419a5717401335cc47202b4fcc711611f67503",
        0o555,
    ),
    TOOLS / "probe_u468_raw_actor6_metricguard_specialbc_v1.py": (
        "ddecd3a85bc2b43c28854afc56678c21943198ac3fde3613765d6a49e440eedb",
        0o555,
    ),
    TOOLS / "run_u468_raw_actor6_metricguard_specialbc_ram_ray_v1.py": (
        "86b05d4f826576717907141aef2c534f140a1531b87c8e2d28aeb624a2657e4d",
        0o555,
    ),
    TOOLS / "analyze_e904_specialist_row_flips_v1.py": (
        "c577dc1280cb36197f5125dbd1a0e6be3ddbd3a8dfb17d7aaf6af53a466d6634",
        0o664,
    ),
    TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v2.py": (
        "e6bce38044f8230662cb6d61d560baa529cf507070045db53c4196bce89794b6",
        0o555,
    ),
    TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v3.py": (
        "2d12ce9e76f6672911948dcc6458573899ec45fc39e748510544001025ede939",
        0o555,
    ),
    TOOLS / "probe_u468_raw_equalblend_ray_thresholds.py": (
        "7339184a12224302f3bf9701c1fbc722e317c3c5a9f68ae4cbfcccda87b8be52",
        0o555,
    ),
    TOOLS / "run_u468_raw_balanced_train_only_endpoint_gate.py": (
        "84b51d8ef51e2184271c4bf358e5d59b74ea631daa79b84d73fffaf44a37037d",
        0o555,
    ),
    TOOLS / "materialize_u468_raw_actor6_equalblend_sgd512_endpoint.py": (
        "fdd259ba846234db8358a630c392c637773645105116d3be7fe2089cbdf3ccd3",
        0o555,
    ),
    TOOLS / "run_u468_raw_equalblend_train_only_promotion_gate.py": (
        "4f36bd0f86cbb811d8142f155027445b4bfb43a1b3e55b2fbf3b5787981aeae0",
        0o555,
    ),
    TOOLS / "run_u468_raw_equalblend_train_only_promotion_gate_armed.py": (
        "c5db208e544f9f7165b189faafc8cbe2b87ea29e0f79f876a6aa6bf42dc837a6",
        0o555,
    ),
    TOOLS / "profile_u468_raw_full_train_margins.py": (
        "c70e52e2398b8551baf0fa3a3a403fbf6636b56c70010fb948a5c85c298977ae",
        0o555,
    ),
    TOOLS / "profile_u468_beta1157_train_margins.py": (
        "f1018994c48ffadcbc5277f0dfd9233086693f90bf30f5bf67462965b4e18142",
        0o664,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_ray_threshold_probe_contract_design202608121.json": (
        "d2b2547cd03fd27b46a19e32499ba233280ed32eebe64cd98bf46570ce4604c4",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_ray_threshold_probe_design202608122.result.json": (
        "d43a1242b0ad28d71544e7e2b4112ce1559818e9e6fb70b8666efa439e91322e",
        0o444,
    ),
    ROOT / "artifacts/u468_raw_full_train_margin_profile_v3_20260802.json": (
        "17af98e175167398a80cd2e1f8930c3d1d85cfe6ceea3143dfe2ba44dc89f1f9",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_materialize_design202608116.preregistration.json": (
        "1d076b31e82d189ef829bc77bd89fe3808fd212bc3113298ae29aed437392882",
        0o444,
    ),
    ROOT / ".ptcg-ppo_u468_raw_actor6_equalblend_sgd512_materialize_design202608116-attempt.json": (
        "0a131e37a27f33c14a1e4f91ed7dba79ac2b340407c94f9e3704b624be20c460",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_materialize_design202608116/equalblend-sgd512-eval-only.pt": (
        "1684493b48b9c6a77696150d925150c21d8028c5fdd45f80e9e838a94d472be2",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_materialize_design202608116/materialization_manifest.json": (
        "51a133ec362ee489b05018728d43fb17f8ab12eb4adf1b223426afa6fafdccb1",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_materialize_design202608116/materialization_result.json": (
        "9b0395c63478434ed6b577566ebd7663c6c4dfb20742ba747f374aeb618896e3",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_contract_design202608116.json": (
        "e0d80b72aed237c3de4b643157b058acf2bff513d11420dd7579db0308b970c2",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117.preregistration.json": (
        "43eee0d5d247fd280d9b81499735d35b6fb693dbe0f734c5ef17d2830375bb3c",
        0o444,
    ),
    ROOT / ".ptcg-ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117-attempt.json": (
        "457c9359f13788c98dc2b0af22e221075710df56a9e1f7fdb3b33313a81728c6",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117/gate_decision.json": (
        "e2585f2b393b15f234d26507c918a4356459439a3c3687bd9253267153d74e90",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117/execution_manifest.json": (
        "9dd33205db048f072e207d09fa28d42632c3912f1276ae269e09fe3f9bf03ace",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117/COMPLETED.json": (
        "86526fecf340ffeefdb5052dfaacd06588f7c9324d8d2dd7dc3b5db805841186",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117/raw_full_u468/flg.json": (
        "b1b83587d39ee29f6d6327082ad3570e754bce77e5193f2feebcd1eeb173295e",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117/raw_full_u468/pokemonfan.json": (
        "e9e89927b45eabb175fdd160802613b9a76912bc40b074fef444ff2331840027",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117/raw_full_u468/core5.json": (
        "29249f098a9c961f1e73e8fadb538278c6698e5328f13152f55fa59a02528dd4",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117/sole_equalblend_endpoint/flg.json": (
        "1a200383535de971e75e94009b167b60ca602493e1d2d8af47ec5d14d19bf8c9",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117/sole_equalblend_endpoint/pokemonfan.json": (
        "4744823636f7b6179820930a01a844eba823642e366f4b635e61680c9a044977",
        0o444,
    ),
    ROOT / "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117/sole_equalblend_endpoint/core5.json": (
        "1dc0df6af28ce8232b9e629d56ac60228e1d4a0385e005b6b4129a5f9bfe01ae",
        0o444,
    ),
    TOOLS / "evaluate_policy_bc.py": (
        "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4",
        0o664,
    ),
    TOOLS / "train_bc_orbit.py": (
        "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
        0o664,
    ),
    TOOLS / "train_ppo.py": (
        "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3",
        0o664,
    ),
    ROOT
    / "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt": (
        "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f",
        0o664,
    ),
    ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip": (
        "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
        0o664,
    ),
    ROOT / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip": (
        "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
        0o664,
    ),
    ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip": (
        "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
        0o600,
    ),
}

ALLOWED_DECISIONS = {
    "NO_GO_CW23_BOUNDARY_QP_BASELINE",
    "NO_GO_CW23_BOUNDARY_QP_PREFLIGHT",
    "GO_CW23_BOUNDARY_QP_PREFLIGHT",
}
EXPECTED_BASELINE_CHECKS = frozenset(
    {
        "rows_exact256",
        "hard96_all_ordered_wrong",
        "hard96_all_set_wrong",
        "hard96_all_count_correct",
        "retention160_all_ordered_correct",
        "retention160_all_set_correct",
        "retention160_all_count_correct",
        "native_policy_BF16",
    }
)
EXPECTED_LINEARIZATION_CHECKS = frozenset(
    {"PF0_native_margin_exact", "PF7_native_margin_exact", "six_zero_native_margins_exact"}
)
EXPECTED_QP_CHECKS = frozenset(
    {
        "gram_full_rank9",
        "nominal_residual_linf_at_most_1e-9",
        "nominal_l2_requires_radial_cap",
        "planned_l2_exact_cap",
    }
)
EXPECTED_TRAIN_CHECKS = frozenset(
    {
        "PF0_target_native_ordered_correct",
        "PF7_target_native_ordered_correct",
        "PF0_repairs_at_least1",
        "PF7_repairs_at_least1",
        "total_hard_repairs_at_least2",
        "all_three_hard_exact_NLL_improve",
        "special9_exact_NLL_improves",
        "special9_first_order_constraint_is_descent",
        "six_zero_margin_guards_stay_correct",
        "retention160_zero_correct_to_wrong",
        "retention160_all_ordered_correct",
        "retention160_aggregate_NLL_nondegrade",
        "each_retention_stratum_within_quantization_tolerance",
        "count_logits_native_exact",
        "value_logits_native_exact",
        "policy_logits_native_BF16",
        "candidate_all256_count_correct",
    }
)
EXPECTED_INTEGRITY_CHECKS = frozenset(
    {
        "nominal_solution_radially_capped",
        "planned_l2_exact_cap",
        "actual_additional_from_CW11_cap",
        "candidate_model_changed",
        "changed_parameter_names_nonempty",
        "changed_parameter_scope_subset_actor6",
        "nonactor_exact_raw",
        "actual_delta_finite",
        "candidate_model_all_finite",
    }
)
EXPECTED_FINAL_CHECKS = frozenset(
    {
        "exact_CW11_context",
        "dependency_sources",
        "final_B256_baseline",
        "sole_boundary_QP_trial",
        "absolute_payload_reconstruction",
    }
)
EXPECTED_RECONSTRUCTION_CHECKS = frozenset(
    {
        "raw_U468_restore_exact",
        "candidate_hash_exact",
        "candidate_actor_bytes_exact",
        "nonactor_exact_raw",
    }
)
EXPECTED_CONTEXT_CHECKS = frozenset(
    {
        "device_cuda",
        "model_exact_CW11",
        "context_candidate_exact_CW11",
        "vector_field_exact",
        "vector_recomputed_exact",
        "vector_shape_finite_l2",
        "ledger_exact34",
        "raw_model_anchor",
        "raw_nonactor_anchor",
        "iteration_rows_exact",
        "selected_row_gate_pass",
        "model_eval80",
        "actor_names_present",
        "actor6_float32_dim65793",
        "actor_bytes_exact",
        "nonactor_exact_raw",
        "raw_parent_update468",
        "raw_parent_model_exact",
        "raw_parent_not_eval_only",
        "model_config_exact_parent",
    }
)
EXPECTED_DEPENDENCY_CHECKS = frozenset(
    {"BC_source_exact_CW15", "PPO_source_exact_CW15"}
)
EXPECTED_OUTER_RESTORE_CHECKS = frozenset(
    {
        "model_hash_exact_CW11",
        "state_keys_exact",
        "training_exact",
        "requires_grad_exact",
        "gradient_presence_exact",
        "precision_exact",
        "deterministic_exact",
        "warn_only_exact",
        "debug_mode_exact",
        "cudnn_benchmark_exact",
        "cudnn_deterministic_exact",
        "cuda_tf32_exact",
        "cudnn_tf32_exact",
        "BC_MAX_ACTION_COUNT_exact",
        "cpu_rng_exact",
        "cuda_rng_exact",
        "python_rng_exact",
        "numpy_rng_exact",
    }
)
EXPECTED_HISTORICAL_CHECKS = frozenset(
    {
        "callback_once",
        "status_exact_CW11",
        "consumer_called",
        "model_exact_CW11",
        "vector_exact_CW11",
        "ledger_exact_CW11",
        "historical_restore_pass",
    }
)


class LauncherError(RuntimeError):
    """Fail-closed one-shot launcher error."""


class LauncherSignal(LauncherError):
    """A catchable external signal received after the attempt was committed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LauncherError(f"{label}: duplicate key {key}")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda token: (_ for _ in ()).throw(
            LauncherError(f"{label}: nonfinite token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise LauncherError(f"{label}: root is not an object")
    if canonical_json(value) != payload:
        raise LauncherError(f"{label}: JSON is not canonical")
    return value


def stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_mode),
        int(value.st_nlink),
    )


def descriptor(path: Path, expected_sha: str | None = None, expected_mode: int | None = None) -> dict[str, Any]:
    path_before = path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    digest_state = hashlib.sha256()
    byte_count = 0
    try:
        opened_before = os.fstat(fd)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest_state.update(chunk)
            byte_count += len(chunk)
        opened_after = os.fstat(fd)
        path_after = path.lstat()
    finally:
        os.close(fd)
    digest = digest_state.hexdigest()

    checks = {
        "regular_single_link": stat.S_ISREG(opened_after.st_mode)
        and stat.S_ISREG(path_after.st_mode)
        and not stat.S_ISLNK(path_after.st_mode)
        and int(opened_after.st_nlink) == 1
        and int(path_after.st_nlink) == 1,
        "identity_stable": stat_identity(path_before)
        == stat_identity(opened_before)
        == stat_identity(opened_after)
        == stat_identity(path_after),
        "complete_read": byte_count == int(opened_after.st_size),
        "sha_exact": expected_sha is None or digest == expected_sha,
        "mode_exact": expected_mode is None
        or stat.S_IMODE(opened_after.st_mode) == expected_mode,
    }
    if not all(checks.values()):
        raise LauncherError(f"input drift for {path}: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": byte_count,
        "mode_octal": format(stat.S_IMODE(opened_after.st_mode), "04o"),
        "device": int(opened_after.st_dev),
        "inode": int(opened_after.st_ino),
        "nlink": int(opened_after.st_nlink),
        "mtime_ns": int(opened_after.st_mtime_ns),
        "ctime_ns": int(opened_after.st_ctime_ns),
        "checks": checks,
    }


def lexists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def safe_descriptor(path: Path) -> dict[str, Any]:
    try:
        if not lexists(path):
            return {"path": str(path.relative_to(ROOT)), "exists": False}
        return {"exists": True, **descriptor(path)}
    except BaseException as error:
        try:
            observed = path.lstat()
            return {
                "path": str(path.relative_to(ROOT)),
                "exists": True,
                "error": f"{type(error).__name__}: {error}",
                "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
                "bytes": int(observed.st_size),
                "device": int(observed.st_dev),
                "inode": int(observed.st_ino),
                "nlink": int(observed.st_nlink),
            }
        except BaseException as nested:
            return {
                "path": str(path.relative_to(ROOT)),
                "exists": lexists(path),
                "error": f"{type(error).__name__}: {error}",
                "nested_error": f"{type(nested).__name__}: {nested}",
            }


def metadata_only(path: Path) -> dict[str, Any]:
    try:
        observed = path.lstat()
        return {
            "path": str(path.relative_to(ROOT)),
            "exists": True,
            "content_hashed": False,
            "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
            "bytes_at_snapshot": int(observed.st_size),
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
            "nlink": int(observed.st_nlink),
            "mtime_ns": int(observed.st_mtime_ns),
            "ctime_ns": int(observed.st_ctime_ns),
        }
    except FileNotFoundError:
        return {
            "path": str(path.relative_to(ROOT)),
            "exists": False,
            "content_hashed": False,
        }
    except BaseException as error:
        return {
            "path": str(path.relative_to(ROOT)),
            "exists": lexists(path),
            "content_hashed": False,
            "metadata_error": f"{type(error).__name__}: {error}",
        }


def live_fd_metadata(fd: int, path: Path) -> dict[str, Any]:
    try:
        observed_fd = os.fstat(fd)
        observed_path = path.lstat()
        return {
            "stable": False,
            "reason": "live_process_group_may_retain_writable_fd",
            "path": str(path.relative_to(ROOT)),
            "content_hashed": False,
            "fd_path_same_device_inode": (
                int(observed_fd.st_dev), int(observed_fd.st_ino)
            )
            == (int(observed_path.st_dev), int(observed_path.st_ino)),
            "fd_bytes_at_snapshot": int(observed_fd.st_size),
            "path_bytes_at_snapshot": int(observed_path.st_size),
            "device": int(observed_fd.st_dev),
            "inode": int(observed_fd.st_ino),
            "nlink": int(observed_fd.st_nlink),
            "mode_octal": format(stat.S_IMODE(observed_fd.st_mode), "04o"),
        }
    except BaseException as error:
        return {
            "stable": False,
            "reason": "live_process_group_may_retain_writable_fd",
            "path": str(path.relative_to(ROOT)),
            "content_hashed": False,
            "metadata_error": f"{type(error).__name__}: {error}",
        }


def exclusive_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    observed_fd: os.stat_result | None = None
    try:
        assert_fd_path_binding(fd, path)
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise LauncherError(f"short evidence write: {path}")
            written += count
        os.fsync(fd)
        os.fchmod(fd, EVIDENCE_MODE)
        os.fsync(fd)
        observed_fd = assert_fd_path_binding(fd, path)
    finally:
        os.close(fd)
    record = descriptor(path, hashlib.sha256(payload).hexdigest(), EVIDENCE_MODE)
    if observed_fd is None or (record["device"], record["inode"]) != (
        int(observed_fd.st_dev),
        int(observed_fd.st_ino),
    ):
        raise LauncherError(f"exclusive evidence descriptor inode drift: {path}")
    return record


def exclusive_json(path: Path, value: Any) -> dict[str, Any]:
    return exclusive_bytes(path, canonical_json(value))


def reserve_evidence(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        assert_fd_path_binding(fd, path)
    except BaseException:
        os.close(fd)
        raise
    return fd


def assert_fd_path_binding(fd: int, path: Path) -> os.stat_result:
    observed_fd = os.fstat(fd)
    observed_path = path.lstat()
    if (
        not stat.S_ISREG(observed_fd.st_mode)
        or not stat.S_ISREG(observed_path.st_mode)
        or stat.S_ISLNK(observed_path.st_mode)
        or int(observed_fd.st_nlink) != 1
        or int(observed_path.st_nlink) != 1
        or stat_identity(observed_fd) != stat_identity(observed_path)
    ):
        raise LauncherError(f"reserved evidence path/FD binding drift: {path}")
    return observed_fd


def write_all_fd(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        count = os.write(fd, payload[offset:])
        if count <= 0:
            raise LauncherError("short reserved evidence write")
        offset += count


def read_all_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def append_fd(fd: int, payload: bytes) -> None:
    os.lseek(fd, 0, os.SEEK_END)
    write_all_fd(fd, payload)


def seal_reserved_stream(fd: int, path: Path) -> tuple[bytes, dict[str, Any]]:
    assert_fd_path_binding(fd, path)
    os.fsync(fd)
    payload = read_all_fd(fd)
    os.fchmod(fd, EVIDENCE_MODE)
    os.fsync(fd)
    observed_fd = assert_fd_path_binding(fd, path)
    record = descriptor(
        path, hashlib.sha256(payload).hexdigest(), EVIDENCE_MODE
    )
    if (record["device"], record["inode"]) != (
        int(observed_fd.st_dev),
        int(observed_fd.st_ino),
    ):
        raise LauncherError(f"reserved stream descriptor inode drift: {path}")
    return payload, record


def commit_reserved_json(fd: int, path: Path, value: Any) -> dict[str, Any]:
    payload = canonical_json(value)
    assert_fd_path_binding(fd, path)
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    write_all_fd(fd, payload)
    os.fsync(fd)
    os.fchmod(fd, EVIDENCE_MODE)
    os.fsync(fd)
    observed_fd = assert_fd_path_binding(fd, path)
    record = descriptor(path, hashlib.sha256(payload).hexdigest(), EVIDENCE_MODE)
    if (record["device"], record["inode"]) != (
        int(observed_fd.st_dev),
        int(observed_fd.st_ino),
    ):
        raise LauncherError(f"reserved JSON descriptor inode drift: {path}")
    return record


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def reap_child_until(pid: int, timeout_seconds: float) -> tuple[bool, int | None]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            waited_pid, wait_status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return True, None
        if waited_pid == pid:
            return True, int(os.waitstatus_to_exitcode(wait_status))
        if time.monotonic() >= deadline:
            return False, None
        time.sleep(0.05)


def ensure_process_group_gone(pgid: int, notes: list[str]) -> bool:
    if not process_group_exists(pgid):
        notes.append("child_process_group_confirmed_gone")
        return False
    os.killpg(pgid, signal.SIGTERM)
    notes.append("remaining_child_process_group_SIGTERM")
    deadline = time.monotonic() + 10.0
    while process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if process_group_exists(pgid):
        os.killpg(pgid, signal.SIGKILL)
        notes.append("remaining_child_process_group_SIGKILL")
        deadline = time.monotonic() + 5.0
        while process_group_exists(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if process_group_exists(pgid):
        raise LauncherError("child process group persisted after SIGKILL")
    notes.append("child_process_group_confirmed_gone")
    return True


def input_snapshot() -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(ROOT)): descriptor(path, digest, mode)
        for path, (digest, mode) in DIRECT_INPUTS.items()
    }


def safe_input_snapshot() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path, (digest, mode) in DIRECT_INPUTS.items():
        key = str(path.relative_to(ROOT))
        try:
            result[key] = descriptor(path, digest, mode)
        except BaseException as error:
            result[key] = {
                "path": key,
                "error": f"{type(error).__name__}: {error}",
                "safe": safe_descriptor(path),
            }
    return result


def same_snapshot(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "sha256",
        "bytes",
        "mode_octal",
        "device",
        "inode",
        "nlink",
        "mtime_ns",
        "ctime_ns",
    )
    return set(left) == set(right) and all(
        all(left[key].get(field) == right[key].get(field) for field in fields)
        for key in left
    )


def create_sealed_memfd(source: bytes) -> tuple[int, dict[str, Any]]:
    flags = LINUX_MFD_CLOEXEC | LINUX_MFD_ALLOW_SEALING
    native_memfd_create = getattr(os, "memfd_create", None)
    if callable(native_memfd_create):
        fd = native_memfd_create("cw23-boundary-qp-frozen", flags)
        backend = "os.memfd_create"
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        if not hasattr(libc, "memfd_create"):
            raise LauncherError("libc has no memfd_create")
        libc_memfd_create = libc.memfd_create
        libc_memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        libc_memfd_create.restype = ctypes.c_int
        fd = int(libc_memfd_create(b"cw23-boundary-qp-frozen", flags))
        if fd < 0:
            observed_errno = ctypes.get_errno()
            raise OSError(observed_errno, os.strerror(observed_errno))
        backend = "libc.memfd_create"
    try:
        written = 0
        while written < len(source):
            count = os.write(fd, source[written:])
            if count <= 0:
                raise LauncherError("short memfd write")
            written += count
        os.lseek(fd, 0, os.SEEK_SET)
        requested = (
            LINUX_F_SEAL_SEAL
            | LINUX_F_SEAL_SHRINK
            | LINUX_F_SEAL_GROW
            | LINUX_F_SEAL_WRITE
        )
        add_seals_command = int(
            getattr(fcntl, "F_ADD_SEALS", LINUX_F_ADD_SEALS)
        )
        get_seals_command = int(
            getattr(fcntl, "F_GET_SEALS", LINUX_F_GET_SEALS)
        )
        fcntl.fcntl(fd, add_seals_command, requested)
        observed = int(fcntl.fcntl(fd, get_seals_command))
        if observed != requested:
            raise LauncherError(f"memfd seal drift: {observed} != {requested}")
        return fd, {
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "bytes": len(source),
            "backend": backend,
            "memfd_flags": flags,
            "add_seals_command": add_seals_command,
            "get_seals_command": get_seals_command,
            "requested_seals": requested,
            "observed_seals": observed,
            "all_four_seals": observed == 15,
        }
    except BaseException:
        os.close(fd)
        raise


def exact_bool_map(value: Any, expected_keys: frozenset[str], *, all_true: bool) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(expected_keys)
        and all(isinstance(item, bool) for item in value.values())
        and (not all_true or all(item is True for item in value.values()))
    )


def nonempty_true_map(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(item is True for item in value.values())
    )


def exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def finite_real(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_child(summary: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, bool]:
    endpoint = document.get("endpoint", {})
    decision = document.get("decision")
    trial = endpoint.get("trial", {}) if isinstance(endpoint, Mapping) else {}
    baseline_checks = endpoint.get("baseline_checks")
    linearization_checks = endpoint.get("baseline_linearization_checks")
    qp_checks = endpoint.get("boundary_qp", {}).get("checks")
    checks = {
        "child_schema_exact": document.get("schema_version") == CHILD_SCHEMA,
        "summary_schema_exact": summary.get("schema_version") == CHILD_SCHEMA,
        "decision_allowed": decision in ALLOWED_DECISIONS,
        "summary_decision_exact": summary.get("decision") == decision,
        "summary_status_exact": summary.get("status") == decision,
        "status_exact": document.get("status") == decision == endpoint.get("decision"),
        "base_exact_CW11": document.get("base", {}).get("model_state_sha256")
        == CW11_MODEL_SHA256,
        "selection_exact": document.get("selection", {}).get("sha256")
        == SELECTION_PAYLOAD_SHA256,
        "cache_exact": endpoint.get("cache", {}).get("cache_sha256")
        == EXPECTED_CACHE_SHA256,
        "outer_restore_pass": endpoint.get("outer_restore_pass") is True,
        "outer_restore_checks_exact_true": exact_bool_map(
            endpoint.get("outer_restore_checks"),
            EXPECTED_OUTER_RESTORE_CHECKS,
            all_true=True,
        ),
        "context_checks_exact_true": exact_bool_map(
            endpoint.get("context_checks"), EXPECTED_CONTEXT_CHECKS, all_true=True
        ),
        "dependency_checks_exact_true": exact_bool_map(
            endpoint.get("dependency_checks"),
            EXPECTED_DEPENDENCY_CHECKS,
            all_true=True,
        ),
        "historical_replay_pass": document.get("historical_exact_CW11_replay", {}).get(
            "pass"
        )
        is True
        and exact_bool_map(
            document.get("historical_exact_CW11_replay", {}).get("checks"),
            EXPECTED_HISTORICAL_CHECKS,
            all_true=True,
        ),
        "historical_CW23_zero_new_validation": document.get(
            "historical_exact_CW11_replay", {}
        ).get("new_validation_rows_opened_for_CW23_selection_or_candidate")
        == 0
        and exact_int(
            document.get("historical_exact_CW11_replay", {}).get(
                "new_validation_rows_opened_for_CW23_selection_or_candidate"
            ),
            0,
        )
        and "new_validation_rows_opened_for_CW22_selection_or_candidate"
        not in document.get("historical_exact_CW11_replay", {}),
        "CW23_contract_exact": document.get("audit", {}).get("CW23_contract", {}).get(
            "train_only"
        )
        is True
        and document.get("audit", {}).get("CW23_contract", {}).get(
            "new_validation_or_test_rows_opened"
        )
        == 0
        and exact_int(
            document.get("audit", {}).get("CW23_contract", {}).get(
                "new_validation_or_test_rows_opened"
            ),
            0,
        )
        and document.get("audit", {}).get("CW23_contract", {}).get(
            "checkpoint_writes"
        )
        == 0
        and exact_int(
            document.get("audit", {}).get("CW23_contract", {}).get(
                "checkpoint_writes"
            ),
            0,
        )
        and document.get("audit", {}).get("CW23_contract", {}).get(
            "optimizer_instances_created"
        )
        == 0
        and exact_int(
            document.get("audit", {}).get("CW23_contract", {}).get(
                "optimizer_instances_created"
            ),
            0,
        )
        and document.get("audit", {}).get("CW23_contract", {}).get(
            "optimizer_step_calls"
        )
        == 0
        and exact_int(
            document.get("audit", {}).get("CW23_contract", {}).get(
                "optimizer_step_calls"
            ),
            0,
        )
        and document.get("audit", {}).get("CW23_contract", {}).get(
            "changed_candidate_train_shadow_count"
        )
        == endpoint.get("changed_candidate_train_shadow_count")
        and type(
            document.get("audit", {}).get("CW23_contract", {}).get(
                "changed_candidate_train_shadow_count"
            )
        )
        is int
        and document.get("audit", {}).get("CW23_contract", {}).get(
            "official_unique_changed_candidate_count_consumed"
        )
        == 0
        and exact_int(
            document.get("audit", {}).get("CW23_contract", {}).get(
                "official_unique_changed_candidate_count_consumed"
            ),
            0,
        )
        and document.get("audit", {}).get("CW23_contract", {}).get(
            "submission_performed"
        )
        is False,
        "official_budget_zero": exact_int(
            document.get("official_unique_changed_candidate_count_consumed"), 0
        ),
        "submission_false": document.get("submission_performed") is False,
        "package_false": document.get("package_upload_performed") is False,
        "summary_budget_zero": exact_int(
            summary.get("official_unique_changed_candidate_count_consumed"), 0
        ),
        "summary_submission_false": summary.get("submission_performed") is False,
        "summary_payload_presence_exact": summary.get("candidate_payload_present")
        is (endpoint.get("candidate_payload") is not None),
        "summary_changed_count_exact": summary.get(
            "changed_candidate_train_shadow_count"
        )
        == endpoint.get("changed_candidate_train_shadow_count")
        and type(summary.get("changed_candidate_train_shadow_count")) is int
        and type(endpoint.get("changed_candidate_train_shadow_count")) is int,
        "output_descriptor_exact": sha256_hex(
            summary.get("output", {}).get("sha256")
        )
        and summary.get("output", {}).get("sha256") == sha256_file(OUTPUT)
        and summary.get("output", {}).get("path") == str(OUTPUT.relative_to(ROOT))
        and exact_int(summary.get("output", {}).get("nlink"), 1)
        and type(summary.get("output", {}).get("bytes")) is int
        and summary.get("output", {}).get("bytes") > 0
        and summary.get("output", {}).get("mode_octal") == "0444"
        and nonempty_true_map(summary.get("output", {}).get("checks")),
    }
    if decision == "GO_CW23_BOUNDARY_QP_PREFLIGHT":
        train_checks = trial.get("train_checks")
        integrity_checks = trial.get("integrity")
        final_checks = endpoint.get("final_checks")
        reconstruction_checks = endpoint.get("pure_payload_reconstruction", {}).get(
            "checks"
        )
        checks.update(
            {
                "GO_reason_exact": endpoint.get("reason")
                == "SOLE_PREREGISTERED_BOUNDARY_QP_CANDIDATE_PASSED_TARGETED_TRAIN_GATE",
                "GO_payload_present": endpoint.get("candidate_payload") is not None
                and isinstance(endpoint.get("candidate_payload"), Mapping)
                and summary.get("candidate_payload_present") is True,
                "GO_single_changed_shadow": exact_int(
                    endpoint.get("changed_candidate_train_shadow_count"), 1
                )
                and exact_int(
                    summary.get("changed_candidate_train_shadow_count"), 1
                ),
                "GO_trial_pass": trial.get("pass") is True,
                "GO_train_checks_exact_true": exact_bool_map(
                    train_checks, EXPECTED_TRAIN_CHECKS, all_true=True
                ),
                "GO_integrity_checks_exact_true": exact_bool_map(
                    integrity_checks, EXPECTED_INTEGRITY_CHECKS, all_true=True
                ),
                "GO_baseline_checks_exact_true": exact_bool_map(
                    baseline_checks, EXPECTED_BASELINE_CHECKS, all_true=True
                ),
                "GO_linearization_checks_exact_true": exact_bool_map(
                    linearization_checks, EXPECTED_LINEARIZATION_CHECKS, all_true=True
                ),
                "GO_QP_checks_exact_true": exact_bool_map(
                    qp_checks, EXPECTED_QP_CHECKS, all_true=True
                ),
                "GO_final_checks_exact_true": exact_bool_map(
                    final_checks, EXPECTED_FINAL_CHECKS, all_true=True
                ),
                "GO_reconstruction_checks_exact_true": exact_bool_map(
                    reconstruction_checks,
                    EXPECTED_RECONSTRUCTION_CHECKS,
                    all_true=True,
                ),
                "GO_reconstruction_pass": endpoint.get(
                    "pure_payload_reconstruction", {}
                ).get("pass")
                is True,
                "GO_hash_exact": sha256_hex(
                    summary.get("candidate_model_state_sha256")
                )
                and summary.get("candidate_model_state_sha256")
                != CW11_MODEL_SHA256
                and summary.get("candidate_model_state_sha256")
                == trial.get("candidate_model_state_sha256")
                == endpoint.get("candidate_payload", {})
                .get("anchor", {})
                .get("terminal_model_state_sha256"),
                "GO_payload_anchor_exact": endpoint.get("candidate_payload", {})
                .get("anchor", {})
                .get("reconstruction_base")
                == "original_raw_U468"
                and endpoint.get("candidate_payload", {})
                .get("anchor", {})
                .get("raw_checkpoint_file_sha256")
                == "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
                and endpoint.get("candidate_payload", {})
                .get("anchor", {})
                .get("raw_model_state_sha256")
                == "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
                and endpoint.get("candidate_payload", {})
                .get("anchor", {})
                .get("CW11_provenance_model_state_sha256")
                == CW11_MODEL_SHA256
                and endpoint.get("candidate_payload", {})
                .get("anchor", {})
                .get("CW11_materialized_eval_only_checkpoint_used")
                is False
                and endpoint.get("candidate_payload", {}).get("selection_sha256")
                == SELECTION_PAYLOAD_SHA256
                and endpoint.get("candidate_payload", {}).get("cache_sha256")
                == EXPECTED_CACHE_SHA256,
                "GO_l2_cap": finite_real(
                    trial.get("actual_additional_from_CW11_l2")
                )
                and 0.0
                < float(trial.get("actual_additional_from_CW11_l2"))
                <= 1.0e-3
                and trial.get("actual_additional_from_CW11_l2")
                == endpoint.get("candidate_payload", {}).get(
                    "actual_additional_from_CW11_l2"
                ),
            }
        )
    elif decision == "NO_GO_CW23_BOUNDARY_QP_BASELINE":
        checks.update(
            {
                "baseline_reason_exact": endpoint.get("reason")
                == "FINAL_B256_BASELINE_CERTIFICATION_FAILED",
                "baseline_no_payload": endpoint.get("candidate_payload") is None
                and summary.get("candidate_payload_present") is False,
                "baseline_zero_changed": exact_int(
                    endpoint.get("changed_candidate_train_shadow_count"), 0
                )
                and exact_int(
                    summary.get("changed_candidate_train_shadow_count"), 0
                ),
                "baseline_checks_exact_with_failure": exact_bool_map(
                    baseline_checks, EXPECTED_BASELINE_CHECKS, all_true=False
                )
                and not all(baseline_checks.values()),
                "baseline_no_trial_or_payload_geometry": "trial" not in endpoint
                and "boundary_qp" not in endpoint,
                "baseline_summary_hash_none": summary.get(
                    "candidate_model_state_sha256"
                )
                is None,
            }
        )
    elif decision == "NO_GO_CW23_BOUNDARY_QP_PREFLIGHT":
        reason = endpoint.get("reason")
        train_checks = trial.get("train_checks")
        integrity_checks = trial.get("integrity")
        final_checks = endpoint.get("final_checks")
        reconstruction_checks = endpoint.get("pure_payload_reconstruction", {}).get(
            "checks"
        )
        failed_train_branch = (
            reason == "SOLE_BOUNDARY_QP_CANDIDATE_FAILED_TARGETED_TRAIN_GATE"
            and trial.get("pass") is False
            and exact_bool_map(train_checks, EXPECTED_TRAIN_CHECKS, all_true=False)
            and exact_bool_map(
                integrity_checks, EXPECTED_INTEGRITY_CHECKS, all_true=False
            )
            and not (
                all(train_checks.values()) and all(integrity_checks.values())
            )
            and "final_checks" not in endpoint
            and "pure_payload_reconstruction" not in endpoint
        )
        failed_reconstruction_branch = (
            reason == "ABSOLUTE_PAYLOAD_RECONSTRUCTION_FAILED"
            and trial.get("pass") is True
            and exact_bool_map(train_checks, EXPECTED_TRAIN_CHECKS, all_true=True)
            and exact_bool_map(
                integrity_checks, EXPECTED_INTEGRITY_CHECKS, all_true=True
            )
            and exact_bool_map(final_checks, EXPECTED_FINAL_CHECKS, all_true=False)
            and final_checks.get("exact_CW11_context") is True
            and final_checks.get("dependency_sources") is True
            and final_checks.get("final_B256_baseline") is True
            and final_checks.get("sole_boundary_QP_trial") is True
            and final_checks.get("absolute_payload_reconstruction") is False
            and exact_bool_map(
                reconstruction_checks,
                EXPECTED_RECONSTRUCTION_CHECKS,
                all_true=False,
            )
            and not all(reconstruction_checks.values())
            and final_checks.get("absolute_payload_reconstruction")
            is all(reconstruction_checks.values())
            and endpoint.get("pure_payload_reconstruction", {}).get("pass") is False
        )
        checks.update(
            {
                "NO_GO_no_payload": endpoint.get("candidate_payload") is None
                and summary.get("candidate_payload_present") is False,
                "NO_GO_single_changed": exact_int(
                    endpoint.get("changed_candidate_train_shadow_count"), 1
                )
                and exact_int(
                    summary.get("changed_candidate_train_shadow_count"), 1
                ),
                "NO_GO_baseline_checks_exact_true": exact_bool_map(
                    baseline_checks, EXPECTED_BASELINE_CHECKS, all_true=True
                ),
                "NO_GO_linearization_checks_exact_true": exact_bool_map(
                    linearization_checks, EXPECTED_LINEARIZATION_CHECKS, all_true=True
                ),
                "NO_GO_QP_checks_exact_true": exact_bool_map(
                    qp_checks, EXPECTED_QP_CHECKS, all_true=True
                ),
                "NO_GO_exact_branch_semantics": failed_train_branch
                or failed_reconstruction_branch,
                "NO_GO_hash_exact": sha256_hex(
                    summary.get("candidate_model_state_sha256")
                )
                and summary.get("candidate_model_state_sha256")
                != CW11_MODEL_SHA256
                and summary.get("candidate_model_state_sha256")
                == trial.get("candidate_model_state_sha256"),
            }
        )
    if not all(checks.values()):
        raise LauncherError(f"child semantic validation failed: {checks}")
    return checks


def target_paths() -> tuple[Path, ...]:
    return (ATTEMPT, CHILD_STDOUT, CHILD_STDERR, EXECUTION, OUTPUT)


def fd_table_audit() -> dict[str, Any]:
    stdio: dict[str, bool] = {}
    for fd in (0, 1, 2):
        try:
            os.fstat(fd)
            stdio[str(fd)] = True
        except OSError:
            stdio[str(fd)] = False
    ambient_inheritable: list[int] = []
    ambient_open: list[int] = []
    for name in os.listdir("/proc/self/fd"):
        try:
            fd = int(name)
            if fd <= 2:
                continue
            os.fstat(fd)
        except (OSError, ValueError):
            continue
        ambient_open.append(fd)
        if os.get_inheritable(fd):
            ambient_inheritable.append(fd)
    return {
        "stdio_open": stdio,
        "ambient_open_fds": sorted(ambient_open),
        "ambient_inheritable_fds": sorted(ambient_inheritable),
    }


def main() -> None:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise LauncherError(f"wrong Python: {sys.executable}")
    expected_process_argv = [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        str(SCRIPT),
    ]
    proc_argv = [
        item.decode("utf-8")
        for item in Path("/proc/self/cmdline").read_bytes().split(b"\0")
        if item
    ]
    fd_audit = fd_table_audit()
    runtime_checks = {
        "cwd_exact_root": Path.cwd().resolve() == ROOT.resolve(),
        "isolated_flag_exact": sys.flags.isolated == 1,
        "dont_write_bytecode_flag_exact": sys.flags.dont_write_bytecode == 1,
        "launcher_file_exact": Path(__file__).resolve() == SCRIPT.resolve(),
        "argv0_exact": Path(sys.argv[0]).resolve() == SCRIPT.resolve(),
        "sys_argv_exact_no_extra_args": sys.argv == [str(SCRIPT)],
        "orig_argv_exact": list(getattr(sys, "orig_argv", ()))
        == expected_process_argv,
        "proc_cmdline_exact": proc_argv == expected_process_argv,
        "stdio_0_1_2_open": all(fd_audit["stdio_open"].values()),
        "no_ambient_inheritable_fds": not fd_audit["ambient_inheritable_fds"],
    }
    if not all(runtime_checks.values()):
        raise LauncherError(f"launcher runtime drift: {runtime_checks}")
    existing = [str(path.relative_to(ROOT)) for path in target_paths() if lexists(path)]
    if existing:
        raise LauncherError(f"one-shot target already exists: {existing}")
    launcher_before = descriptor(SCRIPT, expected_mode=FROZEN_MODE)
    inputs_before = input_snapshot()
    solver_source = SOLVER.read_bytes()
    if hashlib.sha256(solver_source).hexdigest() != SOLVER_SHA256:
        raise LauncherError("solver source drift before attempt")
    argv = [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        "/proc/self/fd/<sealed-solver-fd>",
        "--output",
        str(OUTPUT),
    ]
    marker = {
        "schema_version": SCHEMA,
        "status": "attempt_committed_before_child",
        "launcher": launcher_before,
        "python": str(EXPECTED_PYTHON),
        "runtime_checks": runtime_checks,
        "fd_table_before_attempt": fd_audit,
        "argv": argv,
        "solver_sha256": SOLVER_SHA256,
        "inputs": inputs_before,
        "targets": [str(path.relative_to(ROOT)) for path in target_paths()],
        "official_unique_changed_candidate_count_consumed_before": 0,
        "submission_authorized": False,
    }
    stdout_payload = b""
    stderr_payload = b""
    child_returncode: int | None = None
    memfd_audit: dict[str, Any] | None = None
    validation: dict[str, bool] | None = None
    error_text: str | None = None
    child_summary: dict[str, Any] | None = None
    child_document: dict[str, Any] | None = None
    output_validation_record: dict[str, Any] | None = None
    execution_fd: int | None = None
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    memfd: int | None = None
    devnull_fd: int | None = None
    child_pid: int | None = None
    child_reaped = False
    stream_writers_quiescent = True
    termination_notes: list[str] = []
    previous_signal_handlers: dict[int, Any] = {}
    attempt_record: dict[str, Any] | None = None
    guarded_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    original_signal_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, guarded_signals
    )

    def catchable_signal(signum: int, _frame: Any) -> None:
        signal.pthread_sigmask(signal.SIG_BLOCK, guarded_signals)
        raise LauncherSignal(f"received signal {signum} after attempt commit")

    for signum in guarded_signals:
        previous_signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, catchable_signal)

    try:
        execution_fd = reserve_evidence(EXECUTION)
        attempt_record = exclusive_json(ATTEMPT, marker)
        stdout_fd = reserve_evidence(CHILD_STDOUT)
        stderr_fd = reserve_evidence(CHILD_STDERR)
        memfd, memfd_audit = create_sealed_memfd(solver_source)
        devnull_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            devnull_flags |= os.O_NOFOLLOW
        devnull_fd = os.open("/dev/null", devnull_flags)
        occupied_fds = {
            0,
            1,
            2,
            memfd,
            stdout_fd,
            stderr_fd,
            devnull_fd,
            execution_fd,
        }
        child_solver_fd = next(
            value for value in range(198, 512) if value not in occupied_fds
        )
        if stdout_fd in (0, 1, 2) or stderr_fd in (0, 1, 2):
            raise LauncherError("reserved stream unexpectedly occupied stdio")
        actual_argv = [
            str(EXPECTED_PYTHON),
            "-I",
            "-B",
            f"/proc/self/fd/{child_solver_fd}",
            "--output",
            str(OUTPUT),
        ]
        file_actions = [
            (os.POSIX_SPAWN_DUP2, devnull_fd, 0),
            (os.POSIX_SPAWN_DUP2, stdout_fd, 1),
            (os.POSIX_SPAWN_DUP2, stderr_fd, 2),
            (os.POSIX_SPAWN_DUP2, memfd, child_solver_fd),
            (os.POSIX_SPAWN_CLOSE, devnull_fd),
            (os.POSIX_SPAWN_CLOSE, stdout_fd),
            (os.POSIX_SPAWN_CLOSE, stderr_fd),
            (os.POSIX_SPAWN_CLOSE, memfd),
        ]
        pre_spawn_pending = set(signal.sigpending()).intersection(guarded_signals)
        if pre_spawn_pending:
            raise LauncherSignal(
                "guarded signal arrived before child launch: "
                + ",".join(str(int(value)) for value in sorted(pre_spawn_pending))
            )
        try:
            child_pid = os.posix_spawn(
                str(EXPECTED_PYTHON),
                actual_argv,
                dict(os.environ),
                file_actions=file_actions,
                setpgroup=0,
                setsigmask=original_signal_mask,
                setsigdef=guarded_signals,
            )
        finally:
            os.close(memfd)
            memfd = None
            os.close(devnull_fd)
            devnull_fd = None
        signal.pthread_sigmask(signal.SIG_SETMASK, original_signal_mask)
        waited_pid, wait_status = os.waitpid(child_pid, 0)
        if waited_pid != child_pid:
            raise LauncherError("waitpid returned the wrong child")
        child_reaped = True
        child_returncode = int(os.waitstatus_to_exitcode(wait_status))
        if ensure_process_group_gone(child_pid, termination_notes):
            raise LauncherError("child left live descendants after main-process exit")
        stdout_payload = read_all_fd(stdout_fd)
        stderr_payload = read_all_fd(stderr_fd)
        if child_returncode != 0:
            raise LauncherError(f"child exited {child_returncode}")
        child_summary = strict_json_bytes(stdout_payload, "child stdout")
        output_validation_record = descriptor(OUTPUT, expected_mode=EVIDENCE_MODE)
        child_output_payload = OUTPUT.read_bytes()
        output_validation_after = descriptor(OUTPUT, expected_mode=EVIDENCE_MODE)
        if not same_snapshot(
            {"output": output_validation_record},
            {"output": output_validation_after},
        ) or hashlib.sha256(child_output_payload).hexdigest() != output_validation_record[
            "sha256"
        ]:
            raise LauncherError("child output changed during semantic read")
        child_document = strict_json_bytes(child_output_payload, "child output")
        validation = validate_child(child_summary, child_document)
        if not same_snapshot(inputs_before, input_snapshot()):
            raise LauncherError("direct inputs changed during child")
        launcher_after = descriptor(SCRIPT, expected_mode=FROZEN_MODE)
        if not same_snapshot({"launcher": launcher_before}, {"launcher": launcher_after}):
            raise LauncherError("launcher changed during child")
        current_attempt = descriptor(
            ATTEMPT, attempt_record["sha256"], EVIDENCE_MODE
        )
        if not same_snapshot({"attempt": attempt_record}, {"attempt": current_attempt}):
            raise LauncherError("attempt marker changed after commit")
        signal.pthread_sigmask(signal.SIG_BLOCK, guarded_signals)
    except BaseException as error:
        signal.pthread_sigmask(signal.SIG_BLOCK, guarded_signals)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGTERM)
                termination_notes.append(
                    "child_process_group_SIGTERM_after_launcher_exception"
                )
            except ProcessLookupError:
                termination_notes.append(
                    "child_process_group_absent_after_launcher_exception"
                )
            except BaseException as terminate_error:
                termination_notes.append(
                    f"terminate_failed: {type(terminate_error).__name__}: {terminate_error}"
                )
        if child_pid is not None and not child_reaped:
            reaped, observed_returncode = reap_child_until(child_pid, 10.0)
            if reaped:
                child_reaped = True
                if observed_returncode is not None:
                    child_returncode = observed_returncode
            else:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                    termination_notes.append(
                        "child_process_group_SIGKILL_after_TERM_deadline"
                    )
                    reaped, observed_returncode = reap_child_until(child_pid, 5.0)
                    if reaped:
                        child_reaped = True
                        if observed_returncode is not None:
                            child_returncode = observed_returncode
                    else:
                        stream_writers_quiescent = False
                        termination_notes.append(
                            "child_main_not_reaped_after_SIGKILL_deadline"
                        )
                except BaseException as kill_error:
                    stream_writers_quiescent = False
                    termination_notes.append(
                        f"kill_failed: {type(kill_error).__name__}: {kill_error}"
                    )
        if child_pid is not None:
            try:
                ensure_process_group_gone(child_pid, termination_notes)
            except BaseException as group_error:
                stream_writers_quiescent = False
                termination_notes.append(
                    f"group_cleanup_failed: {type(group_error).__name__}: {group_error}"
                )
        error_text = f"{type(error).__name__}: {error}"
        preserved_traceback = (
            "\n[launcher-preserved-error]\n" + traceback.format_exc()
        ).encode("utf-8", errors="replace")
        if stderr_fd is not None and stream_writers_quiescent:
            try:
                append_fd(stderr_fd, preserved_traceback)
            except BaseException as append_error:
                termination_notes.append(
                    f"stderr_append_failed: {type(append_error).__name__}: {append_error}"
                )
                stderr_payload += preserved_traceback
        else:
            stderr_payload += preserved_traceback
    finally:
        if memfd is not None:
            try:
                os.close(memfd)
            except BaseException as close_error:
                termination_notes.append(
                    f"memfd_close_failed: {type(close_error).__name__}: {close_error}"
                )
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except BaseException as close_error:
                termination_notes.append(
                    f"devnull_close_failed: {type(close_error).__name__}: {close_error}"
                )
    evidence_errors: list[str] = []
    if stdout_fd is not None and not stream_writers_quiescent:
        evidence_errors.append("stdout: unstable_live_writers_not_sealed")
        stdout_record = live_fd_metadata(stdout_fd, CHILD_STDOUT)
        try:
            os.close(stdout_fd)
        except BaseException as error:
            evidence_errors.append(f"stdout_close: {type(error).__name__}: {error}")
    elif stdout_fd is not None:
        try:
            stdout_payload, stdout_record = seal_reserved_stream(
                stdout_fd, CHILD_STDOUT
            )
        except BaseException as error:
            evidence_errors.append(f"stdout: {type(error).__name__}: {error}")
            try:
                os.fchmod(stdout_fd, EVIDENCE_MODE)
                os.fsync(stdout_fd)
            except BaseException as freeze_error:
                evidence_errors.append(
                    f"stdout_freeze: {type(freeze_error).__name__}: {freeze_error}"
                )
            stdout_record = safe_descriptor(CHILD_STDOUT)
        finally:
            try:
                os.close(stdout_fd)
            except BaseException as error:
                evidence_errors.append(
                    f"stdout_close: {type(error).__name__}: {error}"
                )
    else:
        try:
            stdout_record = exclusive_bytes(CHILD_STDOUT, stdout_payload)
        except BaseException as error:
            evidence_errors.append(f"stdout: {type(error).__name__}: {error}")
            stdout_record = safe_descriptor(CHILD_STDOUT)
    if stderr_fd is not None and not stream_writers_quiescent:
        evidence_errors.append("stderr: unstable_live_writers_not_sealed")
        stderr_record = live_fd_metadata(stderr_fd, CHILD_STDERR)
        try:
            os.close(stderr_fd)
        except BaseException as error:
            evidence_errors.append(f"stderr_close: {type(error).__name__}: {error}")
    elif stderr_fd is not None:
        try:
            stderr_payload, stderr_record = seal_reserved_stream(
                stderr_fd, CHILD_STDERR
            )
        except BaseException as error:
            evidence_errors.append(f"stderr: {type(error).__name__}: {error}")
            try:
                os.fchmod(stderr_fd, EVIDENCE_MODE)
                os.fsync(stderr_fd)
            except BaseException as freeze_error:
                evidence_errors.append(
                    f"stderr_freeze: {type(freeze_error).__name__}: {freeze_error}"
                )
            stderr_record = safe_descriptor(CHILD_STDERR)
        finally:
            try:
                os.close(stderr_fd)
            except BaseException as error:
                evidence_errors.append(
                    f"stderr_close: {type(error).__name__}: {error}"
                )
    else:
        try:
            stderr_record = exclusive_bytes(CHILD_STDERR, stderr_payload)
        except BaseException as error:
            evidence_errors.append(f"stderr: {type(error).__name__}: {error}")
            stderr_record = safe_descriptor(CHILD_STDERR)
    inputs_after = safe_input_snapshot()
    launcher_after_safe = safe_descriptor(SCRIPT)
    attempt_after_safe = safe_descriptor(ATTEMPT)
    output_after_safe = (
        safe_descriptor(OUTPUT)
        if stream_writers_quiescent
        else metadata_only(OUTPUT)
    )
    pending_guarded_signals = set(signal.sigpending()).intersection(guarded_signals)
    if pending_guarded_signals:
        pending_text = "pending guarded signals before execution commit: " + ",".join(
            str(int(value)) for value in sorted(pending_guarded_signals)
        )
        termination_notes.append(pending_text)
        if error_text is None:
            error_text = f"LauncherSignal: {pending_text}"
    completed_validated = (
        error_text is None
        and not evidence_errors
        and stream_writers_quiescent
        and child_returncode == 0
        and validation is not None
        and all(validation.values())
        and same_snapshot(inputs_before, inputs_after)
        and same_snapshot(
            {"launcher": launcher_before},
            {"launcher": launcher_after_safe},
        )
        and same_snapshot(
            {"output": output_validation_record or {}},
            {"output": output_after_safe},
        )
        and output_after_safe.get("exists") is True
        and output_after_safe.get("mode_octal") == "0444"
        and attempt_record is not None
        and attempt_after_safe.get("sha256") == attempt_record.get("sha256")
        and attempt_after_safe.get("mode_octal") == "0444"
        and same_snapshot(
            {"attempt": attempt_record},
            {"attempt": attempt_after_safe},
        )
    )
    execution = {
        "schema_version": SCHEMA,
        "status": (
            "one_shot_completed_validated"
            if completed_validated
            else "one_shot_failed_evidence_preserved"
        ),
        "attempt_committed": attempt_record is not None,
        "attempt": attempt_after_safe,
        "launcher_before": launcher_before,
        "launcher_after": launcher_after_safe,
        "solver": {
            "path": str(SOLVER.relative_to(ROOT)),
            "sha256": SOLVER_SHA256,
            "memfd": memfd_audit,
        },
        "argv": argv,
        "python": str(EXPECTED_PYTHON),
        "runtime_checks": runtime_checks,
        "child_returncode": child_returncode,
        "child_error": error_text,
        "termination_notes": termination_notes,
        "stream_writers_quiescent": stream_writers_quiescent,
        "evidence_errors": evidence_errors,
        "child_stdout": stdout_record,
        "child_stderr": stderr_record,
        "child_summary": child_summary,
        "child_validation": validation,
        "output": output_after_safe,
        "inputs_before": inputs_before,
        "inputs_after": inputs_after,
        "inputs_stable": same_snapshot(inputs_before, inputs_after),
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    late_pending_signals = set(signal.sigpending()).intersection(guarded_signals)
    if late_pending_signals:
        late_pending_text = "late pending guarded signals before execution write: " + ",".join(
            str(int(value)) for value in sorted(late_pending_signals)
        )
        if late_pending_text not in termination_notes:
            termination_notes.append(late_pending_text)
        if error_text is None:
            error_text = f"LauncherSignal: {late_pending_text}"
        completed_validated = False
        execution["status"] = "one_shot_failed_evidence_preserved"
        execution["child_error"] = error_text
        execution["termination_notes"] = termination_notes
    execution_publish_error: str | None = None
    try:
        if execution_fd is None:
            execution_record = exclusive_json(EXECUTION, execution)
        else:
            execution_record = commit_reserved_json(execution_fd, EXECUTION, execution)
    except BaseException as error:
        execution_publish_error = f"{type(error).__name__}: {error}"
        emergency = {
            "schema_version": SCHEMA,
            "status": "one_shot_failed_emergency_execution_evidence",
            "attempt_committed": attempt_record is not None,
            "original_child_error": error_text,
            "execution_publish_error": execution_publish_error,
            "child_returncode": child_returncode,
            "termination_notes": termination_notes,
            "attempt": safe_descriptor(ATTEMPT),
            "child_stdout": stdout_record,
            "child_stderr": stderr_record,
            "output": output_after_safe,
            "official_unique_changed_candidate_count_consumed": 0,
            "submission_performed": False,
        }
        try:
            if execution_fd is None:
                execution_record = exclusive_json(EXECUTION, emergency)
            else:
                execution_record = commit_reserved_json(
                    execution_fd, EXECUTION, emergency
                )
        except BaseException as emergency_error:
            execution_record = safe_descriptor(EXECUTION)
            execution_publish_error += (
                "; emergency_failed: "
                f"{type(emergency_error).__name__}: {emergency_error}"
            )
    finally:
        if execution_fd is not None:
            try:
                os.close(execution_fd)
            except BaseException as close_error:
                if execution_publish_error is None:
                    execution_publish_error = (
                        f"execution_close: {type(close_error).__name__}: {close_error}"
                    )
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK, original_signal_mask)
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)
    if execution_publish_error is not None:
        raise LauncherError(
            "execution evidence publication failed after attempt: "
            f"{execution_publish_error}; descriptor={execution_record}"
        )
    if not completed_validated:
        raise LauncherError(
            "one-shot failed after attempt; evidence preserved at "
            f"{execution_record['path']}"
        )
    final_summary = canonical_json(
        {
            "schema_version": SCHEMA,
            "status": execution["status"],
            "decision": child_document["decision"] if child_document else None,
            "candidate_model_state_sha256": (
                child_document.get("endpoint", {}).get("trial", {}).get(
                    "candidate_model_state_sha256"
                )
                if child_document
                else None
            ),
            "execution": execution_record,
            "output": output_after_safe,
            "official_unique_changed_candidate_count_consumed": 0,
            "submission_performed": False,
        }
    )
    try:
        write_all_fd(1, final_summary)
    except (BrokenPipeError, OSError):
        pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-shot launcher for the frozen CW21 historical-CW11 cache512 screen."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import lzma
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "run_cw21_cw11_fixed_pcgrad_trainonly_one_shot_v1.py"
SOLVER = TOOLS / "probe_u468_cw11_fixed_pcgrad_specialbc_cw21_v1.py"
BASE = TOOLS / "run_cw20_raw_fixed_pcgrad_trainonly_one_shot_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-cw21-cw11-fixed-pcgrad-cache512-one-shot-launcher-v1"

ATTEMPT = ROOT / "artifacts/.ptcg-cw21_cw11_fixed_pcgrad_trainonly_20260803_v1-attempt.json"
CHILD_STDOUT = ROOT / "artifacts/cw21_cw11_fixed_pcgrad_trainonly_20260803_v1.stdout.json"
CHILD_STDERR = ROOT / "artifacts/cw21_cw11_fixed_pcgrad_trainonly_20260803_v1.stderr.txt"
EXECUTION = ROOT / "artifacts/cw21_cw11_fixed_pcgrad_trainonly_20260803_v1.execution.json"

FROZEN_MODE = 0o555
SOLVER_SHA256 = "9c18db25e16fe6b9feb1ca72d0682a1baa31ad6dc19e1a6283fd7db2da0f9cf4"
BASE_SHA256 = "ab05db655b9c375929f4aa6219cc693ac6b2e11b89fbec173370e2974dd004d6"
DIRECT_INPUTS: dict[Path, tuple[str, int]] = {
    SOLVER: (SOLVER_SHA256, FROZEN_MODE),
    BASE: (BASE_SHA256, FROZEN_MODE),
    TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py": (
        "8f5b64c8a04b3de9a6d2890aac582967a50f4d708f3a11a64cbd395423d907d2",
        FROZEN_MODE,
    ),
    TOOLS / "probe_u468_cw11_bootstrap_pcgrad_specialbc_cw19_v1.py": (
        "65f16009641481cd13538714520828640d00aed42aae920e24d9b70910423ae8",
        FROZEN_MODE,
    ),
    TOOLS / "probe_u468_cw11_consumed_valid_official6_cw15_cuttingplane_v1.py": (
        "2aeb770f8774b7f7407a93553405089511ab2d2f278653162274a3afb89a2f24",
        FROZEN_MODE,
    ),
    TOOLS / "probe_u468_raw_aggregate512_actor6_gradients.py": (
        "ed85b59bd69558ab6f027ba21df9cb56c744baf9793bbb6b550d46bffbc3389a",
        FROZEN_MODE,
    ),
    TOOLS / "run_u468_raw_trainhard_actor6_balanced_mix_sweep.py": (
        "419feeeb5644adf34a8d56142cfcd87b3a4b79446d95b9b566bf9fcabdb721b1",
        FROZEN_MODE,
    ),
}
ACTOR_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
ACTOR_RANGES = (
    (0, 16384),
    (16384, 32768),
    (32768, 65536),
    (65536, 65664),
    (65664, 65792),
    (65792, 65793),
)
ACTOR_SHAPES = (
    (128, 128),
    (128, 128),
    (128, 256),
    (128,),
    (1, 128),
    (1,),
)


class LauncherError(RuntimeError):
    """Fail-closed CW21 launcher error."""


def exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_base() -> ModuleType:
    before = BASE.lstat()
    source = BASE.read_bytes()
    after = BASE.lstat()
    checks = {
        "regular": stat.S_ISREG(after.st_mode) and not stat.S_ISLNK(after.st_mode),
        "single_link": int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == FROZEN_MODE,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": hashlib.sha256(source).hexdigest() == BASE_SHA256,
    }
    if not all(checks.values()):
        raise LauncherError(f"frozen launcher engine drift: {checks}")
    name = "cw21_frozen_one_shot_engine"
    module = ModuleType(name)
    module.__file__ = str(BASE)
    module.__package__ = ""
    sys.modules[name] = module
    code = compile(source, str(BASE), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def validate_go_payload(endpoint: dict[str, Any]) -> dict[str, Any]:
    payload = endpoint.get("candidate_payload")
    if not isinstance(payload, dict):
        return {"pass": False, "checks": {"payload_dictionary": False}}
    anchor = payload.get("anchor")
    layout = payload.get("actor_layout")
    encoded = payload.get("candidate_actor_float32_le")
    reconstruction = endpoint.get("pure_payload_reconstruction")
    if not all(
        isinstance(value, dict) for value in (anchor, encoded, reconstruction)
    ) or not isinstance(layout, list):
        return {"pass": False, "checks": {"payload_structure": False}}
    layout_records_valid = len(layout) == 6
    if layout_records_valid:
        for record, name, (start, stop), expected_shape in zip(
            layout, ACTOR_NAMES, ACTOR_RANGES, ACTOR_SHAPES, strict=True
        ):
            shape = record.get("shape")
            product = 1
            if isinstance(shape, list) and shape:
                for value in shape:
                    if not isinstance(value, int) or value <= 0:
                        product = -1
                        break
                    product *= value
            else:
                product = -1
            layout_records_valid = layout_records_valid and (
                record.get("name") == name
                and record.get("dtype") == "torch.float32"
                and tuple(shape) == expected_shape
                and record.get("start") == start
                and record.get("stop") == stop
                and record.get("numel") == stop - start == product
            )
    layout_sha = hashlib.sha256(
        json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    chunks = encoded.get("base64_chunks_76")
    bounded_chunks = (
        isinstance(chunks, list)
        and bool(chunks)
        and all(isinstance(value, str) and 1 <= len(value) <= 76 for value in chunks)
        and sum(len(value) for value in chunks) <= 1_000_000
    )
    compressed = b""
    raw = b""
    decode_ok = False
    if bounded_chunks:
        try:
            compressed = base64.b64decode("".join(chunks), validate=True)
            raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
            decode_ok = True
        except (ValueError, binascii.Error, lzma.LZMAError):
            decode_ok = False
    reconstruction_checks = reconstruction.get("checks")
    expected_reconstruction_keys = {
        "raw_U468_restore_exact",
        "terminal_hash_exact",
        "actor_bytes_exact",
        "nonactor_still_exact_raw",
    }
    expected_reconstruction_root_keys = {
        "checks",
        "pass",
        "reconstruction_base_model_state_sha256",
        "model_state_sha256",
    }
    terminal_hash = endpoint.get("candidate_model_state_sha256")
    expected_payload_keys = {
        "anchor",
        "formula",
        "actor_names",
        "actor_layout",
        "actor_layout_sha256",
        "CW11_actor_float32_le_sha256",
        "candidate_actor_float32_le",
        "planned_step_l2",
        "actual_additional_from_CW11_l2",
        "actual_delta_float64_le_sha256",
        "pcgrad_direction_float64_le_sha256",
    }
    expected_encoded_keys = {
        "dtype",
        "raw_bytes",
        "raw_sha256",
        "compression",
        "compressed_bytes",
        "compressed_sha256",
        "base64_chunks_76",
    }
    checks = {
        "payload_keys_exact": set(payload) == expected_payload_keys,
        "terminal_model_hash_hex64": isinstance(terminal_hash, str)
        and len(terminal_hash) == 64
        and all(character in "0123456789abcdef" for character in terminal_hash),
        "anchor_exact": anchor == {
            "reconstruction_base": "original_raw_U468",
            "raw_checkpoint": (
                "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_"
                "design202608090/ppo_stage/B_gold_league/seed-202607336/"
                "checkpoints/update-0468.pt"
            ),
            "raw_checkpoint_sha256": (
                "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
            ),
            "raw_model_state_sha256": (
                "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
            ),
            "CW11_provenance_model_state_sha256": (
                "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
            ),
            "CW11_provenance_vector_float64_le_sha256": (
                "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
            ),
            "CW11_provenance_active_pair_ledger_sha256": (
                "6ae0c07a94627cb6fd0155f265f75d13bd7505b0e5f62cd62c3e26ceb7714436"
            ),
            "CW11_provenance_actor_float32_le_sha256": (
                "fc13661fec1801768a11d2c0366602727df08d6773c982f2ba273e194b8fafa6"
            ),
            "CW11_materialized_eval_only_checkpoint_used": False,
            "terminal_model_state_sha256": terminal_hash,
        },
        "formula_exact": payload.get("formula")
        == (
            "load_original_raw_U468_then_replace_only_six_absolute_"
            "actor_float32_tensors_from_frozen_payload"
        ),
        "actor_names_exact": payload.get("actor_names") == list(ACTOR_NAMES),
        "actor_layout_exact": layout_records_valid
        and payload.get("actor_layout_sha256") == layout_sha,
        "CW11_actor_sha_exact": payload.get("CW11_actor_float32_le_sha256")
        == "fc13661fec1801768a11d2c0366602727df08d6773c982f2ba273e194b8fafa6",
        "step_contract_exact": payload.get("planned_step_l2") == 1.25e-4
        and isinstance(payload.get("actual_additional_from_CW11_l2"), float)
        and 1.245e-4
        <= payload["actual_additional_from_CW11_l2"]
        <= 1.255e-4,
        "delta_hashes_hex64": all(
            isinstance(payload.get(key), str)
            and len(payload[key]) == 64
            and all(character in "0123456789abcdef" for character in payload[key])
            for key in (
                "actual_delta_float64_le_sha256",
                "pcgrad_direction_float64_le_sha256",
            )
        ),
        "encoded_metadata_exact": set(encoded) == expected_encoded_keys
        and encoded.get("dtype") == "<f4"
        and exact_int(encoded.get("raw_bytes"), 263172)
        and encoded.get("compression") == "XZ_preset9_extreme_CRC64",
        "encoded_bounded_and_decoded": bounded_chunks and decode_ok,
        "compressed_identity_exact": exact_int(
            encoded.get("compressed_bytes"), len(compressed)
        )
        and hashlib.sha256(compressed).hexdigest()
        == encoded.get("compressed_sha256"),
        "raw_actor_identity_exact": len(raw) == 263172
        and hashlib.sha256(raw).hexdigest() == encoded.get("raw_sha256"),
        "candidate_actor_changed_from_CW11": encoded.get("raw_sha256")
        != "fc13661fec1801768a11d2c0366602727df08d6773c982f2ba273e194b8fafa6",
        "reconstruction_exact": reconstruction.get("pass") is True
        and set(reconstruction) == expected_reconstruction_root_keys
        and isinstance(reconstruction_checks, dict)
        and set(reconstruction_checks) == expected_reconstruction_keys
        and all(value is True for value in reconstruction_checks.values())
        and reconstruction.get("reconstruction_base_model_state_sha256")
        == "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
        and reconstruction.get("model_state_sha256") == terminal_hash,
    }
    return {"pass": all(checks.values()), "checks": checks}


def validate_child(engine: ModuleType, document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise LauncherError("child stdout is not one JSON object")
    decision = document.get("decision")
    allowed = {
        "GO_CW21_CACHE512_PREFLIGHT",
        "NO_GO_CW21_CACHE512_PREFLIGHT",
    }
    endpoint = document.get("endpoint", {})
    contract = document.get("contract", {})
    historical = document.get("historical_exact_CW11_replay", {})
    train_cache = document.get("train_cache", {})
    integrity = document.get("integrity", {})
    runtime = integrity.get("runtime", {})
    cuda = runtime.get("cuda", {})
    runtime_checks = runtime.get("checks", {})
    bytecode_cache = runtime.get("bytecode_cache_isolation", {})
    source = integrity.get("source", {})
    final_checks = endpoint.get("final_checks")
    expected_go_keys = {
        "integrity",
        "CW11_baseline_retention_contract",
        "cache512_train_gate",
        "pure_payload_reconstruction",
    }
    go_final_exact = (
        isinstance(final_checks, dict)
        and set(final_checks) == expected_go_keys
        and all(final_checks[key] is True for key in expected_go_keys)
    )
    expected_cuda_keys = {
        "available",
        "native_bf16",
        "device_count_exact_1",
        "current_device_exact_0",
        "device_name_exact",
        "compute_capability_exact_12_0",
        "torch_version_exact",
        "cuda_runtime_exact",
    }
    expected_runtime_keys = {
        "cwd_exact",
        "python_exact_my_project_env",
        "isolated",
        "dont_write_bytecode",
        "cublas_workspace_exact",
    }
    expected_source_check_keys = {
        "no_forbidden_named_calls",
        "no_direct_torch_checkpoint_load",
        "exact_two_train_cache_evaluation_sites",
        "exact_one_historical_run_probe_site",
        "stdout_print_present",
    }
    expected_historical_check_keys = {
        "callback_once",
        "exact_historical_CW11_status",
        "historical_consumer_called",
        "historical_CW11_model_exact",
        "historical_CW11_vector_exact",
        "historical_CW11_ledger_exact",
        "outer_restore_pass",
    }
    historical_checks = historical.get("checks")
    payload = endpoint.get("candidate_payload")
    payload_anchor = payload.get("anchor", {}) if isinstance(payload, dict) else {}
    go_payload_audit = validate_go_payload(endpoint) if isinstance(payload, dict) else None
    checks = {
        "schema_exact": document.get("schema_version")
        == "ptcg-u468-cw11-fixed-pcgrad-specialbc-cw21-v1",
        "status_decision_exact": document.get("status") == decision,
        "endpoint_decision_exact": endpoint.get("decision") == decision,
        "decision_allowed": decision in allowed,
        "cache512_scope_exact": contract.get("promotion_scope")
        == "cache512_train_only_preflight_not_fulltrain_or_specialist",
        "generic_not_targeted_disclosed": "not_targeted_special_BC"
        in str(contract.get("special_BC")),
        "resume_forbidden_respected": contract.get(
            "CW11_resume_forbidden_respected"
        )
        is True
        and contract.get("CW11_materialized_eval_only_checkpoint_opened") is False,
        "official_budget_zero": exact_int(
            contract.get("official_candidate_budget_consumed"), 0
        ),
        "all_changed_candidate_nontrain_evaluations_zero": all(
            exact_int(contract.get(key), 0)
            for key in (
                "changed_candidate_fulltrain_evaluation_count",
                "changed_candidate_specialist_or_validation_evaluation_count",
                "changed_candidate_official6_evaluation_count",
                "changed_candidate_official_or_validation_evaluation_count",
            )
        ),
        "historical_replay_disclosed": historical.get("pass") is True
        and exact_int(historical.get("historical_replay_count"), 1)
        and exact_int(historical.get("historical_specialist_valid_rows_replayed"), 4)
        and historical.get("specialist_valid_consumed_for_optimization") is True
        and historical.get("promotion_evidence") is False
        and historical.get("changed_candidate_official_or_validation_evaluation_count")
        is not None
        and exact_int(
            historical.get(
                "changed_candidate_official_or_validation_evaluation_count"
            ),
            0,
        ),
        "historical_checks_all_true": isinstance(historical_checks, dict)
        and set(historical_checks) == expected_historical_check_keys
        and all(value is True for value in historical_checks.values()),
        "new_candidate_train_cache_only": train_cache.get("non_train_members_opened")
        is False,
        "raw_profile_staleness_disclosed": train_cache.get("profile_anchor")
        == "raw_U468_frozen_selection_not_reprofiled_at_CW11",
        "no_writes_inside_child": exact_int(contract.get("checkpoint_writes"), 0)
        and exact_int(contract.get("model_writes"), 0)
        and exact_int(contract.get("result_artifact_writes"), 0),
        "no_optimizer_or_parameter_projection": exact_int(
            contract.get("optimizer_instances"), 0
        )
        and exact_int(contract.get("optimizer_steps"), 0)
        and exact_int(contract.get("backward_calls"), 0)
        and exact_int(contract.get("parameter_space_projections"), 0),
        "pcgrad_projection_disclosed": contract.get(
            "pcgrad_gradient_conflict_projection"
        )
        is True,
        "runtime_gpu_contract": isinstance(cuda, dict)
        and set(cuda) == expected_cuda_keys
        and all(value is True for value in cuda.values()),
        "runtime_process_contract": isinstance(runtime_checks, dict)
        and set(runtime_checks) == expected_runtime_keys
        and all(value is True for value in runtime_checks.values()),
        "runtime_pass": runtime.get("pass") is True,
        "repo_bytecode_cache_isolated": bytecode_cache.get("pass") is True
        and bytecode_cache.get("policy")
        == "repo_imports_compile_from_source_because_cache_prefix_is_ENOTDIR"
        and bytecode_cache.get("checks")
        == {
            "pycache_prefix_exact_dev_null": True,
            "dont_write_bytecode": True,
            "dev_null_character_device": True,
            "dev_null_device_exact_1_3": True,
        },
        "source_exact": source.get("pass") is True
        and source.get("source_sha256") == SOLVER_SHA256
        and isinstance(source.get("checks"), dict)
        and set(source["checks"]) == expected_source_check_keys
        and all(value is True for value in source["checks"].values()),
        "submission_false": document.get("submission_performed") is False,
        "writes_zero": exact_int(document.get("writes_performed"), 0),
        "endpoint_count_bound": contract.get("changed_candidate_train_endpoint_count")
        == endpoint.get("changed_candidate_train_endpoint_count")
        and type(endpoint.get("changed_candidate_train_endpoint_count")) is int
        and endpoint.get("changed_candidate_train_endpoint_count") in {0, 1},
        "endpoint_payload_consistent": (
            decision == "GO_CW21_CACHE512_PREFLIGHT"
            and isinstance(payload, dict)
            and exact_int(endpoint.get("changed_candidate_train_endpoint_count"), 1)
            and go_final_exact
            and endpoint.get("pure_payload_reconstruction", {}).get("pass") is True
            and go_payload_audit is not None
            and go_payload_audit.get("pass") is True
            and payload_anchor.get("reconstruction_base") == "original_raw_U468"
            and payload_anchor.get("raw_model_state_sha256")
            == "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
            and payload_anchor.get("raw_checkpoint_sha256")
            == "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
            and payload_anchor.get("CW11_materialized_eval_only_checkpoint_used")
            is False
            and payload_anchor.get("terminal_model_state_sha256")
            == endpoint.get("candidate_model_state_sha256")
        )
        or (
            decision == "NO_GO_CW21_CACHE512_PREFLIGHT"
            and payload is None
            and type(endpoint.get("changed_candidate_train_endpoint_count")) is int
            and endpoint.get("changed_candidate_train_endpoint_count") in {0, 1}
        ),
    }
    if decision == "NO_GO_CW21_CACHE512_PREFLIGHT":
        changed_count = endpoint.get("changed_candidate_train_endpoint_count")
        if changed_count == 0:
            direction = endpoint.get("pcgrad", {}).get("direction_gate")
            expected_direction_keys = {
                "task_order_exact",
                "original_unprojected_references",
                "arithmetic_mean",
                "four_tasks_plus_retention_robust_descent",
                "minimum_directional_gate_cosine",
                "direction_finite_nonzero",
            }
            coherent = (
                endpoint.get("reason") == "PCGRAD_DIRECTION_GATE_FAILED"
                and final_checks is None
                and isinstance(direction, dict)
                and set(direction) == expected_direction_keys
                and all(isinstance(value, bool) for value in direction.values())
                and not all(value is True for value in direction.values())
            )
        else:
            reason = endpoint.get("reason")
            base_keys = {
                "integrity",
                "CW11_baseline_retention_contract",
                "cache512_train_gate",
            }
            coherent = isinstance(final_checks, dict)
            if reason == "integrity_gate_failed":
                coherent = coherent and set(final_checks) == base_keys and (
                    final_checks.get("integrity") is False
                ) and all(isinstance(value, bool) for value in final_checks.values())
            elif reason == "cache512_train_gate_failed":
                coherent = coherent and set(final_checks) == base_keys and all(
                    isinstance(value, bool) for value in final_checks.values()
                ) and final_checks == {
                    "integrity": True,
                    "CW11_baseline_retention_contract": True,
                    "cache512_train_gate": False,
                }
            elif reason == "payload_reconstruction_gate_failed":
                coherent = coherent and set(final_checks) == expected_go_keys and all(
                    isinstance(value, bool) for value in final_checks.values()
                ) and final_checks == {
                    "integrity": True,
                    "CW11_baseline_retention_contract": True,
                    "cache512_train_gate": True,
                    "pure_payload_reconstruction": False,
                }
            else:
                coherent = False
        checks["no_go_reason_coherent"] = coherent
    else:
        checks["go_reason_coherent"] = (
            endpoint.get("reason") == "all_cache512_train_only_gates_passed"
        )
    if not all(checks.values()):
        raise LauncherError(f"child semantic validation failed: {checks}")
    return {
        "checks": checks,
        "pass": True,
        "decision": decision,
        "expected_GO_final_check_keys": sorted(expected_go_keys),
        "child_reported_inputs_post": engine.validate_child_reported_inputs(document),
    }


def static_audit(engine: ModuleType) -> dict[str, Any]:
    targets = (ATTEMPT, CHILD_STDOUT, CHILD_STDERR, EXECUTION)
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "pycache_prefix_exact_dev_null": sys.pycache_prefix == "/dev/null",
        "launcher_path_exact": Path(__file__).resolve() == SCRIPT.resolve(),
        "targets_absent": not any(path.exists() or path.is_symlink() for path in targets),
    }
    if not all(checks.values()):
        raise LauncherError(f"static checks failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_passed",
        "checks": checks,
        "pass": True,
        "launcher": engine.regular_evidence(
            SCRIPT, None, stat.S_IMODE(SCRIPT.lstat().st_mode)
        ),
        "frozen_engine": engine.regular_evidence(BASE, BASE_SHA256, FROZEN_MODE),
        "direct_inputs": engine.input_snapshot(),
        "command": [
            str(EXPECTED_PYTHON),
            "-I",
            "-B",
            str(SOLVER),
            "--mode",
            "run",
            "--device",
            "cuda",
        ],
        "targets": [str(path.relative_to(ROOT)) for path in targets],
        "writes_performed": 0,
        "submission_performed": False,
    }


def configured_engine() -> ModuleType:
    engine = import_base()
    engine.SCRIPT = SCRIPT
    engine.SOLVER = SOLVER
    engine.SCHEMA = SCHEMA
    engine.ATTEMPT = ATTEMPT
    engine.CHILD_STDOUT = CHILD_STDOUT
    engine.CHILD_STDERR = CHILD_STDERR
    engine.EXECUTION = EXECUTION
    engine.SOLVER_SHA256 = SOLVER_SHA256
    engine.DIRECT_INPUTS = DIRECT_INPUTS
    engine.validate_child = lambda document: validate_child(engine, document)
    engine.static_audit = lambda: static_audit(engine)
    return engine


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise LauncherError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise LauncherError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise LauncherError("requires Python -I -B")
    engine = configured_engine()
    if args.mode == "static":
        print(json.dumps(static_audit(engine), sort_keys=True, ensure_ascii=False))
        return 0
    try:
        return engine.run_once()
    except BaseException as error:
        engine.preserve_failure(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""RAM-only six-panel specialist 60/60 gate for the frozen CW23 payload."""

from __future__ import annotations

import argparse
import ast
import fcntl
import gc
import hashlib
import json
import multiprocessing
import os
import stat
import sys
import traceback
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_cw23_payload_specialist60_gate_v1.py"
LAUNCHER_RELATIVE = "tools/launch_cw23_payload_specialist60_gate_v1.py"
SCHEMA = "ptcg-cw23-payload-specialist60-gate-v1"
FROZEN_MODE = 0o555
EVIDENCE_MODE = 0o444
EXECUTION_BINDING = globals().get("_PTCG_HASH_BOUND_EXECUTION")

BASE = TOOLS / "run_cw23_payload_fulltrain_gate_v1.py"
BASE_SHA256 = "70f7a5dc10f6f0cc6e5dc863b9592a292c2802a62584b3ef7c2c17776fe39130"
FULLTRAIN_RESULT = ROOT / "artifacts/cw23_payload_fulltrain_gate_20260803_v1.json"
FULLTRAIN_RESULT_SHA256 = "1270a08066ddefcb54038b8695ca77a7ee3a402933f338f485821f357e7b4b68"
LEGACY = TOOLS / "run_e904_single_endpoint_specialist.py"
LEGACY_SHA256 = "f749fd660209e2369d6cdf21e91ca0a5234eb3306d50af98165cac502f37e6c7"

CW11_ROOT = ROOT / (
    "artifacts/cw11_4317_specialist_devconsistency_20260802_v1."
    "specialist_dev_consistency"
)
CW11_MANIFEST = CW11_ROOT / "dev_consistency_execution_manifest.json"
CW11_MANIFEST_SHA256 = "47ebd3683b73f397ffff5c6874136c9f12d23ee8eb619a9705445fabd4b73a27"
CW11_DECISION = CW11_ROOT / "dev_consistency_decision.json"
CW11_DECISION_SHA256 = "9c925a66f116a10b0ac996a76fdcbe0c378f0e4d02d45c635a2d47228b189a35"

EVALUATOR = TOOLS / "evaluate_policy_bc.py"
EVALUATOR_SHA256 = "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4"
BC = TOOLS / "train_bc_orbit.py"
BC_SHA256 = "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
PPO = TOOLS / "train_ppo.py"
PPO_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"

PF = ROOT / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip"
FLG = ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip"
CORE5 = ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip"
ARCHIVES = {
    PF: ("71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598", 0o664),
    FLG: ("4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8", 0o664),
    CORE5: ("bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a", 0o600),
}
PANEL_ORDER = ("pokemonfan", "flg", "core5", "dominic", "luca", "szlach")
PANELS = {
    "pokemonfan": (PF, ()),
    "flg": (FLG, ()),
    "core5": (CORE5, ()),
    "dominic": (CORE5, ("Dominic Peel",)),
    "luca": (CORE5, ("Luca",)),
    "szlach": (CORE5, ("szlachetny snieg",)),
}
METRIC_ORDER = (
    "rows",
    "context34_rows",
    "value",
    "count",
    "set",
    "hybrid",
    "ordered",
    "top1",
    "context34_hybrid",
    "context34_ordered",
)
EXACT_RELATIVE_METRICS = {"rows", "context34_rows", "value", "count"}
CW11_METRICS = {
    "pokemonfan": {"rows":15152,"context34_rows":58,"set":13018,"hybrid":13017,"ordered":12867,"value":11033,"count":15022,"top1":13119,"context34_hybrid":57,"context34_ordered":57},
    "flg": {"rows":2312,"context34_rows":3,"set":1762,"hybrid":1749,"ordered":1732,"value":1807,"count":2299,"top1":1776,"context34_hybrid":3,"context34_ordered":3},
    "core5": {"rows":8319,"context34_rows":43,"set":6549,"hybrid":6533,"ordered":6504,"value":6127,"count":8210,"top1":6655,"context34_hybrid":41,"context34_ordered":41},
    "dominic": {"rows":2623,"context34_rows":15,"set":1980,"hybrid":1973,"ordered":1972,"value":1940,"count":2579,"top1":2019,"context34_hybrid":13,"context34_ordered":13},
    "luca": {"rows":1378,"context34_rows":7,"set":1070,"hybrid":1070,"ordered":1060,"value":960,"count":1377,"top1":1087,"context34_hybrid":7,"context34_ordered":7},
    "szlach": {"rows":2567,"context34_rows":7,"set":2071,"hybrid":2071,"ordered":2062,"value":1906,"count":2510,"top1":2110,"context34_hybrid":7,"context34_ordered":7},
}

ATTEMPT = ROOT / "artifacts/.ptcg-cw23_specialist60_noharm_cw11_20260803_v1-attempt.json"
OUTPUT = ROOT / "artifacts/cw23_specialist60_noharm_cw11_20260803_v1.json"


class ProtocolError(RuntimeError):
    """Fail-closed specialist protocol error."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def lexists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def record_identity(value: os.stat_result) -> tuple[int, ...]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_mode), int(value.st_nlink), int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns))


def read_regular(path: Path, expected_sha256: str | None, expected_mode: int, label: str) -> tuple[bytes, dict[str, Any]]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != expected_mode:
        raise ProtocolError(f"{label}: unsafe type/link/mode")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
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
    if not record_identity(before) == record_identity(opened) == record_identity(after_fd) == record_identity(after_path):
        raise ProtocolError(f"{label}: identity changed during read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    if len(payload) != after_fd.st_size or (expected_sha256 is not None and digest != expected_sha256):
        raise ProtocolError(f"{label}: SHA/size drift")
    return payload, {"path":str(path.relative_to(ROOT)),"sha256":digest,"bytes":len(payload),"mode_octal":format(stat.S_IMODE(after_fd.st_mode),"04o"),"device":int(after_fd.st_dev),"inode":int(after_fd.st_ino),"nlink":int(after_fd.st_nlink)}


def hash_descriptor(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def open_held_archive(path: Path, expected_sha256: str, expected_mode: int, label: str) -> dict[str, Any]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != expected_mode:
        raise ProtocolError(f"{label}: unsafe type/link/mode")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        opened = os.fstat(fd)
        digest = hash_descriptor(fd)
        after = os.fstat(fd)
        visible = path.lstat()
        if not record_identity(before) == record_identity(opened) == record_identity(after) == record_identity(visible):
            raise ProtocolError(f"{label}: identity changed while pinning archive")
        if digest != expected_sha256:
            raise ProtocolError(f"{label}: SHA drift")
        return {
            "fd": fd,
            "path": path,
            "proc_path": Path(f"/proc/self/fd/{fd}"),
            "identity": record_identity(after),
            "expected_sha256": expected_sha256,
            "record": {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "bytes": int(after.st_size),
                "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
                "device": int(after.st_dev),
                "inode": int(after.st_ino),
                "nlink": int(after.st_nlink),
                "held_fd_path": f"/proc/self/fd/{fd}",
                "shared_flock": True,
            },
        }
    except BaseException:
        os.close(fd)
        raise


def verify_held_archive(held: Mapping[str, Any], label: str) -> dict[str, Any]:
    fd = int(held["fd"])
    digest = hash_descriptor(fd)
    after = os.fstat(fd)
    visible = Path(held["path"]).lstat()
    if record_identity(after) != tuple(held["identity"]) or record_identity(visible) != tuple(held["identity"]):
        raise ProtocolError(f"{label}: held archive identity changed during evaluation")
    if digest != held["expected_sha256"]:
        raise ProtocolError(f"{label}: held archive content changed during evaluation")
    return {"post_sha256":digest,"post_identity_exact":True,"visible_path_identity_exact":True}


def close_held_archive(held: Mapping[str, Any]) -> None:
    fd = int(held["fd"])
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class TrackedLoader:
    """Expose one DataLoader iterator while retaining its worker handles."""

    def __init__(self, loader: Any, expected_workers: int) -> None:
        self.loader = loader
        self.expected_workers = expected_workers
        self.iterator: Any | None = None
        self.workers: list[Any] = []

    def __iter__(self) -> Any:
        if self.iterator is not None:
            raise ProtocolError("specialist DataLoader was iterated more than once")
        self.iterator = iter(self.loader)
        self.workers = list(getattr(self.iterator, "_workers", ()) or ())
        if len(self.workers) != self.expected_workers:
            self.shutdown()
            raise ProtocolError("specialist DataLoader worker count drift")
        return self.iterator

    def shutdown(self) -> dict[str, Any]:
        iterator = self.iterator
        workers = list(self.workers)
        pids = [worker.pid for worker in workers]
        shutdown_callable = getattr(iterator, "_shutdown_workers", None) if iterator is not None else None
        cleanup_errors: list[str] = []
        graceful_shutdown_error: str | None = None
        alive_after_graceful: list[int | None] = []
        alive_after: list[int | None] = []
        terminated: list[int | None] = []
        killed: list[int | None] = []
        try:
            if iterator is not None and not callable(shutdown_callable):
                graceful_shutdown_error = "ProtocolError: specialist DataLoader lacks worker shutdown API"
            elif callable(shutdown_callable):
                try:
                    shutdown_callable()
                except BaseException as error:
                    graceful_shutdown_error = f"{type(error).__name__}: {error}"
            for worker in workers:
                try:
                    worker.join(timeout=10.0)
                except BaseException as error:
                    cleanup_errors.append(f"join {worker.pid}: {type(error).__name__}: {error}")
            alive_after_graceful = [worker.pid for worker in workers if worker.is_alive()]
            for worker in workers:
                if worker.is_alive():
                    terminated.append(worker.pid)
                    try:
                        worker.terminate()
                        worker.join(timeout=5.0)
                    except BaseException as error:
                        cleanup_errors.append(f"terminate {worker.pid}: {type(error).__name__}: {error}")
            for worker in workers:
                if worker.is_alive():
                    killed.append(worker.pid)
                    try:
                        worker.kill()
                        worker.join(timeout=5.0)
                    except BaseException as error:
                        cleanup_errors.append(f"kill {worker.pid}: {type(error).__name__}: {error}")
            alive_after = [worker.pid for worker in workers if worker.is_alive()]
        finally:
            self.iterator = None
            self.workers = []
        return {
            "requested_workers": self.expected_workers,
            "spawned_workers": len(workers),
            "worker_pids": pids,
            "all_pids_positive": len(pids) == self.expected_workers and all(type(pid) is int and pid > 0 for pid in pids),
            "explicit_shutdown_called": callable(shutdown_callable),
            "graceful_shutdown_error": graceful_shutdown_error,
            "alive_after_graceful_shutdown": alive_after_graceful,
            "forced_terminate_pids": terminated,
            "forced_kill_pids": killed,
            "cleanup_errors": cleanup_errors,
            "alive_after_shutdown": alive_after,
            "cleanup_complete": not alive_after,
            "pass": len(workers) == self.expected_workers and not alive_after and not alive_after_graceful and not terminated and not killed and not cleanup_errors and graceful_shutdown_error is None and len(pids) == self.expected_workers and all(type(pid) is int and pid > 0 for pid in pids),
        }


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ProtocolError(f"{label}: duplicate key {key}")
            result[key] = value
        return result
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(ProtocolError(f"{label}: nonfinite {token}")))
    if not isinstance(value, dict) or canonical_json(value) != payload:
        raise ProtocolError(f"{label}: noncanonical object")
    return value


def publish(path: Path, value: Any) -> dict[str, Any]:
    payload = canonical_json(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    observed: os.stat_result | None = None
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ProtocolError("short publication")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, EVIDENCE_MODE)
        os.fsync(fd)
        observed = os.fstat(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    visible = path.lstat()
    if observed is None or not stat.S_ISREG(visible.st_mode) or stat.S_ISLNK(visible.st_mode) or visible.st_nlink != 1 or stat.S_IMODE(visible.st_mode) != EVIDENCE_MODE or (observed.st_dev, observed.st_ino, observed.st_size) != (visible.st_dev, visible.st_ino, visible.st_size):
        raise ProtocolError("publication identity drift")
    reloaded, record = read_regular(path, sha256_bytes(payload), EVIDENCE_MODE, f"published {path.name}")
    if reloaded != payload:
        raise ProtocolError("publication reload drift")
    return record


def import_exact(path: Path, digest: str, name: str, mode: int = FROZEN_MODE) -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = read_regular(path, digest, mode, name)
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module, evidence


def import_standard_exact(path: Path, digest: str, name: str, mode: int) -> tuple[ModuleType, dict[str, Any]]:
    if name in sys.modules:
        raise ProtocolError(f"standard module was already loaded before exact binding: {name}")
    source, evidence = read_regular(path, digest, mode, f"exact module {name}")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    module.__cached__ = None
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, evidence


def validate_runtime(require_cuda: bool) -> dict[str, Any]:
    checks: dict[str, Any] = {"cwd_exact":Path.cwd().resolve()==ROOT,"python_exact":Path(sys.executable).resolve()==EXPECTED_PYTHON.resolve(),"isolated":sys.flags.isolated==1,"dont_write_bytecode":sys.flags.dont_write_bytecode==1,"optimize_zero":sys.flags.optimize==0}
    if require_cuda:
        import torch
        checks.update({"cuda_available":torch.cuda.is_available(),"cuda_count_positive":torch.cuda.device_count()>0,"cuda_bf16":torch.cuda.is_bf16_supported()})
    if not all(checks.values()):
        raise ProtocolError(f"runtime drift: {checks}")
    return checks


def validate_binding(source_record: Mapping[str, Any], audit_only: bool) -> dict[str, Any]:
    value = EXECUTION_BINDING
    checks = {"mapping":isinstance(value,Mapping),"schema":isinstance(value,Mapping) and value.get("schema_version")=="ptcg-cw23-specialist60-hash-bound-launch-v1","adapter_path":isinstance(value,Mapping) and value.get("adapter_path")==str(SCRIPT),"adapter_sha":isinstance(value,Mapping) and value.get("adapter_sha256")==source_record.get("sha256"),"adapter_bytes":isinstance(value,Mapping) and value.get("adapter_bytes")==source_record.get("bytes"),"adapter_mode":isinstance(value,Mapping) and value.get("adapter_mode_octal")=="0555","held_fd":isinstance(value,Mapping) and value.get("held_fd_identity_exact") is True,"same_bytes":isinstance(value,Mapping) and value.get("compile_exec_same_verified_bytes") is True,"audit_mode":isinstance(value,Mapping) and value.get("audit_only") is audit_only,"launcher":isinstance(value,Mapping) and isinstance(value.get("launcher"),Mapping) and value["launcher"].get("path")==LAUNCHER_RELATIVE and value["launcher"].get("mode_octal")=="0555" and value["launcher"].get("nlink")==1}
    if not all(checks.values()) or not isinstance(value, Mapping):
        raise ProtocolError(f"hash-bound launch drift: {checks}")
    copied = json.loads(canonical_json(value))
    return {"checks":checks,"binding":copied}


def source_audit(source: bytes, require_frozen: bool) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
    strings = [node.value for node in ast.walk(tree) if isinstance(node,ast.Constant) and isinstance(node.value,str)]
    forbidden = sorted(calls.intersection({"save","backward","step"}))
    checks = {"regular_single_link":SCRIPT.is_file() and not SCRIPT.is_symlink() and SCRIPT.stat().st_nlink==1,"mode_exact":(not require_frozen) or stat.S_IMODE(SCRIPT.stat().st_mode)==FROZEN_MODE,"no_save_backward_step":not forbidden,"one_output_path":strings.count(str(OUTPUT.relative_to(ROOT)))==1,"one_attempt_path":strings.count(str(ATTEMPT.relative_to(ROOT)))==1,"valid_split_literal":source.count(b'"valid"')>=1,"legacy_gate_api":all(token in source for token in (b"expected_thresholds",b"evaluate_gates",b"observed_metrics")),"RAM_reconstruction":b"reconstruct_states" in source}
    if not all(checks.values()):
        raise ProtocolError(f"source audit failed: {checks}; forbidden={forbidden}")
    return {"path":str(SCRIPT.relative_to(ROOT)),"sha256":sha256_bytes(source),"bytes":len(source),"mode_octal":format(stat.S_IMODE(SCRIPT.stat().st_mode),"04o"),"checks":checks}


def validate_fulltrain(document: Mapping[str, Any], base: ModuleType) -> None:
    scope = document.get("scope_audit", {})
    checks = {"status_GO":document.get("status")=="GO_CW23_FULLTRAIN_GATE","candidate_exact":document.get("candidate_model_state_sha256")==base.CW23_MODEL_SHA256,"decision_pass":document.get("decision",{}).get("pass") is True,"decision_status":document.get("decision",{}).get("status")=="full_train_gate_pass","materialization_allowed_next":document.get("decision",{}).get("materialization_allowed_next") is True,"execution_checks":isinstance(document.get("execution",{}).get("checks"),Mapping) and all(value is True for value in document["execution"]["checks"].values()),"reconstruction_checks":isinstance(document.get("reconstruction",{}).get("checks"),Mapping) and all(value is True for value in document["reconstruction"]["checks"].values()),"zero_official_count":document.get("official_unique_changed_candidate_count_consumed")==0,"no_submission":document.get("submission_performed") is False and document.get("package_upload_performed") is False,"all_train_rows":scope.get("all_24050_train_rows") is True,"no_validation":scope.get("validation_members_opened") is False,"no_test":scope.get("test_rows_opened") is False,"no_model_writes":scope.get("checkpoint_writes")==0 and scope.get("model_artifact_writes")==0}
    if not all(checks.values()):
        raise ProtocolError(f"fulltrain prerequisite drift: {checks}")


def validate_protocol(legacy: ModuleType, manifest: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    specs = {str(item["name"]):(ROOT/str(item["data_path"]),tuple(() if item.get("team_name") is None else (str(item["team_name"]),)),str(item["data_sha256"])) for item in legacy.PANEL_SPECS}
    expected_specs = {name:(path,teams,ARCHIVES[path][0]) for name,(path,teams) in PANELS.items()}
    ordered = manifest.get("ordered_evaluations", [])
    observed_cw11 = {str(item.get("panel")):{key:int(item.get("metrics",{}).get(key,-1)) for key in METRIC_ORDER} for item in ordered if isinstance(item,Mapping)}
    absolute_gate_count = sum(len(legacy.expected_thresholds(name)) for name in PANEL_ORDER)
    checks = {"panel_order_exact":tuple(str(item["name"]) for item in legacy.PANEL_SPECS)==PANEL_ORDER,"panel_specs_exact":specs==expected_specs,"metric_order_exact":all(set(legacy.expected_thresholds(name))==set(METRIC_ORDER) for name in PANEL_ORDER),"absolute_gate_count60":absolute_gate_count==60,"cw11_manifest_complete":manifest.get("status")=="six_dev_consistency_panels_completed" and manifest.get("all_six_attempted_once") is True and manifest.get("gate_count")==60,"cw11_order_exact":[item.get("panel") for item in ordered]==list(PANEL_ORDER),"cw11_metrics_exact":observed_cw11==CW11_METRICS,"cw11_known_decision":decision.get("status")=="failed_dev_consistency" and decision.get("pass") is False and decision.get("gate_count")==60 and decision.get("passed_gate_count")==51 and len(decision.get("failed_gates",[]))==9}
    if not all(checks.values()):
        raise ProtocolError(f"specialist protocol drift: {checks}")
    return checks


def evaluate_candidate(evaluator: ModuleType, legacy: ModuleType, helper: ModuleType, states: Mapping[str, Mapping[str, Any]], checkpoint: Mapping[str, Any], held_archives: Mapping[Path, Mapping[str, Any]], expected_candidate_sha256: str, expected_raw_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, str]]]]:
    torch = helper.torch
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    candidate_checkpoint = dict(checkpoint)
    candidate_checkpoint["model_state_dict"] = states["candidate"]
    kind = evaluator.checkpoint_kind(candidate_checkpoint)
    config = evaluator.model_config_from_checkpoint(candidate_checkpoint, kind)
    candidate_state_sha256 = helper.model_state_sha256(states["candidate"])
    raw_state_sha256 = helper.model_state_sha256(states["raw"])
    if kind != "ppo" or candidate_state_sha256 != expected_candidate_sha256 or raw_state_sha256 != expected_raw_sha256:
        raise ProtocolError("specialist RAM state identity drift")
    cublas_contract = ("CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    results: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    restores: list[dict[str, Any]] = []
    errors: dict[str, list[dict[str, str]]] = {}
    for ordinal, name in enumerate(PANEL_ORDER, start=1):
        archive, teams = PANELS[name]
        held = held_archives[archive]
        model: Any | None = None
        tracked: TrackedLoader | None = None
        shutdown: dict[str, Any] | None = None
        panel_result: dict[str, Any] | None = None
        panel_errors: list[dict[str, str]] = []
        live_candidate_sha256: str | None = None

        def record_error(stage: str, error: BaseException) -> None:
            panel_errors.append({"stage":stage,"error":f"{type(error).__name__}: {error}","traceback":traceback.format_exc()})

        try:
            if ("CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG")) != cublas_contract:
                raise ProtocolError(f"{name}: CUBLAS environment drift before panel")
            model = evaluator.instantiate_ppo_checkpoint(candidate_checkpoint, config, device)
            live_candidate_sha256 = helper.model_state_sha256(model.state_dict())
            if live_candidate_sha256 != expected_candidate_sha256:
                raise ProtocolError(f"{name}: live specialist candidate drift")
            dataset = evaluator.OrderedZipDecisionDataset(archive_path=Path(held["proc_path"]),split="valid",split_mode="archive",split_seed=20260723,hash_size=int(config["hash_size"]),max_state_entities=int(config["max_state_entities"]),deck_hashes=(),team_names=teams)
            loader = torch.utils.data.DataLoader(dataset,batch_size=256,num_workers=8,collate_fn=partial(evaluator.collate_ordered,max_state_entities=int(config["max_state_entities"]),entity_fields=int(config["entity_fields"]),option_fields=int(config["option_fields"])),pin_memory=True,persistent_workers=False,prefetch_factor=2,multiprocessing_context="fork")
            tracked = TrackedLoader(loader, 8)
            try:
                summary, seconds = evaluator.evaluate(model,tracked,device,canonicalize_order=False,max_rows=None,progress_interval=0)
            finally:
                shutdown = tracked.shutdown()
            if not shutdown["pass"]:
                raise ProtocolError(f"{name}: DataLoader workers survived evaluation")
            archive_post = verify_held_archive(held, f"{name} archive")
            metrics = legacy.observed_metrics({"metrics":summary})
            panel_result = {"metrics":metrics,"official_metrics":summary,"seconds":seconds,"live_candidate_model_state_sha256_before":live_candidate_sha256,"worker_shutdown":shutdown,"archive_post_use":archive_post}
        except BaseException as error:
            record_error("evaluation", error)
        finally:
            if tracked is not None and shutdown is None:
                try:
                    shutdown = tracked.shutdown()
                except BaseException as error:
                    record_error("worker_shutdown", error)
            try:
                archive_post = verify_held_archive(held, f"{name} archive final")
            except BaseException as error:
                archive_post = {"post_identity_exact":False}
                record_error("archive_posthash", error)
            restore = {"ordinal":ordinal,"panel":name,"attempted":model is not None,"candidate_unchanged_before_restore":False,"raw_restore_pass":False,"pass":False}
            if model is not None:
                try:
                    candidate_after = helper.model_state_sha256(model.state_dict())
                    restore["candidate_model_state_sha256_before_restore"] = candidate_after
                    restore["model_eval_mode_before_restore"] = model.training is False
                    restore["candidate_unchanged_before_restore"] = candidate_after == expected_candidate_sha256 and model.training is False
                    if candidate_after != expected_candidate_sha256 or model.training is not False:
                        raise ProtocolError(f"{name}: live candidate changed during evaluation")
                except BaseException as error:
                    record_error("candidate_posthash", error)
                try:
                    model.load_state_dict(states["raw"], strict=True)
                    model.eval()
                    torch.cuda.synchronize(device)
                    observed = helper.model_state_sha256(model.state_dict())
                    restore["raw_model_state_sha256"] = observed
                    restore["raw_restore_pass"] = observed == expected_raw_sha256
                    if observed != expected_raw_sha256:
                        raise ProtocolError(f"{name}: raw restoration hash drift")
                except BaseException as error:
                    record_error("raw_restore", error)
            restore["cublas_environment_after_panel_exact"] = ("CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG")) == cublas_contract
            if not restore["cublas_environment_after_panel_exact"]:
                try:
                    raise ProtocolError(f"{name}: CUBLAS environment drift after panel")
                except BaseException as error:
                    record_error("cublas_postcheck", error)
            restore["pass"] = restore["candidate_unchanged_before_restore"] is True and restore["raw_restore_pass"] is True
            restores.append(restore)
            tracked = None
            model = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except BaseException as error:
                record_error("cuda_cleanup", error)
        if panel_errors:
            errors[name] = panel_errors
        elif panel_result is not None and restore["pass"] and shutdown is not None and shutdown.get("pass") is True and archive_post.get("post_identity_exact") is True:
            panel_result["live_candidate_model_state_sha256_after"] = restore["candidate_model_state_sha256_before_restore"]
            results[name] = panel_result
        else:
            errors[name] = [{"stage":"completion","error":"ProtocolError: panel did not produce a complete auditable result","traceback":""}]
        ledger.append({"ordinal":ordinal,"panel":name,"attempt_count":1,"metrics_completed":panel_result is not None,"evaluation_pass":name in results,"rows":None if panel_result is None else panel_result["metrics"]["rows"],"workers_spawned":0 if shutdown is None else shutdown["spawned_workers"],"workers_alive_after_shutdown":[] if shutdown is None else shutdown["alive_after_shutdown"],"error_count":len(errors.get(name,[]))})
    return results, ledger, {"per_panel":restores,"all_six_attempted":len(restores)==6,"pass":len(restores)==6 and all(item["pass"] for item in restores)}, errors


def decide(results: Mapping[str, Any], legacy: ModuleType, panel_errors: Mapping[str, Any] | None = None) -> dict[str, Any]:
    absolute: list[dict[str, Any]] = []
    noharm: list[dict[str, Any]] = []
    panels: dict[str, Any] = {}
    effective_thresholds: dict[str, Any] = {}
    for name in PANEL_ORDER:
        frozen_thresholds = legacy.expected_thresholds(name)
        effective_thresholds[name] = {
            metric: {
                "operator": "==" if "exact" in frozen_thresholds[metric] else ">=",
                "threshold": int(frozen_thresholds[metric].get("exact", frozen_thresholds[metric].get("minimum"))) if metric in EXACT_RELATIVE_METRICS else max(int(frozen_thresholds[metric]["minimum"]), int(CW11_METRICS[name][metric])),
            }
            for metric in METRIC_ORDER
        }
        if name not in results:
            for metric in METRIC_ORDER:
                comparison = frozen_thresholds[metric]
                exact = "exact" in comparison
                absolute.append({"gate":f"{name}.{metric}","observed":None,"operator":"==" if exact else ">=","threshold":int(comparison["exact"] if exact else comparison["minimum"]),"pass":False,"error":"panel evaluation result absent"})
                noharm.append({"gate":f"{name}.{metric}","observed":None,"operator":"==" if metric in EXACT_RELATIVE_METRICS else ">=","threshold":int(CW11_METRICS[name][metric]),"pass":False,"error":"panel evaluation result absent"})
            panels[name] = {"metrics":None,"errors":[] if panel_errors is None else panel_errors.get(name,[]),"legacy_pass":False,"CW11_noharm_pass":False}
            continue
        metrics = results[name]["metrics"]
        panel_absolute = legacy.evaluate_gates(name, metrics)
        panel_noharm = []
        for metric in METRIC_ORDER:
            observed = int(metrics[metric]); threshold = int(CW11_METRICS[name][metric]); exact = metric in EXACT_RELATIVE_METRICS
            panel_noharm.append({"gate":f"{name}.{metric}","observed":observed,"operator":"==" if exact else ">=","threshold":threshold,"pass":observed==threshold if exact else observed>=threshold})
        absolute.extend(panel_absolute); noharm.extend(panel_noharm)
        panels[name] = {"metrics":metrics,"delta_from_CW11":{key:int(metrics[key])-int(CW11_METRICS[name][key]) for key in METRIC_ORDER},"legacy_pass":all(item["pass"] for item in panel_absolute),"CW11_noharm_pass":all(item["pass"] for item in panel_noharm)}
    passed = len(absolute)==len(noharm)==60 and all(item["pass"] for item in absolute) and all(item["pass"] for item in noharm)
    return {"status":"GO_CW23_SPECIALIST60_GATE" if passed else "NO_GO_CW23_SPECIALIST60_GATE","pass":passed,"legacy_absolute":{"gate_count":len(absolute),"passed":sum(bool(item["pass"]) for item in absolute),"gates":absolute,"failed":[item for item in absolute if not item["pass"]]},"CW11_noharm":{"gate_count":len(noharm),"passed":sum(bool(item["pass"]) for item in noharm),"checks":noharm,"failed":[item for item in noharm if not item["pass"]]},"effective_thresholds":effective_thresholds,"panels":panels,"broad_authorized_next":passed,"gold_authorized":False,"submission_authorized":False}


def production(expected_self_sha: str, launch: Mapping[str, Any]) -> dict[str, Any]:
    validate_runtime(False)
    cublas_present_before = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    cublas_value_before = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        source, self_record = read_regular(SCRIPT, expected_self_sha, FROZEN_MODE, "self")
        audit = source_audit(source, True)
        base, base_record = import_exact(BASE, BASE_SHA256, "cw23_fulltrain_adapter")
        legacy, legacy_record = import_exact(LEGACY, LEGACY_SHA256, "specialist_legacy")
        fulltrain_bytes, fulltrain_record = read_regular(FULLTRAIN_RESULT,FULLTRAIN_RESULT_SHA256,EVIDENCE_MODE,"fulltrain result")
        fulltrain = strict_json(fulltrain_bytes,"fulltrain result"); validate_fulltrain(fulltrain,base)
        cw11_manifest_bytes, cw11_manifest_record = read_regular(CW11_MANIFEST,CW11_MANIFEST_SHA256,EVIDENCE_MODE,"CW11 manifest")
        cw11_decision_bytes, cw11_decision_record = read_regular(CW11_DECISION,CW11_DECISION_SHA256,EVIDENCE_MODE,"CW11 decision")
        protocol_checks = validate_protocol(legacy,strict_json(cw11_manifest_bytes,"CW11 manifest"),strict_json(cw11_decision_bytes,"CW11 decision"))
        cw23_bytes, cw23_record = read_regular(base.CW23,base.CW23_SHA256,EVIDENCE_MODE,"CW23")
        raw_bytes, raw_record = read_regular(base.RAW,base.RAW_FILE_SHA256,0o664,"raw U468")
        formal, formal_record = base.import_frozen(base.FORMAL,base.FORMAL_SHA256,"formal_specialist")
        cw20, cw20_record = base.import_frozen(base.CW20,base.CW20_SHA256,"cw20_specialist")
        exact_bc, bc_record = import_standard_exact(BC,BC_SHA256,"train_bc_orbit",0o664)
        exact_ppo, ppo_record = import_standard_exact(PPO,PPO_SHA256,"train_ppo",0o664)
        exact_evaluator, evaluator_record = import_standard_exact(EVALUATOR,EVALUATOR_SHA256,"evaluate_policy_bc",0o664)
        dependency_records = {"evaluator":evaluator_record,"bc":bc_record,"ppo":ppo_record}
        formal.load_module = base.exact_nested_module_loader
        cw23_document = base.strict_json(cw23_bytes,"CW23")
        if cw23_document.get("cumulative_official_unique_changed_candidate_count") != 1 or cw23_document.get("official_unique_changed_candidate_count_consumed") != 0:
            raise ProtocolError("CW23 official candidate budget predecessor drift")
        candidate_payload = base.validate_cw23(cw23_document)
        states, checkpoint, reconstruction = base.reconstruct_states(candidate_payload,raw_bytes,formal,cw20)
        helper = formal.load_helper(); evaluator = helper.evaluator
        cublas_value_during_imports = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    finally:
        if cublas_present_before:
            if cublas_value_before is None:
                raise ProtocolError("CUBLAS environment presence/value inconsistency")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_value_before
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    cublas_environment = {"present_before":cublas_present_before,"value_before":cublas_value_before,"value_during_bound_imports":cublas_value_during_imports,"present_restored":"CUBLAS_WORKSPACE_CONFIG" in os.environ,"value_restored":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"restored_exact":(("CUBLAS_WORKSPACE_CONFIG" in os.environ)==cublas_present_before and os.environ.get("CUBLAS_WORKSPACE_CONFIG")==cublas_value_before)}
    if not cublas_environment["restored_exact"]:
        raise ProtocolError(f"CUBLAS environment restoration failed: {cublas_environment}")
    runtime = validate_runtime(True)
    post_dependency_records = {"evaluator":read_regular(EVALUATOR,EVALUATOR_SHA256,0o664,"evaluator after import")[1],"bc":read_regular(BC,BC_SHA256,0o664,"BC after import")[1],"ppo":read_regular(PPO,PPO_SHA256,0o664,"PPO after import")[1]}
    module_checks = {"evaluator_path":Path(evaluator.__file__).resolve()==EVALUATOR.resolve(),"evaluator_exact_module":evaluator is exact_evaluator,"evaluator_bc_identity":evaluator.bc is exact_bc,"evaluator_ppo_identity":evaluator.ppo is exact_ppo,"helper_bc_identity":helper.bc is exact_bc,"helper_ppo_identity":helper.ppo is exact_ppo,"evaluator_stable":post_dependency_records["evaluator"]==dependency_records["evaluator"],"bc_path":Path(helper.bc.__file__).resolve()==BC.resolve(),"bc_stable":post_dependency_records["bc"]==dependency_records["bc"],"ppo_path":Path(helper.ppo.__file__).resolve()==PPO.resolve(),"ppo_stable":post_dependency_records["ppo"]==dependency_records["ppo"]}
    if not all(module_checks.values()):
        raise ProtocolError(f"evaluator module drift: {module_checks}")
    start_method_before = multiprocessing.get_start_method(allow_none=True)
    default_context_method = multiprocessing.get_context().get_start_method()
    numerical_runtime_before = {"deterministic_algorithms_disabled":helper.torch.are_deterministic_algorithms_enabled() is False,"cudnn_benchmark_false":helper.torch.backends.cudnn.benchmark is False,"cudnn_deterministic_false":helper.torch.backends.cudnn.deterministic is False,"float32_matmul_precision_before":helper.torch.get_float32_matmul_precision(),"float32_matmul_precision_initial_highest":helper.torch.get_float32_matmul_precision()=="highest","cuda_current_device_zero":helper.torch.cuda.current_device()==0,"cublas_environment_restored_exact":cublas_environment["restored_exact"],"multiprocessing_start_before":start_method_before,"multiprocessing_default_context":default_context_method,"multiprocessing_fork_exact":default_context_method=="fork"}
    if not all(numerical_runtime_before[key] for key in ("deterministic_algorithms_disabled","cudnn_benchmark_false","cudnn_deterministic_false","float32_matmul_precision_initial_highest","cuda_current_device_zero","cublas_environment_restored_exact","multiprocessing_fork_exact")):
        raise ProtocolError(f"legacy specialist runtime drift: {numerical_runtime_before}")
    before = {name:helper.model_state_sha256(state) for name,state in states.items()}
    held_archives: dict[Path, Mapping[str, Any]] = {}
    archive_records: dict[str, Any] = {}
    archive_post_records: dict[str, Any] = {}
    archive_close_errors: list[str] = []
    try:
        for path,(digest,mode) in ARCHIVES.items():
            held = open_held_archive(path,digest,mode,"specialist archive")
            held_archives[path] = held
            archive_records[str(path.relative_to(ROOT))] = held["record"]
        evaluations, ledger, restore, panel_errors = evaluate_candidate(evaluator,legacy,helper,states,checkpoint,held_archives,base.CW23_MODEL_SHA256,base.RAW_MODEL_SHA256)
        for path,held in held_archives.items():
            archive_post_records[str(path.relative_to(ROOT))] = verify_held_archive(held,"specialist archive final")
    finally:
        for held in reversed(list(held_archives.values())):
            try:
                close_held_archive(held)
            except BaseException as error:
                archive_close_errors.append(f"{type(error).__name__}: {error}")
    if archive_close_errors:
        raise ProtocolError(f"specialist archive close failed: {archive_close_errors}")
    after = {name:helper.model_state_sha256(state) for name,state in states.items()}
    final_input_records = {
        "base_adapter":read_regular(BASE,BASE_SHA256,FROZEN_MODE,"base adapter after evaluation")[1],
        "fulltrain_result":read_regular(FULLTRAIN_RESULT,FULLTRAIN_RESULT_SHA256,EVIDENCE_MODE,"fulltrain result after evaluation")[1],
        "legacy_specialist":read_regular(LEGACY,LEGACY_SHA256,FROZEN_MODE,"legacy specialist after evaluation")[1],
        "CW11_manifest":read_regular(CW11_MANIFEST,CW11_MANIFEST_SHA256,EVIDENCE_MODE,"CW11 manifest after evaluation")[1],
        "CW11_decision":read_regular(CW11_DECISION,CW11_DECISION_SHA256,EVIDENCE_MODE,"CW11 decision after evaluation")[1],
        "CW23":read_regular(base.CW23,base.CW23_SHA256,EVIDENCE_MODE,"CW23 after evaluation")[1],
        "raw_U468":read_regular(base.RAW,base.RAW_FILE_SHA256,0o664,"raw U468 after evaluation")[1],
        "formal":read_regular(base.FORMAL,base.FORMAL_SHA256,FROZEN_MODE,"formal after evaluation")[1],
        "CW20":read_regular(base.CW20,base.CW20_SHA256,FROZEN_MODE,"CW20 after evaluation")[1],
        "evaluator":read_regular(EVALUATOR,EVALUATOR_SHA256,0o664,"evaluator after evaluation")[1],
        "bc":read_regular(BC,BC_SHA256,0o664,"BC after evaluation")[1],
        "ppo":read_regular(PPO,PPO_SHA256,0o664,"PPO after evaluation")[1],
    }
    final_input_checks = {"base_adapter":final_input_records["base_adapter"]==base_record,"fulltrain_result":final_input_records["fulltrain_result"]==fulltrain_record,"legacy_specialist":final_input_records["legacy_specialist"]==legacy_record,"CW11_manifest":final_input_records["CW11_manifest"]==cw11_manifest_record,"CW11_decision":final_input_records["CW11_decision"]==cw11_decision_record,"CW23":final_input_records["CW23"]==cw23_record,"raw_U468":final_input_records["raw_U468"]==raw_record,"formal":final_input_records["formal"]==formal_record,"CW20":final_input_records["CW20"]==cw20_record,"evaluator":final_input_records["evaluator"]==dependency_records["evaluator"],"bc":final_input_records["bc"]==dependency_records["bc"],"ppo":final_input_records["ppo"]==dependency_records["ppo"]}
    if not all(final_input_checks.values()):
        raise ProtocolError(f"specialist immutable input drift: {final_input_checks}")
    all_results_present = set(evaluations)==set(PANEL_ORDER)
    cublas_after_evaluation_exact = (("CUBLAS_WORKSPACE_CONFIG" in os.environ)==cublas_present_before and os.environ.get("CUBLAS_WORKSPACE_CONFIG")==cublas_value_before)
    execution_checks = {"all_six_attempted_once":len(ledger)==6 and [item["panel"] for item in ledger]==list(PANEL_ORDER) and all(item["attempt_count"]==1 for item in ledger),"six_successful_fresh_model_instances":all_results_present and len(restore.get("per_panel",[]))==6 and all(item["attempted"] for item in restore["per_panel"]),"all_successful_workers_shutdown":all_results_present and all(evaluations[name]["worker_shutdown"]["pass"] is True for name in PANEL_ORDER),"worker_count_exact8_each_successful_panel":all_results_present and all(evaluations[name]["worker_shutdown"]["spawned_workers"]==8 for name in PANEL_ORDER),"no_worker_alive_after_any_attempt":all(not item["workers_alive_after_shutdown"] for item in ledger),"panel_rows_exact":all_results_present and all(int(evaluations[name]["metrics"]["rows"])==int(CW11_METRICS[name]["rows"]) for name in PANEL_ORDER),"panel_context34_rows_exact":all_results_present and all(int(evaluations[name]["metrics"]["context34_rows"])==int(CW11_METRICS[name]["context34_rows"]) for name in PANEL_ORDER),"state_hashes_unchanged":before==after,"raw_exact":after.get("raw")==base.RAW_MODEL_SHA256,"candidate_exact":after.get("candidate")==base.CW23_MODEL_SHA256,"all_instantiated_models_restored":all(item["pass"] for item in restore.get("per_panel",[]) if item["attempted"]),"all_six_evaluator_restores_pass":restore.get("pass") is True,"float32_matmul_precision_high":helper.torch.get_float32_matmul_precision()=="high","legacy_deterministic_flag_still_disabled":helper.torch.are_deterministic_algorithms_enabled() is False,"legacy_cudnn_deterministic_still_false":helper.torch.backends.cudnn.deterministic is False,"cublas_environment_after_evaluation_exact":cublas_after_evaluation_exact,"immutable_inputs_posthash_exact":all(final_input_checks.values()),"archive_posthash_complete":set(archive_post_records)=={str(path.relative_to(ROOT)) for path in ARCHIVES} and all(item["post_identity_exact"] and item["visible_path_identity_exact"] for item in archive_post_records.values())}
    integrity_keys = ("all_six_attempted_once","no_worker_alive_after_any_attempt","state_hashes_unchanged","raw_exact","candidate_exact","all_instantiated_models_restored","float32_matmul_precision_high","legacy_deterministic_flag_still_disabled","legacy_cudnn_deterministic_still_false","cublas_environment_after_evaluation_exact","immutable_inputs_posthash_exact","archive_posthash_complete")
    if not all(execution_checks[key] for key in integrity_keys):
        raise ProtocolError(f"specialist execution integrity drift: {execution_checks}")
    decision = decide(evaluations,legacy,panel_errors)
    _, self_after = read_regular(SCRIPT,expected_self_sha,FROZEN_MODE,"self after specialist")
    if self_after != self_record:
        raise ProtocolError("self changed during specialist")
    return {"schema_version":SCHEMA,"status":decision["status"],"candidate_model_state_sha256":base.CW23_MODEL_SHA256,"inputs":{"self":{**self_record,"source_audit":audit,"post_evaluation":self_after},"hash_bound_execution":dict(launch),"base_adapter":base_record,"fulltrain_result":fulltrain_record,"legacy_specialist":legacy_record,"CW11_manifest":cw11_manifest_record,"CW11_decision":cw11_decision_record,"CW23":cw23_record,"raw_U468":raw_record,"formal":formal_record,"CW20":cw20_record,"evaluator_dependencies":dependency_records,"evaluator_dependencies_post_import":post_dependency_records,"archives":archive_records,"archives_post_evaluation":archive_post_records,"all_nonarchive_inputs_post_evaluation":final_input_records,"all_nonarchive_input_posthash_checks":final_input_checks},"protocol_checks":protocol_checks,"module_checks":module_checks,"reconstruction":reconstruction,"execution":{"batch_size":256,"workers":8,"split":"valid","split_mode":"archive","split_seed":20260723,"prediction_order":"policy","multiprocessing_context":"fork","model_instance_count":sum(bool(item["attempted"]) for item in restore["per_panel"]),"panel_row_instances":sum(CW11_METRICS[name]["rows"] for name in PANEL_ORDER),"unique_archive_valid_rows":CW11_METRICS["pokemonfan"]["rows"]+CW11_METRICS["flg"]["rows"]+CW11_METRICS["core5"]["rows"],"completion_ledger":ledger,"panel_errors":panel_errors,"checks":execution_checks,"finally_raw_restore":restore,"legacy_numerical_runtime_before":numerical_runtime_before,"CUBLAS_environment_binding":cublas_environment},"evaluations":evaluations,"decision":decision,"runtime":runtime,"scope":{"specialist_valid_consumed":True,"all_six_panels_attempted_once_before_decision":True,"CW11_materialized_eval_only_checkpoint_opened":False,"candidate_RAM_only":True,"checkpoint_writes":0,"model_artifact_writes":0,"train_rows_opened":False,"test_rows_opened":False,"submission_performed":False,"package_upload_performed":False},"official_candidate_budget_ledger":{"before_gate":1,"consumed_this_gate":1,"after_gate":2,"budget":12,"remaining":10},"official_unique_changed_candidate_count_before_gate":1,"official_unique_changed_candidate_count_consumed_this_gate":1,"cumulative_official_unique_changed_candidate_count":2,"official_unique_changed_candidate_budget":12,"submission_performed":False,"package_upload_performed":False}


def audit_only() -> dict[str, Any]:
    runtime = validate_runtime(False)
    source, source_record = read_regular(SCRIPT,None,FROZEN_MODE,"self")
    launch = validate_binding(source_record,True)
    base, base_record = import_exact(BASE,BASE_SHA256,"cw23_base_static")
    legacy, legacy_record = import_exact(LEGACY,LEGACY_SHA256,"specialist_legacy_static")
    fulltrain_bytes, fulltrain_record = read_regular(FULLTRAIN_RESULT,FULLTRAIN_RESULT_SHA256,EVIDENCE_MODE,"fulltrain result")
    validate_fulltrain(strict_json(fulltrain_bytes,"fulltrain result"),base)
    cw23_bytes, cw23_record = read_regular(base.CW23,base.CW23_SHA256,EVIDENCE_MODE,"CW23 static")
    cw23_document = base.strict_json(cw23_bytes,"CW23 static")
    base.validate_cw23(cw23_document)
    budget_checks = {"before_gate_exact1":cw23_document.get("cumulative_official_unique_changed_candidate_count")==1,"trainonly_consumed_zero":cw23_document.get("official_unique_changed_candidate_count_consumed")==0}
    if not all(budget_checks.values()):
        raise ProtocolError(f"CW23 static budget drift: {budget_checks}")
    manifest_bytes, manifest_record = read_regular(CW11_MANIFEST,CW11_MANIFEST_SHA256,EVIDENCE_MODE,"CW11 manifest")
    decision_bytes, decision_record = read_regular(CW11_DECISION,CW11_DECISION_SHA256,EVIDENCE_MODE,"CW11 decision")
    protocol = validate_protocol(legacy,strict_json(manifest_bytes,"CW11 manifest"),strict_json(decision_bytes,"CW11 decision"))
    archives = {str(path.relative_to(ROOT)):read_regular(path,digest,mode,"archive static")[1] for path,(digest,mode) in ARCHIVES.items()}
    dependencies = {"evaluator":read_regular(EVALUATOR,EVALUATOR_SHA256,0o664,"evaluator")[1],"bc":read_regular(BC,BC_SHA256,0o664,"BC")[1],"ppo":read_regular(PPO,PPO_SHA256,0o664,"PPO")[1]}
    interfaces = {"base_reconstruct":callable(base.reconstruct_states),"base_validate_CW23":callable(base.validate_cw23),"legacy_expected_thresholds":callable(legacy.expected_thresholds),"legacy_evaluate_gates":callable(legacy.evaluate_gates),"legacy_observed_metrics":callable(legacy.observed_metrics)}
    if not all(interfaces.values()) or lexists(ATTEMPT) or lexists(OUTPUT):
        raise ProtocolError(f"static interface/target drift: {interfaces}")
    return {"schema_version":SCHEMA,"status":"static_audit_pass","runtime":runtime,"source":{**source_record,"audit":source_audit(source,True)},"hash_bound_execution":launch,"inputs":{"base":base_record,"fulltrain":fulltrain_record,"CW23":cw23_record,"legacy":legacy_record,"CW11_manifest":manifest_record,"CW11_decision":decision_record,"archives":archives,"dependencies":dependencies},"protocol_checks":protocol,"official_budget_checks":budget_checks,"interfaces":interfaces,"attempt_absent":True,"output_absent":True,"cuda_accessed":False,"valid_members_opened":False,"writes_performed":False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--audit-only",action="store_true"); args=parser.parse_args()
    if args.audit_only:
        print(canonical_json(audit_only()).decode(),end=""); return
    validate_runtime(False)
    if lexists(ATTEMPT) or lexists(OUTPUT):
        raise ProtocolError("specialist one-shot target exists")
    source, source_record = read_regular(SCRIPT,None,FROZEN_MODE,"self")
    source_audit(source,True); launch=validate_binding(source_record,False)
    attempt = publish(ATTEMPT,{"schema_version":SCHEMA,"status":"specialist_attempt_committed_before_CUDA_or_valid_open","source":source_record,"hash_bound_execution":launch,"candidate_model_state_sha256":"bf551805d807cb6a77f500a51a12204727ab5cbd371fb1d36a947c63299228cb","fulltrain_result_sha256":FULLTRAIN_RESULT_SHA256,"official_unique_changed_candidate_count_before_gate":1,"official_unique_changed_candidate_count_consumed_this_gate":1,"cumulative_official_unique_changed_candidate_count":2,"official_unique_changed_candidate_budget":12,"remaining_official_unique_changed_candidate_budget":10,"submission_authorized":False})
    try:
        result=production(sha256_bytes(source),launch)
    except BaseException as error:
        failure={"schema_version":SCHEMA,"status":"ERROR_CW23_SPECIALIST60_GATE","error":f"{type(error).__name__}: {error}","traceback":traceback.format_exc(),"attempt":attempt,"official_unique_changed_candidate_count_consumed_this_gate":1,"cumulative_official_unique_changed_candidate_count":2,"submission_performed":False,"package_upload_performed":False}
        publication=publish(OUTPUT,failure); print(canonical_json({"schema_version":SCHEMA,"status":failure["status"],"output":publication}).decode(),end=""); raise
    result["attempt"]=attempt; publication=publish(OUTPUT,result)
    print(canonical_json({"schema_version":SCHEMA,"status":result["status"],"pass":result["decision"]["pass"],"candidate_model_state_sha256":result["candidate_model_state_sha256"],"legacy_gates_passed":result["decision"]["legacy_absolute"]["passed"],"CW11_noharm_gates_passed":result["decision"]["CW11_noharm"]["passed"],"output":publication,"cumulative_official_unique_changed_candidate_count":2,"submission_performed":False}).decode(),end="")


if __name__ == "__main__":
    main()

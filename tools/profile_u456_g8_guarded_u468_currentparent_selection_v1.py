#!/usr/bin/env python3
"""One-shot train-only margin profile and fixed B256 selection for guarded U468.

Audit mode opens no archive member and performs no CUDA work. Formal mode
profiles only the frozen ``train/`` members at the unchanged guarded U468
parent, selects the preregistered five targets/fourteen guards and current-
parent B256 cache rows, and publishes one immutable JSON result. It never
constructs or evaluates a changed model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "profile_u456_g8_guarded_u468_currentparent_selection_v1.py"
SCHEMA = "ptcg-u456-g8-guarded-u468-currentparent-train-profile-selection-v1"
PREREG_SCHEMA = f"{SCHEMA}-execution-preregistration-v1"

MASTER = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_design202608143.master_preregistration.json"
MASTER_SHA256 = "0efc5a08ba1a83ed26b0b518c826c5744d67e98895d2da21449e56291116bfa8"
CORRECTION = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_design202608143.master_correction_v2.json"
CORRECTION_SHA256 = "6accc12201900aba1a50d64d4ffd21796d36180cdb44990694b768a5a351263a"
CORRECTION_V3 = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_design202608143.master_correction_v3.json"
CORRECTION_V3_SHA256 = "709972d0774536fd79ab5627d7b970e0b6ab4b5f2987d3e0e1f5f762b200461d"
AUTHORIZATION = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141.block3_training_integrity_decision.json"
AUTHORIZATION_SHA256 = "d57954cdaeed5d0a2ff5c2a8d459253ba03f1b9a0f8dd1b5c760b23fc6f026bf"
PARENT = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_p12_design202608141/ppo_stage/block3/B_gold_league/seed-202608141/checkpoints/update-0468.pt"
PARENT_FILE_SHA256 = "a9290745b8eb58704c4617594ac5ab49500c6e65ecd9f273a8285c12547fbb83"
PARENT_MODEL_SHA256 = "832c724277d6c2263d76ddbf41622f4ba06621ec3155fa8b7ae2591203db1c5e"

FRAMEWORK = TOOLS / "profile_u468_beta1157_train_margins.py"
FRAMEWORK_SHA256 = "f1018994c48ffadcbc5277f0dfd9233086693f90bf30f5bf67462965b4e18142"
FORMAL = TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal_v3.py"
FORMAL_SHA256 = "2d12ce9e76f6672911948dcc6458573899ec45fc39e748510544001025ede939"
BC_SHA256 = "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
PPO_SHA256 = "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"

PREREGISTRATION = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_design202608143.execution_preregistration.json"
ATTEMPT_MARKER = ROOT / ".ptcg-u456-g8-guarded-u468-currentparent-profile-selection-attempt-v1.json"
OUTPUT = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_design202608143.json"

PANEL_ORDER = ("flg", "pokemonfan", "core5")
DATASETS = {
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
    "pokemonfan": ROOT / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "core5": ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
}
DATA_SHA256 = {
    "flg": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    "pokemonfan": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}
EXPECTED_ROWS = {"flg": 9443, "pokemonfan": 9487, "core5": 5120}
EXPECTED_CONTEXT34_ROWS = {"flg": 42, "pokemonfan": 38, "core5": 20}
KEEP = 10000
BATCH_SIZE = 256
SEED = 202608143
TARGET_MARGIN = 1.0 / 512.0
FORCED_FLG_GUARDS = (
    "f6d1d86202a3010abb8736c70396029c35478a9f79ed9d8e53704c51bfaacd64",
    "20ba6315e9860d54183bab71e4e818bd9847c299c52120e3fbcbc8481972d73f",
)
STRATUM_ORDER = (
    "pf_ctx0_hard",
    "pf_ctx7_hard",
    "dominic_ctx0_hard",
    "pf_ctx0_retention",
    "pf_ctx7_retention",
    "dominic_ctx0_retention",
    "broad_flg_retention",
    "broad_pf_other_retention",
    "broad_core_non_dominic_retention",
    "broad_context34_retention",
)
STRATUM_QUOTAS = {
    "pf_ctx0_hard": 32,
    "pf_ctx7_hard": 32,
    "dominic_ctx0_hard": 32,
    "pf_ctx0_retention": 32,
    "pf_ctx7_retention": 32,
    "dominic_ctx0_retention": 32,
    "broad_flg_retention": 16,
    "broad_pf_other_retention": 16,
    "broad_core_non_dominic_retention": 16,
    "broad_context34_retention": 16,
}


class ProtocolError(RuntimeError):
    """Fail-closed profile/selection error."""


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


def lexists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def read_regular(
    path: Path,
    expected_sha256: str | None,
    label: str,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ProtocolError(f"{label}: unsafe file type or link count")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    visible = os.lstat(path)
    identity = lambda item: (
        int(item.st_dev), int(item.st_ino), int(item.st_size), int(item.st_mtime_ns)
    )
    if (
        identity(before) != identity(after)
        or identity(after) != identity(visible)
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or int(visible.st_nlink) != 1
    ):
        raise ProtocolError(f"{label}: identity changed during held-fd read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    mode = stat.S_IMODE(after.st_mode)
    if len(payload) != int(after.st_size) or (
        expected_sha256 is not None and digest != expected_sha256
    ):
        raise ProtocolError(f"{label}: SHA or size drift")
    if expected_mode is not None and mode != expected_mode:
        raise ProtocolError(f"{label}: mode drift {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "held_fd_identity_exact": True,
    }


def publish(path: Path, payload: bytes) -> dict[str, Any]:
    if path not in {ATTEMPT_MARKER, OUTPUT}:
        raise ProtocolError(f"publication outside exact allowlist: {path}")
    if path.parent.resolve() not in {ROOT.resolve(), (ROOT / "artifacts").resolve()}:
        raise ProtocolError("publication parent drift")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ProtocolError("short O_EXCL publication")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    reloaded, evidence = read_regular(
        path, sha256_bytes(payload), f"published {path.name}", 0o444
    )
    if reloaded != payload:
        raise ProtocolError("published payload mismatch")
    return evidence


def load_module(
    path: Path, digest: str, mode: int, name: str
) -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = read_regular(path, digest, name, mode)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise ProtocolError("run from frozen repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise ProtocolError("requires exact Python -I -B")


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_imports = {"requests", "urllib", "http", "socket", "subprocess", "kaggle"}
    forbidden_calls = {"save", "backward", "step", "unlink", "rename", "replace", "rmtree"}
    import_hits: list[dict[str, Any]] = []
    call_hits: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".", 1)[0] in forbidden_imports:
                    import_hits.append({"line": node.lineno, "name": name})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden_calls:
                call_hits.append({"line": node.lineno, "name": node.func.attr})
    checks = {
        "ast_parse": True,
        "no_network_or_submission_import": not import_hits,
        "no_optimizer_checkpoint_or_destructive_call": not call_hits,
        "profile_keep_covers_each_panel": KEEP >= max(EXPECTED_ROWS.values()),
        "b256_quota_exact256": sum(STRATUM_QUOTAS.values()) == 256,
        "targets_exact5": 3 + 2 == 5,
        "guards_exact14": 6 + 4 + 4 == 14,
    }
    if not all(checks.values()):
        raise ProtocolError(f"static source audit failed: {checks}")
    return {"checks": checks, "import_hits": import_hits, "call_hits": call_hits}


def expected_contract() -> dict[str, Any]:
    return {
        "parent_only": True,
        "changed_candidate_construction_or_evaluation": False,
        "split": "train_only",
        "panel_order": list(PANEL_ORDER),
        "rows": dict(EXPECTED_ROWS),
        "context34_rows": dict(EXPECTED_CONTEXT34_ROWS),
        "total_rows": sum(EXPECTED_ROWS.values()),
        "batch_size": BATCH_SIZE,
        "workers": 0,
        "device": "cuda:0",
        "native_output_dtype": "torch.bfloat16",
        "keep_per_correct_and_wrong_list": KEEP,
        "seed": SEED,
        "b256_stratum_order": list(STRATUM_ORDER),
        "b256_quotas": dict(STRATUM_QUOTAS),
        "target_margin": TARGET_MARGIN,
        "target_quotas": {"context0": 3, "context7": 2},
        "guard_quotas": {"flg": 6, "pokemonfan": 4, "core5": 4},
        "forced_flg_guards": list(FORCED_FLG_GUARDS),
        "validation_test_broad_gold_opened": False,
    }


def expected_bindings(self_sha256: str) -> dict[str, Any]:
    return {
        "runner": {"path": str(SCRIPT.relative_to(ROOT)), "sha256": self_sha256, "mode_octal": "0555"},
        "master": {"path": str(MASTER.relative_to(ROOT)), "sha256": MASTER_SHA256, "mode_octal": "0444"},
        "correction_v2": {"path": str(CORRECTION.relative_to(ROOT)), "sha256": CORRECTION_SHA256, "mode_octal": "0444"},
        "correction_v3": {"path": str(CORRECTION_V3.relative_to(ROOT)), "sha256": CORRECTION_V3_SHA256, "mode_octal": "0444"},
        "authorization": {"path": str(AUTHORIZATION.relative_to(ROOT)), "sha256": AUTHORIZATION_SHA256, "mode_octal": "0444", "status": "GO_P12"},
        "parent": {"path": str(PARENT.relative_to(ROOT)), "file_sha256": PARENT_FILE_SHA256, "runtime_model_state_sha256": PARENT_MODEL_SHA256, "mode_octal": "0444"},
        "framework": {"path": str(FRAMEWORK.relative_to(ROOT)), "sha256": FRAMEWORK_SHA256, "mode_octal": "0664"},
        "formal_v3": {"path": str(FORMAL.relative_to(ROOT)), "sha256": FORMAL_SHA256, "mode_octal": "0555"},
        "bc_source": {"path": "tools/train_bc_orbit.py", "sha256": BC_SHA256, "mode_octal": "0664"},
        "ppo_source": {"path": "tools/train_ppo.py", "sha256": PPO_SHA256, "mode_octal": "0664"},
        "train_archives": {
            panel: {"path": str(DATASETS[panel].relative_to(ROOT)), "sha256": DATA_SHA256[panel]}
            for panel in PANEL_ORDER
        },
    }


def output_state() -> dict[str, bool]:
    return {
        "preregistration_present": PREREGISTRATION.is_file() and not PREREGISTRATION.is_symlink(),
        "attempt_marker_absent": not lexists(ATTEMPT_MARKER),
        "result_absent": not lexists(OUTPUT),
    }


def verify_preregistration(
    expected_sha256: str, self_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, evidence = read_regular(
        PREREGISTRATION, expected_sha256, "execution preregistration", 0o444
    )
    value = json.loads(payload)
    command = [
        str(EXPECTED_PYTHON), "-I", "-B", str(SCRIPT), "--mode", "formal",
        "--expected-tool-sha256", self_sha256,
        "--expected-preregistration-sha256", "<LOCKED_PREREGISTRATION_SHA256>",
    ]
    checks = {
        "schema": value.get("schema_version") == PREREG_SCHEMA,
        "status": value.get("status") == "locked_before_only_formal_parent_profile",
        "bindings": value.get("bindings") == expected_bindings(self_sha256),
        "contract": value.get("profile_selection_contract") == expected_contract(),
        "command": value.get("formal_command_template") == command,
        "command_sha": value.get("formal_command_template_sha256") == sha256_bytes(canonical_json(command)),
        "result": value.get("outputs", {}).get("result") == str(OUTPUT.relative_to(ROOT)),
        "marker": value.get("outputs", {}).get("attempt_marker") == str(ATTEMPT_MARKER.relative_to(ROOT)),
        "attempts": value.get("outputs", {}).get("formal_attempts_authorized") == 1,
        "retry": value.get("outputs", {}).get("retry_authorized") is False,
        "targets_absent": value.get("outputs", {}).get("targets_absent_at_lock") is True,
    }
    if not all(checks.values()):
        raise ProtocolError(f"execution preregistration drift: {checks}")
    return value, {**evidence, "checks": checks}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def nonempty(record: Mapping[str, Any]) -> bool:
    expert = record.get("expert_order")
    return isinstance(expert, list) and len(expert) > 0


def parent_correct(record: Mapping[str, Any]) -> bool:
    return record.get("ordered_correct") is True and nonempty(record) and finite_number(record.get("selection_margin"))


def row_copy(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    expert = result.get("expert_order")
    predicted = result.get("predicted_order")
    result["expert_count"] = len(expert) if isinstance(expert, list) else -1
    result["predicted_count"] = len(predicted) if isinstance(predicted, list) else -1
    result["count_correct"] = result["expert_count"] == result["predicted_count"]
    result["top1_correct"] = bool(expert and predicted and expert[0] == predicted[0])
    return result


def select_targets(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for context, quota in ((0, 3), (7, 2)):
        pool = [
            row for row in all_rows
            if row.get("source") == "pokemonfan"
            and int(row.get("context", -1)) == context
            and row.get("ordered_correct") is False
            and row.get("set_correct") is False
            and isinstance(row.get("expert_order"), list)
            and isinstance(row.get("predicted_order"), list)
            and len(row["expert_order"]) == len(row["predicted_order"]) == 1
            and int(row.get("min_count", -1)) == int(row.get("max_count", -2)) == 1
            and finite_number(row.get("selection_margin"))
        ]
        ranked = sorted(
            pool,
            key=lambda row: (
                TARGET_MARGIN - float(row["selection_margin"]),
                str(row["line_sha256"]),
            ),
        )
        if len(ranked) < quota:
            raise ProtocolError(f"insufficient singleton PF context{context} targets")
        for rank, row in enumerate(ranked[:quota], start=1):
            item = row_copy(row)
            item.update({
                "role": "target",
                "target_context_quota": context,
                "target_rank_within_context": rank,
                "required_linear_margin_change": TARGET_MARGIN - float(row["selection_margin"]),
                "endpoint_margin_minimum": TARGET_MARGIN,
            })
            selected.append(item)
    if len(selected) != 5 or len({row["line_sha256"] for row in selected}) != 5:
        raise ProtocolError("target selection cardinality drift")
    return selected


def select_guards(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_sha = {str(row["line_sha256"]): row for row in all_rows}
    if len(by_sha) != len(all_rows):
        raise ProtocolError("full profile line SHA collision")
    result: list[dict[str, Any]] = []
    used: set[str] = set()
    for digest in FORCED_FLG_GUARDS:
        row = by_sha.get(digest)
        if row is None or row.get("source") != "flg" or not parent_correct(row):
            raise ProtocolError(f"forced FLG guard is absent or not parent-correct: {digest}")
        item = row_copy(row)
        item.update({"role": "guard", "guard_panel": "flg", "forced_P12_exposed_guard": True})
        result.append(item)
        used.add(digest)
    for panel, quota in (("flg", 6), ("pokemonfan", 4), ("core5", 4)):
        need = quota - sum(row["guard_panel"] == panel for row in result)
        pool = sorted(
            (
                row for row in all_rows
                if row.get("source") == panel
                and str(row["line_sha256"]) not in used
                and parent_correct(row)
            ),
            key=lambda row: (float(row["selection_margin"]), str(row["line_sha256"])),
        )
        if len(pool) < need:
            raise ProtocolError(f"insufficient {panel} guard capacity")
        for row in pool[:need]:
            item = row_copy(row)
            item.update({"role": "guard", "guard_panel": panel, "forced_P12_exposed_guard": False})
            result.append(item)
            used.add(str(row["line_sha256"]))
    counts = Counter(row["guard_panel"] for row in result)
    if counts != Counter({"flg": 6, "pokemonfan": 4, "core5": 4}) or len(used) != 14:
        raise ProtocolError(f"guard selection cardinality drift: {counts}")
    return result


def classify_stratum(record: Mapping[str, Any], *, hard: bool) -> str | None:
    source = str(record.get("source"))
    context = int(record.get("context", -1))
    team = str(record.get("team_name", ""))
    if hard:
        if source == "pokemonfan" and team == "Pokemon Fan" and context == 0:
            return "pf_ctx0_hard"
        if source == "pokemonfan" and team == "Pokemon Fan" and context == 7:
            return "pf_ctx7_hard"
        if source == "core5" and team == "Dominic Peel" and context == 0:
            return "dominic_ctx0_hard"
        return None
    if context == 34:
        return "broad_context34_retention"
    if source == "pokemonfan" and team == "Pokemon Fan" and context == 0:
        return "pf_ctx0_retention"
    if source == "pokemonfan" and team == "Pokemon Fan" and context == 7:
        return "pf_ctx7_retention"
    if source == "core5" and team == "Dominic Peel" and context == 0:
        return "dominic_ctx0_retention"
    if source == "flg":
        return "broad_flg_retention"
    if source == "pokemonfan":
        return "broad_pf_other_retention"
    if source == "core5" and team != "Dominic Peel":
        return "broad_core_non_dominic_retention"
    return None


def select_with_episode_cap(
    pool: Sequence[Mapping[str, Any]],
    quota: int,
    forced: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], Any],
    stratum: str,
) -> list[dict[str, Any]]:
    eligible = {str(row["line_sha256"]): row for row in pool}
    selected: list[dict[str, Any]] = []
    used_sha: set[str] = set()
    used_episode: set[str] = set()
    for row in sorted(forced, key=key):
        digest = str(row["line_sha256"])
        episode = str(row.get("episode_id", ""))
        if digest not in eligible or not episode or episode in used_episode:
            raise ProtocolError(f"{stratum}: forced row ineligible or episode collision")
        selected.append(row_copy(row))
        used_sha.add(digest)
        used_episode.add(episode)
    for row in sorted(pool, key=key):
        if len(selected) == quota:
            break
        digest = str(row["line_sha256"])
        episode = str(row.get("episode_id", ""))
        if digest in used_sha or not episode or episode in used_episode:
            continue
        selected.append(row_copy(row))
        used_sha.add(digest)
        used_episode.add(episode)
    if len(selected) != quota:
        raise ProtocolError(f"{stratum}: only selected {len(selected)} of {quota}")
    return selected


def build_b256(
    all_rows: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    guards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    forced_by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in targets:
        stratum = classify_stratum(row, hard=True)
        if stratum is None:
            raise ProtocolError("target has no hard stratum")
        forced_by_stratum[stratum].append(row)
    for row in guards:
        stratum = classify_stratum(row, hard=False)
        if stratum is None:
            raise ProtocolError("guard has no retention stratum")
        forced_by_stratum[stratum].append(row)

    selected: list[dict[str, Any]] = []
    for stratum in STRATUM_ORDER:
        hard = stratum.endswith("_hard")
        pool = [
            row for row in all_rows
            if classify_stratum(row, hard=hard) == stratum
            and nonempty(row)
            and finite_number(row.get("selection_margin"))
            and ((row.get("ordered_correct") is False) if hard else parent_correct(row))
            and (
                not hard
                or int(row.get("min_count", -1)) == int(row.get("max_count", -2))
            )
        ]
        rank_key: Callable[[Mapping[str, Any]], Any]
        if hard:
            rank_key = lambda row: (-float(row["decision_margin"]), str(row["line_sha256"]))
        else:
            rank_key = lambda row: (float(row["selection_margin"]), str(row["line_sha256"]))
        rows = select_with_episode_cap(
            pool,
            STRATUM_QUOTAS[stratum],
            forced_by_stratum.get(stratum, []),
            rank_key,
            stratum,
        )
        for row in rows:
            row["stratum"] = stratum
            row["category"] = "hard" if hard else ("c34" if stratum == "broad_context34_retention" else "fragile")
            row["profile_selection_margin_rank_basis"] = (
                "descending_decision_margin_then_line_sha" if hard else "ascending_selection_margin_then_line_sha"
            )
        selected.extend(rows)
    if len(selected) != 256 or len({row["line_sha256"] for row in selected}) != 256:
        raise ProtocolError("B256 row count or uniqueness drift")
    forced_sha = {row["line_sha256"] for row in (*targets, *guards)}
    selected_sha = {row["line_sha256"] for row in selected}
    if not forced_sha.issubset(selected_sha):
        raise ProtocolError("B256 does not contain every target and guard")
    for index, row in enumerate(selected):
        row["final_b256_slot_zero_based"] = index
    counts = Counter(row["stratum"] for row in selected)
    if dict(counts) != STRATUM_QUOTAS:
        raise ProtocolError(f"B256 quota drift: {counts}")
    return selected


def sanitize_profiles(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    replacements: Counter[str] = Counter()
    for panel, profile in profiles.items():
        for list_name in ("near_wrong", "fragile_correct"):
            rows = profile.get(list_name)
            if not isinstance(rows, list):
                raise ProtocolError("profile retained list malformed")
            for row in rows:
                for key in ("selection_margin", "count_margin", "decision_margin", "predicted_log_probability"):
                    value = row.get(key)
                    if isinstance(value, float) and not math.isfinite(value):
                        row[key] = None
                        row[f"{key}_nonfinite_reason"] = "no_competing_allowed_choice" if value > 0 else "nonfinite_model_diagnostic"
                        replacements[f"{panel}.{list_name}.{key}"] += 1
    return {
        "encoding": "JSON null plus sibling nonfinite_reason",
        "replacement_counts": dict(sorted(replacements.items())),
        "replacement_total": sum(replacements.values()),
    }


def load_static() -> dict[str, Any]:
    source, self_evidence = read_regular(SCRIPT, None, "profile runner", 0o555)
    master_bytes, master_evidence = read_regular(MASTER, MASTER_SHA256, "master", 0o444)
    correction_bytes, correction_evidence = read_regular(CORRECTION, CORRECTION_SHA256, "correction v2", 0o444)
    correction_v3_bytes, correction_v3_evidence = read_regular(CORRECTION_V3, CORRECTION_V3_SHA256, "correction v3", 0o444)
    auth_bytes, auth_evidence = read_regular(AUTHORIZATION, AUTHORIZATION_SHA256, "guarded block3 authorization", 0o444)
    parent_bytes, parent_evidence = read_regular(PARENT, PARENT_FILE_SHA256, "guarded U468 parent", 0o444)
    framework, framework_evidence = load_module(FRAMEWORK, FRAMEWORK_SHA256, 0o664, "current_parent_profile_framework")
    formal, formal_evidence = load_module(FORMAL, FORMAL_SHA256, 0o555, "current_parent_profile_formal")
    master = json.loads(master_bytes)
    correction = json.loads(correction_bytes)
    correction_v3 = json.loads(correction_v3_bytes)
    authorization = json.loads(auth_bytes)
    semantic = {
        "master_locked": master.get("status") == "locked_before_parent_profile_or_changed_candidate_construction",
        "parent_exact": master.get("parent", {}).get("runtime_model_state_sha256") == PARENT_MODEL_SHA256,
        "correction_cap_0p001": correction.get("corrected_value") == 0.001 and correction.get("all_other_master_fields_unchanged") is True,
        "correction_v3_net_gate": correction_v3.get("corrected_value") == "reported_but_not_a_zero_CW_hard_gate" and correction_v3.get("all_other_master_and_v2_fields_unchanged") is True,
        "authorization_GO": authorization.get("status") == "GO_P12",
        "framework_interface": callable(framework.profile_archive),
        "formal_interface": callable(formal.load_helper) and callable(formal.cuda_runtime),
    }
    if not all(semantic.values()):
        raise ProtocolError(f"static semantic input drift: {semantic}")
    archive_evidence: dict[str, Any] = {}
    for panel in PANEL_ORDER:
        _, evidence = read_regular(DATASETS[panel], DATA_SHA256[panel], f"{panel} train archive")
        archive_evidence[panel] = evidence
    return {
        "source": source,
        "self": self_evidence,
        "master": master_evidence,
        "correction": correction_evidence,
        "correction_v3": correction_v3_evidence,
        "authorization": auth_evidence,
        "parent_bytes": parent_bytes,
        "parent": parent_evidence,
        "framework": framework,
        "framework_evidence": framework_evidence,
        "formal": formal,
        "formal_evidence": formal_evidence,
        "archives": archive_evidence,
        "semantic": semantic,
    }


def run_formal(inputs: Mapping[str, Any], prereg_sha256: str) -> dict[str, Any]:
    prereg, prereg_evidence = verify_preregistration(prereg_sha256, inputs["self"]["sha256"])
    state = output_state()
    if not all((state["preregistration_present"], state["attempt_marker_absent"], state["result_absent"])):
        raise ProtocolError(f"formal target state drift: {state}")
    marker = publish(ATTEMPT_MARKER, canonical_json({
        "schema_version": f"{SCHEMA}-attempt-v1",
        "attempt": 1,
        "attempts_authorized": 1,
        "runner_sha256": inputs["self"]["sha256"],
        "preregistration_sha256": prereg_sha256,
        "parent_file_sha256": PARENT_FILE_SHA256,
        "parent_model_state_sha256": PARENT_MODEL_SHA256,
        "changed_candidate_constructed_or_evaluated": False,
        "retry_authorized": False,
    }))
    formal = inputs["formal"]
    framework = inputs["framework"]
    helper = formal.load_helper()
    dependency_checks = {
        "framework_bc_same_module": framework.bc is helper.bc,
        "framework_ppo_same_module": framework.ppo is helper.ppo,
        "bc_source_sha": sha256_bytes(Path(helper.bc.__file__).read_bytes()) == BC_SHA256,
        "ppo_source_sha": sha256_bytes(Path(helper.ppo.__file__).read_bytes()) == PPO_SHA256,
    }
    if not all(dependency_checks.values()):
        raise ProtocolError(f"profile dependency drift: {dependency_checks}")
    cuda_runtime = formal.cuda_runtime(helper)
    random.seed(SEED)
    helper.torch.manual_seed(SEED)
    helper.torch.cuda.manual_seed_all(SEED)
    checkpoint = helper.checkpoint_from_bytes(inputs["parent_bytes"], "guarded U468 parent")
    if checkpoint.get("update") != 468 or helper.evaluator.checkpoint_kind(checkpoint) != "ppo":
        raise ProtocolError("guarded parent is not PPO U468")
    parent_hash = helper.model_state_sha256(checkpoint["model_state_dict"])
    if parent_hash != PARENT_MODEL_SHA256:
        raise ProtocolError("guarded parent model hash drift")
    model, model_config, kind = helper.instantiate_checkpoint(checkpoint, helper.torch.device("cuda:0"))
    if kind != "ppo" or helper.model_state_sha256(model.state_dict()) != PARENT_MODEL_SHA256:
        raise ProtocolError("live guarded parent construction drift")
    profiles: dict[str, Any] = {}
    model_hash_after_panel: dict[str, str] = {}
    for panel in PANEL_ORDER:
        profile = framework.profile_archive(
            panel,
            DATASETS[panel],
            model,
            model_config,
            helper.torch.device("cuda:0"),
            BATCH_SIZE,
            KEEP,
        )
        profile["source"] = panel
        for list_name in ("near_wrong", "fragile_correct"):
            for row in profile[list_name]:
                row["source"] = panel
        profiles[panel] = profile
        model_hash_after_panel[panel] = helper.model_state_sha256(model.state_dict())
    shape_checks = {
        panel: {
            "rows_exact": int(profiles[panel]["rows"]) == EXPECTED_ROWS[panel],
            "all_rows_retained_across_two_lists": len(profiles[panel]["near_wrong"]) + len(profiles[panel]["fragile_correct"]) == EXPECTED_ROWS[panel],
            "train_members_only": bool(profiles[panel]["opened_members"]) and all(str(name).startswith("train/") for name in profiles[panel]["opened_members"]),
            "non_train_not_opened": profiles[panel]["non_train_members_opened"] is False,
            "archive_sha_exact": profiles[panel]["archive_sha256"] == DATA_SHA256[panel],
            "context34_exact": int(profiles[panel]["contexts"].get("34", -1)) == EXPECTED_CONTEXT34_ROWS[panel],
            "model_unchanged": model_hash_after_panel[panel] == PARENT_MODEL_SHA256,
        }
        for panel in PANEL_ORDER
    }
    if not all(all(checks.values()) for checks in shape_checks.values()):
        raise ProtocolError(f"profile shape/integrity failed: {shape_checks}")
    all_rows = [row for panel in PANEL_ORDER for list_name in ("near_wrong", "fragile_correct") for row in profiles[panel][list_name]]
    targets = select_targets(all_rows)
    guards = select_guards(all_rows)
    b256 = build_b256(all_rows, targets, guards)
    b256_sha = sha256_bytes(canonical_json(b256))
    selection = {
        "status": "completed_current_parent_train_only_selection",
        "target_margin": TARGET_MARGIN,
        "targets": targets,
        "guards": guards,
        "b256_rows": b256,
        "b256_rows_canonical_sha256": b256_sha,
        "contracts": {
            "target_count_exact5": len(targets) == 5,
            "guard_count_exact14": len(guards) == 14,
            "b256_count_exact256": len(b256) == 256,
            "all_target_guard_rows_in_b256": {row["line_sha256"] for row in (*targets, *guards)}.issubset({row["line_sha256"] for row in b256}),
            "all_rows_unique": len({row["line_sha256"] for row in b256}) == 256,
            "stratum_quotas_exact": dict(Counter(row["stratum"] for row in b256)) == STRATUM_QUOTAS,
            "single_candidate_recipe_inputs_only": True,
        },
    }
    if not all(selection["contracts"].values()):
        raise ProtocolError(f"selection terminal checks failed: {selection['contracts']}")
    nonfinite = sanitize_profiles(profiles)
    result = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_parent_only_train_profile_selection",
        "preregistration": prereg_evidence,
        "attempt_marker": marker,
        "input_lock": {
            "runner": inputs["self"],
            "master": inputs["master"],
            "correction_v2": inputs["correction"],
            "correction_v3": inputs["correction_v3"],
            "authorization": inputs["authorization"],
            "parent": inputs["parent"],
            "framework": inputs["framework_evidence"],
            "formal_v3": inputs["formal_evidence"],
            "train_archives": inputs["archives"],
        },
        "base": {
            "kind": "general_BC_plus_guarded_PPO_U468_before_special_BC",
            "checkpoint_file_sha256": PARENT_FILE_SHA256,
            "checkpoint_update": 468,
            "model_state_sha256": parent_hash,
            "live_model_final_state_sha256": helper.model_state_sha256(model.state_dict()),
            "changed_candidate_constructed_or_evaluated": False,
        },
        "runtime": cuda_runtime,
        "dependency_checks": dependency_checks,
        "profile_integrity": shape_checks,
        "profiles": profiles,
        "selection": selection,
        "nonfinite_encoding": nonfinite,
        "scope_audit": {
            "train_rows_profiled": sum(EXPECTED_ROWS.values()),
            "validation_test_broad_gold_rows_opened": 0,
            "optimizer_instances_steps_or_backward": 0,
            "changed_candidates_constructed_or_evaluated": 0,
            "checkpoint_or_model_artifact_writes": 0,
            "evidence_result_writes": 1,
            "network_package_upload_submission": False,
            "official_unique_changed_candidate_count_consumed": 0,
        },
        "preregistered_contract": prereg["profile_selection_contract"],
    }
    evidence = publish(OUTPUT, canonical_json(result))
    return {"status": result["status"], "result": evidence, "selection": {"targets": len(targets), "guards": len(guards), "b256": len(b256), "b256_sha256": b256_sha}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "formal"), required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--expected-preregistration-sha256")
    args = parser.parse_args()
    validate_runtime()
    inputs = load_static()
    if args.expected_tool_sha256 != inputs["self"]["sha256"]:
        raise ProtocolError("runner SHA mismatch")
    audit = static_audit(inputs["source"])
    if args.mode == "audit":
        print(canonical_json({
            "schema_version": f"{SCHEMA}-audit-v1",
            "status": "AUDIT_PASS_ZERO_ARCHIVE_MEMBERS_ZERO_CUDA",
            "runner": inputs["self"],
            "bindings": expected_bindings(inputs["self"]["sha256"]),
            "contract": expected_contract(),
            "static": audit,
            "semantic": inputs["semantic"],
            "output_state": output_state(),
            "archive_members_opened": 0,
            "cuda_accessed": False,
            "writes_performed": False,
        }).decode("utf-8"), end="")
        return
    if not args.expected_preregistration_sha256:
        raise ProtocolError("formal mode requires preregistration SHA")
    expected_argv = [
        str(SCRIPT), "--mode", "formal", "--expected-tool-sha256",
        inputs["self"]["sha256"], "--expected-preregistration-sha256",
        args.expected_preregistration_sha256,
    ]
    observed_argv = [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    if observed_argv != expected_argv:
        raise ProtocolError(f"formal argv drift: {observed_argv} != {expected_argv}")
    receipt = run_formal(inputs, args.expected_preregistration_sha256)
    print(canonical_json(receipt).decode("utf-8"), end="")


if __name__ == "__main__":
    main()

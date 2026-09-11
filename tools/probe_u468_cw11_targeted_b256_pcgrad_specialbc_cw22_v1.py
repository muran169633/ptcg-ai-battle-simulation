#!/usr/bin/env python3
"""One exact-CW11 targeted special-BC train-only endpoint for CW22.

The exact CW11 state is reconstructed inside the already-consumed historical
callback.  The new B256 cache opens only frozen ``train/`` members.  Candidate
selection uses no new validation, broad, Gold, package, upload, or submission
action; the historical reconstruction's already-consumed four valid rows are
reported separately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import stat
import sys
import types
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-targeted-b256-pcgrad-specialbc-cw22-v1"
SEED = 202608305

CW21 = TOOLS / "probe_u468_cw11_fixed_pcgrad_specialbc_cw21_v1.py"
CW21_SHA256 = "9c18db25e16fe6b9feb1ca72d0682a1baa31ad6dc19e1a6283fd7db2da0f9cf4"
CW21_MODE = 0o555
SELECTOR = TOOLS / "select_cw22_targeted_b256_from_cw11_profile_v2.py"
SELECTOR_SHA256 = "fe3ade2f38ea9e2972976654ffd584e85cc17183a7c1787c444261e515dd6c30"
SELECTOR_MODE = 0o555
SELECTION = ROOT / "artifacts/cw22_targeted_b256_selection_v2_20260803.json"
SELECTION_SHA256 = "035f071902c3f43acd5cec328400d6172a990bd19864d2d794cb2cdff5f7bee7"
SELECTION_MODE = 0o444
SELECTION_SCHEMA = "ptcg-cw22-targeted-train-b256-selection-v2"
SELECTION_PAYLOAD_SHA256 = "c4a9b711d8f5636daea8a428b9fd5e506bffcfdaeb614c7faedf7c75e4165a21"
PROFILE = ROOT / "artifacts/u468_cw11_full_train_margin_profile_v1_20260803.json"
PROFILE_SHA256 = "da684858d2c459416234cdc3cae1ffc32a5ec23a1ec376f66a0cf440776a846e"
PROFILE_MODE = 0o444

CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
CW11_VECTOR_SHA256 = "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
CW11_LEDGER_SHA256 = "6ae0c07a94627cb6fd0155f265f75d13bd7505b0e5f62cd62c3e26ceb7714436"
CW11_ACTOR_FLOAT32_LE_SHA256 = (
    "fc13661fec1801768a11d2c0366602727df08d6773c982f2ba273e194b8fafa6"
)
RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
RAW_NONACTOR_SHA256 = "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
EXPECTED_ACTOR_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)

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

BATCH_SIZE = 256
CONTEXT34_ROWS = 16
CONTEXT34_SAMPLE_WEIGHT = 1.0 / 3.0
EXPECTED_B256_CACHE_SHA256 = (
    "a936e106c6ef0e02895046ab70fa9480a9426f2b371d11c6d25ed1447771519b"
)
TASK_ORDER = (
    "pf_ctx0_hard",
    "pf_ctx7_hard",
    "dominic_ctx0_hard",
    "retention160",
)
RADII = (1.25e-4, 2.5e-4, 5.0e-4, 7.5e-4)
ADDITIONAL_FROM_CW11_CAP = 1.0e-3
RETENTION_LOSS_TOLERANCE = 1.0e-6
FIRST_ORDER_COSINE_EPSILON = 1.0e-12


class ProtocolError(RuntimeError):
    """Fail-closed CW22 protocol error."""


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


def regular_evidence(
    path: Path, expected_sha: str, expected_mode: int, label: str
) -> dict[str, Any]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise ProtocolError(f"unsafe {label} identity/mode")
    digest = sha256_file(path)
    after = path.lstat()
    checks = {
        "sha_exact": digest == expected_sha,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} evidence drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": int(after.st_size),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def load_locked_source(
    path: Path,
    expected_sha: str,
    expected_mode: int,
    module_name: str,
) -> tuple[ModuleType, dict[str, Any]]:
    evidence = regular_evidence(path, expected_sha, expected_mode, module_name)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha:
        raise ProtocolError(f"{module_name}: source changed after evidence")
    if module_name in sys.modules:
        raise ProtocolError(f"{module_name}: module name already occupied")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, evidence


def load_json_strict(path: Path, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ProtocolError(f"{label}: nonfinite JSON constant {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ProtocolError(f"{label}: root is not an object")
    return value


def is_dominic_special(row: Mapping[str, Any]) -> bool:
    metadata = row.get("train_row_metadata")
    signatures = (
        metadata.get("selected_option_signatures")
        if isinstance(metadata, Mapping)
        else None
    )
    return isinstance(signatures, Mapping) and sum(
        isinstance(signature, Mapping)
        and int(signature.get("type", -1)) == 10
        and int(signature.get("area", -1)) == 5
        for signature in signatures.values()
    ) >= 2


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.append((node.func.id, node.lineno))
        elif isinstance(node.func, ast.Attribute):
            calls.append((node.func.attr, node.lineno))
    forbidden = {
        "backward",
        "step",
        "zero_grad",
        "load_state_dict",
        "save",
        "submit",
        "upload",
        "package_submission",
    }
    hits = [f"{name}@{line}" for name, line in calls if name in forbidden]
    checks = {
        "no_optimizer_backward_checkpoint_or_external_calls": not hits,
        "exact_four_autograd_grad_call_iterations": sum(
            1 for name, _ in calls if name == "grad"
        )
        == 1,
        "exact_one_historical_run_probe_site": sum(
            1 for name, _ in calls if name == "run_probe"
        )
        == 1,
        "exact_one_publish_site": sum(
            1 for name, _ in calls if name == "publish_o_excl"
        )
        == 1,
        "dev_null_pycache_present": b'sys.pycache_prefix = "/dev/null"' in source,
        "stdout_summary_present": any(name == "print" for name, _ in calls),
    }
    if not all(checks.values()):
        raise ProtocolError(f"source audit failed: {checks}; hits={hits}")
    return {
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "checks": checks,
        "forbidden_hits": hits,
        "autograd_grad_call_site_count": 1,
        "autograd_grad_runtime_calls": len(TASK_ORDER),
        "pass": True,
    }


def validate_runtime(require_cuda: bool) -> dict[str, Any]:
    checks = {
        "repo_root": Path.cwd().resolve() == ROOT,
        "my_project_env": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1
        and sys.dont_write_bytecode is True,
        "pycache_prefix_dev_null": sys.pycache_prefix == "/dev/null",
    }
    if not all(checks.values()):
        raise ProtocolError(f"runtime drift: {checks}")
    result: dict[str, Any] = {
        "python": str(Path(sys.executable).resolve()),
        "checks": checks,
        "pass": True,
    }
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise ProtocolError("CUDA unavailable")
        result["cuda"] = {
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    return result


def validate_selection(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = payload.get("rows")
    fields = payload.get("selection_sha256_payload_fields")
    if not isinstance(rows, list) or not isinstance(fields, list):
        raise ProtocolError("selection rows/fields malformed")
    compact = [{str(key): row[str(key)] for key in fields} for row in rows]
    observed_selection_sha = hashlib.sha256(canonical_json(compact)).hexdigest()
    strata = Counter(str(row.get("stratum")) for row in rows)
    categories = Counter(str(row.get("category")) for row in rows)

    checks = {
        "schema_exact": payload.get("schema_version") == SELECTION_SCHEMA,
        "status_exact": payload.get("status")
        == "completed_frozen_train_only_selection",
        "base_exact_CW11": payload.get("base_model_state_sha256")
        == CW11_MODEL_SHA256,
        "selection_sha_field_exact": payload.get("selection_sha256")
        == SELECTION_PAYLOAD_SHA256,
        "selection_sha_recomputed_exact": observed_selection_sha
        == SELECTION_PAYLOAD_SHA256,
        "rows_exact256": len(rows) == BATCH_SIZE,
        "category_counts_exact": dict(categories)
        == {"hard": 96, "fragile": 144, "c34": 16},
        "stratum_counts_exact": dict(strata)
        == {
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
        },
        "Dominic_special_full9": sum(
            str(row.get("stratum")) == "dominic_ctx0_hard"
            and is_dominic_special(row)
            for row in rows
        )
        == 9,
        "FLG_guard_exact16_context3": sum(
            str(row.get("stratum")) == "broad_flg_retention"
            and int(row.get("context", -1)) == 3
            for row in rows
        )
        == 16,
        "line_sha_unique": len({str(row.get("line_sha256")) for row in rows})
        == BATCH_SIZE,
        "slots_exact": [int(row.get("final_b256_slot_zero_based", -1)) for row in rows]
        == list(range(BATCH_SIZE)),
        "all_train_members": all(
            str(row.get("member", "")).startswith("train/")
            and str(row.get("member", "")).endswith(".jsonl")
            and ".." not in str(row.get("member", "")).split("/")
            for row in rows
        ),
        "selector_scope_zero_valid": payload.get("scope", {}).get(
            "validation_or_test_rows_opened"
        )
        == 0,
        "selector_terminal_checks_all_true": all(
            value is True for value in payload.get("terminal_checks", {}).values()
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"selection contract drift: {checks}")
    return [dict(row) for row in rows], checks


def tensor_bytes_sha(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def tensor_batch_manifest(batch: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    manifest = []
    digest = hashlib.sha256()
    for name, tensor in sorted(batch.items()):
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
        record = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        manifest.append(record)
        digest.update(canonical_json(record))
    return digest.hexdigest(), manifest


def load_train_b256(
    rows: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
    bc: ModuleType,
    ppo: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wanted: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {
        source: {} for source in DATASETS
    }
    for row in rows:
        source = str(row["source"])
        key = (str(row["member"]), int(row["line_index"]))
        if source not in wanted or key in wanted[source]:
            raise ProtocolError("selection source/key drift")
        wanted[source][key] = row
    features: dict[tuple[str, str, int], dict[str, Any]] = {}
    opened: dict[str, list[str]] = {}
    for source, path in DATASETS.items():
        by_member: dict[str, set[int]] = defaultdict(set)
        for member, line_index in wanted[source]:
            if (
                not member.startswith("train/")
                or not member.endswith(".jsonl")
                or ".." in member.split("/")
            ):
                raise ProtocolError("attempted noncanonical/non-train member")
            by_member[member].add(line_index)
        opened[source] = []
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ProtocolError(f"{source}: duplicate ZIP member names")
            for member in sorted(by_member):
                archive.getinfo(member)
                opened[source].append(member)
                remaining = set(by_member[member])
                with archive.open(member) as handle:
                    for line_index, raw in enumerate(handle):
                        if line_index not in remaining:
                            continue
                        identity = wanted[source][(member, line_index)]
                        if hashlib.sha256(raw).hexdigest() != identity["line_sha256"]:
                            raise ProtocolError("selected train line SHA drift")
                        source_row = json.loads(raw)
                        if str(source_row.get("split", "")) != "train":
                            raise ProtocolError("selected source row is not train")
                        if (
                            str(source_row.get("episode_id", ""))
                            != str(identity["episode_id"])
                            or str(source_row.get("team_name", ""))
                            != str(identity["team_name"])
                        ):
                            raise ProtocolError("selected source metadata drift")
                        previous_max = bc.MAX_ACTION_COUNT
                        bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
                        try:
                            feature = bc.featurize_row(
                                source_row,
                                int(model_config["hash_size"]),
                                int(model_config["max_state_entities"]),
                            )
                        finally:
                            bc.MAX_ACTION_COUNT = previous_max
                        if feature is None:
                            raise ProtocolError("selected row no longer featurizes")
                        expert = [int(value) for value in source_row.get("action", [])]
                        checks = {
                            "expert": expert
                            == [int(value) for value in identity["expert_order"]],
                            "context": int(feature["context"])
                            == int(identity["context"]),
                            "min_count": int(feature["min_count"])
                            == int(identity["min_count"]),
                            "max_count": int(feature["max_count"])
                            == int(identity["max_count"]),
                        }
                        if not all(checks.values()):
                            raise ProtocolError(f"selected decision drift: {checks}")
                        feature["action_sequence"] = expert
                        # CW22 is an equal-row targeted panel.  Source-archive
                        # replay weights are intentionally replaced; only the
                        # established context-34 ordered-action downweight is
                        # retained.
                        feature["sample_weight"] = 1.0
                        if int(feature["context"]) == ppo.SKILL_ORDER_CONTEXT:
                            feature["sample_weight"] = CONTEXT34_SAMPLE_WEIGHT
                        features[(source, member, line_index)] = feature
                        remaining.remove(line_index)
                        if not remaining:
                            break
                if remaining:
                    raise ProtocolError(f"selected lines missing: {source}/{member}")
    ordered_features = [
        features[(str(row["source"]), str(row["member"]), int(row["line_index"]))]
        for row in rows
    ]
    batch = bc.collate_decisions(
        ordered_features,
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    context34_mask = batch["contexts"] == ppo.SKILL_ORDER_CONTEXT
    checks = {
        "batch_rows_exact256": int(batch["action_counts"].shape[0]) == BATCH_SIZE,
        "context34_rows_exact16": int(context34_mask.sum()) == CONTEXT34_ROWS,
        "context34_sample_weights_exact_one_third": bool(
            __import__("torch").allclose(
                batch["sample_weights"][context34_mask],
                __import__("torch").full_like(
                    batch["sample_weights"][context34_mask],
                    CONTEXT34_SAMPLE_WEIGHT,
                ),
                rtol=0.0,
                atol=1e-7,
            )
        ),
        "non_context34_sample_weights_exact_one": bool(
            __import__("torch").allclose(
                batch["sample_weights"][~context34_mask],
                __import__("torch").ones_like(
                    batch["sample_weights"][~context34_mask]
                ),
                rtol=0.0,
                atol=0.0,
            )
        ),
        "all_sample_weights_finite_positive": bool(
            __import__("torch").isfinite(batch["sample_weights"]).all()
            and (batch["sample_weights"] > 0).all()
        ),
        "all_action_counts_positive": bool((batch["action_counts"] > 0).all()),
        "all_sources_opened_train_members_only": all(
            members and all(member.startswith("train/") for member in members)
            for members in opened.values()
        ),
    }
    cache_sha, manifest = tensor_batch_manifest(batch)
    checks["cache_sha_exact_CPU_preflight"] = cache_sha == EXPECTED_B256_CACHE_SHA256
    if not all(checks.values()):
        raise ProtocolError(f"collated B256 gate failed: {checks}")
    return batch, {
        "checks": checks,
        "opened_train_members": opened,
        "non_train_members_opened": False,
        "cache_sha256": cache_sha,
        "tensor_manifest": manifest,
    }


def ordered_nll_per_row(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> Any:
    import torch

    logits = outputs["policy_logits"].float()
    option_mask = batch["option_mask"].bool()
    action_counts = batch["action_counts"]
    sequences = batch["action_sequences"]
    selected = torch.zeros_like(option_mask)
    result = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    for step in range(sequences.shape[1]):
        active = step < action_counts
        if not bool(active.any()):
            break
        chosen = sequences[:, step]
        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        active_chosen = chosen[active]
        if not bool(
            (
                option_mask[active_rows, active_chosen]
                & ~selected[active_rows, active_chosen]
            ).all()
        ):
            raise ProtocolError("illegal/duplicate expert action")
        allowed = option_mask & ~selected
        log_probs = torch.log_softmax(logits.masked_fill(~allowed, -1e9), dim=1)
        safe_chosen = chosen.clamp(0, logits.shape[1] - 1)
        chosen_log_prob = log_probs.gather(1, safe_chosen.unsqueeze(1)).squeeze(1)
        result -= torch.where(active, chosen_log_prob, torch.zeros_like(result))
        selected.scatter_(1, safe_chosen.unsqueeze(1), active.unsqueeze(1))
    return result


def ordered_margin(
    logits: Any, option_mask: Any, expert_order: Sequence[int]
) -> float:
    remaining = option_mask.clone()
    margins: list[float] = []
    for chosen in expert_order:
        allowed = remaining.nonzero(as_tuple=False).squeeze(1)
        competitors = allowed[allowed != int(chosen)]
        if int(competitors.numel()) == 0:
            continue
        margins.append(float(logits[int(chosen)] - logits[competitors].max()))
        remaining[int(chosen)] = False
    return min(margins) if margins else float("inf")


def aggregate_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not entries:
        raise ProtocolError("empty evaluation aggregation")
    weight_sum = sum(float(row["sample_weight"]) for row in entries)
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ProtocolError("invalid evaluation sample-weight sum")
    unweighted_nll = sum(float(row["ordered_nll"]) for row in entries) / len(entries)
    weighted_nll = sum(
        float(row["ordered_nll"]) * float(row["sample_weight"])
        for row in entries
    ) / weight_sum
    return {
        "rows": len(entries),
        "ordered_nll": weighted_nll,
        "ordered_nll_weighted": weighted_nll,
        "ordered_nll_unweighted": unweighted_nll,
        "sample_weight_sum": weight_sum,
        "ordered_correct": sum(bool(row["ordered_correct"]) for row in entries),
        "set_correct": sum(bool(row["set_correct"]) for row in entries),
        "count_correct": sum(bool(row["count_correct"]) for row in entries),
        "selection_margin_min": min(float(row["selection_margin"]) for row in entries),
        "selection_margin_mean": sum(float(row["selection_margin"]) for row in entries)
        / len(entries),
    }


def evaluate_outputs(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    ppo: ModuleType,
) -> tuple[dict[str, Any], dict[str, bool], list[list[int]]]:
    per_row_nll = ordered_nll_per_row(outputs, batch)
    predictions, _, _, _ = ppo.sample_ordered_actions(
        outputs,
        batch,
        deterministic=True,
        canonicalize_order=False,
    )
    logits = outputs["policy_logits"].float().detach().cpu()
    masks = batch["option_mask"].bool().detach().cpu()
    entries: list[dict[str, Any]] = []
    correct: dict[str, bool] = {}
    predicted_lists: list[list[int]] = []
    for index, identity in enumerate(rows):
        count = int(batch["action_counts"][index])
        expert = [
            int(value)
            for value in batch["action_sequences"][index, :count].detach().cpu().tolist()
        ]
        predicted = [int(value) for value in predictions[index]]
        predicted_lists.append(predicted)
        line_sha = str(identity["line_sha256"])
        if line_sha in correct:
            raise ProtocolError("duplicate line SHA in evaluation")
        is_ordered = predicted == expert
        correct[line_sha] = is_ordered
        entries.append(
            {
                "source": str(identity["source"]),
                "stratum": str(identity["stratum"]),
                "category": str(identity["category"]),
                "line_sha256": line_sha,
                "ordered_nll": float(per_row_nll[index].detach().cpu()),
                "sample_weight": float(
                    batch["sample_weights"][index].detach().cpu()
                ),
                "ordered_correct": is_ordered,
                "set_correct": set(predicted) == set(expert),
                "count_correct": len(predicted) == len(expert),
                "selection_margin": ordered_margin(
                    logits[index], masks[index], expert
                ),
            }
        )
    strata_names = sorted({str(row["stratum"]) for row in rows})
    by_stratum = {
        name: aggregate_entries([row for row in entries if row["stratum"] == name])
        for name in strata_names
    }
    hard = [row for row in entries if row["category"] == "hard"]
    retention = [row for row in entries if row["category"] != "hard"]
    policy_sha = tensor_bytes_sha(outputs["policy_logits"])
    return {
        "rows": len(entries),
        "hard": aggregate_entries(hard),
        "retention": aggregate_entries(retention),
        "by_stratum": by_stratum,
        "native_output": {
            "policy_logits_dtype": str(outputs["policy_logits"].dtype),
            "policy_logits_sha256": policy_sha,
            "count_logits_dtype": str(outputs["count_logits"].dtype),
            "count_logits_sha256": tensor_bytes_sha(outputs["count_logits"]),
            "value_logits_dtype": str(outputs["value_logits"].dtype),
            "value_logits_sha256": tensor_bytes_sha(outputs["value_logits"]),
        },
    }, correct, predicted_lists


def zero_gradient(parameters: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    return {
        name: torch.zeros_like(parameter.detach(), device="cpu", dtype=torch.float64)
        for name, parameter in parameters.items()
    }


def clone_gradient(gradient: Mapping[str, Any]) -> dict[str, Any]:
    return {name: tensor.clone() for name, tensor in gradient.items()}


def gradient_dot(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    import torch

    return float(
        sum(
            torch.dot(left[name].reshape(-1), right[name].reshape(-1))
            for name in EXPECTED_ACTOR_NAMES
        )
    )


def gradient_norm(gradient: Mapping[str, Any]) -> float:
    value = gradient_dot(gradient, gradient)
    if value <= 0.0 or not math.isfinite(value):
        raise ProtocolError("gradient norm is zero/nonfinite")
    return math.sqrt(value)


def gradient_add(
    left: Mapping[str, Any], right: Mapping[str, Any], scale: float
) -> dict[str, Any]:
    return {
        name: left[name] + float(scale) * right[name]
        for name in EXPECTED_ACTOR_NAMES
    }


def gradient_scale(gradient: Mapping[str, Any], scale: float) -> dict[str, Any]:
    return {name: tensor * float(scale) for name, tensor in gradient.items()}


def gradient_sha(gradient: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in EXPECTED_ACTOR_NAMES:
        value = gradient[name].detach().cpu().contiguous().to(torch.float64)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def gradient_report(gradient: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    per_tensor = {}
    for name in EXPECTED_ACTOR_NAMES:
        tensor = gradient[name]
        per_tensor[name] = {
            "l2": float(tensor.norm()),
            "max_abs": float(tensor.abs().max()),
            "nonzero_elements": int(torch.count_nonzero(tensor)),
            "finite": bool(torch.isfinite(tensor).all()),
        }
    return {
        "l2": gradient_norm(gradient),
        "float64_le_sha256": gradient_sha(gradient),
        "all_six_nonzero": all(row["nonzero_elements"] > 0 for row in per_tensor.values()),
        "finite": all(row["finite"] for row in per_tensor.values()),
        "per_tensor": per_tensor,
    }


def compute_task_gradients(
    model: Any,
    parameters: Mapping[str, Any],
    batch_cpu: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    device: Any,
    ppo: ModuleType,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any], dict[str, Any], dict[str, bool], list[list[int]], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ProtocolError("gradient buffers present before targeted probe")
    batch = {key: value.to(device, non_blocking=True) for key, value in batch_cpu.items()}
    outputs = ppo.model_forward(model, batch, device)
    if outputs["policy_logits"].dtype != torch.bfloat16:
        raise ProtocolError("native CUDA policy logits are not BF16")
    baseline_report, baseline_correct, baseline_predictions = evaluate_outputs(
        outputs, batch, rows, ppo
    )
    baseline_native = {
        "count_logits": outputs["count_logits"].detach().clone(),
        "value_logits": outputs["value_logits"].detach().clone(),
    }
    logits = outputs["policy_logits"].float()
    task_losses: dict[str, Any] = {}
    for task in TASK_ORDER[:3]:
        indices = [index for index, row in enumerate(rows) if row["stratum"] == task]
        if len(indices) != 32:
            raise ProtocolError(f"{task}: hard quota drift")
        margins = []
        for index in indices:
            expert = int(rows[index]["expert_order"][0])
            predicted = baseline_predictions[index]
            if len(predicted) != 1 or predicted[0] == expert:
                raise ProtocolError(f"{task}: final B256 hard baseline is not wrong")
            margins.append(logits[index, expert] - logits[index, predicted[0]])
        task_losses[task] = functional.softplus(-torch.stack(margins)).mean()
    retention_indices = [
        index for index, row in enumerate(rows) if str(row["category"]) != "hard"
    ]
    if len(retention_indices) != 160:
        raise ProtocolError("retention160 quota drift")
    per_row_nll = ordered_nll_per_row(outputs, batch)
    retention_mask = torch.tensor(retention_indices, dtype=torch.long, device=device)
    retention_weights = batch["sample_weights"][retention_mask].float()
    task_losses["retention160"] = (
        per_row_nll[retention_mask] * retention_weights
    ).sum() / retention_weights.sum().clamp_min(1.0)

    parameter_tuple = tuple(parameters[name] for name in EXPECTED_ACTOR_NAMES)
    gradients: dict[str, Any] = {}
    losses: dict[str, float] = {}
    per_task: dict[str, Any] = {}
    for task_index, task in enumerate(TASK_ORDER):
        values = torch.autograd.grad(
            task_losses[task],
            parameter_tuple,
            retain_graph=task_index + 1 < len(TASK_ORDER),
            create_graph=False,
            allow_unused=False,
            materialize_grads=False,
        )
        gradient = zero_gradient(parameters)
        for name, value in zip(EXPECTED_ACTOR_NAMES, values):
            if not bool(torch.isfinite(value).all()):
                raise ProtocolError(f"{task}/{name}: nonfinite gradient")
            gradient[name].copy_(value.detach().cpu().to(torch.float64))
        gradients[task] = gradient
        losses[task] = float(task_losses[task].detach().cpu())
        per_task[task] = gradient_report(gradient)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ProtocolError("autograd.grad materialized .grad buffers")
    del outputs, task_losses, per_row_nll
    return (
        gradients,
        losses,
        {"per_task": per_task, "autograd_grad_calls": len(TASK_ORDER), "backward_calls": 0},
        baseline_report,
        baseline_correct,
        baseline_predictions,
        baseline_native,
    )


def fixed_order_pcgrad(
    gradients: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    projected: dict[str, Any] = {}
    trace = []
    for task_index, task in enumerate(TASK_ORDER):
        current = clone_gradient(gradients[task])
        others = TASK_ORDER[task_index + 1 :] + TASK_ORDER[:task_index]
        for other in others:
            reference = gradients[other]
            before = gradient_dot(current, reference)
            reference_square = gradient_dot(reference, reference)
            if reference_square <= 0.0:
                raise ProtocolError(f"zero PCGrad reference: {other}")
            projected_conflict = before < 0.0
            coefficient = before / reference_square if projected_conflict else 0.0
            if projected_conflict:
                current = gradient_add(current, reference, -coefficient)
            after = gradient_dot(current, reference)
            trace.append(
                {
                    "task": task,
                    "reference": other,
                    "dot_before": before,
                    "projection_applied": projected_conflict,
                    "coefficient": coefficient,
                    "dot_after": after,
                }
            )
        projected[task] = current
    result = zero_gradient(next(iter(gradients.values())))
    for task in TASK_ORDER:
        result = gradient_add(result, projected[task], 1.0 / len(TASK_ORDER))
    norm = gradient_norm(result)
    effects = {}
    for task in TASK_ORDER:
        task_norm = gradient_norm(gradients[task])
        dot = gradient_dot(gradients[task], result)
        cosine = dot / (task_norm * norm)
        effects[task] = {
            "dot": dot,
            "cosine": cosine,
            "strict_first_order_descent": dot > 0.0,
            "robust_first_order_descent": dot > 0.0
            and cosine > FIRST_ORDER_COSINE_EPSILON,
        }
    return result, {
        "task_order": list(TASK_ORDER),
        "other_order": "cyclic",
        "reference_kind": "original_unprojected",
        "aggregation": "arithmetic_mean",
        "trace": trace,
        "direction": gradient_report(result),
        "effects": effects,
        "all_tasks_robust_first_order_descent": all(
            row["robust_first_order_descent"] for row in effects.values()
        ),
    }


def compare_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_correct: Mapping[str, bool],
    candidate_correct: Mapping[str, bool],
    rows: Sequence[Mapping[str, Any]],
    immutable_native: Mapping[str, bool],
) -> dict[str, Any]:
    hard_rows = [row for row in rows if row["category"] == "hard"]
    retention_rows = [row for row in rows if row["category"] != "hard"]
    repairs = [
        str(row["line_sha256"])
        for row in hard_rows
        if not baseline_correct[str(row["line_sha256"])]
        and candidate_correct[str(row["line_sha256"])]
    ]
    retention_flips = [
        str(row["line_sha256"])
        for row in retention_rows
        if baseline_correct[str(row["line_sha256"])]
        and not candidate_correct[str(row["line_sha256"])]
    ]
    repairs_by_stratum = {
        task: sum(
            str(row["line_sha256"]) in repairs
            for row in hard_rows
            if row["stratum"] == task
        )
        for task in TASK_ORDER[:3]
    }
    hard_improvements = {
        task: float(baseline["by_stratum"][task]["ordered_nll"])
        - float(candidate["by_stratum"][task]["ordered_nll"])
        for task in TASK_ORDER[:3]
    }
    retention_strata = sorted(
        {
            str(row["stratum"])
            for row in retention_rows
        }
    )
    retention_improvements_by_stratum = {
        stratum: float(baseline["by_stratum"][stratum]["ordered_nll"])
        - float(candidate["by_stratum"][stratum]["ordered_nll"])
        for stratum in retention_strata
    }
    dominic_special_repairs = sum(
        str(row["line_sha256"]) in repairs and is_dominic_special(row)
        for row in hard_rows
        if row["stratum"] == "dominic_ctx0_hard"
    )
    checks = {
        "all_three_hard_losses_improve": all(
            value > 0.0 for value in hard_improvements.values()
        ),
        "pf_ctx0_repairs_at_least1": repairs_by_stratum["pf_ctx0_hard"] >= 1,
        "pf_ctx7_repairs_at_least1": repairs_by_stratum["pf_ctx7_hard"] >= 1,
        "dominic_repairs_at_least1": repairs_by_stratum["dominic_ctx0_hard"] >= 1,
        "dominic_special9_repairs_at_least1": dominic_special_repairs >= 1,
        "total_hard_repairs_at_least3": len(repairs) >= 3,
        "retention160_zero_correct_to_wrong": not retention_flips,
        "retention160_all_ordered_correct": candidate["retention"]["ordered_correct"]
        == 160,
        "retention160_loss_nondegrade": float(candidate["retention"]["ordered_nll"])
        <= float(baseline["retention"]["ordered_nll"])
        + RETENTION_LOSS_TOLERANCE,
        "each_retention_stratum_loss_nondegrade": all(
            value >= -RETENTION_LOSS_TOLERANCE
            for value in retention_improvements_by_stratum.values()
        ),
        "count_logits_native_exact": immutable_native["count_logits_native_exact"],
        "value_logits_native_exact": immutable_native["value_logits_native_exact"],
        "policy_logits_native_BF16": immutable_native[
            "policy_logits_native_BF16"
        ],
        "count_logits_dtype_exact": immutable_native["count_logits_dtype_exact"],
        "value_logits_dtype_exact": immutable_native["value_logits_dtype_exact"],
        "candidate_all256_count_correct": candidate["hard"]["count_correct"]
        == 96
        and candidate["retention"]["count_correct"] == 160,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "hard_loss_improvements": hard_improvements,
        "hard_repairs": len(repairs),
        "hard_repairs_by_stratum": repairs_by_stratum,
        "dominic_special9_repairs": dominic_special_repairs,
        "hard_repair_line_sha256": repairs,
        "retention_correct_to_wrong_count": len(retention_flips),
        "retention_correct_to_wrong_line_sha256": retention_flips,
        "retention_loss_improvement": float(baseline["retention"]["ordered_nll"])
        - float(candidate["retention"]["ordered_nll"]),
        "retention_loss_improvements_by_stratum": retention_improvements_by_stratum,
        "immutable_native": dict(immutable_native),
    }


def validate_exact_context(
    context: Mapping[str, Any], cw20: ModuleType, cw15: ModuleType
) -> dict[str, Any]:
    import numpy as np

    required = {
        "helper",
        "model",
        "checkpoint",
        "model_config",
        "raw_actor",
        "raw_model_state_sha256",
        "raw_nonactor_sha256",
        "terminal_cumulative_float64",
        "terminal_cumulative_float64_le_sha256",
        "success_iteration",
        "cw10_success_iteration",
        "selected_row_gate",
        "active_pair_ledger",
        "expanded_row_count",
        "candidate_model_state_sha256",
    }
    if not required.issubset(context):
        raise ProtocolError("historical callback context schema drift")
    helper = context["helper"]
    model = context["model"]
    state = model.state_dict()
    named = dict(model.named_parameters())
    terminal = np.asarray(context["terminal_cumulative_float64"], dtype=np.float64)
    ledger_sha = hashlib.sha256(cw15.canonical_json(context["active_pair_ledger"])).hexdigest()
    checkpoint = context["checkpoint"]
    checkpoint_state = checkpoint.get("model_state_dict") if isinstance(checkpoint, Mapping) else None
    checks = {
        "device_cuda": next(model.parameters()).device.type == "cuda",
        "model_exact_CW11": helper.model_state_sha256(state) == CW11_MODEL_SHA256,
        "context_candidate_exact_CW11": context["candidate_model_state_sha256"]
        == CW11_MODEL_SHA256,
        "vector_field_exact": context["terminal_cumulative_float64_le_sha256"]
        == CW11_VECTOR_SHA256,
        "vector_recomputed_exact": cw15.float64_vector_sha256(terminal)
        == CW11_VECTOR_SHA256,
        "vector_shape_finite_l2": terminal.shape == (cw15.ACTOR6_FLAT_LENGTH,)
        and bool(np.isfinite(terminal).all())
        and math.isclose(
            float(np.linalg.norm(terminal)),
            0.00792176975336988,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "ledger_exact34": ledger_sha == CW11_LEDGER_SHA256
        and len(context["active_pair_ledger"]) == 34,
        "raw_model_anchor": context["raw_model_state_sha256"] == RAW_MODEL_SHA256,
        "raw_nonactor_anchor": context["raw_nonactor_sha256"] == RAW_NONACTOR_SHA256,
        "iteration_rows_exact": int(context["success_iteration"]) == 3
        and int(context["cw10_success_iteration"]) == 9
        and int(context["expanded_row_count"]) == 33,
        "selected_row_gate_pass": context["selected_row_gate"].get("pass") is True,
        "model_eval80": model.training is False and len(state) == 80,
        "actor_names_present": all(name in named for name in EXPECTED_ACTOR_NAMES),
        "actor6_float32_dim65793": all(
            named[name].dtype == helper.torch.float32 for name in EXPECTED_ACTOR_NAMES
        )
        and sum(int(named[name].numel()) for name in EXPECTED_ACTOR_NAMES) == 65793,
        "actor_bytes_exact": hashlib.sha256(
            cw20.actor_bytes(named, EXPECTED_ACTOR_NAMES, np)
        ).hexdigest()
        == CW11_ACTOR_FLOAT32_LE_SHA256,
        "nonactor_exact_raw": cw20.nonactor_sha(model, EXPECTED_ACTOR_NAMES, helper)
        == RAW_NONACTOR_SHA256,
        "raw_parent_update468": isinstance(checkpoint, Mapping)
        and int(checkpoint.get("update", -1)) == 468,
        "raw_parent_model_exact": isinstance(checkpoint_state, Mapping)
        and helper.model_state_sha256(checkpoint_state) == RAW_MODEL_SHA256,
        "raw_parent_not_eval_only": isinstance(checkpoint, Mapping)
        and "evaluation_only" not in checkpoint
        and "resume_forbidden" not in checkpoint,
        "model_config_exact_parent": isinstance(checkpoint, Mapping)
        and dict(context["model_config"]) == dict(checkpoint.get("model_config", {})),
    }
    if not all(checks.values()):
        raise ProtocolError(f"exact CW11 context drift: {checks}")
    return checks


def run_targeted_core(
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cw20: ModuleType,
    cw19: ModuleType,
    cw15: ModuleType,
    modules: Mapping[str, ModuleType],
) -> dict[str, Any]:
    import numpy as np
    import torch

    context_checks = validate_exact_context(context, cw20, cw15)
    helper = context["helper"]
    model = context["model"]
    bc = helper.bc
    ppo = helper.ppo
    dependency_checks = {
        "BC_source_exact_CW15": sha256_file(Path(bc.__file__))
        == cw15.MODULE_SHAS[cw15.BC],
        "PPO_source_exact_CW15": sha256_file(Path(ppo.__file__))
        == cw15.MODULE_SHAS[cw15.PPO],
    }
    if not all(dependency_checks.values()):
        raise ProtocolError(f"BC/PPO dependency drift: {dependency_checks}")
    batch_cpu, cache_audit = load_train_b256(
        rows, context["model_config"], bc, ppo
    )
    device = next(model.parameters()).device
    named = dict(model.named_parameters())
    requires_grad_before = {
        name: bool(parameter.requires_grad) for name, parameter in named.items()
    }
    for parameter in named.values():
        parameter.requires_grad_(False)
        parameter.grad = None
    for name in EXPECTED_ACTOR_NAMES:
        named[name].requires_grad_(True)
    parameters = {name: named[name] for name in EXPECTED_ACTOR_NAMES}
    parameter_sequence = modules["geometry"].configure_actor6(model)
    if any(parameters[name] is not value for name, value in zip(EXPECTED_ACTOR_NAMES, parameter_sequence)):
        raise ProtocolError("actor6 implementation tensor identity drift")
    cw11_actor = cw20.clone_actor(parameters, EXPECTED_ACTOR_NAMES)
    cw11_flat = cw20.flat_actor(parameters, EXPECTED_ACTOR_NAMES, np)
    try:
        (
            gradients,
            losses,
            gradient_audit,
            baseline_report,
            baseline_correct,
            baseline_predictions,
            baseline_native,
        ) = compute_task_gradients(
            model, parameters, batch_cpu, rows, device, ppo
        )
        baseline_checks = {
            "rows_exact256": baseline_report["rows"] == 256,
            "hard96_all_ordered_wrong": baseline_report["hard"]["rows"] == 96
            and baseline_report["hard"]["ordered_correct"] == 0,
            "hard96_all_set_wrong": baseline_report["hard"]["set_correct"] == 0,
            "hard96_all_count_correct": baseline_report["hard"]["count_correct"] == 96,
            "retention160_all_ordered_correct": baseline_report["retention"]["rows"]
            == 160
            and baseline_report["retention"]["ordered_correct"] == 160,
            "retention160_all_set_correct": baseline_report["retention"]["set_correct"]
            == 160,
            "retention160_all_count_correct": baseline_report["retention"]["count_correct"]
            == 160,
            "profile_category_contract_preserved_in_final_B256": all(
                (
                    not baseline_correct[str(row["line_sha256"])]
                    if row["category"] == "hard"
                    else baseline_correct[str(row["line_sha256"])]
                )
                for row in rows
            ),
            "native_policy_BF16": baseline_report["native_output"][
                "policy_logits_dtype"
            ]
            == "torch.bfloat16",
        }
        if not all(baseline_checks.values()):
            return {
                "decision": "NO_GO_CW22_TARGETED_B256_BASELINE",
                "reason": "FINAL_B256_BASELINE_CERTIFICATION_FAILED",
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline": baseline_report,
                "baseline_checks": baseline_checks,
                "changed_candidate_train_shadow_count": 0,
                "candidate_payload": None,
            }
        direction, pcgrad = fixed_order_pcgrad(gradients)
        if not pcgrad["all_tasks_robust_first_order_descent"]:
            return {
                "decision": "NO_GO_CW22_TARGETED_B256_DIRECTION",
                "reason": "TARGETED_PCGRAD_DIRECTION_GATE_FAILED",
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline": baseline_report,
                "baseline_checks": baseline_checks,
                "task_losses": losses,
                "gradient_audit": gradient_audit,
                "pcgrad": pcgrad,
                "changed_candidate_train_shadow_count": 0,
                "candidate_payload": None,
            }
        direction_flat = np.concatenate(
            [
                direction[name].detach().cpu().numpy().reshape(-1)
                for name in EXPECTED_ACTOR_NAMES
            ]
        )
        direction_l2 = float(np.linalg.norm(direction_flat))
        trials: list[dict[str, Any]] = []
        selected_flat = None
        selected_hash = None
        selected_trial = None
        selected_actor_bytes = None
        for radius in RADII:
            planned = -float(radius) * direction_flat / direction_l2
            cw20.apply_flat_actor(
                parameters, EXPECTED_ACTOR_NAMES, cw11_flat + planned, torch
            )
            actual_flat = cw20.flat_actor(parameters, EXPECTED_ACTOR_NAMES, np)
            actual_delta = actual_flat - cw11_flat
            actual_l2 = float(np.linalg.norm(actual_delta))
            candidate_hash = helper.model_state_sha256(model.state_dict())
            with torch.no_grad():
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch_cpu.items()
                }
                outputs = ppo.model_forward(model, batch, device)
                candidate_report, candidate_correct, _ = evaluate_outputs(
                    outputs, batch, rows, ppo
                )
                immutable = {
                    "count_logits_native_exact": bool(
                        torch.equal(outputs["count_logits"], baseline_native["count_logits"])
                    ),
                    "value_logits_native_exact": bool(
                        torch.equal(outputs["value_logits"], baseline_native["value_logits"])
                    ),
                    "policy_logits_native_BF16": outputs["policy_logits"].dtype
                    == torch.bfloat16,
                    "count_logits_dtype_exact": str(outputs["count_logits"].dtype)
                    == baseline_report["native_output"]["count_logits_dtype"],
                    "value_logits_dtype_exact": str(outputs["value_logits"].dtype)
                    == baseline_report["native_output"]["value_logits_dtype"],
                }
            train_gate = compare_candidate(
                baseline_report,
                candidate_report,
                baseline_correct,
                candidate_correct,
                rows,
                immutable,
            )
            integrity = {
                "actual_l2_within_radius_rounding": abs(actual_l2 - radius) <= 5e-7,
                "additional_from_CW11_cap": actual_l2 <= ADDITIONAL_FROM_CW11_CAP,
                "candidate_model_changed": candidate_hash != CW11_MODEL_SHA256,
                "nonactor_exact_raw": cw20.nonactor_sha(
                    model, EXPECTED_ACTOR_NAMES, helper
                )
                == RAW_NONACTOR_SHA256,
                "all_six_actor_tensors_changed": all(
                    not torch.equal(parameters[name], cw11_actor[name])
                    for name in EXPECTED_ACTOR_NAMES
                ),
                "actual_delta_finite": bool(np.isfinite(actual_delta).all()),
            }
            trial_pass = all(integrity.values()) and train_gate["pass"]
            trial = {
                "radius": radius,
                "candidate_model_state_sha256": candidate_hash,
                "actual_additional_from_CW11_l2": actual_l2,
                "actual_delta_float64_le_sha256": cw19.float64_sha(actual_delta, np),
                "integrity": integrity,
                "candidate": candidate_report,
                "train_gate": train_gate,
                "pass": trial_pass,
            }
            trials.append(trial)
            if trial_pass:
                selected_flat = actual_flat.copy()
                selected_hash = candidate_hash
                selected_trial = trial
                selected_actor_bytes = cw20.actor_bytes(
                    parameters, EXPECTED_ACTOR_NAMES, np
                )
                cw20.restore_actor(parameters, EXPECTED_ACTOR_NAMES, cw11_actor, torch)
                break
            cw20.restore_actor(parameters, EXPECTED_ACTOR_NAMES, cw11_actor, torch)
        if selected_flat is None:
            return {
                "decision": "NO_GO_CW22_TARGETED_B256_PREFLIGHT",
                "reason": "NO_PREREGISTERED_RADIUS_PASSED_TARGETED_TRAIN_GATE",
                "context_checks": context_checks,
                "dependency_checks": dependency_checks,
                "cache": cache_audit,
                "baseline": baseline_report,
                "baseline_checks": baseline_checks,
                "task_losses": losses,
                "gradient_audit": gradient_audit,
                "pcgrad": pcgrad,
                "radii_contract": list(RADII),
                "trials": trials,
                "changed_candidate_train_shadow_count": len(trials),
                "candidate_payload": None,
            }
        cw20.apply_flat_actor(
            parameters, EXPECTED_ACTOR_NAMES, selected_flat, torch
        )
        layout = cw20.actor_layout(parameters, EXPECTED_ACTOR_NAMES)
        payload = {
            "anchor": {
                "reconstruction_base": "original_raw_U468",
                "raw_checkpoint": (
                    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_"
                    "design202608090/ppo_stage/B_gold_league/seed-202607336/"
                    "checkpoints/update-0468.pt"
                ),
                "raw_checkpoint_file_sha256": (
                    "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
                ),
                "raw_model_state_sha256": RAW_MODEL_SHA256,
                "CW11_provenance_model_state_sha256": CW11_MODEL_SHA256,
                "CW11_provenance_vector_float64_le_sha256": CW11_VECTOR_SHA256,
                "CW11_provenance_active_pair_ledger_sha256": CW11_LEDGER_SHA256,
                "CW11_provenance_actor_float32_le_sha256": CW11_ACTOR_FLOAT32_LE_SHA256,
                "CW11_materialized_eval_only_checkpoint_used": False,
                "terminal_model_state_sha256": selected_hash,
            },
            "formula": (
                "load_original_raw_U468_then_replace_only_six_absolute_"
                "actor_float32_tensors_from_frozen_payload"
            ),
            "actor_names": list(EXPECTED_ACTOR_NAMES),
            "actor_layout": layout,
            "actor_layout_sha256": hashlib.sha256(
                json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "candidate_actor_float32_le": cw20.xz_payload(selected_actor_bytes),
            "actual_additional_from_CW11_l2": selected_trial[
                "actual_additional_from_CW11_l2"
            ],
            "targeted_pcgrad_direction_float64_le_sha256": cw19.float64_sha(
                direction_flat, np
            ),
            "selection_sha256": SELECTION_PAYLOAD_SHA256,
            "cache_sha256": cache_audit["cache_sha256"],
        }
        modules["cutting"].restore_raw_actor(
            modules["ram"], parameter_sequence, context["raw_actor"], torch
        )
        raw_restored_hash = helper.model_state_sha256(model.state_dict())
        decoded = cw20.decode_xz(payload["candidate_actor_float32_le"])
        cw20.copy_actor_bytes(
            parameters, EXPECTED_ACTOR_NAMES, layout, decoded, np, torch
        )
        reconstructed_hash = helper.model_state_sha256(model.state_dict())
        reconstruction_checks = {
            "raw_U468_restore_exact": raw_restored_hash == RAW_MODEL_SHA256,
            "candidate_hash_exact": reconstructed_hash == selected_hash,
            "candidate_actor_bytes_exact": cw20.actor_bytes(
                parameters, EXPECTED_ACTOR_NAMES, np
            )
            == selected_actor_bytes,
            "nonactor_exact_raw": cw20.nonactor_sha(
                model, EXPECTED_ACTOR_NAMES, helper
            )
            == RAW_NONACTOR_SHA256,
        }
        cw20.restore_actor(parameters, EXPECTED_ACTOR_NAMES, cw11_actor, torch)
        reconstruction_pass = all(reconstruction_checks.values())
        final_checks = {
            "exact_CW11_context": all(context_checks.values()),
            "dependency_sources": all(dependency_checks.values()),
            "final_B256_baseline": all(baseline_checks.values()),
            "PCGrad_first_order_direction": pcgrad[
                "all_tasks_robust_first_order_descent"
            ],
            "selected_trial_integrity_and_train_gate": selected_trial["pass"]
            and selected_trial["train_gate"]["pass"]
            and all(selected_trial["integrity"].values()),
            "absolute_payload_reconstruction": reconstruction_pass,
        }
        passed = all(final_checks.values())
        return {
            "decision": (
                "GO_CW22_TARGETED_B256_PREFLIGHT"
                if passed
                else "NO_GO_CW22_TARGETED_B256_PREFLIGHT"
            ),
            "reason": (
                "SMALLEST_PREREGISTERED_RADIUS_PASSED_ALL_TARGETED_TRAIN_GATES"
                if passed
                else "ABSOLUTE_PAYLOAD_RECONSTRUCTION_FAILED"
            ),
            "context_checks": context_checks,
            "dependency_checks": dependency_checks,
            "cache": cache_audit,
            "baseline": baseline_report,
            "baseline_checks": baseline_checks,
            "task_losses": losses,
            "gradient_audit": gradient_audit,
            "pcgrad": pcgrad,
            "radii_contract": list(RADII),
            "trials": trials,
            "selected_trial": selected_trial,
            "final_checks": final_checks,
            "pure_payload_reconstruction": {
                "checks": reconstruction_checks,
                "pass": reconstruction_pass,
                "raw_model_state_sha256": raw_restored_hash,
                "candidate_model_state_sha256": reconstructed_hash,
            },
            "changed_candidate_train_shadow_count": len(trials),
            "candidate_payload": payload if passed else None,
        }
    finally:
        cw20.restore_actor(parameters, EXPECTED_ACTOR_NAMES, cw11_actor, torch)
        for name, parameter in named.items():
            parameter.requires_grad_(requires_grad_before[name])
            parameter.grad = None


def callback_with_restore(
    context: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cw20: ModuleType,
    cw19: ModuleType,
    cw15: ModuleType,
    modules: Mapping[str, ModuleType],
) -> dict[str, Any]:
    import numpy as np
    import torch

    model = context["model"]
    helper = context["helper"]
    named = dict(model.named_parameters())
    state_before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    flags_before = {name: bool(parameter.requires_grad) for name, parameter in named.items()}
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in named.items()
    }
    training_before = bool(model.training)
    precision_before = torch.get_float32_matmul_precision()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    debug_before = torch.get_deterministic_debug_mode()
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cuda_tf32_before = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32_before = torch.backends.cudnn.allow_tf32
    cpu_rng_before = torch.random.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state_all()
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    bc_max_before = helper.bc.MAX_ACTION_COUNT
    result: dict[str, Any]
    try:
        torch.set_float32_matmul_precision("high")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        result = run_targeted_core(context, rows, cw20, cw19, cw15, modules)
    finally:
        with torch.no_grad():
            current = model.state_dict()
            for name, value in state_before.items():
                if name not in current:
                    raise ProtocolError("state key removed during CW22 callback")
                current[name].copy_(value)
        for name, parameter in named.items():
            parameter.requires_grad_(flags_before[name])
            parameter.grad = (
                None
                if gradients_before[name] is None
                else gradients_before[name].to(parameter.device, parameter.dtype)
            )
        model.train(training_before)
        helper.bc.MAX_ACTION_COUNT = bc_max_before
        torch.set_float32_matmul_precision(precision_before)
        torch.use_deterministic_algorithms(deterministic_before, warn_only=warn_only_before)
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32_before
        torch.backends.cudnn.allow_tf32 = cudnn_tf32_before
        torch.random.set_rng_state(cpu_rng_before)
        torch.cuda.set_rng_state_all(cuda_rng_before)
        random.setstate(python_rng_before)
        np.random.set_state(numpy_rng_before)
    restore_checks = {
        "model_hash_exact_CW11": helper.model_state_sha256(model.state_dict())
        == CW11_MODEL_SHA256,
        "state_keys_exact": set(model.state_dict()) == set(state_before),
        "training_exact": bool(model.training) == training_before,
        "requires_grad_exact": all(
            bool(parameter.requires_grad) == flags_before[name]
            for name, parameter in named.items()
        ),
        "gradient_presence_exact": all(
            (parameter.grad is None) == (gradients_before[name] is None)
            for name, parameter in named.items()
        ),
        "precision_exact": torch.get_float32_matmul_precision() == precision_before,
        "deterministic_exact": torch.are_deterministic_algorithms_enabled()
        == deterministic_before,
        "warn_only_exact": torch.is_deterministic_algorithms_warn_only_enabled()
        == warn_only_before,
        "debug_mode_exact": torch.get_deterministic_debug_mode() == debug_before,
        "cudnn_benchmark_exact": torch.backends.cudnn.benchmark
        == cudnn_benchmark_before,
        "cudnn_deterministic_exact": torch.backends.cudnn.deterministic
        == cudnn_deterministic_before,
        "cuda_tf32_exact": torch.backends.cuda.matmul.allow_tf32 == cuda_tf32_before,
        "cudnn_tf32_exact": torch.backends.cudnn.allow_tf32 == cudnn_tf32_before,
        "BC_MAX_ACTION_COUNT_exact": helper.bc.MAX_ACTION_COUNT == bc_max_before,
        "cpu_rng_exact": torch.equal(torch.random.get_rng_state(), cpu_rng_before),
        "cuda_rng_exact": all(
            torch.equal(left, right)
            for left, right in zip(torch.cuda.get_rng_state_all(), cuda_rng_before)
        ),
        "python_rng_exact": random.getstate() == python_rng_before,
        "numpy_rng_exact": all(
            np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
            for left, right in zip(np.random.get_state(), numpy_rng_before)
        ),
    }
    if not all(restore_checks.values()):
        raise ProtocolError(f"CW22 outer restore failed: {restore_checks}")
    result["outer_restore_checks"] = restore_checks
    result["outer_restore_pass"] = True
    return result


def archive_evidence() -> dict[str, Any]:
    return {
        source: regular_evidence(path, DATA_SHA256[source], DATA_MODE[source], source)
        for source, path in DATASETS.items()
    }


def production_run() -> dict[str, Any]:
    import numpy as np
    import torch

    runtime = validate_runtime(require_cuda=True)
    source = source_audit()
    selector_evidence = regular_evidence(
        SELECTOR, SELECTOR_SHA256, SELECTOR_MODE, "selector"
    )
    selection_evidence = regular_evidence(
        SELECTION, SELECTION_SHA256, SELECTION_MODE, "selection"
    )
    profile_evidence = regular_evidence(PROFILE, PROFILE_SHA256, PROFILE_MODE, "profile")
    selection_payload = load_json_strict(SELECTION, "CW22 selection")
    rows, selection_checks = validate_selection(selection_payload)
    archives_before = archive_evidence()

    cw21, cw21_evidence = load_locked_source(
        CW21, CW21_SHA256, CW21_MODE, "cw22_targeted_frozen_cw21"
    )
    cw20, cw20_evidence = cw21.import_cw20()
    cw20_runtime = cw21.runtime_audit(cw20, require_cuda=True)
    cw19, cw19_evidence = cw20.import_cw19()
    cw15, cw15_evidence = cw19.import_locked(
        cw19.CW15, "cw22_targeted_frozen_cw15"
    )
    modules = cw15.frozen_modules()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    primary_source, primary_evidence = modules["cw11"].read_regular_bytes(
        cw15.PRIMARY,
        cw15.MODULE_SHAS[cw15.PRIMARY],
        "CW22 targeted frozen primary",
        expected_mode=0o555,
    )
    holder: dict[str, Any] = {}

    def consume(context: Mapping[str, Any]) -> None:
        if holder:
            raise ProtocolError("CW22 callback called more than once")
        holder["endpoint"] = callback_with_restore(
            context, rows, cw20, cw19, cw15, modules
        )

    historical = modules["cw11"].run_probe(
        modules["primary"],
        primary_source,
        primary_evidence,
        candidate_consumer=consume,
    )
    historical_checks = {
        "callback_once": "endpoint" in holder,
        "status_exact_CW11": historical.get("status")
        == "exploratory_33row_specialist_valid_CW11_success",
        "consumer_called": historical.get("second_stage", {})
        .get("decision", {})
        .get("candidate_consumer_called")
        is True,
        "model_exact_CW11": historical.get("second_stage", {})
        .get("decision", {})
        .get("candidate_model_state_sha256_before_CW10_finally_restore")
        == CW11_MODEL_SHA256,
        "vector_exact_CW11": historical.get("second_stage", {})
        .get("decision", {})
        .get("terminal_cumulative_float64_le_sha256")
        == CW11_VECTOR_SHA256,
        "ledger_exact_CW11": historical.get("second_stage", {})
        .get("active_pair_contract", {})
        .get("final_canonical_ledger_sha256")
        == CW11_LEDGER_SHA256,
        "historical_restore_pass": historical.get("final_integrity", {}).get("pass")
        is True,
    }
    if not all(historical_checks.values()):
        raise ProtocolError(f"historical replay drift: {historical_checks}")
    archives_after = archive_evidence()
    archive_stability = {
        source_name: {
            "sha_exact": archives_before[source_name]["sha256"]
            == archives_after[source_name]["sha256"]
            == DATA_SHA256[source_name],
            "identity_exact": (
                archives_before[source_name]["device"],
                archives_before[source_name]["inode"],
                archives_before[source_name]["bytes"],
            )
            == (
                archives_after[source_name]["device"],
                archives_after[source_name]["inode"],
                archives_after[source_name]["bytes"],
            ),
        }
        for source_name in DATASETS
    }
    if not all(all(checks.values()) for checks in archive_stability.values()):
        raise ProtocolError(f"archive changed during CW22: {archive_stability}")
    endpoint = holder["endpoint"]
    result = {
        "schema_version": SCHEMA,
        "status": endpoint["decision"],
        "decision": endpoint["decision"],
        "seed": SEED,
        "endpoint": endpoint,
        "base": {
            "kind": "general_BC_plus_PPO_plus_exact_CW11_then_targeted_special_BC",
            "model_state_sha256": CW11_MODEL_SHA256,
            "vector_sha256": CW11_VECTOR_SHA256,
            "ledger_sha256": CW11_LEDGER_SHA256,
            "raw_nonactor_sha256": RAW_NONACTOR_SHA256,
            "materialized_eval_only_CW11_opened": False,
            "resume_forbidden_respected": True,
        },
        "selection": {
            "sha256": SELECTION_PAYLOAD_SHA256,
            "rows": len(rows),
            "checks": selection_checks,
            "scope": selection_payload["scope"],
            "training_weight_contract": {
                "source_archive_weights_replaced": True,
                "non_context34_equal_row_weight": 1.0,
                "context34_ordered_row_weight": CONTEXT34_SAMPLE_WEIGHT,
            },
            "optimization_contract": {
                "hard_gradient_surrogate": (
                    "softplus of expert minus frozen exact-CW11 wrong threat"
                ),
                "hard_observed_gate": "full ordered PL NLL and deterministic action",
                "retention_gradient_and_gate": "sample-weighted ordered PL NLL",
            },
        },
        "historical_exact_CW11_replay": {
            "checks": historical_checks,
            "pass": True,
            "historical_specialist_valid_rows_replayed": 4,
            "validation_replayed_for_historical_reconstruction": True,
            "new_validation_rows_opened_for_CW22_selection_or_candidate": 0,
            "promotion_evidence": False,
        },
        "inputs": {
            "selector": selector_evidence,
            "selection": selection_evidence,
            "profile": profile_evidence,
            "archives_before": archives_before,
            "archive_stability": archive_stability,
            "CW21": cw21_evidence,
            "CW20": cw20_evidence,
            "CW19": cw19_evidence,
            "CW15": cw15_evidence,
            "primary": primary_evidence,
        },
        "audit": {
            "source": source,
            "runtime": runtime,
            "CW20_runtime": cw20_runtime,
        },
        "official_unique_changed_candidate_count_consumed": 0,
        "cumulative_official_unique_changed_candidate_count": 1,
        "submission_performed": False,
        "package_upload_performed": False,
    }
    result["audit"]["finite_nested_and_json_serializable"] = True
    canonical_json(result)
    return result


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise ProtocolError("short CW22 result write")
            written += count
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    observed = path.lstat()
    digest = sha256_file(path)
    checks = {
        "regular_single_link": stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and int(observed.st_nlink) == 1,
        "mode_0444": stat.S_IMODE(observed.st_mode) == 0o444,
        "size_exact": int(observed.st_size) == len(payload),
        "sha_exact": digest == hashlib.sha256(payload).hexdigest(),
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW22 result publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "nlink": int(observed.st_nlink),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "static":
        if args.device != "cpu" or args.output is not None:
            raise ProtocolError("static requires CPU and forbids output")
        static_selection = load_json_strict(SELECTION, "CW22 static selection")
        static_rows, static_selection_checks = validate_selection(static_selection)
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_passed",
                    "source": source_audit(),
                    "runtime": validate_runtime(require_cuda=False),
                    "CW21": regular_evidence(CW21, CW21_SHA256, CW21_MODE, "CW21"),
                    "selector": regular_evidence(
                        SELECTOR, SELECTOR_SHA256, SELECTOR_MODE, "selector"
                    ),
                    "selection": regular_evidence(
                        SELECTION, SELECTION_SHA256, SELECTION_MODE, "selection"
                    ),
                    "selection_contract": {
                        "rows": len(static_rows),
                        "checks": static_selection_checks,
                        "pass": all(static_selection_checks.values()),
                    },
                    "profile": regular_evidence(
                        PROFILE, PROFILE_SHA256, PROFILE_MODE, "profile"
                    ),
                    "writes": 0,
                    "validation_or_test_rows_opened": 0,
                    "submission_performed": False,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if args.device != "cuda" or args.output is None:
        raise ProtocolError("run requires CUDA and output")
    output = args.output.resolve()
    if output.parent != ROOT / "artifacts":
        raise ProtocolError("output must be a direct child of artifacts")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    result = production_run()
    artifact = publish_o_excl(output, canonical_json(result))
    endpoint = result["endpoint"]
    print(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "reason": endpoint["reason"],
                "output": artifact,
                "cache_sha256": endpoint["cache"]["cache_sha256"],
                "selected_radius": (
                    None
                    if endpoint.get("selected_trial") is None
                    else endpoint["selected_trial"]["radius"]
                ),
                "candidate_model_state_sha256": (
                    None
                    if endpoint.get("candidate_payload") is None
                    else endpoint["candidate_payload"]["anchor"][
                        "terminal_model_state_sha256"
                    ]
                ),
                "new_validation_rows_opened": 0,
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

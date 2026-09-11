#!/usr/bin/env python3
"""Train replay policy clones and publish a frozen PPO opponent league locally.

This runner intentionally does not call Kaggle, upload artifacts, or submit a
model.  It consumes the manifest produced by the gold replay pipeline, trains
one behavior-cloning model per policy, evaluates each clone on episode-disjoint
holdout rows, and writes a local league manifest for ``train_ppo.py``.

The source archive's holdout split is never passed to ``train_bc_orbit.py``.
Instead, this runner deterministically partitions only the source train
episodes into an internal train/dev archive.  The source holdout is then used
once by ``evaluate_policy_bc.py`` for the frozen clone-quality report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train_bc_orbit.py"
EVAL_SCRIPT = REPO_ROOT / "tools" / "evaluate_policy_bc.py"
INPUT_SCHEMA = "ptcg-gold-policy-clones-v1"
OUTPUT_SCHEMA = "ptcg-gold-clone-run-v1"
LEAGUE_SCHEMA = "ptcg-ppo-opponent-league-v1"
DERIVED_SCHEMA = "ptcg-clone-train-dev-v2"
ROW_SPLIT_CONTRACT = "member_prefix_equals_row_split-v1"
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    submission_id: int
    team_name: str
    deck_hash: str
    archetype: str
    train_archive: Path
    train_split: str
    holdout_archive: Path
    holdout_split: str
    train_archive_sha256: str | None
    holdout_archive_sha256: str | None
    deck_path: Path | None
    declared_train_episodes: int | None
    declared_train_decisions: int | None
    declared_holdout_episodes: int | None
    declared_holdout_decisions: int | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class SplitInventory:
    episode_ids: frozenset[str]
    decisions: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def optional_sha256(value: Any, label: str) -> str | None:
    if value is None or value == "":
        return None
    parsed = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", parsed):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return parsed


def resolve_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def parse_count(
    nested: dict[str, Any],
    policy: dict[str, Any],
    nested_keys: tuple[str, ...],
    policy_keys: tuple[str, ...],
    label: str,
) -> int | None:
    value = first_present(nested, nested_keys)
    if value is None:
        value = first_present(policy, policy_keys)
    return optional_nonnegative_int(value, label)


def parse_policy(
    raw: dict[str, Any],
    manifest_base: Path,
    index: int,
) -> PolicySpec:
    policy_id = raw.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError(f"policies[{index}].policy_id must be non-empty")
    submission_id = optional_nonnegative_int(
        raw.get("submission_id"),
        f"policies[{index}].submission_id",
    )
    if submission_id is None:
        raise ValueError(f"policies[{index}].submission_id is required")
    team_name = str(raw.get("team_name", ""))
    deck_hash = str(raw.get("deck_hash", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", deck_hash):
        raise ValueError(
            f"policies[{index}].deck_hash must be a lowercase SHA-256"
        )
    archetype = str(raw.get("archetype", "unknown"))

    train = raw.get("train") if isinstance(raw.get("train"), dict) else {}
    holdout = (
        raw.get("holdout") if isinstance(raw.get("holdout"), dict) else {}
    )
    common_archive = first_present(raw, ("archive_path", "archive"))
    train_archive_value = first_present(
        train,
        ("archive_path", "archive", "path"),
    )
    if train_archive_value is None:
        train_archive_value = first_present(
            raw,
            ("train_archive_path", "train_archive"),
        )
    if train_archive_value is None:
        train_archive_value = common_archive
    holdout_archive_value = first_present(
        holdout,
        ("archive_path", "archive", "path"),
    )
    if holdout_archive_value is None:
        holdout_archive_value = first_present(
            raw,
            ("holdout_archive_path", "holdout_archive"),
        )
    if holdout_archive_value is None:
        holdout_archive_value = common_archive or train_archive_value

    train_archive = resolve_path(
        train_archive_value,
        manifest_base,
        f"policies[{index}].train archive",
    )
    holdout_archive = resolve_path(
        holdout_archive_value,
        manifest_base,
        f"policies[{index}].holdout archive",
    )
    train_split = str(
        first_present(train, ("split",))
        or raw.get("train_split")
        or "train"
    )
    holdout_split = str(
        first_present(holdout, ("split",))
        or raw.get("holdout_split")
        or "valid"
    )
    if train_split not in SPLITS:
        raise ValueError(f"Unsupported train split {train_split!r}")
    if holdout_split not in SPLITS:
        raise ValueError(f"Unsupported holdout split {holdout_split!r}")
    common_archive_sha = raw.get("archive_sha256")
    train_archive_sha = first_present(
        train,
        ("archive_sha256", "sha256"),
    )
    holdout_archive_sha = first_present(
        holdout,
        ("archive_sha256", "sha256"),
    )

    deck_path_value = first_present(
        raw,
        ("resolved_deck_path", "deck_path", "deck_csv_path"),
    )
    deck_path = (
        resolve_path(
            deck_path_value,
            manifest_base,
            f"policies[{index}].deck_path",
        )
        if deck_path_value is not None
        else None
    )
    return PolicySpec(
        policy_id=policy_id,
        submission_id=submission_id,
        team_name=team_name,
        deck_hash=deck_hash,
        archetype=archetype,
        train_archive=train_archive,
        train_split=train_split,
        holdout_archive=holdout_archive,
        holdout_split=holdout_split,
        train_archive_sha256=optional_sha256(
            train_archive_sha or common_archive_sha,
            f"policies[{index}].train archive sha256",
        ),
        holdout_archive_sha256=optional_sha256(
            holdout_archive_sha or common_archive_sha,
            f"policies[{index}].holdout archive sha256",
        ),
        deck_path=deck_path,
        declared_train_episodes=parse_count(
            train,
            raw,
            ("episodes", "episode_count"),
            ("train_episodes",),
            f"policies[{index}].train episodes",
        ),
        declared_train_decisions=parse_count(
            train,
            raw,
            ("decisions", "rows", "decision_count"),
            ("train_decisions",),
            f"policies[{index}].train decisions",
        ),
        declared_holdout_episodes=parse_count(
            holdout,
            raw,
            ("episodes", "episode_count"),
            ("holdout_episodes",),
            f"policies[{index}].holdout episodes",
        ),
        declared_holdout_decisions=parse_count(
            holdout,
            raw,
            ("decisions", "rows", "decision_count"),
            ("holdout_decisions",),
            f"policies[{index}].holdout decisions",
        ),
        raw=raw,
    )


def load_pipeline_manifest(path: Path) -> tuple[dict[str, Any], list[PolicySpec]]:
    manifest = read_json(path)
    schema = manifest.get("schema_version") or manifest.get("schema")
    if schema != INPUT_SCHEMA:
        raise ValueError(
            f"{path}: unsupported schema {schema!r}; expected {INPUT_SCHEMA!r}"
        )
    raw_policies = manifest.get("policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError(f"{path}: policies must be a non-empty list")
    policies: list[PolicySpec] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_policies):
        if not isinstance(raw, dict):
            raise ValueError(f"policies[{index}] must be an object")
        policy = parse_policy(raw, path.parent.resolve(), index)
        if policy.policy_id in seen_ids:
            raise ValueError(f"Duplicate policy_id {policy.policy_id!r}")
        seen_ids.add(policy.policy_id)
        policies.append(policy)
    return manifest, policies


def safe_policy_name(policy_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", policy_id).strip(".-_")
    slug = slug[:64] or "policy"
    suffix = hashlib.sha256(policy_id.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{suffix}"


def archive_jsonl_members(
    archive: zipfile.ZipFile,
    split: str,
) -> list[str]:
    prefix = f"{split}/"
    return sorted(
        name
        for name in archive.namelist()
        if name.startswith(prefix) and name.endswith(".jsonl")
    )


def iter_split_lines(
    path: Path,
    split: str,
) -> Iterator[tuple[bytes, dict[str, Any]]]:
    with zipfile.ZipFile(path) as archive:
        members = archive_jsonl_members(archive, split)
        if not members:
            raise ValueError(f"{path}: no JSONL members for split {split!r}")
        for member in members:
            with archive.open(member) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{path}:{member}:{line_number}: invalid JSON"
                        ) from error
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"{path}:{member}:{line_number}: row is not an object"
                        )
                    yield line, row


def scan_split(path: Path, split: str) -> SplitInventory:
    episodes: set[str] = set()
    decisions = 0
    for _, row in iter_split_lines(path, split):
        episode_id = row.get("episode_id")
        if episode_id in {None, ""}:
            raise ValueError(f"{path}:{split}: row missing episode_id")
        episodes.add(str(episode_id))
        decisions += 1
    return SplitInventory(frozenset(episodes), decisions)


def validate_declared_count(
    actual: int,
    declared: int | None,
    label: str,
) -> None:
    if declared is not None and declared != actual:
        raise ValueError(
            f"{label} manifest count mismatch: declared={declared}, actual={actual}"
        )


def inventory_policy(policy: PolicySpec) -> tuple[SplitInventory, SplitInventory]:
    if not policy.train_archive.is_file():
        raise FileNotFoundError(policy.train_archive)
    if not policy.holdout_archive.is_file():
        raise FileNotFoundError(policy.holdout_archive)
    train = scan_split(policy.train_archive, policy.train_split)
    holdout = scan_split(policy.holdout_archive, policy.holdout_split)
    validate_declared_count(
        len(train.episode_ids),
        policy.declared_train_episodes,
        f"{policy.policy_id} train episodes",
    )
    validate_declared_count(
        train.decisions,
        policy.declared_train_decisions,
        f"{policy.policy_id} train decisions",
    )
    validate_declared_count(
        len(holdout.episode_ids),
        policy.declared_holdout_episodes,
        f"{policy.policy_id} holdout episodes",
    )
    validate_declared_count(
        holdout.decisions,
        policy.declared_holdout_decisions,
        f"{policy.policy_id} holdout decisions",
    )
    overlap = train.episode_ids & holdout.episode_ids
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            f"{policy.policy_id}: train/holdout episode leakage "
            f"({len(overlap)} overlapping IDs; examples={examples})"
        )
    if len(train.episode_ids) < 2:
        raise ValueError(
            f"{policy.policy_id}: at least two train episodes are required "
            "to derive disjoint train/dev splits"
        )
    if not holdout.episode_ids or not holdout.decisions:
        raise ValueError(f"{policy.policy_id}: holdout split is empty")
    return train, holdout


def choose_internal_dev_episodes(
    episode_ids: Iterable[str],
    *,
    policy_id: str,
    seed: int,
    fraction: float,
    minimum: int,
) -> frozenset[str]:
    episodes = list(set(episode_ids))
    if len(episodes) < 2:
        raise ValueError("At least two train episodes are required")
    desired = max(minimum, int(math.ceil(len(episodes) * fraction)))
    desired = min(max(desired, 1), len(episodes) - 1)

    def rank(episode_id: str) -> bytes:
        payload = (
            f"ptcg-clone-dev-v1:{seed}:{policy_id}:{episode_id}".encode("utf-8")
        )
        return hashlib.sha256(payload).digest()

    return frozenset(sorted(episodes, key=rank)[:desired])


def source_manifest(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist().count("manifest.json") != 1:
            raise ValueError(f"{path}: expected exactly one manifest.json")
        value = json.loads(archive.read("manifest.json"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest.json is not an object")
    return value


def derived_signature(
    policy: PolicySpec,
    *,
    source_sha256: str,
    seed: int,
    dev_fraction: float,
    min_dev_episodes: int,
) -> dict[str, Any]:
    return {
        "schema_version": DERIVED_SCHEMA,
        "policy_id": policy.policy_id,
        "submission_id": policy.submission_id,
        "deck_hash": policy.deck_hash,
        "source_archive": str(policy.train_archive),
        "source_archive_sha256": source_sha256,
        "source_split": policy.train_split,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "min_dev_episodes": min_dev_episodes,
        "row_split_contract": ROW_SPLIT_CONTRACT,
    }


def read_archive_manifest(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        value = json.loads(archive.read("manifest.json"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest.json must be an object")
    return value


def validate_resumable_derived_archive(
    path: Path,
    expected_signature: dict[str, Any],
) -> dict[str, Any]:
    """Validate both the manifest and every row before reusing an archive."""
    manifest = read_archive_manifest(path)
    if manifest.get("schema_version") != DERIVED_SCHEMA:
        raise ValueError(
            f"{path}: derived schema is stale; expected {DERIVED_SCHEMA!r}"
        )
    if manifest.get("clone_runner_signature") != expected_signature:
        raise ValueError(f"{path}: clone runner signature mismatch")
    split_policy = manifest.get("split_policy")
    if (
        not isinstance(split_policy, dict)
        or split_policy.get("row_split_contract") != ROW_SPLIT_CONTRACT
    ):
        raise ValueError(f"{path}: missing row split rewrite contract")

    actual_counts = {split: 0 for split in SPLITS}
    episodes = {split: set() for split in SPLITS}
    with zipfile.ZipFile(path) as archive:
        for split in ("train", "valid"):
            members = archive_jsonl_members(archive, split)
            if not members:
                raise ValueError(f"{path}: missing {split} JSONL members")
            for member in members:
                with archive.open(member) as handle:
                    for line_number, line in enumerate(handle, start=1):
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(
                                f"{path}:{member}:{line_number}: row is not "
                                "an object"
                            )
                        if row.get("split") != split:
                            raise ValueError(
                                f"{path}:{member}:{line_number}: row split "
                                f"{row.get('split')!r} does not match member "
                                f"prefix {split!r}"
                            )
                        episode_id = row.get("episode_id")
                        if episode_id in {None, ""}:
                            raise ValueError(
                                f"{path}:{member}:{line_number}: missing "
                                "episode_id"
                            )
                        actual_counts[split] += 1
                        episodes[split].add(str(episode_id))
    overlap = episodes["train"] & episodes["valid"]
    if overlap:
        raise ValueError(
            f"{path}: derived train/dev episode leakage "
            f"({len(overlap)} overlapping IDs)"
        )
    declared_counts = manifest.get("split_decisions")
    if not isinstance(declared_counts, dict):
        raise ValueError(f"{path}: missing split_decisions")
    for split in SPLITS:
        if int(declared_counts.get(split, -1)) != actual_counts[split]:
            raise ValueError(
                f"{path}: split_decisions.{split} does not match rows"
            )
    return manifest


def build_derived_train_archive(
    policy: PolicySpec,
    destination: Path,
    train_inventory: SplitInventory,
    *,
    seed: int,
    dev_fraction: float,
    min_dev_episodes: int,
    source_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    signature = derived_signature(
        policy,
        source_sha256=source_sha256,
        seed=seed,
        dev_fraction=dev_fraction,
        min_dev_episodes=min_dev_episodes,
    )
    if destination.is_file():
        if resume:
            return validate_resumable_derived_archive(
                destination,
                signature,
            )
        raise FileExistsError(
            f"{destination} already exists with a different or non-resumable "
            "derivation; choose a new output directory or use --force"
        )

    dev_episodes = choose_internal_dev_episodes(
        train_inventory.episode_ids,
        policy_id=policy.policy_id,
        seed=seed,
        fraction=dev_fraction,
        minimum=min_dev_episodes,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {split: 0 for split in SPLITS}
    episode_counts = {split: set() for split in SPLITS}
    original_manifest = source_manifest(policy.train_archive)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as target:
            for output_split, include_dev in (
                ("train", False),
                ("valid", True),
            ):
                member = f"{output_split}/part-00000.jsonl"
                with target.open(member, "w", force_zip64=True) as output:
                    for line, row in iter_split_lines(
                        policy.train_archive,
                        policy.train_split,
                    ):
                        episode_id = str(row["episode_id"])
                        if (episode_id in dev_episodes) != include_dev:
                            continue
                        row["split"] = output_split
                        output.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                            + b"\n"
                        )
                        counts[output_split] += 1
                        episode_counts[output_split].add(episode_id)
            manifest = {
                **original_manifest,
                "schema_version": DERIVED_SCHEMA,
                "clone_runner_signature": signature,
                "split_policy": {
                    "mode": "episode_disjoint_internal_dev",
                    "seed": seed,
                    "dev_fraction": dev_fraction,
                    "min_dev_episodes": min_dev_episodes,
                    "frozen_holdout_excluded": True,
                    "row_split_contract": ROW_SPLIT_CONTRACT,
                },
                "split_decisions": counts,
                "split_episodes": {
                    split: len(episode_counts[split])
                    for split in SPLITS
                },
                "source_train_inventory": {
                    "episodes": len(train_inventory.episode_ids),
                    "decisions": train_inventory.decisions,
                },
            }
            target.writestr(
                "manifest.json",
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8"),
            )
        if counts["train"] + counts["valid"] != train_inventory.decisions:
            raise RuntimeError("Derived train/dev rows do not preserve source rows")
        if not counts["train"] or not counts["valid"]:
            raise RuntimeError("Derived train or dev split is empty")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def compute_deck_hash(path: Path) -> str:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.append(int(line.strip()))
    if len(values) != 60:
        raise ValueError(f"{path}: contains {len(values)} cards; expected 60")
    canonical = ",".join(str(value) for value in sorted(values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def append_option(command: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def build_train_command(
    args: argparse.Namespace,
    policy: PolicySpec,
    derived_archive: Path,
    train_output: Path,
    expected_train_rows: int,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data",
        str(derived_archive),
        "--output-dir",
        str(train_output),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--categorical-dim",
        str(args.categorical_dim),
        "--model-dim",
        str(args.model_dim),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--dropout",
        str(args.dropout),
        "--seed",
        str(args.seed),
        "--expected-train-rows",
        str(expected_train_rows),
        "--target-accuracy",
        str(args.train_target_accuracy),
        "--split-mode",
        "archive",
        "--skip-test",
        "--deck-hash",
        policy.deck_hash,
        "--device",
        args.device,
    ]
    if policy.team_name:
        command.extend(("--team-name", policy.team_name))
    append_option(command, "--max-train-rows", args.max_train_rows)
    append_option(command, "--max-valid-rows", args.max_valid_rows)
    if args.base_checkpoint is not None:
        command.extend(("--init-checkpoint", str(args.base_checkpoint)))
    if args.use_trajectory_weights:
        command.append("--use-trajectory-weights")
    return command


def build_eval_command(
    args: argparse.Namespace,
    policy: PolicySpec,
    checkpoint: Path,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--data",
        str(policy.holdout_archive),
        "--split",
        policy.holdout_split,
        "--split-mode",
        "archive",
        "--batch-size",
        str(args.eval_batch_size),
        "--workers",
        str(args.eval_workers),
        "--prediction-order",
        "auto",
        "--device",
        args.device,
        "--progress-interval",
        str(args.eval_progress_interval),
        "--json-output",
        str(output),
    ]
    if policy.team_name:
        command.extend(("--team-name", policy.team_name))
    command.extend(("--deck-hash", policy.deck_hash))
    append_option(command, "--max-rows", args.max_holdout_rows)
    return command


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_logged_command(
    command: list[str],
    log_path: Path,
    *,
    timeout: float | None,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("ab") as log:
        header = (
            f"\n[{utc_now()}] command="
            + json.dumps(command, ensure_ascii=False)
            + "\n"
        )
        log.write(header.encode("utf-8"))
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process(process)
            return_code = 124
            log.write(f"[{utc_now()}] timed out\n".encode("utf-8"))
    return return_code, time.monotonic() - started


def valid_resumable_evaluation(
    path: Path,
    *,
    checkpoint: Path,
    holdout_archive: Path,
    holdout_split: str,
    signature_path: Path,
    expected_signature: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file() or not checkpoint.is_file():
        return None
    if not signature_path.is_file():
        return None
    try:
        result = read_json(path)
        saved_signature = read_json(signature_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if saved_signature != expected_signature:
        return None
    if Path(str(result.get("checkpoint", ""))) != checkpoint.resolve():
        return None
    if Path(str(result.get("data", ""))) != holdout_archive.resolve():
        return None
    if result.get("split") != holdout_split:
        return None
    if result.get("checkpoint_sha256") != file_sha256(checkpoint):
        return None
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not metrics.get("rows"):
        return None
    return result


def quality_from_evaluation(
    result: dict[str, Any],
    *,
    min_rows: int,
    min_set: float,
    min_ordered: float,
    min_hybrid: float,
) -> dict[str, Any]:
    metrics = result["metrics"]
    rows = int(metrics.get("rows", 0) or 0)
    set_accuracy = float(metrics.get("set_exact_accuracy", 0.0) or 0.0)
    ordered_accuracy = float(
        metrics.get("ordered_exact_accuracy", 0.0) or 0.0
    )
    hybrid_accuracy = float(
        metrics.get("hybrid_order_exact_accuracy", 0.0) or 0.0
    )
    checks = {
        "minimum_holdout_rows": rows >= min_rows,
        "minimum_set_exact_accuracy": set_accuracy >= min_set,
        "minimum_ordered_exact_accuracy": ordered_accuracy >= min_ordered,
        "minimum_hybrid_order_exact_accuracy": hybrid_accuracy >= min_hybrid,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "rows": rows,
        "set_exact_accuracy": set_accuracy,
        "ordered_exact_accuracy": ordered_accuracy,
        "hybrid_order_exact_accuracy": hybrid_accuracy,
        "thresholds": {
            "minimum_holdout_rows": min_rows,
            "minimum_set_exact_accuracy": min_set,
            "minimum_ordered_exact_accuracy": min_ordered,
            "minimum_hybrid_order_exact_accuracy": min_hybrid,
        },
    }


def materialize_unique_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_sha256(destination) == file_sha256(source):
        return
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    try:
        # Use a real copy rather than a hard link. A future resume can overwrite
        # ``best.pt`` in place; a hard link would silently mutate an already
        # published frozen league checkpoint.
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def policy_public_metadata(policy: PolicySpec) -> dict[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "submission_id": policy.submission_id,
        "team_name": policy.team_name,
        "deck_hash": policy.deck_hash,
        "archetype": policy.archetype,
    }


def remaining_timeout(
    started: float,
    max_wall_seconds: float | None,
    per_policy_seconds: float | None,
) -> float | None:
    values = [
        value
        for value in (
            per_policy_seconds,
            (
                max_wall_seconds - (time.monotonic() - started)
                if max_wall_seconds is not None
                else None
            ),
        )
        if value is not None
    ]
    return max(min(values), 0.0) if values else None


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "min_train_episodes",
        "min_train_rows",
        "min_holdout_episodes",
        "min_holdout_rows",
        "min_dev_episodes",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("workers", "eval_workers", "eval_progress_interval"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in (
        "max_policies",
        "max_train_rows",
        "max_valid_rows",
        "max_holdout_rows",
        "max_total_train_rows",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_wall_seconds", "timeout_per_policy_seconds"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.internal_dev_fraction < 1.0:
        raise ValueError("--internal-dev-fraction must be in (0, 1)")
    for name in (
        "min_set_exact",
        "min_ordered_exact",
        "min_hybrid_order_exact",
        "train_target_accuracy",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.model_dim % args.heads:
        raise ValueError("--model-dim must be divisible by --heads")
    if args.base_checkpoint is not None and not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    if not TRAIN_SCRIPT.is_file() or not EVAL_SCRIPT.is_file():
        raise FileNotFoundError("Training or evaluation helper is missing")
    if args.resume and args.force:
        raise ValueError("--resume and --force are mutually exclusive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train episode-disjoint gold policy clones and build a local PPO "
            "opponent league. This command never uploads or submits."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-id", action="append", default=[])
    parser.add_argument("--max-policies", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--categorical-dim", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--train-target-accuracy", type=float, default=0.70)
    parser.add_argument("--use-trajectory-weights", action="store_true")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--internal-dev-fraction", type=float, default=0.10)
    parser.add_argument("--min-dev-episodes", type=int, default=2)
    parser.add_argument("--min-train-episodes", type=int, default=10)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-holdout-episodes", type=int, default=3)
    parser.add_argument("--min-holdout-rows", type=int, default=300)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--max-holdout-rows", type=int)
    parser.add_argument("--max-total-train-rows", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--timeout-per-policy-seconds", type=float)

    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--eval-progress-interval", type=int, default=100)
    parser.add_argument("--min-set-exact", type=float, default=0.65)
    parser.add_argument("--min-ordered-exact", type=float, default=0.60)
    parser.add_argument("--min-hybrid-order-exact", type=float, default=0.60)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.base_checkpoint is not None:
        args.base_checkpoint = args.base_checkpoint.expanduser().resolve()
    validate_args(args)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    input_manifest, policies = load_pipeline_manifest(args.manifest)

    requested = set(args.policy_id)
    known = {policy.policy_id for policy in policies}
    missing = requested - known
    if missing:
        raise ValueError(f"Unknown --policy-id values: {sorted(missing)}")
    if requested:
        policies = [policy for policy in policies if policy.policy_id in requested]
    if args.max_policies is not None:
        policies = policies[: args.max_policies]
    if not policies:
        raise ValueError("No policies selected")

    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise NotADirectoryError(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume and not args.force and not args.dry_run:
            raise FileExistsError(
                f"{args.output_dir} is non-empty; use --resume, --force, "
                "or choose a new output directory"
            )
    if args.force and args.output_dir.exists() and not args.dry_run:
        # Never delete broad user paths. Only known per-run JSON files are
        # replaced; per-policy artifacts are rebuilt in place as encountered.
        pass

    input_manifest_sha = file_sha256(args.manifest)
    base_checkpoint_sha = (
        file_sha256(args.base_checkpoint)
        if args.base_checkpoint is not None
        else None
    )
    started = time.monotonic()
    total_train_budget = 0
    results: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []

    for policy in policies:
        safe_name = safe_policy_name(policy.policy_id)
        policy_root = args.output_dir / "policies" / safe_name
        derived_archive = policy_root / "training_input.zip"
        train_output = policy_root / "train"
        checkpoint = train_output / "best.pt"
        eval_output = policy_root / "frozen_holdout_eval.json"
        training_signature_path = policy_root / "training_signature.json"
        evaluation_signature_path = policy_root / "evaluation_signature.json"
        result: dict[str, Any] = {
            **policy_public_metadata(policy),
            "safe_name": safe_name,
            "status": "pending",
            "paths": {
                "source_train_archive": str(policy.train_archive),
                "source_holdout_archive": str(policy.holdout_archive),
                "derived_train_archive": str(derived_archive),
                "checkpoint": str(checkpoint),
                "evaluation": str(eval_output),
            },
        }
        try:
            train_inventory, holdout_inventory = inventory_policy(policy)
            result["episode_isolation"] = {
                "verified": True,
                "train_episodes": len(train_inventory.episode_ids),
                "train_decisions": train_inventory.decisions,
                "holdout_episodes": len(holdout_inventory.episode_ids),
                "holdout_decisions": holdout_inventory.decisions,
                "overlap_episodes": 0,
            }
            sufficiency = {
                "minimum_train_episodes": (
                    len(train_inventory.episode_ids) >= args.min_train_episodes
                ),
                "minimum_train_rows": (
                    train_inventory.decisions >= args.min_train_rows
                ),
                "minimum_holdout_episodes": (
                    len(holdout_inventory.episode_ids)
                    >= args.min_holdout_episodes
                ),
                "minimum_holdout_rows": (
                    holdout_inventory.decisions >= args.min_holdout_rows
                ),
            }
            result["data_sufficiency"] = sufficiency
            if not all(sufficiency.values()):
                result["status"] = "insufficient_data"
                result["quality_pass"] = False
                results.append(result)
                continue

            source_sha = file_sha256(policy.train_archive)
            holdout_sha = (
                source_sha
                if policy.holdout_archive == policy.train_archive
                else file_sha256(policy.holdout_archive)
            )
            if (
                policy.train_archive_sha256 is not None
                and source_sha != policy.train_archive_sha256
            ):
                raise ValueError(
                    f"{policy.policy_id}: train archive SHA-256 mismatch"
                )
            if (
                policy.holdout_archive_sha256 is not None
                and holdout_sha != policy.holdout_archive_sha256
            ):
                raise ValueError(
                    f"{policy.policy_id}: holdout archive SHA-256 mismatch"
                )
            internal_dev = choose_internal_dev_episodes(
                train_inventory.episode_ids,
                policy_id=policy.policy_id,
                seed=args.seed,
                fraction=args.internal_dev_fraction,
                minimum=args.min_dev_episodes,
            )
            estimated_dev_ratio = len(internal_dev) / len(
                train_inventory.episode_ids
            )
            estimated_train_rows = max(
                1,
                int(train_inventory.decisions * (1.0 - estimated_dev_ratio)),
            )
            budget_rows = min(
                estimated_train_rows,
                args.max_train_rows or estimated_train_rows,
            )
            if (
                args.max_total_train_rows is not None
                and total_train_budget + budget_rows
                > args.max_total_train_rows
            ):
                result["status"] = "skipped_total_train_budget"
                result["quality_pass"] = False
                results.append(result)
                continue
            total_train_budget += budget_rows

            train_command = build_train_command(
                args,
                policy,
                derived_archive,
                train_output,
                budget_rows,
            )
            eval_command = build_eval_command(
                args,
                policy,
                checkpoint,
                eval_output,
            )
            plan = {
                "policy_id": policy.policy_id,
                "source_train_sha256": source_sha,
                "train_command": train_command,
                "eval_command": eval_command,
                "budget_train_rows": budget_rows,
                "frozen_holdout_excluded_from_training": True,
            }
            plans.append(plan)
            if args.dry_run:
                result["status"] = "dry_run"
                result["plan"] = plan
                results.append(result)
                continue

            if args.force and derived_archive.exists():
                derived_archive.unlink()
            derived_manifest = build_derived_train_archive(
                policy,
                derived_archive,
                train_inventory,
                seed=args.seed,
                dev_fraction=args.internal_dev_fraction,
                min_dev_episodes=args.min_dev_episodes,
                source_sha256=source_sha,
                resume=args.resume,
            )
            actual_train_rows = int(
                derived_manifest["split_decisions"]["train"]
            )
            expected_train_rows = min(
                actual_train_rows,
                args.max_train_rows or actual_train_rows,
            )
            train_command = build_train_command(
                args,
                policy,
                derived_archive,
                train_output,
                expected_train_rows,
            )
            result["derived_split"] = {
                "train_episodes": derived_manifest["split_episodes"]["train"],
                "dev_episodes": derived_manifest["split_episodes"]["valid"],
                "train_decisions": actual_train_rows,
                "dev_decisions": derived_manifest["split_decisions"]["valid"],
                "source_archive_sha256": source_sha,
            }
            expected_training_signature = {
                "schema_version": "ptcg-clone-training-signature-v2",
                "policy_id": policy.policy_id,
                "input_manifest_sha256": input_manifest_sha,
                "source_train_archive_sha256": source_sha,
                "derived_train_archive_sha256": file_sha256(derived_archive),
                "base_checkpoint_sha256": base_checkpoint_sha,
                "command": train_command,
            }
            if training_signature_path.is_file():
                saved_training_signature = read_json(training_signature_path)
                if (
                    args.resume
                    and saved_training_signature
                    != expected_training_signature
                ):
                    raise ValueError(
                        f"{policy.policy_id}: resume training signature "
                        "mismatch; use --force or a new output directory"
                    )
            atomic_write_json(
                training_signature_path,
                expected_training_signature,
            )

            reuse_training = (
                args.resume
                and checkpoint.is_file()
                and (train_output / "summary.json").is_file()
            )
            if not reuse_training:
                timeout = remaining_timeout(
                    started,
                    args.max_wall_seconds,
                    args.timeout_per_policy_seconds,
                )
                if timeout is not None and timeout <= 0:
                    result["status"] = "skipped_wall_budget"
                    result["quality_pass"] = False
                    results.append(result)
                    continue
                return_code, seconds = run_logged_command(
                    train_command,
                    policy_root / "train.log",
                    timeout=timeout,
                )
                result["train_seconds"] = seconds
                result["train_return_code"] = return_code
                if return_code != 0 or not checkpoint.is_file():
                    result["status"] = (
                        "train_timeout" if return_code == 124 else "train_failed"
                    )
                    result["quality_pass"] = False
                    results.append(result)
                    if args.stop_on_error:
                        break
                    continue
            else:
                result["training_resumed"] = True

            evaluation = (
                valid_resumable_evaluation(
                    eval_output,
                    checkpoint=checkpoint,
                    holdout_archive=policy.holdout_archive,
                    holdout_split=policy.holdout_split,
                    signature_path=evaluation_signature_path,
                    expected_signature={
                        "schema_version": (
                            "ptcg-clone-evaluation-signature-v1"
                        ),
                        "policy_id": policy.policy_id,
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "holdout_archive_sha256": holdout_sha,
                        "command": build_eval_command(
                            args,
                            policy,
                            checkpoint,
                            eval_output,
                        ),
                    },
                )
                if args.resume
                else None
            )
            expected_evaluation_signature = {
                "schema_version": "ptcg-clone-evaluation-signature-v1",
                "policy_id": policy.policy_id,
                "checkpoint_sha256": file_sha256(checkpoint),
                "holdout_archive_sha256": holdout_sha,
                "command": build_eval_command(
                    args,
                    policy,
                    checkpoint,
                    eval_output,
                ),
            }
            if evaluation is None:
                timeout = remaining_timeout(
                    started,
                    args.max_wall_seconds,
                    args.timeout_per_policy_seconds,
                )
                if timeout is not None and timeout <= 0:
                    result["status"] = "skipped_wall_budget"
                    result["quality_pass"] = False
                    results.append(result)
                    continue
                return_code, seconds = run_logged_command(
                    build_eval_command(
                        args,
                        policy,
                        checkpoint,
                        eval_output,
                    ),
                    policy_root / "eval.log",
                    timeout=timeout,
                )
                result["eval_seconds"] = seconds
                result["eval_return_code"] = return_code
                if return_code != 0 or not eval_output.is_file():
                    result["status"] = (
                        "eval_timeout" if return_code == 124 else "eval_failed"
                    )
                    result["quality_pass"] = False
                    results.append(result)
                    if args.stop_on_error:
                        break
                    continue
                evaluation = read_json(eval_output)
                atomic_write_json(
                    evaluation_signature_path,
                    expected_evaluation_signature,
                )
            else:
                result["evaluation_resumed"] = True

            quality = quality_from_evaluation(
                evaluation,
                min_rows=args.min_holdout_rows,
                min_set=args.min_set_exact,
                min_ordered=args.min_ordered_exact,
                min_hybrid=args.min_hybrid_order_exact,
            )
            result["quality"] = quality
            result["quality_pass"] = quality["pass"]
            result["checkpoint_sha256"] = file_sha256(checkpoint)
            result["status"] = (
                "quality_pass" if quality["pass"] else "quality_rejected"
            )
            if policy.deck_path is None:
                result["ppo_eligible"] = False
                result["ppo_ineligible_reason"] = "missing_deck_path"
            elif not policy.deck_path.is_file():
                result["ppo_eligible"] = False
                result["ppo_ineligible_reason"] = "deck_path_not_found"
            else:
                actual_deck_hash = compute_deck_hash(policy.deck_path)
                result["deck_path"] = str(policy.deck_path)
                result["deck_file_hash_matches"] = (
                    actual_deck_hash == policy.deck_hash
                )
                result["ppo_eligible"] = bool(
                    quality["pass"] and actual_deck_hash == policy.deck_hash
                )
                if actual_deck_hash != policy.deck_hash:
                    result["ppo_ineligible_reason"] = "deck_hash_mismatch"
                elif not quality["pass"]:
                    result["ppo_ineligible_reason"] = "quality_threshold"
            results.append(result)
        except Exception as error:
            result["status"] = "error"
            result["quality_pass"] = False
            result["error"] = f"{type(error).__name__}: {error}"
            results.append(result)
            if args.stop_on_error:
                break

    if args.dry_run:
        summary = {
            "schema_version": OUTPUT_SCHEMA,
            "dry_run": True,
            "input_manifest": str(args.manifest),
            "input_manifest_sha256": input_manifest_sha,
            "selected_policies": len(policies),
            "plans": plans,
            "results": results,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    league_entries: list[dict[str, Any]] = []
    excluded_entries: list[dict[str, Any]] = []
    for result in results:
        if not result.get("ppo_eligible"):
            excluded_entries.append(
                {
                    **{
                        key: result.get(key)
                        for key in (
                            "policy_id",
                            "submission_id",
                            "team_name",
                            "deck_hash",
                            "archetype",
                            "status",
                            "ppo_ineligible_reason",
                        )
                    },
                    "quality": result.get("quality"),
                }
            )
            continue
        safe_name = str(result["safe_name"])
        source_checkpoint = Path(result["paths"]["checkpoint"])
        league_checkpoint = (
            args.output_dir / "league_checkpoints" / f"{safe_name}.pt"
        )
        materialize_unique_checkpoint(source_checkpoint, league_checkpoint)
        deck_path = Path(result["deck_path"])
        expected_name = f"{league_checkpoint.stem}@{deck_path.stem}"
        entry = {
            **{
                key: result[key]
                for key in (
                    "policy_id",
                    "submission_id",
                    "team_name",
                    "deck_hash",
                    "archetype",
                )
            },
            "name": expected_name,
            "checkpoint": str(league_checkpoint),
            "checkpoint_sha256": file_sha256(league_checkpoint),
            "deck": str(deck_path),
            "quality": result["quality"],
            "ppo_cli_args": [
                "--extra-opponent",
                str(league_checkpoint),
                str(deck_path),
            ],
        }
        league_entries.append(entry)

    league_manifest = {
        "schema_version": LEAGUE_SCHEMA,
        "generated_at": utc_now(),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": input_manifest_sha,
        "selection_thresholds": {
            "minimum_holdout_rows": args.min_holdout_rows,
            "minimum_set_exact_accuracy": args.min_set_exact,
            "minimum_ordered_exact_accuracy": args.min_ordered_exact,
            "minimum_hybrid_order_exact_accuracy": (
                args.min_hybrid_order_exact
            ),
        },
        "episode_isolation_required": True,
        "frozen_holdout_excluded_from_training": True,
        "opponents": league_entries,
        "excluded": excluded_entries,
        "ppo_cli_args": [
            item
            for entry in league_entries
            for item in entry["ppo_cli_args"]
        ],
        "safety": {
            "uses_public_replay_actions_only": True,
            "uses_open_submission_code": False,
            "uploads_or_submissions_performed": False,
        },
    }
    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "generated_at": utc_now(),
        "input_manifest": str(args.manifest),
        "input_manifest_sha256": input_manifest_sha,
        "base_checkpoint": (
            str(args.base_checkpoint) if args.base_checkpoint else None
        ),
        "base_checkpoint_sha256": base_checkpoint_sha,
        "selected_policies": len(policies),
        "completed_quality_pass": sum(
            result.get("status") == "quality_pass" for result in results
        ),
        "league_opponents": len(league_entries),
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
        "league_manifest": str(args.output_dir / "league_manifest.json"),
    }
    atomic_write_json(args.output_dir / "league_manifest.json", league_manifest)
    atomic_write_json(args.output_dir / "clone_quality_summary.json", summary)
    atomic_write_json(
        args.output_dir / "run_config.json",
        {
            "schema_version": OUTPUT_SCHEMA,
            "created_at": utc_now(),
            "argv": sys.argv,
            "input_manifest_schema": (
                input_manifest.get("schema_version")
                or input_manifest.get("schema")
            ),
            "input_manifest": str(args.manifest),
            "input_manifest_sha256": input_manifest_sha,
            "base_checkpoint": (
                str(args.base_checkpoint) if args.base_checkpoint else None
            ),
            "base_checkpoint_sha256": base_checkpoint_sha,
            "dry_run": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

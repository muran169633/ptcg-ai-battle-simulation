#!/usr/bin/env python3
"""Safely merge base and recovery gold-clone PPO league manifests.

The merger is local-only: it does not train, upload, or submit anything.  Every
selected checkpoint and deck is re-hashed from disk.  Duplicate policy IDs are
an error unless ``--prefer-later`` explicitly authorizes a later valid input to
replace the earlier selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEAGUE_SCHEMA = "ptcg-ppo-opponent-league-v1"
MERGE_SCHEMA = "ptcg-ppo-opponent-league-merge-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class InputLeague:
    index: int
    role: str
    path: Path
    sha256: str
    manifest: dict[str, Any]
    safety_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedCandidate:
    policy_id: str
    entry_index: int
    source: InputLeague
    entry: dict[str, Any]
    checkpoint: Path
    checkpoint_sha256: str
    deck: Path
    deck_hash: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def resolve_input_path(value: Any, manifest_path: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def compute_deck_hash(path: Path) -> str:
    values: list[int] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                values.append(int(line.strip()))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid deck CSV") from error
    if len(values) != 60:
        raise ValueError(f"{path}: contains {len(values)} cards; expected 60")
    canonical = ",".join(str(card) for card in sorted(values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_safety_reasons(manifest: dict[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    safety = manifest.get("safety")
    if not isinstance(safety, dict):
        reasons.append("manifest_safety_missing")
    else:
        if safety.get("uses_public_replay_actions_only") is not True:
            reasons.append("public_replay_only_not_verified")
        if safety.get("uses_open_submission_code") is not False:
            reasons.append("open_submission_code_safety_not_verified")
        if safety.get("uploads_or_submissions_performed") is not False:
            reasons.append("upload_submission_safety_not_verified")
    if manifest.get("episode_isolation_required") is not True:
        reasons.append("episode_isolation_not_verified")
    if manifest.get("frozen_holdout_excluded_from_training") is not True:
        reasons.append("frozen_holdout_exclusion_not_verified")
    return tuple(reasons)


def load_input_league(path: Path, index: int, role: str) -> InputLeague:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = read_json_object(path)
    if manifest.get("schema_version") != LEAGUE_SCHEMA:
        raise ValueError(
            f"{path}: unsupported schema {manifest.get('schema_version')!r}; "
            f"expected {LEAGUE_SCHEMA!r}"
        )
    opponents = manifest.get("opponents")
    if not isinstance(opponents, list):
        raise ValueError(f"{path}: opponents must be a list")
    excluded = manifest.get("excluded", [])
    if not isinstance(excluded, list):
        raise ValueError(f"{path}: excluded must be a list")
    return InputLeague(
        index=index,
        role=role,
        path=path,
        sha256=file_sha256(path),
        manifest=manifest,
        safety_reasons=manifest_safety_reasons(manifest),
    )


def source_metadata(source: InputLeague) -> dict[str, Any]:
    return {
        "input_index": source.index,
        "role": source.role,
        "manifest": str(source.path),
        "manifest_sha256": source.sha256,
    }


def candidate_identity(entry: Any, entry_index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {"policy_id": None, "entry_index": entry_index}
    return {
        "policy_id": entry.get("policy_id"),
        "submission_id": entry.get("submission_id"),
        "team_name": entry.get("team_name"),
        "entry_index": entry_index,
    }


def validate_candidate(
    source: InputLeague,
    entry: Any,
    entry_index: int,
) -> tuple[ValidatedCandidate | None, list[str]]:
    reasons = list(source.safety_reasons)
    if not isinstance(entry, dict):
        return None, reasons + ["entry_not_object"]

    policy_id = entry.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        reasons.append("policy_id_missing")
    quality = entry.get("quality")
    if not isinstance(quality, dict) or quality.get("pass") is not True:
        reasons.append("quality_pass_not_true")
    elif isinstance(quality.get("checks"), dict) and not all(
        value is True for value in quality["checks"].values()
    ):
        reasons.append("quality_checks_not_all_true")
    if entry.get("ppo_eligible") is False:
        reasons.append("ppo_eligible_false")

    checkpoint = resolve_input_path(entry.get("checkpoint"), source.path)
    declared_checkpoint_sha = entry.get("checkpoint_sha256")
    actual_checkpoint_sha: str | None = None
    if checkpoint is None:
        reasons.append("checkpoint_path_missing")
    elif not checkpoint.is_file():
        reasons.append("checkpoint_not_found")
    elif (
        not isinstance(declared_checkpoint_sha, str)
        or SHA256_PATTERN.fullmatch(declared_checkpoint_sha) is None
    ):
        reasons.append("checkpoint_sha256_invalid")
    else:
        actual_checkpoint_sha = file_sha256(checkpoint)
        if actual_checkpoint_sha != declared_checkpoint_sha:
            reasons.append("checkpoint_sha256_mismatch")

    deck = resolve_input_path(entry.get("deck"), source.path)
    declared_deck_hash = entry.get("deck_hash")
    actual_deck_hash: str | None = None
    if deck is None:
        reasons.append("deck_path_missing")
    elif not deck.is_file():
        reasons.append("deck_not_found")
    elif (
        not isinstance(declared_deck_hash, str)
        or SHA256_PATTERN.fullmatch(declared_deck_hash) is None
    ):
        reasons.append("deck_hash_invalid")
    else:
        try:
            actual_deck_hash = compute_deck_hash(deck)
        except ValueError:
            reasons.append("deck_csv_invalid")
        else:
            if actual_deck_hash != declared_deck_hash:
                reasons.append("deck_hash_mismatch")

    cli_args = entry.get("ppo_cli_args")
    if (
        not isinstance(cli_args, list)
        or len(cli_args) != 3
        or cli_args[0] != "--extra-opponent"
    ):
        reasons.append("ppo_cli_args_invalid")
    elif checkpoint is not None and deck is not None:
        cli_checkpoint = resolve_input_path(cli_args[1], source.path)
        cli_deck = resolve_input_path(cli_args[2], source.path)
        if cli_checkpoint != checkpoint or cli_deck != deck:
            reasons.append("ppo_cli_args_path_mismatch")

    if checkpoint is not None and deck is not None:
        expected_name = f"{checkpoint.stem}@{deck.stem}"
        if entry.get("name") != expected_name:
            reasons.append("opponent_name_path_mismatch")

    if reasons:
        return None, sorted(set(reasons))
    assert isinstance(policy_id, str)
    assert checkpoint is not None
    assert actual_checkpoint_sha is not None
    assert deck is not None
    assert actual_deck_hash is not None
    return (
        ValidatedCandidate(
            policy_id=policy_id,
            entry_index=entry_index,
            source=source,
            entry=entry,
            checkpoint=checkpoint,
            checkpoint_sha256=actual_checkpoint_sha,
            deck=deck,
            deck_hash=actual_deck_hash,
        ),
        [],
    )


def safe_checkpoint_name(candidate: ValidatedCandidate) -> str:
    slug = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        candidate.policy_id,
    ).strip(".-_")
    slug = slug[:64] or "policy"
    policy_suffix = hashlib.sha256(
        candidate.policy_id.encode("utf-8")
    ).hexdigest()[:10]
    return (
        f"{slug}-{policy_suffix}-"
        f"{candidate.checkpoint_sha256[:12]}.pt"
    )


def copy_checkpoint_without_overwrite(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> str:
    """Copy atomically; an existing different file is always an error."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise FileExistsError(f"{destination} exists and is not a file")
        if file_sha256(destination) != expected_sha256:
            raise FileExistsError(
                f"Refusing to overwrite different checkpoint {destination}"
            )
        return "reused_identical"

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        if file_sha256(temporary) != expected_sha256:
            raise RuntimeError(
                f"Copied checkpoint hash mismatch for {destination}"
            )
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                not destination.is_file()
                or file_sha256(destination) != expected_sha256
            ):
                raise FileExistsError(
                    f"Concurrent different checkpoint at {destination}"
                )
            return "reused_identical_race"
        if file_sha256(destination) != expected_sha256:
            raise RuntimeError(
                f"Materialized checkpoint hash mismatch for {destination}"
            )
        return "copied"
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass --overwrite-output explicitly"
        )
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
    try:
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} appeared during merge; refusing to overwrite"
            )
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"{path} appeared during merge; refusing to overwrite"
                ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def merge_leagues(
    inputs: list[InputLeague],
    *,
    prefer_later: bool,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    selected: dict[str, ValidatedCandidate] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    carried_excluded: list[dict[str, Any]] = []

    for source in inputs:
        for excluded_index, excluded in enumerate(
            source.manifest.get("excluded", [])
        ):
            carried_excluded.append(
                {
                    "source": source_metadata(source),
                    "excluded_index": excluded_index,
                    "entry": excluded,
                }
            )
        for entry_index, entry in enumerate(source.manifest["opponents"]):
            candidate, reasons = validate_candidate(
                source,
                entry,
                entry_index,
            )
            if candidate is None:
                rejected.append(
                    {
                        "source": source_metadata(source),
                        **candidate_identity(entry, entry_index),
                        "reasons": reasons,
                    }
                )
                continue
            previous = selected.get(candidate.policy_id)
            event = {
                "source": source_metadata(source),
                "entry_index": entry_index,
                "checkpoint_sha256": candidate.checkpoint_sha256,
                "deck_hash": candidate.deck_hash,
            }
            if previous is not None and not prefer_later:
                raise ValueError(
                    f"Duplicate policy_id {candidate.policy_id!r} from "
                    f"{previous.source.path} and {source.path}; pass "
                    "--prefer-later to explicitly replace the earlier entry"
                )
            if previous is not None:
                histories[candidate.policy_id].append(
                    {
                        **event,
                        "action": "replaced_previous",
                        "replaced_source": source_metadata(previous.source),
                        "replaced_checkpoint_sha256": (
                            previous.checkpoint_sha256
                        ),
                    }
                )
            else:
                histories[candidate.policy_id] = [
                    {**event, "action": "selected"}
                ]
            selected[candidate.policy_id] = candidate

    candidates = list(selected.values())
    materialized: dict[str, tuple[Path, str]] = {}
    planned_names: dict[str, str] = {}
    for candidate in candidates:
        if checkpoint_dir is None:
            checkpoint = candidate.checkpoint
            copy_status = "referenced"
        else:
            checkpoint = (
                checkpoint_dir.resolve() / safe_checkpoint_name(candidate)
            )
            copy_status = "pending_copy"
        expected_name = f"{checkpoint.stem}@{candidate.deck.stem}"
        other_policy = planned_names.get(expected_name)
        if other_policy is not None and other_policy != candidate.policy_id:
            raise ValueError(
                f"PPO opponent name collision {expected_name!r} for "
                f"{other_policy!r} and {candidate.policy_id!r}"
            )
        planned_names[expected_name] = candidate.policy_id
        materialized[candidate.policy_id] = (checkpoint, copy_status)

    if checkpoint_dir is not None:
        checkpoint_dir = checkpoint_dir.resolve()
        if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
            raise NotADirectoryError(checkpoint_dir)
        # Check every existing target before copying any source.
        for candidate in candidates:
            checkpoint, _ = materialized[candidate.policy_id]
            if (
                checkpoint.exists()
                and (
                    not checkpoint.is_file()
                    or file_sha256(checkpoint)
                    != candidate.checkpoint_sha256
                )
            ):
                raise FileExistsError(
                    f"Refusing to overwrite different checkpoint {checkpoint}"
                )
        for candidate in candidates:
            checkpoint, _ = materialized[candidate.policy_id]
            status = copy_checkpoint_without_overwrite(
                candidate.checkpoint,
                checkpoint,
                candidate.checkpoint_sha256,
            )
            materialized[candidate.policy_id] = (checkpoint, status)

    opponents: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for candidate in candidates:
        checkpoint, materialization = materialized[candidate.policy_id]
        # Verify references and copies again at the point recorded in output.
        if file_sha256(checkpoint) != candidate.checkpoint_sha256:
            raise RuntimeError(
                f"Final checkpoint hash mismatch for {candidate.policy_id}"
            )
        if compute_deck_hash(candidate.deck) != candidate.deck_hash:
            raise RuntimeError(
                f"Final deck hash mismatch for {candidate.policy_id}"
            )
        name = f"{checkpoint.stem}@{candidate.deck.stem}"
        normalized = {
            **candidate.entry,
            "name": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": candidate.checkpoint_sha256,
            "deck": str(candidate.deck),
            "deck_hash": candidate.deck_hash,
            "ppo_cli_args": [
                "--extra-opponent",
                str(checkpoint),
                str(candidate.deck),
            ],
            "merge_provenance": {
                "selected_source": source_metadata(candidate.source),
                "selected_entry_index": candidate.entry_index,
                "materialization": materialization,
            },
        }
        opponents.append(normalized)
        selections.append(
            {
                "policy_id": candidate.policy_id,
                "selected_source": source_metadata(candidate.source),
                "selected_entry_index": candidate.entry_index,
                "checkpoint_sha256": candidate.checkpoint_sha256,
                "deck_hash": candidate.deck_hash,
                "history": histories[candidate.policy_id],
            }
        )

    return {
        "schema_version": LEAGUE_SCHEMA,
        "generated_at": utc_now(),
        "episode_isolation_required": True,
        "frozen_holdout_excluded_from_training": True,
        "opponents": opponents,
        "excluded": carried_excluded,
        "rejected": rejected,
        "ppo_cli_args": [
            item
            for opponent in opponents
            for item in opponent["ppo_cli_args"]
        ],
        "safety": {
            "uses_public_replay_actions_only": True,
            "uses_open_submission_code": False,
            "uploads_or_submissions_performed": False,
            "all_selected_entries_reverified": True,
            "checkpoint_sha256_reverified": True,
            "deck_hash_reverified": True,
        },
        "merge": {
            "schema_version": MERGE_SCHEMA,
            "prefer_later": prefer_later,
            "checkpoint_mode": (
                "copy" if checkpoint_dir is not None else "reference"
            ),
            "checkpoint_dir": (
                str(checkpoint_dir) if checkpoint_dir is not None else None
            ),
            "inputs": [
                {
                    **source_metadata(source),
                    "opponent_entries": len(
                        source.manifest.get("opponents", [])
                    ),
                    "excluded_entries": len(
                        source.manifest.get("excluded", [])
                    ),
                    "manifest_safety_reasons": list(
                        source.safety_reasons
                    ),
                }
                for source in inputs
            ],
            "selections": selections,
            "selected_policy_count": len(opponents),
            "rejected_candidate_count": len(rejected),
            "carried_excluded_count": len(carried_excluded),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely merge a base gold-clone PPO league with recovery leagues. "
            "This command never trains, uploads, or submits."
        )
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--recovery",
        type=Path,
        action="append",
        default=[],
        help="Recovery league manifest; repeat in desired precedence order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefer-later",
        action="store_true",
        help=(
            "Explicitly allow a later valid input to replace an earlier entry "
            "with the same policy_id."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Copy selected checkpoints into this directory. Omit to reference "
            "the verified source files in place."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Explicitly replace only the output manifest if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [args.base, *args.recovery]
    inputs = [
        load_input_league(
            path,
            index,
            "base" if index == 0 else "recovery",
        )
        for index, path in enumerate(input_paths)
    ]
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else None
    )
    merged = merge_leagues(
        inputs,
        prefer_later=args.prefer_later,
        checkpoint_dir=checkpoint_dir,
    )
    output = args.output.expanduser().resolve()
    atomic_write_json(
        output,
        merged,
        overwrite=args.overwrite_output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": file_sha256(output),
                "selected": len(merged["opponents"]),
                "rejected": len(merged["rejected"]),
                "carried_excluded": len(merged["excluded"]),
                "checkpoint_mode": merged["merge"]["checkpoint_mode"],
                "uploads_or_submissions_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

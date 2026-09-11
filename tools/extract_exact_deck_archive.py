#!/usr/bin/env python3
"""Extract one exact-deck specialist archive from an existing BC archive.

The source rows already contain the canonical ``deck_hash`` and strict split,
so this tool does not reinterpret replay labels or hidden state.  It also
recovers the submitted 60-card list from the original official replay named by
one of the selected rows.  The resulting archive is therefore cheap to build,
keeps the source train/valid/test protocol, and is directly usable by the BC
and head-to-head tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import prepare_bc_week as base
from prepare_gold8_week import extract_replay_decks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/gold8_recent7_20260808/general_top20.zip"
DEFAULT_OUTPUT_ROOT = ROOT / "data/portfolio4_recent7_20260809"
SPECIALIST_SCHEMA = "ptcg-exact-deck-archive-from-bc-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def source_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    if "manifest.json" not in archive.namelist():
        raise RuntimeError("Source BC archive has no manifest.json")
    manifest = json.loads(archive.read("manifest.json"))
    if manifest.get("schema_version") != base.SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected source schema {manifest.get('schema_version')!r}"
        )
    return manifest


def recover_deck(
    manifest: dict[str, Any],
    episode_ids_by_date: dict[str, list[str]],
    expected_hash: str,
) -> tuple[list[int], Path, str]:
    """Open only replay members named by selected BC rows."""

    source_paths = {
        str(raw.get("date") or ""): Path(str(raw.get("path") or ""))
        for raw in manifest.get("sources", [])
        if isinstance(raw, dict)
    }
    checked: list[str] = []
    for dataset_date, episode_ids in sorted(episode_ids_by_date.items(), reverse=True):
        replay_path = source_paths.get(dataset_date)
        if replay_path is None or not replay_path.is_file():
            continue
        with zipfile.ZipFile(replay_path) as replay_archive:
            members = set(replay_archive.namelist())
            for episode_id in episode_ids:
                candidates = (f"{episode_id}.json", episode_id)
                member = next((name for name in candidates if name in members), None)
                if member is None:
                    continue
                checked.append(f"{dataset_date}:{episode_id}")
                episode = json.loads(replay_archive.read(member))
                hashes, decks = extract_replay_decks(episode)
                for deck_hash, deck in zip(hashes, decks):
                    if deck_hash == expected_hash and deck is not None:
                        return deck, replay_path.resolve(), member
    raise RuntimeError(
        "Could not recover the exact 60-card list from selected replay IDs; "
        f"checked={checked[:20]}"
    )


def extract_archive(
    source_path: Path,
    output_path: Path,
    deck_path: Path,
    deck_hash: str,
    slug: str,
    label: str,
    frames_per_shard: int,
    replay_candidates_per_date: int,
    overwrite: bool,
) -> dict[str, Any]:
    if len(deck_hash) != 64 or any(char not in "0123456789abcdef" for char in deck_hash):
        raise ValueError("--deck-hash must be a lowercase SHA-256 hex digest")
    if frames_per_shard < 1 or replay_candidates_per_date < 1:
        raise ValueError("Shard size and replay candidate count must be positive")
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    deck_path = deck_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    split_counts: Counter[str] = Counter()
    split_episodes: dict[str, set[str]] = {
        split: set() for split in ("train", "valid", "test")
    }
    team_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    episode_ids_by_date: dict[str, list[str]] = defaultdict(list)
    seen_candidates: dict[str, set[str]] = defaultdict(set)

    with zipfile.ZipFile(source_path) as source:
        manifest = source_manifest(source)
        members = sorted(
            name for name in source.namelist() if name.endswith(".jsonl")
        )
        if not members:
            raise RuntimeError("Source BC archive has no JSONL shards")
        with base.DecisionArchiveWriter(
            output_path, frames_per_shard, overwrite
        ) as writer:
            for member_index, member in enumerate(members, 1):
                base.log(f"[{member_index}/{len(members)}] filtering {member}")
                with source.open(member) as rows:
                    for line in rows:
                        row = json.loads(line)
                        if row.get("deck_hash") != deck_hash:
                            continue
                        split = str(row.get("split") or "")
                        if split not in split_episodes:
                            raise ValueError(f"Selected row has invalid split {split!r}")
                        writer.add(split, row)
                        split_counts[split] += 1
                        episode_id = str(row.get("episode_id") or "")
                        dataset_date = str(row.get("dataset_date") or "")
                        if not episode_id:
                            raise ValueError("Selected row has no episode_id")
                        split_episodes[split].add(episode_id)
                        team_counts[str(row.get("team_name") or "")] += 1
                        context_counts[str(row.get("select_context", "unknown"))] += 1
                        date_counts[dataset_date] += 1
                        if (
                            len(episode_ids_by_date[dataset_date])
                            < replay_candidates_per_date
                            and episode_id not in seen_candidates[dataset_date]
                        ):
                            episode_ids_by_date[dataset_date].append(episode_id)
                            seen_candidates[dataset_date].add(episode_id)

            missing = [
                split for split in ("train", "valid", "test")
                if split_counts[split] <= 0
            ]
            if missing:
                raise RuntimeError(
                    f"Exact deck {deck_hash} has no rows in source splits {missing}"
                )

            deck, replay_source, replay_member = recover_deck(
                manifest, episode_ids_by_date, deck_hash
            )
            canonical = ",".join(str(card) for card in sorted(deck))
            recovered_hash = hashlib.sha256(canonical.encode()).hexdigest()
            if recovered_hash != deck_hash:
                raise RuntimeError("Recovered deck hash does not match requested hash")
            atomic_write_text(
                deck_path,
                "".join(f"{card}\n" for card in sorted(deck)),
                overwrite,
            )

            output_manifest = {
                "schema_version": base.SCHEMA_VERSION,
                "specialist_schema_version": SPECIALIST_SCHEMA,
                "competition": manifest.get("competition"),
                "profile": {
                    "slug": slug,
                    "label": label,
                    "deck_hash": deck_hash,
                    "deck": str(deck_path),
                },
                "dates": manifest.get("dates"),
                "split_policy": manifest.get("split_policy"),
                "team_filter": manifest.get("team_filter"),
                "label_alignment": manifest.get("label_alignment"),
                "hidden_information_policy": manifest.get("hidden_information_policy"),
                "loss_trajectory_weights": manifest.get("loss_trajectory_weights"),
                "target_bc_exact_accuracy": manifest.get(
                    "target_bc_exact_accuracy", 0.75
                ),
                "stats": {
                    "decisions": sum(split_counts.values()),
                    "episodes": sum(len(values) for values in split_episodes.values()),
                },
                "split_decisions": dict(split_counts),
                "split_episodes": {
                    split: len(values) for split, values in split_episodes.items()
                },
                "team_decisions": dict(team_counts.most_common()),
                "context_decisions": dict(context_counts.most_common()),
                "date_decisions": dict(sorted(date_counts.items())),
                "shards": dict(writer.shard_index),
                "source_archive": {
                    "path": str(source_path),
                    "sha256": sha256_file(source_path),
                },
                "deck_recovery": {
                    "source": str(replay_source),
                    "member": replay_member,
                    "visualize_not_copied_to_rows": True,
                },
            }
            writer.finish(output_manifest)

    verified = base.verify_archive(output_path)
    result = {
        "archive": str(output_path),
        "archive_sha256": sha256_file(output_path),
        "deck": str(deck_path),
        "deck_sha256": sha256_file(deck_path),
        "deck_hash": deck_hash,
        "split_decisions": verified["split_decisions"],
        "split_episodes": verified["split_episodes"],
        "split_policy": verified["split_policy"],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--deck-hash", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--frames-per-shard", type=int, default=50_000)
    parser.add_argument("--replay-candidates-per-date", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    result = extract_archive(
        source_path=args.source,
        output_path=output_root / "archives" / f"{args.slug}.zip",
        deck_path=output_root / "decks" / f"{args.deck_hash}.csv",
        deck_hash=args.deck_hash,
        slug=args.slug,
        label=args.label,
        frames_per_shard=args.frames_per_shard,
        replay_candidates_per_date=args.replay_candidates_per_date,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

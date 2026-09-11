#!/usr/bin/env python3
"""Build an auditable four-file submission archive from an Orbit V5 BC checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

import torch


SUPPORTED_FEATURE_VERSIONS = {
    "ptcg-bc-orbit-entity-transformer-v5",
    "ptcg-bc-orbit-entity-transformer-v6-priority",
}
EXPECTED_MEMBERS = ("main.py", "deck.csv", "model.pt", "policy_runtime.py")
MODEL_CONFIG_KEYS = (
    "hash_size",
    "categorical_dim",
    "model_dim",
    "layers",
    "heads",
    "dropout",
    "max_state_entities",
    "entity_fields",
    "option_fields",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deck_hash(path: Path) -> str:
    cards = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError(f"deck must contain 60 cards, found {len(cards)}")
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_deterministic_archive(archive_path: Path, source_dir: Path) -> None:
    """Write a byte-reproducible gzip-compressed tar with normalized metadata."""
    with archive_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with tarfile.open(
                fileobj=gzip_handle,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as tar:
                for name in EXPECTED_MEMBERS:
                    source = source_dir / name
                    info = tarfile.TarInfo(name=name)
                    info.size = source.stat().st_size
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as source_handle:
                        tar.addfile(info, source_handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("submissions/ptcg_bc_orbit_v5"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    for path in (args.checkpoint, args.deck):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in ("main.py", "policy_runtime.py"):
        if not (args.template_dir / name).is_file():
            raise FileNotFoundError(args.template_dir / name)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.archive.exists():
        raise FileExistsError(args.archive)

    resolved_deck_hash = deck_hash(args.deck)
    main_source = (args.template_dir / "main.py").read_text(encoding="utf-8")
    match = re.search(r'^DECK_HASH = "([0-9a-f]{64})"$', main_source, re.MULTILINE)
    if match is None:
        raise ValueError("Template main.py has no auditable DECK_HASH constant")
    if match.group(1) != resolved_deck_hash:
        raise ValueError(
            "Template deck hash does not match requested deck: "
            f"template={match.group(1)} requested={resolved_deck_hash}"
        )

    source_checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    feature_version = source_checkpoint.get("feature_version")
    if feature_version not in SUPPORTED_FEATURE_VERSIONS:
        raise ValueError(
            "Checkpoint is not a supported Orbit BC checkpoint: "
            f"{feature_version!r}"
        )
    runtime_source = (args.template_dir / "policy_runtime.py").read_text(
        encoding="utf-8"
    )
    runtime_match = re.search(
        r'^FEATURE_VERSION = "([^"]+)"$', runtime_source, re.MULTILINE
    )
    if runtime_match is None or runtime_match.group(1) != feature_version:
        raise ValueError(
            "Template policy_runtime.py feature version does not match "
            f"checkpoint: runtime={runtime_match.group(1) if runtime_match else None!r} "
            f"checkpoint={feature_version!r}"
        )
    config = source_checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no BC config mapping")
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f"Checkpoint config is missing keys: {missing}")
    state = source_checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no model state")

    args.output_dir.mkdir(parents=True)
    shutil.copy2(args.template_dir / "main.py", args.output_dir / "main.py")
    shutil.copy2(
        args.template_dir / "policy_runtime.py",
        args.output_dir / "policy_runtime.py",
    )
    shutil.copy2(args.deck, args.output_dir / "deck.csv")

    bundled: dict[str, Any] = {
        "feature_version": feature_version,
        "model_config": {key: config[key] for key in MODEL_CONFIG_KEYS},
        "epoch": source_checkpoint.get("epoch"),
        "metrics": source_checkpoint.get("valid_metrics"),
        "model_state_dict": state,
    }
    model_path = args.output_dir / "model.pt"
    torch.save(bundled, model_path)
    reloaded = torch.load(model_path, map_location="cpu", weights_only=True)
    if reloaded["model_state_dict"].keys() != state.keys():
        raise RuntimeError("Bundled state keys differ from source")
    unequal = [
        name
        for name in state
        if not torch.equal(state[name], reloaded["model_state_dict"][name])
    ]
    if unequal:
        raise RuntimeError(f"Bundled tensors differ from source: {unequal[:5]}")

    for name in ("main.py", "policy_runtime.py"):
        source = (args.output_dir / name).read_text(encoding="utf-8")
        compile(source, str(args.output_dir / name), "exec")

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_archive(args.archive, args.output_dir)
    with tarfile.open(args.archive, mode="r:gz") as tar:
        members = tuple(member.name for member in tar.getmembers())
        if members != EXPECTED_MEMBERS:
            raise RuntimeError(f"Unexpected archive members: {members}")
        if any(not member.isfile() for member in tar.getmembers()):
            raise RuntimeError("Archive contains a non-file member")

    archive_sha256 = sha256_file(args.archive)
    sidecar = args.archive.with_suffix(args.archive.suffix + ".sha256")
    sidecar.write_text(
        f"{archive_sha256}  {args.archive.name}\n",
        encoding="utf-8",
    )
    manifest_path = args.manifest or args.archive.with_suffix(
        args.archive.suffix + ".manifest.json"
    )
    manifest = {
        "schema_version": "ptcg-bc-submission-package-v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_epoch": source_checkpoint.get("epoch"),
        "source_valid_metrics": source_checkpoint.get("valid_metrics"),
        "deck": str(args.deck.resolve()),
        "deck_hash": resolved_deck_hash,
        "deck_sha256": sha256_file(args.deck),
        "template_dir": str(args.template_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "archive": str(args.archive.resolve()),
        "archive_sha256": archive_sha256,
        "archive_bytes": args.archive.stat().st_size,
        "archive_determinism": {
            "gzip_mtime": 0,
            "tar_mtime": 0,
            "uid": 0,
            "gid": 0,
            "mode": "0644",
            "member_order": list(EXPECTED_MEMBERS),
        },
        "members": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in EXPECTED_MEMBERS
        },
        "model_tensor_count": len(state),
        "model_tensors_equal_source": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

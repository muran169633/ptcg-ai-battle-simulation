#!/usr/bin/env python3
"""Read-only identity audit for two independently built PPO archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import torch

from package_ppo_submission import (
    EXPECTED_MEMBERS,
    MODEL_KEYS,
    read_template_contract,
    sha256_file,
)
from train_ppo import compute_deck_hash, read_deck


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}"
        )


def read_archive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    if len(raw) < 10 or raw[:2] != b"\x1f\x8b":
        raise ValueError(f"{path}: not a gzip archive")
    require_equal(raw[4:8], b"\x00\x00\x00\x00", f"{path} gzip mtime")
    payloads: dict[str, bytes] = {}
    member_rows: dict[str, dict[str, Any]] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        require_equal(
            tuple(member.name for member in members),
            EXPECTED_MEMBERS,
            f"{path} member order",
        )
        for member in members:
            if not member.isfile():
                raise ValueError(f"{path}: {member.name} is not a regular file")
            require_equal(member.mtime, 0, f"{path}:{member.name} mtime")
            require_equal(member.mode, 0o644, f"{path}:{member.name} mode")
            require_equal(member.uid, 0, f"{path}:{member.name} uid")
            require_equal(member.gid, 0, f"{path}:{member.name} gid")
            require_equal(member.uname, "", f"{path}:{member.name} uname")
            require_equal(member.gname, "", f"{path}:{member.name} gname")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"{path}: cannot read {member.name}")
            payload = handle.read()
            payloads[member.name] = payload
            member_rows[member.name] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "members": member_rows,
        "payloads": payloads,
    }


def validate_manifest(
    path: Path,
    *,
    archive: dict[str, Any],
    checkpoint_sha256: str,
    deck_hash: str,
    action_order_mode: str,
    contract_path: Path,
    deployment_contract_path: Path | None,
    source_files: dict[str, Path],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest must be an object")
    require_equal(
        value.get("source_checkpoint_sha256"),
        checkpoint_sha256,
        f"{path} source checkpoint",
    )
    require_equal(
        value.get("archive_sha256"),
        archive["sha256"],
        f"{path} archive SHA-256",
    )
    require_equal(
        value.get("members"),
        archive["members"],
        f"{path} member identities",
    )
    require_equal(
        value.get("action_order_mode"),
        action_order_mode,
        f"{path} action-order mode",
    )
    require_equal(
        value.get("semantic_deck_hash"),
        deck_hash,
        f"{path} semantic deck hash",
    )
    require_equal(
        (value.get("template_contract") or {}).get("sha256"),
        sha256_file(contract_path),
        f"{path} template contract",
    )
    expected_sources = {
        name: {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        }
        for name, source in source_files.items()
    }
    require_equal(
        value.get("template_sources"),
        expected_sources,
        f"{path} template sources",
    )
    require_equal(
        value.get("byte_reproducible_packaging"),
        True,
        f"{path} reproducible flag",
    )
    expected_deployment = (
        {
            "path": str(deployment_contract_path.resolve()),
            "sha256": sha256_file(deployment_contract_path),
            "schema_version": "ptcg-gold-push-deployment-contract-v1",
        }
        if deployment_contract_path is not None
        else None
    )
    require_equal(
        value.get("deployment_contract"),
        expected_deployment,
        f"{path} deployment contract",
    )
    return value


def audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = (
        args.archive_a,
        args.archive_b,
        args.manifest_a,
        args.manifest_b,
        args.checkpoint,
        args.main_file,
        args.deck_file,
        args.policy_runtime_file,
        args.template_contract,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.deployment_contract is not None and not args.deployment_contract.is_file():
        raise FileNotFoundError(args.deployment_contract)
    source_files = {
        "main.py": args.main_file,
        "deck.csv": args.deck_file,
        "policy_runtime.py": args.policy_runtime_file,
    }
    contract = read_template_contract(
        args.template_contract,
        action_order_mode=args.action_order_mode,
        source_files=source_files,
    )
    checkpoint_sha = sha256_file(args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    archive_a = read_archive(args.archive_a)
    archive_b = read_archive(args.archive_b)
    require_equal(
        archive_a["sha256"],
        archive_b["sha256"],
        "independent archive SHA-256",
    )
    require_equal(
        archive_a["payloads"],
        archive_b["payloads"],
        "independent archive member payloads",
    )
    for name, source in source_files.items():
        require_equal(
            archive_a["payloads"][name],
            source.read_bytes(),
            f"bundled {name}",
        )
    bundled = torch.load(
        io.BytesIO(archive_a["payloads"]["model.pt"]),
        map_location="cpu",
        weights_only=True,
    )
    require_equal(
        tuple(bundled),
        tuple(MODEL_KEYS),
        "bundled checkpoint keys",
    )
    source_state = checkpoint.get("model_state_dict")
    bundled_state = bundled.get("model_state_dict")
    if not isinstance(source_state, dict) or not isinstance(bundled_state, dict):
        raise ValueError("Source or bundled checkpoint lacks model state")
    require_equal(
        tuple(source_state),
        tuple(bundled_state),
        "model state keys",
    )
    unequal = [
        name
        for name in source_state
        if not torch.equal(source_state[name], bundled_state[name])
    ]
    require_equal(unequal, [], "model tensors")
    deck = read_deck(args.deck_file)
    deck_hash = compute_deck_hash(deck)
    require_equal(
        checkpoint.get("learner_deck_hash"),
        deck_hash,
        "checkpoint learner deck hash",
    )
    manifest_a = validate_manifest(
        args.manifest_a,
        archive=archive_a,
        checkpoint_sha256=checkpoint_sha,
        deck_hash=deck_hash,
        action_order_mode=args.action_order_mode,
        contract_path=args.template_contract,
        source_files=source_files,
        deployment_contract_path=args.deployment_contract,
    )
    manifest_b = validate_manifest(
        args.manifest_b,
        archive=archive_b,
        checkpoint_sha256=checkpoint_sha,
        deck_hash=deck_hash,
        action_order_mode=args.action_order_mode,
        contract_path=args.template_contract,
        source_files=source_files,
        deployment_contract_path=args.deployment_contract,
    )
    require_equal(
        manifest_a.get("members"),
        manifest_b.get("members"),
        "independent manifest members",
    )
    return {
        "schema_version": "ptcg-ppo-package-pair-audit-v1",
        "pass": True,
        "read_only": True,
        "action_order_mode": args.action_order_mode,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha,
        },
        "deck_hash": deck_hash,
        "template_contract": {
            "path": str(args.template_contract.resolve()),
            "sha256": sha256_file(args.template_contract),
            "template_version": contract.get("template_version"),
        },
        "deployment_contract": (
            {
                "path": str(args.deployment_contract.resolve()),
                "sha256": sha256_file(args.deployment_contract),
            }
            if args.deployment_contract is not None
            else None
        ),
        "archive_sha256": archive_a["sha256"],
        "archive_bytes": archive_a["bytes"],
        "members": archive_a["members"],
        "independent_archive_bytes_identical": True,
        "model_tensors_equal_source": True,
        "external_submission_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-a", type=Path, required=True)
    parser.add_argument("--archive-b", type=Path, required=True)
    parser.add_argument("--manifest-a", type=Path, required=True)
    parser.add_argument("--manifest-b", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--main-file", type=Path, required=True)
    parser.add_argument("--deck-file", type=Path, required=True)
    parser.add_argument("--policy-runtime-file", type=Path, required=True)
    parser.add_argument("--template-contract", type=Path, required=True)
    parser.add_argument("--deployment-contract", type=Path)
    parser.add_argument(
        "--action-order-mode",
        choices=("raw", "canonical", "hybrid"),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(audit(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

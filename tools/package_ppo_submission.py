#!/usr/bin/env python3
"""Build a four-file PTCG PPO submission archive from a training checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import torch


EXPECTED_MEMBERS = ("main.py", "deck.csv", "model.pt", "policy_runtime.py")
DEPLOYMENT_SCHEMA = "ptcg-gold-push-deployment-contract-v1"
MODEL_KEYS = (
    "feature_version",
    "bc_feature_version",
    "model_config",
    "model_state_dict",
    "reward",
    "update",
    "action_distribution",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_deck_hash(path: Path) -> str:
    cards = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError(f"{path} contains {len(cards)} cards; expected 60")
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_template_contract(
    path: Path,
    *,
    action_order_mode: str,
    source_files: dict[str, Path],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("Template contract must be a JSON object")
    decode = contract.get("decode")
    if not isinstance(decode, dict):
        raise ValueError("Template contract is missing decode metadata")
    if decode.get("order_mode") != action_order_mode:
        raise ValueError(
            "Template contract action-order mode does not match "
            "--action-order-mode"
        )
    declared_files = contract.get("template_files")
    if not isinstance(declared_files, dict):
        raise ValueError("Template contract is missing template_files")
    expected_files = {
        name: sha256_file(source)
        for name, source in source_files.items()
    }
    if declared_files != expected_files:
        raise ValueError(
            "Template contract file SHA-256 values do not match sources"
        )
    deck = contract.get("deck")
    if not isinstance(deck, dict):
        raise ValueError("Template contract is missing deck metadata")
    declared_deck_hash = deck.get("semantic_hash", deck.get("deck_hash"))
    actual_deck_hash = semantic_deck_hash(source_files["deck.csv"])
    if declared_deck_hash != actual_deck_hash:
        raise ValueError("Template contract semantic deck hash does not match")
    return contract


def resolve_contract_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def validate_deployment_contract(
    path: Path,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    deck: Path,
    deck_file_sha256: str,
    deck_hash: str,
    action_order_mode: str,
    template_contract: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("Unexpected deployment contract schema")
    candidate = value.get("candidate")
    candidate_deck = value.get("candidate_deck")
    action_order = value.get("action_order")
    submission_template = value.get("submission_template")
    if not all(
        isinstance(row, dict)
        for row in (
            candidate,
            candidate_deck,
            action_order,
            submission_template,
        )
    ):
        raise ValueError("Deployment contract sections are missing")
    assert isinstance(candidate, dict)
    assert isinstance(candidate_deck, dict)
    assert isinstance(action_order, dict)
    assert isinstance(submission_template, dict)
    expected = {
        "candidate.path": (
            resolve_contract_path(candidate.get("path"), path.parent, "candidate.path"),
            checkpoint.resolve(),
        ),
        "candidate.sha256": (candidate.get("sha256"), checkpoint_sha256),
        "candidate_deck.path": (
            resolve_contract_path(
                candidate_deck.get("path"),
                path.parent,
                "candidate_deck.path",
            ),
            deck.resolve(),
        ),
        "candidate_deck.file_sha256": (
            candidate_deck.get("file_sha256"),
            deck_file_sha256,
        ),
        "candidate_deck.semantic_hash": (
            candidate_deck.get("semantic_hash"),
            deck_hash,
        ),
        "action_order.mode": (
            action_order.get("mode"),
            action_order_mode,
        ),
        "action_order.canonical_order": (
            action_order.get("canonical_order"),
            action_order_mode == "canonical",
        ),
        "action_order.hybrid_order": (
            action_order.get("hybrid_order"),
            action_order_mode == "hybrid",
        ),
        "submission_template.contract": (
            resolve_contract_path(
                submission_template.get("contract"),
                path.parent,
                "submission_template.contract",
            ),
            template_contract.resolve(),
        ),
        "submission_template.contract_sha256": (
            submission_template.get("contract_sha256"),
            sha256_file(template_contract),
        ),
    }
    mismatches = {
        label: {"actual": actual, "expected": expected_value}
        for label, (actual, expected_value) in expected.items()
        if actual != expected_value
    }
    if mismatches:
        raise ValueError(f"Deployment contract mismatch: {mismatches}")
    return value


def deterministic_torch_save(value: Any, path: Path) -> None:
    """Serialize independently of the destination directory or file mtime."""
    buffer = io.BytesIO()
    torch.save(value, buffer)
    path.write_bytes(buffer.getvalue())


def write_deterministic_archive(root: Path, archive_path: Path) -> None:
    """Write a byte-reproducible gzip-compressed tar with normalized metadata."""
    with archive_path.open("xb") as raw_handle:
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
            ) as tar_handle:
                for name in EXPECTED_MEMBERS:
                    payload = (root / name).read_bytes()
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    tar_handle.addfile(info, io.BytesIO(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("submissions/ptcg_ppo_terminal01_v1"),
    )
    parser.add_argument(
        "--main-file",
        type=Path,
        help="Optional action-order-specific main.py override.",
    )
    parser.add_argument(
        "--deck-file",
        type=Path,
        help="Optional exact deck.csv source override.",
    )
    parser.add_argument(
        "--policy-runtime-file",
        type=Path,
        help="Optional exact policy_runtime.py source override.",
    )
    parser.add_argument("--template-contract", type=Path, required=True)
    parser.add_argument("--deployment-contract", type=Path)
    parser.add_argument(
        "--action-order-mode",
        choices=("raw", "canonical", "hybrid"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    source_files = {
        "main.py": args.main_file or args.template_dir / "main.py",
        "deck.csv": args.deck_file or args.template_dir / "deck.csv",
        "policy_runtime.py": (
            args.policy_runtime_file
            or args.template_dir / "policy_runtime.py"
        ),
    }
    for source in source_files.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    template_contract = read_template_contract(
        args.template_contract,
        action_order_mode=args.action_order_mode,
        source_files=source_files,
    )
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {args.output_dir}"
        )
    if args.archive.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing archive: {args.archive}"
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("feature_version") != "ptcg-selfplay-ppo-terminal01-v1":
        raise ValueError("Checkpoint is not the expected PPO feature version")
    missing = [key for key in MODEL_KEYS if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing submission keys: {missing}")
    deck_hash = semantic_deck_hash(source_files["deck.csv"])
    if checkpoint.get("learner_deck_hash") != deck_hash:
        raise ValueError(
            "Checkpoint learner_deck_hash does not match packaged deck.csv"
        )
    deployment_contract: dict[str, Any] | None = None
    if args.deployment_contract is not None:
        deployment_contract = validate_deployment_contract(
            args.deployment_contract,
            checkpoint=args.checkpoint,
            checkpoint_sha256=sha256_file(args.checkpoint),
            deck=source_files["deck.csv"],
            deck_file_sha256=sha256_file(source_files["deck.csv"]),
            deck_hash=deck_hash,
            action_order_mode=args.action_order_mode,
            template_contract=args.template_contract,
        )

    args.output_dir.mkdir(parents=True)
    for name, source in source_files.items():
        shutil.copy2(source, args.output_dir / name)

    slim_checkpoint: dict[str, Any] = {
        key: checkpoint[key]
        for key in MODEL_KEYS
    }
    model_path = args.output_dir / "model.pt"
    deterministic_torch_save(slim_checkpoint, model_path)

    reloaded = torch.load(model_path, map_location="cpu", weights_only=True)
    source_state = checkpoint["model_state_dict"]
    bundled_state = reloaded["model_state_dict"]
    if source_state.keys() != bundled_state.keys():
        raise RuntimeError("Bundled model state keys differ from source")
    unequal = [
        name
        for name in source_state
        if not torch.equal(source_state[name], bundled_state[name])
    ]
    if unequal:
        raise RuntimeError(f"Bundled model changed tensors: {unequal[:5]}")
    if reloaded.get("update") != checkpoint.get("update"):
        raise RuntimeError("Bundled model update metadata changed")

    for name in ("main.py", "policy_runtime.py"):
        source = (args.output_dir / name).read_text(encoding="utf-8")
        compile(source, str(args.output_dir / name), "exec")

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_archive(args.output_dir, args.archive)

    with tarfile.open(args.archive, mode="r:gz") as archive:
        members = tuple(member.name for member in archive.getmembers())
        if members != EXPECTED_MEMBERS:
            raise RuntimeError(
                f"Unexpected archive members: {members}; expected {EXPECTED_MEMBERS}"
            )
        if any(not member.isfile() for member in archive.getmembers()):
            raise RuntimeError("Submission archive contains a non-file member")

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
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_update": checkpoint.get("update"),
        "template_dir": str(args.template_dir.resolve()),
        "template_contract": {
            "path": str(args.template_contract.resolve()),
            "sha256": sha256_file(args.template_contract),
            "schema_version": template_contract.get("schema_version"),
            "template_version": template_contract.get("template_version"),
        },
        "template_sources": {
            name: {
                "path": str(source.resolve()),
                "sha256": sha256_file(source),
            }
            for name, source in source_files.items()
        },
        "action_order_mode": args.action_order_mode,
        "semantic_deck_hash": deck_hash,
        "deployment_contract": (
            {
                "path": str(args.deployment_contract.resolve()),
                "sha256": sha256_file(args.deployment_contract),
                "schema_version": deployment_contract.get("schema_version"),
            }
            if args.deployment_contract is not None
            and deployment_contract is not None
            else None
        ),
        "output_dir": str(args.output_dir.resolve()),
        "archive": str(args.archive.resolve()),
        "archive_sha256": archive_sha256,
        "archive_bytes": args.archive.stat().st_size,
        "members": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in EXPECTED_MEMBERS
        },
        "model_tensor_count": len(bundled_state),
        "model_tensors_equal_source": True,
        "byte_reproducible_packaging": True,
        "archive_metadata": {
            "member_order": list(EXPECTED_MEMBERS),
            "gzip_mtime": 0,
            "member_mtime": 0,
            "member_mode": "0644",
            "uid": 0,
            "gid": 0,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

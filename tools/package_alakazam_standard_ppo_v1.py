#!/usr/bin/env python3
"""Read-only fixed packaging plan for one Alakazam standard PPO candidate.

The generated commands deliberately reuse package_ppo_submission.py and
validate_ppo_submission.py. This script itself never writes a bundle and never
performs an external submission.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_VERSION = "ptcg-ppo-alakazam-standard-pl-v1"
PPO_FEATURE_VERSION = "ptcg-selfplay-ppo-terminal01-v1"
BC_FEATURE_VERSION = "ptcg-bc-orbit-entity-transformer-v5"
EXPECTED_ACTION_DISTRIBUTION = (
    "masked cardinality categorical + ordered Plackett-Luce without replacement"
)
EXPECTED_DECK_HASH = (
    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
)
EXPECTED_CANDIDATE_SHA256 = (
    "bc3bbf185e4f61e4d3adc5dd11241f0f6001be274e90ef42e3f341f22b052393"
)
EXPECTED_BC_SHA256 = (
    "f5500086c16a02c19f3f2abce5e144446bd079248fd9fc3f4a620f3d079c7626"
)
EXPECTED_DECK_FILE_SHA256 = (
    "0598646548d081832ec311c15fdc369b32c6f5e63175b0cfd1904d21fd082451"
)
CANDIDATE_RELATIVE = Path(
    "artifacts/top3_bc77_20260810/alakazam_control/"
    "ppo_terminal01_corrected_v2_soup_u0_u10_v1/bc12p5_u10_87p5.pt"
)
BC_ANCHOR_RELATIVE = Path(
    "artifacts/gold8_recent7_20260808/alakazam_control/specialist_bc/best.pt"
)
DECK_SOURCE_RELATIVE = Path(
    "data/gold8_recent7_20260808/decks/"
    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf.csv"
)
TEMPLATE_RELATIVE = Path(
    "submission_templates/ptcg_ppo_alakazam_standard_pl_v1"
)
EXPECTED_TEMPLATE_FILES = {
    "main.py": "de6c3f354fb797fe34785cc6679c031c1488e207099583e70d2101e39afc8313",
    "deck.csv": EXPECTED_DECK_FILE_SHA256,
    "policy_runtime.py": (
        "fe7182a588962fd9c5e07e9e62433beaa1c5b3fa047bb8170e22435e4f93e997"
    ),
}
LEGACY_MARNIE_FILES = {
    "main.py": "616c06b523ab856faf91c3b51a7fda8f594a91376b61c4fb22ef93447252019f",
    "deck.csv": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
    "policy_runtime.py": (
        "fe7182a588962fd9c5e07e9e62433beaa1c5b3fa047bb8170e22435e4f93e997"
    ),
}
GENERIC_PACKAGER_SHA256 = (
    "7adb37f38cd055999df21794776ec264b58dd5ee703e10057ca91990f0bbf724"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: actual={actual} expected={expected}"
        )


def read_deck(path: Path) -> list[int]:
    deck = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"{path} contains {len(deck)} cards; expected 60")
    return deck


def compute_deck_hash(deck: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(deck))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalized_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    raw = checkpoint.get("model_config")
    if not isinstance(raw, dict):
        raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint has no model config")
    return {
        "hash_size": int(raw["hash_size"]),
        "categorical_dim": int(raw["categorical_dim"]),
        "model_dim": int(raw["model_dim"]),
        "layers": int(raw["layers"]),
        "heads": int(raw["heads"]),
        "dropout": float(raw["dropout"]),
        "max_state_entities": int(raw["max_state_entities"]),
        "entity_fields": int(raw.get("entity_fields", 20)),
        "option_fields": int(raw.get("option_fields", 24)),
    }


def audit_and_plan(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    candidate = repo_root / CANDIDATE_RELATIVE
    bc_anchor = repo_root / BC_ANCHOR_RELATIVE
    deck_source = repo_root / DECK_SOURCE_RELATIVE
    template_dir = repo_root / TEMPLATE_RELATIVE
    contract_path = template_dir / "CONTRACT.json"
    packager = repo_root / "tools/package_ppo_submission.py"
    validator = repo_root / "tools/validate_ppo_submission.py"

    require_sha(candidate, EXPECTED_CANDIDATE_SHA256, "fixed candidate")
    require_sha(bc_anchor, EXPECTED_BC_SHA256, "Alakazam BC anchor")
    require_sha(deck_source, EXPECTED_DECK_FILE_SHA256, "Alakazam deck")
    require_sha(packager, GENERIC_PACKAGER_SHA256, "generic PPO packager")
    if not validator.is_file():
        raise FileNotFoundError(validator)
    for name, expected in EXPECTED_TEMPLATE_FILES.items():
        require_sha(template_dir / name, expected, f"template {name}")
    legacy_dir = repo_root / "submissions/ptcg_ppo_terminal01_v1"
    for name, expected in LEGACY_MARNIE_FILES.items():
        require_sha(
            legacy_dir / name,
            expected,
            f"read-only legacy Marnie {name}",
        )

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("template_version") != TEMPLATE_VERSION:
        raise ValueError("template contract version mismatch")
    if contract.get("template_files") != EXPECTED_TEMPLATE_FILES:
        raise ValueError("template contract hashes mismatch")
    decode = contract.get("decode")
    if decode != {
        "order_mode": "raw",
        "cardinality": "legal_masked_greedy",
        "selection": "greedy_plackett_luce_without_replacement",
        "canonicalize_order": False,
        "sort_selected_indices": False,
        "context_overrides": [],
    }:
        raise ValueError("template decode contract is not standard PPO")

    checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(
        bc_anchor,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("feature_version") != PPO_FEATURE_VERSION:
        raise ValueError("candidate PPO feature version mismatch")
    if checkpoint.get("bc_feature_version") != BC_FEATURE_VERSION:
        raise ValueError("candidate BC feature version mismatch")
    if checkpoint.get("learner_deck_hash") != EXPECTED_DECK_HASH:
        raise ValueError("candidate learner deck hash mismatch")
    if checkpoint.get("action_distribution") != EXPECTED_ACTION_DISTRIBUTION:
        raise ValueError("candidate is not ordered Plackett-Luce PPO")
    if bc_checkpoint.get("feature_version") != BC_FEATURE_VERSION:
        raise ValueError("BC anchor feature version mismatch")
    if normalized_model_config(checkpoint) != normalized_model_config(
        bc_checkpoint
    ):
        raise ValueError("candidate and BC anchor configs differ")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("candidate has no model state")
    count_weight = state.get("count_head.2.weight")
    if not isinstance(count_weight, torch.Tensor):
        raise ValueError("candidate count head is missing")
    if tuple(count_weight.shape) != (61, 128):
        raise ValueError("candidate count head is not the expected 61x128")

    source_deck = read_deck(deck_source)
    template_deck = read_deck(template_dir / "deck.csv")
    if template_deck != source_deck:
        raise ValueError("template deck differs from source deck")
    if compute_deck_hash(template_deck) != EXPECTED_DECK_HASH:
        raise ValueError("template deck hash mismatch")
    compile(
        (template_dir / "main.py").read_text(encoding="utf-8"),
        str(template_dir / "main.py"),
        "exec",
    )

    package_command = [
        sys.executable,
        str(packager.resolve()),
        "--checkpoint",
        str(candidate.resolve()),
        "--template-dir",
        str(template_dir.resolve()),
        "--template-contract",
        str(contract_path.resolve()),
        "--action-order-mode",
        "raw",
        "--output-dir",
        "<OUTPUT_DIR>",
        "--archive",
        "<ARCHIVE>",
        "--manifest",
        "<MANIFEST>",
    ]
    validation_games = int(contract["action_exact_validation"]["games"])
    action_exact_command = [
        sys.executable,
        str(validator.resolve()),
        "--archive",
        "<ARCHIVE>",
        "--checkpoint",
        str(candidate.resolve()),
        "--bc-checkpoint",
        str(bc_anchor.resolve()),
        "--games",
        str(validation_games),
        "--action-order-mode",
        "raw",
    ]
    return {
        "schema_version": "ptcg-alakazam-standard-ppo-package-plan-v1",
        "read_only": True,
        "template_version": TEMPLATE_VERSION,
        "candidate": {
            "path": str(candidate.resolve()),
            "sha256": EXPECTED_CANDIDATE_SHA256,
            "update": checkpoint.get("update"),
        },
        "deck": {
            "path": str(deck_source.resolve()),
            "file_sha256": EXPECTED_DECK_FILE_SHA256,
            "deck_hash": EXPECTED_DECK_HASH,
        },
        "bc_anchor": {
            "path": str(bc_anchor.resolve()),
            "sha256": EXPECTED_BC_SHA256,
        },
        "template_dir": str(template_dir.resolve()),
        "decode": decode,
        "legacy_marnie_template_unchanged": True,
        "package": {
            "implementation": str(packager.resolve()),
            "command": package_command,
            "executed": False,
        },
        "action_exact_validation": {
            "implementation": str(validator.resolve()),
            "command": action_exact_command,
            "games": validation_games,
            "comparison": (
                "packaged agent versus evaluator sample_ordered_actions with "
                "deterministic=True and canonicalize_order=False"
            ),
            "required_action_mismatches": 0,
            "required_invalid_games": 0,
            "executed": False,
            "required_before_submission": True,
        },
        "archive_created": False,
        "external_submission_performed": False,
    }


def main() -> None:
    print(json.dumps(audit_and_plan(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

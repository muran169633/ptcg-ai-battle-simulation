#!/usr/bin/env python3
"""Apply exact checkpoint/deck-hash compatibility to proxy selection.

Validation accuracy can rank only deployable checkpoints.  A checkpoint whose
training config declares another deck hash may be evaluated for diagnostics,
but the PPO engine correctly refuses to bind it to a different 60-card list.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROXY_ROOT = ROOT / "artifacts/top100_proxy_recent14_20260811_v1"
DATA_MANIFEST = ROOT / "data/top100_proxy_recent14_20260811_v1/manifest.json"
INPUT = PROXY_ROOT / "proxy_checkpoint_selection.json"
OUTPUT = PROXY_ROOT / "proxy_checkpoint_selection_compatible.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declared_deck_hashes(checkpoint: Path) -> list[str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config") or {}
    raw = config.get("deck_hashes") or []
    return [str(value) for value in raw]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    data = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    decisions = {}
    for slug, selected in source["selections"].items():
        expected_hash = str(data["profiles"][slug]["deck_hash"])
        candidates = {}
        for label in ("initialization", "refreshed"):
            candidate = selected[label]
            checkpoint = Path(candidate["checkpoint"])
            declared = declared_deck_hashes(checkpoint)
            candidates[label] = candidate | {
                "declared_deck_hashes": declared,
                "exact_deck_compatible": expected_hash in declared,
            }
        eligible = [
            (label, candidate)
            for label, candidate in candidates.items()
            if candidate["exact_deck_compatible"]
        ]
        if not eligible:
            raise RuntimeError(f"No exact-deck-compatible checkpoint for {slug}")
        label, candidate = max(
            eligible,
            key=lambda item: float(item[1]["valid_exact_accuracy"]),
        )
        checkpoint = Path(candidate["checkpoint"])
        decisions[slug] = {
            "expected_deck_hash": expected_hash,
            "selected": label,
            "selected_checkpoint": str(checkpoint.resolve()),
            "selected_checkpoint_sha256": sha256(checkpoint),
            "selected_valid_exact_accuracy": candidate["valid_exact_accuracy"],
            "selected_test_exact_accuracy": candidate["test_exact_accuracy"],
            "candidates": candidates,
            "selection_rule": "highest valid accuracy among exact-deck-compatible checkpoints",
            "test_not_used_for_selection": True,
        }
    output = {
        "schema_version": "ptcg-top100-proxy-compatible-selection-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_selection": {"path": str(INPUT), "sha256": sha256(INPUT)},
        "data_manifest": {"path": str(DATA_MANIFEST), "sha256": sha256(DATA_MANIFEST)},
        "selections": decisions,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

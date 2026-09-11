#!/usr/bin/env python3
"""Pinned Orbit-v5 featurizer for reproducible legacy Parquet conversion.

The frozen July 31 behavior source already contains the exact Orbit-v5 entity
ordering used by the incumbent general BC model.  It was later extended to
emit ``action_sequence`` while retaining the v5 feature-version string.  The
existing v5 Parquet contract predates that field and stores it as null, so this
adapter deliberately removes only that key before Arrow serialization.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


FEATURE_VERSION = "ptcg-bc-orbit-entity-transformer-v5"
_SOURCE_SHA256 = "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "frozen_behavior_source__sha256_"
    "f6c92841f2f6974408be05600b80f532b099a816a40ef2a5b2168fe45dc713ab"
    / "train_bc_orbit.py"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not _SOURCE.is_file():
    raise FileNotFoundError(_SOURCE)
_actual_sha256 = _sha256_file(_SOURCE)
if _actual_sha256 != _SOURCE_SHA256:
    raise RuntimeError(
        "Frozen Orbit-v5 feature source hash drift: "
        f"expected {_SOURCE_SHA256}, got {_actual_sha256}"
    )

_module_name = "_ptcg_frozen_orbit_v5_feature_source"
_spec = importlib.util.spec_from_file_location(_module_name, _SOURCE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load frozen Orbit-v5 source: {_SOURCE}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_module_name] = _module
_spec.loader.exec_module(_module)

if _module.FEATURE_VERSION != FEATURE_VERSION:
    raise RuntimeError(
        "Frozen feature contract mismatch: "
        f"expected {FEATURE_VERSION!r}, got {_module.FEATURE_VERSION!r}"
    )

ENTITY_NUMERIC_SIZE = _module.ENTITY_NUMERIC_SIZE
GLOBAL_NUMERIC_SIZE = _module.GLOBAL_NUMERIC_SIZE
OPTION_NUMERIC_SIZE = _module.OPTION_NUMERIC_SIZE
# ``evaluate_policy_bc`` temporarily raises this limit to the engine/PPO
# contract when it measures long ordered selections.  Export the frozen
# source's value and mirror temporary caller changes into the wrapped module;
# otherwise the adapter exposes a subtly different interface from the source
# it pins.
MAX_ACTION_COUNT = _module.MAX_ACTION_COUNT
resolve_card = _module.resolve_card
collate_decisions = _module.collate_decisions
EntityOptionPolicy = _module.EntityOptionPolicy


def featurize_row(
    row: dict[str, Any], hash_size: int, max_state_entities: int
) -> dict[str, Any] | None:
    """Return the historical v5 record, with sequence data left null."""

    previous_max_action_count = _module.MAX_ACTION_COUNT
    _module.MAX_ACTION_COUNT = MAX_ACTION_COUNT
    try:
        features = _module.featurize_row(row, hash_size, max_state_entities)
    finally:
        _module.MAX_ACTION_COUNT = previous_max_action_count
    if features is None:
        return None
    result = dict(features)
    result.pop("action_sequence", None)
    return result

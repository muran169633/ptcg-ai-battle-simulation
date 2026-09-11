#!/usr/bin/env python3
"""Corrected fail-closed launcher for the frozen U468 exact-P12 design."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_WRAPPER = REPO_ROOT / "tools/run_ppo_special_bc_u468_actorheadonly_exactp12_v1.py"
BASE_WRAPPER_SHA256 = "25bda69bb7cb57fdc0220ee88449054b526fd15249b8c18b08a060e74fc4a443"
DESIGN_PATH = REPO_ROOT / (
    "artifacts/ppo_u464_g8ppo4_u468_exactp12specialbc_"
    "design202608091.design_preregistration_v3.json"
)
DESIGN_SHA256 = "d5d4fecfee5c0c3b278476777a7c5f56db3086d25b37585be6e37840e6ee0e57"


def require_regular_bytes(path: Path, expected: str, label: str) -> bytes:
    current = os.lstat(path)
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise ValueError(f"{label} is not a regular non-symlink file")
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {observed}"
        )
    return raw


def corrected_transform(raw: bytes) -> str:
    source = raw.decode("utf-8")
    unique_replacements = {
        "if tuple(BATCH_INDICES) != tuple(range(32)) or STEPS != 32:": (
            "if tuple(BATCH_INDICES) != "
            "(0, 1, 2, 7, 8, 9, 10, 14, 15, 16, 18, 19) or STEPS != 12:"
        ),
        'integrity["optimizer_steps"] == 32,': (
            'integrity["optimizer_steps"] == 12,'
        ),
        'integrity["rows"] == 8192,': 'integrity["rows"] == 3072,',
        'integrity["context34_rows"] == 32,': (
            'integrity["context34_rows"] == 12,'
        ),
    }
    for old, new in unique_replacements.items():
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"authenticated source transform count for {old!r} is {count}, expected 1"
            )
        source = source.replace(old, new, 1)
    batch_key = '"batch_order_exact_natural_0_to_31"'
    if source.count(batch_key) != 2:
        raise RuntimeError(
            "authenticated source batch-order key count is not exactly 2"
        )
    source = source.replace(batch_key, '"batch_order_exact_historical_p12"')
    source = source.replace("s32eqp12", "exacthistoricalp12")
    source = source.replace("S32", "P12")
    return source


def main() -> int:
    raw = require_regular_bytes(BASE_WRAPPER, BASE_WRAPPER_SHA256, "base wrapper")
    require_regular_bytes(DESIGN_PATH, DESIGN_SHA256, "authoritative v3 design")
    base = types.ModuleType("_ptcg_u468_exactp12_base_wrapper")
    base.__file__ = str(BASE_WRAPPER)
    base.__package__ = None
    exec(compile(raw, str(BASE_WRAPPER), "exec"), base.__dict__)
    base.transformed_executor_source = corrected_transform
    base.DESIGN_PATH = DESIGN_PATH
    base.DESIGN_SHA256 = DESIGN_SHA256
    base.__file__ = str(Path(__file__).resolve())
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())

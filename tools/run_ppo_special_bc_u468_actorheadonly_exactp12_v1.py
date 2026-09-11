#!/usr/bin/env python3
"""Execute the frozen U468 exact-historical-P12 special-BC design.

The audited S32 executor remains immutable.  This wrapper authenticates its
source bytes, applies a small fail-closed transformation for the separately
preregistered P12 constants, executes those transformed bytes as an isolated
module, and overrides only the new design/output bindings.  The inherited
executor still performs its descriptor, optimizer, gradient, publication, and
round-trip integrity checks.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SOURCE_EXECUTOR = REPO_ROOT / "tools/run_ppo_special_bc_u468_actorheadonly_s32eqp12_v3.py"
SOURCE_EXECUTOR_SHA256 = "3c398bcec22f4d379cb8b49ba683b9e2bfb959d21dec483de1c11601e37b40df"
DESIGN_PATH = REPO_ROOT / (
    "artifacts/ppo_u464_g8ppo4_u468_exactp12specialbc_"
    "design202608091.design_preregistration_v2.json"
)
DESIGN_SHA256 = "152a2cab1db5708dfb790f1f3c2730e3c6dc1a7cc85d3ba614324978930b139f"
AUTHORIZATION_PATH = REPO_ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_"
    "design202608090.specialist_retention_decision.json"
)
AUTHORIZATION_SHA256 = "218c407e3b8c4622629b7c5b6bb637c3c23ba134d4ee07cb1af0d3b68342ab9e"

BRANCH = "ppo_u464_g8ppo4_u468_exactp12specialbc_design202608091"
BATCH_INDICES = (0, 1, 2, 7, 8, 9, 10, 14, 15, 16, 18, 19)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_hash(path: Path, expected: str, label: str) -> bytes:
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


def transformed_executor_source(raw: bytes) -> str:
    source = raw.decode("utf-8")
    replacements = {
        "if tuple(BATCH_INDICES) != tuple(range(32)) or STEPS != 32:": (
            "if tuple(BATCH_INDICES) != "
            "(0, 1, 2, 7, 8, 9, 10, 14, 15, 16, 18, 19) or STEPS != 12:"
        ),
        '"batch_order_exact_natural_0_to_31"': (
            '"batch_order_exact_historical_p12"'
        ),
        'integrity["optimizer_steps"] == 32,': (
            'integrity["optimizer_steps"] == 12,'
        ),
        'integrity["rows"] == 8192,': 'integrity["rows"] == 3072,',
        'integrity["context34_rows"] == 32,': (
            'integrity["context34_rows"] == 12,'
        ),
    }
    for old, new in replacements.items():
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"authenticated source transform count for {old!r} is {count}, expected 1"
            )
        source = source.replace(old, new, 1)
    source = source.replace("s32eqp12", "exacthistoricalp12")
    source = source.replace("S32", "P12")
    return source


def main() -> int:
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError("current working directory must be the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("wrapper must use my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("wrapper requires outer Python flags -I -B")
    source_raw = require_regular_hash(
        SOURCE_EXECUTOR, SOURCE_EXECUTOR_SHA256, "source executor"
    )
    require_regular_hash(DESIGN_PATH, DESIGN_SHA256, "authoritative P12 design")
    require_regular_hash(
        AUTHORIZATION_PATH, AUTHORIZATION_SHA256, "design-only authorization"
    )

    runtime = types.ModuleType("_ptcg_u468_exact_historical_p12_runtime")
    runtime.__file__ = str(Path(__file__).resolve())
    runtime.__package__ = None
    runtime.__doc__ = __doc__
    source = transformed_executor_source(source_raw)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), runtime.__dict__)

    runtime.BRANCH = BRANCH
    runtime.DESIGN_PATH = DESIGN_PATH
    runtime.DESIGN_SHA256 = DESIGN_SHA256
    runtime.AUTHORIZATION_PATH = AUTHORIZATION_PATH
    runtime.AUTHORIZATION_SHA256 = AUTHORIZATION_SHA256
    runtime.PREFLIGHT_DIR = REPO_ROOT / f"artifacts/{BRANCH}.p12_preflight_v1"
    runtime.PREFLIGHT_PATH = runtime.PREFLIGHT_DIR / "preflight_audit.json"
    runtime.FORMAL_DIR = REPO_ROOT / f"artifacts/{BRANCH}/special_stage"
    runtime.CHECKPOINT_PATH = runtime.FORMAL_DIR / (
        "special-bc-actorheadonly-pokemonfan-exactp12-0012.pt"
    )
    runtime.MANIFEST_PATH = runtime.FORMAL_DIR / "special_bc_manifest.json"
    runtime.ATTEMPT_MARKER = REPO_ROOT / (
        ".ptcg-u468-exactp12-specialbc-attempt-202608012-202608091.json"
    )
    runtime.STEPS = 12
    runtime.BATCH_INDICES = BATCH_INDICES
    runtime.SPECIAL_LR_SCALE = 0.05
    runtime.SPECIAL_LEARNING_RATE = 3.6e-5 * 0.05
    runtime.L2_MIN = 0.00177912
    runtime.L2_MAX = 0.00266868
    return int(runtime.main())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the authenticated FLG special-BC executor with preregistered hard batches."""

from __future__ import annotations

import hashlib
import importlib.util
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/run_design202608209_balancedppo_flg_twoweek_special_bc.py"
SOURCE_SHA256 = "8a15c2af66d42cccebc445e89f6cb2d78bf38cf563b45012e105225251438d5e"
info = SOURCE.lstat()
if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("authenticated special-BC executor mismatch")
spec = importlib.util.spec_from_file_location("design202608209_hardbatch_base", SOURCE)
if spec is None or spec.loader is None: raise RuntimeError("cannot import special-BC executor")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
base.BATCH_INDICES = (22, 24, 12, 18, 1, 8, 0, 6)
base.OUTPUT_ROOT = ROOT / "artifacts/design202608210_balancedppo_flg_hardbatch_special_bc"
base.PREFLIGHT_ROOT = ROOT / "artifacts/design202608210_balancedppo_flg_hardbatch_special_bc_preflight"

if __name__ == "__main__":
    raise SystemExit(base.main())

#!/usr/bin/env python3
"""Audit or execute the legacy-quota bootstrap revision of alpha0.75 PPO."""

from __future__ import annotations

import hashlib
import importlib.util
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
BASE = TOOLS / "exec_design202608176_alpha075_dualanchor_u477_u478.py"
BASE_SHA256 = "142755830ae80fff6f16513cdd82ae8c564ace875ae2dcffdcbfc1b2ab36485e"

info = BASE.lstat()
if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
    raise RuntimeError("design202608176 launcher is not an authenticated regular file")
if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("design202608176 launcher mismatch")

spec = importlib.util.spec_from_file_location("design202608176_launcher", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load authenticated design202608176 launcher")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.SELF = Path(__file__).resolve()
base.SCHEMA = "ptcg-design202608178-alpha075-dualanchor-u477-u478-v2"
base.SEED = 202608178
base.BOOTSTRAP = ROOT / "artifacts/design202608177_alpha075_fresh_training_bootstrap_v2/alpha075-fresh-bootstrap-legacyquota-u476.pt"
base.BOOTSTRAP_SHA256 = "9537dd73afa79aa1bef6801e390eb9b9fd5ce614c00b45cce91f3254155da10d"
base.BOOTSTRAP_MANIFEST = ROOT / "artifacts/design202608177_alpha075_fresh_training_bootstrap_v2/manifest.json"
base.BOOTSTRAP_MANIFEST_SHA256 = "866f5eacacfdfc3fd94e16b7ca2b464d80089d5d4a4adb06f75480924c9f3de1"
base.OUTPUT_ROOT = ROOT / "artifacts/design202608178_alpha075_dualanchor_ppo2x192_u478_v2"
base.OUTPUT_DIR = base.OUTPUT_ROOT / "B_gold_league/seed-202608178"
base.TERMINAL = base.OUTPUT_DIR / "checkpoints/update-0478.pt"
base.ATTEMPT = ROOT / ".ptcg-design202608178-alpha075-u477-u478-v2-attempt.json"
base.LOG = ROOT / "artifacts/design202608178_alpha075_dualanchor_ppo2x192_u478_v2.log"
base.DEPENDENCIES = dict(base.DEPENDENCIES) | {BASE: BASE_SHA256}


if __name__ == "__main__":
    raise SystemExit(base.main())

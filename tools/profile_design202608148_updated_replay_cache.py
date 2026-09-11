#!/usr/bin/env python3
"""Versioned seed-148 specialization of the audited updated-replay profiler."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
BASE = TOOLS / "profile_design202608147_updated_replay_cache.py"
BASE_SHA256 = "7cd7d3e5751b329aa8cf6b6ce460e7a1497a85b3dc3c561861a1d786bc7224ce"
info = os.lstat(BASE)
if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
    raise RuntimeError("authenticated cache profiler base must be a regular file")
if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("authenticated cache profiler base SHA-256 mismatch")
spec = importlib.util.spec_from_file_location("_design202608148_cache_base", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load authenticated cache profiler base")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.SEED = 202608148
_write_exclusive = base.write_exclusive


def write_versioned(path, payload):
    payload["schema_version"] = "ptcg-design202608148-updated-replay-cache-profile-v1"
    payload["design_id"] = "design202608148"
    _write_exclusive(path, payload)


base.write_exclusive = write_versioned


if __name__ == "__main__":
    raise SystemExit(base.main())

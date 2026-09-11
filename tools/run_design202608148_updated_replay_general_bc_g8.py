#!/usr/bin/env python3
"""Corrected terminal-step specialization for updated-replay G8 at U468."""

from __future__ import annotations

import hashlib
import inspect
import os
import stat
import sys
from pathlib import Path


sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
PREVIOUS = TOOLS / "run_design202608147_updated_replay_general_bc_g8.py"
PREVIOUS_SHA256 = "91241e072386aafdea1df7db03c9da1cf008e9f031b2d019b9bb1df5cc7c8107"
info = os.lstat(PREVIOUS)
if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
    raise RuntimeError("authenticated design147 G8 runner must be a regular file")
if hashlib.sha256(PREVIOUS.read_bytes()).hexdigest() != PREVIOUS_SHA256:
    raise RuntimeError("authenticated design147 G8 runner SHA-256 mismatch")

import run_design202608147_updated_replay_general_bc_g8 as previous  # noqa: E402


base = previous.base
base.__file__ = str(Path(__file__).resolve())
base.DESIGN_ID = 202608148
base.G8_SEED = 202608148
base.GENERAL_CACHE_SHA256 = "896521dfa6ed5a772796f758182019206dd1f34399e2374aef2080ca9b563917"

# The authenticated historical base has exactly two U464-specific literals:
# it assumes replay AdamW starts at step 16 and therefore hardcodes terminal
# step 24.  U468 starts at step 24, so the general invariant is parent + G8.
source = inspect.getsource(base.main)
terminal_check = 'state_steps(replay_after) != {24}'
terminal_record = '"replay_step_after": 24'
if source.count(terminal_check) != 1 or source.count(terminal_record) != 1:
    raise RuntimeError("authenticated G8 terminal-step patch points drifted")
source = source.replace(
    terminal_check,
    "state_steps(replay_after) != {PARENT_REPLAY_STEP + G8_STEPS}",
)
source = source.replace(
    terminal_record,
    '"replay_step_after": PARENT_REPLAY_STEP + G8_STEPS',
)
exec(compile(source, str(Path(__file__).resolve()), "exec"), base.__dict__)


if __name__ == "__main__":
    raise SystemExit(base.main())

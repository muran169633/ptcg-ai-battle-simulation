#!/usr/bin/env python3
"""Authenticated reduced-step-size specialization of design202608148 S8."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
BASE = ROOT / "tools/run_design202608148_u472_special_s8.py"
BASE_SHA = "3cc362b79682f5b3e8f67be22204155ecbeaa56708be5b55e4595ce5edb56520"
POSTMORTEM = ROOT / "artifacts/design202608148_u472_special_s8_failure_postmortem.json"
POSTMORTEM_SHA = "82f86cbd8513dbba1ff288841e6f7eccc1d0e5f3b0ffbca740986c15153bddfd"


def authenticate(path: Path, expected: str) -> None:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"unsafe authenticated input: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"authenticated input hash drift: {path}")


authenticate(BASE, BASE_SHA)
authenticate(POSTMORTEM, POSTMORTEM_SHA)
source = BASE.read_text(encoding="utf-8")
replacements = {
    'SCHEMA = "ptcg-design202608148-u472-special-s8-execution-v1"':
        'SCHEMA = "ptcg-design202608149-u472-special-s8-execution-v1"',
    "LR = 1e-6": "LR = 6.75e-7",
    'output.parent != ROOT / "artifacts/ppo_guardedu468_updatedreplay_g8_ppo4x96_u472_freshjointactor6_s8_design202608148"':
        'output.parent != ROOT / "artifacts"',
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise RuntimeError(f"authenticated base patch signature drift: {old}")
    source = source.replace(old, new)
if "LR = 1e-6" in source or "ptcg-design202608148-u472-special-s8-execution-v1" in source:
    raise RuntimeError("design149 patch was incomplete")
namespace = {"__name__": "__main__", "__file__": str(BASE)}
exec(compile(source, str(BASE), "exec"), namespace)

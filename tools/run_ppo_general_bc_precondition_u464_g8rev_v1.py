#!/usr/bin/env python3
"""Apply the preregistered reverse-order G8 general-BC preconditioner.

This is intentionally a thin, hashable specialization of the audited G8
implementation.  It keeps the parent, data, optimizer state, learning rate,
and eight selected batches fixed, changing only their application order.  The
result therefore preserves the complete PPO/quota state while moving the
learner into a different deterministic policy/optimizer cell before PPO.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_ppo_general_bc_precondition_u464_g8 as base  # noqa: E402


base.__file__ = str(Path(__file__).resolve())
base.DESIGN_ID = 202608120
base.G8_SEED = 202608013
base.G8_BATCH_INDICES = (11, 8, 7, 6, 5, 4, 3, 1)


if __name__ == "__main__":
    raise SystemExit(base.main())

#!/usr/bin/env python3
"""Time-forward Pokemon-Fan specialization of the frozen U456 BC executor."""

from __future__ import annotations

import run_ppo_special_bc as base


# The clean time-forward specialist train split has 9,487 rows.  Bind a cache
# that fits wholly inside that split while retaining the original 16-step
# actor-only update and all other integrity checks from the base executor.
base.SPECIAL_CACHE_BATCHES = 32


if __name__ == "__main__":
    base.main()

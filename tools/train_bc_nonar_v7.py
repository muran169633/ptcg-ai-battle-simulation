#!/usr/bin/env python3
"""Train the integrated non-autoregressive V7 PTCG BC policy."""

from __future__ import annotations

import bc_nonar_v7
import train_bc_orbit


if __name__ == "__main__":
    bc_nonar_v7.install_into_trainer()
    train_bc_orbit.main()

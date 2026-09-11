#!/usr/bin/env python3
"""Explicit loader for the opt-in, local deterministic PTCG engine.

This module never replaces ``cg.sim.lib`` and is not imported by submission or
training code. Callers must provide the path to a separately built local
library, making deterministic-engine use deliberate and auditable.
"""

from __future__ import annotations

import ctypes
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int),
    ]


def load_seeded_engine(path: str | Path) -> ctypes.CDLL:
    """Load and initialize one explicit local seeded-engine build."""

    resolved = Path(path).expanduser().resolve(strict=True)
    return _load_seeded_engine(str(resolved))


@lru_cache(maxsize=None)
def _load_seeded_engine(resolved_path: str) -> ctypes.CDLL:
    """Initialize each normalized shared-library path at most once."""

    engine = ctypes.CDLL(resolved_path)

    engine.GameInitialize.restype = None
    engine.GameInitialize.argtypes = []
    engine.BattleStartSeeded.restype = StartData
    engine.BattleStartSeeded.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint32,
    ]
    engine.BattleFinish.restype = None
    engine.BattleFinish.argtypes = [ctypes.c_void_p]
    engine.GetBattleData.restype = SerialData
    engine.GetBattleData.argtypes = [ctypes.c_void_p]
    engine.Select.restype = ctypes.c_int
    engine.Select.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]

    engine.GameInitialize()
    return engine


class SeededRawBattle:
    """One local-engine battle with an explicit uint32 seed."""

    def __init__(
        self,
        engine_path: str | Path,
        deck0: list[int],
        deck1: list[int],
        seed: int,
    ) -> None:
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("Each deck must contain exactly 60 cards")
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed must be in the uint32 range")

        self.engine = load_seeded_engine(engine_path)
        cards = deck0 + deck1
        argument = (ctypes.c_int * len(cards))(*cards)
        start = self.engine.BattleStartSeeded(argument, seed)
        self.ptr = start.battlePtr
        self.closed = False
        if not self.ptr:
            raise ValueError(
                "BattleStartSeeded failed: "
                f"player={start.errorPlayer} type={start.errorType}"
            )
        self.observation = self._get_observation()

    def _get_observation(self) -> dict[str, Any]:
        serial = self.engine.GetBattleData(self.ptr)
        observation = json.loads(serial.json.decode())
        observation["search_begin_input"] = ctypes.string_at(
            serial.data,
            serial.count,
        ).decode("ascii")
        return observation

    @property
    def result(self) -> int:
        return int((self.observation.get("current") or {}).get("result", -1))

    def step(self, action: list[int]) -> tuple[dict[str, Any], int]:
        argument = (ctypes.c_int * len(action))(*action)
        error = int(self.engine.Select(self.ptr, argument, len(action)))
        if error == 0:
            self.observation = self._get_observation()
        return self.observation, error

    def close(self) -> None:
        if not self.closed and self.ptr:
            self.engine.BattleFinish(self.ptr)
            self.ptr = None
            self.closed = True

    def __enter__(self) -> SeededRawBattle:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

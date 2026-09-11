#!/usr/bin/env python3
"""CW24 v17 FP32 structural absolute-QP train-only protocol.

Production wiring is implemented but remains deliberately unarmed while this
source is writable.  It may be armed only by an independent review followed by
an exact mode-0555 freeze.  Audit and solver-unit modes import no torch and
perform no writes.

Locked design
-------------

* reproduce frozen v12 steps 1..29 byte-for-byte;
* allow at most 32 stage-2 native-BF16 trial forwards (29 + 32 = 61);
* form every structural margin and VJP with a real autocast-disabled FP32
  ``model(batch)`` forward, never by casting already-BF16 logits;
* keep append-only current-point cuts in absolute actor6 coordinates relative
  to CW11 and solve the unique minimum-norm cutting-plane QP;
* move toward that absolute target by at most 2.5e-5, with a .000999 safe
  target radius and the frozen v15 quantization-safe writeback projector;
* use the general-BC/PPO/special-BC scalar only as a binding-nullspace
  secondary direction capped at one quarter of the trust radius;
* evaluate exactly one native-BF16 endpoint per stage-2 trial, stop at the
  first complete native hard pass, and expose no actor payload on NO_GO.

The module imports neither torch nor any production module in audit/unit mode.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import struct
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_cw11_fp32_absolute_qp_specialbc_cw24_v17.py"
OUTPUT = ROOT / "artifacts/cw24_cw11_fp32_absolute_qp_specialbc_trainonly_v17.json"
ATTEMPT_MARKER = ROOT / "artifacts/.ptcg-cw24-cw11-fp32-absolute-qp-v17-attempt.json"
FAILURE = ROOT / "artifacts/cw24_cw11_fp32_absolute_qp_specialbc_failure_v17.json"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-cw11-fp32-absolute-qp-specialbc-cw24-v17"
PROTOCOL_ID = 202608053
STAGE1_EXECUTION_SEED = 202608052

STAGE1_REFERENCE_STEPS = 29
STAGE2_MAX_TRIALS = 32
TOTAL_CHANGED_TRAIN_SHADOW_CAP = 61
SAFE_TARGET_RADIUS = 0.000999
HARD_ACTUAL_RADIUS = 0.00099998
TRUST_RADIUS = 2.5e-5
SECONDARY_NORM_FRACTION = 0.25
PINV_RCOND = 1.0e-12
DUAL_TOL = 1.0e-12
KKT_TOL = 1.0e-9
TIE_TOL = 1.0e-15
FP32_PROGRESS_TOL = 1.0e-12
EXPECTED_SCIPY_VERSION = "1.16.3"
DUAL_WARM_SUPPORT_TOL = 1.0e-10
DUAL_WARM_MAXITER = 20_000
DUAL_WARM_MAXFUN = 200_000
DUAL_WARM_MAXLS = 100
SAFE_RADIUS_SEQUENCE = (0.000999, 0.000995, 0.000980)
FLOAT32_WRITEBACK_STEP_TOL = 2.0e-8
MAX_DYNAMIC_PAIRS = 64
TERMINAL_NO_NEW_CUT_REASONS = frozenset(
    {
        "NONPAIR_HARD_GATE_FAILURE_NO_LEGAL_STRUCTURAL_CUT",
        "LEGAL_STRUCTURAL_CUT_COLLISION_NO_NEW_GEOMETRY",
        "UNLINEARIZABLE_PAIR_FAILURE_NO_LEGAL_STRUCTURAL_CUT",
    }
)

PF7_TARGET_KIND = "PF7_positive_step29_local_q"
PF0_TARGET_KIND = "PF0_terminal_positive_local_q"
ZERO_TARGET_KIND = "zero_native_floor"
DYNAMIC_TARGET_KIND = "sticky_dynamic_stored_threshold"
HISTORICAL_TARGET_KIND = "preserve_CW11_native_margin"

PF7_LINE_SHA256 = "d35040093be24508b97bbfb4a9f0ea98d0f2ef2d7f6af1c9d40119d31de8f123"
PF0A_LINE_SHA256 = "92b3f535ff3ac2c1aa4e95a3ba1c71fc23fba4e3aa0a93f32ce4e2c4ea180379"
PF0B_LINE_SHA256 = "c691171c4e7093f498f29fda034e456a52e277c1d27defffc886217f08ca935b"
RETENTION_0142_SHA256 = "0142a2a3d5bdf5a9a7edc19761db708549701625055a07a4e90013884c752f43"
RETENTION_243F_SHA256 = "243f4a21d6c0b178a6306bdb63ac738aeb5bb13f6be0ccfaa68eebe02eac2d99"
TOP1_FDE6_SHA256 = "fde6fab074495c238d4c4ef42a2f12ec595da78e35836578f7a3828002037b88"
HISTORICAL_PAIR_CONTRACT: tuple[tuple[str, int, int, int, float], ...] = (
    (RETENTION_0142_SHA256, 100, 3, 2, 0.0078125),
    (RETENTION_243F_SHA256, 228, 4, 2, 0.0078125),
    (TOP1_FDE6_SHA256, 271, 0, 2, 0.0),
)
V13_SOURCE = TOOLS / "probe_u468_cw11_two_stage_pf7_specialbc_cw24_v13.py"
V13_SOURCE_SHA256 = "a99762d1a2c98b0521199d845d3f041fff40b3787ff3ceda0fd8207e1c6d0ff7"
V15_SOURCE = TOOLS / "probe_u468_cw11_quant_safe_projected_specialbc_cw24_v15.py"
V15_SOURCE_SHA256 = "95dbbd243c82812c8fd85d8958593f1f143eb229d7599361c9dcaa72d25589f1"
V13_HANDOFF_NEEDLE = '''                        "stage2_initial_loss": loss_audit,
                    }
            else:
'''
V13_HANDOFF_REPLACEMENT = '''                        "stage2_initial_loss": loss_audit,
                    }
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = None
                    outputs = None
                    return cw24_v17_stage2_handoff(dict(locals()))
            else:
'''

FROZEN_INPUTS: tuple[tuple[str, Path, str, int], ...] = (
    (
        "frozen_v16_source",
        TOOLS / "probe_u468_cw11_typed_contract_fix_specialbc_cw24_v16.py",
        "f2902c04f2dea665d0c6a17e98c854cb49f328ff34c3394fbe3dcd446c134c83",
        0o555,
    ),
    (
        "frozen_v16_result",
        ROOT / "artifacts/cw24_cw11_typed_contract_fix_specialbc_trainonly_v16.json",
        "5d02eb10fe8f0a6909042ce2b4794a6369eba83ae83be09328e43a90d4c337a2",
        0o444,
    ),
    (
        "frozen_v16_consumed_attempt",
        ROOT / "artifacts/.ptcg-cw24-cw11-typed-contract-fix-specialbc-v16-attempt.json",
        "fe2885d61be9cde34730a8205c91d0d84fe0cd05e62deffbc10beca4aff724fa",
        0o444,
    ),
    (
        "frozen_v15_source",
        TOOLS / "probe_u468_cw11_quant_safe_projected_specialbc_cw24_v15.py",
        "95dbbd243c82812c8fd85d8958593f1f143eb229d7599361c9dcaa72d25589f1",
        0o555,
    ),
    (
        "frozen_v15_consumed_attempt",
        ROOT / "artifacts/.ptcg-cw24-cw11-quant-safe-projected-specialbc-v15-attempt.json",
        "7b58d270b3d324bfddd7af92f01a72f212ab9a9d5434d1b3538f90c793ae2081",
        0o444,
    ),
    (
        "frozen_v13_source",
        TOOLS / "probe_u468_cw11_two_stage_pf7_specialbc_cw24_v13.py",
        "a99762d1a2c98b0521199d845d3f041fff40b3787ff3ceda0fd8207e1c6d0ff7",
        0o555,
    ),
    (
        "frozen_v13_consumed_attempt",
        ROOT / "artifacts/.ptcg-cw24-cw11-two-stage-pf7-specialbc-v13-attempt.json",
        "924e4741a407351246e25c19e476e74e7cd504886403114c540638f72c00316d",
        0o444,
    ),
    (
        "frozen_v12_source",
        TOOLS / "probe_u468_cw11_projected_nonlinear_specialbc_cw24_v12.py",
        "003e476325a08ff5be400167fccbd43705fac9a4071d806101ddb02969ea2301",
        0o555,
    ),
    (
        "frozen_v12_result",
        ROOT / "artifacts/cw24_cw11_projected_nonlinear_specialbc_trainonly_v12.json",
        "effee98dc03c6d505f7bc2007dae88e28190c0fbba80f22c7902363592b554dc",
        0o444,
    ),
    (
        "frozen_v1_source",
        TOOLS / "probe_u468_cw11_two_stage_top1_specialbc_cw24_v1.py",
        "a3c96eed95a6b558c61303e2190eaa4f353970add4509e8770c3dfcc1621db39",
        0o555,
    ),
    (
        "frozen_CW22_source",
        TOOLS / "probe_u468_cw11_targeted_b256_pcgrad_specialbc_cw22_v1.py",
        "f307f3f26355125f97c857f14b97a73d9a7a4f2ea39ca8b0a578e801055e82b4",
        0o555,
    ),
    (
        "frozen_CW23_source",
        TOOLS / "probe_u468_cw11_boundary_qp_specialbc_cw23_v1.py",
        "4a2d209af71a7ea5d955bd373f1fbada936688fc5bc597615c41192dc360b4b8",
        0o555,
    ),
    (
        "frozen_plain_SGD_reference",
        TOOLS / "run_u468_raw_actor6_equalblend_sgd512_shadow.py",
        "e04f7b7579ef42d0c6f643db833779fb837ae86e3205d948b5564d08f8b30f8e",
        0o555,
    ),
    (
        "frozen_dynamic_pair_reference",
        TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v1.py",
        "9c5a7377d82b18e437ca89d064ce8be60554646573e3aca27f4e40f4989ab77c",
        0o555,
    ),
    (
        "frozen_v1_result",
        ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v1.json",
        "9e3be911bfba95338538f315d47d3f7ce9f598e830ace607332d46697cb0cda5",
        0o444,
    ),
    (
        "frozen_v2_result",
        ROOT / "artifacts/cw24_cw11_two_stage_top1_specialbc_trainonly_v2.json",
        "36cae87fbaf4947f1caf1d6229a9d8ade75044eb25bc33398b731fd429d9e465",
        0o444,
    ),
    (
        "frozen_B352_selection",
        ROOT / "artifacts/cw24_top1_b352_selection_v1.json",
        "1f72b26eccd436ca0d341f837ec42949f5823bf7f0be73461aaeab28166cc222",
        0o444,
    ),
    (
        "FP32_direct_forward_reference",
        TOOLS / "probe_u468_raw_direct512_actor6_multiobjective.py",
        "92e6c599da44ac42a2a497280b0db86e78fc5cd91a40423c9447b4f29c177119",
        0o555,
    ),
    (
        "frozen_v4_historical_cut_source",
        TOOLS / "probe_u468_cw11_native_cuttingplane_specialbc_cw24_v4.py",
        "134eebec7b1e54b68bd47256fde5c8b80dea3cd8d8ea4399428ec4c1de7d1454",
        0o555,
    ),
    (
        "frozen_v4_historical_cut_result",
        ROOT / "artifacts/cw24_cw11_native_cuttingplane_specialbc_trainonly_v4.json",
        "db0c2c356fac5779e14698e3ff3ee9b36967f2aa6ddecee55531bb37e8b3377c",
        0o444,
    ),
)


class ProtocolError(RuntimeError):
    """Fail-closed v17 protocol error."""


class QPInfeasibleError(ProtocolError):
    """Certified infeasible append-only halfspace system, safe for NO_GO."""

    def __init__(self, message: str, certificate: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.certificate = dict(certificate)


def canonical_json(value: Any) -> bytes:
    def check(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ProtocolError("nonfinite JSON value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ProtocolError("non-string JSON key")
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float64_sha(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def path_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    return False


def immutable_file_evidence(
    label: str, path: Path, expected_sha256: str, expected_mode: int
) -> dict[str, Any]:
    before = path.lstat()
    digest = sha256_file(path)
    after = path.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
        "sha256_exact": digest == expected_sha256,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"immutable input drift for {label}: {checks}")
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": int(after.st_size),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


@dataclass(frozen=True, order=True)
class CutKey:
    """Canonical append-only cut identity."""

    actor_sha256: str
    line_sha256: str
    positive_option: int
    negative_option: int
    target_float64_le_sha256: str
    linearization_version: str

    def as_list(self) -> list[Any]:
        return [
            self.actor_sha256,
            self.line_sha256,
            self.positive_option,
            self.negative_option,
            self.target_float64_le_sha256,
            self.linearization_version,
        ]


def canonical_target(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError("cut target is nonfinite")
    return 0.0 if result == 0.0 else result


def target_identity(value: float) -> dict[str, str]:
    target = canonical_target(value)
    packed = struct.pack("<d", target)
    return {
        "float_hex": target.hex(),
        "float64_le_sha256": hashlib.sha256(packed).hexdigest(),
    }


def make_cut_key(
    *,
    actor_sha256: str,
    line_sha256: str,
    positive_option: int,
    negative_option: int,
    target_kind: str,
    target: float,
    linearization_version: str = "fp32_direct_model_v1",
) -> CutKey:
    identity = target_identity(target)
    return CutKey(
        actor_sha256=str(actor_sha256),
        line_sha256=str(line_sha256),
        positive_option=int(positive_option),
        negative_option=int(negative_option),
        target_float64_le_sha256=identity["float64_le_sha256"],
        linearization_version=str(linearization_version),
    )


@dataclass(frozen=True)
class NormalizedCut:
    key: CutKey
    normal: Any
    rhs: float
    target: float
    current_margin: float
    gradient_norm: float
    target_kinds: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()


def normalize_absolute_cut(
    *,
    key: CutKey,
    current_delta: Any,
    current_margin: float,
    gradient: Any,
    target: float,
    target_kind: str = "unspecified",
    origins: Sequence[str] = (),
) -> NormalizedCut:
    """Convert ``m + g·(x-d) >= b`` to normalized ``a·x >= c``."""

    import numpy as np

    d = np.asarray(current_delta, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    if d.ndim != 1 or g.shape != d.shape:
        raise ProtocolError("cut vector shape mismatch")
    if not np.isfinite(d).all() or not np.isfinite(g).all():
        raise ProtocolError("cut vector is nonfinite")
    if not math.isfinite(current_margin) or not math.isfinite(target):
        raise ProtocolError("cut scalar is nonfinite")
    target = canonical_target(target)
    expected_target_identity = target_identity(target)["float64_le_sha256"]
    if key.target_float64_le_sha256 != expected_target_identity:
        raise ProtocolError("cut key is not bound to its exact float64 target")
    norm = float(np.linalg.norm(g))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ProtocolError("cut gradient is zero/nonfinite")
    normal = np.asarray(g / norm, dtype=np.float64)
    rhs = float((target - current_margin + float(g @ d)) / norm)
    if not math.isfinite(rhs):
        raise ProtocolError("normalized cut rhs is nonfinite")
    if not math.isclose(float(np.linalg.norm(normal)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ProtocolError("normalized cut does not have unit norm")
    return NormalizedCut(
        key=key,
        normal=normal,
        rhs=rhs,
        target=float(target),
        current_margin=float(current_margin),
        gradient_norm=norm,
        target_kinds=(str(target_kind),),
        origins=tuple(sorted(set(str(value) for value in origins))),
    )


@dataclass
class AppendOnlyCutLedger:
    _cuts: dict[CutKey, NormalizedCut] = field(default_factory=dict)

    def add(self, cut: NormalizedCut) -> None:
        existing = self._cuts.get(cut.key)
        if existing is not None:
            same = (
                existing.rhs == cut.rhs
                and existing.target == cut.target
                and existing.current_margin == cut.current_margin
                and existing.gradient_norm == cut.gradient_norm
                and float64_sha(existing.normal) == float64_sha(cut.normal)
            )
            if not same:
                raise ProtocolError("same cut key was rebound to different geometry")
            merged_kinds = tuple(
                sorted(set(existing.target_kinds) | set(cut.target_kinds))
            )
            merged_origins = tuple(
                sorted(set(existing.origins) | set(cut.origins))
            )
            if (
                merged_kinds != existing.target_kinds
                or merged_origins != existing.origins
            ):
                self._cuts[cut.key] = NormalizedCut(
                    key=existing.key,
                    normal=existing.normal,
                    rhs=existing.rhs,
                    target=existing.target,
                    current_margin=existing.current_margin,
                    gradient_norm=existing.gradient_norm,
                    target_kinds=merged_kinds,
                    origins=merged_origins,
                )
            return
        self._cuts[cut.key] = cut

    def extend(self, cuts: Iterable[NormalizedCut]) -> None:
        for cut in cuts:
            self.add(cut)

    def ordered(self) -> list[NormalizedCut]:
        return [self._cuts[key] for key in sorted(self._cuts)]

    def __len__(self) -> int:
        return len(self._cuts)


def _canonical_tied_index(
    values: Any,
    indices: Sequence[int],
    keys: Sequence[CutKey],
) -> int:
    """Choose the smallest value, resolving numerical ties by CutKey."""

    minimum = min(float(values[index]) for index in indices)
    tied = [
        index
        for index in indices
        if abs(float(values[index]) - minimum) <= TIE_TOL
    ]
    return min(tied, key=lambda index: keys[index])


def classify_halfspace_feasibility_rowspace(
    matrix: Any,
    rhs: Any,
) -> dict[str, Any]:
    """Classify ``A x >= b`` using only ``m`` row-space variables.

    Every ``x`` has the same constraint values as its projection into the row
    space of ``A``.  Writing that projection as ``x=A.T z`` yields the exactly
    equivalent feasibility system ``(A A.T) z >= b`` even when ``A`` is rank
    deficient.  This keeps the exceptional HiGHS fallback at cut dimension
    ``m`` rather than actor6 dimension 65,793.
    """

    import numpy as np
    import scipy
    from scipy.optimize import linprog

    if scipy.__version__ != EXPECTED_SCIPY_VERSION:
        raise ProtocolError(
            f"SciPy version drift: {scipy.__version__} != {EXPECTED_SCIPY_VERSION}"
        )
    value = np.asarray(matrix, dtype=np.float64)
    bound = np.asarray(rhs, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0:
        raise ProtocolError("row-space feasibility matrix must be nonempty 2D")
    row_count, original_dimension = value.shape
    if bound.shape != (row_count,):
        raise ProtocolError("row-space feasibility rhs shape drift")
    if not np.isfinite(value).all() or not np.isfinite(bound).all():
        raise ProtocolError("row-space feasibility input is nonfinite")
    gram = np.asarray(value @ value.T, dtype=np.float64)
    feasibility = linprog(
        np.zeros(row_count, dtype=np.float64),
        A_ub=-gram,
        b_ub=-bound,
        bounds=[(None, None)] * row_count,
        method="highs",
    )
    status = int(feasibility.status)
    if bool(feasibility.success):
        if status != 0:
            raise ProtocolError("successful row-space HiGHS status is not zero")
        coefficients = np.asarray(feasibility.x, dtype=np.float64)
        if coefficients.shape != (row_count,) or not np.isfinite(
            coefficients
        ).all():
            raise ProtocolError("row-space HiGHS returned malformed coefficients")
        minimum_residual = float(np.min(gram @ coefficients - bound))
        if minimum_residual < -1.0e-7:
            raise ProtocolError(
                "row-space HiGHS feasible solution failed residual verification"
            )
        classification = "feasible"
    elif status == 2:
        minimum_residual = None
        classification = "infeasible"
    else:
        minimum_residual = None
        classification = "indeterminate"
    return {
        "classification": classification,
        "status": status,
        "success": bool(feasibility.success),
        "solver": "scipy.optimize.linprog/HiGHS-row-space",
        "scipy_version": scipy.__version__,
        "method": "highs",
        "row_variable_count": row_count,
        "original_variable_count": original_dimension,
        "gram_shape": [row_count, row_count],
        "matrix_float64_le_sha256": float64_sha(value),
        "rhs_float64_le_sha256": float64_sha(bound),
        "gram_float64_le_sha256": float64_sha(gram),
        "minimum_verified_residual": minimum_residual,
    }


def solve_minimum_norm_active_set(
    cuts: Sequence[NormalizedCut],
    *,
    pinv_rcond: float = PINV_RCOND,
    dual_tol: float = DUAL_TOL,
    kkt_tol: float = KKT_TOL,
) -> tuple[Any, dict[str, Any]]:
    """Solve ``min .5||x||^2`` subject to append-only linear cuts.

    A frozen dual L-BFGS-B call is used only to identify a likely support.  Its
    status and approximate values are never accepted as a certificate.  A
    deterministic pseudoinverse active-set polish then adds primal violations,
    drops negative multipliers, and independently proves primal/dual KKT.  If
    an intermediate equality set is rank-inconsistent, a canonical one-delete
    lookahead shrinks it; this avoids confusing a temporarily overfull working
    set with a globally infeasible halfspace system.
    """

    import numpy as np
    import scipy
    from scipy.optimize import minimize

    if scipy.__version__ != EXPECTED_SCIPY_VERSION:
        raise ProtocolError(
            f"SciPy version drift: {scipy.__version__} != {EXPECTED_SCIPY_VERSION}"
        )

    ordered = sorted(cuts, key=lambda cut: cut.key)
    if not ordered:
        raise ProtocolError("absolute QP requires at least one cut")
    dimension = int(np.asarray(ordered[0].normal).size)
    if dimension <= 0:
        raise ProtocolError("absolute QP dimension is empty")
    matrix = np.stack(
        [np.asarray(cut.normal, dtype=np.float64) for cut in ordered], axis=0
    )
    rhs = np.asarray([cut.rhs for cut in ordered], dtype=np.float64)
    keys = [cut.key for cut in ordered]
    if matrix.shape != (len(ordered), dimension):
        raise ProtocolError("absolute QP matrix shape drift")
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ProtocolError("absolute QP inputs are nonfinite")
    row_norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(row_norms, 1.0, rtol=0.0, atol=1e-12):
        raise ProtocolError("absolute QP received a non-normalized cut")

    full_gram = matrix @ matrix.T

    def dual_value(value: Any) -> float:
        candidate = np.asarray(value, dtype=np.float64)
        return float(0.5 * candidate @ (full_gram @ candidate) - rhs @ candidate)

    def dual_gradient(value: Any) -> Any:
        candidate = np.asarray(value, dtype=np.float64)
        return np.asarray(full_gram @ candidate - rhs, dtype=np.float64)

    warm = minimize(
        dual_value,
        np.zeros(len(ordered), dtype=np.float64),
        method="L-BFGS-B",
        jac=dual_gradient,
        bounds=[(0.0, None)] * len(ordered),
        options={
            "ftol": 1.0e-15,
            "gtol": 1.0e-12,
            "maxiter": DUAL_WARM_MAXITER,
            "maxfun": DUAL_WARM_MAXFUN,
            "maxls": DUAL_WARM_MAXLS,
            "maxcor": min(50, len(ordered)),
        },
    )
    warm_lambda = np.asarray(warm.x, dtype=np.float64)
    warm_residual = dual_gradient(warm_lambda)
    if (
        warm_lambda.shape != (len(ordered),)
        or not np.isfinite(warm_lambda).all()
        or not np.isfinite(warm_residual).all()
        or not math.isfinite(float(warm.fun))
    ):
        raise ProtocolError("dual warm start is nonfinite or malformed")
    if float(np.min(warm_lambda)) < -dual_tol:
        raise ProtocolError("dual warm start violated its nonnegative bounds")
    if not bool(warm.success):
        # Classify only a solver-certified infeasible halfspace system as a
        # structured train-only NO_GO.  A warm-start failure on a feasible
        # system, or an indeterminate HiGHS status, remains a fatal protocol
        # error and must never be mislabeled as expected infeasibility.
        feasibility = classify_halfspace_feasibility_rowspace(
            matrix,
            rhs,
        )
        if feasibility["classification"] == "infeasible":
            raise QPInfeasibleError(
                "append-only absolute halfspace system certified infeasible by "
                "row-space HiGHS",
                feasibility,
            )
        if feasibility["classification"] == "feasible":
            raise ProtocolError(
                "dual warm start failed although row-space HiGHS certified "
                "primal feasibility"
            )
        raise ProtocolError(
            "dual warm start failed and row-space HiGHS feasibility "
            "classification was indeterminate"
        )
    warm_lambda[(warm_lambda < 0.0) & (warm_lambda >= -dual_tol)] = 0.0

    support_threshold = DUAL_WARM_SUPPORT_TOL
    active = [
        index
        for index, value in enumerate(warm_lambda)
        if float(value) > support_threshold
    ]
    active.sort(key=lambda index: keys[index])
    trace: list[dict[str, Any]] = []
    visited: set[tuple[int, ...]] = set()
    x = np.zeros(dimension, dtype=np.float64)
    active_lambda = np.zeros(0, dtype=np.float64)
    max_iterations = 20 * len(ordered) * len(ordered) + 50

    for iteration in range(max_iterations):
        state = tuple(active)
        if state in visited:
            raise ProtocolError("dual active-set polish cycled")
        visited.add(state)

        if active:
            active_matrix = matrix[active]
            active_rhs = rhs[active]
            gram = active_matrix @ active_matrix.T
            active_lambda = np.asarray(
                np.linalg.pinv(gram, rcond=pinv_rcond, hermitian=True) @ active_rhs,
                dtype=np.float64,
            )
            x = np.asarray(active_matrix.T @ active_lambda, dtype=np.float64)
            equality_residual = active_matrix @ x - active_rhs
            equality_linf = float(np.max(np.abs(equality_residual)))
            if equality_linf > kkt_tol:
                removal_candidates: list[dict[str, Any]] = []
                for local_remove, removed_global in enumerate(active):
                    reduced = [
                        global_index
                        for local, global_index in enumerate(active)
                        if local != local_remove
                    ]
                    if reduced:
                        reduced_matrix = matrix[reduced]
                        reduced_rhs = rhs[reduced]
                        reduced_gram = reduced_matrix @ reduced_matrix.T
                        reduced_lambda = np.asarray(
                            np.linalg.pinv(
                                reduced_gram,
                                rcond=pinv_rcond,
                                hermitian=True,
                            )
                            @ reduced_rhs,
                            dtype=np.float64,
                        )
                        reduced_x = np.asarray(
                            reduced_matrix.T @ reduced_lambda,
                            dtype=np.float64,
                        )
                        reduced_equality_linf = float(
                            np.max(
                                np.abs(reduced_matrix @ reduced_x - reduced_rhs)
                            )
                        )
                        reduced_dual_min = float(np.min(reduced_lambda))
                        reduced_dual_objective = float(
                            reduced_rhs @ reduced_lambda
                            - 0.5
                            * reduced_lambda
                            @ (reduced_gram @ reduced_lambda)
                        )
                    else:
                        reduced_lambda = np.zeros(0, dtype=np.float64)
                        reduced_x = np.zeros(dimension, dtype=np.float64)
                        reduced_equality_linf = 0.0
                        reduced_dual_min = 0.0
                        reduced_dual_objective = 0.0
                    reduced_residual = matrix @ reduced_x - rhs
                    removal_candidates.append(
                        {
                            "local": local_remove,
                            "global": removed_global,
                            "consistent": reduced_equality_linf <= kkt_tol,
                            "equality_linf": reduced_equality_linf,
                            "dual_feasible": reduced_dual_min >= -dual_tol,
                            "dual_min": reduced_dual_min,
                            "primal_min": float(np.min(reduced_residual)),
                            "dual_objective": reduced_dual_objective,
                        }
                    )

                pool = [item for item in removal_candidates if item["consistent"]]
                if pool:
                    dual_feasible_pool = [item for item in pool if item["dual_feasible"]]
                    if dual_feasible_pool:
                        pool = dual_feasible_pool
                    for field_name in (
                        "primal_min",
                        "dual_min",
                        "dual_objective",
                    ):
                        best = max(float(item[field_name]) for item in pool)
                        pool = [
                            item
                            for item in pool
                            if abs(float(item[field_name]) - best) <= TIE_TOL
                        ]
                else:
                    best = min(float(item["equality_linf"]) for item in removal_candidates)
                    pool = [
                        item
                        for item in removal_candidates
                        if abs(float(item["equality_linf"]) - best) <= TIE_TOL
                    ]
                chosen = min(pool, key=lambda item: keys[int(item["global"])])
                removed_global = active.pop(int(chosen["local"]))
                trace.append(
                    {
                        "iteration": iteration,
                        "action": "drop_rank_inconsistent_member",
                        "key": keys[removed_global].as_list(),
                        "warm_lambda": float(warm_lambda[removed_global]),
                        "equality_residual_linf": equality_linf,
                        "one_delete_lookahead": {
                            "consistent": bool(chosen["consistent"]),
                            "equality_linf": float(chosen["equality_linf"]),
                            "dual_feasible": bool(chosen["dual_feasible"]),
                            "dual_min": float(chosen["dual_min"]),
                            "primal_min": float(chosen["primal_min"]),
                            "dual_objective": float(chosen["dual_objective"]),
                        },
                    }
                )
                continue
            negative = [
                local
                for local, value in enumerate(active_lambda)
                if float(value) < -dual_tol
            ]
            if negative:
                local_remove = _canonical_tied_index(
                    active_lambda,
                    negative,
                    [keys[index] for index in active],
                )
                removed_global = active.pop(local_remove)
                trace.append(
                    {
                        "iteration": iteration,
                        "action": "remove_negative_dual",
                        "key": keys[removed_global].as_list(),
                        "lambda": float(active_lambda[local_remove]),
                    }
                )
                continue
        else:
            x.fill(0.0)
            active_lambda = np.zeros(0, dtype=np.float64)

        residual = matrix @ x - rhs
        violated = [
            index for index, value in enumerate(residual) if float(value) < -kkt_tol
        ]
        if not violated:
            break
        add_index = _canonical_tied_index(residual, violated, keys)
        if add_index in active:
            raise ProtocolError("active-set polish stalled on an already-active cut")
        active.append(add_index)
        active.sort(key=lambda index: keys[index])
        trace.append(
            {
                "iteration": iteration,
                "action": "add_most_violated",
                "key": keys[add_index].as_list(),
                "residual": float(residual[add_index]),
            }
        )
    else:
        raise ProtocolError("dual active-set polish iteration cap reached")

    residual = matrix @ x - rhs
    full_lambda = np.zeros(len(ordered), dtype=np.float64)
    for local, global_index in enumerate(active):
        full_lambda[global_index] = active_lambda[local]
    stationarity_vector = x - matrix.T @ full_lambda
    complementarity = full_lambda * residual
    primal_objective = 0.5 * float(x @ x)
    dual_objective = float(rhs @ full_lambda - 0.5 * full_lambda @ (matrix @ matrix.T) @ full_lambda)
    duality_gap = primal_objective - dual_objective
    active_gram = matrix[active] @ matrix[active].T if active else np.zeros((0, 0))
    if active:
        singular = np.linalg.svd(active_gram, compute_uv=False)
        rank_threshold = float(singular[0]) * pinv_rcond if singular.size else 0.0
        positive = singular[singular > rank_threshold]
        rank = int(positive.size)
        condition = float(positive[0] / positive[-1]) if positive.size else math.inf
    else:
        rank = 0
        condition = 1.0
    checks = {
        "primal_feasible": float(np.min(residual)) >= -kkt_tol,
        "dual_feasible": float(np.min(full_lambda)) >= -dual_tol,
        "stationarity": float(np.linalg.norm(stationarity_vector)) <= kkt_tol,
        "complementarity": float(np.max(np.abs(complementarity))) <= kkt_tol,
        "strong_duality": abs(duality_gap) <= kkt_tol,
        "finite": all(
            math.isfinite(value)
            for value in (
                primal_objective,
                dual_objective,
                duality_gap,
                condition,
            )
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"absolute QP KKT certificate failed: {checks}")
    binding = [
        index for index, value in enumerate(residual) if float(value) <= kkt_tol
    ]
    audit = {
        "objective": "minimize_0p5_absolute_CW11_delta_l2_squared",
        "dimension": dimension,
        "constraint_count": len(ordered),
        "active_count": len(active),
        "binding_count": len(binding),
        "active_keys": [keys[index].as_list() for index in active],
        "binding_keys": [keys[index].as_list() for index in binding],
        "active_indices": active,
        "binding_indices": binding,
        "iterations": len(trace),
        "iteration_cap": max_iterations,
        "trace": trace,
        "warm_start": {
            "solver": "scipy.optimize.minimize/L-BFGS-B",
            "scipy_version": scipy.__version__,
            "role": "required_support_identification_precondition_not_KKT_certificate",
            "success_required_precondition": bool(warm.success),
            "status_required_precondition": int(warm.status),
            "message_required_precondition": str(warm.message),
            "iterations": int(warm.nit),
            "function_evaluations": int(warm.nfev),
            "objective": float(warm.fun),
            "support_threshold": support_threshold,
            "support_count": int(
                np.count_nonzero(warm_lambda > support_threshold)
            ),
            "lambda_float64_le_sha256": float64_sha(warm_lambda),
            "residual_float64_le_sha256": float64_sha(warm_residual),
        },
        "pinv_rcond": pinv_rcond,
        "dual_tolerance": dual_tol,
        "KKT_tolerance": kkt_tol,
        "rank": rank,
        "condition_number_nonzero_spectrum": condition,
        "matrix_float64_le_sha256": float64_sha(matrix),
        "rhs_float64_le_sha256": float64_sha(rhs),
        "gram_float64_le_sha256": float64_sha(active_gram),
        "solution_float64_le_sha256": float64_sha(x),
        "minimum_residual": float(np.min(residual)),
        "minimum_dual": float(np.min(full_lambda)),
        "stationarity_l2": float(np.linalg.norm(stationarity_vector)),
        "complementarity_linf": float(np.max(np.abs(complementarity))),
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "duality_gap": duality_gap,
        "checks": checks,
    }
    return x, audit


def project_to_row_nullspace(vector: Any, rows: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    value = np.asarray(vector, dtype=np.float64)
    matrix = np.asarray(rows, dtype=np.float64)
    if value.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != value.size:
        raise ProtocolError("nullspace projection shape mismatch")
    if matrix.shape[0] == 0:
        return value.copy(), {"row_count": 0, "projected": False, "rank": 0}
    gram = matrix @ matrix.T
    coefficients = np.linalg.pinv(gram, rcond=PINV_RCOND, hermitian=True) @ (matrix @ value)
    projected = np.asarray(value - matrix.T @ coefficients, dtype=np.float64)
    residual = matrix @ projected
    tolerance = max(KKT_TOL, float(np.linalg.norm(value)) * 1e-11)
    checks = {
        "finite": bool(np.isfinite(projected).all()),
        "row_orthogonal": float(np.max(np.abs(residual))) <= tolerance,
    }
    if not all(checks.values()):
        raise ProtocolError(f"secondary nullspace projection failed: {checks}")
    return projected, {
        "row_count": int(matrix.shape[0]),
        "projected": True,
        "rank": int(np.linalg.matrix_rank(gram, tol=PINV_RCOND)),
        "residual_linf": float(np.max(np.abs(residual))),
        "checks": checks,
    }


def compose_trust_step(
    *,
    current_delta: Any,
    absolute_qp_solution: Any,
    primary_pf7_gradient: Any,
    secondary_descent: Any,
    binding_rows: Any,
    safe_radius: float = SAFE_TARGET_RADIUS,
    trust_radius: float = TRUST_RADIUS,
    secondary_fraction: float = SECONDARY_NORM_FRACTION,
) -> tuple[Any, dict[str, Any]]:
    """Compose primary chord and fail-closed binding-nullspace secondary."""

    import numpy as np

    current = np.asarray(current_delta, dtype=np.float64)
    nominal = np.asarray(absolute_qp_solution, dtype=np.float64)
    g7 = np.asarray(primary_pf7_gradient, dtype=np.float64)
    h = np.asarray(secondary_descent, dtype=np.float64)
    binding = np.asarray(binding_rows, dtype=np.float64)
    if current.ndim != 1 or any(value.shape != current.shape for value in (nominal, g7, h)):
        raise ProtocolError("trust-step vector shape mismatch")
    if binding.ndim != 2 or binding.shape[1] != current.size:
        raise ProtocolError("trust-step binding matrix shape mismatch")
    if not all(np.isfinite(value).all() for value in (current, nominal, g7, h, binding)):
        raise ProtocolError("trust-step input is nonfinite")
    if not (0.0 <= secondary_fraction <= SECONDARY_NORM_FRACTION):
        raise ProtocolError("secondary fraction exceeds locked cap")
    current_norm = float(np.linalg.norm(current))
    if current_norm > HARD_ACTUAL_RADIUS + KKT_TOL:
        raise ProtocolError("accepted current endpoint exceeds hard actual radius")

    nominal_norm = float(np.linalg.norm(nominal))
    target_scale = min(1.0, safe_radius / nominal_norm) if nominal_norm > 0.0 else 1.0
    target = np.asarray(nominal * target_scale, dtype=np.float64)
    primary = target - current
    primary_raw_norm = float(np.linalg.norm(primary))
    if primary_raw_norm > trust_radius:
        primary *= trust_radius / primary_raw_norm
    primary_norm = float(np.linalg.norm(primary))

    null_rows = np.concatenate([g7.reshape(1, -1), binding], axis=0)
    secondary, null_audit = project_to_row_nullspace(h, null_rows)
    secondary_raw_norm = float(np.linalg.norm(secondary))
    secondary_cap = secondary_fraction * trust_radius
    if secondary_raw_norm > secondary_cap > 0.0:
        secondary *= secondary_cap / secondary_raw_norm
    elif secondary_cap == 0.0:
        secondary.fill(0.0)
    secondary_norm_before_safety = float(np.linalg.norm(secondary))

    reasons: list[str] = []
    if float(g7 @ secondary) < -FP32_PROGRESS_TOL:
        reasons.append("secondary_reduces_primary_PF7")
    if binding.shape[0] and float(np.min(binding @ secondary)) < -FP32_PROGRESS_TOL:
        reasons.append("secondary_reduces_binding_safety")
    combined = primary + secondary
    combined_norm_before_cap = float(np.linalg.norm(combined))
    if combined_norm_before_cap > trust_radius:
        combined *= trust_radius / combined_norm_before_cap

    primary_pf7_effect = float(g7 @ primary)
    combined_pf7_effect = float(g7 @ combined)
    if combined_pf7_effect < primary_pf7_effect - FP32_PROGRESS_TOL:
        reasons.append("eta_scaling_reduces_primary_PF7")
    if binding.shape[0]:
        primary_binding = binding @ primary
        combined_binding = binding @ combined
        if float(np.min(combined_binding - primary_binding)) < -FP32_PROGRESS_TOL:
            reasons.append("eta_scaling_reduces_binding_safety")
    trial = current + combined
    if float(np.linalg.norm(trial)) > safe_radius + KKT_TOL:
        reasons.append("secondary_leaves_safe_absolute_ball")

    secondary_used = not reasons and secondary_norm_before_safety > 0.0
    if reasons:
        secondary.fill(0.0)
        combined = primary.copy()
        trial = current + combined
    final_norm = float(np.linalg.norm(combined))
    trial_norm = float(np.linalg.norm(trial))
    checks = {
        "step_within_eta": final_norm <= trust_radius + KKT_TOL,
        "absolute_endpoint_within_R": trial_norm <= safe_radius + KKT_TOL,
        "secondary_cap": float(np.linalg.norm(secondary)) <= secondary_cap + KKT_TOL,
        "secondary_post_eta_PF7_nonharm": float(g7 @ combined)
        >= primary_pf7_effect - FP32_PROGRESS_TOL,
        "secondary_post_eta_binding_nonharm": not binding.shape[0]
        or float(np.min(binding @ combined - binding @ primary)) >= -FP32_PROGRESS_TOL,
    }
    if not all(checks.values()):
        raise ProtocolError(f"composed trust step failed: {checks}")
    return trial, {
        "safe_target_radius": safe_radius,
        "accepted_current_l2": current_norm,
        "accepted_current_within_hard_actual_radius": current_norm
        <= HARD_ACTUAL_RADIUS + KKT_TOL,
        "trust_radius": trust_radius,
        "secondary_norm_fraction_cap": secondary_fraction,
        "absolute_qp_solution_l2": nominal_norm,
        "absolute_target_scale": target_scale,
        "absolute_target_l2": float(np.linalg.norm(target)),
        "primary_chord_raw_l2": primary_raw_norm,
        "primary_chord_l2": primary_norm,
        "secondary_raw_l2": secondary_raw_norm,
        "secondary_l2": float(np.linalg.norm(secondary)),
        "secondary_used": secondary_used,
        "secondary_fail_closed_reasons": reasons,
        "combined_before_eta_cap_l2": combined_norm_before_cap,
        "final_step_l2": final_norm,
        "trial_absolute_l2": trial_norm,
        "primary_PF7_predicted_change": primary_pf7_effect,
        "final_PF7_predicted_change": float(g7 @ combined),
        "nullspace_projection": null_audit,
        "trial_float64_le_sha256": float64_sha(trial),
        "checks": checks,
    }


def fp32_direct_outputs(model: Any, batch: Mapping[str, Any]) -> Mapping[str, Any]:
    """The only permitted structural-forward path for v17 VJPs."""

    import torch

    with torch.autocast(device_type="cuda", enabled=False):
        outputs = model(batch)
    expected_keys = {"policy_logits", "count_logits", "value_logits"}
    batch_rows = int(batch["sample_weights"].shape[0])
    option_shape = tuple(batch["option_mask"].shape)
    checks = {
        "mapping": isinstance(outputs, Mapping),
        "keys_exact": isinstance(outputs, Mapping) and set(outputs) == expected_keys,
        "all_tensors": isinstance(outputs, Mapping)
        and all(isinstance(value, torch.Tensor) for value in outputs.values()),
        "all_CUDA": isinstance(outputs, Mapping)
        and all(
            isinstance(value, torch.Tensor) and value.device.type == "cuda"
            for value in outputs.values()
        ),
        "all_float32": isinstance(outputs, Mapping)
        and all(
            isinstance(value, torch.Tensor) and value.dtype == torch.float32
            for value in outputs.values()
        ),
        "all_finite": isinstance(outputs, Mapping)
        and all(
            isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
            for value in outputs.values()
        ),
        "all_batch_dimension_exact": isinstance(outputs, Mapping)
        and all(
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.shape[0]) == batch_rows
            for value in outputs.values()
        ),
        "policy_shape_exact_option_mask": isinstance(outputs, Mapping)
        and "policy_logits" in outputs
        and isinstance(outputs["policy_logits"], torch.Tensor)
        and tuple(outputs["policy_logits"].shape) == option_shape,
    }
    if not all(checks.values()):
        raise ProtocolError(f"FP32 direct model output contract drift: {checks}")
    return outputs


def positive_hinge_scalar(margin: float, target: float) -> float:
    margin_value = float(margin)
    target_value = canonical_target(float(target))
    if not math.isfinite(margin_value):
        raise ProtocolError("hinge margin is nonfinite")
    return max(0.0, target_value - margin_value)


def pf0_buffer_hinge(
    outputs: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]
) -> Any:
    """Mean FP32 hinge at terminal PF0 margin plus two local BF16 q."""

    import torch

    logits = outputs["policy_logits"]
    if logits.dtype != torch.float32 or not pairs:
        raise ProtocolError("PF0 buffer hinge requires nonempty FP32 pairs")
    terms = []
    for pair in pairs:
        row = int(pair["row_index"])
        positive = int(pair["positive_option"])
        negative = int(pair["negative_option"])
        margin = logits[row, positive] - logits[row, negative]
        target = logits.new_tensor(canonical_target(float(pair["threshold"])))
        terms.append(torch.relu(target - margin))
    result = torch.stack(terms).mean()
    if result.ndim != 0 or not bool(torch.isfinite(result)):
        raise ProtocolError("PF0 buffer hinge is nonfinite/non-scalar")
    return result


def secondary_loss(
    *,
    union: Any,
    pf0: Any,
    pf7: Any,
    dominic: Any,
    protected_kl: Any,
    pf0_buffer_penalty: Any,
) -> Any:
    """Locked secondary scalar; PF7 pair pressure belongs only to primary."""

    return (
        0.15 * union
        + 0.10 * pf0
        + 0.35 * pf7
        + 0.05 * dominic
        + 0.05 * protected_kl
        + 1.00 * pf0_buffer_penalty
    )


@dataclass
class Stage2ShadowLedger:
    """Pure accounting state used by production and CPU protocol tests."""

    accepted_actor_sha256: str
    accepted_fp32_pf7_margin: float
    accepted_native_pf7_cell: float
    trial_actor_hashes: set[str] = field(default_factory=set)
    stage2_trial_count: int = 0
    native_gate_forward_count: int = 0
    accepted_stage2_count: int = 0
    rejected_stage2_count: int = 0
    terminal_decision: str | None = None
    terminal_no_go_reason: str | None = None
    go_count: int = 0

    @property
    def terminal_go_reached(self) -> bool:
        return self.terminal_decision == "GO"

    def mark_terminal_no_go_without_native_trial(self, reason: str) -> None:
        """Close a pre-native or exhausted-budget path without changing counts."""

        if self.terminal_decision is not None:
            raise ProtocolError("terminal NO_GO attempted after terminal decision")
        if not isinstance(reason, str) or not reason:
            raise ProtocolError("terminal NO_GO reason must be a nonempty string")
        self.terminal_decision = "NO_GO"
        self.terminal_no_go_reason = reason

    def record_native_trial(
        self,
        *,
        actor_sha256: str,
        fp32_pf7_margin: float,
        native_pf7_cell: float,
        all_non_pf7_hard_gates_pass: bool,
        complete_native_gate_pass: bool,
        violated_cuts_added: int,
        restored_after_reject: bool,
        terminal_no_go_no_new_cut_reason: str | None = None,
    ) -> str:
        if self.terminal_decision is not None:
            raise ProtocolError("stage2 trial attempted after terminal decision")
        if self.stage2_trial_count >= STAGE2_MAX_TRIALS:
            raise ProtocolError("stage2 32-trial budget exhausted")
        if actor_sha256 in self.trial_actor_hashes or actor_sha256 == self.accepted_actor_sha256:
            raise ProtocolError("repeated actor state is not a new trial")
        if not math.isfinite(fp32_pf7_margin) or not math.isfinite(native_pf7_cell):
            raise ProtocolError("trial PF7 evidence is nonfinite")
        if type(violated_cuts_added) is not int or violated_cuts_added < 0:
            raise ProtocolError("violated cut count must be a nonnegative int")
        self.trial_actor_hashes.add(actor_sha256)
        self.stage2_trial_count += 1
        self.native_gate_forward_count += 1
        # The frozen v12 complete native gate is authoritative.  It is checked
        # before the v17 continuation heuristics and can never be vetoed by a
        # float32/native-PF7 nondegrade comparison.
        if complete_native_gate_pass:
            if not all_non_pf7_hard_gates_pass:
                raise ProtocolError("complete native pass contradicted non-PF7 summary")
            if restored_after_reject or violated_cuts_added != 0:
                raise ProtocolError("GO trial has reject-only side effects")
            if terminal_no_go_no_new_cut_reason is not None:
                raise ProtocolError("GO trial has terminal NO_GO metadata")
            self.accepted_stage2_count += 1
            self.accepted_actor_sha256 = actor_sha256
            self.accepted_fp32_pf7_margin = fp32_pf7_margin
            self.accepted_native_pf7_cell = native_pf7_cell
            self.terminal_decision = "GO"
            self.go_count += 1
            return "GO"
        pf7_fp32_nondegrade = (
            fp32_pf7_margin >= self.accepted_fp32_pf7_margin - FP32_PROGRESS_TOL
        )
        pf7_native_nondegrade = native_pf7_cell >= self.accepted_native_pf7_cell
        accept = all_non_pf7_hard_gates_pass and pf7_fp32_nondegrade and pf7_native_nondegrade
        if accept:
            if restored_after_reject or violated_cuts_added != 0:
                raise ProtocolError("accepted trial has reject-only side effects")
            if terminal_no_go_no_new_cut_reason is not None:
                raise ProtocolError("accepted trial has terminal NO_GO metadata")
            self.accepted_stage2_count += 1
            self.accepted_actor_sha256 = actor_sha256
            self.accepted_fp32_pf7_margin = fp32_pf7_margin
            self.accepted_native_pf7_cell = native_pf7_cell
            return "ACCEPT_CONTINUE"
        if not restored_after_reject:
            raise ProtocolError("rejected trial must restore accepted actor")
        self.rejected_stage2_count += 1
        if violated_cuts_added > 0:
            if terminal_no_go_no_new_cut_reason is not None:
                raise ProtocolError("cuttable rejection also declared uncuttable")
            return "REJECT_RESTORE_CONTINUE"
        if terminal_no_go_no_new_cut_reason not in TERMINAL_NO_NEW_CUT_REASONS:
            raise ProtocolError("zero-cut rejection lacks structured terminal reason")
        self.terminal_decision = "NO_GO"
        self.terminal_no_go_reason = str(terminal_no_go_no_new_cut_reason)
        return "REJECT_RESTORE_TERMINAL_NO_GO"

    def audit(self) -> dict[str, Any]:
        checks = {
            "one_native_gate_forward_per_stage2_trial": self.native_gate_forward_count
            == self.stage2_trial_count,
            "accepted_plus_rejected_equals_trials": self.accepted_stage2_count
            + self.rejected_stage2_count
            == self.stage2_trial_count,
            "stage2_within_32": self.stage2_trial_count <= STAGE2_MAX_TRIALS,
            "total_changed_shadows_within_61": STAGE1_REFERENCE_STEPS
            + self.stage2_trial_count
            <= TOTAL_CHANGED_TRAIN_SHADOW_CAP,
            "at_most_one_GO": self.go_count in {0, 1},
            "terminal_decision_typed": self.terminal_decision
            in {None, "GO", "NO_GO"},
            "terminal_GO_state_exact": self.terminal_go_reached == (self.go_count == 1),
            "terminal_NO_GO_reason_exact": (
                self.terminal_no_go_reason is not None
                if self.terminal_decision == "NO_GO"
                else self.terminal_no_go_reason is None
            ),
        }
        if not all(checks.values()):
            raise ProtocolError(f"stage2 shadow ledger failed: {checks}")
        return {
            "stage1_exact_changed_train_shadows": STAGE1_REFERENCE_STEPS,
            "stage2_changed_train_shadows": self.stage2_trial_count,
            "native_gate_forward_count": self.native_gate_forward_count,
            "total_changed_train_shadows": STAGE1_REFERENCE_STEPS
            + self.stage2_trial_count,
            "accepted_stage2_count": self.accepted_stage2_count,
            "rejected_stage2_count": self.rejected_stage2_count,
            "unique_trial_actor_sha256": sorted(self.trial_actor_hashes),
            "terminal_GO_reached": self.terminal_go_reached,
            "terminal_decision": self.terminal_decision,
            "terminal_no_go_reason": self.terminal_no_go_reason,
            "GO_count": self.go_count,
            "checks": checks,
        }


def load_json_no_duplicates(raw: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ProtocolError(f"{label}: nonfinite JSON constant {value}")

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ProtocolError(f"{label}: root is not an object")
    return value


def binding_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(record["path"]),
        "sha256": str(record["sha256"]),
        "bytes": int(record["bytes"]),
        "mode_octal": str(record["mode_octal"]),
        "device": int(record["device"]),
        "inode": int(record["inode"]),
        "nlink": int(record["nlink"]),
    }


def dependency_evidence() -> dict[str, Any]:
    return {
        label: immutable_file_evidence(label, path, digest, mode)
        for label, path, digest, mode in FROZEN_INPUTS
    }


def validate_pre_cuda_runtime() -> dict[str, Any]:
    checks = {
        "repo_root": Path.cwd().resolve() == ROOT,
        "my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1
        and sys.dont_write_bytecode is True,
        "pycache_prefix_dev_null": sys.pycache_prefix == "/dev/null",
        "torch_not_imported_before_claim": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        raise ProtocolError(f"pre-CUDA runtime drift: {checks}")
    return {
        "python": str(Path(sys.executable).resolve()),
        "checks": checks,
        "pass": True,
    }


def transform_v13_source() -> tuple[bytes, dict[str, Any]]:
    before = V13_SOURCE.lstat()
    raw = V13_SOURCE.read_bytes()
    after = V13_SOURCE.lstat()
    source = raw.decode("utf-8")
    checks = {
        "frozen_source_sha_exact": hashlib.sha256(raw).hexdigest()
        == V13_SOURCE_SHA256,
        "frozen_source_mode_0555": stat.S_IMODE(after.st_mode) == 0o555,
        "frozen_source_regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "frozen_source_identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
        "one_exact_step29_handoff_site": source.count(V13_HANDOFF_NEEDLE) == 1,
        "frozen_source_has_no_v17_handoff": "cw24_v17_stage2_handoff" not in source,
    }
    if not all(checks.values()):
        raise ProtocolError(f"frozen v13 handoff transform drift: {checks}")
    transformed_text = source.replace(
        V13_HANDOFF_NEEDLE, V13_HANDOFF_REPLACEMENT, 1
    )
    transformed = transformed_text.encode("utf-8")
    compile(transformed, str(V13_SOURCE), "exec", dont_inherit=True)
    transform_checks = {
        "one_source_replacement": source.count(V13_HANDOFF_NEEDLE) == 1,
        "one_handoff_call": transformed.count(
            b"cw24_v17_stage2_handoff(dict(locals()))"
        )
        == 1,
        "stage2_SGD_source_retained_but_unreachable_after_handoff": transformed.count(
            b"project_non_outward_sgd_direction("
        )
        >= 2,
        "syntax_valid": True,
    }
    if not all(transform_checks.values()):
        raise ProtocolError(f"transformed v13 source audit failed: {transform_checks}")
    return transformed, {
        "frozen_source": {
            "path": str(V13_SOURCE.relative_to(ROOT)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
            "nlink": int(after.st_nlink),
        },
        "transformed_sha256": hashlib.sha256(transformed).hexdigest(),
        "transformed_bytes": len(transformed),
        "replacement_sha256": hashlib.sha256(
            V13_HANDOFF_REPLACEMENT.encode("utf-8")
        ).hexdigest(),
        "checks": {**checks, **transform_checks},
    }


def load_locked_module(
    path: Path,
    expected_sha256: str,
    expected_mode: int,
    module_name: str,
) -> tuple[ModuleType, dict[str, Any]]:
    evidence = immutable_file_evidence(
        module_name, path, expected_sha256, expected_mode
    )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ProtocolError(f"{module_name}: held source digest drift")
    if module_name in sys.modules:
        raise ProtocolError(f"module name already occupied: {module_name}")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, evidence


def load_transformed_v13_module(
    handoff: Any,
) -> tuple[ModuleType, dict[str, Any]]:
    transformed, audit = transform_v13_source()
    module_name = "cw24_v17_step29_handoff_transformed_v13"
    if module_name in sys.modules:
        raise ProtocolError("transformed v13 module name already occupied")
    module = types.ModuleType(module_name)
    module.__file__ = str(V13_SOURCE)
    module.__package__ = ""
    module.cw24_v17_stage2_handoff = handoff
    sys.modules[module_name] = module
    try:
        exec(
            compile(transformed, str(V13_SOURCE), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, audit


def self_evidence(require_frozen: bool) -> dict[str, Any]:
    before = SCRIPT.lstat()
    source = SCRIPT.read_bytes()
    after = SCRIPT.lstat()
    mode = stat.S_IMODE(after.st_mode)
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ),
        "mode_allowed": mode == 0o555
        if require_frozen
        else mode in {0o644, 0o664, 0o555},
        "source_audit": source_audit()["pass"],
    }
    if not all(checks.values()):
        raise ProtocolError(f"v17 self evidence failed: {checks}")
    return {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "mode_octal": format(mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def publication_staging_residues(path: Path) -> list[str]:
    prefix = f".{path.name}.staging-"
    return sorted(
        str(candidate.relative_to(ROOT))
        for candidate in path.parent.iterdir()
        if candidate.name.startswith(prefix)
    )


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    """Publish complete verified bytes via same-directory no-clobber link.

    The final pathname is never the write target.  A private staging inode is
    written, fsynced, chmodded, reread through the held fd, and only then
    atomically hard-linked at the absent final name.  Any exception removes
    only names that still resolve to the exact inode created by this call.
    """

    if not hasattr(os, "O_NOFOLLOW"):
        raise ProtocolError("O_NOFOLLOW is required for one-shot publication")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    staging = path.with_name(
        f".{path.name}.staging-{os.getpid()}-{payload_sha256[:16]}"
    )
    if not path_absent(path):
        raise ProtocolError(f"v17 no-clobber publication target exists: {path.name}")
    if not path_absent(staging):
        raise ProtocolError(f"v17 publication staging path exists: {staging.name}")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd: int | None = None
    created_inode: tuple[int, int] | None = None
    try:
        fd = os.open(staging, flags, 0o600)
        opened = os.fstat(fd)
        created_inode = (int(opened.st_dev), int(opened.st_ino))
        opened_checks = {
            "staging_regular": stat.S_ISREG(opened.st_mode),
            "staging_mode_0600": stat.S_IMODE(opened.st_mode) == 0o600,
            "staging_single_link": int(opened.st_nlink) == 1,
            "final_still_absent_before_write": path_absent(path),
        }
        if not all(opened_checks.values()):
            raise ProtocolError(f"v17 staging open drift: {opened_checks}")

        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ProtocolError("short v17 staging publication write")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        held_digest = hashlib.sha256()
        held_bytes = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            held_digest.update(chunk)
            held_bytes += len(chunk)
        completed_fd = os.fstat(fd)
        completed_path = staging.lstat()
        prelink_checks = {
            "held_fd_same_created_inode": (
                int(completed_fd.st_dev), int(completed_fd.st_ino)
            )
            == created_inode,
            "staging_path_same_held_identity": stat_identity(completed_path)
            == stat_identity(completed_fd),
            "staging_regular_single_link": stat.S_ISREG(completed_fd.st_mode)
            and not stat.S_ISLNK(completed_path.st_mode)
            and int(completed_fd.st_nlink) == 1,
            "staging_mode_0444": stat.S_IMODE(completed_fd.st_mode) == 0o444,
            "held_size_exact": int(completed_fd.st_size)
            == held_bytes
            == len(payload),
            "held_sha_exact": held_digest.hexdigest() == payload_sha256,
            "final_absent_before_atomic_link": path_absent(path),
        }
        if not all(prelink_checks.values()):
            raise ProtocolError(f"v17 staging prelink drift: {prelink_checks}")

        os.link(staging, path, follow_symlinks=False)
        linked_fd = os.fstat(fd)
        linked_staging = staging.lstat()
        linked_final = path.lstat()
        link_checks = {
            "atomic_no_clobber_link_same_held_inode": stat_identity(linked_fd)
            == stat_identity(linked_staging)
            == stat_identity(linked_final),
            "linked_regular_two_links": stat.S_ISREG(linked_final.st_mode)
            and not stat.S_ISLNK(linked_final.st_mode)
            and int(linked_final.st_nlink) == 2,
            "linked_mode_0444": stat.S_IMODE(linked_final.st_mode) == 0o444,
            "linked_size_exact": int(linked_final.st_size) == len(payload),
        }
        if not all(link_checks.values()):
            raise ProtocolError(f"v17 atomic link drift: {link_checks}")

        os.unlink(staging)
        final_fd = os.fstat(fd)
        final_path = path.lstat()
        final_checks = {
            "final_same_held_inode": stat_identity(final_fd)
            == stat_identity(final_path),
            "final_regular_single_link": stat.S_ISREG(final_path.st_mode)
            and not stat.S_ISLNK(final_path.st_mode)
            and int(final_path.st_nlink) == 1,
            "final_mode_0444": stat.S_IMODE(final_path.st_mode) == 0o444,
            "final_size_exact": int(final_path.st_size) == len(payload),
            "held_sha_exact": held_digest.hexdigest() == payload_sha256,
            "staging_absent_after_promotion": path_absent(staging),
        }
        if not all(final_checks.values()):
            raise ProtocolError(f"v17 final publication drift: {final_checks}")
        fsync_directory(path.parent)
        os.close(fd)
        fd = None
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": payload_sha256,
            "bytes": int(final_path.st_size),
            "mode_octal": format(stat.S_IMODE(final_path.st_mode), "04o"),
            "device": int(final_path.st_dev),
            "inode": int(final_path.st_ino),
            "nlink": int(final_path.st_nlink),
            "publication_method": (
                "same_directory_private_staging_fsync_then_atomic_no_clobber_link"
            ),
            "checks": {**opened_checks, **prelink_checks, **link_checks, **final_checks},
        }
    except BaseException:
        if created_inode is not None:
            for cleanup_path in (path, staging):
                try:
                    observed = cleanup_path.lstat()
                except FileNotFoundError:
                    continue
                except BaseException:
                    continue
                if (int(observed.st_dev), int(observed.st_ino)) == created_inode:
                    try:
                        os.unlink(cleanup_path)
                    except BaseException:
                        pass
            try:
                fsync_directory(path.parent)
            except BaseException:
                pass
        raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except BaseException:
                pass


def claim_attempt() -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    if not path_absent(OUTPUT):
        raise ProtocolError("v17 output already exists before attempt claim")
    if not path_absent(ATTEMPT_MARKER):
        raise ProtocolError("v17 one-shot attempt was already consumed")
    if not path_absent(FAILURE):
        raise ProtocolError("v17 failure target already exists before attempt claim")
    staging_residues = {
        str(path.relative_to(ROOT)): publication_staging_residues(path)
        for path in (ATTEMPT_MARKER, OUTPUT, FAILURE)
    }
    if any(staging_residues.values()):
        raise ProtocolError(
            f"v17 publication staging residue exists before claim: {staging_residues}"
        )
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    transform = transform_v13_source()[1]
    payload = {
        "schema_version": f"{SCHEMA}-attempt",
        "status": "claimed_before_module_exec_or_CUDA",
        "source": binding_summary(source),
        "dependencies": {
            name: binding_summary(record)
            for name, record in dependencies.items()
        },
        "step29_handoff_transform": transform,
        "runtime": runtime,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent_by_lstat_at_claim": True,
        "failure": str(FAILURE.relative_to(ROOT)),
        "failure_absent_by_lstat_at_claim": True,
        "publication_staging_residues_absent_at_claim": True,
        "CUDA_preflight_state": "torch_not_imported_before_claim",
        "protocol_id": PROTOCOL_ID,
        "stage1_execution_seed": STAGE1_EXECUTION_SEED,
        "stage2_randomness": "none_deterministic_linear_algebra_only",
        "stage1_reference_steps": STAGE1_REFERENCE_STEPS,
        "stage2_maximum_native_trials": STAGE2_MAX_TRIALS,
        "total_maximum_changed_train_shadows": TOTAL_CHANGED_TRAIN_SHADOW_CAP,
        "retry_authorized": False,
        "network_package_upload_submission": False,
    }
    publication = publish_o_excl(ATTEMPT_MARKER, canonical_json(payload))
    return {"payload": payload, "publication": publication}


def no_go_material_audit(value: Any) -> dict[str, Any]:
    violations: list[str] = []
    forbidden_exact = {
        "actor_bytes",
        "candidate_actor_float32_le",
        "actual_delta",
        "state_dict",
        "model_state_dict",
    }

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                child_path = f"{path}.{key}"
                lowered = str(key).lower()
                if key == "candidate_payload" and child is not None:
                    violations.append(f"{child_path}:non_null")
                if key in forbidden_exact:
                    violations.append(f"{child_path}:forbidden_material_key")
                if "delta" in lowered and isinstance(child, (list, tuple)):
                    violations.append(f"{child_path}:delta_array")
                if any(token in lowered for token in ("base64", "compressed", "xz_payload")):
                    if (
                        child is not None
                        and child is not False
                        and child != 0
                        and child != ""
                    ):
                        violations.append(f"{child_path}:encoded_material")
                walk(child, child_path)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
        elif isinstance(item, (bytes, bytearray, memoryview)):
            violations.append(f"{path}:binary_value")

    walk(value, "$<result>")
    if violations:
        raise ProtocolError(f"v17 NO_GO material rejected: {violations[:8]}")
    return {
        "pass": True,
        "candidate_payloads_all_null": True,
        "actor_bytes_absent": True,
        "actual_delta_arrays_absent": True,
        "encoded_or_binary_material_absent": True,
    }


def flatten_vjp(
    gradients: Sequence[Any], expected_dimension: int
) -> Any:
    import numpy as np
    import torch

    if not gradients or any(value is None for value in gradients):
        raise ProtocolError("FP32 VJP produced a missing actor6 gradient")
    pieces = []
    for value in gradients:
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ProtocolError("FP32 VJP produced a nonfinite actor6 gradient")
        pieces.append(
            value.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .reshape(-1)
            .astype(np.float64, copy=False)
        )
    result = np.concatenate(pieces).astype(np.float64, copy=False)
    if result.shape != (expected_dimension,) or not bool(np.isfinite(result).all()):
        raise ProtocolError("FP32 VJP flattened actor6 geometry drift")
    return result


def constraint_specs(
    *,
    rows: Sequence[Mapping[str, Any]],
    teacher_policy_logits: Any,
    stage2_pair_contract: Mapping[str, Any],
    dynamic_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    result: list[dict[str, Any]] = []

    def add(
        pair: Mapping[str, Any],
        target: float,
        kind: str,
        origins: Sequence[str],
        name: str,
    ) -> None:
        row = int(pair["row_index"])
        positive = int(pair["positive_option"])
        negative = int(pair["negative_option"])
        line = str(pair["line_sha256"])
        target_value = canonical_target(float(target))
        checks = {
            "row_in_range": 0 <= row < len(rows),
            "line_matches_row": 0 <= row < len(rows)
            and str(rows[row]["line_sha256"]) == line,
            "options_distinct_nonnegative": positive >= 0
            and negative >= 0
            and positive != negative,
            "target_finite": math.isfinite(target_value),
        }
        if not all(checks.values()):
            raise ProtocolError(f"constraint identity drift for {name}: {checks}")
        result.append(
            {
                "name": str(name),
                "row_index": row,
                "line_sha256": line,
                "positive_option": positive,
                "negative_option": negative,
                "target": target_value,
                "target_kind": str(kind),
                "origins": sorted(set(str(value) for value in origins)),
            }
        )

    pf7 = dict(stage2_pair_contract["PF7"])
    add(
        pf7,
        float(pf7["threshold"]),
        PF7_TARGET_KIND,
        ["fixed_PF7", *pf7.get("origins", [])],
        "PF7_positive_step29_local_q",
    )
    for offset, pair_value in enumerate(stage2_pair_contract["PF0"]):
        pair = dict(pair_value)
        add(
            pair,
            float(pair["threshold"]),
            PF0_TARGET_KIND,
            [f"fixed_PF0_{offset}", *pair.get("origins", [])],
            f"PF0_terminal_plus_one_q_{offset}",
        )
    for offset, pair_value in enumerate(stage2_pair_contract["zero"]):
        pair = dict(pair_value)
        add(
            pair,
            0.0,
            ZERO_TARGET_KIND,
            [f"fixed_zero_{offset}", *pair.get("origins", [])],
            f"zero_native_floor_{offset}",
        )
    if len(dynamic_pairs) > MAX_DYNAMIC_PAIRS:
        raise ProtocolError("dynamic pair ledger exceeds locked cap64")
    for offset, pair_value in enumerate(dynamic_pairs):
        pair = dict(pair_value)
        add(
            pair,
            float(pair["threshold"]),
            DYNAMIC_TARGET_KIND,
            ["sticky_dynamic", *pair.get("origins", [])],
            f"dynamic_{offset}_{str(pair['line_sha256'])[:12]}",
        )

    teacher = teacher_policy_logits.detach().float().cpu()
    if not isinstance(teacher, torch.Tensor) or tuple(teacher.shape[:1]) != (len(rows),):
        raise ProtocolError("CW11 teacher-policy shape drift for historical cuts")
    for line, row, positive, negative, target in HISTORICAL_PAIR_CONTRACT:
        observed = float(teacher[row, positive] - teacher[row, negative])
        checks = {
            "row_line_exact": str(rows[row]["line_sha256"]) == line,
            "expert_option_exact": int(rows[row]["expert_order"][0]) == positive,
            "CW11_native_margin_exact": observed == target,
        }
        if not all(checks.values()):
            raise ProtocolError(f"historical v4 cut drift for {line[:12]}: {checks}")
        add(
            {
                "row_index": row,
                "line_sha256": line,
                "positive_option": positive,
                "negative_option": negative,
            },
            target,
            HISTORICAL_TARGET_KIND,
            ["frozen_v4_CW11_native_strongest_threat"],
            f"historical_{line[:12]}",
        )

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for spec in result:
        identity = target_identity(float(spec["target"]))
        key = (
            str(spec["line_sha256"]),
            int(spec["row_index"]),
            int(spec["positive_option"]),
            int(spec["negative_option"]),
            identity["float64_le_sha256"],
        )
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = {
                **spec,
                "names": [str(spec["name"])],
                "target_kinds": [str(spec["target_kind"])],
                "origins": list(spec["origins"]),
                "target_identity": identity,
            }
            continue
        existing["names"] = sorted(
            set(existing["names"]) | {str(spec["name"])}
        )
        existing["target_kinds"] = sorted(
            set(existing["target_kinds"]) | {str(spec["target_kind"])}
        )
        existing["origins"] = sorted(
            set(existing["origins"]) | set(spec["origins"])
        )
    ordered = [grouped[key] for key in sorted(grouped)]
    # Fixed PF/zero plus all three historical geometries contribute twelve
    # unique contracts before dynamic cuts.  Dynamic rows may legitimately
    # coalesce with historical geometry (243f initially, and potentially 0142
    # later), so their raw count is not a lower bound on unique QP rows.
    if len(ordered) < 12:
        raise ProtocolError("unexpected fixed/dynamic historical constraint collapse")
    return ordered


def fp32_structural_bundle(
    *,
    model: Any,
    batch: Mapping[str, Any],
    parameters: Mapping[str, Any],
    names: Sequence[str],
    current_delta: Any,
    actor_sha256: str,
    specs: Sequence[Mapping[str, Any]],
    teacher_policy_logits: Any,
    contract: Mapping[str, Any],
    stage2_pair_contract: Mapping[str, Any],
    ppo: ModuleType,
    v12: ModuleType,
    v13: ModuleType,
) -> dict[str, Any]:
    import numpy as np
    import torch

    delta = np.asarray(current_delta, dtype=np.float64)
    if delta.ndim != 1 or delta.size != 65793 or not bool(np.isfinite(delta).all()):
        raise ProtocolError("structural bundle actor6 delta shape/finite drift")
    if any(parameters[name].grad is not None for name in names):
        raise ProtocolError("structural VJP entered with accumulated actor gradients")
    outputs = fp32_direct_outputs(model, batch)
    logits = outputs["policy_logits"]
    parameter_tuple = tuple(parameters[name] for name in names)
    cuts: list[NormalizedCut] = []
    records: list[dict[str, Any]] = []
    primary_pf7_gradient = None
    primary_pf7_margin = None
    for spec in specs:
        row = int(spec["row_index"])
        positive = int(spec["positive_option"])
        negative = int(spec["negative_option"])
        margin_tensor = logits[row, positive] - logits[row, negative]
        if margin_tensor.ndim != 0 or margin_tensor.dtype != torch.float32:
            raise ProtocolError("structural margin is not scalar FP32")
        margin = float(margin_tensor.detach().cpu())
        gradients = torch.autograd.grad(
            margin_tensor,
            parameter_tuple,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        gradient = flatten_vjp(gradients, int(delta.size))
        target = canonical_target(float(spec["target"]))
        kinds = tuple(sorted(str(value) for value in spec["target_kinds"]))
        key = make_cut_key(
            actor_sha256=actor_sha256,
            line_sha256=str(spec["line_sha256"]),
            positive_option=positive,
            negative_option=negative,
            target_kind="+".join(kinds),
            target=target,
        )
        cut = normalize_absolute_cut(
            key=key,
            current_delta=delta,
            current_margin=margin,
            gradient=gradient,
            target=target,
            target_kind="+".join(kinds),
            origins=tuple(str(value) for value in spec["origins"]),
        )
        cuts.append(cut)
        record = {
            "names": list(spec["names"]),
            "row_index": row,
            "line_sha256": str(spec["line_sha256"]),
            "positive_option": positive,
            "negative_option": negative,
            "target": target,
            "target_identity": target_identity(target),
            "target_kinds": list(kinds),
            "origins": list(spec["origins"]),
            "FP32_margin": margin,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_float64_le_sha256": float64_sha(gradient),
            "normalized_rhs": float(cut.rhs),
            "normalized_normal_float64_le_sha256": float64_sha(cut.normal),
            "violated_at_linearization_point": margin < target,
            "cut_key": key.as_list(),
        }
        records.append(record)
        if PF7_TARGET_KIND in kinds:
            if primary_pf7_gradient is not None:
                raise ProtocolError("multiple canonical PF7 primary constraints")
            primary_pf7_gradient = gradient.copy()
            primary_pf7_margin = margin

    masks = contract["masks"]
    union = v12.complete_ordered_scalar(
        outputs, v12.masked_batch(batch, masks["union"]), ppo
    )
    pf0 = v12.complete_ordered_scalar(
        outputs, v12.masked_batch(batch, masks["pf0"]), ppo
    )
    pf7 = v12.complete_ordered_scalar(
        outputs, v12.masked_batch(batch, masks["pf7"]), ppo
    )
    dominic = v12.complete_ordered_scalar(
        outputs, v12.masked_batch(batch, masks["dominic"]), ppo
    )
    protected_kl = v13.masked_policy_kl(
        outputs, batch, teacher_policy_logits, masks["protected"]
    )
    pf0_buffer_pairs = []
    for pair_value in stage2_pair_contract["PF0"]:
        pair = dict(pair_value)
        pair["threshold"] = float(pair["threshold"]) + float(pair["temperature"])
        pair["threshold_source"] = "terminal_threshold_plus_two_step29_local_q"
        pf0_buffer_pairs.append(pair)
    pf0_buffer = pf0_buffer_hinge(outputs, pf0_buffer_pairs)
    secondary = secondary_loss(
        union=union,
        pf0=pf0,
        pf7=pf7,
        dominic=dominic,
        protected_kl=protected_kl,
        pf0_buffer_penalty=pf0_buffer,
    )
    if secondary.ndim != 0 or not bool(torch.isfinite(secondary)):
        raise ProtocolError("secondary scalar is nonfinite/non-scalar")
    secondary_gradients = torch.autograd.grad(
        secondary,
        parameter_tuple,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    secondary_descent = -flatten_vjp(secondary_gradients, int(delta.size))
    if primary_pf7_gradient is None or primary_pf7_margin is None:
        raise ProtocolError("structural bundle lacks canonical PF7 primary")
    gradient_absence = all(parameters[name].grad is None for name in names)
    if not gradient_absence:
        raise ProtocolError("autograd.grad accumulated forbidden parameter gradients")
    return {
        "cuts": cuts,
        "records": records,
        "primary_pf7_gradient": primary_pf7_gradient,
        "primary_pf7_margin": float(primary_pf7_margin),
        "secondary_descent": secondary_descent,
        "audit": {
            "forward": "direct_model_batch_autocast_disabled",
            "floating_output_dtypes": {
                name: str(value.dtype)
                for name, value in outputs.items()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            },
            "constraint_count_after_geometry_dedup": len(records),
            "constraints": records,
            "secondary_formula": (
                "0.15*union+0.10*pf0+0.35*pf7+0.05*dominic+"
                "0.05*protected_KL+1.00*PF0_terminal_plus_two_q_buffer"
            ),
            "secondary_components": {
                "union": float(union.detach().cpu()),
                "PF0": float(pf0.detach().cpu()),
                "PF7": float(pf7.detach().cpu()),
                "Dominic": float(dominic.detach().cpu()),
                "protected_KL": float(protected_kl.detach().cpu()),
                "PF0_buffer": float(pf0_buffer.detach().cpu()),
                "total": float(secondary.detach().cpu()),
            },
            "secondary_descent_l2": float(np.linalg.norm(secondary_descent)),
            "secondary_descent_float64_le_sha256": float64_sha(
                secondary_descent
            ),
            "actor_parameter_gradients_absent_after_VJPs": gradient_absence,
        },
    }


def descriptive_cut_slacks(
    cuts: Sequence[NormalizedCut],
    *,
    ball_free_solution: Any,
    capped_absolute_target: Any,
    planned_trial: Any,
) -> dict[str, Any]:
    import numpy as np

    matrix = np.stack([np.asarray(cut.normal, dtype=np.float64) for cut in cuts])
    rhs = np.asarray([float(cut.rhs) for cut in cuts], dtype=np.float64)

    def describe(value: Any) -> dict[str, Any]:
        vector = np.asarray(value, dtype=np.float64)
        slack = matrix @ vector - rhs
        return {
            "minimum_slack": float(np.min(slack)),
            "violated_count": int(np.count_nonzero(slack < 0.0)),
            "slack_float64_le_sha256": float64_sha(slack),
        }

    return {
        "semantics": "descriptive_only_after_radius_or_trust_cap",
        "certified": False,
        "used_for_selection": False,
        "negative_slack_is_allowed_and_does_not_trigger_NO_GO": True,
        "ball_free_solution_crosscheck": describe(ball_free_solution),
        "safe_radius_capped_absolute_target": describe(capped_absolute_target),
        "planned_trust_trial": describe(planned_trial),
    }


def restore_accepted_actor_exact(
    *,
    accepted_actor_bytes: bytes,
    accepted_actor_sha256: str,
    accepted_model_sha256: str,
    accepted_nonactor_sha256: str,
    accepted_delta_sha256: str,
    parameters: Mapping[str, Any],
    names: Sequence[str],
    layout: Sequence[Mapping[str, Any]],
    cw11_flat: Any,
    model: Any,
    helper: Any,
    cw20: ModuleType,
    cw22: ModuleType,
    optimizer: Any,
) -> dict[str, Any]:
    import numpy as np
    import torch

    cw20.copy_actor_bytes(
        parameters,
        names,
        layout,
        accepted_actor_bytes,
        np,
        torch,
    )
    for name in names:
        parameters[name].grad = None
    restored_bytes = cw20.actor_bytes(parameters, names, np)
    restored_flat = cw20.flat_actor(parameters, names, np)
    restored_delta = restored_flat - cw11_flat
    checks = {
        "actor_bytes_exact": restored_bytes == accepted_actor_bytes,
        "actor_sha_exact": hashlib.sha256(restored_bytes).hexdigest()
        == accepted_actor_sha256,
        "model_sha_exact": helper.model_state_sha256(model.state_dict())
        == accepted_model_sha256,
        "nonactor_sha_exact": cw20.nonactor_sha(model, names, helper)
        == accepted_nonactor_sha256,
        "delta_sha_exact": float64_sha(restored_delta) == accepted_delta_sha256,
        "actor_gradients_absent": all(parameters[name].grad is None for name in names),
        "optimizer_state_empty": not optimizer.state
        and not optimizer.state_dict()["state"],
    }
    if not all(checks.values()):
        raise ProtocolError(f"rejected actor restore drift: {checks}")
    return {
        "checks": checks,
        "actor_sha256": accepted_actor_sha256,
        "model_sha256": accepted_model_sha256,
        "nonactor_sha256": accepted_nonactor_sha256,
        "delta_float64_le_sha256": accepted_delta_sha256,
    }


def verify_candidate_identity_exact(
    *,
    expected_actor_bytes: bytes,
    expected_actor_sha256: str,
    expected_model_sha256: str,
    expected_nonactor_sha256: str,
    parameters: Mapping[str, Any],
    names: Sequence[str],
    model: Any,
    helper: Any,
    cw20: ModuleType,
    optimizer: Any,
) -> dict[str, bool]:
    import numpy as np

    observed_bytes = cw20.actor_bytes(parameters, names, np)
    checks = {
        "actor_bytes_exact": observed_bytes == expected_actor_bytes,
        "actor_sha_exact": hashlib.sha256(observed_bytes).hexdigest()
        == expected_actor_sha256,
        "model_sha_exact": helper.model_state_sha256(model.state_dict())
        == expected_model_sha256,
        "nonactor_sha_exact": cw20.nonactor_sha(model, names, helper)
        == expected_nonactor_sha256,
        "actor_gradients_absent": all(parameters[name].grad is None for name in names),
        "optimizer_state_empty": not optimizer.state
        and not optimizer.state_dict()["state"],
    }
    return checks


def run_stage2_handoff(local_state: Mapping[str, Any], *, v15: ModuleType) -> dict[str, Any]:
    """Continue from the exact frozen-v12 step29 state without another SGD step."""

    import numpy as np
    import torch

    required = {
        "context",
        "rows",
        "cw20",
        "cw22",
        "cw23",
        "v1",
        "v12",
        "modules",
        "helper",
        "model",
        "ppo",
        "context_checks",
        "dependency_checks",
        "cache_audit",
        "named",
        "parameters",
        "parameter_sequence",
        "cw11_actor",
        "cw11_flat",
        "batch",
        "baseline",
        "baseline_native",
        "teacher_policy_logits",
        "contract",
        "dynamic_pairs",
        "baseline_added",
        "baseline_loss_audit",
        "baseline_base_soft_loss",
        "optimizer",
        "actor_hashes",
        "stage1_trajectory",
        "stage2_pair_contract",
        "transition",
        "projection",
        "candidate_hash",
        "actor_hash",
        "global_step",
        "stage2_started",
        "total_loss",
        "outputs",
    }
    if not required.issubset(local_state):
        missing = sorted(required - set(local_state))
        raise ProtocolError(f"step29 handoff locals missing: {missing}")
    context = local_state["context"]
    rows = local_state["rows"]
    cw20 = local_state["cw20"]
    cw22 = local_state["cw22"]
    cw23 = local_state["cw23"]
    v1 = local_state["v1"]
    v12 = local_state["v12"]
    v13 = sys.modules["cw24_v17_step29_handoff_transformed_v13"]
    modules = local_state["modules"]
    helper = local_state["helper"]
    model = local_state["model"]
    ppo = local_state["ppo"]
    parameters = local_state["parameters"]
    parameter_sequence = local_state["parameter_sequence"]
    cw11_actor = local_state["cw11_actor"]
    cw11_flat = np.asarray(local_state["cw11_flat"], dtype=np.float64)
    batch = local_state["batch"]
    baseline = local_state["baseline"]
    baseline_native = local_state["baseline_native"]
    teacher_policy_logits = local_state["teacher_policy_logits"]
    contract = local_state["contract"]
    dynamic_pairs = local_state["dynamic_pairs"]
    optimizer = local_state["optimizer"]
    names = tuple(cw22.EXPECTED_ACTOR_NAMES)
    stage1_trajectory = local_state["stage1_trajectory"]
    stage2_pair_contract = local_state["stage2_pair_contract"]
    transition = local_state["transition"]
    device = next(model.parameters()).device

    handoff_checks = {
        "global_step_exact29": type(local_state["global_step"]) is int
        and local_state["global_step"] == STAGE1_REFERENCE_STEPS,
        "stage2_objective_built": local_state["stage2_started"] is True,
        "stage1_trajectory_exact29": len(stage1_trajectory)
        == STAGE1_REFERENCE_STEPS,
        "stage1_every_reference_check_true": all(
            all(record["v12_reference_exact_checks"].values())
            for record in stage1_trajectory
        ),
        "stage1_terminal_actor_exact": local_state["actor_hash"]
        == v13.STAGE1_REFERENCE_ACTOR_SHA256,
        "stage1_terminal_model_exact": local_state["candidate_hash"]
        == v13.STAGE1_REFERENCE_MODEL_SHA256,
        "stage1_terminal_delta_exact": local_state["projection"][
            "actual_delta_float64_le_sha256"
        ]
        == v13.STAGE1_REFERENCE_DELTA_SHA256,
        "single_plain_SGD_state_empty": not optimizer.state
        and not optimizer.state_dict()["state"],
        "stage1_gradients_cleared_before_handoff": all(
            parameters[name].grad is None for name in names
        ),
        "stage1_graph_references_cleared_before_handoff": local_state["total_loss"]
        is None
        and local_state["outputs"] is None,
        "stage1_execution_seed_exact": v13.SEED
        == v12.SEED
        == STAGE1_EXECUTION_SEED,
        "actor6_dimension_exact65793": cw11_flat.shape == (65793,),
        "stage2_pair_contract_present": isinstance(stage2_pair_contract, Mapping),
        "transition_checks_true": all(transition["checks"].values()),
    }
    if not all(handoff_checks.values()):
        raise ProtocolError(f"step29 handoff contract failed: {handoff_checks}")

    layout = cw20.actor_layout(parameters, names)
    accepted_actor_bytes = cw20.actor_bytes(parameters, names, np)
    accepted_actor_sha = hashlib.sha256(accepted_actor_bytes).hexdigest()
    accepted_model_sha = helper.model_state_sha256(model.state_dict())
    accepted_nonactor_sha = cw20.nonactor_sha(model, names, helper)
    accepted_delta = cw20.flat_actor(parameters, names, np) - cw11_flat
    accepted_delta_sha = float64_sha(accepted_delta)
    initial_checks = {
        "actor_sha_exact": accepted_actor_sha
        == v13.STAGE1_REFERENCE_ACTOR_SHA256,
        "model_sha_exact": accepted_model_sha
        == v13.STAGE1_REFERENCE_MODEL_SHA256,
        "delta_sha_exact": accepted_delta_sha
        == v13.STAGE1_REFERENCE_DELTA_SHA256,
        "delta_within_hard_radius": float(np.linalg.norm(accepted_delta))
        <= HARD_ACTUAL_RADIUS,
        "nonactor_exact_raw": accepted_nonactor_sha == cw22.RAW_NONACTOR_SHA256,
    }
    if not all(initial_checks.values()):
        raise ProtocolError(f"step29 actor state drift: {initial_checks}")
    initial_native_gate = stage1_trajectory[-1]["native_terminal_gate"]
    accepted_native_pf7 = float(
        initial_native_gate["PF_native"]["pf7_boundary"][
            "native_fixed_pair_margin"
        ]
    )
    cut_ledger = AppendOnlyCutLedger()
    v15_projection_ledger: list[dict[str, Any]] = []
    stage2_trajectory: list[dict[str, Any]] = []
    stage1_actor_hashes = set(str(value) for value in local_state["actor_hashes"])
    selected: dict[str, Any] | None = None
    terminal_reason = "STAGE2_32_NATIVE_TRIAL_BUDGET_EXHAUSTED"
    initial_bundle = fp32_structural_bundle(
        model=model,
        batch=batch,
        parameters=parameters,
        names=names,
        current_delta=accepted_delta,
        actor_sha256=accepted_actor_sha,
        specs=constraint_specs(
            rows=rows,
            teacher_policy_logits=teacher_policy_logits,
            stage2_pair_contract=stage2_pair_contract,
            dynamic_pairs=dynamic_pairs,
        ),
        teacher_policy_logits=teacher_policy_logits,
        contract=contract,
        stage2_pair_contract=stage2_pair_contract,
        ppo=ppo,
        v12=v12,
        v13=v13,
    )
    initial_identity_after_current_FP32 = verify_candidate_identity_exact(
        expected_actor_bytes=accepted_actor_bytes,
        expected_actor_sha256=accepted_actor_sha,
        expected_model_sha256=accepted_model_sha,
        expected_nonactor_sha256=accepted_nonactor_sha,
        parameters=parameters,
        names=names,
        model=model,
        helper=helper,
        cw20=cw20,
        optimizer=optimizer,
    )
    if not all(initial_identity_after_current_FP32.values()):
        restore_accepted_actor_exact(
            accepted_actor_bytes=accepted_actor_bytes,
            accepted_actor_sha256=accepted_actor_sha,
            accepted_model_sha256=accepted_model_sha,
            accepted_nonactor_sha256=accepted_nonactor_sha,
            accepted_delta_sha256=accepted_delta_sha,
            parameters=parameters,
            names=names,
            layout=layout,
            cw11_flat=cw11_flat,
            model=model,
            helper=helper,
            cw20=cw20,
            cw22=cw22,
            optimizer=optimizer,
        )
        raise ProtocolError(
            "accepted identity mutated during initial FP32 structural VJPs"
        )
    accepted_fp32_pf7 = float(initial_bundle["primary_pf7_margin"])
    shadow_ledger = Stage2ShadowLedger(
        accepted_actor_sha,
        accepted_fp32_pf7,
        accepted_native_pf7,
    )

    def serializable_cut_ledger() -> dict[str, Any]:
        ordered = cut_ledger.ordered()
        return {
            "append_only": True,
            "linearization_coordinate": "absolute_actor6_delta_from_CW11",
            "count": len(ordered),
            "keys": [cut.key.as_list() for cut in ordered],
            "records": [
                {
                    "key": cut.key.as_list(),
                    "target": float(cut.target),
                    "target_identity": target_identity(float(cut.target)),
                    "target_kinds": list(cut.target_kinds),
                    "origins": list(cut.origins),
                    "current_margin": float(cut.current_margin),
                    "gradient_l2": float(cut.gradient_norm),
                    "normalized_rhs": float(cut.rhs),
                    "normalized_normal_float64_le_sha256": float64_sha(
                        cut.normal
                    ),
                }
                for cut in ordered
            ],
        }

    def common_endpoint() -> dict[str, Any]:
        return {
            "context_checks": dict(local_state["context_checks"]),
            "dependency_checks": dict(local_state["dependency_checks"]),
            "cache": local_state["cache_audit"],
            "baseline_checks": contract["baseline_checks"],
            "objective_mask_counts": contract["mask_counts"],
            "baseline_loss": local_state["baseline_loss_audit"],
            "baseline_dynamic_pairs_added": [
                dict(value) for value in local_state["baseline_added"]
            ],
            "fixed_pair_contract": [dict(value) for value in contract["fixed_pairs"]],
            "zero_margin_contract": [
                dict(value) for value in contract["zero_contract"]
            ],
            "stage1_handoff": {
                "implementation": "single exact transformed-v13 step29 callback",
                "checks": handoff_checks,
                "initial_actor_checks": initial_checks,
                "optimizer_instances_created": 1,
                "optimizer_step_calls": STAGE1_REFERENCE_STEPS,
                "optimizer_discarded_before_stage2": True,
                "stage2_backward_calls": 0,
                "stage2_optimizer_step_calls": 0,
            },
            "stage1_transition": transition,
            "stage1_trajectory": stage1_trajectory,
            "stage2_trajectory": stage2_trajectory,
            "stage2_shadow_ledger": shadow_ledger.audit(),
            "absolute_cut_ledger": serializable_cut_ledger(),
            "v15_writeback_projection_ledger": list(v15_projection_ledger),
            "v15_pre_native_writeback_attempts_count_as_changed_train_shadows": False,
            "dynamic_pair_ledger_terminal": [dict(value) for value in dynamic_pairs],
            "changed_candidate_train_shadow_count": STAGE1_REFERENCE_STEPS
            + shadow_ledger.stage2_trial_count,
            "maximum_changed_train_shadows": TOTAL_CHANGED_TRAIN_SHADOW_CAP,
            "native_selection_forward_count_stage2": shadow_ledger.native_gate_forward_count,
            "verification_forward_count_stage2": 0,
        }

    first_bundle = initial_bundle
    for stage2_step in range(1, STAGE2_MAX_TRIALS + 1):
        global_step = STAGE1_REFERENCE_STEPS + stage2_step
        if shadow_ledger.terminal_decision is not None:
            raise ProtocolError("loop continued after terminal stage2 decision")
        if stage2_step == 1:
            bundle = first_bundle
            identity_after_current_FP32 = initial_identity_after_current_FP32
        else:
            bundle = fp32_structural_bundle(
                model=model,
                batch=batch,
                parameters=parameters,
                names=names,
                current_delta=accepted_delta,
                actor_sha256=accepted_actor_sha,
                specs=constraint_specs(
                    rows=rows,
                    teacher_policy_logits=teacher_policy_logits,
                    stage2_pair_contract=stage2_pair_contract,
                    dynamic_pairs=dynamic_pairs,
                ),
                teacher_policy_logits=teacher_policy_logits,
                contract=contract,
                stage2_pair_contract=stage2_pair_contract,
                ppo=ppo,
                v12=v12,
                v13=v13,
            )
            identity_after_current_FP32 = verify_candidate_identity_exact(
                expected_actor_bytes=accepted_actor_bytes,
                expected_actor_sha256=accepted_actor_sha,
                expected_model_sha256=accepted_model_sha,
                expected_nonactor_sha256=accepted_nonactor_sha,
                parameters=parameters,
                names=names,
                model=model,
                helper=helper,
                cw20=cw20,
                optimizer=optimizer,
            )
            if not all(identity_after_current_FP32.values()):
                restore_accepted_actor_exact(
                    accepted_actor_bytes=accepted_actor_bytes,
                    accepted_actor_sha256=accepted_actor_sha,
                    accepted_model_sha256=accepted_model_sha,
                    accepted_nonactor_sha256=accepted_nonactor_sha,
                    accepted_delta_sha256=accepted_delta_sha,
                    parameters=parameters,
                    names=names,
                    layout=layout,
                    cw11_flat=cw11_flat,
                    model=model,
                    helper=helper,
                    cw20=cw20,
                    cw22=cw22,
                    optimizer=optimizer,
                )
                raise ProtocolError(
                    "accepted identity mutated during current FP32 structural VJPs"
                )
        cuts_before_current = len(cut_ledger)
        cut_ledger.extend(bundle["cuts"])
        current_cuts_added = len(cut_ledger) - cuts_before_current
        ordered_cuts = cut_ledger.ordered()
        try:
            absolute_solution, qp_audit = solve_minimum_norm_active_set(
                ordered_cuts
            )
        except QPInfeasibleError as error:
            infeasible_certificate = dict(error.certificate)
            infeasible_matrix = np.stack(
                [np.asarray(cut.normal, dtype=np.float64) for cut in ordered_cuts],
                axis=0,
            )
            infeasible_rhs = np.asarray(
                [float(cut.rhs) for cut in ordered_cuts], dtype=np.float64
            )
            infeasible_certificate_checks = {
                "classification_exact_infeasible": infeasible_certificate.get(
                    "classification"
                )
                == "infeasible",
                "HiGHS_status_exact_2": infeasible_certificate.get("status") == 2,
                "success_exact_false": infeasible_certificate.get("success")
                is False,
                "solver_exact": infeasible_certificate.get("solver")
                == "scipy.optimize.linprog/HiGHS-row-space",
                "scipy_version_exact": infeasible_certificate.get("scipy_version")
                == EXPECTED_SCIPY_VERSION,
                "row_count_exact": infeasible_certificate.get(
                    "row_variable_count"
                )
                == len(ordered_cuts),
                "actor_dimension_exact": infeasible_certificate.get(
                    "original_variable_count"
                )
                == accepted_delta.size,
                "matrix_hash_exact": infeasible_certificate.get(
                    "matrix_float64_le_sha256"
                )
                == float64_sha(infeasible_matrix),
                "rhs_hash_exact": infeasible_certificate.get(
                    "rhs_float64_le_sha256"
                )
                == float64_sha(infeasible_rhs),
            }
            if not all(infeasible_certificate_checks.values()):
                raise ProtocolError(
                    "row-space infeasibility certificate drift: "
                    f"{infeasible_certificate_checks}"
                ) from error
            terminal_reason = "NO_GO_CERTIFIED_APPEND_ONLY_QP_INFEASIBLE"
            stage2_trajectory.append(
                {
                    "stage": 2,
                    "stage2_step": stage2_step,
                    "global_step": global_step,
                    "native_selection_forward_performed": False,
                    "pre_native_proposal_counts_as_changed_shadow": False,
                    "decision": "TERMINAL_NO_GO_BEFORE_NATIVE",
                    "reason": terminal_reason,
                    "absolute_cut_count": len(ordered_cuts),
                    "ordered_cut_keys": [
                        cut.key.as_list() for cut in ordered_cuts
                    ],
                    "rowspace_HiGHS_infeasibility_certificate": {
                        **infeasible_certificate,
                        "checks": infeasible_certificate_checks,
                    },
                    "accepted_actor_unchanged": True,
                    "identity_after_current_FP32": identity_after_current_FP32,
                }
            )
            break
        qp_ball_free_checks = {
            "solver_objective_ball_free": qp_audit["objective"]
            == "minimize_0p5_absolute_CW11_delta_l2_squared",
            "KKT_certificate_all_true": all(qp_audit["checks"].values()),
            "no_radius_constraint_in_solver": True,
            "solution_hash_exact": qp_audit["solution_float64_le_sha256"]
            == float64_sha(absolute_solution),
        }
        if not all(qp_ball_free_checks.values()):
            raise ProtocolError(f"ball-free QP certificate drift: {qp_ball_free_checks}")
        binding_indices = [int(value) for value in qp_audit["binding_indices"]]
        binding_rows = (
            np.stack([ordered_cuts[index].normal for index in binding_indices])
            if binding_indices
            else np.zeros((0, accepted_delta.size), dtype=np.float64)
        )
        planned_trial, composer = compose_trust_step(
            current_delta=accepted_delta,
            absolute_qp_solution=absolute_solution,
            primary_pf7_gradient=bundle["primary_pf7_gradient"],
            secondary_descent=bundle["secondary_descent"],
            binding_rows=binding_rows,
        )
        capped_target = absolute_solution * float(composer["absolute_target_scale"])
        cap_slacks = descriptive_cut_slacks(
            ordered_cuts,
            ball_free_solution=absolute_solution,
            capped_absolute_target=capped_target,
            planned_trial=planned_trial,
        )
        accepted_snapshot = {
            "actor_bytes": accepted_actor_bytes,
            "actor_sha256": accepted_actor_sha,
            "model_sha256": accepted_model_sha,
            "nonactor_sha256": accepted_nonactor_sha,
            "delta_sha256": accepted_delta_sha,
        }
        cw20.apply_flat_actor(
            parameters,
            names,
            cw11_flat + planned_trial,
            torch,
        )
        try:
            writeback = v15.v15_safe_project_actor6(
                parameters,
                names,
                cw11_flat,
                cw20,
                np,
                torch,
                v12,
                global_step,
                v15_projection_ledger,
            )
        except Exception as error:
            try:
                restore_accepted_actor_exact(
                    accepted_actor_bytes=accepted_actor_bytes,
                    accepted_actor_sha256=accepted_actor_sha,
                    accepted_model_sha256=accepted_model_sha,
                    accepted_nonactor_sha256=accepted_nonactor_sha,
                    accepted_delta_sha256=accepted_delta_sha,
                    parameters=parameters,
                    names=names,
                    layout=layout,
                    cw11_flat=cw11_flat,
                    model=model,
                    helper=helper,
                    cw20=cw20,
                    cw22=cw22,
                    optimizer=optimizer,
                )
            except Exception as restore_error:
                raise ProtocolError(
                    "fatal v15 writeback integrity failure and accepted restore failed"
                ) from restore_error
            raise
        actual_delta = np.asarray(writeback["actual_delta"], dtype=np.float64)
        actual_l2 = float(np.linalg.norm(actual_delta))
        actual_step_l2 = (
            float(np.linalg.norm(actual_delta - accepted_delta))
            if actual_delta.shape == accepted_delta.shape
            else math.inf
        )
        candidate_actor_bytes = cw20.actor_bytes(parameters, names, np)
        candidate_actor_sha = hashlib.sha256(candidate_actor_bytes).hexdigest()
        candidate_model_sha = helper.model_state_sha256(model.state_dict())
        candidate_nonactor_sha = cw20.nonactor_sha(model, names, helper)
        pre_native_integrity_checks = {
            "actual_delta_shape65793": actual_delta.shape == (65793,),
            "actual_delta_finite": bool(np.isfinite(actual_delta).all()),
            "actual_l2_within_hard": math.isfinite(actual_l2)
            and actual_l2 <= HARD_ACTUAL_RADIUS,
            "actual_step_within_eta_plus_writeback_tolerance": actual_step_l2
            <= TRUST_RADIUS + FLOAT32_WRITEBACK_STEP_TOL,
            "candidate_nonactor_exact": candidate_nonactor_sha
            == accepted_nonactor_sha
            == cw22.RAW_NONACTOR_SHA256,
            "actor_gradients_absent": all(
                parameters[name].grad is None for name in names
            ),
            "optimizer_state_still_empty": not optimizer.state
            and not optimizer.state_dict()["state"],
        }
        pre_native_novelty_checks = {
            "candidate_actor_unique": candidate_actor_sha
            not in stage1_actor_hashes
            and candidate_actor_sha not in shadow_ledger.trial_actor_hashes,
            "candidate_differs_from_accepted": candidate_actor_sha
            != accepted_actor_sha,
        }
        pre_native_checks = {
            **pre_native_integrity_checks,
            **pre_native_novelty_checks,
        }
        if not all(pre_native_integrity_checks.values()):
            restore_accepted_actor_exact(
                accepted_actor_bytes=accepted_actor_bytes,
                accepted_actor_sha256=accepted_actor_sha,
                accepted_model_sha256=accepted_model_sha,
                accepted_nonactor_sha256=accepted_nonactor_sha,
                accepted_delta_sha256=accepted_delta_sha,
                parameters=parameters,
                names=names,
                layout=layout,
                cw11_flat=cw11_flat,
                model=model,
                helper=helper,
                cw20=cw20,
                cw22=cw22,
                optimizer=optimizer,
            )
            raise ProtocolError(
                "fatal pre-native writeback integrity failure after verified restore: "
                f"{pre_native_integrity_checks}"
            )
        if not all(pre_native_novelty_checks.values()):
            restore = restore_accepted_actor_exact(
                accepted_actor_bytes=accepted_actor_bytes,
                accepted_actor_sha256=accepted_actor_sha,
                accepted_model_sha256=accepted_model_sha,
                accepted_nonactor_sha256=accepted_nonactor_sha,
                accepted_delta_sha256=accepted_delta_sha,
                parameters=parameters,
                names=names,
                layout=layout,
                cw11_flat=cw11_flat,
                model=model,
                helper=helper,
                cw20=cw20,
                cw22=cw22,
                optimizer=optimizer,
            )
            terminal_reason = "PRE_NATIVE_CANDIDATE_STALLED_RESTORED"
            stage2_trajectory.append(
                {
                    "stage": 2,
                    "stage2_step": stage2_step,
                    "global_step": global_step,
                    "native_selection_forward_performed": False,
                    "pre_native_proposal_counts_as_changed_shadow": False,
                    "decision": "TERMINAL_NO_GO_BEFORE_NATIVE",
                    "reason": terminal_reason,
                    "pre_native_checks": pre_native_checks,
                    "pre_native_integrity_checks": pre_native_integrity_checks,
                    "pre_native_novelty_checks": pre_native_novelty_checks,
                    "restore": restore,
                    "QP_ball_free_certificate": {
                        "certified": True,
                        "checks": qp_ball_free_checks,
                        "solver": qp_audit,
                    },
                    "cap_slack_diagnostic": cap_slacks,
                }
            )
            break

        # Candidate FP32 evidence is frozen before the sole native selection
        # forward.  Any mutation or VJP failure is therefore a pre-native
        # proposal failure and consumes no changed train-shadow.
        candidate_bundle = fp32_structural_bundle(
            model=model,
            batch=batch,
            parameters=parameters,
            names=names,
            current_delta=actual_delta,
            actor_sha256=candidate_actor_sha,
            specs=constraint_specs(
                rows=rows,
                teacher_policy_logits=teacher_policy_logits,
                stage2_pair_contract=stage2_pair_contract,
                dynamic_pairs=dynamic_pairs,
            ),
            teacher_policy_logits=teacher_policy_logits,
            contract=contract,
            stage2_pair_contract=stage2_pair_contract,
            ppo=ppo,
            v12=v12,
            v13=v13,
        )
        candidate_fp32_pf7 = float(candidate_bundle["primary_pf7_margin"])
        identity_after_candidate_FP32 = verify_candidate_identity_exact(
            expected_actor_bytes=candidate_actor_bytes,
            expected_actor_sha256=candidate_actor_sha,
            expected_model_sha256=candidate_model_sha,
            expected_nonactor_sha256=candidate_nonactor_sha,
            parameters=parameters,
            names=names,
            model=model,
            helper=helper,
            cw20=cw20,
            optimizer=optimizer,
        )
        if not all(identity_after_candidate_FP32.values()):
            restore_accepted_actor_exact(
                accepted_actor_bytes=accepted_snapshot["actor_bytes"],
                accepted_actor_sha256=accepted_snapshot["actor_sha256"],
                accepted_model_sha256=accepted_snapshot["model_sha256"],
                accepted_nonactor_sha256=accepted_snapshot["nonactor_sha256"],
                accepted_delta_sha256=accepted_snapshot["delta_sha256"],
                parameters=parameters,
                names=names,
                layout=layout,
                cw11_flat=cw11_flat,
                model=model,
                helper=helper,
                cw20=cw20,
                cw22=cw22,
                optimizer=optimizer,
            )
            raise ProtocolError(
                "candidate identity mutated during pre-native FP32 structural VJPs"
            )

        with torch.no_grad():
            native_outputs = ppo.model_forward(model, batch, device)
            current = v12.evaluate_snapshot(
                native_outputs, batch, rows, ppo, cw22
            )
            _, native_loss = v13.stage2_objective(
                native_outputs,
                batch,
                teacher_policy_logits,
                contract,
                stage2_pair_contract,
                dynamic_pairs,
                ppo,
                v12,
            )
        changed_names = [
            name for name in names if not torch.equal(parameters[name], cw11_actor[name])
        ]
        gate = v12.terminal_gate(
            outputs=native_outputs,
            rows=rows,
            current=current,
            baseline=baseline,
            baseline_native=baseline_native,
            contract=contract,
            base_soft_loss=float(native_loss["base_soft_loss"]),
            baseline_base_soft_loss=float(local_state["baseline_base_soft_loss"]),
            actual_delta=actual_delta,
            candidate_hash=candidate_model_sha,
            changed_names=changed_names,
            model=model,
            parameters=parameters,
            helper=helper,
            cw20=cw20,
            cw22=cw22,
        )
        identity_after_native_gate = verify_candidate_identity_exact(
            expected_actor_bytes=candidate_actor_bytes,
            expected_actor_sha256=candidate_actor_sha,
            expected_model_sha256=candidate_model_sha,
            expected_nonactor_sha256=candidate_nonactor_sha,
            parameters=parameters,
            names=names,
            model=model,
            helper=helper,
            cw20=cw20,
            optimizer=optimizer,
        )
        if not all(identity_after_native_gate.values()):
            restore_accepted_actor_exact(
                accepted_actor_bytes=accepted_snapshot["actor_bytes"],
                accepted_actor_sha256=accepted_snapshot["actor_sha256"],
                accepted_model_sha256=accepted_snapshot["model_sha256"],
                accepted_nonactor_sha256=accepted_snapshot["nonactor_sha256"],
                accepted_delta_sha256=accepted_snapshot["delta_sha256"],
                parameters=parameters,
                names=names,
                layout=layout,
                cw11_flat=cw11_flat,
                model=model,
                helper=helper,
                cw20=cw20,
                cw22=cw22,
                optimizer=optimizer,
            )
            raise ProtocolError("candidate identity mutated during native gate forward")
        candidate_native_pf7 = float(
            gate["PF_native"]["pf7_boundary"]["native_fixed_pair_margin"]
        )
        all_non_pf7 = v13.pf0_ab_pass(gate) and all(
            value
            for name, value in gate["checks"].items()
            if name != "three_PF_native_order_set_and_margin"
        )
        complete_native = gate["pass"] is True
        # First complete frozen-v12 native gate is authoritative and terminal.
        # No diagnostics, dynamic discovery, or additional FP32 forward/VJP is
        # permitted before recording GO and stopping selection.
        if complete_native:
            ledger_decision = shadow_ledger.record_native_trial(
                actor_sha256=candidate_actor_sha,
                fp32_pf7_margin=candidate_fp32_pf7,
                native_pf7_cell=candidate_native_pf7,
                all_non_pf7_hard_gates_pass=all_non_pf7,
                complete_native_gate_pass=True,
                violated_cuts_added=0,
                restored_after_reject=False,
            )
            if ledger_decision != "GO":
                raise ProtocolError("authoritative native GO did not terminate ledger")
            stage2_trajectory.append(
                {
                    "stage": 2,
                    "stage2_step": stage2_step,
                    "global_step": global_step,
                    "candidate_model_state_sha256": candidate_model_sha,
                    "candidate_actor_float32_le_sha256": candidate_actor_sha,
                    "candidate_actual_delta_float64_le_sha256": float64_sha(
                        actual_delta
                    ),
                    "candidate_actual_l2": actual_l2,
                    "actual_step_from_accepted_l2": actual_step_l2,
                    "current_point_cuts_added": current_cuts_added,
                    "FP32_current_structural": bundle["audit"],
                    "identity_after_current_FP32": identity_after_current_FP32,
                    "FP32_candidate_structural_pre_native": candidate_bundle[
                        "audit"
                    ],
                    "QP_ball_free_certificate": {
                        "certified": True,
                        "checks": qp_ball_free_checks,
                        "solver": qp_audit,
                    },
                    "trust_step_composer": composer,
                    "cap_slack_diagnostic": cap_slacks,
                    "v15_writeback": {
                        key: value
                        for key, value in writeback.items()
                        if key != "actual_delta"
                    },
                    "pre_native_checks": pre_native_checks,
                    "identity_after_candidate_FP32": identity_after_candidate_FP32,
                    "native_selection_forward_performed": True,
                    "native_selection_forward_ordinal": shadow_ledger.stage2_trial_count,
                    "native_terminal_gate": gate,
                    "native_loss": native_loss,
                    "identity_after_native_gate": identity_after_native_gate,
                    "all_non_PF7_hard_gates_pass": all_non_pf7,
                    "complete_native_gate_pass": True,
                    "post_native_diagnostics_evaluated": False,
                    "post_native_dynamic_discovery_evaluated": False,
                    "post_native_additional_FP32_VJP_evaluated": False,
                    "decision": "GO_FIRST_COMPLETE_NATIVE_GATE",
                }
            )
            selected = {
                "global_step": global_step,
                "stage2_step": stage2_step,
                "actor_bytes": candidate_actor_bytes,
                "actor_sha256": candidate_actor_sha,
                "model_sha256": candidate_model_sha,
                "actual_delta_sha256": float64_sha(actual_delta),
                "actual_l2": actual_l2,
                "changed_names": changed_names,
                "gate": gate,
                "loss": native_loss,
                "dynamic_pairs": [dict(value) for value in dynamic_pairs],
            }
            terminal_reason = "FIRST_COMPLETE_FROZEN_V12_NATIVE_GATE"
            break

        predecision_record = {
            "stage": 2,
            "stage2_step": stage2_step,
            "global_step": global_step,
            "candidate_model_state_sha256": candidate_model_sha,
            "candidate_actor_float32_le_sha256": candidate_actor_sha,
            "candidate_actual_delta_float64_le_sha256": float64_sha(actual_delta),
            "candidate_actual_l2": actual_l2,
            "actual_step_from_accepted_l2": actual_step_l2,
            "current_point_cuts_added": current_cuts_added,
            "FP32_current_structural": bundle["audit"],
            "identity_after_current_FP32": identity_after_current_FP32,
            "FP32_candidate_structural_pre_native": candidate_bundle["audit"],
            "QP_ball_free_certificate": {
                "certified": True,
                "checks": qp_ball_free_checks,
                "solver": qp_audit,
            },
            "trust_step_composer": composer,
            "cap_slack_diagnostic": cap_slacks,
            "v15_writeback": {
                key: value for key, value in writeback.items() if key != "actual_delta"
            },
            "pre_native_checks": pre_native_checks,
            "identity_after_candidate_FP32": identity_after_candidate_FP32,
            "native_selection_forward_performed": True,
            "native_selection_forward_ordinal": shadow_ledger.stage2_trial_count + 1,
            "native_terminal_gate": gate,
            "native_loss": native_loss,
            "identity_after_native_gate": identity_after_native_gate,
            "all_non_PF7_hard_gates_pass": all_non_pf7,
            "complete_native_gate_pass": False,
        }

        fatal_gate_names = {
            "native_policy_BF16",
            "count_logits_native_exact_CW11",
            "value_logits_native_exact_CW11",
            "actual_actor_delta_finite",
            "actual_actor_delta_within_hard_radius",
            "candidate_changed",
            "changed_scope_actor6_only",
            "nonactor_exact_raw",
            "candidate_model_all_finite",
            "base_soft_loss_finite",
        }
        fatal_gate_failures = sorted(
            name for name in fatal_gate_names if gate["checks"].get(name) is not True
        )
        if fatal_gate_failures:
            restore_accepted_actor_exact(
                accepted_actor_bytes=accepted_snapshot["actor_bytes"],
                accepted_actor_sha256=accepted_snapshot["actor_sha256"],
                accepted_model_sha256=accepted_snapshot["model_sha256"],
                accepted_nonactor_sha256=accepted_snapshot["nonactor_sha256"],
                accepted_delta_sha256=accepted_snapshot["delta_sha256"],
                parameters=parameters,
                names=names,
                layout=layout,
                cw11_flat=cw11_flat,
                model=model,
                helper=helper,
                cw20=cw20,
                cw22=cw22,
                optimizer=optimizer,
            )
            raise ProtocolError(
                f"fatal native gate invariant failed after verified restore: {fatal_gate_failures}"
            )
        if gate["checks"].get("base_soft_loss_improves_by_floor") is not True:
            restore = restore_accepted_actor_exact(
                accepted_actor_bytes=accepted_snapshot["actor_bytes"],
                accepted_actor_sha256=accepted_snapshot["actor_sha256"],
                accepted_model_sha256=accepted_snapshot["model_sha256"],
                accepted_nonactor_sha256=accepted_snapshot["nonactor_sha256"],
                accepted_delta_sha256=accepted_snapshot["delta_sha256"],
                parameters=parameters,
                names=names,
                layout=layout,
                cw11_flat=cw11_flat,
                model=model,
                helper=helper,
                cw20=cw20,
                cw22=cw22,
                optimizer=optimizer,
            )
            ledger_decision = shadow_ledger.record_native_trial(
                actor_sha256=candidate_actor_sha,
                fp32_pf7_margin=candidate_fp32_pf7,
                native_pf7_cell=candidate_native_pf7,
                all_non_pf7_hard_gates_pass=False,
                complete_native_gate_pass=False,
                violated_cuts_added=0,
                restored_after_reject=True,
                terminal_no_go_no_new_cut_reason=(
                    "NONPAIR_HARD_GATE_FAILURE_NO_LEGAL_STRUCTURAL_CUT"
                ),
            )
            if ledger_decision != "REJECT_RESTORE_TERMINAL_NO_GO":
                raise ProtocolError("base-soft uncuttable terminal decision drift")
            stage2_trajectory.append(
                {
                    **predecision_record,
                    "decision": ledger_decision,
                    "fatal_gate_failures": [],
                    "uncuttable_gate_failure": "base_soft_loss_improves_by_floor",
                    "restore": restore,
                    "violated_new_cuts_added_after_restore": 0,
                    "terminal_no_new_cut_reason": (
                        "NONPAIR_HARD_GATE_FAILURE_NO_LEGAL_STRUCTURAL_CUT"
                    ),
                }
            )
            terminal_reason = "NO_GO_STALLED_UNCUTTABLE_BASE_SOFT_FLOOR"
            break

        diagnostics = v12.soft_floor_diagnostics(rows, baseline, current, cw22)
        candidates = v13.stage2_obligations(
            native_outputs, rows, baseline, current, contract, v1, v12
        )
        dynamic_added = v12.merge_dynamic_pairs(
            dynamic_pairs, candidates, global_step
        )
        v13.attach_dynamic_temperatures(dynamic_pairs)
        updated_candidate_bundle = fp32_structural_bundle(
            model=model,
            batch=batch,
            parameters=parameters,
            names=names,
            current_delta=actual_delta,
            actor_sha256=candidate_actor_sha,
            specs=constraint_specs(
                rows=rows,
                teacher_policy_logits=teacher_policy_logits,
                stage2_pair_contract=stage2_pair_contract,
                dynamic_pairs=dynamic_pairs,
            ),
            teacher_policy_logits=teacher_policy_logits,
            contract=contract,
            stage2_pair_contract=stage2_pair_contract,
            ppo=ppo,
            v12=v12,
            v13=v13,
        )
        identity_after_post_native_FP32 = verify_candidate_identity_exact(
            expected_actor_bytes=candidate_actor_bytes,
            expected_actor_sha256=candidate_actor_sha,
            expected_model_sha256=candidate_model_sha,
            expected_nonactor_sha256=candidate_nonactor_sha,
            parameters=parameters,
            names=names,
            model=model,
            helper=helper,
            cw20=cw20,
            optimizer=optimizer,
        )
        if not all(identity_after_post_native_FP32.values()):
            restore_accepted_actor_exact(
                accepted_actor_bytes=accepted_snapshot["actor_bytes"],
                accepted_actor_sha256=accepted_snapshot["actor_sha256"],
                accepted_model_sha256=accepted_snapshot["model_sha256"],
                accepted_nonactor_sha256=accepted_snapshot["nonactor_sha256"],
                accepted_delta_sha256=accepted_snapshot["delta_sha256"],
                parameters=parameters,
                names=names,
                layout=layout,
                cw11_flat=cw11_flat,
                model=model,
                helper=helper,
                cw20=cw20,
                cw22=cw22,
                optimizer=optimizer,
            )
            raise ProtocolError(
                "candidate identity mutated during post-native FP32 structural VJPs"
            )
        candidate_bundle = updated_candidate_bundle
        candidate_fp32_pf7 = float(candidate_bundle["primary_pf7_margin"])
        heuristic_accept = (
            all_non_pf7
            and candidate_fp32_pf7
            >= shadow_ledger.accepted_fp32_pf7_margin - FP32_PROGRESS_TOL
            and candidate_native_pf7 >= shadow_ledger.accepted_native_pf7_cell
        )
        base_trial_record = {
            **predecision_record,
            "FP32_candidate_structural_post_native_gate_false": candidate_bundle[
                "audit"
            ],
            "identity_after_post_native_FP32": identity_after_post_native_FP32,
            "soft_floor_diagnostics": diagnostics,
            "dynamic_pairs_added": [dict(value) for value in dynamic_added],
            "dynamic_pair_count": len(dynamic_pairs),
            "continuation_heuristic_accept": heuristic_accept,
        }
        if heuristic_accept:
            ledger_decision = shadow_ledger.record_native_trial(
                actor_sha256=candidate_actor_sha,
                fp32_pf7_margin=candidate_fp32_pf7,
                native_pf7_cell=candidate_native_pf7,
                all_non_pf7_hard_gates_pass=all_non_pf7,
                complete_native_gate_pass=False,
                violated_cuts_added=0,
                restored_after_reject=False,
            )
            if ledger_decision != "ACCEPT_CONTINUE":
                raise ProtocolError("accepted continuation ledger decision drift")
            stage2_trajectory.append(
                {**base_trial_record, "decision": "ACCEPT_CONTINUE"}
            )
            accepted_actor_bytes = candidate_actor_bytes
            accepted_actor_sha = candidate_actor_sha
            accepted_model_sha = candidate_model_sha
            accepted_nonactor_sha = candidate_nonactor_sha
            accepted_delta = actual_delta.copy()
            accepted_delta_sha = float64_sha(accepted_delta)
            continue

        violated_candidate_cuts = [
            cut
            for cut, record in zip(
                candidate_bundle["cuts"], candidate_bundle["records"]
            )
            if float(record["FP32_margin"]) < float(record["target"])
        ]
        restore = restore_accepted_actor_exact(
            accepted_actor_bytes=accepted_snapshot["actor_bytes"],
            accepted_actor_sha256=accepted_snapshot["actor_sha256"],
            accepted_model_sha256=accepted_snapshot["model_sha256"],
            accepted_nonactor_sha256=accepted_snapshot["nonactor_sha256"],
            accepted_delta_sha256=accepted_snapshot["delta_sha256"],
            parameters=parameters,
            names=names,
            layout=layout,
            cw11_flat=cw11_flat,
            model=model,
            helper=helper,
            cw20=cw20,
            cw22=cw22,
            optimizer=optimizer,
        )
        cut_count_before_reject = len(cut_ledger)
        cut_ledger.extend(violated_candidate_cuts)
        violated_cuts_added = len(cut_ledger) - cut_count_before_reject
        terminal_no_cut_reason = None
        if violated_cuts_added == 0:
            terminal_no_cut_reason = (
                "LEGAL_STRUCTURAL_CUT_COLLISION_NO_NEW_GEOMETRY"
                if violated_candidate_cuts
                else "NONPAIR_HARD_GATE_FAILURE_NO_LEGAL_STRUCTURAL_CUT"
            )
        ledger_decision = shadow_ledger.record_native_trial(
            actor_sha256=candidate_actor_sha,
            fp32_pf7_margin=candidate_fp32_pf7,
            native_pf7_cell=candidate_native_pf7,
            all_non_pf7_hard_gates_pass=all_non_pf7,
            complete_native_gate_pass=False,
            violated_cuts_added=violated_cuts_added,
            restored_after_reject=True,
            terminal_no_go_no_new_cut_reason=terminal_no_cut_reason,
        )
        stage2_trajectory.append(
            {
                **base_trial_record,
                "decision": ledger_decision,
                "violated_structural_cuts_observed": len(
                    violated_candidate_cuts
                ),
                "violated_new_cuts_added_after_restore": violated_cuts_added,
                "restore": restore,
                "terminal_no_new_cut_reason": terminal_no_cut_reason,
            }
        )
        if ledger_decision == "REJECT_RESTORE_TERMINAL_NO_GO":
            terminal_reason = "NO_GO_STALLED_NO_NEW_LEGAL_STRUCTURAL_CUT"
            break
        if ledger_decision != "REJECT_RESTORE_CONTINUE":
            raise ProtocolError("rejected continuation ledger decision drift")

    if selected is None:
        if shadow_ledger.terminal_decision is None:
            shadow_ledger.mark_terminal_no_go_without_native_trial(
                terminal_reason
            )
        if shadow_ledger.terminal_decision != "NO_GO":
            raise ProtocolError("NO_GO endpoint lacks typed terminal ledger state")
        endpoint = {
            "decision": "NO_GO_CW24_V17_FP32_ABSOLUTE_QP_TRAIN_GATE",
            "reason": terminal_reason,
            **common_endpoint(),
            "candidate_payload": None,
        }
        endpoint["NO_GO_material_audit"] = no_go_material_audit(endpoint)
        return endpoint

    if shadow_ledger.terminal_decision != "GO" or shadow_ledger.go_count != 1:
        raise ProtocolError("selected endpoint lacks unique terminal GO state")
    payload = {
        "anchor": {
            "reconstruction_base": "original_raw_U468",
            "raw_checkpoint": str(cw23.RAW.relative_to(ROOT)),
            "raw_checkpoint_file_sha256": cw23.RAW_SHA256,
            "raw_model_state_sha256": cw22.RAW_MODEL_SHA256,
            "CW11_provenance_model_state_sha256": cw22.CW11_MODEL_SHA256,
            "CW11_provenance_vector_float64_le_sha256": cw22.CW11_VECTOR_SHA256,
            "CW11_provenance_active_pair_ledger_sha256": cw22.CW11_LEDGER_SHA256,
            "stage1_reference_actor_sha256": v13.STAGE1_REFERENCE_ACTOR_SHA256,
            "terminal_model_state_sha256": selected["model_sha256"],
        },
        "formula": (
            "load_original_raw_U468_then_replace_all_six_absolute_actor_float32_"
            "tensors_from_v17_payload"
        ),
        "actor_names": list(names),
        "changed_actor_names": selected["changed_names"],
        "actor_layout": layout,
        "actor_layout_sha256": hashlib.sha256(
            json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "candidate_actor_float32_le": cw20.xz_payload(selected["actor_bytes"]),
        "actual_additional_from_CW11_l2": selected["actual_l2"],
        "actual_delta_float64_le_sha256": selected["actual_delta_sha256"],
        "selected_global_step": selected["global_step"],
        "selected_stage2_step": selected["stage2_step"],
        "selection_sha256": v1.ROWS_CANONICAL_SHA256,
        "cache_sha256": local_state["cache_audit"]["cache_sha256"],
    }
    modules["cutting"].restore_raw_actor(
        modules["ram"], parameter_sequence, context["raw_actor"], torch
    )
    raw_restored_hash = helper.model_state_sha256(model.state_dict())
    decoded = cw20.decode_xz(payload["candidate_actor_float32_le"])
    cw20.copy_actor_bytes(parameters, names, layout, decoded, np, torch)
    reconstructed_hash = helper.model_state_sha256(model.state_dict())
    replay_flat = cw20.flat_actor(parameters, names, np)
    replay_delta = replay_flat - cw11_flat
    with torch.no_grad():
        replay_outputs = ppo.model_forward(model, batch, device)
        replay_snapshot = v12.evaluate_snapshot(
            replay_outputs, batch, rows, ppo, cw22
        )
        _, replay_loss = v13.stage2_objective(
            replay_outputs,
            batch,
            teacher_policy_logits,
            contract,
            stage2_pair_contract,
            selected["dynamic_pairs"],
            ppo,
            v12,
        )
    replay_changed = [
        name for name in names if not torch.equal(parameters[name], cw11_actor[name])
    ]
    replay_gate = v12.terminal_gate(
        outputs=replay_outputs,
        rows=rows,
        current=replay_snapshot,
        baseline=baseline,
        baseline_native=baseline_native,
        contract=contract,
        base_soft_loss=float(replay_loss["base_soft_loss"]),
        baseline_base_soft_loss=float(local_state["baseline_base_soft_loss"]),
        actual_delta=replay_delta,
        candidate_hash=reconstructed_hash,
        changed_names=replay_changed,
        model=model,
        parameters=parameters,
        helper=helper,
        cw20=cw20,
        cw22=cw22,
    )
    replay_identity_after_forward = verify_candidate_identity_exact(
        expected_actor_bytes=selected["actor_bytes"],
        expected_actor_sha256=selected["actor_sha256"],
        expected_model_sha256=selected["model_sha256"],
        expected_nonactor_sha256=cw22.RAW_NONACTOR_SHA256,
        parameters=parameters,
        names=names,
        model=model,
        helper=helper,
        cw20=cw20,
        optimizer=optimizer,
    )
    reconstruction_checks = {
        "raw_U468_restore_exact": raw_restored_hash == cw22.RAW_MODEL_SHA256,
        "candidate_hash_exact": reconstructed_hash == selected["model_sha256"],
        "candidate_actor_bytes_exact": cw20.actor_bytes(parameters, names, np)
        == selected["actor_bytes"],
        "actual_delta_sha_exact": float64_sha(replay_delta)
        == selected["actual_delta_sha256"],
        "nonactor_exact_raw": cw20.nonactor_sha(model, names, helper)
        == cw22.RAW_NONACTOR_SHA256,
        "repeat_native_terminal_gate": replay_gate["pass"] is True,
        "repeat_native_terminal_gate_full_equality": replay_gate
        == selected["gate"],
        "repeat_stage2_loss_exact": replay_loss == selected["loss"],
        "identity_exact_after_verification_forward": all(
            replay_identity_after_forward.values()
        ),
    }
    if not all(reconstruction_checks.values()):
        raise ProtocolError(
            f"v17 payload reconstruction verification failed: {reconstruction_checks}"
        )
    endpoint = {
        "decision": "GO_CW24_V17_FP32_ABSOLUTE_QP_TRAIN_GATE",
        "reason": terminal_reason,
        **common_endpoint(),
        "verification_forward_count_stage2": 1,
        "pure_payload_reconstruction": {
            "selection_forward": False,
            "same_selected_actor_hash": selected["actor_sha256"],
            "checks": reconstruction_checks,
            "pass": True,
            "repeat_native_terminal_gate": replay_gate,
            "repeat_stage2_loss": replay_loss,
            "identity_after_verification_forward": replay_identity_after_forward,
        },
        "candidate_payload": payload,
    }
    return endpoint


def validate_claim_binding_before_module_exec(
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = validate_pre_cuda_runtime()
    source = self_evidence(require_frozen=True)
    dependencies = dependency_evidence()
    transform = transform_v13_source()[1]
    if not isinstance(claim.get("payload"), Mapping) or not isinstance(
        claim.get("publication"), Mapping
    ):
        raise ProtocolError("in-memory attempt claim shape drift")
    claim_payload = claim["payload"]
    claim_publication = claim["publication"]
    attempt_evidence = immutable_file_evidence(
        "current_v17_attempt",
        ATTEMPT_MARKER,
        str(claim_publication["sha256"]),
        0o444,
    )
    attempt_before = ATTEMPT_MARKER.lstat()
    attempt_raw = ATTEMPT_MARKER.read_bytes()
    attempt_after = ATTEMPT_MARKER.lstat()
    attempt_payload = load_json_no_duplicates(
        attempt_raw, "current v17 attempt before module exec"
    )
    fresh_transform_summary = transform
    fresh_dependency_summaries = {
        name: binding_summary(record) for name, record in dependencies.items()
    }
    checks = {
        "attempt_payload_exact_in_memory_claim": attempt_payload == claim_payload,
        "attempt_canonical_bytes_exact": attempt_raw == canonical_json(claim_payload),
        "attempt_publication_binding_exact": binding_summary(attempt_evidence)
        == binding_summary(claim_publication),
        "attempt_identity_stable": (
            attempt_before.st_dev,
            attempt_before.st_ino,
            attempt_before.st_size,
            attempt_before.st_mtime_ns,
        )
        == (
            attempt_after.st_dev,
            attempt_after.st_ino,
            attempt_after.st_size,
            attempt_after.st_mtime_ns,
        ),
        "attempt_identity_exact_claim": (
            int(attempt_after.st_dev),
            int(attempt_after.st_ino),
            int(attempt_after.st_nlink),
        )
        == (
            int(claim_publication["device"]),
            int(claim_publication["inode"]),
            int(claim_publication["nlink"]),
        ),
        "runtime_exact_claim_and_fresh": runtime == claim_payload.get("runtime"),
        "source_exact_claim_and_fresh": binding_summary(source)
        == claim_payload.get("source"),
        "dependencies_exact_claim_and_fresh": fresh_dependency_summaries
        == claim_payload.get("dependencies"),
        "transform_exact_claim_and_fresh": fresh_transform_summary
        == claim_payload.get("step29_handoff_transform"),
        "output_absent_before_any_module_exec": path_absent(OUTPUT),
        "failure_absent_before_any_module_exec": path_absent(FAILURE),
        "torch_still_not_imported": "torch" not in sys.modules,
    }
    if not all(checks.values()):
        raise ProtocolError(f"attempt-to-execution binding drift: {checks}")
    return {
        "runtime": runtime,
        "source": source,
        "dependencies": dependencies,
        "transform": transform,
        "attempt": attempt_evidence,
        "checks": checks,
    }


def run_v17_production(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Run the one-shot train-only protocol after its pre-CUDA attempt claim."""

    pre_module_binding = validate_claim_binding_before_module_exec(claim)
    pre_cuda_runtime = pre_module_binding["runtime"]
    source = pre_module_binding["source"]
    dependencies = pre_module_binding["dependencies"]
    v15, v15_evidence = load_locked_module(
        V15_SOURCE,
        V15_SOURCE_SHA256,
        0o555,
        "cw24_v17_frozen_v15_projector",
    )

    def handoff(local_state: Mapping[str, Any]) -> dict[str, Any]:
        return run_stage2_handoff(local_state, v15=v15)

    v13, transform_audit = load_transformed_v13_module(handoff)
    if transform_audit != pre_module_binding["transform"]:
        raise ProtocolError("step29 transform drifted across first module exec boundary")
    reference_payload, reference_audit = v13.load_v12_reference()
    v12, v12_evidence = v13.load_locked_module(
        v13.V12_SOURCE,
        v13.V12_SOURCE_SHA256,
        v13.V12_SOURCE_MODE,
        "cw24_v17_frozen_v12",
    )
    v1, v1_evidence = v12.load_locked_module(
        v12.V1,
        v12.V1_SHA256,
        v12.SOURCE_MODE,
        "cw24_v17_frozen_v1",
    )
    rows, selection_audit = v1.load_selection()
    cw23, cw23_evidence = v1.load_locked_module(
        v1.CW23,
        v1.CW23_SHA256,
        v1.CW23_MODE,
        "cw24_v17_frozen_cw23",
    )
    cw22, cw22_evidence = cw23.load_cw22()
    original_core = cw22.run_targeted_core
    original_seed = cw22.SEED
    callback_calls = 0

    def bound_core(
        context: Mapping[str, Any],
        ignored_rows: Sequence[Mapping[str, Any]],
        cw20: ModuleType,
        cw19: ModuleType,
        cw15: ModuleType,
        modules: Mapping[str, ModuleType],
    ) -> dict[str, Any]:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls != 1:
            raise ProtocolError("v17 CW22 callback executed more than once")
        if len(ignored_rows) != 256 or len(rows) != 352:
            raise ProtocolError("frozen CW22/B352 callback row contract drift")
        return v13.run_two_stage_core(
            context,
            rows,
            cw20,
            cw19,
            cw15,
            modules,
            cw22=cw22,
            cw23=cw23,
            v1=v1,
            v12=v12,
            reference_payload=reference_payload,
            reference_audit=reference_audit,
        )

    cw22.run_targeted_core = bound_core
    cw22.SEED = STAGE1_EXECUTION_SEED
    try:
        result = cw22.production_run()
    finally:
        cw22.run_targeted_core = original_core
        cw22.SEED = original_seed

    post_source = self_evidence(require_frozen=True)
    post_dependencies = dependency_evidence()
    post_transform = transform_v13_source()[1]
    attempt_before = ATTEMPT_MARKER.lstat()
    attempt_raw = ATTEMPT_MARKER.read_bytes()
    attempt_payload = load_json_no_duplicates(attempt_raw, "v17 attempt marker")
    attempt_stat = ATTEMPT_MARKER.lstat()
    post_run_checks = {
        "v17_source_byte_identity": post_source == source,
        "all_dependencies_byte_identity": post_dependencies == dependencies,
        "step29_handoff_transform_identity": post_transform == transform_audit,
        "CW22_callback_executed_once": callback_calls == 1,
        "CW22_core_restored": cw22.run_targeted_core is original_core,
        "CW22_seed_restored": cw22.SEED == original_seed,
        "attempt_payload_exact": attempt_payload == claim["payload"],
        "attempt_sha_exact": hashlib.sha256(attempt_raw).hexdigest()
        == claim["publication"]["sha256"],
        "attempt_identity_stable": (
            attempt_before.st_dev,
            attempt_before.st_ino,
            attempt_before.st_size,
            attempt_before.st_mtime_ns,
        )
        == (
            attempt_stat.st_dev,
            attempt_stat.st_ino,
            attempt_stat.st_size,
            attempt_stat.st_mtime_ns,
        ),
        "attempt_identity_exact_claim": (
            int(attempt_stat.st_dev),
            int(attempt_stat.st_ino),
            int(attempt_stat.st_nlink),
        )
        == (
            int(claim["publication"]["device"]),
            int(claim["publication"]["inode"]),
            int(claim["publication"]["nlink"]),
        ),
        "attempt_mode_0444": stat.S_IMODE(attempt_stat.st_mode) == 0o444,
        "attempt_regular_single_link": stat.S_ISREG(attempt_stat.st_mode)
        and not stat.S_ISLNK(attempt_stat.st_mode)
        and int(attempt_stat.st_nlink) == 1,
        "output_absent_before_publication": path_absent(OUTPUT),
        "failure_absent_before_publication": path_absent(FAILURE),
    }
    if not all(post_run_checks.values()):
        raise ProtocolError(f"v17 post-run identity drift: {post_run_checks}")
    if not isinstance(result, dict) or not isinstance(result.get("endpoint"), dict):
        raise ProtocolError("CW22 returned malformed v17 result")
    endpoint = result["endpoint"]
    decision = endpoint.get("decision")
    allowed_decisions = {
        "GO_CW24_V17_FP32_ABSOLUTE_QP_TRAIN_GATE",
        "NO_GO_CW24_V17_FP32_ABSOLUTE_QP_TRAIN_GATE",
    }
    changed = endpoint.get("changed_candidate_train_shadow_count")
    stage2_native = endpoint.get("native_selection_forward_count_stage2")
    payload_present = endpoint.get("candidate_payload") is not None
    stage2_records = endpoint.get("stage2_trajectory")
    records_typed = isinstance(stage2_records, list) and all(
        isinstance(record, Mapping) for record in stage2_records
    )
    native_records = (
        [
            record
            for record in stage2_records
            if record.get("native_selection_forward_performed") is True
        ]
        if records_typed
        else []
    )
    pre_native_records = (
        [
            record
            for record in stage2_records
            if record.get("native_selection_forward_performed") is False
        ]
        if records_typed
        else []
    )
    native_actor_hashes = [
        record.get("candidate_actor_float32_le_sha256")
        for record in native_records
    ]
    shadow_audit = endpoint.get("stage2_shadow_ledger", {})
    verification_count = endpoint.get("verification_forward_count_stage2")
    is_go = decision == "GO_CW24_V17_FP32_ABSOLUTE_QP_TRAIN_GATE"
    native_flags_are_typed = records_typed and all(
        type(record.get("native_selection_forward_performed")) is bool
        for record in stage2_records
    )
    complete_native_records = [
        record
        for record in native_records
        if record.get("complete_native_gate_pass") is True
    ]
    native_identity_checks_exact = records_typed and all(
        isinstance(record.get("identity_after_current_FP32"), Mapping)
        and bool(record["identity_after_current_FP32"])
        and all(record["identity_after_current_FP32"].values())
        and isinstance(record.get("identity_after_candidate_FP32"), Mapping)
        and bool(record["identity_after_candidate_FP32"])
        and all(record["identity_after_candidate_FP32"].values())
        and isinstance(record.get("identity_after_native_gate"), Mapping)
        and bool(record["identity_after_native_gate"])
        and all(record["identity_after_native_gate"].values())
        for record in native_records
    )
    native_gate_summary_exact = records_typed and all(
        isinstance(record.get("native_terminal_gate"), Mapping)
        and type(record["native_terminal_gate"].get("pass")) is bool
        and record["native_terminal_gate"]["pass"]
        is record.get("complete_native_gate_pass")
        for record in native_records
    )
    terminal_stage2_record = stage2_records[-1] if records_typed and stage2_records else None
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 2
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    typed_result_checks = {
        "decision_exact_enum": type(decision) is str
        and decision in allowed_decisions,
        "payload_present_iff_GO": payload_present
        == (decision == "GO_CW24_V17_FP32_ABSOLUTE_QP_TRAIN_GATE"),
        "changed_shadow_count_typed_range": type(changed) is int
        and STAGE1_REFERENCE_STEPS
        <= changed
        <= TOTAL_CHANGED_TRAIN_SHADOW_CAP,
        "native_stage2_count_typed_range": type(stage2_native) is int
        and 0 <= stage2_native <= STAGE2_MAX_TRIALS,
        "changed_count_exact_29_plus_native": type(changed) is int
        and type(stage2_native) is int
        and changed == STAGE1_REFERENCE_STEPS + stage2_native,
        "stage1_exact29": len(endpoint.get("stage1_trajectory", []))
        == STAGE1_REFERENCE_STEPS
        and all(
            all(record["v12_reference_exact_checks"].values())
            for record in endpoint.get("stage1_trajectory", [])
        ),
        "train_only_B352": endpoint.get("cache", {}).get("non_train_members_opened")
        is False,
        "outer_restore_pass": endpoint.get("outer_restore_pass") is True,
        "top_level_reason_nonempty": isinstance(endpoint.get("reason"), str)
        and bool(endpoint["reason"]),
        "stage2_trajectory_typed": records_typed,
        "native_forward_flags_bool_and_partition_exact": native_flags_are_typed
        and len(native_records) + len(pre_native_records) == len(stage2_records),
        "native_record_count_exact": records_typed
        and type(stage2_native) is int
        and len(native_records) == stage2_native,
        "native_ordinals_exact": records_typed
        and [
            record.get("native_selection_forward_ordinal")
            for record in native_records
        ]
        == list(range(1, len(native_records) + 1)),
        "native_actor_hashes_unique_typed": all(
            isinstance(value, str) and len(value) == 64
            for value in native_actor_hashes
        )
        and len(set(native_actor_hashes)) == len(native_actor_hashes),
        "native_actor_hashes_equal_ledger": sorted(native_actor_hashes)
        == shadow_audit.get("unique_trial_actor_sha256"),
        "native_identity_rehashes_all_true": native_identity_checks_exact,
        "native_gate_summary_exact": native_gate_summary_exact,
        "ledger_native_count_exact": shadow_audit.get(
            "native_gate_forward_count"
        )
        == stage2_native,
        "ledger_total_count_exact": shadow_audit.get(
            "total_changed_train_shadows"
        )
        == changed,
        "GO_exactly_one_complete_native_record_NO_GO_zero": len(
            complete_native_records
        )
        == (1 if is_go else 0),
        "GO_is_terminal_first_complete_native_record": (
            not is_go
            or (
                terminal_stage2_record is complete_native_records[0]
                and terminal_stage2_record.get("decision")
                == "GO_FIRST_COMPLETE_NATIVE_GATE"
                and shadow_audit.get("terminal_decision") == "GO"
                and shadow_audit.get("GO_count") == 1
            )
        ),
        "NO_GO_ledger_never_reached_GO": is_go
        or (
            shadow_audit.get("terminal_decision") == "NO_GO"
            and shadow_audit.get("GO_count") == 0
        ),
        "final_ledger_terminal_decision_exact": shadow_audit.get(
            "terminal_decision"
        )
        == ("GO" if is_go else "NO_GO"),
        "pre_native_proposals_explicitly_not_shadows": all(
            record.get("pre_native_proposal_counts_as_changed_shadow") is False
            for record in pre_native_records
        ),
        "v15_pre_native_materializations_explicitly_not_shadows": endpoint.get(
            "v15_pre_native_writeback_attempts_count_as_changed_train_shadows"
        )
        is False,
        "verification_count_GO1_NO_GO0": type(verification_count) is int
        and verification_count
        == (1 if is_go else 0),
        "final_official_count_zero_int": type(
            result.get("official_unique_changed_candidate_count_consumed")
        )
        is int
        and result["official_unique_changed_candidate_count_consumed"] == 0,
        "final_cumulative_count_two_int": type(
            result.get("cumulative_official_unique_changed_candidate_count")
        )
        is int
        and result["cumulative_official_unique_changed_candidate_count"] == 2,
        "final_submission_false_bool": result.get("submission_performed") is False,
        "final_package_upload_false_bool": result.get("package_upload_performed")
        is False,
    }
    if not all(typed_result_checks.values()):
        raise ProtocolError(f"v17 typed result contract failed: {typed_result_checks}")
    if decision.startswith("NO_GO"):
        no_go_audit = no_go_material_audit(result)
    else:
        no_go_audit = None

    frozen_engine_selection = result["selection"]
    result["schema_version"] = SCHEMA
    result["status"] = decision
    result["decision"] = decision
    result["seed"] = STAGE1_EXECUTION_SEED
    result["protocol_id"] = PROTOCOL_ID
    result["stage1_execution_seed"] = STAGE1_EXECUTION_SEED
    result["stage2_randomness"] = "none_deterministic_linear_algebra_only"
    result["base"]["kind"] = (
        "general_BC_plus_PPO_plus_exact_CW11_then_exact_v12_step29_then_"
        "FP32_absolute_QP_plus_special_BC_secondary"
    )
    result["selection"] = {
        "rows": 352,
        "sha256": v1.ROWS_CANONICAL_SHA256,
        "checks": selection_audit["checks"],
        "counts": selection_audit["counts"],
        "scope": selection_audit["scope"],
        "optimization_contract": {
            "stage1_execution_seed": STAGE1_EXECUTION_SEED,
            "stage2_randomness": "none",
            "stage1_exact_v12_steps": STAGE1_REFERENCE_STEPS,
            "stage2_maximum_native_trials": STAGE2_MAX_TRIALS,
            "total_maximum_changed_train_shadows": TOTAL_CHANGED_TRAIN_SHADOW_CAP,
            "stage2_optimizer_instances": 0,
            "stage2_backward_calls": 0,
            "stage2_FP32_direct_VJPs": True,
            "absolute_ball_free_minimum_norm_QP": True,
            "safe_target_radius": SAFE_TARGET_RADIUS,
            "hard_actual_radius": HARD_ACTUAL_RADIUS,
            "trust_radius": TRUST_RADIUS,
            "v15_writeback_only_before_unique_native_trial": True,
            "first_complete_native_gate_stop": True,
        },
        "frozen_v12_selection": reference_payload["selection"],
        "frozen_engine_B256_reconstruction_selection": frozen_engine_selection,
    }
    historical = result["historical_exact_CW11_replay"]
    historical.pop("new_validation_rows_opened_for_CW22_selection_or_candidate", None)
    historical["new_validation_rows_opened_for_CW24_v17_selection_or_candidate"] = 0
    inputs = result.setdefault("inputs", {})
    inputs["CW24_v17_source"] = source
    inputs["CW24_v17_source_post_run"] = post_source
    inputs["CW24_v17_dependencies"] = dependencies
    inputs["CW24_v17_dependencies_post_run"] = post_dependencies
    inputs["CW24_v17_step29_handoff_transform"] = transform_audit
    inputs["CW24_v17_frozen_v15_projector"] = v15_evidence
    inputs["CW24_v17_frozen_v12_module"] = v12_evidence
    inputs["CW24_v17_frozen_v12_reference"] = reference_audit
    inputs["CW24_v17_v1_loader"] = v1_evidence
    inputs["CW24_v17_CW23_loader"] = cw23_evidence
    inputs["CW24_v17_CW22_engine"] = cw22_evidence
    inputs["CW24_v17_attempt"] = claim["publication"]
    inputs["CW24_v17_pre_module_claim_binding"] = pre_module_binding
    audit = result.setdefault("audit", {})
    audit["source"] = source
    audit["CW24_v17_contract"] = {
        "python_exact": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "pre_cuda_runtime": pre_cuda_runtime,
        "pre_module_claim_binding_checks": pre_module_binding["checks"],
        "post_run_checks": post_run_checks,
        "typed_result_checks": typed_result_checks,
        "train_only": True,
        "new_validation_or_test_rows_opened": 0,
        "checkpoint_writes": 0,
        "model_artifact_writes": 0,
        "optimizer_instances_created": 1,
        "optimizer_step_calls": STAGE1_REFERENCE_STEPS,
        "stage2_optimizer_step_calls": 0,
        "candidate_payload_present_iff_GO": payload_present
        == decision.startswith("GO_"),
        "NO_GO_material_audit": no_go_audit,
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
    }
    result["exploratory_disclosure"] = {
        "train_only_result_is_not_promotion_evidence": True,
        "stage1_is_exact_frozen_v12_reproduction_not_new_selection": True,
        "PPO_term_is_policy_KL_not_new_rollout": True,
        "fulltrain_exact_CW11_comparator_required_after_GO": True,
        "specialist_then_broad_then_Gold_required_after_fulltrain_GO": True,
    }
    result["official_unique_changed_candidate_count_consumed"] = 0
    result["cumulative_official_unique_changed_candidate_count"] = 2
    result["submission_performed"] = False
    result["package_upload_performed"] = False
    if not is_go:
        audit["CW24_v17_contract"]["NO_GO_material_audit"] = no_go_material_audit(
            result
        )
    canonical_json(result)
    return result


def best_effort_file_observation(
    path: Path,
    label: str,
    expected: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    """Collect one held-fd observation; never raise or return file contents."""

    record: dict[str, Any] = {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "completed": False,
    }
    fd: int | None = None
    try:
        before = path.lstat()
    except FileNotFoundError:
        record.update({"completed": True, "absent": True})
        return record, None
    except BaseException as error:
        record.update(
            {
                "absent": None,
                "error_type": type(error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8", errors="replace")
                ).hexdigest(),
            }
        )
        return record, None

    record.update(
        {
            "absent": False,
            "lstat_device": int(before.st_dev),
            "lstat_inode": int(before.st_ino),
            "lstat_nlink": int(before.st_nlink),
            "lstat_mode_octal": format(stat.S_IMODE(before.st_mode), "04o"),
            "lstat_bytes": int(before.st_size),
            "regular": stat.S_ISREG(before.st_mode),
            "symlink": stat.S_ISLNK(before.st_mode),
        }
    )
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        record["completed"] = True
        record["safe_regular_read"] = False
        record["matches_expected_binding"] = False
        return record, None
    try:
        if not hasattr(os, "O_NOFOLLOW"):
            raise ProtocolError("O_NOFOLLOW unavailable during failure forensics")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            total += len(chunk)
        after_fd = os.fstat(fd)
        after_path = path.lstat()
        identity_stable = (
            stat_identity(before)
            == stat_identity(opened)
            == stat_identity(after_fd)
            == stat_identity(after_path)
        )
        binding = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest.hexdigest(),
            "bytes": total,
            "mode_octal": format(stat.S_IMODE(after_fd.st_mode), "04o"),
            "device": int(after_fd.st_dev),
            "inode": int(after_fd.st_ino),
            "nlink": int(after_fd.st_nlink),
        }
        expected_fields = (
            "path",
            "sha256",
            "bytes",
            "mode_octal",
            "device",
            "inode",
            "nlink",
        )
        expected_subset = (
            {key: expected[key] for key in expected_fields if key in expected}
            if isinstance(expected, Mapping)
            else None
        )
        record.update(
            {
                "completed": True,
                "safe_regular_read": True,
                "identity_stable_before_open_after_read": identity_stable,
                "binding": binding,
                "expected_binding": expected_subset,
                "matches_expected_binding": expected_subset is not None
                and identity_stable
                and all(binding[key] == value for key, value in expected_subset.items()),
            }
        )
        return record, b"".join(chunks)
    except BaseException as error:
        record.update(
            {
                "error_type": type(error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8", errors="replace")
                ).hexdigest(),
                "matches_expected_binding": False,
            }
        )
        return record, None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except BaseException as close_error:
                record["close_error_type"] = type(close_error).__name__


def best_effort_transform_observation(
    expected: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected_source = (
        expected.get("frozen_source") if isinstance(expected, Mapping) else None
    )
    source_record, raw = best_effort_file_observation(
        V13_SOURCE, "failed_v17_transform_source", expected_source
    )
    record: dict[str, Any] = {
        "completed": False,
        "source_observation": source_record,
    }
    if raw is None or not isinstance(source_record.get("binding"), Mapping):
        record["error"] = "transform source bytes unavailable"
        return record
    try:
        source = raw.decode("utf-8")
        replacement_count = source.count(V13_HANDOFF_NEEDLE)
        transformed = source.replace(
            V13_HANDOFF_NEEDLE, V13_HANDOFF_REPLACEMENT, 1
        ).encode("utf-8")
        syntax_valid = True
        try:
            compile(transformed, str(V13_SOURCE), "exec", dont_inherit=True)
        except BaseException:
            syntax_valid = False
        checks = {
            "frozen_source_sha_exact": hashlib.sha256(raw).hexdigest()
            == V13_SOURCE_SHA256,
            "frozen_source_mode_0555": source_record["binding"]["mode_octal"]
            == "0555",
            "frozen_source_regular_single_link": source_record.get(
                "safe_regular_read"
            )
            is True
            and source_record["binding"]["nlink"] == 1,
            "frozen_source_identity_stable": source_record.get(
                "identity_stable_before_open_after_read"
            )
            is True,
            "one_exact_step29_handoff_site": replacement_count == 1,
            "frozen_source_has_no_v17_handoff": "cw24_v17_stage2_handoff"
            not in source,
            "one_source_replacement": replacement_count == 1,
            "one_handoff_call": transformed.count(
                b"cw24_v17_stage2_handoff(dict(locals()))"
            )
            == 1,
            "stage2_SGD_source_retained_but_unreachable_after_handoff": transformed.count(
                b"project_non_outward_sgd_direction("
            )
            >= 2,
            "syntax_valid": syntax_valid,
        }
        observed = {
            "frozen_source": dict(source_record["binding"]),
            "transformed_sha256": hashlib.sha256(transformed).hexdigest(),
            "transformed_bytes": len(transformed),
            "replacement_sha256": hashlib.sha256(
                V13_HANDOFF_REPLACEMENT.encode("utf-8")
            ).hexdigest(),
            "checks": checks,
        }
        record.update(
            {
                "completed": True,
                "observed_transform": observed,
                "expected_transform": expected,
                "matches_expected_transform": isinstance(expected, Mapping)
                and observed == expected,
            }
        )
    except BaseException as error:
        record.update(
            {
                "error_type": type(error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8", errors="replace")
                ).hexdigest(),
            }
        )
    return record


def best_effort_failure_forensics(
    claim: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Collect every binding independently without masking the root failure."""

    passed_claim = claim if isinstance(claim, Mapping) else {}
    passed_payload = passed_claim.get("payload", {})
    passed_publication = passed_claim.get("publication", {})
    attempt_record, attempt_raw = best_effort_file_observation(
        ATTEMPT_MARKER,
        "failed_v17_attempt",
        passed_publication if isinstance(passed_publication, Mapping) else None,
    )
    parsed_attempt: Mapping[str, Any] | None = None
    attempt_parse: dict[str, Any] = {"completed": False}
    if attempt_raw is not None:
        try:
            parsed_attempt = load_json_no_duplicates(
                attempt_raw, "v17 failed-attempt forensic marker"
            )
            attempt_parse = {
                "completed": True,
                "canonical_bytes_exact": attempt_raw
                == canonical_json(parsed_attempt),
                "matches_in_memory_claim": isinstance(passed_payload, Mapping)
                and parsed_attempt == passed_payload,
            }
        except BaseException as error:
            attempt_parse = {
                "completed": False,
                "error_type": type(error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8", errors="replace")
                ).hexdigest(),
            }
    effective_payload: Mapping[str, Any] = (
        passed_payload
        if isinstance(passed_payload, Mapping) and passed_payload
        else parsed_attempt
        if isinstance(parsed_attempt, Mapping)
        else {}
    )

    source_record, _ = best_effort_file_observation(
        SCRIPT,
        "failed_v17_source",
        effective_payload.get("source")
        if isinstance(effective_payload.get("source"), Mapping)
        else None,
    )
    expected_dependencies = effective_payload.get("dependencies", {})
    dependency_records: dict[str, Any] = {}
    for label, path, digest, mode in FROZEN_INPUTS:
        expected = (
            expected_dependencies.get(label)
            if isinstance(expected_dependencies, Mapping)
            else None
        )
        if not isinstance(expected, Mapping):
            expected = {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest,
                "mode_octal": format(mode, "04o"),
            }
        dependency_records[label] = best_effort_file_observation(
            path, label, expected
        )[0]
    expected_transform = effective_payload.get("step29_handoff_transform")
    transform_record = best_effort_transform_observation(
        expected_transform if isinstance(expected_transform, Mapping) else None
    )
    output_record = best_effort_file_observation(
        OUTPUT, "failed_v17_output", None
    )[0]
    staging_residues: dict[str, Any] = {}
    for path in (ATTEMPT_MARKER, OUTPUT, FAILURE):
        try:
            staging_residues[str(path.relative_to(ROOT))] = (
                publication_staging_residues(path)
            )
        except BaseException as error:
            staging_residues[str(path.relative_to(ROOT))] = {
                "error_type": type(error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8", errors="replace")
                ).hexdigest(),
            }
    try:
        failure_absent = path_absent(FAILURE)
    except BaseException:
        failure_absent = False
    checks = {
        "source_matches_claim": source_record.get("matches_expected_binding")
        is True,
        "dependencies_match_claim": all(
            record.get("matches_expected_binding") is True
            for record in dependency_records.values()
        ),
        "transform_matches_claim": transform_record.get(
            "matches_expected_transform"
        )
        is True,
        "attempt_payload_matches_claim": attempt_parse.get(
            "matches_in_memory_claim"
        )
        is True,
        "attempt_identity_stable": attempt_record.get(
            "identity_stable_before_open_after_read"
        )
        is True,
        "attempt_matches_claim_publication": attempt_record.get(
            "matches_expected_binding"
        )
        is True,
        "output_final_absent_after_failed_atomic_publication": output_record.get(
            "absent"
        )
        is True,
        "all_publication_staging_residues_absent": all(
            value == [] for value in staging_residues.values()
        ),
        "failure_target_absent_before_forensic_publication": failure_absent,
    }
    torch_state: dict[str, Any] = {"imported": "torch" in sys.modules}
    if "torch" in sys.modules:
        try:
            torch_state["CUDA_initialized"] = bool(
                sys.modules["torch"].cuda.is_initialized()
            )
        except BaseException as error:
            torch_state["CUDA_state_error_type"] = type(error).__name__
    return {
        "completed": True,
        "all_collectors_independent": True,
        "checks": checks,
        "bindings_pass": all(checks.values()),
        "observed_bindings": {
            "source": source_record,
            "dependencies": dependency_records,
            "transform": transform_record,
            "attempt": attempt_record,
            "attempt_parse": attempt_parse,
            "output": output_record,
            "publication_staging_residues": staging_residues,
        },
        "output_state": output_record,
        "torch_state": torch_state,
    }


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    source_text = source.decode("utf-8")
    tree = ast.parse(source_text, filename=str(SCRIPT))
    calls: list[str] = []
    functions: dict[str, ast.FunctionDef] = {}
    publication_targets: list[str] = []

    def dotted_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix is not None else None
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "publish_o_excl"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                publication_targets.append(node.args[0].id)

    required_functions = {
        "classify_halfspace_feasibility_rowspace",
        "fp32_direct_outputs",
        "pf0_buffer_hinge",
        "fp32_structural_bundle",
        "run_stage2_handoff",
        "publish_o_excl",
        "validate_claim_binding_before_module_exec",
        "best_effort_file_observation",
        "best_effort_transform_observation",
        "best_effort_failure_forensics",
        "run_v17_production",
        "main",
    }
    if not required_functions.issubset(functions):
        raise ProtocolError(
            "v17 static source functions missing: "
            f"{sorted(required_functions - set(functions))}"
        )
    all_call_names = [
        dotted_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    direct_source = ast.get_source_segment(
        source_text, functions["fp32_direct_outputs"]
    )
    rowspace_source = ast.get_source_segment(
        source_text, functions["classify_halfspace_feasibility_rowspace"]
    )
    hinge_source = ast.get_source_segment(
        source_text, functions["pf0_buffer_hinge"]
    )
    bundle_source = ast.get_source_segment(
        source_text, functions["fp32_structural_bundle"]
    )
    handoff_source = ast.get_source_segment(
        source_text, functions["run_stage2_handoff"]
    )
    publisher_source = ast.get_source_segment(
        source_text, functions["publish_o_excl"]
    )
    claim_binding_source = ast.get_source_segment(
        source_text, functions["validate_claim_binding_before_module_exec"]
    )
    forensic_source = ast.get_source_segment(
        source_text, functions["best_effort_failure_forensics"]
    )
    forensic_file_source = ast.get_source_segment(
        source_text, functions["best_effort_file_observation"]
    )
    main_source = ast.get_source_segment(source_text, functions["main"])
    if any(
        value is None
        for value in (
            direct_source,
            rowspace_source,
            hinge_source,
            bundle_source,
            handoff_source,
            publisher_source,
            claim_binding_source,
            forensic_source,
            forensic_file_source,
            main_source,
        )
    ):
        raise ProtocolError("v17 static source extraction failed")
    assert direct_source is not None
    assert rowspace_source is not None
    assert hinge_source is not None
    assert bundle_source is not None
    assert handoff_source is not None
    assert publisher_source is not None
    assert claim_binding_source is not None
    assert forensic_source is not None
    assert forensic_file_source is not None
    assert main_source is not None

    handoff_call_names = [
        dotted_name(node.func)
        for node in ast.walk(functions["run_stage2_handoff"])
        if isinstance(node, ast.Call)
    ]
    direct_call_names = [
        dotted_name(node.func)
        for node in ast.walk(functions["fp32_direct_outputs"])
        if isinstance(node, ast.Call)
    ]
    hinge_call_names = [
        dotted_name(node.func)
        for node in ast.walk(functions["pf0_buffer_hinge"])
        if isinstance(node, ast.Call)
    ]
    bundle_call_names = [
        dotted_name(node.func)
        for node in ast.walk(functions["fp32_structural_bundle"])
        if isinstance(node, ast.Call)
    ]
    stage2_optimizer_constructors = {
        "torch.optim.SGD",
        "torch.optim.Adam",
        "torch.optim.AdamW",
        "SGD",
        "Adam",
        "AdamW",
    }
    immediate_go_markers = (
        "native_outputs = ppo.model_forward(model, batch, device)",
        "if complete_native:",
        "diagnostics = v12.soft_floor_diagnostics(rows, baseline, current, cw22)",
        "updated_candidate_bundle = fp32_structural_bundle(",
    )
    immediate_go_positions = tuple(
        handoff_source.find(marker) for marker in immediate_go_markers
    )
    pre_native_FP32_position = handoff_source.find(
        "# Candidate FP32 evidence is frozen before the sole native selection"
    )
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    transform = transform_v13_source()[1]
    checks = {
        "syntax_valid": True,
        "stage_budget_exact_29_plus_32": STAGE1_REFERENCE_STEPS == 29
        and STAGE2_MAX_TRIALS == 32
        and TOTAL_CHANGED_TRAIN_SHADOW_CAP == 61,
        "execution_seed_exact_frozen_v12": STAGE1_EXECUTION_SEED == 202608052,
        "protocol_id_not_execution_seed": PROTOCOL_ID == 202608053
        and PROTOCOL_ID != STAGE1_EXECUTION_SEED,
        "safe_radius_exact": SAFE_TARGET_RADIUS == 0.000999,
        "hard_actual_radius_exact": HARD_ACTUAL_RADIUS == 0.00099998,
        "trust_radius_exact": TRUST_RADIUS == 2.5e-5,
        "pinv_rcond_exact": PINV_RCOND == 1e-12,
        "KKT_tolerance_exact": KKT_TOL == 1e-9,
        "secondary_cap_exact": SECONDARY_NORM_FRACTION == 0.25,
        "HiGHS_fallback_uses_cut_rowspace_not_actor_dimension": all(
            marker in rowspace_source
            for marker in (
                "gram = np.asarray(value @ value.T, dtype=np.float64)",
                "np.zeros(row_count, dtype=np.float64)",
                "A_ub=-gram",
                "b_ub=-bound",
                "bounds=[(None, None)] * row_count",
            )
        )
        and "np.zeros(original_dimension" not in rowspace_source,
        "real_FP32_direct_forward_present": (
            'with torch.autocast(device_type="cuda", enabled=False):'
            in direct_source
            and "outputs = model(batch)" in direct_source
            and direct_call_names.count("model") == 1
        ),
        "FP32_output_schema_fail_closed": all(
            marker in direct_source
            for marker in (
                'expected_keys = {"policy_logits", "count_logits", "value_logits"}',
                '"keys_exact"',
                '"all_tensors"',
                '"all_CUDA"',
                '"all_float32"',
                '"all_finite"',
                '"all_batch_dimension_exact"',
                '"policy_shape_exact_option_mask"',
            )
        ),
        "no_BF16_logits_cast_as_structural_path": (
            'outputs["policy_logits"].float()' not in direct_source
        ),
        "no_optimizer_constructor_in_v17": not any(
            name in {"SGD", "Adam", "AdamW"} for name in calls
        )
        and not any(
            name is not None and name.startswith("torch.optim.")
            for name in all_call_names
        ),
        "stage2_no_backward_or_optimizer_step": not any(
            name is not None
            and (name.endswith(".backward") or name.endswith(".step"))
            for name in handoff_call_names
        ),
        "stage2_no_optimizer_constructor": not any(
            name in stage2_optimizer_constructors
            or (name is not None and name.startswith("torch.optim."))
            for name in handoff_call_names
        ),
        "production_implemented": len(functions["run_v17_production"].body) > 10,
        "single_step29_handoff_transform": all(transform["checks"].values()),
        "single_v15_projector_call_site_in_stage2": handoff_call_names.count(
            "v15.v15_safe_project_actor6"
        )
        == 1,
        "exact_two_native_model_forward_sites_selection_plus_verification": (
            handoff_call_names.count("ppo.model_forward") == 2
        ),
        "candidate_FP32_bundle_precedes_native_selection": (
            pre_native_FP32_position >= 0
            and immediate_go_positions[0] >= 0
            and pre_native_FP32_position < immediate_go_positions[0]
        ),
        "first_complete_native_GO_precedes_all_post_native_work": all(
            position >= 0 for position in immediate_go_positions
        )
        and immediate_go_positions == tuple(sorted(immediate_go_positions)),
        "certified_QP_infeasibility_evidence_serialized": all(
            marker in handoff_source
            for marker in (
                'except QPInfeasibleError as error:',
                '"HiGHS_status_exact_2"',
                '"matrix_hash_exact"',
                '"rhs_hash_exact"',
                '"ordered_cut_keys"',
                '"rowspace_HiGHS_infeasibility_certificate"',
            )
        ),
        "GO_replay_requires_full_gate_equality": all(
            marker in handoff_source
            for marker in (
                '"repeat_native_terminal_gate_full_equality"',
                '== selected["gate"]',
            )
        ),
        "pre_native_integrity_is_fatal_and_novelty_is_structured": all(
            marker in handoff_source
            for marker in (
                '"fatal v15 writeback integrity failure',
                "pre_native_integrity_checks",
                "pre_native_novelty_checks",
                '"fatal pre-native writeback integrity failure',
                '"PRE_NATIVE_CANDIDATE_STALLED_RESTORED"',
            )
        ),
        "successful_endpoint_ledger_terminal_is_never_null": all(
            marker in source_text
            for marker in (
                "mark_terminal_no_go_without_native_trial",
                '"NO_GO endpoint lacks typed terminal ledger state"',
                '"final_ledger_terminal_decision_exact"',
            )
        ),
        "PF0_two_q_buffer_is_explicit_FP32_ReLU_hinge": (
            hinge_call_names.count("torch.relu") == 1
            and "torch.relu(target - margin)" in hinge_source
            and bundle_call_names.count("pf0_buffer_hinge") == 1
            and "v13.tempered_pair_penalty" not in bundle_call_names
            and "tempered_pair_penalty" not in bundle_source
        ),
        "pre_module_claim_binding_is_stable": all(
            marker in claim_binding_source
            for marker in (
                '"attempt_identity_stable"',
                '"output_absent_before_any_module_exec"',
                '"failure_absent_before_any_module_exec"',
                '"torch_still_not_imported"',
            )
        ),
        "failure_forensics_rehashes_bindings": all(
            marker in forensic_source
            for marker in (
                '"source_matches_claim"',
                '"dependencies_match_claim"',
                '"transform_matches_claim"',
                '"attempt_identity_stable"',
                '"output_state"',
                '"observed_bindings"',
                '"all_collectors_independent"',
            )
        )
        and all(
            marker in forensic_file_source
            for marker in (
                "opened = os.fstat(fd)",
                "after_fd = os.fstat(fd)",
                "after_path = path.lstat()",
                '"identity_stable_before_open_after_read"',
                '"device"',
                '"inode"',
                '"nlink"',
            )
        ),
        "publication_is_staged_atomic_no_clobber_and_fd_bound": all(
            marker in publisher_source
            for marker in (
                "fd = os.open(staging, flags, 0o600)",
                "opened = os.fstat(fd)",
                "completed_fd = os.fstat(fd)",
                "os.link(staging, path, follow_symlinks=False)",
                '"atomic_no_clobber_link_same_held_inode"',
                "os.unlink(staging)",
                '"final_same_held_inode"',
                "for cleanup_path in (path, staging):",
            )
        )
        and "os.open(path, flags" not in publisher_source,
        "claim_and_failure_publication_are_inside_root_try": (
            main_source.find("try:")
            < main_source.find("claim = claim_attempt()")
            < main_source.find("result = run_v17_production(claim)")
            and "except BaseException as publication_error:" in main_source
        ),
        "failure_artifact_external_action_fields_typed": all(
            marker in main_source
            for marker in (
                '"submission_performed": False',
                '"package_upload_performed": False',
                '"official_unique_changed_candidate_count_consumed": 0',
                '"cumulative_official_unique_changed_candidate_count": 2',
            )
        ),
        "publication_requires_O_NOFOLLOW": (
            'raise ProtocolError("O_NOFOLLOW is required for one-shot publication")'
            in source_text
        ),
        "no_checkpoint_save": (b"torch" + b".save") not in source,
        "no_network_import": not any(
            name.split(".")[0] in {"requests", "urllib", "httpx", "socket"}
            for name in imports
        ),
        "single_attempt_publication_site": publication_targets.count(
            "ATTEMPT_MARKER"
        )
        == 1,
        "single_output_publication_site": publication_targets.count("OUTPUT") == 1,
        "single_failure_publication_site": publication_targets.count("FAILURE")
        == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"v17 static source audit failed: {checks}")
    return {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "mode_octal": format(stat.S_IMODE(SCRIPT.stat().st_mode), "04o"),
        "checks": checks,
        "pass": True,
        "step29_handoff_transform": transform,
    }


def cpu_solver_unit() -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import linprog

    tests: dict[str, bool] = {}

    d = np.asarray([0.25, -0.5], dtype=np.float64)
    g = np.asarray([3.0, 4.0], dtype=np.float64)
    key = make_cut_key(
        actor_sha256="a" * 64,
        line_sha256="b" * 64,
        positive_option=1,
        negative_option=0,
        target_kind="unit",
        target=0.3,
    )
    cut = normalize_absolute_cut(
        key=key,
        current_delta=d,
        current_margin=-0.2,
        gradient=g,
        target=0.3,
    )
    probe = np.asarray([0.7, -0.1], dtype=np.float64)
    lhs_original = -0.2 + float(g @ (probe - d))
    tests["normalized_cut_algebra"] = math.isclose(
        float(cut.normal @ probe - cut.rhs),
        (lhs_original - 0.3) / float(np.linalg.norm(g)),
        rel_tol=0.0,
        abs_tol=1e-14,
    )
    plus_zero = target_identity(0.0)
    minus_zero = target_identity(-0.0)
    tests["target_identity_canonicalizes_signed_zero"] = plus_zero == minus_zero
    next_target = float(np.nextafter(np.float64(0.3), np.float64(math.inf)))
    next_key = make_cut_key(
        actor_sha256="a" * 64,
        line_sha256="b" * 64,
        positive_option=1,
        negative_option=0,
        target_kind="unit",
        target=next_target,
    )
    tests["nextafter_targets_have_distinct_binary_keys"] = (
        next_key != key
        and next_key.target_float64_le_sha256
        != key.target_float64_le_sha256
    )
    next_cut = normalize_absolute_cut(
        key=next_key,
        current_delta=d,
        current_margin=-0.2,
        gradient=g,
        target=next_target,
    )
    target_ledger = AppendOnlyCutLedger()
    target_ledger.extend([cut, next_cut])
    tests["nextafter_target_cuts_coexist"] = len(target_ledger) == 2
    mismatch_rejected = False
    try:
        normalize_absolute_cut(
            key=key,
            current_delta=d,
            current_margin=-0.2,
            gradient=g,
            target=next_target,
        )
    except ProtocolError:
        mismatch_rejected = True
    tests["same_key_different_target_rejected"] = mismatch_rejected
    nonfinite_target_rejected = False
    try:
        make_cut_key(
            actor_sha256="a" * 64,
            line_sha256="b" * 64,
            positive_option=1,
            negative_option=0,
            target_kind="unit",
            target=float("nan"),
        )
    except ProtocolError:
        nonfinite_target_rejected = True
    tests["nonfinite_target_identity_rejected"] = nonfinite_target_rejected
    identity_roundtrip = json.loads(canonical_json(target_identity(0.3)))
    tests["target_identity_JSON_roundtrip_exact"] = identity_roundtrip == target_identity(
        0.3
    )
    tests["PF0_positive_hinge_scalar_exact"] = (
        positive_hinge_scalar(-0.1, 0.2) == 0.30000000000000004
        and positive_hinge_scalar(0.2, 0.2) == 0.0
        and positive_hinge_scalar(0.3, 0.2) == 0.0
    )

    metadata_a = normalize_absolute_cut(
        key=key,
        current_delta=d,
        current_margin=-0.2,
        gradient=g,
        target=0.3,
        target_kind="kind_a",
        origins=("origin_a",),
    )
    metadata_b = normalize_absolute_cut(
        key=key,
        current_delta=d,
        current_margin=-0.2,
        gradient=g,
        target=0.3,
        target_kind="kind_b",
        origins=("origin_b", "origin_a"),
    )
    metadata_ledger = AppendOnlyCutLedger()
    metadata_ledger.extend([metadata_a, metadata_b])
    metadata_merged = metadata_ledger.ordered()[0]
    tests["duplicate_geometry_merges_metadata_without_new_cut"] = (
        len(metadata_ledger) == 1
        and metadata_merged.target_kinds == ("kind_a", "kind_b")
        and metadata_merged.origins == ("origin_a", "origin_b")
    )

    def unit_cut(name: str, normal: Sequence[float], rhs: float) -> NormalizedCut:
        vector = np.asarray(normal, dtype=np.float64)
        vector /= np.linalg.norm(vector)
        return NormalizedCut(
            key=make_cut_key(
                actor_sha256="0" * 64,
                line_sha256=name.ljust(64, "0")[:64],
                positive_option=1,
                negative_option=0,
                target_kind="unit",
                target=rhs,
            ),
            normal=vector,
            rhs=float(rhs),
            target=float(rhs),
            current_margin=0.0,
            gradient_norm=1.0,
        )

    simple = [unit_cut("b", [0.0, 1.0], 2.0), unit_cut("a", [1.0, 0.0], 1.0)]
    x, simple_audit = solve_minimum_norm_active_set(simple)
    tests["simple_solution"] = bool(np.allclose(x, [1.0, 2.0], rtol=0.0, atol=1e-12))
    tests["canonical_active_key_order"] = simple_audit["active_keys"] == sorted(
        simple_audit["active_keys"]
    )
    slack_audit = descriptive_cut_slacks(
        simple,
        ball_free_solution=x,
        capped_absolute_target=0.5 * x,
        planned_trial=np.zeros_like(x),
    )
    tests["cap_slacks_are_descriptive_and_may_be_negative"] = (
        slack_audit["certified"] is False
        and slack_audit["used_for_selection"] is False
        and slack_audit["negative_slack_is_allowed_and_does_not_trigger_NO_GO"]
        is True
        and slack_audit["ball_free_solution_crosscheck"]["violated_count"] == 0
        and slack_audit["safe_radius_capped_absolute_target"]["violated_count"]
        == 2
        and slack_audit["planned_trust_trial"]["violated_count"] == 2
    )
    tie = [unit_cut("b", [0.0, 1.0], 1.0), unit_cut("a", [1.0, 0.0], 1.0)]
    tie_x_forward, tie_audit = solve_minimum_norm_active_set(tie)
    tie_x_reverse, tie_reverse_audit = solve_minimum_norm_active_set(list(reversed(tie)))
    tests["input_order_invariant"] = bool(
        np.array_equal(tie_x_forward, tie_x_reverse)
        and tie_audit["solution_float64_le_sha256"]
        == tie_reverse_audit["solution_float64_le_sha256"]
    )

    duplicate = [unit_cut("a", [1.0, 0.0], 1.0), unit_cut("c", [1.0, 0.0], 1.0)]
    x_duplicate, duplicate_audit = solve_minimum_norm_active_set(duplicate)
    tests["rank_deficient_pinv"] = bool(
        np.allclose(x_duplicate, [1.0, 0.0], rtol=0.0, atol=1e-12)
        and duplicate_audit["rank"] == 1
    )

    blocker_matrix = np.asarray(
        [
            [-0.9447416720524403, 0.32781576088949627],
            [-0.6461154892010335, -0.7632396573911167],
            [0.2631331704617782, -0.9647595216439859],
        ],
        dtype=np.float64,
    )
    blocker_rhs = np.asarray(
        [0.7732465791362175, 1.3462383361657737, 0.5568454170457465],
        dtype=np.float64,
    )
    blocker_cuts = [
        NormalizedCut(
            key=make_cut_key(
                actor_sha256="0" * 64,
                line_sha256=f"{index:064x}",
                positive_option=1,
                negative_option=0,
                target_kind="fuzz",
                target=float(blocker_rhs[index]),
            ),
            normal=blocker_matrix[index],
            rhs=float(blocker_rhs[index]),
            target=float(blocker_rhs[index]),
            current_margin=0.0,
            gradient_norm=1.0,
        )
        for index in range(3)
    ]
    blocker_x, blocker_audit = solve_minimum_norm_active_set(blocker_cuts)
    blocker_expected = np.asarray(
        [-1.1252444473959426, -0.8840903218867645], dtype=np.float64
    )
    tests["rank_inconsistent_workset_regression"] = bool(
        np.allclose(blocker_x, blocker_expected, rtol=0.0, atol=1e-12)
        and blocker_audit["active_indices"] == [0, 2]
        and blocker_audit["minimum_residual"] >= -KKT_TOL
    )

    small_feasibility_cases = (
        (
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            np.asarray([1.0, 2.0], dtype=np.float64),
            "feasible",
        ),
        (
            np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64),
            np.asarray([1.0, 0.0], dtype=np.float64),
            "infeasible",
        ),
        (
            np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            np.asarray([1.0, 0.5], dtype=np.float64),
            "feasible",
        ),
    )
    rowspace_reports = []
    rowspace_matches_original = True
    for case_matrix, case_rhs, expected_classification in small_feasibility_cases:
        original = linprog(
            np.zeros(case_matrix.shape[1], dtype=np.float64),
            A_ub=-case_matrix,
            b_ub=-case_rhs,
            bounds=[(None, None)] * case_matrix.shape[1],
            method="highs",
        )
        original_classification = (
            "feasible"
            if bool(original.success)
            else "infeasible"
            if int(original.status) == 2
            else "indeterminate"
        )
        rowspace = classify_halfspace_feasibility_rowspace(
            case_matrix,
            case_rhs,
        )
        rowspace_reports.append(rowspace)
        rowspace_matches_original = rowspace_matches_original and (
            original_classification
            == rowspace["classification"]
            == expected_classification
            and rowspace["row_variable_count"] == case_matrix.shape[0]
            and rowspace["original_variable_count"] == case_matrix.shape[1]
        )
    tests["rowspace_feasibility_matches_original_small_problems"] = (
        rowspace_matches_original
    )
    wide_matrix = np.zeros((2, 65793), dtype=np.float64)
    wide_matrix[0, 0] = 1.0
    wide_matrix[1, 0] = -1.0
    wide_report = classify_halfspace_feasibility_rowspace(
        wide_matrix,
        np.asarray([1.0, 0.0], dtype=np.float64),
    )
    tests["rowspace_fallback_dimension_is_cut_count_not_actor6"] = (
        wide_report["classification"] == "infeasible"
        and wide_report["row_variable_count"] == 2
        and wide_report["original_variable_count"] == 65793
        and wide_report["gram_shape"] == [2, 2]
    )

    infeasible_classified_exactly = False
    try:
        solve_minimum_norm_active_set(
            [unit_cut("a", [1.0, 0.0], 1.0), unit_cut("b", [-1.0, 0.0], 0.0)]
        )
    except QPInfeasibleError as error:
        certificate = error.certificate
        infeasible_classified_exactly = (
            certificate.get("classification") == "infeasible"
            and certificate.get("status") == 2
            and certificate.get("success") is False
            and certificate.get("solver")
            == "scipy.optimize.linprog/HiGHS-row-space"
            and certificate.get("scipy_version") == EXPECTED_SCIPY_VERSION
            and certificate.get("row_variable_count") == 2
            and certificate.get("original_variable_count") == 2
            and isinstance(certificate.get("matrix_float64_le_sha256"), str)
            and isinstance(certificate.get("rhs_float64_le_sha256"), str)
            and isinstance(certificate.get("gram_float64_le_sha256"), str)
        )
    except ProtocolError:
        infeasible_classified_exactly = False
    tests["rank_inconsistent_constraints_classified_QP_infeasible"] = (
        infeasible_classified_exactly
    )

    trial, compose_audit = compose_trust_step(
        current_delta=np.asarray([0.0, 0.0]),
        absolute_qp_solution=np.asarray([2.0, 0.0]),
        primary_pf7_gradient=np.asarray([1.0, 0.0]),
        secondary_descent=np.asarray([0.0, 1.0]),
        binding_rows=np.asarray([[1.0, 0.0]]),
        safe_radius=1.0,
        trust_radius=0.25,
        secondary_fraction=0.25,
    )
    tests["absolute_R_then_eta"] = bool(
        np.allclose(trial, [0.25, 0.0], rtol=0.0, atol=1e-12)
    )
    tests["secondary_eta_scaling_harm_fails_closed"] = (
        compose_audit["secondary_used"] is False
        and "eta_scaling_reduces_primary_PF7"
        in compose_audit["secondary_fail_closed_reasons"]
    )

    trial2, compose_audit2 = compose_trust_step(
        current_delta=np.asarray([0.0, 0.0]),
        absolute_qp_solution=np.asarray([0.05, 0.0]),
        primary_pf7_gradient=np.asarray([1.0, 0.0]),
        secondary_descent=np.asarray([0.0, 1.0]),
        binding_rows=np.asarray([[1.0, 0.0]]),
        safe_radius=1.0,
        trust_radius=0.25,
        secondary_fraction=0.25,
    )
    tests["secondary_survives_with_trust_headroom"] = bool(
        compose_audit2["secondary_used"]
        and np.allclose(trial2, [0.05, 0.0625], rtol=0.0, atol=1e-12)
    )

    inset_trial, inset_audit = compose_trust_step(
        current_delta=np.asarray([0.0009995, 0.0]),
        absolute_qp_solution=np.asarray([0.000998, 0.0]),
        primary_pf7_gradient=np.asarray([1.0, 0.0]),
        secondary_descent=np.asarray([0.0, 0.0]),
        binding_rows=np.zeros((0, 2), dtype=np.float64),
    )
    tests["accepted_current_between_safe_and_hard_can_move_inward"] = bool(
        inset_audit["accepted_current_l2"] > SAFE_TARGET_RADIUS
        and inset_audit["accepted_current_within_hard_actual_radius"] is True
        and inset_audit["checks"]["absolute_endpoint_within_R"] is True
        and float(np.linalg.norm(inset_trial)) <= SAFE_TARGET_RADIUS + KKT_TOL
    )

    ledger = Stage2ShadowLedger("anchor", -0.004, -0.004)
    first = ledger.record_native_trial(
        actor_sha256="candidate1",
        fp32_pf7_margin=-0.003,
        native_pf7_cell=-0.004,
        all_non_pf7_hard_gates_pass=True,
        complete_native_gate_pass=False,
        violated_cuts_added=0,
        restored_after_reject=False,
    )
    second = ledger.record_native_trial(
        actor_sha256="candidate2",
        fp32_pf7_margin=-0.0035,
        native_pf7_cell=-0.006,
        all_non_pf7_hard_gates_pass=False,
        complete_native_gate_pass=False,
        violated_cuts_added=1,
        restored_after_reject=True,
    )
    ledger_audit = ledger.audit()
    tests["accept_reject_restore_state_machine"] = (
        first == "ACCEPT_CONTINUE"
        and second == "REJECT_RESTORE_CONTINUE"
        and ledger.accepted_actor_sha256 == "candidate1"
    )
    tests["one_native_forward_per_shadow"] = ledger_audit["checks"][
        "one_native_gate_forward_per_stage2_trial"
    ]
    tests["total_budget_accounting"] = ledger_audit["total_changed_train_shadows"] == 31

    go_side_effect_failed_closed = False
    try:
        Stage2ShadowLedger("anchor", -0.004, -0.004).record_native_trial(
            actor_sha256="go-with-side-effect",
            fp32_pf7_margin=0.001,
            native_pf7_cell=0.001,
            all_non_pf7_hard_gates_pass=True,
            complete_native_gate_pass=True,
            violated_cuts_added=1,
            restored_after_reject=True,
        )
    except ProtocolError:
        go_side_effect_failed_closed = True
    tests["GO_reject_side_effects_forbidden"] = go_side_effect_failed_closed

    native_go_overrides = Stage2ShadowLedger("anchor", 0.5, 0.5)
    native_go_override_result = native_go_overrides.record_native_trial(
        actor_sha256="native-go-worse-heuristics",
        fp32_pf7_margin=-1.0,
        native_pf7_cell=-1.0,
        all_non_pf7_hard_gates_pass=True,
        complete_native_gate_pass=True,
        violated_cuts_added=0,
        restored_after_reject=False,
    )
    tests["complete_native_GO_precedes_nondegrade_heuristics"] = (
        native_go_override_result == "GO"
        and native_go_overrides.terminal_decision == "GO"
    )

    complete_summary_contradiction_rejected = False
    try:
        Stage2ShadowLedger("anchor", 0.0, 0.0).record_native_trial(
            actor_sha256="bad-native-summary",
            fp32_pf7_margin=0.0,
            native_pf7_cell=0.0,
            all_non_pf7_hard_gates_pass=False,
            complete_native_gate_pass=True,
            violated_cuts_added=0,
            restored_after_reject=False,
        )
    except ProtocolError:
        complete_summary_contradiction_rejected = True
    tests["complete_native_summary_contradiction_rejected"] = (
        complete_summary_contradiction_rejected
    )

    terminal_no_cut = Stage2ShadowLedger("anchor", 0.0, 0.0)
    no_cut_result = terminal_no_cut.record_native_trial(
        actor_sha256="uncuttable-reject",
        fp32_pf7_margin=-1.0,
        native_pf7_cell=-1.0,
        all_non_pf7_hard_gates_pass=False,
        complete_native_gate_pass=False,
        violated_cuts_added=0,
        restored_after_reject=True,
        terminal_no_go_no_new_cut_reason=(
            "NONPAIR_HARD_GATE_FAILURE_NO_LEGAL_STRUCTURAL_CUT"
        ),
    )
    after_no_go_failed_closed = False
    try:
        terminal_no_cut.record_native_trial(
            actor_sha256="after-terminal-no-go",
            fp32_pf7_margin=0.0,
            native_pf7_cell=0.0,
            all_non_pf7_hard_gates_pass=True,
            complete_native_gate_pass=False,
            violated_cuts_added=0,
            restored_after_reject=False,
        )
    except ProtocolError:
        after_no_go_failed_closed = True
    tests["zero_cut_restore_is_terminal_structured_NO_GO"] = (
        no_cut_result == "REJECT_RESTORE_TERMINAL_NO_GO"
        and terminal_no_cut.terminal_decision == "NO_GO"
        and terminal_no_cut.rejected_stage2_count == 1
    )
    tests["post_terminal_NO_GO_trial_forbidden"] = after_no_go_failed_closed

    pre_native_terminal = Stage2ShadowLedger("anchor", 0.0, 0.0)
    pre_native_terminal.mark_terminal_no_go_without_native_trial(
        "UNIT_PRE_NATIVE_STALL"
    )
    pre_native_terminal_audit = pre_native_terminal.audit()
    tests["pre_native_or_budget_NO_GO_has_typed_terminal_ledger"] = (
        pre_native_terminal_audit["terminal_decision"] == "NO_GO"
        and pre_native_terminal_audit["terminal_no_go_reason"]
        == "UNIT_PRE_NATIVE_STALL"
        and pre_native_terminal_audit["stage2_changed_train_shadows"] == 0
        and pre_native_terminal_audit["native_gate_forward_count"] == 0
    )

    terminal = Stage2ShadowLedger("anchor", -0.004, -0.004)
    terminal_result = terminal.record_native_trial(
        actor_sha256="valid-go",
        fp32_pf7_margin=0.001,
        native_pf7_cell=0.001,
        all_non_pf7_hard_gates_pass=True,
        complete_native_gate_pass=True,
        violated_cuts_added=0,
        restored_after_reject=False,
    )
    after_go_failed_closed = False
    try:
        terminal.record_native_trial(
            actor_sha256="after-go",
            fp32_pf7_margin=0.002,
            native_pf7_cell=0.002,
            all_non_pf7_hard_gates_pass=True,
            complete_native_gate_pass=False,
            violated_cuts_added=0,
            restored_after_reject=False,
        )
    except ProtocolError:
        after_go_failed_closed = True
    terminal_audit = terminal.audit()
    tests["terminal_GO_updates_once_and_stops"] = bool(
        terminal_result == "GO"
        and terminal_audit["GO_count"] == 1
        and terminal_audit["terminal_GO_reached"] is True
        and after_go_failed_closed
    )

    capped = Stage2ShadowLedger("anchor", 0.0, 0.0)
    for index in range(STAGE2_MAX_TRIALS):
        capped.record_native_trial(
            actor_sha256=f"reject-{index}",
            fp32_pf7_margin=-1.0,
            native_pf7_cell=-1.0,
            all_non_pf7_hard_gates_pass=False,
            complete_native_gate_pass=False,
            violated_cuts_added=1,
            restored_after_reject=True,
        )
    thirty_third_failed_closed = False
    try:
        capped.record_native_trial(
            actor_sha256="reject-32",
            fp32_pf7_margin=-1.0,
            native_pf7_cell=-1.0,
            all_non_pf7_hard_gates_pass=False,
            complete_native_gate_pass=False,
            violated_cuts_added=1,
            restored_after_reject=True,
        )
    except ProtocolError:
        thirty_third_failed_closed = True
    tests["stage2_32_allowed_33_forbidden"] = bool(
        capped.stage2_trial_count == STAGE2_MAX_TRIALS
        and capped.audit()["total_changed_train_shadows"]
        == TOTAL_CHANGED_TRAIN_SHADOW_CAP
        and thirty_third_failed_closed
    )

    if not all(tests.values()):
        raise ProtocolError(f"CPU solver unit failed: {tests}")
    return {
        "schema_version": SCHEMA,
        "status": "solver_unit_only_pass",
        "numpy_version": np.__version__,
        "CUDA_initialized": False,
        "torch_imported": "torch" in sys.modules,
        "tests": tests,
        "simple_solver": simple_audit,
        "rank_deficient_solver": duplicate_audit,
        "rank_inconsistent_workset_solver": blocker_audit,
        "rowspace_feasibility_small_problem_reports": rowspace_reports,
        "rowspace_feasibility_actor6_width_report": wide_report,
        "secondary_fail_closed": compose_audit,
        "secondary_with_headroom": compose_audit2,
        "shadow_ledger": ledger_audit,
        "writes_performed": 0,
    }


def audit_only() -> dict[str, Any]:
    if Path.cwd().resolve() != ROOT:
        raise ProtocolError("audit-only must run from repository root")
    if "torch" in sys.modules:
        raise ProtocolError("audit-only imported torch before the no-CUDA claim")
    if (
        not path_absent(OUTPUT)
        or not path_absent(ATTEMPT_MARKER)
        or not path_absent(FAILURE)
    ):
        raise ProtocolError("v17 output/attempt/failure targets must be absent")
    staging_residues = {
        str(path.relative_to(ROOT)): publication_staging_residues(path)
        for path in (ATTEMPT_MARKER, OUTPUT, FAILURE)
    }
    if any(staging_residues.values()):
        raise ProtocolError(
            f"v17 publication staging residues must be absent: {staging_residues}"
        )
    dependencies = {
        label: immutable_file_evidence(label, path, digest, mode)
        for label, path, digest, mode in FROZEN_INPUTS
    }
    source = source_audit()
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_pass_production_wiring_unfrozen",
        "source": source,
        "frozen_inputs": dependencies,
        "locked_contract": {
            "stage1": "exact frozen v12 through step29",
            "stage1_execution_seed": STAGE1_EXECUTION_SEED,
            "protocol_id_not_execution_seed": PROTOCOL_ID,
            "stage2_randomness": "none",
            "stage1_changed_train_shadows": STAGE1_REFERENCE_STEPS,
            "stage2_max_native_trials": STAGE2_MAX_TRIALS,
            "total_changed_train_shadow_cap": TOTAL_CHANGED_TRAIN_SHADOW_CAP,
            "one_native_BF16_terminal_gate_per_stage2_trial": True,
            "structural_forward": "direct model(batch) with CUDA autocast disabled",
            "absolute_QP": "min 0.5||x||^2 subject to append-only normalized cuts",
            "cut_formula": "a=g/||g||; c=(b-m+g.dot(d))/||g||; a.dot(x)>=c",
            "PF7_target": "+one actual step29 local BF16 q",
            "PF0_A_B_hard_floor": "+one respective local BF16 q",
            "PF0_secondary_buffer": "+two respective local BF16 q",
            "PF0_secondary_buffer_loss": (
                "mean_FP32_ReLU_hinge(max(0,target_minus_margin)); "
                "not_v13_tempered_softplus"
            ),
            "six_zero_floor": 0.0,
            "dynamic_pairs": "stored thresholds, sticky",
            "historical_threat_cuts": [
                {
                    "line_sha256": line,
                    "row_index": row,
                    "positive_option": positive,
                    "negative_option": negative,
                    "target": target,
                    "target_identity": target_identity(target),
                    "kind": HISTORICAL_TARGET_KIND,
                }
                for line, row, positive, negative, target in HISTORICAL_PAIR_CONTRACT
            ],
            "safe_target_radius": SAFE_TARGET_RADIUS,
            "hard_actual_radius": HARD_ACTUAL_RADIUS,
            "trust_radius": TRUST_RADIUS,
            "pinv_rcond": PINV_RCOND,
            "KKT_tolerance": KKT_TOL,
            "secondary_norm_cap": SECONDARY_NORM_FRACTION * TRUST_RADIUS,
            "secondary": "locked loss descent projected through PF7 plus binding-cut nullspace",
            "no_arc_or_grid_scan": True,
            "NO_GO_payload": None,
        },
        "production_implemented": True,
        "production_armed": stat.S_IMODE(SCRIPT.stat().st_mode) == 0o555,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_absent": True,
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
        "attempt_marker_absent": True,
        "failure": str(FAILURE.relative_to(ROOT)),
        "failure_absent": True,
        "publication_staging_residues_absent": True,
        "python": str(Path(sys.executable).resolve()),
        "python_is_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "CUDA_initialized": False,
        "torch_imported": False,
        "writes_performed": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--solver-unit-only", action="store_true")
    modes.add_argument("--production", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        print(canonical_json(audit_only()).decode("utf-8"), end="")
        return
    if args.solver_unit_only:
        if "torch" in sys.modules:
            raise ProtocolError("CPU solver unit imported torch")
        print(canonical_json(cpu_solver_unit()).decode("utf-8"), end="")
        return
    claim: Mapping[str, Any] | None = None
    try:
        claim = claim_attempt()
        result = run_v17_production(claim)
        payload = canonical_json(result)
        publish_o_excl(OUTPUT, payload)
    except BaseException as error:
        try:
            attempt_consumed = not path_absent(ATTEMPT_MARKER)
        except BaseException:
            attempt_consumed = True
        if claim is None and not attempt_consumed:
            raise
        failure_forensics = best_effort_failure_forensics(claim)
        attempt_reference: Mapping[str, Any] = {}
        if isinstance(claim, Mapping) and isinstance(
            claim.get("publication"), Mapping
        ):
            attempt_reference = claim["publication"]
        else:
            observed_attempt = (
                failure_forensics.get("observed_bindings", {}).get("attempt", {})
            )
            if isinstance(observed_attempt.get("binding"), Mapping):
                attempt_reference = observed_attempt["binding"]
        failure_payload = {
            "schema_version": f"{SCHEMA}-failure",
            "status": "production_failed_after_consuming_one_shot_attempt",
            "protocol_id": PROTOCOL_ID,
            "stage1_execution_seed": STAGE1_EXECUTION_SEED,
            "attempt": attempt_reference,
            "exception_type": type(error).__name__,
            "exception_message_sha256": hashlib.sha256(
                str(error).encode("utf-8", errors="replace")
            ).hexdigest(),
            "post_failure_forensics": failure_forensics,
            "candidate_payload": None,
            "retry_authorized": False,
            "submission_performed": False,
            "package_upload_performed": False,
            "official_unique_changed_candidate_count_consumed": 0,
            "cumulative_official_unique_changed_candidate_count": 2,
        }
        failure_payload["NO_GO_material_audit"] = no_go_material_audit(
            failure_payload
        )
        try:
            publish_o_excl(FAILURE, canonical_json(failure_payload))
        except BaseException as publication_error:
            try:
                error.add_note(
                    "v17 failure publication error sha256="
                    + hashlib.sha256(
                        (
                            type(publication_error).__name__
                            + ":"
                            + str(publication_error)
                        ).encode("utf-8", errors="replace")
                    ).hexdigest()
                )
            except BaseException:
                pass
        raise
    print(payload.decode("utf-8"), end="")


if __name__ == "__main__":
    main()

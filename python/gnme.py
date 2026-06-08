from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import prod
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from numpy.typing import NDArray

ComplexVector = NDArray[np.complex128]
ComplexMatrix = NDArray[np.complex128]

__version__ = "0.2.2-cudss-linear-solver-no-stale-gpu-kwargs"
__artifact_build__ = "2026-06-08 verified: no solve_kwargs gpu/use_indirect false"


# -----------------------------------------------------------------------------
# Sparse subsystem maps
# -----------------------------------------------------------------------------

def _basis_index(indices: Sequence[int], dims: Sequence[int]) -> int:
    """
    Row-major subsystem basis index:
        |i0 i1 ... in> -> (((i0*d1 + i1)*d2 + i2) ...)
    """
    out = 0
    for i, d in zip(indices, dims):
        out = out * int(d) + int(i)
    return out


@lru_cache(maxsize=None)
def partial_transpose_vec_map(
    dims: tuple[int, ...],
    subsys: tuple[int, ...],
) -> sp.csr_matrix:
    """
    Sparse permutation P such that:

        vec(X^T_subsys) = P @ vec(X)

    where vec uses column-major/F order, matching cp.vec(X, order="F").
    """
    dims = tuple(int(x) for x in dims)
    subsys = tuple(sorted(set(int(x) for x in subsys)))

    n = len(dims)
    if any(s < 0 or s >= n for s in subsys):
        raise ValueError(f"Invalid partial-transpose subsystem(s): {subsys}")

    d = prod(dims)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for I in range(d):
        ket = list(np.unravel_index(I, dims, order="C"))

        for J in range(d):
            bra = list(np.unravel_index(J, dims, order="C"))

            ket_pt = ket.copy()
            bra_pt = bra.copy()

            for s in subsys:
                ket_pt[s], bra_pt[s] = bra_pt[s], ket_pt[s]

            I_pt = _basis_index(ket_pt, dims)
            J_pt = _basis_index(bra_pt, dims)

            # F-order vectorization index: vec(X)[I + J*d] = X[I, J]
            col = I + J * d
            row = I_pt + J_pt * d

            rows.append(row)
            cols.append(col)
            data.append(1.0)

    return sp.coo_matrix(
        (data, (rows, cols)),
        shape=(d * d, d * d),
        dtype=np.float64,
    ).tocsr()


def partial_transpose_expr_sparse(
    X: cp.Expression,
    dims: Sequence[int],
    subsys: Sequence[int],
) -> cp.Expression:
    dims_t = tuple(int(x) for x in dims)
    subsys_t = tuple(sorted(set(int(x) for x in subsys)))

    d = prod(dims_t)
    P = partial_transpose_vec_map(dims_t, subsys_t)

    v = P @ cp.vec(X, order="F")
    return cp.reshape(v, (d, d), order="F")


@lru_cache(maxsize=None)
def ordered_marginal_vec_map(
    dims: tuple[int, ...],
    keep: tuple[int, ...],
) -> tuple[sp.csr_matrix, int]:
    """
    Sparse map L such that:

        vec(X_keep_ordered) = L @ vec(X)

    where vec uses column-major/F order, matching cp.vec(X, order="F").

    `keep` gives the output subsystem order. Example:
        keep=(2,3,5,0) means output order 2 ⊗ 3 ⊗ 5 ⊗ 0.
    """
    dims = tuple(int(x) for x in dims)
    keep = tuple(int(x) for x in keep)

    n = len(dims)
    if len(set(keep)) != len(keep):
        raise ValueError(f"Repeated subsystem in keep={keep}")
    if any(k < 0 or k >= n for k in keep):
        raise ValueError(f"Invalid keep={keep} for dims={dims}")

    trace = tuple(i for i in range(n) if i not in keep)

    d_in = prod(dims)
    out_dims = tuple(dims[i] for i in keep)
    d_out = prod(out_dims)

    trace_ranges = [range(dims[i]) for i in trace]

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for a_flat in range(d_out):
        a_multi = np.unravel_index(a_flat, out_dims, order="C")

        for b_flat in range(d_out):
            b_multi = np.unravel_index(b_flat, out_dims, order="C")
            out_vec_idx = a_flat + b_flat * d_out

            for traced_vals in (product(*trace_ranges) if trace else [()]):
                ket = [0] * n
                bra = [0] * n

                for pos, subsystem in enumerate(keep):
                    ket[subsystem] = int(a_multi[pos])
                    bra[subsystem] = int(b_multi[pos])

                for pos, subsystem in enumerate(trace):
                    ket[subsystem] = int(traced_vals[pos])
                    bra[subsystem] = int(traced_vals[pos])

                I = _basis_index(ket, dims)
                J = _basis_index(bra, dims)
                in_vec_idx = I + J * d_in

                rows.append(out_vec_idx)
                cols.append(in_vec_idx)
                data.append(1.0)

    L = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(d_out * d_out, d_in * d_in),
        dtype=np.float64,
    ).tocsr()

    return L, d_out


def marginal_vec_expr(
    X: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
) -> cp.Expression:
    """
    Vectorized ordered marginal:
        vec(X_keep_ordered), using column-major/F order.
    """
    dims_t = tuple(int(x) for x in dims)
    keep_t = tuple(int(x) for x in keep)

    L, _ = ordered_marginal_vec_map(dims_t, keep_t)
    return L @ cp.vec(X, order="F")


def marginal_matrix_expr(
    X: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
) -> cp.Expression:
    """
    Matrix ordered marginal X_keep, ordered according to `keep`.
    Use this only when a matrix expression is needed, e.g. for PSD/PPT.
    """
    dims_t = tuple(int(x) for x in dims)
    keep_t = tuple(int(x) for x in keep)

    L, d_out = ordered_marginal_vec_map(dims_t, keep_t)
    v = L @ cp.vec(X, order="F")
    return cp.reshape(v, (d_out, d_out), order="F")


# -----------------------------------------------------------------------------
# Constraint compression helpers
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _diag_and_upper_f_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    F-order vec indices for a Hermitian n x n matrix's independent entries.
    Diagonal entries are real; strict upper-triangle entries are complex.
    """
    diag = np.array([i + i * n for i in range(n)], dtype=np.int64)
    upper = np.array([i + j * n for j in range(n) for i in range(j)], dtype=np.int64)
    return diag, upper


def hermitian_vec_eq(
    lhs: cp.Expression,
    rhs: cp.Expression | np.ndarray,
    n: int,
) -> list[cp.Constraint]:
    """
    Equality of two Hermitian n x n matrices represented as F-order vectors.

    This imposes only independent Hermitian entries instead of duplicating the
    lower triangle. It is equivalent when both sides are Hermitian.
    """
    diag, upper = _diag_and_upper_f_indices(int(n))
    return [
        cp.real(lhs[diag]) == cp.real(rhs[diag]),
        lhs[upper] == rhs[upper],
    ]


# -----------------------------------------------------------------------------
# Cached 3-qubit ring-inflation SDP
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InflationResult:
    score: float | None
    raw_score: float | None
    status: str
    solve_time: float | None
    setup_time: float | None
    num_iters: int | None


class RingInflationSDP3Q:
    """
    Cached level-2 ring-inflation SDP for three qubit parties.

    This class builds the CVXPY expression graph once, then updates a parameter
    for each new 8x8 rho. This is the main speed improvement for scans over many
    reduced density matrices.
    """

    dims: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    d: int = 64
    rho_dim: int = 8
    rho_vec_dim: int = 64

    def __init__(self) -> None:
        dims = self.dims
        d = self.d

        self.t = cp.Variable(name="t")
        self.tau = cp.Variable((d, d), hermitian=True, name="tau")
        self.gamma = cp.Variable((d, d), hermitian=True, name="gamma")

        self.white_vec = (np.eye(self.rho_dim, dtype=np.complex128) / self.rho_dim).reshape(
            self.rho_vec_dim,
            order="F",
        )

        # Parameterized as white + t*(rho-white). This avoids rebuilding the
        # problem and also makes t=0 a fixed feasible noise point.
        self.delta_vec = cp.Parameter((self.rho_vec_dim,), complex=True, name="rho_minus_white_vec")

        constraints: list[cp.Constraint] = [
            self.t >= 0,
            self.t <= 1,
            self.tau >> 0,
            self.gamma >> 0,
            cp.trace(self.tau) == 1,
            cp.trace(self.gamma) == 1,
        ]

        # tau_012 = I/8 + t*(rho - I/8)
        constraints += hermitian_vec_eq(
            marginal_vec_expr(self.tau, dims, [0, 1, 2]),
            self.white_vec + self.t * self.delta_vec,
            self.rho_dim,
        )

        # Ring-inflation marginal equality constraints.
        constraints += hermitian_vec_eq(
            marginal_vec_expr(self.gamma, dims, [0, 1, 3, 4]),
            marginal_vec_expr(self.tau, dims, [0, 1, 3, 4]),
            16,
        )

        constraints += hermitian_vec_eq(
            marginal_vec_expr(self.gamma, dims, [1, 2, 4, 5]),
            marginal_vec_expr(self.tau, dims, [1, 2, 4, 5]),
            16,
        )

        constraints += hermitian_vec_eq(
            marginal_vec_expr(self.gamma, dims, [2, 3, 5, 0]),
            marginal_vec_expr(self.tau, dims, [2, 0, 5, 3]),
            16,
        )

        # Copy-swap symmetry constraints.
        constraints += hermitian_vec_eq(
            marginal_vec_expr(self.tau, dims, [3, 4, 5, 0, 1, 2]),
            cp.vec(self.tau, order="F"),
            64,
        )

        constraints += hermitian_vec_eq(
            marginal_vec_expr(self.gamma, dims, [3, 4, 5, 0, 1, 2]),
            cp.vec(self.gamma, order="F"),
            64,
        )

        # PPT constraints.
        tau_T012 = partial_transpose_expr_sparse(self.tau, dims, [0, 1, 2])

        gamma_0124 = marginal_matrix_expr(self.gamma, dims, [0, 1, 2, 4])
        gamma_1235 = marginal_matrix_expr(self.gamma, dims, [1, 2, 3, 5])
        gamma_2340 = marginal_matrix_expr(self.gamma, dims, [2, 3, 4, 0])

        gamma_0124_T4 = partial_transpose_expr_sparse(gamma_0124, [2, 2, 2, 2], [3])
        gamma_1235_T5 = partial_transpose_expr_sparse(gamma_1235, [2, 2, 2, 2], [3])
        gamma_2340_T0 = partial_transpose_expr_sparse(gamma_2340, [2, 2, 2, 2], [3])

        constraints += [
            (tau_T012 + tau_T012.H) / 2 >> 0,
            (gamma_0124_T4 + gamma_0124_T4.H) / 2 >> 0,
            (gamma_1235_T5 + gamma_1235_T5.H) / 2 >> 0,
            (gamma_2340_T0 + gamma_2340_T0.H) / 2 >> 0,
        ]

        self.problem = cp.Problem(cp.Maximize(cp.real(self.t)), constraints)

    def solve(
        self,
        rho: ComplexMatrix,
        *,
        verbose: bool = False,
        max_iters: int = 10_000,
        tol: float = 1e-4,
        warm_start: bool = True,
        gpu: bool = False,
        use_indirect: bool = False,
        acceleration_lookback: int = 10,
        normalize: bool = True,
        return_info: bool = False,
        **scs_options: Any,
    ) -> float | InflationResult:
        rho = _validate_3q_density_matrix(rho)
        rho_vec = rho.reshape(self.rho_vec_dim, order="F")
        self.delta_vec.value = rho_vec - self.white_vec

        solve_kwargs: dict[str, Any] = dict(
            solver=cp.SCS,
            verbose=verbose,
            warm_start=warm_start,
            max_iters=max_iters,
            eps=tol,
            acceleration_lookback=acceleration_lookback,
            normalize=normalize,
        )

        # ARTIFACT_VERIFICATION_SENTINEL: GPU uses scs.LinearSolver.CUDSS; no gpu=True/use_indirect=False kwargs are passed to SCS.
        # Newer scs-python builds select GPU/cuDSS through the LinearSolver
        # backend enum. Do not pass gpu=True/gpu=False to SCS here: in current
        # scs-python it is not a valid keyword for the CPU extension, and it may
        # be rejected before backend dispatch.
        if gpu:
            if use_indirect:
                raise ValueError(
                    "gpu=True selects the direct cuDSS backend; do not also set "
                    "use_indirect=True."
                )

            try:
                import scs
            except ImportError as exc:  # pragma: no cover - environment issue
                raise RuntimeError(
                    "gpu=True requires an SCS build with the cuDSS backend."
                ) from exc

            try:
                solve_kwargs["linear_solver"] = scs.LinearSolver.CUDSS
            except AttributeError as exc:  # pragma: no cover - old scs-python
                raise RuntimeError(
                    "This SCS build does not expose scs.LinearSolver.CUDSS. "
                    "Install/rebuild scs-python with cuDSS support, or call with "
                    "gpu=False."
                ) from exc
        elif use_indirect:
            # Keep CPU indirect mode available without passing the old
            # use_indirect=False keyword on normal CPU/direct solves. Prefer the
            # newer LinearSolver enum when available, but fall back to CVXPY's
            # traditional use_indirect=True option for older SCS wrappers.
            try:
                import scs
                solve_kwargs["linear_solver"] = scs.LinearSolver.CPU_INDIRECT
            except Exception:
                solve_kwargs["use_indirect"] = True

        solve_kwargs.update(scs_options)
        self.problem.solve(**solve_kwargs)

        raw = None if self.problem.value is None else float(np.real(self.problem.value))
        score = None if raw is None else min(raw, 1.0)

        if not return_info:
            return score

        stats = self.problem.solver_stats
        return InflationResult(
            score=score,
            raw_score=raw,
            status=self.problem.status,
            solve_time=getattr(stats, "solve_time", None),
            setup_time=getattr(stats, "setup_time", None),
            num_iters=getattr(stats, "num_iters", None),
        )


_TEMPLATE_3Q: RingInflationSDP3Q | None = None


def get_ring_inflation_sdp_3q(*, rebuild: bool = False) -> RingInflationSDP3Q:
    """Return the cached 3-qubit SDP template."""
    global _TEMPLATE_3Q
    if rebuild or _TEMPLATE_3Q is None:
        _TEMPLATE_3Q = RingInflationSDP3Q()
    return _TEMPLATE_3Q


def inflation_score(
    rho: ComplexMatrix,
    verbose: bool = False,
    max_iters: int = 10_000,
    tol: float = 1e-4,
    *,
    warm_start: bool = True,
    gpu: bool = False,
    use_indirect: bool = False,
    acceleration_lookback: int = 10,
    normalize: bool = True,
    return_info: bool = False,
    rebuild_problem: bool = False,
    **scs_options: Any,
) -> float | InflationResult:
    """
    Level-2 ring-inflation score for a 3-qubit density matrix rho.

    Parameters
    ----------
    rho:
        8x8 density matrix over A,B,C, each a qubit.

    verbose, max_iters, tol:
        Passed to SCS. `tol` is mapped to SCS `eps` for compatibility with your
        current call style.

    gpu:
        If True, select the SCS cuDSS backend via
        `linear_solver=scs.LinearSolver.CUDSS`. This requires scs-python to be
        built with cuDSS support and the relevant NVIDIA libraries visible at
        runtime.

    return_info:
        If True, return an InflationResult containing raw score, clipped score,
        solver status, timing, and iterations.

    rebuild_problem:
        Force construction of a fresh CVXPY problem template.
    """
    sdp = get_ring_inflation_sdp_3q(rebuild=rebuild_problem)
    return sdp.solve(
        rho,
        verbose=verbose,
        max_iters=max_iters,
        tol=tol,
        warm_start=warm_start,
        gpu=gpu,
        use_indirect=use_indirect,
        acceleration_lookback=acceleration_lookback,
        normalize=normalize,
        return_info=return_info,
        **scs_options,
    )


# -----------------------------------------------------------------------------
# Validation / utility functions
# -----------------------------------------------------------------------------

def _validate_3q_density_matrix(rho: ComplexMatrix) -> ComplexMatrix:
    rho = np.asarray(rho, dtype=np.complex128)

    if rho.shape != (8, 8):
        raise ValueError(
            f"inflation_score currently supports only 3-qubit / 8x8 RDMs; "
            f"got shape {rho.shape}. For higher-dimensional tripartitions, "
            f"the SDP dimensions and PPT marginal dimensions must be generalized."
        )

    # Cheap cleanup for numerical RDMs from simulations.
    rho = (rho + rho.conj().T) / 2
    tr = np.trace(rho)
    if abs(tr) == 0:
        raise ValueError("rho has zero trace")
    rho = rho / tr

    return rho


def reset_cache() -> None:
    """Clear the cached CVXPY problem and sparse maps."""
    global _TEMPLATE_3Q
    _TEMPLATE_3Q = None
    partial_transpose_vec_map.cache_clear()
    ordered_marginal_vec_map.cache_clear()
    _diag_and_upper_f_indices.cache_clear()

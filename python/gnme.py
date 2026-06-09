from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import prod, sqrt
from typing import Any, Literal, Sequence

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from numpy.typing import NDArray

ComplexVector = NDArray[np.complex128]
ComplexMatrix = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]

__version__ = "0.5.1-sdp-optimize-only-swap-blocks-objective-fix"
__artifact_build__ = (
    "2026-06-09 v2 optimize-only; removed fast/auto/feasible/bisection/project_real/ppt approximations; "
    "exact copy-swap block parametrization; real-domain auto template; direct sparse marginal/PT maps; "
    "objective uses bare real scalar t to avoid CVXPY cp.real(real-variable) canonicalization bug; no stale gpu kwargs"
)


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
def _basis_permutation_for_keep(
    dims: tuple[int, ...],
    keep: tuple[int, ...],
) -> np.ndarray:
    """
    Return perm such that Y = X_keep_full_order has
        Y[a,b] = X[perm[a], perm[b]]
    when keep is a full subsystem permutation.
    """
    dims = tuple(int(x) for x in dims)
    keep = tuple(int(x) for x in keep)
    n = len(dims)
    if sorted(keep) != list(range(n)):
        raise ValueError(f"keep={keep} is not a full subsystem permutation")

    d = prod(dims)
    out_dims = tuple(dims[i] for i in keep)
    perm = np.empty(d, dtype=np.int64)
    for a_flat in range(d):
        a_multi = np.unravel_index(a_flat, out_dims, order="C")
        ket = [0] * n
        for pos, subsystem in enumerate(keep):
            ket[subsystem] = int(a_multi[pos])
        perm[a_flat] = _basis_index(ket, dims)
    return perm


@lru_cache(maxsize=None)
def partial_transpose_vec_map(
    dims: tuple[int, ...],
    subsys: tuple[int, ...],
) -> sp.csc_matrix:
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

            # F-order vectorization index: vec(X)[I + J*d] = X[I, J].
            cols.append(I + J * d)
            rows.append(I_pt + J_pt * d)

    data = np.ones(len(rows), dtype=np.float64)
    return sp.coo_matrix((data, (rows, cols)), shape=(d * d, d * d)).tocsc()


def partial_transpose_vec_expr_from_vec(
    X_vec: cp.Expression,
    dims: Sequence[int],
    subsys: Sequence[int],
) -> cp.Expression:
    dims_t = tuple(int(x) for x in dims)
    subsys_t = tuple(sorted(set(int(x) for x in subsys)))
    P = partial_transpose_vec_map(dims_t, subsys_t)
    return P @ X_vec


def matrix_from_vec(v: cp.Expression, d: int) -> cp.Expression:
    return cp.reshape(v, (int(d), int(d)), order="F")


def partial_transpose_expr_sparse(
    X: cp.Expression,
    dims: Sequence[int],
    subsys: Sequence[int],
) -> cp.Expression:
    """Compatibility wrapper retained for external callers."""
    dims_t = tuple(int(x) for x in dims)
    d = prod(dims_t)
    v = partial_transpose_vec_expr_from_vec(cp.vec(X, order="F"), dims_t, subsys)
    return matrix_from_vec(v, d)


@lru_cache(maxsize=None)
def ordered_marginal_vec_map(
    dims: tuple[int, ...],
    keep: tuple[int, ...],
) -> tuple[sp.csc_matrix, int]:
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
                rows.append(out_vec_idx)
                cols.append(I + J * d_in)

    data = np.ones(len(rows), dtype=np.float64)
    L = sp.coo_matrix((data, (rows, cols)), shape=(d_out * d_out, d_in * d_in)).tocsc()
    return L, d_out


def marginal_vec_expr_from_vec(
    X_vec: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
) -> cp.Expression:
    dims_t = tuple(int(x) for x in dims)
    keep_t = tuple(int(x) for x in keep)
    L, _ = ordered_marginal_vec_map(dims_t, keep_t)
    return L @ X_vec


def marginal_vec_expr(
    X: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
) -> cp.Expression:
    """Compatibility wrapper retained for external callers."""
    return marginal_vec_expr_from_vec(cp.vec(X, order="F"), dims, keep)


def marginal_matrix_expr_from_vec(
    X_vec: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
) -> cp.Expression:
    dims_t = tuple(int(x) for x in dims)
    keep_t = tuple(int(x) for x in keep)
    L, d_out = ordered_marginal_vec_map(dims_t, keep_t)
    return matrix_from_vec(L @ X_vec, d_out)


def marginal_matrix_expr(
    X: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
) -> cp.Expression:
    """Compatibility wrapper retained for external callers."""
    return marginal_matrix_expr_from_vec(cp.vec(X, order="F"), dims, keep)


def partial_transposed_marginal_matrix_expr_from_vec(
    X_vec: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
    pt_subsys_in_output_order: Sequence[int],
) -> cp.Expression:
    """
    Directly build reshape(P_pt @ L_keep @ vec(X)). This avoids the slower
    nested reshape(vec(reshape(...))) graph produced by applying partial
    transpose to an already-reshaped marginal expression.
    """
    dims_t = tuple(int(x) for x in dims)
    keep_t = tuple(int(x) for x in keep)
    out_dims = tuple(dims_t[i] for i in keep_t)

    L, d_out = ordered_marginal_vec_map(dims_t, keep_t)
    P = partial_transpose_vec_map(out_dims, tuple(int(x) for x in pt_subsys_in_output_order))
    return matrix_from_vec((P @ L) @ X_vec, d_out)


# -----------------------------------------------------------------------------
# Exact copy-swap block basis
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def copy_swap_block_basis(copy_dim: int = 8) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Orthonormal basis matrices for the +1 and -1 eigenspaces of the copy-swap
    operator |a,b> -> |b,a> on C^copy_dim ⊗ C^copy_dim.

    Any Hermitian matrix X satisfying P X P^T = X can be written exactly as:

        X = U_plus X_plus U_plus.T + U_minus X_minus U_minus.T

    with X_plus Hermitian on dim copy_dim*(copy_dim+1)/2 and X_minus Hermitian
    on dim copy_dim*(copy_dim-1)/2. This removes the copy-swap equality
    constraints and replaces one 64x64 PSD cone by 36x36 and 28x28 cones.
    """
    m = int(copy_dim)
    d = m * m

    plus_rows: list[int] = []
    plus_cols: list[int] = []
    plus_data: list[float] = []
    minus_rows: list[int] = []
    minus_cols: list[int] = []
    minus_data: list[float] = []

    pcol = 0
    mcol = 0
    inv_sqrt2 = 1.0 / sqrt(2.0)

    for a in range(m):
        for b in range(a, m):
            idx_ab = a * m + b
            idx_ba = b * m + a
            if a == b:
                plus_rows.append(idx_ab)
                plus_cols.append(pcol)
                plus_data.append(1.0)
            else:
                plus_rows.extend([idx_ab, idx_ba])
                plus_cols.extend([pcol, pcol])
                plus_data.extend([inv_sqrt2, inv_sqrt2])

                minus_rows.extend([idx_ab, idx_ba])
                minus_cols.extend([mcol, mcol])
                minus_data.extend([inv_sqrt2, -inv_sqrt2])
                mcol += 1
            pcol += 1

    U_plus = sp.coo_matrix((plus_data, (plus_rows, plus_cols)), shape=(d, pcol)).tocsc()
    U_minus = sp.coo_matrix((minus_data, (minus_rows, minus_cols)), shape=(d, mcol)).tocsc()
    return U_plus, U_minus


# -----------------------------------------------------------------------------
# Constraint compression helpers
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _diag_and_upper_f_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    F-order vec indices for a Hermitian n x n matrix's independent entries.
    Diagonal entries are real; strict upper-triangle entries are complex.
    """
    n = int(n)
    diag = np.array([i + i * n for i in range(n)], dtype=np.int64)
    upper = np.array([i + j * n for j in range(n) for i in range(j)], dtype=np.int64)
    return diag, upper


@lru_cache(maxsize=None)
def _upper_f_indices_including_diag(n: int) -> np.ndarray:
    """F-order indices for i <= j."""
    n = int(n)
    return np.array([i + j * n for j in range(n) for i in range(j + 1)], dtype=np.int64)


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


def symmetric_vec_eq(
    lhs: cp.Expression,
    rhs: cp.Expression | np.ndarray,
    n: int,
) -> list[cp.Constraint]:
    """
    Equality of two real symmetric n x n matrices represented as F-order vecs.
    Uses the independent upper triangle including the diagonal.
    """
    upper = _upper_f_indices_including_diag(int(n))
    return [lhs[upper] == rhs[upper]]


def matrix_vec_eq(
    lhs: cp.Expression,
    rhs: cp.Expression | np.ndarray,
    n: int,
    *,
    real_problem: bool,
) -> list[cp.Constraint]:
    if real_problem:
        return symmetric_vec_eq(lhs, rhs, n)
    return hermitian_vec_eq(lhs, rhs, n)


@lru_cache(maxsize=None)
def _copy_swap_symmetry_index_groups(
    dims: tuple[int, ...],
    keep: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build compressed index groups for P X P^T == X on a Hermitian matrix.

    This fallback is exact, but slower than the block-parametrized form above.
    """
    dims = tuple(int(x) for x in dims)
    keep = tuple(int(x) for x in keep)
    d = prod(dims)
    perm = _basis_permutation_for_keep(dims, keep)

    same_lhs: list[int] = []
    same_rhs: list[int] = []
    conj_lhs: list[int] = []
    conj_rhs: list[int] = []
    real_idx: list[int] = []

    for j in range(d):
        for i in range(j + 1):
            key = (i, j)
            ip = int(perm[i])
            jp = int(perm[j])

            mapped_key = (ip, jp) if ip <= jp else (jp, ip)
            idx = i + j * d
            mapped_idx = mapped_key[0] + mapped_key[1] * d

            if mapped_key == key:
                if i < j and ip == j and jp == i:
                    real_idx.append(idx)
                continue

            if key < mapped_key:
                if ip <= jp:
                    same_lhs.append(idx)
                    same_rhs.append(mapped_idx)
                else:
                    conj_lhs.append(idx)
                    conj_rhs.append(mapped_idx)

    return (
        np.asarray(same_lhs, dtype=np.int64),
        np.asarray(same_rhs, dtype=np.int64),
        np.asarray(conj_lhs, dtype=np.int64),
        np.asarray(conj_rhs, dtype=np.int64),
        np.asarray(real_idx, dtype=np.int64),
    )


def copy_swap_symmetry_constraints(
    X_vec: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
    *,
    real_problem: bool,
) -> list[cp.Constraint]:
    """Compressed exact fallback for P X P^T == X."""
    groups = _copy_swap_symmetry_index_groups(tuple(int(x) for x in dims), tuple(int(x) for x in keep))
    same_lhs, same_rhs, conj_lhs, conj_rhs, real_idx = groups

    constraints: list[cp.Constraint] = []
    if same_lhs.size:
        constraints.append(X_vec[same_lhs] == X_vec[same_rhs])
    if conj_lhs.size:
        if real_problem:
            constraints.append(X_vec[conj_lhs] == X_vec[conj_rhs])
        else:
            constraints.append(X_vec[conj_lhs] == cp.conj(X_vec[conj_rhs]))
    if real_idx.size and not real_problem:
        constraints.append(cp.imag(X_vec[real_idx]) == 0)
    return constraints


# -----------------------------------------------------------------------------
# Solver helpers
# -----------------------------------------------------------------------------

def _make_solve_kwargs(
    *,
    solver: Any,
    verbose: bool,
    warm_start: bool,
    max_iters: int,
    tol: float,
    gpu: bool,
    use_indirect: bool,
    acceleration_lookback: int,
    normalize: bool,
    extra_options: dict[str, Any],
) -> dict[str, Any]:
    """Build CVXPY/SCS kwargs while avoiding stale gpu/use_indirect kwargs."""
    solve_kwargs: dict[str, Any] = dict(
        solver=solver,
        verbose=verbose,
        warm_start=warm_start,
    )

    if solver == cp.SCS or str(solver).upper() == "SCS":
        solve_kwargs.update(
            max_iters=int(max_iters),
            eps=float(tol),
            acceleration_lookback=int(acceleration_lookback),
            normalize=bool(normalize),
        )

        # ARTIFACT_VERIFICATION_SENTINEL: GPU uses scs.LinearSolver.CUDSS; no gpu=True/use_indirect=False kwargs are passed to SCS.
        if gpu:
            if use_indirect:
                raise ValueError("gpu=True selects the direct cuDSS backend; do not also set use_indirect=True.")
            try:
                import scs
            except ImportError as exc:  # pragma: no cover - environment issue
                raise RuntimeError("gpu=True requires an SCS build with cuDSS support.") from exc
            try:
                solve_kwargs["linear_solver"] = scs.LinearSolver.CUDSS
            except AttributeError as exc:  # pragma: no cover - old scs-python
                raise RuntimeError(
                    "This SCS build does not expose scs.LinearSolver.CUDSS. "
                    "Install/rebuild scs-python with cuDSS support, or call with gpu=False."
                ) from exc
        elif use_indirect:
            try:
                import scs
                solve_kwargs["linear_solver"] = scs.LinearSolver.CPU_INDIRECT
            except Exception:
                solve_kwargs["use_indirect"] = True

    solve_kwargs.update(extra_options)
    return solve_kwargs


# -----------------------------------------------------------------------------
# Cached 3-qubit ring-inflation SDP optimization template
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InflationResult:
    score: float | None
    raw_score: float | None
    status: str
    solve_time: float | None
    setup_time: float | None
    num_iters: int | None
    strategy: str = "optimize"
    real_problem: bool = False


class _RingInflationTemplate3Q:
    """Parameterized maximize-t ring-inflation SDP template."""

    dims: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    d: int = 64
    rho_dim: int = 8
    rho_vec_dim: int = 64

    def __init__(self, *, real_problem: bool, use_swap_blocks: bool = True) -> None:
        self.real_problem = bool(real_problem)
        self.use_swap_blocks = bool(use_swap_blocks)
        dims = self.dims
        d = self.d

        self.t = cp.Variable(name="t")

        constraints: list[cp.Constraint] = [self.t >= 0, self.t <= 1]

        if self.real_problem:
            self.delta_vec = cp.Parameter((self.rho_vec_dim,), name="rho_minus_white_vec")
            white_mat = np.eye(self.rho_dim, dtype=np.float64) / self.rho_dim
        else:
            self.delta_vec = cp.Parameter((self.rho_vec_dim,), complex=True, name="rho_minus_white_vec")
            white_mat = np.eye(self.rho_dim, dtype=np.complex128) / self.rho_dim

        self.white_vec = white_mat.reshape(self.rho_vec_dim, order="F")

        if self.use_swap_blocks:
            U_plus, U_minus = copy_swap_block_basis(8)
            plus_dim = U_plus.shape[1]
            minus_dim = U_minus.shape[1]

            if self.real_problem:
                self.tau_plus = cp.Variable((plus_dim, plus_dim), symmetric=True, name="tau_plus")
                self.tau_minus = cp.Variable((minus_dim, minus_dim), symmetric=True, name="tau_minus")
                self.gamma_plus = cp.Variable((plus_dim, plus_dim), symmetric=True, name="gamma_plus")
                self.gamma_minus = cp.Variable((minus_dim, minus_dim), symmetric=True, name="gamma_minus")
            else:
                self.tau_plus = cp.Variable((plus_dim, plus_dim), hermitian=True, name="tau_plus")
                self.tau_minus = cp.Variable((minus_dim, minus_dim), hermitian=True, name="tau_minus")
                self.gamma_plus = cp.Variable((plus_dim, plus_dim), hermitian=True, name="gamma_plus")
                self.gamma_minus = cp.Variable((minus_dim, minus_dim), hermitian=True, name="gamma_minus")

            self.tau = U_plus @ self.tau_plus @ U_plus.T + U_minus @ self.tau_minus @ U_minus.T
            self.gamma = U_plus @ self.gamma_plus @ U_plus.T + U_minus @ self.gamma_minus @ U_minus.T
            constraints += [
                self.tau_plus >> 0,
                self.tau_minus >> 0,
                self.gamma_plus >> 0,
                self.gamma_minus >> 0,
            ]
        else:
            if self.real_problem:
                self.tau = cp.Variable((d, d), symmetric=True, name="tau")
                self.gamma = cp.Variable((d, d), symmetric=True, name="gamma")
            else:
                self.tau = cp.Variable((d, d), hermitian=True, name="tau")
                self.gamma = cp.Variable((d, d), hermitian=True, name="gamma")
            constraints += [self.tau >> 0, self.gamma >> 0]

        tau_vec = cp.vec(self.tau, order="F")
        gamma_vec = cp.vec(self.gamma, order="F")

        # The marginal equality below already fixes Tr(tau)=1 because the target
        # marginal has trace 1. The first gamma/tau marginal equality then fixes
        # Tr(gamma)=1. Explicit trace constraints are therefore redundant and are
        # intentionally omitted.

        # tau_012 = I/8 + t*(rho - I/8)
        constraints += matrix_vec_eq(
            marginal_vec_expr_from_vec(tau_vec, dims, [0, 1, 2]),
            self.white_vec + self.t * self.delta_vec,
            self.rho_dim,
            real_problem=self.real_problem,
        )

        # Ring-inflation marginal equality constraints.
        constraints += matrix_vec_eq(
            marginal_vec_expr_from_vec(gamma_vec, dims, [0, 1, 3, 4]),
            marginal_vec_expr_from_vec(tau_vec, dims, [0, 1, 3, 4]),
            16,
            real_problem=self.real_problem,
        )
        constraints += matrix_vec_eq(
            marginal_vec_expr_from_vec(gamma_vec, dims, [1, 2, 4, 5]),
            marginal_vec_expr_from_vec(tau_vec, dims, [1, 2, 4, 5]),
            16,
            real_problem=self.real_problem,
        )
        constraints += matrix_vec_eq(
            marginal_vec_expr_from_vec(gamma_vec, dims, [2, 3, 5, 0]),
            marginal_vec_expr_from_vec(tau_vec, dims, [2, 0, 5, 3]),
            16,
            real_problem=self.real_problem,
        )

        if not self.use_swap_blocks:
            # Exact fallback if the block parametrization is disabled.
            copy_keep = [3, 4, 5, 0, 1, 2]
            constraints += copy_swap_symmetry_constraints(
                tau_vec, dims, copy_keep, real_problem=self.real_problem
            )
            constraints += copy_swap_symmetry_constraints(
                gamma_vec, dims, copy_keep, real_problem=self.real_problem
            )

        # Full original PPT constraints. No tau_only/none approximation remains.
        tau_T012 = matrix_from_vec(
            partial_transpose_vec_expr_from_vec(tau_vec, dims, [0, 1, 2]),
            d,
        )
        constraints.append(
            (tau_T012 + tau_T012.T) / 2 >> 0
            if self.real_problem
            else (tau_T012 + tau_T012.H) / 2 >> 0
        )

        gamma_0124_T4 = partial_transposed_marginal_matrix_expr_from_vec(
            gamma_vec, dims, [0, 1, 2, 4], [3]
        )
        gamma_1235_T5 = partial_transposed_marginal_matrix_expr_from_vec(
            gamma_vec, dims, [1, 2, 3, 5], [3]
        )
        gamma_2340_T0 = partial_transposed_marginal_matrix_expr_from_vec(
            gamma_vec, dims, [2, 3, 4, 0], [3]
        )
        constraints += [
            (gamma_0124_T4 + gamma_0124_T4.T) / 2 >> 0
            if self.real_problem
            else (gamma_0124_T4 + gamma_0124_T4.H) / 2 >> 0,
            (gamma_1235_T5 + gamma_1235_T5.T) / 2 >> 0
            if self.real_problem
            else (gamma_1235_T5 + gamma_1235_T5.H) / 2 >> 0,
            (gamma_2340_T0 + gamma_2340_T0.T) / 2 >> 0
            if self.real_problem
            else (gamma_2340_T0 + gamma_2340_T0.H) / 2 >> 0,
        ]

        self.problem = cp.Problem(cp.Maximize(self.t), constraints)

    def set_rho(self, rho: ComplexMatrix | RealMatrix) -> None:
        if self.real_problem:
            rho_arr = np.asarray(rho, dtype=np.float64)
            rho_vec = rho_arr.reshape(self.rho_vec_dim, order="F")
        else:
            rho_arr = np.asarray(rho, dtype=np.complex128)
            rho_vec = rho_arr.reshape(self.rho_vec_dim, order="F")
        self.delta_vec.value = rho_vec - self.white_vec


class RingInflationSDP3Q:
    """
    Cached level-2 ring-inflation SDP for three qubit parties.

    This optimize-only version deliberately removes the approximate/probe paths.
    Every `solve` call solves the SDP:

        maximize t
        subject to the full level-2 ring-inflation constraints.

    Exact implementation optimizations retained:
      1. `domain="auto"` uses a real symmetric SDP only when rho is numerically real.
      2. Copy-swap symmetry is imposed by an exact +1/-1 block parametrization by default.
      3. Marginals and partial transposes are built through cached sparse vector maps.
      4. Redundant trace constraints are omitted because the fixed marginal equalities
         already enforce unit trace.

    Removed on purpose:
      - fast/probe mode
      - bisection/feasibility-only mode
      - project_real mode
      - tau_only/none PPT relaxations
    """

    dims: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    d: int = 64
    rho_dim: int = 8
    rho_vec_dim: int = 64

    def __init__(
        self,
        *,
        domain: Literal["auto", "real", "complex"] = "auto",
        real_tol: float = 1e-12,
        use_swap_blocks: bool = True,
        build_now: bool = False,
    ) -> None:
        if domain not in {"auto", "real", "complex"}:
            raise ValueError("domain must be 'auto', 'real', or 'complex'")
        self.domain = domain
        self.real_tol = float(real_tol)
        self.use_swap_blocks = bool(use_swap_blocks)
        self._opt_templates: dict[bool, _RingInflationTemplate3Q] = {}

        if build_now:
            if domain == "real":
                self._get_opt_template(True)
            elif domain == "complex":
                self._get_opt_template(False)

    @property
    def problem(self) -> cp.Problem | None:
        """Most recently built optimization problem, if any."""
        if self._opt_templates:
            return next(reversed(self._opt_templates.values())).problem
        return None

    def _choose_real_problem(self, rho: ComplexMatrix) -> bool:
        if self.domain == "real":
            max_imag = float(np.max(np.abs(np.imag(rho))))
            if max_imag > self.real_tol:
                raise ValueError(
                    f"domain='real' was requested, but rho has max imaginary part {max_imag:g}."
                )
            return True
        if self.domain == "complex":
            return False
        return bool(np.max(np.abs(np.imag(rho))) <= self.real_tol)

    def _get_opt_template(self, real_problem: bool) -> _RingInflationTemplate3Q:
        if real_problem not in self._opt_templates:
            self._opt_templates[real_problem] = _RingInflationTemplate3Q(
                real_problem=real_problem,
                use_swap_blocks=self.use_swap_blocks,
            )
        return self._opt_templates[real_problem]

    def _coerce_rho_for_domain(self, rho: ComplexMatrix, real_problem: bool) -> ComplexMatrix | RealMatrix:
        if real_problem:
            return np.asarray(np.real(rho), dtype=np.float64)
        return np.asarray(rho, dtype=np.complex128)

    def solve(
        self,
        rho: ComplexMatrix,
        *,
        verbose: bool = False,
        max_iters: int = 2_000,
        tol: float = 1e-3,
        warm_start: bool = True,
        gpu: bool = False,
        use_indirect: bool = False,
        acceleration_lookback: int = 0,
        normalize: bool = True,
        return_info: bool = False,
        strategy: Literal["optimize"] = "optimize",
        solver: Any = cp.SCS,
        **solver_options: Any,
    ) -> float | InflationResult:
        """
        Compute the level-2 ring-inflation score by solving the maximize-t SDP.

        `strategy` is retained only for notebook compatibility. The only valid
        value is "optimize".
        """
        if strategy != "optimize":
            raise ValueError("This module is optimize-only. Use strategy='optimize' or omit strategy.")

        rho = _validate_3q_density_matrix(rho)
        real_problem = self._choose_real_problem(rho)
        rho_domain = self._coerce_rho_for_domain(rho, real_problem)

        template = self._get_opt_template(real_problem)
        template.set_rho(rho_domain)

        solve_kwargs = _make_solve_kwargs(
            solver=solver,
            verbose=verbose,
            warm_start=warm_start,
            max_iters=max_iters,
            tol=tol,
            gpu=gpu,
            use_indirect=use_indirect,
            acceleration_lookback=acceleration_lookback,
            normalize=normalize,
            extra_options=solver_options,
        )
        template.problem.solve(**solve_kwargs)

        raw = None if template.problem.value is None else float(np.real(template.problem.value))
        score = None if raw is None else min(max(raw, 0.0), 1.0)

        if not return_info:
            return score

        stats = template.problem.solver_stats
        return InflationResult(
            score=score,
            raw_score=raw,
            status=template.problem.status,
            solve_time=getattr(stats, "solve_time", None),
            setup_time=getattr(stats, "setup_time", None),
            num_iters=getattr(stats, "num_iters", None),
            strategy="optimize",
            real_problem=real_problem,
        )


_TEMPLATE_3Q: RingInflationSDP3Q | None = None


def get_ring_inflation_sdp_3q(*, rebuild: bool = False, **kwargs: Any) -> RingInflationSDP3Q:
    """Return the cached 3-qubit SDP wrapper."""
    global _TEMPLATE_3Q
    if rebuild or _TEMPLATE_3Q is None:
        _TEMPLATE_3Q = RingInflationSDP3Q(**kwargs)
    return _TEMPLATE_3Q


def inflation_score(
    rho: ComplexMatrix,
    verbose: bool = False,
    max_iters: int = 2_000,
    tol: float = 1e-3,
    *,
    warm_start: bool = True,
    gpu: bool = False,
    use_indirect: bool = False,
    acceleration_lookback: int = 0,
    normalize: bool = True,
    return_info: bool = False,
    rebuild_problem: bool = False,
    strategy: Literal["optimize"] = "optimize",
    **solver_options: Any,
) -> float | InflationResult:
    """Level-2 ring-inflation score for a 3-qubit density matrix rho."""
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
        strategy=strategy,
        **solver_options,
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
    """Clear the cached CVXPY wrapper and sparse maps."""
    global _TEMPLATE_3Q
    _TEMPLATE_3Q = None
    partial_transpose_vec_map.cache_clear()
    ordered_marginal_vec_map.cache_clear()
    _basis_permutation_for_keep.cache_clear()
    _copy_swap_symmetry_index_groups.cache_clear()
    copy_swap_block_basis.cache_clear()
    _diag_and_upper_f_indices.cache_clear()
    _upper_f_indices_including_diag.cache_clear()

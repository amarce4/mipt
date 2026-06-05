from __future__ import annotations
from numpy.typing import NDArray
import numpy as np

from functools import lru_cache
from itertools import product
from math import prod
from typing import Sequence

import scipy.sparse as sp
import cvxpy as cp

ComplexVector = NDArray[np.complex128]
ComplexMatrix = NDArray[np.complex128]

@lru_cache(maxsize=None)
def partial_transpose_vec_map(
    dims: tuple[int, ...],
    subsys: tuple[int, ...],
) -> sp.csr_matrix:
    """
    Sparse permutation P such that:

        vec(X^T_subsys) = P @ vec(X)

    using column-major/F vectorization.
    """
    dims = tuple(int(x) for x in dims)
    subsys = tuple(sorted(set(int(x) for x in subsys)))

    n = len(dims)
    d = prod(dims)

    rows = []
    cols = []
    data = []

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

            # vec index, column-major:
            # vec(X)[I + J*d] = X[I, J]
            col = I + J * d
            row = I_pt + J_pt * d

            rows.append(row)
            cols.append(col)
            data.append(1.0)

    return sp.coo_matrix(
        (data, (rows, cols)),
        shape=(d * d, d * d),
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

def _basis_index(indices: Sequence[int], dims: Sequence[int]) -> int:
    """
    Row-major subsystem basis index:
        |i0 i1 ... in> -> (((i0*d1 + i1)*d2 + i2) ...)
    """
    out = 0
    for i, d in zip(indices, dims):
        out = out * d + i
    return out


@lru_cache(maxsize=None)
def ordered_marginal_vec_map(
    dims: tuple[int, ...],
    keep: tuple[int, ...],
) -> tuple[sp.csr_matrix, int]:
    """
    Sparse map L such that:

        vec( X_keep_ordered ) = L @ vec(X)

    where vec uses column-major/F order, matching cp.vec(X, order="F").

    dims:
        Full subsystem dimensions.

    keep:
        Subsystems to keep, in the desired output order.

    Returns:
        L, d_out
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
    trace_iter = product(*trace_ranges) if trace else [()]

    rows = []
    cols = []
    data = []

    for a_flat in range(d_out):
        a_multi = np.unravel_index(a_flat, out_dims, order="C")

        for b_flat in range(d_out):
            b_multi = np.unravel_index(b_flat, out_dims, order="C")

            # Column-major vectorization index for output matrix element:
            # vec(Y)[row + col*d_out] = Y[row, col]
            out_vec_idx = a_flat + b_flat * d_out

            # Need a fresh iterator each time.
            trace_iter = product(*trace_ranges) if trace else [()]

            for traced_vals in trace_iter:
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

                # Column-major vec index for X[I, J].
                in_vec_idx = I + J * d_in

                rows.append(out_vec_idx)
                cols.append(in_vec_idx)
                data.append(1.0)

    L = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(d_out * d_out, d_in * d_in),
    ).tocsr()

    return L, d_out


def marginal_vec_expr(
    X: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
) -> cp.Expression:
    """
    Vectorized ordered marginal:
        vec(X_keep) in column-major/F order.
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
    Use this only when you need a matrix, e.g. for PSD or partial transpose.
    """
    dims_t = tuple(int(x) for x in dims)
    keep_t = tuple(int(x) for x in keep)

    L, d_out = ordered_marginal_vec_map(dims_t, keep_t)
    v = L @ cp.vec(X, order="F")
    return cp.reshape(v, (d_out, d_out), order="F")

def inflation_score(rho, verbose=False, max_iters=10_000, tol=1e-4):
    rho = np.asarray(rho, dtype=np.complex128)

    dims = [2, 2, 2, 2, 2, 2]
    d = int(np.prod(dims))

    if rho.shape != (8, 8):
        raise ValueError(f"Expected rho shape (8, 8), got {rho.shape}")

    t = cp.Variable(name="t")
    tau = cp.Variable((d, d), hermitian=True, name="tau")
    gamma = cp.Variable((d, d), hermitian=True, name="gamma")

    I8 = np.eye(8, dtype=np.complex128)
    rho_vec = rho.reshape(8 * 8, order="F")
    I8_vec = I8.reshape(8 * 8, order="F")

    constraints: list[cp.Constraint] = []

    constraints += [
        t >= 0,
        t <= 1,          # Important safeguard, especially near rho = I/8.
        tau >> 0,
        gamma >> 0,
        cp.trace(tau) == 1,
        cp.trace(gamma) == 1,
    ]

    # White-noise mixture:
    # tau_012 = t*rho + (1-t)*I/8
    constraints.append(
        marginal_vec_expr(tau, dims, [0, 1, 2])
        ==
        t * rho_vec + (1 - t) * I8_vec / 8
    )

    # Marginal equality constraints.
    constraints += [
        # gamma_0134 = tau_0134
        marginal_vec_expr(gamma, dims, [0, 1, 3, 4])
        ==
        marginal_vec_expr(tau, dims, [0, 1, 3, 4]),

        # gamma_1245 = tau_1245
        marginal_vec_expr(gamma, dims, [1, 2, 4, 5])
        ==
        marginal_vec_expr(tau, dims, [1, 2, 4, 5]),

        # gamma_2350 = tau_2053
        marginal_vec_expr(gamma, dims, [2, 3, 5, 0])
        ==
        marginal_vec_expr(tau, dims, [2, 0, 5, 3]),
    ]

    # Copy-swap / permutation symmetry:
    # tau_345012 = tau_012345
    # gamma_345012 = gamma_012345
    constraints += [
        marginal_vec_expr(tau, dims, [3, 4, 5, 0, 1, 2])
        ==
        cp.vec(tau, order="F"),

        marginal_vec_expr(gamma, dims, [3, 4, 5, 0, 1, 2])
        ==
        cp.vec(gamma, order="F"),
    ]

    # PPT constraints.
    tau_T012 = partial_transpose_expr_sparse(tau, dims, [0, 1, 2])

    gamma_0124 = marginal_matrix_expr(gamma, dims, [0, 1, 2, 4])
    gamma_1235 = marginal_matrix_expr(gamma, dims, [1, 2, 3, 5])
    gamma_2340 = marginal_matrix_expr(gamma, dims, [2, 3, 4, 0])

    gamma_0124_T4 = partial_transpose_expr_sparse(gamma_0124, [2, 2, 2, 2], [3])
    gamma_1235_T5 = partial_transpose_expr_sparse(gamma_1235, [2, 2, 2, 2], [3])
    gamma_2340_T0 = partial_transpose_expr_sparse(gamma_2340, [2, 2, 2, 2], [3])

    constraints += [
        # Symmetrization helps CVXPY/MOSEK numerically.
        (tau_T012 + tau_T012.H) / 2 >> 0,
        (gamma_0124_T4 + gamma_0124_T4.H) / 2 >> 0,
        (gamma_1235_T5 + gamma_1235_T5.H) / 2 >> 0,
        (gamma_2340_T0 + gamma_2340_T0.H) / 2 >> 0,
    ]

    objective = cp.Maximize(cp.real(t))
    problem = cp.Problem(objective, constraints)

    if verbose:
        print("is_dcp:", problem.is_dcp())
        print("num constraints:", len(problem.constraints))

        print("Compiling problem data...")
        data, chain, inverse_data = problem.get_problem_data(cp.MOSEK)
        print("Compilation succeeded.")

        print("Solving...")

    problem.solve(
        solver=cp.SCS,
        verbose=verbose,
        warm_start=True,
        max_iters=max_iters,
        eps=tol
    )

    if verbose:
        print("Solve succeeded.")

    score = problem.value

    if (score >= 1):
        score = 1

    return score
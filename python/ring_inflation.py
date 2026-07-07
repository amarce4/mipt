"""
Level-2 ring-inflation robustness SDP for tripartite GNME.

Implements Eq. (12) of:
    Lyu, Lauand, Witczak-Krempa,
    "Network-Irreducible Multiparty Entanglement in Quantum Matter",
    arXiv:2512.11118v2 (2026).

The input is a state rho_ABC. For a six-qubit tripartite state partitioned
as A=(q0,q1), B=(q2,q3), C=(q4,q5), use party_dims=(4, 4, 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod, sqrt
from typing import Sequence

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


@dataclass
class RingInflationResult:
    """Summary of the SDP solution."""

    t_max: float | None
    white_noise_at_boundary: float | None
    certifies_gnme: bool | None
    status: str
    objective_value: float | None
    target_dimension: int
    inflated_dimension: int
    problem: cp.Problem


def _as_density_matrix(
    rho: np.ndarray,
    dimension: int,
    *,
    tol: float,
) -> np.ndarray:
    """Validate a statevector or density matrix and return a normalized rho."""
    arr = np.asarray(rho, dtype=np.complex128)

    if arr.ndim == 1:
        if arr.shape != (dimension,):
            raise ValueError(
                f"Statevector must have length {dimension}; got {arr.shape}."
            )
        norm = float(np.vdot(arr, arr).real)
        if norm <= tol:
            raise ValueError("Statevector has zero norm.")
        arr = arr / np.sqrt(norm)
        arr = np.outer(arr, arr.conj())
    elif arr.ndim == 2:
        if arr.shape != (dimension, dimension):
            raise ValueError(
                f"Density matrix must have shape {(dimension, dimension)}; "
                f"got {arr.shape}."
            )
    else:
        raise ValueError("rho must be a statevector or a square density matrix.")

    herm_err = np.linalg.norm(arr - arr.conj().T, ord="fro")
    if herm_err > tol:
        raise ValueError(f"rho is not Hermitian within tolerance: error={herm_err:g}.")
    arr = (arr + arr.conj().T) / 2.0

    trace = np.trace(arr)
    if abs(trace.imag) > tol or trace.real <= tol:
        raise ValueError(f"rho has invalid trace: {trace}.")
    if abs(trace.real - 1.0) > tol:
        raise ValueError(f"rho must have unit trace; got trace={trace.real:g}.")
    arr = arr / trace.real

    min_eig = float(np.linalg.eigvalsh(arr).min())
    if min_eig < -tol:
        raise ValueError(f"rho is not PSD within tolerance: min eigenvalue={min_eig:g}.")

    return arr


def _basis_permutation_matrix(dims: Sequence[int], order: Sequence[int]) -> sp.csr_matrix:
    """
    Return P such that P @ X @ P.T expresses X in the subsystem order
    [old_axis[order[0]], old_axis[order[1]], ...].
    """
    dims = tuple(int(d) for d in dims)
    order = tuple(int(i) for i in order)
    n = prod(dims)
    old_flat_for_new_flat = (
        np.arange(n).reshape(dims).transpose(order).reshape(-1)
    )
    return sp.csr_matrix(
        (np.ones(n), (np.arange(n), old_flat_for_new_flat)),
        shape=(n, n),
    )


def _marginal(
    expr: cp.Expression,
    dims: Sequence[int],
    keep: Sequence[int],
) -> tuple[cp.Expression, tuple[int, ...]]:
    """
    Partial trace leaving subsystems in exactly the order specified by keep.
    Subsystem labels refer to the original expr ordering.
    """
    current = expr
    current_dims = list(int(d) for d in dims)
    current_labels = list(range(len(current_dims)))
    keep = list(int(i) for i in keep)

    if len(set(keep)) != len(keep):
        raise ValueError("keep contains repeated subsystem indices.")

    for original_axis in reversed(range(len(dims))):
        if original_axis not in keep:
            axis = current_labels.index(original_axis)
            current = cp.partial_trace(current, tuple(current_dims), axis=axis)
            current_dims.pop(axis)
            current_labels.pop(axis)

    if current_labels != keep:
        order = [current_labels.index(label) for label in keep]
        permutation = _basis_permutation_matrix(current_dims, order)
        current = permutation @ current @ permutation.T
        current_dims = [current_dims[i] for i in order]

    return current, tuple(current_dims)


def _hermitian_part(expr: cp.Expression) -> cp.Expression:
    """Expose mathematically Hermitian transformed expressions to CVXPY."""
    return (expr + expr.H) / 2.0


def _swap_sector_isometries(block_dim: int) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Isometries Q_+, Q_- for the symmetric/antisymmetric sectors under
    swapping the two ABC copies.
    """
    d = int(block_dim)
    n = d * d
    n_plus = d * (d + 1) // 2
    n_minus = d * (d - 1) // 2

    plus_rows: list[int] = []
    plus_cols: list[int] = []
    plus_data: list[float] = []
    minus_rows: list[int] = []
    minus_cols: list[int] = []
    minus_data: list[float] = []

    plus_col = 0
    for i in range(d):
        plus_rows.append(i * d + i)
        plus_cols.append(plus_col)
        plus_data.append(1.0)
        plus_col += 1

    minus_col = 0
    s = 1.0 / sqrt(2.0)
    for i in range(d):
        for j in range(i + 1, d):
            ij = i * d + j
            ji = j * d + i

            plus_rows.extend([ij, ji])
            plus_cols.extend([plus_col, plus_col])
            plus_data.extend([s, s])
            plus_col += 1

            minus_rows.extend([ij, ji])
            minus_cols.extend([minus_col, minus_col])
            minus_data.extend([s, -s])
            minus_col += 1

    q_plus = sp.csc_matrix(
        (plus_data, (plus_rows, plus_cols)), shape=(n, n_plus)
    )
    q_minus = sp.csc_matrix(
        (minus_data, (minus_rows, minus_cols)), shape=(n, n_minus)
    )
    return q_plus, q_minus


def ring_inflation_tmax(
    rho: np.ndarray,
    party_dims: Sequence[int] = (4, 4, 4),
    *,
    symmetry_reduced: bool = True,
    real_only: bool | None = None,
    solver: str = cp.MOSEK,
    verbose: bool = False,
    mosek_params: dict | None = None,
    state_tolerance: float = 1e-9,
    gnme_tolerance: float = 1e-6,
) -> RingInflationResult:
    """
    Solve the level-2 ring-inflation robustness SDP for rho_ABC.

    Parameters
    ----------
    rho:
        Either a pure statevector of length D or a mixed-state D x D density
        matrix, with D = d_A*d_B*d_C.
    party_dims:
        Dimensions of A, B, C. For six physical qubits grouped as
        A=(q0,q1), B=(q2,q3), C=(q4,q5), use (4, 4, 4).
    symmetry_reduced:
        Use the swap-sector parametrization from the paper's numerical tip.
        This removes the explicit tau_21=tau and gamma_21=gamma constraints.
    real_only:
        If None, use real symmetric SDP variables when rho is real to numerical
        precision; otherwise use complex Hermitian variables. Set False to
        force the general complex formulation.
    solver:
        CVXPY solver identifier; defaults to cp.MOSEK.
    gnme_tolerance:
        Report GNME certification only when t_max < 1 - gnme_tolerance.

    Returns
    -------
    RingInflationResult
        t_max is the maximum target weight in
            t*rho + (1-t)*I/D
        that admits the level-2 inflation extension. If t_max < 1, the
        relaxation certifies GNME.
    """
    party_dims = tuple(int(d) for d in party_dims)
    if len(party_dims) != 3 or any(d <= 0 for d in party_dims):
        raise ValueError("party_dims must contain three positive dimensions: (dA,dB,dC).")

    d_a, d_b, d_c = party_dims
    dimension = prod(party_dims)
    rho = _as_density_matrix(rho, dimension, tol=state_tolerance)

    max_imag = float(np.max(np.abs(rho.imag)))
    if real_only is None:
        real_only = max_imag <= state_tolerance
    if real_only:
        if max_imag > state_tolerance:
            raise ValueError(
                f"real_only=True, but rho contains imaginary entries up to {max_imag:g}."
            )
        rho_model = rho.real
    else:
        rho_model = rho

    # Article ordering: A1, B1, C1, A2, B2, C2.
    inflated_dims = (d_a, d_b, d_c, d_a, d_b, d_c)
    inflated_dimension = dimension * dimension
    t = cp.Variable(name="t")

    # Page 12 explicitly interprets t as being in [0, 1].
    constraints: list[cp.Constraint] = [t >= 0, t <= 1]

    if symmetry_reduced:
        q_plus, q_minus = _swap_sector_isometries(dimension)
        n_plus = q_plus.shape[1]
        n_minus = q_minus.shape[1]
        if real_only:
            y_plus = cp.Variable((n_plus, n_plus), symmetric=True, name="tau_plus")
            y_minus = cp.Variable((n_minus, n_minus), symmetric=True, name="tau_minus")
            z_plus = cp.Variable((n_plus, n_plus), symmetric=True, name="gamma_plus")
            z_minus = cp.Variable((n_minus, n_minus), symmetric=True, name="gamma_minus")
        else:
            y_plus = cp.Variable((n_plus, n_plus), hermitian=True, name="tau_plus")
            y_minus = cp.Variable((n_minus, n_minus), hermitian=True, name="tau_minus")
            z_plus = cp.Variable((n_plus, n_plus), hermitian=True, name="gamma_plus")
            z_minus = cp.Variable((n_minus, n_minus), hermitian=True, name="gamma_minus")

        tau = q_plus @ y_plus @ q_plus.T + q_minus @ y_minus @ q_minus.T
        gamma = q_plus @ z_plus @ q_plus.T + q_minus @ z_minus @ q_minus.T

        # Avoid cp.real(...) around an expression that is already real.
        # In current CVXPY releases that construction can survive into
        # canonicalization and raise NotImplementedError for symmetric variables.
        trace_tau = cp.trace(y_plus) + cp.trace(y_minus)
        trace_gamma = cp.trace(z_plus) + cp.trace(z_minus)
        if not real_only:
            trace_tau = cp.real(trace_tau)
            trace_gamma = cp.real(trace_gamma)

        constraints += [
            y_plus >> 0,
            y_minus >> 0,
            z_plus >> 0,
            z_minus >> 0,
            trace_tau == 1,
            trace_gamma == 1,
        ]
    else:
        if real_only:
            tau = cp.Variable(
                (inflated_dimension, inflated_dimension), symmetric=True, name="tau"
            )
            gamma = cp.Variable(
                (inflated_dimension, inflated_dimension), symmetric=True, name="gamma"
            )
        else:
            tau = cp.Variable(
                (inflated_dimension, inflated_dimension), hermitian=True, name="tau"
            )
            gamma = cp.Variable(
                (inflated_dimension, inflated_dimension), hermitian=True, name="gamma"
            )

        copy_swap = _basis_permutation_matrix(inflated_dims, (3, 4, 5, 0, 1, 2))
        trace_tau = cp.trace(tau)
        trace_gamma = cp.trace(gamma)
        if not real_only:
            trace_tau = cp.real(trace_tau)
            trace_gamma = cp.real(trace_gamma)

        constraints += [
            tau >> 0,
            gamma >> 0,
            trace_tau == 1,
            trace_gamma == 1,
            tau == copy_swap @ tau @ copy_swap.T,
            gamma == copy_swap @ gamma @ copy_swap.T,
        ]

    # tau_(A1 B1 C1) = t*rho_ABC + (1-t)*I/D.
    tau_abc, _ = _marginal(tau, inflated_dims, (0, 1, 2))
    noisy_target = t * rho_model + (1 - t) * np.eye(dimension) / dimension
    constraints.append(tau_abc == noisy_target)

    # Matching four-party marginals in Eq. (12).
    gamma_ab, _ = _marginal(gamma, inflated_dims, (0, 1, 3, 4))
    tau_ab, _ = _marginal(tau, inflated_dims, (0, 1, 3, 4))
    gamma_bc, _ = _marginal(gamma, inflated_dims, (1, 2, 4, 5))
    tau_bc, _ = _marginal(tau, inflated_dims, (1, 2, 4, 5))
    gamma_ca, _ = _marginal(gamma, inflated_dims, (2, 3, 5, 0))
    tau_ca, _ = _marginal(tau, inflated_dims, (2, 0, 5, 3))
    constraints += [
        gamma_ab == tau_ab,
        gamma_bc == tau_bc,
        gamma_ca == tau_ca,
    ]

    # PPT relaxations in Eq. (12).
    # T1 is partial transpose over the full first ABC block of tau.
    tau_t1 = cp.partial_transpose(tau, (dimension, dimension), axis=0)

    gamma_abcb2, dims_abcb2 = _marginal(gamma, inflated_dims, (0, 1, 2, 4))
    gamma_bca2c2, dims_bca2c2 = _marginal(gamma, inflated_dims, (1, 2, 3, 5))
    # Note: this PPT marginal is C1 A2 B2 A1; it contains B2, not C2.
    gamma_ca2b2a1, dims_ca2b2a1 = _marginal(gamma, inflated_dims, (2, 3, 4, 0))

    constraints += [
        _hermitian_part(tau_t1) >> 0,
        _hermitian_part(cp.partial_transpose(gamma_abcb2, dims_abcb2, axis=3)) >> 0,
        _hermitian_part(cp.partial_transpose(gamma_bca2c2, dims_bca2c2, axis=3)) >> 0,
        _hermitian_part(cp.partial_transpose(gamma_ca2b2a1, dims_ca2b2a1, axis=3)) >> 0,
    ]

    problem = cp.Problem(cp.Maximize(t), constraints)
    solve_kwargs: dict = {"solver": solver, "verbose": verbose}
    if mosek_params is not None:
        solve_kwargs["mosek_params"] = mosek_params
    problem.solve(**solve_kwargs)

    solved = problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and t.value is not None
    t_max = float(np.real(t.value)) if solved else None
    boundary_noise = (1.0 - t_max) if solved else None
    certifies = (t_max < 1.0 - gnme_tolerance) if solved else None
    objective = float(problem.value) if solved and problem.value is not None else None

    return RingInflationResult(
        t_max=t_max,
        white_noise_at_boundary=boundary_noise,
        certifies_gnme=certifies,
        status=problem.status,
        objective_value=objective,
        target_dimension=dimension,
        inflated_dimension=inflated_dimension,
        problem=problem,
    )

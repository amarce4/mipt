
"""
gilbert_uqn.py

A practical Gilbert / Frank-Wolfe implementation for estimating Hilbert-Schmidt
distance to a convex hull of pure-state atoms, with Step 2 implemented as a
support oracle over user-supplied pure-state families.

The intended use case is a UQN/network-state convex hull:

    C = conv{ |psi(theta, family)><psi(theta, family)| }

where each family.make_state(theta) returns one allowed pure network state.

Important sign convention
-------------------------
For minimizing

    || rho - rho1 ||_F^2,

the standard Gilbert/Frank-Wolfe atom search chooses an atom rho2 maximizing

    Tr((rho - rho1) rho2) = <psi|rho-rho1|psi>.

Equivalently, it minimizes <psi|rho1-rho|psi>.  This module uses that
"descent" convention by default.  A "paper_literal" mode is included for
checking the quoted wording from arXiv:2512.11118, but the actual accepted
candidate is still tested by the true Hilbert-Schmidt descent gain.

Dependencies
------------
Required:
    numpy

Optional but strongly recommended:
    scipy

If scipy is unavailable, random preselection still works, but local refinement
and memory QP are disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Optional, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
RealVector = NDArray[np.float64]


OracleMode = Literal["descent", "paper_literal"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UQNFamily:
    """
    A parameterized pure-state atom family.

    Parameters
    ----------
    name:
        Descriptive name, e.g. "B2_family_0" or "AB|C".
    n_params:
        Number of real parameters expected by make_state.
    make_state:
        Callable theta -> psi, where psi is a normalized or normalizable
        complex state vector.
    theta_sampler:
        Optional custom sampler. If omitted, theta is sampled uniformly from
        [0, 2*pi)^n_params.
    batch_make_states:
        Optional vectorized callable accepting an array of shape
        (batch, n_params) and returning states of shape (batch, dim).
        When supplied, Step-2 random preselection is much faster.
    """

    name: str
    n_params: int
    make_state: Callable[[RealVector], ComplexVector]
    theta_sampler: Optional[Callable[[np.random.Generator], RealVector]] = None
    batch_make_states: Optional[Callable[[NDArray[np.float64]], NDArray[np.complex128]]] = None

    def sample_theta(self, rng: np.random.Generator) -> RealVector:
        if self.theta_sampler is not None:
            theta = self.theta_sampler(rng)
            return np.asarray(theta, dtype=np.float64)
        return rng.uniform(0.0, 2.0 * np.pi, size=self.n_params).astype(np.float64)

    def sample_theta_batch(self, rng: np.random.Generator, count: int) -> NDArray[np.float64]:
        if count <= 0:
            raise ValueError("count must be positive.")
        if self.theta_sampler is None:
            return rng.uniform(0.0, 2.0 * np.pi, size=(count, self.n_params)).astype(np.float64)
        return np.vstack([self.sample_theta(rng) for _ in range(count)]).astype(np.float64)


@dataclass
class AtomResult:
    """
    Result returned by the Step-2 support oracle.
    """

    family_name: str
    theta: RealVector
    psi: ComplexVector
    rho2: ComplexMatrix
    support_value: float
    descent_gain: float
    success: bool
    mode: OracleMode
    refined: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryQPResult:
    """
    Result of the small convex QP over remembered atoms.
    """

    weights: RealVector
    sigma: ComplexMatrix
    distance: float
    success: bool
    objective: float
    message: str = ""


@dataclass
class GilbertResult:
    """
    Full output of the Gilbert algorithm.
    """

    sigma: ComplexMatrix
    distance: float
    distances: list[float]
    atoms: list[AtomResult]
    weights: Optional[RealVector]
    converged: bool
    stop_reason: str


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

def normalize_ket(psi: ComplexVector, *, zero_tol: float = 1e-15) -> ComplexVector:
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm <= zero_tol:
        raise ValueError("Cannot normalize a zero or near-zero state vector.")
    return psi / norm


def ket_to_dm(psi: ComplexVector) -> ComplexMatrix:
    psi = normalize_ket(psi)
    return np.outer(psi, psi.conj()).astype(np.complex128)


def hermitize(X: ComplexMatrix) -> ComplexMatrix:
    X = np.asarray(X, dtype=np.complex128)
    return 0.5 * (X + X.conj().T)


def trace_real(X: ComplexMatrix) -> float:
    return float(np.real(np.trace(X)))


def hs_inner(A: ComplexMatrix, B: ComplexMatrix) -> float:
    """
    Hilbert-Schmidt inner product Re Tr(A^dagger B).
    For Hermitian A,B this is Re Tr(A B).
    """
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    return float(np.real(np.trace(A.conj().T @ B)))


def hs_norm(A: ComplexMatrix) -> float:
    return float(np.sqrt(max(hs_inner(A, A), 0.0)))


def real_expectation(psi: ComplexVector, H: ComplexMatrix) -> float:
    psi = normalize_ket(psi)
    H = np.asarray(H, dtype=np.complex128)
    return float(np.real(np.vdot(psi, H @ psi)))


def validate_square_matrix(X: ComplexMatrix, *, name: str = "matrix") -> ComplexMatrix:
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {X.shape}.")
    return X


def validate_same_shape(rho: ComplexMatrix, sigma: ComplexMatrix) -> tuple[ComplexMatrix, ComplexMatrix]:
    rho = validate_square_matrix(rho, name="rho")
    sigma = validate_square_matrix(sigma, name="sigma")
    if rho.shape != sigma.shape:
        raise ValueError(f"rho and sigma must have the same shape, got {rho.shape} and {sigma.shape}.")
    return rho, sigma


def project_density_matrix(
    rho: ComplexMatrix,
    *,
    normalize_trace: bool = True,
    hermitize_input: bool = True,
) -> ComplexMatrix:
    """
    Lightweight cleanup: Hermitize and optionally normalize trace.
    This does not project onto the PSD cone.
    """
    rho = validate_square_matrix(rho, name="rho")
    if hermitize_input:
        rho = hermitize(rho)
    if normalize_trace:
        tr = np.trace(rho)
        if abs(tr) <= 1e-15:
            raise ValueError("Cannot normalize density matrix with zero trace.")
        rho = rho / tr
    return rho.astype(np.complex128)


# ---------------------------------------------------------------------------
# Gilbert geometry
# ---------------------------------------------------------------------------

def gilbert_descent_gain(
    rho: ComplexMatrix,
    rho1: ComplexMatrix,
    rho2: ComplexMatrix,
) -> float:
    """
    Positive iff moving from rho1 toward rho2 decreases ||rho-rho1||_F^2
    to first order.

    Directional derivative of f(sigma)=||rho-sigma||_F^2 at rho1 along
    d = rho2-rho1 is

        -2 Tr((rho-rho1)(rho2-rho1)).

    So descent requires this returned quantity to be positive.
    """
    rho, rho1 = validate_same_shape(rho, rho1)
    rho2 = validate_square_matrix(rho2, name="rho2")
    if rho2.shape != rho.shape:
        raise ValueError(f"rho2 shape mismatch: got {rho2.shape}, expected {rho.shape}.")
    return hs_inner(rho - rho1, rho2 - rho1)


def gilbert_line_search(
    rho: ComplexMatrix,
    rho1: ComplexMatrix,
    rho2: ComplexMatrix,
) -> tuple[float, ComplexMatrix, float]:
    """
    Exact Step-3 line search:

        minimize_p ||rho - [p*rho1 + (1-p)*rho2]||_F^2, p in [0,1].

    Returns
    -------
    p:
        Weight on rho1.
    rho_new:
        p*rho1 + (1-p)*rho2.
    distance:
        Hilbert-Schmidt distance ||rho-rho_new||_F.
    """
    rho, rho1 = validate_same_shape(rho, rho1)
    rho2 = validate_square_matrix(rho2, name="rho2")
    if rho2.shape != rho.shape:
        raise ValueError(f"rho2 shape mismatch: got {rho2.shape}, expected {rho.shape}.")

    u = rho1 - rho2
    b = rho - rho2
    denom = hs_inner(u, u)

    if denom <= 1e-15:
        p = 1.0
    else:
        p = hs_inner(b, u) / denom
        p = float(np.clip(p, 0.0, 1.0))

    rho_new = p * rho1 + (1.0 - p) * rho2
    distance = hs_norm(rho - rho_new)
    return p, rho_new.astype(np.complex128), distance


def full_state_eigen_oracle(
    rho: ComplexMatrix,
    rho1: ComplexMatrix,
    *,
    mode: OracleMode = "descent",
) -> AtomResult:
    """
    Debug oracle over all pure states, not UQN-restricted.

    For all pure states, maximizing <psi|H|psi> is simply the top eigenvector
    of H. This is useful for testing sign conventions and the Gilbert loop.
    """
    rho, rho1 = validate_same_shape(rho, rho1)

    if mode == "descent":
        H = hermitize(rho - rho1)
    elif mode == "paper_literal":
        H = hermitize(rho1 - rho)
    else:
        raise ValueError("mode must be 'descent' or 'paper_literal'.")

    evals, evecs = np.linalg.eigh(H)
    idx = int(np.argmax(evals))
    psi = normalize_ket(evecs[:, idx])
    rho2 = ket_to_dm(psi)
    gain = gilbert_descent_gain(rho, rho1, rho2)

    return AtomResult(
        family_name="all_pure_states_eigen_oracle",
        theta=np.array([], dtype=np.float64),
        psi=psi,
        rho2=rho2,
        support_value=float(evals[idx]),
        descent_gain=gain,
        success=gain > 0.0,
        mode=mode,
        refined=False,
        metadata={"eigenvalue": float(evals[idx])},
    )


# ---------------------------------------------------------------------------
# Step-2 support oracle over parameterized UQN families
# ---------------------------------------------------------------------------

def _support_hamiltonian(rho: ComplexMatrix, rho1: ComplexMatrix, mode: OracleMode) -> ComplexMatrix:
    if mode == "descent":
        return hermitize(rho - rho1)
    if mode == "paper_literal":
        return hermitize(rho1 - rho)
    raise ValueError("mode must be 'descent' or 'paper_literal'.")


def _evaluate_family_theta(fam: UQNFamily, theta: RealVector, H: ComplexMatrix) -> tuple[float, ComplexVector]:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (fam.n_params,):
        raise ValueError(
            f"Family {fam.name!r} expected theta shape {(fam.n_params,)}, got {theta.shape}."
        )
    psi = normalize_ket(fam.make_state(theta))
    value = real_expectation(psi, H)
    return value, psi


def _batch_expectations(states: NDArray[np.complex128], H: ComplexMatrix) -> NDArray[np.float64]:
    """
    Vectorized values Re <psi|H|psi> for states with shape (batch, dim).
    """
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim != 2:
        raise ValueError(f"states must have shape (batch, dim), got {states.shape}.")
    H = np.asarray(H, dtype=np.complex128)
    if H.shape != (states.shape[1], states.shape[1]):
        raise ValueError(f"H shape {H.shape} is incompatible with states shape {states.shape}.")
    norms = np.linalg.norm(states, axis=1)
    if np.any(norms <= 1e-15):
        raise ValueError("At least one batch state has zero or near-zero norm.")
    states = states / norms[:, None]
    return np.einsum("bi,ij,bj->b", states.conj(), H, states, optimize=True).real.astype(np.float64)


def random_preselection(
    H: ComplexMatrix,
    families: Sequence[UQNFamily],
    *,
    rng: np.random.Generator,
    n_random_per_family: int = 2048,
    keep: int = 16,
) -> list[tuple[float, UQNFamily, RealVector]]:
    """
    Random preselection for Step 2.

    Uses family.batch_make_states when available, otherwise falls back to the
    scalar make_state loop.

    Returns a descending list of (support_value, family, theta) candidates.
    """
    if keep <= 0:
        raise ValueError("keep must be positive.")
    if n_random_per_family <= 0:
        raise ValueError("n_random_per_family must be positive.")
    if len(families) == 0:
        raise ValueError("At least one UQNFamily is required.")

    candidates: list[tuple[float, UQNFamily, RealVector]] = []

    for fam in families:
        thetas = fam.sample_theta_batch(rng, n_random_per_family)

        if fam.batch_make_states is not None:
            states = fam.batch_make_states(thetas)
            values = _batch_expectations(states, H)
            local_keep = min(keep, n_random_per_family)
            # argpartition avoids sorting the full batch.
            idx = np.argpartition(values, -local_keep)[-local_keep:]
            idx = idx[np.argsort(values[idx])[::-1]]
            candidates.extend((float(values[i]), fam, thetas[i].copy()) for i in idx)
        else:
            for theta in thetas:
                val, _ = _evaluate_family_theta(fam, theta, H)
                candidates.append((val, fam, theta.copy()))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:keep]


def refine_candidate(
    H: ComplexMatrix,
    fam: UQNFamily,
    theta0: RealVector,
    *,
    maxiter: int = 250,
    method: str = "BFGS",
) -> tuple[float, RealVector, ComplexVector, bool, str]:
    """
    Local refinement of one preselected candidate.

    Uses scipy.optimize.minimize on -<psi(theta)|H|psi(theta)>.
    """
    theta0 = np.asarray(theta0, dtype=np.float64)

    if minimize is None:
        val, psi = _evaluate_family_theta(fam, theta0, H)
        return val, theta0, psi, False, "scipy unavailable; returned unrefined candidate"

    def objective(theta: RealVector) -> float:
        val, _ = _evaluate_family_theta(fam, np.asarray(theta, dtype=np.float64), H)
        return -val

    method_upper = method.upper()
    if method_upper in {"BFGS", "CG", "L-BFGS-B"}:
        options = {"maxiter": int(maxiter), "gtol": 1e-8}
    elif method_upper == "POWELL":
        options = {"maxiter": int(maxiter), "xtol": 1e-6, "ftol": 1e-8}
    elif method_upper == "NELDER-MEAD":
        options = {"maxiter": int(maxiter), "xatol": 1e-6, "fatol": 1e-8}
    else:
        options = {"maxiter": int(maxiter)}

    result = minimize(
        objective,
        theta0,
        method=method,
        options=options,
    )

    theta = np.asarray(result.x, dtype=np.float64)
    val, psi = _evaluate_family_theta(fam, theta, H)
    return val, theta, psi, bool(result.success), str(result.message)


def step2_support_oracle(
    rho: ComplexMatrix,
    rho1: ComplexMatrix,
    families: Sequence[UQNFamily],
    *,
    rng: Optional[np.random.Generator] = None,
    n_random_per_family: int = 2048,
    n_refine: int = 16,
    maxiter: int = 250,
    mode: OracleMode = "descent",
    require_positive_descent: bool = True,
    refine: bool = True,
    local_method: str = "BFGS",
) -> AtomResult:
    """
    Step 2 support oracle.

    Parameters
    ----------
    rho:
        Target state.
    rho1:
        Current convex approximation.
    families:
        Allowed pure-state atom families.
    rng:
        Random generator.
    n_random_per_family:
        Number of random samples drawn per family.
    n_refine:
        Number of best random candidates to refine locally.
    maxiter:
        Max local optimizer iterations.
    mode:
        "descent":
            Maximize <psi|rho-rho1|psi>. This is the standard Gilbert
            direction.
        "paper_literal":
            Maximize <psi|rho1-rho|psi> exactly as in the quoted step.
            The returned candidate is still checked by true descent_gain.
    require_positive_descent:
        If True, success means descent_gain > 0. If False, the best support
        candidate is returned even if it is not a descent direction.
    refine:
        Whether to run local refinement after random preselection.
    local_method:
        scipy.optimize.minimize method, e.g. "BFGS", "L-BFGS-B",
        "Nelder-Mead".

    Returns
    -------
    AtomResult
        rho2 = |psi><psi| and diagnostic information.
    """
    rho, rho1 = validate_same_shape(rho, rho1)
    rng = np.random.default_rng() if rng is None else rng

    H = _support_hamiltonian(rho, rho1, mode)
    preselected = random_preselection(
        H,
        families,
        rng=rng,
        n_random_per_family=n_random_per_family,
        keep=max(1, n_refine),
    )

    best_val = -np.inf
    best_fam: Optional[UQNFamily] = None
    best_theta: Optional[RealVector] = None
    best_psi: Optional[ComplexVector] = None
    best_refined_success = False
    best_message = ""

    for initial_val, fam, theta0 in preselected:
        if refine:
            val, theta, psi, refined_success, message = refine_candidate(
                H,
                fam,
                theta0,
                maxiter=maxiter,
                method=local_method,
            )
        else:
            val, psi = _evaluate_family_theta(fam, theta0, H)
            theta = theta0
            refined_success = False
            message = "local refinement disabled"

        # Keep local optimizer only if it did not degrade badly due to parameterization issues.
        # In normal use val should be >= initial_val up to numerical tolerance.
        if val + 1e-12 < initial_val:
            val = initial_val
            theta = theta0
            _, psi = _evaluate_family_theta(fam, theta0, H)
            refined_success = False
            message = "refinement degraded candidate; reverted to preselected theta"

        if val > best_val:
            best_val = float(val)
            best_fam = fam
            best_theta = theta
            best_psi = psi
            best_refined_success = bool(refined_success)
            best_message = message

    if best_fam is None or best_theta is None or best_psi is None:
        raise RuntimeError("Step-2 oracle failed to produce any candidate.")

    rho2 = ket_to_dm(best_psi)
    gain = gilbert_descent_gain(rho, rho1, rho2)

    if require_positive_descent:
        success = gain > 0.0
    else:
        success = True

    return AtomResult(
        family_name=best_fam.name,
        theta=best_theta,
        psi=best_psi,
        rho2=rho2,
        support_value=best_val,
        descent_gain=gain,
        success=success,
        mode=mode,
        refined=refine and minimize is not None,
        metadata={
            "local_optimizer_success": best_refined_success,
            "local_optimizer_message": best_message,
            "n_random_per_family": int(n_random_per_family),
            "n_refine": int(n_refine),
            "local_method": local_method,
        },
    )


# ---------------------------------------------------------------------------
# Gilbert with optional memory
# ---------------------------------------------------------------------------

def solve_memory_qp(
    rho: ComplexMatrix,
    atoms: Sequence[ComplexMatrix],
    *,
    initial_weights: Optional[RealVector] = None,
    maxiter: int = 1000,
) -> MemoryQPResult:
    """
    Solve the small convex QP

        minimize_w ||rho - sum_i w_i atoms[i]||_F^2
        subject to w_i >= 0, sum_i w_i = 1.

    This is the "memory" version of the Gilbert update. It is a QP over a
    simplex, not an SDP.

    Parameters
    ----------
    rho:
        Target density matrix.
    atoms:
        Sequence of atom density matrices.
    initial_weights:
        Optional feasible starting point.

    Returns
    -------
    MemoryQPResult
    """
    rho = validate_square_matrix(rho, name="rho")
    if len(atoms) == 0:
        raise ValueError("At least one atom is required for memory QP.")

    atoms_arr = [validate_square_matrix(A, name=f"atoms[{i}]") for i, A in enumerate(atoms)]
    for i, A in enumerate(atoms_arr):
        if A.shape != rho.shape:
            raise ValueError(f"atoms[{i}] shape mismatch: got {A.shape}, expected {rho.shape}.")

    m = len(atoms_arr)

    if m == 1:
        sigma = atoms_arr[0].copy()
        dist = hs_norm(rho - sigma)
        return MemoryQPResult(
            weights=np.ones(1, dtype=np.float64),
            sigma=sigma,
            distance=dist,
            success=True,
            objective=dist * dist,
            message="single atom",
        )

    # Build Gram matrix G_ij = <A_i, A_j> and vector c_i = <rho, A_i>.
    G = np.empty((m, m), dtype=np.float64)
    c = np.empty(m, dtype=np.float64)

    for i in range(m):
        c[i] = hs_inner(rho, atoms_arr[i])
        for j in range(i, m):
            gij = hs_inner(atoms_arr[i], atoms_arr[j])
            G[i, j] = gij
            G[j, i] = gij

    rho_norm2 = hs_inner(rho, rho)

    def objective(w: RealVector) -> float:
        w = np.asarray(w, dtype=np.float64)
        return float(w @ G @ w - 2.0 * c @ w + rho_norm2)

    def gradient(w: RealVector) -> RealVector:
        w = np.asarray(w, dtype=np.float64)
        return (2.0 * G @ w - 2.0 * c).astype(np.float64)

    if initial_weights is None:
        w0 = np.full(m, 1.0 / m, dtype=np.float64)
    else:
        w0 = np.asarray(initial_weights, dtype=np.float64)
        if w0.shape != (m,):
            raise ValueError(f"initial_weights must have shape {(m,)}, got {w0.shape}.")
        w0 = np.maximum(w0, 0.0)
        s = float(np.sum(w0))
        if s <= 1e-15:
            w0 = np.full(m, 1.0 / m, dtype=np.float64)
        else:
            w0 = w0 / s

    if minimize is None:
        # Fallback: return current weighted average.
        sigma = sum(float(w0[i]) * atoms_arr[i] for i in range(m))
        dist = hs_norm(rho - sigma)
        return MemoryQPResult(
            weights=w0,
            sigma=sigma.astype(np.complex128),
            distance=dist,
            success=False,
            objective=dist * dist,
            message="scipy unavailable; returned initial feasible average",
        )

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                    "jac": lambda w: np.ones_like(w, dtype=np.float64)}]
    bounds = [(0.0, 1.0)] * m

    result = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(maxiter), "ftol": 1e-12},
    )

    w = np.asarray(result.x, dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    if s <= 1e-15:
        w = np.full(m, 1.0 / m, dtype=np.float64)
    else:
        w = w / s

    sigma = sum(float(w[i]) * atoms_arr[i] for i in range(m))
    sigma = np.asarray(sigma, dtype=np.complex128)
    dist = hs_norm(rho - sigma)
    obj = dist * dist

    return MemoryQPResult(
        weights=w,
        sigma=sigma,
        distance=dist,
        success=bool(result.success),
        objective=obj,
        message=str(result.message),
    )


def gilbert_uqn(
    rho: ComplexMatrix,
    families: Sequence[UQNFamily],
    *,
    rho1_init: Optional[ComplexMatrix] = None,
    rng: Optional[np.random.Generator] = None,
    max_iter: int = 200,
    tol_abs: float = 1e-8,
    tol_rel: float = 1e-8,
    n_random_per_family: int = 2048,
    n_refine: int = 16,
    local_maxiter: int = 250,
    mode: OracleMode = "descent",
    use_memory: bool = True,
    memory_size: int = 50,
    local_method: str = "BFGS",
    refine: bool = True,
    verbose: bool = False,
) -> GilbertResult:
    """
    Estimate distance from rho to conv{UQN atoms} by Gilbert iteration.

    Parameters
    ----------
    rho:
        Target density matrix.
    families:
        Allowed pure atom families.
    rho1_init:
        Optional initial convex approximation. If omitted, the first atom is
        chosen by applying the Step-2 oracle against maximally mixed rho1.
    rng:
        NumPy random generator.
    max_iter:
        Maximum Gilbert iterations.
    tol_abs:
        Absolute distance improvement tolerance.
    tol_rel:
        Relative distance improvement tolerance.
    n_random_per_family:
        Random samples per family per Step-2 call.
    n_refine:
        Number of preselected candidates refined per Step-2 call.
    local_maxiter:
        Local optimizer iterations inside Step 2.
    mode:
        "descent" is recommended.
    use_memory:
        If True, solve the convex simplex QP over remembered atoms after each
        new atom is found.
    memory_size:
        Maximum number of atom density matrices retained in memory.
    local_method:
        scipy optimizer method used for Step-2 local refinement.
    refine:
        Whether to run local optimizer refinement after random preselection.
    verbose:
        Print per-iteration diagnostics.

    Returns
    -------
    GilbertResult
    """
    rho = project_density_matrix(rho, normalize_trace=True, hermitize_input=True)
    dim = rho.shape[0]
    rng = np.random.default_rng() if rng is None else rng

    atoms_results: list[AtomResult] = []
    atom_dms: list[ComplexMatrix] = []
    weights: Optional[RealVector] = None

    if rho1_init is None:
        # A neutral starting point for the first oracle call. The returned atom
        # becomes the first convex approximation.
        rho_mm = np.eye(dim, dtype=np.complex128) / dim
        first = step2_support_oracle(
            rho,
            rho_mm,
            families,
            rng=rng,
            n_random_per_family=n_random_per_family,
            n_refine=n_refine,
            maxiter=local_maxiter,
            mode="descent",
            require_positive_descent=False,
            refine=refine,
            local_method=local_method,
        )
        rho1 = first.rho2.copy()
        atoms_results.append(first)
        atom_dms.append(first.rho2.copy())
        weights = np.ones(1, dtype=np.float64)
    else:
        rho1 = project_density_matrix(rho1_init, normalize_trace=True, hermitize_input=True)
        if rho1.shape != rho.shape:
            raise ValueError(f"rho1_init shape mismatch: got {rho1.shape}, expected {rho.shape}.")

    distance = hs_norm(rho - rho1)
    distances = [distance]

    if verbose:
        print(f"init distance={distance:.12g}")

    converged = False
    stop_reason = "max_iter reached"

    for it in range(max_iter):
        prev_distance = distance

        atom = step2_support_oracle(
            rho,
            rho1,
            families,
            rng=rng,
            n_random_per_family=n_random_per_family,
            n_refine=n_refine,
            maxiter=local_maxiter,
            mode=mode,
            require_positive_descent=True,
            refine=refine,
            local_method=local_method,
        )

        if not atom.success:
            stop_reason = (
                "no positive descent atom found; increase n_random_per_family, "
                "n_refine, or improve the UQN ansatz/local optimizer"
            )
            if verbose:
                print(
                    f"iter={it:04d} stop: no descent atom; "
                    f"support={atom.support_value:.6g} gain={atom.descent_gain:.6g}"
                )
            break

        atoms_results.append(atom)
        atom_dms.append(atom.rho2.copy())

        # Keep only the newest memory_size atoms. The current rho1 itself is
        # retained implicitly by the convex combination from previous memory.
        if use_memory:
            if len(atom_dms) > memory_size:
                atom_dms = atom_dms[-memory_size:]
                # AtomResult list is kept as full history; weights only correspond
                # to current atom_dms.

            qp = solve_memory_qp(rho, atom_dms, maxiter=1000)
            rho_next = qp.sigma
            distance = qp.distance
            weights = qp.weights
        else:
            _, rho_next, distance = gilbert_line_search(rho, rho1, atom.rho2)
            weights = None

        rho1 = rho_next
        distances.append(distance)

        abs_improvement = prev_distance - distance
        rel_improvement = abs_improvement / max(prev_distance, 1e-15)

        if verbose:
            print(
                f"iter={it:04d} dist={distance:.12g} "
                f"abs_imp={abs_improvement:.3g} rel_imp={rel_improvement:.3g} "
                f"support={atom.support_value:.6g} gain={atom.descent_gain:.6g} "
                f"family={atom.family_name}"
            )

        if abs_improvement < -1e-10:
            warnings.warn(
                f"Distance increased by {-abs_improvement:.3e}. "
                "This may indicate a bad oracle candidate or numerical issues.",
                RuntimeWarning,
            )

        if abs_improvement >= 0.0 and (abs_improvement <= tol_abs or rel_improvement <= tol_rel):
            converged = True
            stop_reason = "distance improvement below tolerance"
            break

    return GilbertResult(
        sigma=rho1,
        distance=distance,
        distances=distances,
        atoms=atoms_results,
        weights=weights,
        converged=converged,
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# Example atom families
# ---------------------------------------------------------------------------

def single_qubit_state(theta: float, phi: float) -> ComplexVector:
    """
    Bloch-sphere parameterization:

        cos(theta/2)|0> + exp(i phi) sin(theta/2)|1>.
    """
    return np.array(
        [
            np.cos(theta / 2.0),
            np.exp(1j * phi) * np.sin(theta / 2.0),
        ],
        dtype=np.complex128,
    )


def product_3q_state(params: RealVector) -> ComplexVector:
    """
    Fully product 3-qubit pure state in A,B,C order.

    params = [theta_a, phi_a, theta_b, phi_b, theta_c, phi_c]
    """
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (6,):
        raise ValueError(f"product_3q_state expected 6 params, got {params.shape}.")

    theta_a, phi_a, theta_b, phi_b, theta_c, phi_c = params
    a = single_qubit_state(theta_a, phi_a)
    b = single_qubit_state(theta_b, phi_b)
    c = single_qubit_state(theta_c, phi_c)
    return normalize_ket(np.kron(np.kron(a, b), c))


def normalized_complex_vector_from_reals(params: RealVector, dim: int) -> ComplexVector:
    """
    Convert 2*dim real parameters into a normalized complex vector.

    This is deliberately redundant due to global phase and norm, but it is
    convenient and robust for local search.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (2 * dim,):
        raise ValueError(f"Expected {2 * dim} real params for dim={dim}, got {params.shape}.")

    z = params[:dim] + 1j * params[dim:]
    norm = np.linalg.norm(z)
    if norm <= 1e-15:
        z = np.zeros(dim, dtype=np.complex128)
        z[0] = 1.0
        return z
    return z / norm


def random_normal_theta_sampler(n: int, scale: float = 1.0) -> Callable[[np.random.Generator], RealVector]:
    """
    Useful for unconstrained real-vector parameterizations, such as
    normalized_complex_vector_from_reals.
    """
    def sampler(rng: np.random.Generator) -> RealVector:
        return rng.normal(loc=0.0, scale=scale, size=n).astype(np.float64)
    return sampler


def biseparable_ab_c_state(params: RealVector) -> ComplexVector:
    """
    Biseparable 3-qubit atom |phi_AB> tensor |chi_C>, in A,B,C order.

    params[0:8]   -> arbitrary 2-qubit AB state, 4 complex amplitudes
    params[8:12]  -> arbitrary 1-qubit C state, 2 complex amplitudes
    """
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (12,):
        raise ValueError(f"biseparable_ab_c_state expected 12 params, got {params.shape}.")

    ab = normalized_complex_vector_from_reals(params[:8], dim=4)
    c = normalized_complex_vector_from_reals(params[8:12], dim=2)
    return normalize_ket(np.kron(ab, c))


def biseparable_ac_b_state(params: RealVector) -> ComplexVector:
    """
    Biseparable 3-qubit atom |phi_AC> tensor |chi_B>, returned in A,B,C order.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (12,):
        raise ValueError(f"biseparable_ac_b_state expected 12 params, got {params.shape}.")

    ac = normalized_complex_vector_from_reals(params[:8], dim=4)
    b = normalized_complex_vector_from_reals(params[8:12], dim=2)

    # kron(ac,b) has tensor axes A,C,B. Convert to A,B,C.
    psi_acb = np.kron(ac, b).reshape(2, 2, 2)
    psi_abc = np.transpose(psi_acb, (0, 2, 1)).reshape(8)
    return normalize_ket(psi_abc)


def biseparable_bc_a_state(params: RealVector) -> ComplexVector:
    """
    Biseparable 3-qubit atom |chi_A> tensor |phi_BC>, returned in A,B,C order.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (12,):
        raise ValueError(f"biseparable_bc_a_state expected 12 params, got {params.shape}.")

    bc = normalized_complex_vector_from_reals(params[:8], dim=4)
    a = normalized_complex_vector_from_reals(params[8:12], dim=2)
    return normalize_ket(np.kron(a, bc))


def make_product_3q_family() -> UQNFamily:
    return UQNFamily(
        name="3q_product",
        n_params=6,
        make_state=product_3q_state,
    )


def make_biseparable_3q_families() -> list[UQNFamily]:
    sampler = random_normal_theta_sampler(12, scale=1.0)
    return [
        UQNFamily("AB|C", 12, biseparable_ab_c_state, theta_sampler=sampler),
        UQNFamily("AC|B", 12, biseparable_ac_b_state, theta_sampler=sampler),
        UQNFamily("BC|A", 12, biseparable_bc_a_state, theta_sampler=sampler),
    ]


def make_custom_family(
    name: str,
    n_params: int,
    make_state: Callable[[RealVector], ComplexVector],
    theta_sampler: Optional[Callable[[np.random.Generator], RealVector]] = None,
    batch_make_states: Optional[Callable[[NDArray[np.float64]], NDArray[np.complex128]]] = None,
) -> UQNFamily:
    """
    Convenience wrapper for your actual UQN/B2/network ansatz.

    Example
    -------
    def my_uqn_state(theta):
        # Build your allowed network pure state here.
        return psi

    fam = make_custom_family("B2_family_0", n_params=24, make_state=my_uqn_state)
    """
    return UQNFamily(
        name=name,
        n_params=n_params,
        make_state=make_state,
        theta_sampler=theta_sampler,
        batch_make_states=batch_make_states,
    )




# ---------------------------------------------------------------------------
# 6-qubit pair-product UQN atom families
# ---------------------------------------------------------------------------

def _permute_state_axes(
    psi: ComplexVector,
    *,
    current_order: Sequence[int],
    target_order: Sequence[int],
    local_dim: int = 2,
) -> ComplexVector:
    """
    Permute a pure state from current tensor-axis order to target tensor-axis order.

    Example
    -------
    If psi is ordered as qubits [0,2,1,3], and target_order=[0,1,2,3],
    this function returns the state vector in standard [0,1,2,3] order.
    """
    psi = normalize_ket(psi)
    current_order = tuple(int(x) for x in current_order)
    target_order = tuple(int(x) for x in target_order)

    if sorted(current_order) != sorted(target_order):
        raise ValueError(
            f"current_order and target_order must contain the same labels, "
            f"got {current_order} and {target_order}."
        )

    n = len(current_order)
    tensor = psi.reshape((local_dim,) * n)
    perm = [current_order.index(q) for q in target_order]
    return normalize_ket(np.transpose(tensor, perm).reshape(local_dim ** n))


def pairproduct_6q_state_from_pairs(
    params: RealVector,
    pairs: Sequence[tuple[int, int]],
    *,
    target_order: Sequence[int] = (0, 1, 2, 3, 4, 5),
) -> ComplexVector:
    """
    Six-qubit pair-product pure state.

    Each pair receives an arbitrary two-qubit pure state. The full atom is

        |phi_pair0> tensor |phi_pair1> tensor |phi_pair2>,

    then permuted into target_order.

    Parameters
    ----------
    params:
        24 real parameters:
            params[0:8]   -> first pair state, 4 complex amplitudes
            params[8:16]  -> second pair state
            params[16:24] -> third pair state

    pairs:
        Three disjoint qubit pairs, e.g. [(0,1), (2,3), (4,5)].

    target_order:
        Desired final qubit order. Defaults to standard [0,1,2,3,4,5].

    Returns
    -------
    psi:
        Complex vector of shape (64,).
    """
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (24,):
        raise ValueError(f"pairproduct_6q_state_from_pairs expected 24 params, got {params.shape}.")

    pairs = [tuple(map(int, pair)) for pair in pairs]
    if len(pairs) != 3:
        raise ValueError(f"Expected exactly three pairs, got {pairs}.")

    flat = [q for pair in pairs for q in pair]
    if sorted(flat) != [0, 1, 2, 3, 4, 5]:
        raise ValueError(
            "For this 6-qubit helper, pairs must be a disjoint partition of "
            "{0,1,2,3,4,5}."
        )

    phi0 = normalized_complex_vector_from_reals(params[0:8], dim=4)
    phi1 = normalized_complex_vector_from_reals(params[8:16], dim=4)
    phi2 = normalized_complex_vector_from_reals(params[16:24], dim=4)

    psi_pair_order = normalize_ket(np.kron(np.kron(phi0, phi1), phi2))
    current_order = tuple(flat)

    return _permute_state_axes(
        psi_pair_order,
        current_order=current_order,
        target_order=tuple(target_order),
        local_dim=2,
    )


def pairproduct_6q_01_23_45_state(params: RealVector) -> ComplexVector:
    """
    Basic six-qubit UQN atom:

        |phi_01> tensor |phi_23> tensor |phi_45>,

    returned in standard qubit order [0,1,2,3,4,5].

    This is the direct implementation of the pair family:
        (0,1), (2,3), (4,5).

    It returns a pure state vector of shape (64,), so the target rho passed
    into gilbert_uqn must be a 64x64 density matrix.
    """
    return pairproduct_6q_state_from_pairs(
        params,
        pairs=[(0, 1), (2, 3), (4, 5)],
        target_order=(0, 1, 2, 3, 4, 5),
    )


def make_pairproduct_6q_family(
    pairs: Sequence[tuple[int, int]] = ((0, 1), (2, 3), (4, 5)),
    *,
    name: Optional[str] = None,
) -> UQNFamily:
    """
    Make a six-qubit pair-product UQN atom family.

    Examples
    --------
    make_pairproduct_6q_family()
        -> pairs [(0,1), (2,3), (4,5)]

    make_pairproduct_6q_family([(0,3), (1,4), (2,5)])
        -> arbitrary two-qubit state on each of those three pairs,
           returned in standard [0,1,2,3,4,5] order.
    """
    pairs_tuple = tuple(tuple(map(int, pair)) for pair in pairs)
    if name is None:
        name = "6q_pairproduct_" + "_".join(f"{a}{b}" for a, b in pairs_tuple)

    sampler = random_normal_theta_sampler(24, scale=1.0)

    def make_state(theta: RealVector) -> ComplexVector:
        return pairproduct_6q_state_from_pairs(
            theta,
            pairs=pairs_tuple,
            target_order=(0, 1, 2, 3, 4, 5),
        )

    def batch_make_states(thetas: NDArray[np.float64]) -> NDArray[np.complex128]:
        return _pairproduct_6q_batch_states_from_pairs(
            thetas,
            pairs=pairs_tuple,
            target_order=(0, 1, 2, 3, 4, 5),
        )

    return UQNFamily(
        name=name,
        n_params=24,
        make_state=make_state,
        theta_sampler=sampler,
        batch_make_states=batch_make_states,
    )


def make_pairproduct_6q_01_23_45_family() -> UQNFamily:
    """
    Convenience constructor for the basic six-qubit family:

        (0,1), (2,3), (4,5).
    """
    return make_pairproduct_6q_family(
        pairs=((0, 1), (2, 3), (4, 5)),
        name="6q_pairproduct_01_23_45",
    )




# ---------------------------------------------------------------------------
# Layered 6-qubit UQN families with additional two-qubit unitary layers
# ---------------------------------------------------------------------------

BASIC_SOURCE_PAIRS_6Q: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
    (0, 1),
    (2, 3),
    (4, 5),
)

# Natural closed-ring / triangle local layer complementary to the source pairs.
# Equivalent to [(1,2), (3,4), (5,0)] up to pair ordering.
TRIANGLE_LOCAL_PAIRS_6Q: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = (
    (0, 5),
    (1, 2),
    (3, 4),
)


def normalized_complex_matrix_from_reals(params: RealVector, rows: int, cols: int) -> NDArray[np.complex128]:
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (2 * rows * cols,):
        raise ValueError(
            f"Expected {2 * rows * cols} real parameters for a {rows}x{cols} complex matrix, "
            f"got {params.shape}."
        )
    z = params[: rows * cols] + 1j * params[rows * cols :]
    return z.reshape(rows, cols).astype(np.complex128)


def hermitian_4x4_from_16_reals(params: RealVector) -> ComplexMatrix:
    """
    Map 16 real parameters to a 4x4 Hermitian generator.

    The parameterization is:
        4 real diagonal entries,
        6 complex off-diagonal entries = 12 more reals.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (16,):
        raise ValueError(f"hermitian_4x4_from_16_reals expected 16 params, got {params.shape}.")

    H = np.zeros((4, 4), dtype=np.complex128)
    H[np.diag_indices(4)] = params[:4]
    k = 4
    for i in range(4):
        for j in range(i + 1, 4):
            z = params[k] + 1j * params[k + 1]
            H[i, j] = z
            H[j, i] = z.conjugate()
            k += 2
    return H


def two_qubit_unitary_from_16_reals(params: RealVector, *, generator_scale: float = 1.0) -> ComplexMatrix:
    """
    Arbitrary U(4) parameterization via U = exp(i * generator_scale * H),
    where H is Hermitian.

    This covers the connected unitary group. A global phase is irrelevant for
    pure-state projectors, so this is more than enough for the atom search.
    """
    H = hermitian_4x4_from_16_reals(params)
    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(1j * generator_scale * evals)
    return (evecs * phases[None, :]) @ evecs.conj().T


def hermitian_4x4_from_16_reals_batch(params_batch: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Batch version of hermitian_4x4_from_16_reals."""
    params_batch = np.asarray(params_batch, dtype=np.float64)
    if params_batch.ndim != 2 or params_batch.shape[1] != 16:
        raise ValueError(f"Expected params_batch shape (batch,16), got {params_batch.shape}.")

    batch = params_batch.shape[0]
    H = np.zeros((batch, 4, 4), dtype=np.complex128)
    idx = np.arange(4)
    H[:, idx, idx] = params_batch[:, :4]
    k = 4
    for i in range(4):
        for j in range(i + 1, 4):
            z = params_batch[:, k] + 1j * params_batch[:, k + 1]
            H[:, i, j] = z
            H[:, j, i] = np.conj(z)
            k += 2
    return H


def two_qubit_unitaries_from_16_reals_batch(
    params_batch: NDArray[np.float64],
    *,
    generator_scale: float = 1.0,
) -> NDArray[np.complex128]:
    """Vectorized U=exp(iH) for a batch of 4x4 Hermitian generators."""
    H = hermitian_4x4_from_16_reals_batch(params_batch)
    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(1j * generator_scale * evals)
    return np.einsum("bik,bk,bjk->bij", evecs, phases, evecs.conj(), optimize=True)


def apply_two_qubit_unitaries_to_states_batch(
    states: NDArray[np.complex128],
    Us: NDArray[np.complex128],
    pair: tuple[int, int],
    *,
    n_qubits: int = 6,
) -> NDArray[np.complex128]:
    """Batch apply different two-qubit unitaries to different state vectors."""
    states = np.asarray(states, dtype=np.complex128)
    Us = np.asarray(Us, dtype=np.complex128)
    if states.ndim != 2 or states.shape[1] != 2**n_qubits:
        raise ValueError(f"states must have shape (batch,{2**n_qubits}), got {states.shape}.")
    if Us.shape != (states.shape[0], 4, 4):
        raise ValueError(f"Us must have shape ({states.shape[0]},4,4), got {Us.shape}.")

    a, b = map(int, pair)
    if a == b or not (0 <= a < n_qubits) or not (0 <= b < n_qubits):
        raise ValueError(f"Invalid qubit pair {pair} for n_qubits={n_qubits}.")

    batch = states.shape[0]
    tensor = states.reshape((batch,) + (2,) * n_qubits)
    axes_front = (a, b)
    axes_rest = tuple(q for q in range(n_qubits) if q not in axes_front)
    perm = (0,) + tuple(q + 1 for q in axes_front + axes_rest)
    inv_perm = np.argsort(perm)

    work = np.transpose(tensor, perm).reshape(batch, 4, -1)
    work = np.einsum("bij,bjk->bik", Us, work, optimize=True)
    out = np.transpose(work.reshape((batch, 2, 2) + (2,) * (n_qubits - 2)), inv_perm).reshape(batch, 2**n_qubits)
    norms = np.linalg.norm(out, axis=1)
    return (out / norms[:, None]).astype(np.complex128)


def apply_two_qubit_unitary_to_state(
    psi: ComplexVector,
    U: ComplexMatrix,
    pair: tuple[int, int],
    *,
    n_qubits: int = 6,
) -> ComplexVector:
    """
    Apply a two-qubit unitary to the specified pair of qubit axes.

    Qubit order convention is the same as NumPy reshape((2,)*n_qubits):
    qubit 0 is the first tensor axis and the most significant computational
    basis bit.
    """
    psi = normalize_ket(psi)
    U = np.asarray(U, dtype=np.complex128)
    if U.shape != (4, 4):
        raise ValueError(f"U must have shape (4,4), got {U.shape}.")

    a, b = map(int, pair)
    if a == b or not (0 <= a < n_qubits) or not (0 <= b < n_qubits):
        raise ValueError(f"Invalid qubit pair {pair} for n_qubits={n_qubits}.")

    tensor = psi.reshape((2,) * n_qubits)
    axes_front = (a, b)
    axes_rest = tuple(q for q in range(n_qubits) if q not in axes_front)
    perm = axes_front + axes_rest
    inv_perm = np.argsort(perm)

    work = np.transpose(tensor, perm).reshape(4, -1)
    work = U @ work
    out = np.transpose(work.reshape((2, 2) + (2,) * (n_qubits - 2)), inv_perm).reshape(2**n_qubits)
    return normalize_ket(out)


def _validate_pair_layer(layer: Sequence[tuple[int, int]], *, n_qubits: int = 6) -> tuple[tuple[int, int], ...]:
    layer_tuple = tuple(tuple(map(int, pair)) for pair in layer)
    flat = [q for pair in layer_tuple for q in pair]
    if len(set(flat)) != len(flat):
        raise ValueError(f"Pairs in one unitary layer must be disjoint, got {layer_tuple}.")
    if any(q < 0 or q >= n_qubits for q in flat):
        raise ValueError(f"Layer {layer_tuple} contains invalid qubit index for n_qubits={n_qubits}.")
    return layer_tuple


def normalize_unitary_pair_layers(
    unitary_pair_layers: Sequence[Sequence[tuple[int, int]]] | Sequence[tuple[int, int]],
    *,
    n_qubits: int = 6,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """
    Accept either one layer such as ((0,5),(1,2),(3,4)) or a sequence of
    layers such as [((0,5),(1,2),(3,4)), ((0,1),(2,3),(4,5))].
    """
    if len(unitary_pair_layers) == 0:  # type: ignore[arg-type]
        return tuple()

    first = list(unitary_pair_layers)[0]  # type: ignore[arg-type]
    if len(first) == 2 and all(isinstance(x, (int, np.integer)) for x in first):  # type: ignore[arg-type]
        return (_validate_pair_layer(unitary_pair_layers, n_qubits=n_qubits),)  # type: ignore[arg-type]

    return tuple(_validate_pair_layer(layer, n_qubits=n_qubits) for layer in unitary_pair_layers)  # type: ignore[arg-type]


def layered_pairunitary_6q_state(
    params: RealVector,
    *,
    source_pairs: Sequence[tuple[int, int]] = BASIC_SOURCE_PAIRS_6Q,
    unitary_pair_layers: Sequence[Sequence[tuple[int, int]]] = (TRIANGLE_LOCAL_PAIRS_6Q,),
    generator_scale: float = 1.0,
) -> ComplexVector:
    """
    Six-qubit UQN atom with arbitrary pair sources plus additional two-qubit
    unitary layers.

    State form:

        |psi> = [prod over layers and pairs U_pair] |
            phi_source0> tensor |phi_source1> tensor |phi_source2>

    For the natural triangle/UQN layer with source pairs
    (0,1),(2,3),(4,5), use unitary pairs (0,5),(1,2),(3,4).

    Parameter layout:
        first 24 reals:
            arbitrary two-qubit pure source states, 8 reals per pair.
        then 16 reals per two-qubit unitary:
            Hermitian generator H for U=exp(iH).

    If unitary_pair_layers contains L layers with 3 disjoint pairs each, the
    total parameter count is 24 + 48*L.
    """
    layers = normalize_unitary_pair_layers(unitary_pair_layers, n_qubits=6)
    n_unitaries = sum(len(layer) for layer in layers)
    expected = 24 + 16 * n_unitaries

    params = np.asarray(params, dtype=np.float64)
    if params.shape != (expected,):
        raise ValueError(f"layered_pairunitary_6q_state expected {expected} params, got {params.shape}.")

    psi = pairproduct_6q_state_from_pairs(params[:24], pairs=source_pairs, target_order=(0, 1, 2, 3, 4, 5))

    offset = 24
    for layer in layers:
        for pair in layer:
            U = two_qubit_unitary_from_16_reals(
                params[offset : offset + 16],
                generator_scale=generator_scale,
            )
            psi = apply_two_qubit_unitary_to_state(psi, U, pair, n_qubits=6)
            offset += 16

    return normalize_ket(psi)


def _pairproduct_6q_batch_states_from_pairs(
    params_batch: NDArray[np.float64],
    pairs: Sequence[tuple[int, int]],
    *,
    target_order: Sequence[int] = (0, 1, 2, 3, 4, 5),
) -> NDArray[np.complex128]:
    """
    Batch version for source-only 6-qubit pair-product states.
    """
    params_batch = np.asarray(params_batch, dtype=np.float64)
    if params_batch.ndim != 2 or params_batch.shape[1] != 24:
        raise ValueError(f"Expected params_batch shape (batch,24), got {params_batch.shape}.")

    pairs = [tuple(map(int, pair)) for pair in pairs]
    flat = [q for pair in pairs for q in pair]
    if sorted(flat) != [0, 1, 2, 3, 4, 5]:
        raise ValueError("pairs must be a disjoint partition of {0,1,2,3,4,5}.")

    batch = params_batch.shape[0]
    phis = []
    for k in range(3):
        block = params_batch[:, 8*k : 8*(k+1)]
        z = block[:, :4] + 1j * block[:, 4:]
        norms = np.linalg.norm(z, axis=1)
        bad = norms <= 1e-15
        if np.any(bad):
            z[bad] = 0.0
            z[bad, 0] = 1.0
            norms = np.linalg.norm(z, axis=1)
        phis.append(z / norms[:, None])

    # Batch Kronecker: (B,4)x(B,4)x(B,4) -> (B,64)
    psi = np.einsum("bi,bj,bk->bijk", phis[0], phis[1], phis[2], optimize=True).reshape(batch, 64)

    current_order = tuple(flat)
    target_order = tuple(int(x) for x in target_order)
    if current_order != target_order:
        perm = [current_order.index(q) for q in target_order]
        psi = np.transpose(psi.reshape((batch,) + (2,) * 6), (0,) + tuple(p + 1 for p in perm)).reshape(batch, 64)

    return psi.astype(np.complex128)


def _layered_pairunitary_6q_batch_states(
    params_batch: NDArray[np.float64],
    *,
    source_pairs: Sequence[tuple[int, int]],
    unitary_pair_layers: Sequence[Sequence[tuple[int, int]]],
    generator_scale: float,
) -> NDArray[np.complex128]:
    """
    Vectorized batch maker for layered UQN states.

    NumPy's stacked eigh is used for all 4x4 unitary generators in a layer,
    and each pair unitary is applied to the full batch at once. This is much
    faster than evaluating random preselection candidates one at a time.
    """
    params_batch = np.asarray(params_batch, dtype=np.float64)
    layers = normalize_unitary_pair_layers(unitary_pair_layers, n_qubits=6)
    n_unitaries = sum(len(layer) for layer in layers)
    expected = 24 + 16 * n_unitaries
    if params_batch.ndim != 2 or params_batch.shape[1] != expected:
        raise ValueError(f"Expected params_batch shape (batch,{expected}), got {params_batch.shape}.")

    states = _pairproduct_6q_batch_states_from_pairs(
        params_batch[:, :24],
        pairs=source_pairs,
        target_order=(0, 1, 2, 3, 4, 5),
    )

    offset = 24
    for layer in layers:
        for pair in layer:
            Us = two_qubit_unitaries_from_16_reals_batch(
                params_batch[:, offset : offset + 16],
                generator_scale=generator_scale,
            )
            states = apply_two_qubit_unitaries_to_states_batch(states, Us, pair, n_qubits=6)
            offset += 16

    return states.astype(np.complex128)


def make_layered_pairunitary_6q_family(
    *,
    source_pairs: Sequence[tuple[int, int]] = BASIC_SOURCE_PAIRS_6Q,
    unitary_pair_layers: Sequence[Sequence[tuple[int, int]]] = (TRIANGLE_LOCAL_PAIRS_6Q,),
    name: Optional[str] = None,
    generator_scale: float = 1.0,
    theta_scale: float = 0.35,
    enable_batch_preselection: bool = True,
) -> UQNFamily:
    """
    General six-qubit UQN family with arbitrary source-pair states and arbitrary
    two-qubit unitary layers.

    This is the main constructor to use when you want the additional unitary
    layers included.
    """
    source_pairs_tuple = tuple(tuple(map(int, pair)) for pair in source_pairs)
    layers = normalize_unitary_pair_layers(unitary_pair_layers, n_qubits=6)
    n_unitaries = sum(len(layer) for layer in layers)
    n_params = 24 + 16 * n_unitaries

    if name is None:
        source_name = "src" + "_".join(f"{a}{b}" for a, b in source_pairs_tuple)
        layer_name = "L" + "_".join("-".join(f"{a}{b}" for a, b in layer) for layer in layers)
        name = f"6q_layered_pairunitary_{source_name}_{layer_name}"

    def sampler(rng: np.random.Generator) -> RealVector:
        theta = np.empty(n_params, dtype=np.float64)
        # Source states: normal complex vectors.
        theta[:24] = rng.normal(0.0, 1.0, size=24)
        # Unitary generators: smaller initial scale is usually better for
        # preselection/local search than huge random exp(iH) jumps.
        theta[24:] = rng.normal(0.0, theta_scale, size=n_params - 24)
        return theta

    def make_state(theta: RealVector) -> ComplexVector:
        return layered_pairunitary_6q_state(
            theta,
            source_pairs=source_pairs_tuple,
            unitary_pair_layers=layers,
            generator_scale=generator_scale,
        )

    if enable_batch_preselection:
        def batch_make_states(thetas: NDArray[np.float64]) -> NDArray[np.complex128]:
            return _layered_pairunitary_6q_batch_states(
                thetas,
                source_pairs=source_pairs_tuple,
                unitary_pair_layers=layers,
                generator_scale=generator_scale,
            )
    else:
        batch_make_states = None

    return UQNFamily(
        name=name,
        n_params=n_params,
        make_state=make_state,
        theta_sampler=sampler,
        batch_make_states=batch_make_states,
    )


def make_triangle_uqn_6q_family(
    *,
    num_local_layers: int = 1,
    source_pairs: Sequence[tuple[int, int]] = BASIC_SOURCE_PAIRS_6Q,
    local_pairs: Sequence[tuple[int, int]] = TRIANGLE_LOCAL_PAIRS_6Q,
    name: Optional[str] = None,
    generator_scale: float = 1.0,
    theta_scale: float = 0.35,
    enable_batch_preselection: bool = True,
) -> UQNFamily:
    """
    Convenience constructor for the natural 6-qubit triangle/UQN ansatz:

        sources:      (0,1), (2,3), (4,5)
        local layer:  (0,5), (1,2), (3,4)

    num_local_layers repeats the local unitary layer. One layer has 72 real
    parameters: 24 source parameters + 3*16 unitary-generator parameters.
    """
    if num_local_layers < 0:
        raise ValueError("num_local_layers must be nonnegative.")
    layers = tuple(tuple(tuple(map(int, pair)) for pair in local_pairs) for _ in range(num_local_layers))
    if name is None:
        name = f"6q_triangle_uqn_{num_local_layers}local_layers"
    return make_layered_pairunitary_6q_family(
        source_pairs=source_pairs,
        unitary_pair_layers=layers,
        name=name,
        generator_scale=generator_scale,
        theta_scale=theta_scale,
        enable_batch_preselection=enable_batch_preselection,
    )


def make_alternating_ring_uqn_6q_family(
    *,
    depth: int = 1,
    even_pairs: Sequence[tuple[int, int]] = BASIC_SOURCE_PAIRS_6Q,
    odd_pairs: Sequence[tuple[int, int]] = TRIANGLE_LOCAL_PAIRS_6Q,
    include_even_unitary_layers: bool = False,
    name: Optional[str] = None,
    generator_scale: float = 1.0,
    theta_scale: float = 0.35,
    enable_batch_preselection: bool = True,
) -> UQNFamily:
    """
    Optional brickwork/ring variant.

    The source states already cover arbitrary pure states on even_pairs, so an
    immediate even-pair unitary layer is redundant. If include_even_unitary_layers
    is False, this applies only the odd/local layer depth times. If True, each
    depth block applies odd_pairs then even_pairs.
    """
    if depth < 0:
        raise ValueError("depth must be nonnegative.")
    layers: list[tuple[tuple[int, int], ...]] = []
    for _ in range(depth):
        layers.append(tuple(tuple(map(int, p)) for p in odd_pairs))
        if include_even_unitary_layers:
            layers.append(tuple(tuple(map(int, p)) for p in even_pairs))
    if name is None:
        suffix = "odd_even" if include_even_unitary_layers else "odd"
        name = f"6q_ring_uqn_depth{depth}_{suffix}"
    return make_layered_pairunitary_6q_family(
        source_pairs=even_pairs,
        unitary_pair_layers=tuple(layers),
        name=name,
        generator_scale=generator_scale,
        theta_scale=theta_scale,
        enable_batch_preselection=enable_batch_preselection,
    )


# ---------------------------------------------------------------------------
# Faster memory QP and convenience Gilbert wrapper
# ---------------------------------------------------------------------------

def solve_memory_qp_pure_states(
    rho: ComplexMatrix,
    psis: Sequence[ComplexVector],
    *,
    initial_weights: Optional[RealVector] = None,
    maxiter: int = 1000,
) -> MemoryQPResult:
    """
    Fast memory QP specialized to pure atoms rho_i=|psi_i><psi_i|.

    It uses:
        G_ij = Tr(rho_i rho_j) = |<psi_i|psi_j>|^2
        c_i  = Tr(rho rho_i)   = <psi_i|rho|psi_i>

    This avoids repeatedly forming dense 64x64 atom matrices while solving the
    simplex QP.
    """
    rho = validate_square_matrix(rho, name="rho")
    if len(psis) == 0:
        raise ValueError("At least one pure atom is required.")

    Psi = np.vstack([normalize_ket(psi) for psi in psis]).astype(np.complex128)
    m, dim = Psi.shape
    if rho.shape != (dim, dim):
        raise ValueError(f"rho shape {rho.shape} is incompatible with atom dim {dim}.")

    if m == 1:
        sigma = ket_to_dm(Psi[0])
        dist = hs_norm(rho - sigma)
        return MemoryQPResult(np.ones(1), sigma, dist, True, dist * dist, "single atom")

    overlaps = Psi @ Psi.conj().T
    G = (np.abs(overlaps) ** 2).real.astype(np.float64)
    c = np.einsum("bi,ij,bj->b", Psi.conj(), rho, Psi, optimize=True).real.astype(np.float64)
    rho_norm2 = hs_inner(rho, rho)

    def objective(w: RealVector) -> float:
        w = np.asarray(w, dtype=np.float64)
        return float(w @ G @ w - 2.0 * c @ w + rho_norm2)

    def gradient(w: RealVector) -> RealVector:
        return (2.0 * G @ np.asarray(w, dtype=np.float64) - 2.0 * c).astype(np.float64)

    if initial_weights is None or np.asarray(initial_weights).shape != (m,):
        w0 = np.full(m, 1.0 / m, dtype=np.float64)
    else:
        w0 = np.maximum(np.asarray(initial_weights, dtype=np.float64), 0.0)
        s = float(np.sum(w0))
        w0 = np.full(m, 1.0 / m, dtype=np.float64) if s <= 1e-15 else w0 / s

    # Keep the starting feasible point as a monotonic fallback.  In Gilbert
    # with memory, w0 should normally represent at least the exact line-search
    # update, so the QP must not return something worse than w0 merely because
    # SLSQP stopped early or drifted numerically.
    initial_objective = objective(w0)

    if minimize is None:
        weights = w0
        sigma = sum(float(weights[i]) * ket_to_dm(Psi[i]) for i in range(m))
        dist = hs_norm(rho - sigma)
        return MemoryQPResult(weights, sigma, dist, False, dist * dist, "scipy unavailable")

    constraints = [{
        "type": "eq",
        "fun": lambda w: float(np.sum(w) - 1.0),
        "jac": lambda w: np.ones_like(w, dtype=np.float64),
    }]
    bounds = [(0.0, 1.0)] * m

    result = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(maxiter), "ftol": 1e-12, "disp": False},
    )

    weights = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    s = float(np.sum(weights))
    weights = np.full(m, 1.0 / m, dtype=np.float64) if s <= 1e-15 else weights / s

    candidate_objective = objective(weights)
    used_fallback = False
    if candidate_objective > initial_objective + 1e-12:
        # SLSQP should not return a worse feasible point for this convex QP,
        # but finite precision / premature termination can happen.  Reverting
        # here is essential for monotonic Gilbert distances.
        weights = w0.copy()
        candidate_objective = initial_objective
        used_fallback = True

    sigma = sum(float(weights[i]) * ket_to_dm(Psi[i]) for i in range(m))
    sigma = np.asarray(sigma, dtype=np.complex128)
    dist = hs_norm(rho - sigma)

    message = str(result.message)
    if used_fallback:
        message = message + "; reverted to initial feasible weights to preserve monotonicity"

    return MemoryQPResult(
        weights=weights,
        sigma=sigma,
        distance=dist,
        success=bool(result.success) and not used_fallback,
        objective=dist * dist,
        message=message,
    )


def prune_memory_by_weights(
    psis: list[ComplexVector],
    weights: Optional[RealVector],
    *,
    memory_size: int,
    min_weight: float = 1e-10,
) -> tuple[list[ComplexVector], Optional[RealVector]]:
    """
    Keep the most relevant remembered atoms by QP weight.
    """
    if len(psis) <= memory_size and weights is None:
        return psis, weights
    if weights is None or len(weights) != len(psis):
        return psis[-memory_size:], None

    weights = np.asarray(weights, dtype=np.float64)
    keep = np.where(weights > min_weight)[0]
    if keep.size == 0:
        keep = np.array([int(np.argmax(weights))])
    if keep.size > memory_size:
        keep = keep[np.argsort(weights[keep])[-memory_size:]]
    keep = np.sort(keep)

    new_psis = [psis[int(i)] for i in keep]
    new_w = weights[keep].copy()
    s = float(np.sum(new_w))
    new_w = None if s <= 1e-15 else new_w / s
    return new_psis, new_w


def gilbert_uqn_fast(
    rho: ComplexMatrix,
    families: Sequence[UQNFamily],
    *,
    rng: Optional[np.random.Generator] = None,
    rho1_init: Optional[ComplexMatrix] = None,
    max_iter: int = 100,
    tol_abs: float = 1e-7,
    tol_rel: float = 1e-7,
    n_random_per_family: int = 1024,
    n_refine: int = 6,
    local_maxiter: int = 80,
    local_method: str = "Powell",
    refine: bool = True,
    mode: OracleMode = "descent",
    memory_size: int = 48,
    memory_prune_tol: float = 1e-9,
    verbose: bool = False,
) -> GilbertResult:
    """
    Faster Gilbert wrapper for expensive 6-qubit UQN atom searches.

    Differences from gilbert_uqn:
        - Uses vectorized random preselection when the family supports it.
        - Uses a pure-state memory QP with Gram entries |<psi_i|psi_j>|^2.
        - Prunes low-weight memory atoms after each memory QP.
        - Defaults to fewer local refinements and Powell, which often works
          better than finite-difference BFGS for high-dimensional unitary
          generator parameters. For very expensive unitary-layer families, set
          refine=False for a pure random-preselection oracle.

    This still follows the same Gilbert algorithm: Step 2 support-oracle atom
    search followed by Step 3 convex recombination over the segment/memory
    simplex.
    """
    rho = project_density_matrix(rho, normalize_trace=True, hermitize_input=True)
    dim = rho.shape[0]
    rng = np.random.default_rng() if rng is None else rng

    atoms_results: list[AtomResult] = []
    memory_psis: list[ComplexVector] = []
    weights: Optional[RealVector] = None

    if rho1_init is None:
        rho_mm = np.eye(dim, dtype=np.complex128) / dim
        first = step2_support_oracle(
            rho,
            rho_mm,
            families,
            rng=rng,
            n_random_per_family=n_random_per_family,
            n_refine=max(1, n_refine),
            maxiter=local_maxiter,
            mode="descent",
            require_positive_descent=False,
            refine=refine,
            local_method=local_method,
        )
        rho1 = first.rho2.copy()
        atoms_results.append(first)
        memory_psis.append(first.psi.copy())
        weights = np.ones(1, dtype=np.float64)
    else:
        rho1 = project_density_matrix(rho1_init, normalize_trace=True, hermitize_input=True)
        if rho1.shape != rho.shape:
            raise ValueError(f"rho1_init shape mismatch: got {rho1.shape}, expected {rho.shape}.")

    distance = hs_norm(rho - rho1)
    distances = [distance]
    converged = False
    stop_reason = "max_iter reached"

    if verbose:
        print(f"init distance={distance:.12g}")

    for it in range(max_iter):
        prev_distance = distance

        atom = step2_support_oracle(
            rho,
            rho1,
            families,
            rng=rng,
            n_random_per_family=n_random_per_family,
            n_refine=max(1, n_refine),
            maxiter=local_maxiter,
            mode=mode,
            require_positive_descent=True,
            refine=refine,
            local_method=local_method,
        )

        if not atom.success:
            stop_reason = "no positive descent atom found"
            if verbose:
                print(
                    f"iter={it:04d} stop: no descent atom; "
                    f"support={atom.support_value:.6g} gain={atom.descent_gain:.6g}"
                )
            break

        atoms_results.append(atom)

        # The exact Step-3 line search is a guaranteed non-increasing fallback
        # whenever atom.descent_gain > 0.  The memory QP should improve on this,
        # but never be allowed to replace it with a worse point.
        p_line, rho_line, line_distance = gilbert_line_search(rho, rho1, atom.rho2)

        old_weights = None
        if weights is not None and len(weights) == len(memory_psis):
            old_weights = np.asarray(weights, dtype=np.float64).copy()

        memory_psis.append(atom.psi.copy())

        # Initial QP point = exact line-search candidate represented in the
        # enlarged memory simplex.  This is the key monotonicity fix.
        initial_qp_weights = None
        if old_weights is not None:
            initial_qp_weights = np.concatenate([p_line * old_weights, [1.0 - p_line]])

        qp = solve_memory_qp_pure_states(
            rho,
            memory_psis,
            initial_weights=initial_qp_weights,
            maxiter=1000,
        )

        monotone_tol = max(1e-12, 1e-10 * max(1.0, prev_distance))
        if qp.distance <= line_distance + monotone_tol and qp.distance <= prev_distance + monotone_tol:
            rho1 = qp.sigma
            distance = qp.distance
            weights = qp.weights
        else:
            # Fall back to the exact line-search update.  This should be rare,
            # but it protects the algorithm from QP or memory numerical issues.
            rho1 = rho_line
            distance = line_distance
            weights = initial_qp_weights

        # Prune only if the pruned simplex can reproduce a point no worse than
        # the accepted one.  Otherwise keep the larger memory; a hard prune can
        # destroy monotonicity by removing atoms needed for the current iterate.
        if len(memory_psis) > memory_size and weights is not None and len(weights) == len(memory_psis):
            pruned_psis, pruned_weights = prune_memory_by_weights(
                memory_psis,
                weights,
                memory_size=memory_size,
                min_weight=memory_prune_tol,
            )
            if len(pruned_psis) < len(memory_psis) and pruned_weights is not None:
                qp_pruned = solve_memory_qp_pure_states(
                    rho,
                    pruned_psis,
                    initial_weights=pruned_weights,
                    maxiter=1000,
                )
                if qp_pruned.distance <= distance + monotone_tol:
                    memory_psis = pruned_psis
                    weights = qp_pruned.weights
                    rho1 = qp_pruned.sigma
                    distance = qp_pruned.distance

        distances.append(distance)
        abs_improvement = prev_distance - distance
        rel_improvement = abs_improvement / max(prev_distance, 1e-15)

        if verbose:
            print(
                f"iter={it:04d} dist={distance:.12g} "
                f"abs_imp={abs_improvement:.3g} rel_imp={rel_improvement:.3g} "
                f"support={atom.support_value:.6g} gain={atom.descent_gain:.6g} "
                f"mem={len(memory_psis)} family={atom.family_name}          ", end='\r', flush=True
            )

        if abs_improvement < -monotone_tol:
            warnings.warn(
                f"Distance increased by {-abs_improvement:.3e}. This indicates a monotonicity bug or severe numerical issues.",
                RuntimeWarning,
            )

        if abs_improvement >= 0.0 and (abs_improvement <= tol_abs or rel_improvement <= tol_rel):
            converged = True
            stop_reason = "distance improvement below tolerance"
            break

    return GilbertResult(
        sigma=rho1,
        distance=distance,
        distances=distances,
        atoms=atoms_results,
        weights=weights,
        converged=converged,
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# Small demo / smoke test
# ---------------------------------------------------------------------------

def ghz_state(n_qubits: int = 3) -> ComplexVector:
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0 / np.sqrt(2.0)
    psi[-1] = 1.0 / np.sqrt(2.0)
    return psi


def white_noise_mixture(psi: ComplexVector, p: float) -> ComplexMatrix:
    """
    rho = (1-p)|psi><psi| + p I/d.
    """
    psi = normalize_ket(psi)
    dim = psi.size
    return (1.0 - p) * ket_to_dm(psi) + p * np.eye(dim, dtype=np.complex128) / dim


def _smoke_test() -> None:
    rng = np.random.default_rng(1234)
    rho = white_noise_mixture(ghz_state(3), p=0.5)

    families = [make_product_3q_family()]
    result = gilbert_uqn(
        rho,
        families,
        rng=rng,
        max_iter=5,
        n_random_per_family=128,
        n_refine=4,
        local_maxiter=50,
        use_memory=True,
        verbose=True,
    )

    print("final distance:", result.distance)
    print("stop reason:", result.stop_reason)


if __name__ == "__main__":
    _smoke_test()

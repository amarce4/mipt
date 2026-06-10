
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
    """

    name: str
    n_params: int
    make_state: Callable[[RealVector], ComplexVector]
    theta_sampler: Optional[Callable[[np.random.Generator], RealVector]] = None

    def sample_theta(self, rng: np.random.Generator) -> RealVector:
        if self.theta_sampler is not None:
            theta = self.theta_sampler(rng)
            return np.asarray(theta, dtype=np.float64)
        return rng.uniform(0.0, 2.0 * np.pi, size=self.n_params).astype(np.float64)


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
        for _ in range(n_random_per_family):
            theta = fam.sample_theta(rng)
            val, _ = _evaluate_family_theta(fam, theta, H)
            candidates.append((val, fam, theta))

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

    result = minimize(
        objective,
        theta0,
        method=method,
        options={"maxiter": int(maxiter), "gtol": 1e-8},
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
            refine=True,
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
            refine=True,
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

    return UQNFamily(
        name=name,
        n_params=24,
        make_state=make_state,
        theta_sampler=sampler,
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

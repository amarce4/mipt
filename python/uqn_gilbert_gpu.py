r"""
GPU-capable Gilbert-with-memory estimator for tripartite unitary quantum networks (UQN).

The principal entry point is ``gilbert_uqn_distance_3q``. It accepts a precomputed
three-qubit 8x8 density matrix (including ``qiskit.quantum_info.DensityMatrix``),
constructs the paper's extended state rho_ext = rho \otimes I_8/8, and runs
Gilbert optimization against pure six-qubit UQN atoms on three parties of local
dimension four. The UQN oracle includes all three tripartite topology classes in
Fig. 5 of the supplement to arXiv:2512.11118:

    - biseparable: one arbitrary bipartite resource and one isolated party;
    - chain: two two-qubit link resources;
    - triangle: three two-qubit link resources.

The expensive support-oracle and simplex/QP calculations are performed in
PyTorch and can execute on CUDA. This remains a heuristic upper-bound estimator:
the support problem over UQN pure atoms is non-convex.

Important deterministic safeguard:
    Computational-basis projectors are valid UQN atoms and are retained in the
    active set. Consequently, diagonal states (including the p=1 output of a
    measurement-after-every-layer MIPT circuit) are represented exactly rather
    than reported as false positive distances.

Dependencies:
    pip install numpy torch

For CUDA, install a PyTorch build appropriate to the local CUDA/driver setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence
import math
import time

import numpy as np
from numpy.typing import NDArray
import torch

Array = NDArray[np.complex128]

# Six-qubit order in the extended UQN calculation:
# [A0, A1, B0, B1, C0, C1], with visible physical qubits A0,B0,C0.
VISIBLE_QUBITS = (0, 2, 4)


@dataclass
class UQNGilbert3QResult:
    """Result returned for an input three-qubit density matrix."""

    distance: float
    squared_distance: float
    sigma: Array
    selected_iteration: int
    identity_bound: float
    extended_distance: float
    iterations: int
    status: str
    reduced_history: list[float]
    extended_history: list[float]
    gaps: list[float]
    oracle_history: list[dict]
    device: str
    elapsed_seconds: float


# -----------------------------------------------------------------------------
# Input handling and basic matrix operations
# -----------------------------------------------------------------------------


def ketbra_np(psi: Array) -> Array:
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm <= 1e-15:
        raise ValueError("Cannot construct a projector from a zero vector.")
    psi = psi / norm
    return np.outer(psi, psi.conj())


def hs_distance_np(a: Array, b: Array) -> float:
    delta = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(max(np.real(np.vdot(delta, delta)), 0.0)))


def _normalize_density_np(rho: Array, dim: int, psd_tol: float = 1e-8) -> Array:
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (dim, dim):
        raise ValueError(f"Expected shape {(dim, dim)}, received {rho.shape}.")
    rho = (rho + rho.conj().T) / 2.0
    tr = np.trace(rho)
    if abs(tr) <= 1e-14:
        raise ValueError("Density matrix has zero trace.")
    rho = rho / tr
    minimum = float(np.min(np.linalg.eigvalsh(rho)))
    if minimum < -psd_tol:
        raise ValueError(f"Density matrix is not PSD; minimum eigenvalue={minimum:g}.")
    return rho


def qiskit_3q_to_internal_order(
    rho_qiskit: Array,
    qiskit_qubit_order: Sequence[int] = (0, 1, 2),
) -> Array:
    """
    Convert Qiskit's little-endian 3-qubit basis into internal |ABC> order.

    ``qiskit_qubit_order`` specifies which saved Qiskit qubit is A, B, C.
    For ``save_density_matrix(qubits=[0,1,2])``, use the default.
    """
    rho_qiskit = np.asarray(rho_qiskit, dtype=np.complex128)
    if rho_qiskit.shape != (8, 8):
        raise ValueError(f"Expected 8x8 matrix, received {rho_qiskit.shape}.")
    order = tuple(int(q) for q in qiskit_qubit_order)
    if sorted(order) != [0, 1, 2]:
        raise ValueError("qiskit_qubit_order must be a permutation of (0,1,2).")

    perm: list[int] = []
    for i in range(8):
        bits_abc = ((i >> 2) & 1, (i >> 1) & 1, i & 1)
        qbits = {order[pos]: bits_abc[pos] for pos in range(3)}
        perm.append(sum(qbits[q] << q for q in range(3)))
    idx = np.asarray(perm, dtype=int)
    return rho_qiskit[np.ix_(idx, idx)]


def as_density_matrix_3q(
    rho_like: Any,
    input_order: Literal["qiskit", "internal"] = "qiskit",
    qiskit_qubit_order: Sequence[int] = (0, 1, 2),
) -> Array:
    """Accept only a precomputed 8x8 three-qubit state or 8-component vector."""
    if rho_like.__class__.__name__ == "QuantumCircuit":
        raise TypeError("Pass a precomputed 8x8 DensityMatrix, not a QuantumCircuit.")
    raw = rho_like.data if hasattr(rho_like, "data") else rho_like
    arr = np.asarray(raw, dtype=np.complex128)
    if arr.shape == (8,):
        rho = ketbra_np(arr)
    elif arr.shape == (8, 8):
        rho = arr
    else:
        raise ValueError(f"Expected shape (8,) or (8,8), received {arr.shape}.")
    if input_order == "qiskit":
        rho = qiskit_3q_to_internal_order(rho, qiskit_qubit_order)
    elif input_order != "internal":
        raise ValueError("input_order must be 'qiskit' or 'internal'.")
    return _normalize_density_np(rho, 8)


def extend_3q_state(rho: Array) -> Array:
    """
    Build rho_ext = rho_ABC tensor I_abc/8 in interleaved party order
    [A0,A1,B0,B1,C0,C1] = [A,a,B,b,C,c].
    """
    rho = _normalize_density_np(rho, 8)
    ancilla = np.eye(8, dtype=np.complex128) / 8.0
    raw = np.kron(rho, ancilla).reshape((2,) * 12)  # A,B,C,a,b,c | primed
    axes = (0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11)
    return np.transpose(raw, axes).reshape(64, 64)


def _choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chosen = torch.device(device)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but torch.cuda.is_available() is False.")
    return chosen


def _choose_complex_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.complex64 if device.type == "cuda" else torch.complex128
    if dtype == "complex64":
        return torch.complex64
    if dtype == "complex128":
        return torch.complex128
    raise ValueError("dtype must be 'auto', 'complex64', or 'complex128'.")


def _real_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if complex_dtype == torch.complex64 else torch.float64


def _hs_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    delta = a - b
    return torch.real(torch.sum(torch.conj(delta) * delta))


def _hs_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(a) * b))


def _projector(psi: torch.Tensor) -> torch.Tensor:
    psi = psi / torch.linalg.vector_norm(psi)
    return psi[:, None] * torch.conj(psi[None, :])


def _reduce_visible_3q(rho6: torch.Tensor) -> torch.Tensor:
    """Trace ancillas A1,B1,C1 from order [A0,A1,B0,B1,C0,C1]."""
    tensor = rho6.reshape((2,) * 12)
    tensor = tensor.permute(0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11).reshape(8, 8, 8, 8)
    return torch.einsum("iaja->ij", tensor)


def _basis_projectors_6q(device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    eye = torch.eye(64, dtype=dtype, device=device)
    return [eye[:, k:k+1] @ torch.conj(eye[:, k:k+1]).T for k in range(64)]


# -----------------------------------------------------------------------------
# CUDA-capable UQN support oracle
# -----------------------------------------------------------------------------


class BatchedUQNSupportOracle:
    """
    Batched gradient support oracle over the Fig. 5 tripartite UQN families.

    Each call seeks an atom tau maximizing Tr[(target-sigma) tau]. Because this
    problem is nonconvex, the result is heuristic and should be monitored using
    the reported Gilbert gap and benchmarks.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        starts: int = 32,
        steps: int = 150,
        learning_rate: float = 0.04,
        seed: Optional[int] = None,
        include_biseparable: bool = True,
        include_chain: bool = True,
        include_triangle: bool = True,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.real_dtype = _real_dtype(dtype)
        self.starts = int(starts)
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.include_biseparable = bool(include_biseparable)
        self.include_chain = bool(include_chain)
        self.include_triangle = bool(include_triangle)
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.fixed_00 = torch.zeros((2, 2), dtype=dtype, device=device)
        self.fixed_00[0, 0] = 1.0

    def _rand_complex(self, shape: tuple[int, ...]) -> torch.nn.Parameter:
        re = torch.randn(shape, dtype=self.real_dtype, device=self.device, generator=self.generator)
        im = torch.randn(shape, dtype=self.real_dtype, device=self.device, generator=self.generator)
        return torch.nn.Parameter(torch.complex(re, im))

    @staticmethod
    def _unit_vectors(z: torch.Tensor) -> torch.Tensor:
        return z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-12)

    def _unitaries4(self, z: torch.Tensor) -> torch.Tensor:
        # QR parameterization of U(4), batched over independent random starts.
        q, r = torch.linalg.qr(z)
        diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
        phase = diagonal / torch.abs(diagonal).clamp_min(1e-12)
        return q * torch.conj(phase).unsqueeze(-2)

    @staticmethod
    def _values(psi: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        psi = psi / torch.linalg.vector_norm(psi, dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.real(torch.einsum("si,ij,sj->s", torch.conj(psi), residual, psi))

    def _optimize(self, residual: torch.Tensor, params: list[torch.nn.Parameter], build) -> tuple[torch.Tensor, float]:
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        best_value = -math.inf
        best_psi: Optional[torch.Tensor] = None

        for _ in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            psi = build(params)
            values = self._values(psi, residual)
            loss = -values.sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                vmax, imax = torch.max(values, dim=0)
                value = float(vmax.item())
                if value > best_value:
                    best_value = value
                    best_psi = psi[imax].detach().clone()

        with torch.no_grad():
            psi = build(params)
            values = self._values(psi, residual)
            vmax, imax = torch.max(values, dim=0)
            value = float(vmax.item())
            if value > best_value or best_psi is None:
                best_value = value
                best_psi = psi[imax].detach().clone()

        best_psi = best_psi / torch.linalg.vector_norm(best_psi)
        return best_psi, best_value

    def _triangle_raw(self, s_ab: torch.Tensor, s_bc: torch.Tensor, s_ca: torch.Tensor) -> torch.Tensor:
        # Resources: AB[A0,B0], BC[B1,C0], CA[C1,A1].
        raw6 = torch.einsum("sab,scd,sef->safbcde", s_ab, s_bc, s_ca)
        return raw6.reshape(-1, 4, 4, 4)

    def _apply_local(self, raw: torch.Tensor, UA: torch.Tensor, UB: torch.Tensor, UC: torch.Tensor) -> torch.Tensor:
        return torch.einsum("sai,sbj,sck,sijk->sabc", UA, UB, UC, raw).reshape(-1, 64)

    def biseparable(self, residual: torch.Tensor, isolated: int) -> tuple[torch.Tensor, float, str]:
        # A general pure state on the joined two-party Hilbert space (dimension 16)
        # tensor a pure state of the isolated party (dimension 4).
        pair = self._rand_complex((self.starts, 16))
        single = self._rand_complex((self.starts, 4))

        def build(params):
            pair_v = self._unit_vectors(params[0]).reshape(-1, 4, 4)
            single_v = self._unit_vectors(params[1])
            if isolated == 0:      # A | BC
                return torch.einsum("sa,sbc->sabc", single_v, pair_v).reshape(-1, 64)
            if isolated == 1:      # B | AC
                return torch.einsum("sac,sb->sabc", pair_v, single_v).reshape(-1, 64)
            if isolated == 2:      # C | AB
                return torch.einsum("sab,sc->sabc", pair_v, single_v).reshape(-1, 64)
            raise ValueError("isolated must be 0, 1, or 2.")

        psi, val = self._optimize(residual, [pair, single], build)
        return psi, val, ("biseparable:A|BC", "biseparable:B|AC", "biseparable:C|AB")[isolated]

    def chain(self, residual: torch.Tensor, omitted_link: str) -> tuple[torch.Tensor, float, str]:
        # A chain is the triangle circuit with one source fixed to |00>.
        x = self._rand_complex((self.starts, 4))
        y = self._rand_complex((self.starts, 4))
        UA = self._rand_complex((self.starts, 4, 4))
        UB = self._rand_complex((self.starts, 4, 4))
        UC = self._rand_complex((self.starts, 4, 4))
        fixed = self.fixed_00.unsqueeze(0).expand(self.starts, -1, -1)

        def build(params):
            left = self._unit_vectors(params[0]).reshape(-1, 2, 2)
            right = self._unit_vectors(params[1]).reshape(-1, 2, 2)
            ua = self._unitaries4(params[2])
            ub = self._unitaries4(params[3])
            uc = self._unitaries4(params[4])
            if omitted_link == "CA":
                raw = self._triangle_raw(left, right, fixed)       # AB, BC
            elif omitted_link == "AB":
                raw = self._triangle_raw(fixed, left, right)       # BC, CA
            elif omitted_link == "BC":
                raw = self._triangle_raw(left, fixed, right)       # AB, CA
            else:
                raise ValueError("omitted_link must be AB, BC, or CA.")
            return self._apply_local(raw, ua, ub, uc)

        psi, val = self._optimize(residual, [x, y, UA, UB, UC], build)
        return psi, val, f"chain:omit_{omitted_link}"

    def triangle(self, residual: torch.Tensor) -> tuple[torch.Tensor, float, str]:
        ab = self._rand_complex((self.starts, 4))
        bc = self._rand_complex((self.starts, 4))
        ca = self._rand_complex((self.starts, 4))
        UA = self._rand_complex((self.starts, 4, 4))
        UB = self._rand_complex((self.starts, 4, 4))
        UC = self._rand_complex((self.starts, 4, 4))

        def build(params):
            s_ab = self._unit_vectors(params[0]).reshape(-1, 2, 2)
            s_bc = self._unit_vectors(params[1]).reshape(-1, 2, 2)
            s_ca = self._unit_vectors(params[2]).reshape(-1, 2, 2)
            ua = self._unitaries4(params[3])
            ub = self._unitaries4(params[4])
            uc = self._unitaries4(params[5])
            return self._apply_local(self._triangle_raw(s_ab, s_bc, s_ca), ua, ub, uc)

        psi, val = self._optimize(residual, [ab, bc, ca, UA, UB, UC], build)
        return psi, val, "triangle"

    def __call__(self, residual: torch.Tensor) -> tuple[torch.Tensor, dict]:
        residual = (residual + torch.conj(residual.T)) / 2.0
        candidates: list[tuple[torch.Tensor, float, str]] = []
        if self.include_biseparable:
            candidates.extend(self.biseparable(residual, cut) for cut in range(3))
        if self.include_chain:
            candidates.extend(self.chain(residual, omitted) for omitted in ("AB", "BC", "CA"))
        if self.include_triangle:
            candidates.append(self.triangle(residual))
        if not candidates:
            raise ValueError("At least one UQN topology family must be enabled.")
        best = max(candidates, key=lambda x: x[1])
        psi, support, topology = best
        atom = _projector(psi)
        return atom, {
            "topology": topology,
            "support": support,
            "supports": {name: value for _, value, name in candidates},
            "starts": self.starts,
            "steps": self.steps,
        }


# -----------------------------------------------------------------------------
# Gilbert and Gilbert-with-memory operations, fully torch/GPU capable
# -----------------------------------------------------------------------------


@torch.no_grad()
def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    idx = torch.arange(1, v.numel() + 1, dtype=v.dtype, device=v.device)
    mask = u - cssv / idx > 0
    if not bool(torch.any(mask)):
        return torch.ones_like(v) / v.numel()
    rho = torch.nonzero(mask, as_tuple=False)[-1, 0]
    theta = cssv[rho] / idx[rho]
    return torch.clamp(v - theta, min=0.0)


@torch.no_grad()
def _memory_update(
    target: torch.Tensor,
    states: Sequence[torch.Tensor],
    max_steps: int = 2000,
    tol: float = 1e-10,
) -> torch.Tensor:
    """Page-9 Gilbert-with-memory QP, solved by projected FISTA on GPU/CPU."""
    A = torch.stack([state.reshape(-1) for state in states], dim=1)
    b = target.reshape(-1)
    Q = torch.real(torch.conj(A).T @ A)
    c = torch.real(torch.conj(A).T @ b)
    Q = (Q + Q.T) / 2.0
    L = 2.0 * torch.linalg.eigvalsh(Q)[-1].clamp_min(1e-12)
    real_dtype = Q.dtype
    x = torch.ones(len(states), dtype=real_dtype, device=target.device) / len(states)
    y = x.clone()
    t = torch.tensor(1.0, dtype=real_dtype, device=target.device)
    for _ in range(int(max_steps)):
        grad = 2.0 * (Q @ y - c)
        x_new = _project_simplex(y - grad / L)
        if float(torch.linalg.vector_norm(x_new - x, ord=1).item()) <= tol:
            x = x_new
            break
        t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    out = sum(weight.to(target.dtype) * state for weight, state in zip(x, states))
    return (out + torch.conj(out.T)) / 2.0


@torch.no_grad()
def _line_search(target: torch.Tensor, sigma: torch.Tensor, atom: torch.Tensor) -> torch.Tensor:
    direction = atom - sigma
    denom = _hs_inner(direction, direction).clamp_min(1e-15)
    alpha = torch.clamp(_hs_inner(target - sigma, direction) / denom, 0.0, 1.0)
    out = sigma + alpha.to(target.dtype) * direction
    return (out + torch.conj(out.T)) / 2.0


def _fast_diagonal_solution(
    target_ext: torch.Tensor,
    rho3: torch.Tensor,
    fixed_basis: Sequence[torch.Tensor],
    atol: float,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Return exact diagonal UQN representation when the input state is diagonal."""
    off = rho3 - torch.diag(torch.diagonal(rho3))
    if float(torch.sqrt(_hs_sq(off, torch.zeros_like(off))).item()) > atol:
        return None
    probs = torch.real(torch.diagonal(target_ext))
    sigma_ext = sum(p.to(target_ext.dtype) * atom for p, atom in zip(probs, fixed_basis))
    return sigma_ext, _reduce_visible_3q(sigma_ext)


def gilbert_uqn_distance_3q(
    rho_like: Any,
    *,
    input_order: Literal["qiskit", "internal"] = "qiskit",
    qiskit_qubit_order: Sequence[int] = (0, 1, 2),
    device: str = "auto",
    dtype: str = "auto",
    max_iter: int = 100,
    memory: int = 50,
    include_basis_active_set: bool = True,
    basis_diagonal_atol: float = 1e-12,
    oracle_starts: int = 32,
    oracle_steps: int = 150,
    learning_rate: float = 0.04,
    retry_multiplier: int = 2,
    gap_tol: float = 1e-9,
    distance_tol: float = 1e-9,
    qp_steps: int = 2000,
    qp_tol: float = 1e-10,
    seed: Optional[int] = None,
) -> UQNGilbert3QResult:
    """
    Estimate the paper-style UQN distance upper bound for an arbitrary mixed
    three-qubit state, with CUDA acceleration when available.

    The returned ``distance`` is the lowest reduced three-qubit distance among
    all valid UQN iterates seen, including the basis-seeded initialization. The
    extended-space optimization is the procedure used to generate those states.

    ``status='oracle_failed_to_find_descent'`` indicates an inconclusive
    heuristic-oracle failure; it does not mean the UQN distance has converged.
    """
    start_time = time.perf_counter()
    rho_np = as_density_matrix_3q(rho_like, input_order, qiskit_qubit_order)
    rho_ext_np = extend_3q_state(rho_np)
    chosen_device = _choose_device(device)
    complex_dtype = _choose_complex_dtype(dtype, chosen_device)
    rho3 = torch.as_tensor(rho_np, dtype=complex_dtype, device=chosen_device)
    target = torch.as_tensor(rho_ext_np, dtype=complex_dtype, device=chosen_device)
    identity3 = torch.eye(8, dtype=complex_dtype, device=chosen_device) / 8.0
    identity_bound = float(torch.sqrt(_hs_sq(rho3, identity3)).item())
    basis = _basis_projectors_6q(chosen_device, complex_dtype)

    exact = _fast_diagonal_solution(target, rho3, basis, basis_diagonal_atol)
    if exact is not None:
        sigma_ext, sigma3 = exact
        elapsed = time.perf_counter() - start_time
        return UQNGilbert3QResult(
            distance=0.0,
            squared_distance=0.0,
            sigma=sigma3.detach().cpu().numpy().astype(np.complex128),
            selected_iteration=0,
            identity_bound=identity_bound,
            extended_distance=0.0,
            iterations=0,
            status="exact_diagonal_uqn_decomposition",
            reduced_history=[0.0],
            extended_history=[0.0],
            gaps=[],
            oracle_history=[],
            device=str(chosen_device),
            elapsed_seconds=elapsed,
        )

    # Deterministic valid initialization: best convex fit within all product
    # computational-basis atoms; this is never worse than I/64.
    if include_basis_active_set:
        sigma = _memory_update(target, basis, max_steps=qp_steps, tol=qp_tol)
        fixed_states = basis
    else:
        sigma = torch.eye(64, dtype=complex_dtype, device=chosen_device) / 64.0
        fixed_states = []

    sigma3 = _reduce_visible_3q(sigma)
    reduced_history = [float(torch.sqrt(_hs_sq(rho3, sigma3)).item())]
    extended_history = [float(torch.sqrt(_hs_sq(target, sigma)).item())]
    best_distance = reduced_history[0]
    best_sigma3 = sigma3.detach().clone()
    selected_iteration = 0

    if best_distance <= distance_tol:
        return UQNGilbert3QResult(
            distance=best_distance,
            squared_distance=best_distance * best_distance,
            sigma=best_sigma3.cpu().numpy().astype(np.complex128),
            selected_iteration=0,
            identity_bound=identity_bound,
            extended_distance=extended_history[-1],
            iterations=0,
            status="basis_active_set_reached_tolerance",
            reduced_history=reduced_history,
            extended_history=extended_history,
            gaps=[],
            oracle_history=[],
            device=str(chosen_device),
            elapsed_seconds=time.perf_counter() - start_time,
        )

    dynamic_atoms: list[torch.Tensor] = []
    gaps: list[float] = []
    oracle_history: list[dict] = []
    status = "max_iter_reached"

    oracle = BatchedUQNSupportOracle(
        device=chosen_device,
        dtype=complex_dtype,
        starts=oracle_starts,
        steps=oracle_steps,
        learning_rate=learning_rate,
        seed=seed,
    )

    for iteration in range(1, int(max_iter) + 1):
        residual = target - sigma
        atom, info = oracle(residual)
        gap = float(_hs_inner(residual, atom - sigma).item())

        # Retry an apparent stall with a stronger multi-start search. An
        # approximate non-convex oracle cannot use gap<=0 as a certificate.
        retried = False
        if gap <= gap_tol and retry_multiplier > 1:
            retried = True
            stronger = BatchedUQNSupportOracle(
                device=chosen_device,
                dtype=complex_dtype,
                starts=oracle_starts * retry_multiplier,
                steps=oracle_steps * retry_multiplier,
                learning_rate=learning_rate,
                seed=None if seed is None else seed + 100000 + iteration,
            )
            atom2, info2 = stronger(residual)
            gap2 = float(_hs_inner(residual, atom2 - sigma).item())
            if gap2 > gap:
                atom, info, gap = atom2, info2, gap2

        info = dict(info)
        info.update({"iteration": iteration, "gap": gap, "retried": retried})
        gaps.append(gap)
        oracle_history.append(info)

        if gap <= gap_tol:
            status = "oracle_failed_to_find_descent"
            break

        dynamic_atoms.append(atom.detach())
        if memory > 0:
            states = list(fixed_states) + [sigma] + dynamic_atoms[-int(memory):]
            sigma_new = _memory_update(target, states, max_steps=qp_steps, tol=qp_tol)
        else:
            sigma_new = _line_search(target, sigma, atom)

        sigma = sigma_new
        sigma3 = _reduce_visible_3q(sigma)
        reduced = float(torch.sqrt(_hs_sq(rho3, sigma3)).item())
        extended = float(torch.sqrt(_hs_sq(target, sigma)).item())
        reduced_history.append(reduced)
        extended_history.append(extended)

        if reduced < best_distance:
            best_distance = reduced
            best_sigma3 = sigma3.detach().clone()
            selected_iteration = iteration

        if best_distance <= distance_tol:
            status = "distance_tolerance_reached"
            break

    elapsed = time.perf_counter() - start_time
    return UQNGilbert3QResult(
        distance=best_distance,
        squared_distance=best_distance * best_distance,
        sigma=best_sigma3.detach().cpu().numpy().astype(np.complex128),
        selected_iteration=selected_iteration,
        identity_bound=identity_bound,
        extended_distance=extended_history[-1],
        iterations=len(extended_history) - 1,
        status=status,
        reduced_history=reduced_history,
        extended_history=extended_history,
        gaps=gaps,
        oracle_history=oracle_history,
        device=str(chosen_device),
        elapsed_seconds=elapsed,
    )


# -----------------------------------------------------------------------------
# Diagnostics and benchmark helpers
# -----------------------------------------------------------------------------


def ghz3_density() -> Array:
    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = psi[7] = 1.0 / np.sqrt(2.0)
    return ketbra_np(psi)


def w3_density() -> Array:
    psi = np.zeros(8, dtype=np.complex128)
    psi[1] = psi[2] = psi[4] = 1.0 / np.sqrt(3.0)
    return ketbra_np(psi)


def white_noise_mixture(rho: Array, p: float) -> Array:
    return (1.0 - float(p)) * rho + float(p) * np.eye(8, dtype=np.complex128) / 8.0


def state_diagnostics(rho_like: Any, input_order: Literal["qiskit", "internal"] = "qiskit") -> dict:
    rho = as_density_matrix_3q(rho_like, input_order=input_order)
    identity = np.eye(8, dtype=np.complex128) / 8.0
    off = rho - np.diag(np.diagonal(rho))
    return {
        "purity": float(np.real(np.trace(rho @ rho))),
        "distance_to_identity": hs_distance_np(rho, identity),
        "offdiagonal_hs_norm": float(np.sqrt(max(np.real(np.vdot(off, off)), 0.0))),
        "min_eigenvalue": float(np.min(np.linalg.eigvalsh(rho))),
    }


def run_white_noise_benchmark(
    family: Literal["GHZ3", "W3"] = "GHZ3",
    p_values: Sequence[float] = (0.0, 0.2, 0.4, 0.5, 0.55, 0.57, 0.6),
    **kwargs,
) -> list[UQNGilbert3QResult]:
    base = ghz3_density() if family == "GHZ3" else w3_density()
    results: list[UQNGilbert3QResult] = []
    for i, p in enumerate(p_values):
        call_kwargs = dict(kwargs)
        if call_kwargs.get("seed") is not None:
            call_kwargs["seed"] = int(call_kwargs["seed"]) + i
        result = gilbert_uqn_distance_3q(
            white_noise_mixture(base, float(p)), input_order="internal", **call_kwargs
        )
        print(
            f"{family} p={p:.4g}: D={result.distance:.10g}, "
            f"D_id={result.identity_bound:.10g}, status={result.status}, "
            f"iters={result.iterations}, time={result.elapsed_seconds:.3g}s, "
            f"device={result.device}"
        )
        results.append(result)
    return results


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    # Exact membership smoke test: every diagonal 3-qubit state is a product-basis mixture.
    diag = np.diag([0.18, 0.11, 0.09, 0.16, 0.12, 0.14, 0.10, 0.10]).astype(np.complex128)
    result = gilbert_uqn_distance_3q(diag, input_order="internal", max_iter=1)
    print("diagonal smoke test:", result.distance, result.status)


def gilbert_uqn_distance_3q_best_of(
    rho_like: Any,
    *,
    oracle_seeds: Sequence[int] = (1001, 2001, 3001),
    **kwargs,
) -> UQNGilbert3QResult:
    """
    Run independent heuristic-oracle trials and return the smallest valid upper
    bound. Because each trial produces a valid UQN approximant, taking the
    minimum returned distance is legitimate and helps diagnose oracle noise.
    """
    results = [gilbert_uqn_distance_3q(rho_like, seed=int(seed), **kwargs)
               for seed in oracle_seeds]
    return min(results, key=lambda result: result.distance)

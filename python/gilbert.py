"""
GPU Gilbert-with-memory estimator for three-qubit UQN distance.

This module implements the three-qubit construction in Supplemental Sec. A.5
of arXiv:2512.11118:

    rho_ext = rho \\otimes I_8 / 8

Gilbert iterations are carried out on the six-qubit extended state, with the
six qubits grouped as three parties of local dimension four:

    A = (A_visible, A_ancilla)
    B = (B_visible, B_ancilla)
    C = (C_visible, C_ancilla)

The pure unitary-network (UQN) support oracle includes the three tripartite
network topology families illustrated in Fig. 5 of the supplement, up to
party permutations:

    * biseparable: A|BC, B|AC, C|AB
    * two-link chains: omit_AB, omit_BC, omit_CA
    * triangle: AB, BC, CA

The support oracle is hybrid:

    1. A reusable GPU warm-start bank provides high-scoring candidates for the
       current Gilbert residual.
    2. Only the top candidates from promising topology families are refined by
       differentiable continuous optimization on the GPU.
    3. Refined candidates are cached back into the bank and may be reused for
       subsequent iterations or subsequent input density matrices.

Step 3 of ordinary Gilbert is replaced, by default, with the Gilbert-with-
memory simplex/QP update of Supplemental Eqs. (9)-(10). All reported reduced
three-qubit distances use exactly

    D = sqrt(Tr[(rho - sigma)^2]).

Interpretation
--------------
The continuous support optimization is non-convex. Every sigma returned by the
algorithm is a valid network approximant, so the returned D is an upper bound;
a positive D is not by itself a GNME certificate. A direct biseparable solver
is run first and retained as a mandatory UQN baseline, because the biseparable
set is a subset of UQN. The full UQN result can therefore never be worse than
the direct biseparable candidate on the same input state.

Dependencies
------------
    pip install numpy torch

A CUDA-enabled PyTorch installation is required for device='cuda'.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence
import time

import numpy as np
from numpy.typing import NDArray
import torch

Array = NDArray[np.complex128]

__version__ = '2026-06-04-progressive-oracle-v2'
Topology = Literal[
    "biseparable:A|BC", "biseparable:B|AC", "biseparable:C|AB",
    "chain:omit_AB", "chain:omit_BC", "chain:omit_CA", "triangle"
]

TOPOLOGIES: tuple[Topology, ...] = (
    "biseparable:A|BC", "biseparable:B|AC", "biseparable:C|AB",
    "chain:omit_AB", "chain:omit_BC", "chain:omit_CA", "triangle",
)


@dataclass(frozen=True)
class HybridBankConfig:
    triangle_atoms: int = 32768
    chain_atoms_per_topology: int = 8192
    biseparable_atoms_per_cut: int = 8192
    generation_batch_size: int = 4096
    seed: Optional[int] = 1234


@dataclass
class FamilySeedBank:
    topology: str
    psi: torch.Tensor
    params: dict[str, torch.Tensor]

    @property
    def size(self) -> int:
        return int(self.psi.shape[0])

    def append(self, psi: torch.Tensor, params: dict[str, torch.Tensor]) -> None:
        psi = psi.detach().reshape(1, 64).to(device=self.psi.device, dtype=self.psi.dtype)
        self.psi = torch.cat((self.psi, psi), dim=0).contiguous()
        for name, old in self.params.items():
            new = params[name].detach().reshape((1,) + tuple(old.shape[1:])).to(
                device=old.device, dtype=old.dtype
            )
            self.params[name] = torch.cat((old, new), dim=0).contiguous()


@dataclass
class HybridUQNBank3Q:
    families: dict[str, FamilySeedBank]
    basis_psi: torch.Tensor
    config: HybridBankConfig
    device: str
    dtype: str
    build_seconds: float
    refined_atoms_added: int = 0

    @property
    def size(self) -> int:
        return int(self.basis_psi.shape[0] + sum(f.size for f in self.families.values()))

    @property
    def memory_megabytes(self) -> float:
        total = self.basis_psi.numel() * self.basis_psi.element_size()
        for fam in self.families.values():
            total += fam.psi.numel() * fam.psi.element_size()
            total += sum(p.numel() * p.element_size() for p in fam.params.values())
        return float(total / 1024**2)

    def add_refined_seed(self, topology: str, psi: torch.Tensor, params: dict[str, torch.Tensor]) -> None:
        self.families[topology].append(psi, params)
        self.refined_atoms_added += 1

    def save(self, path: str | Path) -> None:
        payload = {
            "config": self.config.__dict__,
            "dtype": self.dtype,
            "build_seconds": self.build_seconds,
            "refined_atoms_added": self.refined_atoms_added,
            "basis_psi": self.basis_psi.detach().cpu(),
            "families": {
                name: {
                    "psi": fam.psi.detach().cpu(),
                    "params": {key: val.detach().cpu() for key, val in fam.params.items()},
                }
                for name, fam in self.families.items()
            },
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "auto", dtype: str = "auto") -> "HybridUQNBank3Q":
        chosen = _choose_device(device)
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        selected_dtype = _choose_complex_dtype(
            payload.get("dtype", "complex64") if dtype == "auto" else dtype, chosen
        )
        families: dict[str, FamilySeedBank] = {}
        for name, fam in payload["families"].items():
            families[name] = FamilySeedBank(
                topology=name,
                psi=fam["psi"].to(device=chosen, dtype=selected_dtype),
                params={key: val.to(device=chosen, dtype=selected_dtype) for key, val in fam["params"].items()},
            )
        return cls(
            families=families,
            basis_psi=payload["basis_psi"].to(device=chosen, dtype=selected_dtype),
            config=HybridBankConfig(**payload["config"]),
            device=str(chosen),
            dtype=_dtype_name(selected_dtype),
            build_seconds=float(payload.get("build_seconds", 0.0)),
            refined_atoms_added=int(payload.get("refined_atoms_added", 0)),
        )


@dataclass
class UQNGilbertHybridResult:
    distance: float
    squared_distance: float
    sigma: Array
    extended_distance: float
    identity_bound: float
    initial_reduced_distance: float
    iterations: int
    selected_iteration: int
    status: str
    reduced_history: list[float]
    extended_history: list[float]
    gaps: list[float]
    oracle_history: list[dict]
    elapsed_seconds: float
    bank_size: int
    refined_atoms_added: int
    device: str
    biseparable_distance: Optional[float] = None
    biseparable_status: Optional[str] = None
    selected_source: str = "extended_uqn"


# -----------------------------------------------------------------------------
# Input conversion and matrix utilities
# -----------------------------------------------------------------------------


def _choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chosen = torch.device(device)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False.")
    return chosen


def _choose_complex_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.complex64 if device.type == "cuda" else torch.complex128
    if dtype == "complex64":
        return torch.complex64
    if dtype == "complex128":
        return torch.complex128
    raise ValueError("dtype must be 'auto', 'complex64', or 'complex128'.")


def _dtype_name(dtype: torch.dtype) -> str:
    return "complex64" if dtype == torch.complex64 else "complex128"


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype == torch.complex64 else torch.float64


def ketbra_np(psi: Array) -> Array:
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm <= 1e-15:
        raise ValueError("Cannot construct a projector from a zero vector.")
    psi = psi / norm
    return np.outer(psi, psi.conj()).astype(np.complex128)


def hs_distance_np(a: Array, b: Array) -> float:
    delta = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(max(float(np.real(np.vdot(delta, delta))), 0.0)))


def qiskit_3q_to_internal_order(
    rho_qiskit: Array,
    qiskit_qubit_order: Sequence[int] = (0, 1, 2),
) -> Array:
    """Convert Qiskit little-endian order to internal |A B C> order."""
    rho_qiskit = np.asarray(rho_qiskit, dtype=np.complex128)
    if rho_qiskit.shape != (8, 8):
        raise ValueError(f"Expected an 8x8 density matrix; got {rho_qiskit.shape}.")
    order = tuple(int(q) for q in qiskit_qubit_order)
    if sorted(order) != [0, 1, 2]:
        raise ValueError("qiskit_qubit_order must be a permutation of (0, 1, 2).")
    indices: list[int] = []
    for internal_index in range(8):
        bits_abc = ((internal_index >> 2) & 1, (internal_index >> 1) & 1, internal_index & 1)
        qbits = {order[pos]: bits_abc[pos] for pos in range(3)}
        indices.append(sum(qbits[q] << q for q in range(3)))
    idx = np.asarray(indices, dtype=int)
    return rho_qiskit[np.ix_(idx, idx)].astype(np.complex128)


def as_density_matrix_3q(
    rho_like: Any,
    input_order: Literal["qiskit", "internal"] = "qiskit",
    qiskit_qubit_order: Sequence[int] = (0, 1, 2),
    psd_tol: float = 1e-8,
) -> Array:
    """Accept a precomputed three-qubit 8x8 density matrix or 8-vector."""
    if rho_like.__class__.__name__ == "QuantumCircuit":
        raise TypeError("Pass a precomputed three-qubit density matrix, not a circuit.")
    raw = rho_like.data if hasattr(rho_like, "data") else rho_like
    arr = np.asarray(raw, dtype=np.complex128)
    if arr.shape == (8,):
        rho = ketbra_np(arr)
    elif arr.shape == (8, 8):
        rho = arr
    else:
        raise ValueError(f"Expected shape (8,) or (8,8); got {arr.shape}.")
    if input_order == "qiskit":
        rho = qiskit_3q_to_internal_order(rho, qiskit_qubit_order)
    elif input_order != "internal":
        raise ValueError("input_order must be 'qiskit' or 'internal'.")
    rho = (rho + rho.conj().T) / 2.0
    tr = np.trace(rho)
    if abs(tr) <= 1e-14:
        raise ValueError("Density matrix has zero trace.")
    rho = rho / tr
    min_eig = float(np.min(np.linalg.eigvalsh(rho)))
    if min_eig < -psd_tol:
        raise ValueError(f"Density matrix is not PSD; min eigenvalue={min_eig:g}.")
    return rho.astype(np.complex128)


def extend_3q_state(rho: Array) -> Array:
    """Construct rho tensor I_8/8 in interleaved [A,a,B,b,C,c] order."""
    anc = np.eye(8, dtype=np.complex128) / 8.0
    raw = np.kron(rho, anc).reshape((2,) * 12)  # A B C a b c | A' B' C' a' b' c'
    axes = (0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11)
    return np.transpose(raw, axes).reshape(64, 64).astype(np.complex128)


def state_diagnostics(
    rho_like: Any,
    input_order: Literal["qiskit", "internal"] = "qiskit",
    qiskit_qubit_order: Sequence[int] = (0, 1, 2),
) -> dict:
    rho = as_density_matrix_3q(rho_like, input_order, qiskit_qubit_order)
    ident = np.eye(8, dtype=np.complex128) / 8.0
    return {
        "purity": float(np.real(np.trace(rho @ rho))),
        "distance_to_identity": hs_distance_np(rho, ident),
        "min_eigenvalue": float(np.min(np.linalg.eigvalsh(rho))),
    }


# -----------------------------------------------------------------------------
# UQN families and warm-start bank construction
# -----------------------------------------------------------------------------


def _rand_complex(
    shape: tuple[int, ...], device: torch.device, dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    real_dtype = _real_dtype(dtype)
    re = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
    im = torch.randn(shape, dtype=real_dtype, device=device, generator=generator)
    return torch.complex(re, im)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)


def _haar_unitaries4(
    count: int, device: torch.device, dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    z = _rand_complex((count, 4, 4), device, dtype, generator)
    q, r = torch.linalg.qr(z)
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    phase = diag / torch.abs(diag).clamp_min(1e-12)
    return (q * torch.conj(phase).unsqueeze(-2)).contiguous()


def _fixed_00(count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros((count, 2, 2), dtype=dtype, device=device)
    out[:, 0, 0] = 1.0
    return out


def _triangle_raw(ab: torch.Tensor, bc: torch.Tensor, ca: torch.Tensor) -> torch.Tensor:
    # AB[A0,B0], BC[B1,C0], CA[C1,A1] -> local grouping A0A1,B0B1,C0C1.
    return torch.einsum("sab,scd,sef->safbcde", ab, bc, ca).reshape(-1, 4, 4, 4)


def _apply_local(raw: torch.Tensor, ua: torch.Tensor, ub: torch.Tensor, uc: torch.Tensor) -> torch.Tensor:
    psi = torch.einsum("sai,sbj,sck,sijk->sabc", ua, ub, uc, raw).reshape(-1, 64)
    return _normalize_rows(psi)


def _build_psi(topology: str, params: dict[str, torch.Tensor]) -> torch.Tensor:
    if topology == "biseparable:A|BC":
        single = _normalize_rows(params["single"])
        pair = _normalize_rows(params["pair"]).reshape(-1, 4, 4)
        return _normalize_rows(torch.einsum("sa,sbc->sabc", single, pair).reshape(-1, 64))
    if topology == "biseparable:B|AC":
        single = _normalize_rows(params["single"])
        pair = _normalize_rows(params["pair"]).reshape(-1, 4, 4)
        return _normalize_rows(torch.einsum("sac,sb->sabc", pair, single).reshape(-1, 64))
    if topology == "biseparable:C|AB":
        single = _normalize_rows(params["single"])
        pair = _normalize_rows(params["pair"]).reshape(-1, 4, 4)
        return _normalize_rows(torch.einsum("sab,sc->sabc", pair, single).reshape(-1, 64))

    count = int(next(iter(params.values())).shape[0])
    fixed = _fixed_00(count, next(iter(params.values())).device, next(iter(params.values())).dtype)
    ua, ub, uc = params["UA"], params["UB"], params["UC"]
    if topology == "chain:omit_CA":
        raw = _triangle_raw(_normalize_rows(params["r1"]).reshape(-1, 2, 2),
                            _normalize_rows(params["r2"]).reshape(-1, 2, 2), fixed)
    elif topology == "chain:omit_AB":
        raw = _triangle_raw(fixed, _normalize_rows(params["r1"]).reshape(-1, 2, 2),
                            _normalize_rows(params["r2"]).reshape(-1, 2, 2))
    elif topology == "chain:omit_BC":
        raw = _triangle_raw(_normalize_rows(params["r1"]).reshape(-1, 2, 2), fixed,
                            _normalize_rows(params["r2"]).reshape(-1, 2, 2))
    elif topology == "triangle":
        raw = _triangle_raw(_normalize_rows(params["AB"]).reshape(-1, 2, 2),
                            _normalize_rows(params["BC"]).reshape(-1, 2, 2),
                            _normalize_rows(params["CA"]).reshape(-1, 2, 2))
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return _apply_local(raw, ua, ub, uc)


def _sample_family(
    topology: str,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> FamilySeedBank:
    if topology.startswith("biseparable"):
        params = {
            "single": _normalize_rows(_rand_complex((count, 4), device, dtype, generator)),
            "pair": _normalize_rows(_rand_complex((count, 16), device, dtype, generator)),
        }
    elif topology.startswith("chain"):
        params = {
            "r1": _normalize_rows(_rand_complex((count, 4), device, dtype, generator)),
            "r2": _normalize_rows(_rand_complex((count, 4), device, dtype, generator)),
            "UA": _haar_unitaries4(count, device, dtype, generator),
            "UB": _haar_unitaries4(count, device, dtype, generator),
            "UC": _haar_unitaries4(count, device, dtype, generator),
        }
    elif topology == "triangle":
        params = {
            "AB": _normalize_rows(_rand_complex((count, 4), device, dtype, generator)),
            "BC": _normalize_rows(_rand_complex((count, 4), device, dtype, generator)),
            "CA": _normalize_rows(_rand_complex((count, 4), device, dtype, generator)),
            "UA": _haar_unitaries4(count, device, dtype, generator),
            "UB": _haar_unitaries4(count, device, dtype, generator),
            "UC": _haar_unitaries4(count, device, dtype, generator),
        }
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return FamilySeedBank(topology=topology, psi=_build_psi(topology, params), params=params)


def build_hybrid_uqn_bank_3q(
    *,
    device: str = "auto",
    dtype: str = "auto",
    triangle_atoms: int = 32768,
    chain_atoms_per_topology: int = 8192,
    biseparable_atoms_per_cut: int = 8192,
    generation_batch_size: int = 4096,
    seed: Optional[int] = 1234,
) -> HybridUQNBank3Q:
    """Build a reusable parameterized warm-start bank on CPU or GPU."""
    started = time.perf_counter()
    chosen = _choose_device(device)
    cdtype = _choose_complex_dtype(dtype, chosen)
    generator = torch.Generator(device=chosen)
    if seed is not None:
        generator.manual_seed(int(seed))

    counts = {
        "biseparable:A|BC": int(biseparable_atoms_per_cut),
        "biseparable:B|AC": int(biseparable_atoms_per_cut),
        "biseparable:C|AB": int(biseparable_atoms_per_cut),
        "chain:omit_AB": int(chain_atoms_per_topology),
        "chain:omit_BC": int(chain_atoms_per_topology),
        "chain:omit_CA": int(chain_atoms_per_topology),
        "triangle": int(triangle_atoms),
    }
    families: dict[str, FamilySeedBank] = {}
    for topology, total in counts.items():
        pieces: list[FamilySeedBank] = []
        remaining = total
        while remaining > 0:
            take = min(remaining, int(generation_batch_size))
            pieces.append(_sample_family(topology, take, chosen, cdtype, generator))
            remaining -= take
        if pieces:
            params = {key: torch.cat([p.params[key] for p in pieces], dim=0).contiguous()
                      for key in pieces[0].params}
            families[topology] = FamilySeedBank(
                topology=topology,
                psi=torch.cat([p.psi for p in pieces], dim=0).contiguous(),
                params=params,
            )
        else:
            families[topology] = _sample_family(topology, 1, chosen, cdtype, generator)
            families[topology].psi = families[topology].psi[:0]
            for key in families[topology].params:
                families[topology].params[key] = families[topology].params[key][:0]

    basis = torch.eye(64, device=chosen, dtype=cdtype)
    cfg = HybridBankConfig(
        triangle_atoms=int(triangle_atoms),
        chain_atoms_per_topology=int(chain_atoms_per_topology),
        biseparable_atoms_per_cut=int(biseparable_atoms_per_cut),
        generation_batch_size=int(generation_batch_size),
        seed=seed,
    )
    return HybridUQNBank3Q(
        families=families,
        basis_psi=basis,
        config=cfg,
        device=str(chosen),
        dtype=_dtype_name(cdtype),
        build_seconds=time.perf_counter() - started,
    )


# -----------------------------------------------------------------------------
# Batched support search and continuous refinement
# -----------------------------------------------------------------------------


def _projector(psi: torch.Tensor) -> torch.Tensor:
    psi = psi / torch.linalg.vector_norm(psi).clamp_min(1e-12)
    return psi[:, None] * torch.conj(psi[None, :])


def _hs_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(a) * b))


def _hs_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _hs_inner(a - b, a - b)


def _support_values(psi: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.einsum("bi,ij,bj->b", torch.conj(psi), residual, psi))


def _slice_params(params: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, indices).detach().clone() for key, value in params.items()}


def _cat_params(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not a:
        return b
    return {key: torch.cat((a[key], b[key]), dim=0).contiguous() for key in a}


def _unitary_increment(base: torch.Tensor, raw_delta: torch.Tensor) -> torch.Tensor:
    """Unitary Cayley retraction about a warm-start unitary.

    For skew-Hermitian K, (I-K)^(-1)(I+K) is unitary. This is
    considerably cheaper than a batched matrix exponential for the small
    local 4x4 unitaries used in every oracle refinement step.
    """
    skew = 0.5 * (raw_delta - torch.conj(raw_delta.transpose(-1, -2)))
    eye = torch.eye(4, dtype=base.dtype, device=base.device).expand(base.shape[0], -1, -1)
    update = torch.linalg.solve(eye - skew, eye + skew)
    return update @ base


def _extract_seed(topology: str, built_params: dict[str, torch.Tensor], idx: int) -> dict[str, torch.Tensor]:
    return {key: value[idx].detach().clone() for key, value in built_params.items()}


def _refine_family(
    topology: str,
    seed_params: dict[str, torch.Tensor],
    residual: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
    """Continuously refine a batch of warm starts within one UQN topology."""
    if not seed_params or next(iter(seed_params.values())).shape[0] == 0:
        raise ValueError("Cannot refine an empty seed batch.")

    vector_keys = [key for key in seed_params if key not in {"UA", "UB", "UC"}]
    raw_vec = {key: torch.nn.Parameter(seed_params[key].detach().clone()) for key in vector_keys}
    base_u = {key: seed_params[key].detach().clone() for key in ("UA", "UB", "UC") if key in seed_params}
    delta_u = {
        key: torch.nn.Parameter(torch.zeros_like(base_u[key])) for key in base_u
    }
    optim_vars = list(raw_vec.values()) + list(delta_u.values())
    optimizer = torch.optim.Adam(optim_vars, lr=float(learning_rate)) if optim_vars else None

    def forward() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        built: dict[str, torch.Tensor] = {key: value for key, value in raw_vec.items()}
        for key, value in base_u.items():
            built[key] = _unitary_increment(value, delta_u[key])
        return _build_psi(topology, built), built

    with torch.no_grad():
        psi, built = forward()
        values = _support_values(psi, residual)
        best_val_tensor, best_idx_tensor = torch.max(values, dim=0)
        best_val = float(best_val_tensor.item())
        best_psi = psi[int(best_idx_tensor.item())].detach().clone()
        best_params = _extract_seed(topology, built, int(best_idx_tensor.item()))

    for _ in range(int(steps)):
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        psi, _ = forward()
        values = _support_values(psi, residual)
        loss = -torch.sum(values)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            psi_eval, built_eval = forward()
            vals_eval = _support_values(psi_eval, residual)
            val, idx = torch.max(vals_eval, dim=0)
            val_float = float(val.item())
            if val_float > best_val:
                best_val = val_float
                ii = int(idx.item())
                best_psi = psi_eval[ii].detach().clone()
                best_params = _extract_seed(topology, built_eval, ii)

    return best_psi, best_val, best_params


class HybridContinuousSupportOracle3Q:
    """Warm-start bank plus continuous GPU refinement for the UQN support problem."""

    def __init__(
        self,
        bank: HybridUQNBank3Q,
        *,
        topk_per_family: int = 4,
        fresh_starts_per_family: int = 1,
        families_to_refine: int = 4,
        refinement_steps: int = 35,
        learning_rate: float = 0.04,
        support_batch_size: int = 32768,
        cache_refined_atoms: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.bank = bank
        self.topk_per_family = int(topk_per_family)
        self.fresh_starts_per_family = int(fresh_starts_per_family)
        self.families_to_refine = int(families_to_refine)
        self.refinement_steps = int(refinement_steps)
        self.learning_rate = float(learning_rate)
        self.support_batch_size = int(support_batch_size)
        self.cache_refined_atoms = bool(cache_refined_atoms)
        self.generator = torch.Generator(device=bank.basis_psi.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def _top_seeds(self, residual: torch.Tensor) -> tuple[dict[str, dict], torch.Tensor, float]:
        summaries: dict[str, dict] = {}
        best_psi = self.bank.basis_psi[0]
        basis_values = torch.real(torch.diagonal(residual))
        basis_value, basis_idx = torch.max(basis_values, dim=0)
        best_value = float(basis_value.item())
        best_psi = self.bank.basis_psi[int(basis_idx.item())]

        for topology, fam in self.bank.families.items():
            if fam.size == 0:
                continue
            vals_parts: list[torch.Tensor] = []
            for start in range(0, fam.size, self.support_batch_size):
                vals_parts.append(_support_values(fam.psi[start:start + self.support_batch_size], residual))
            values = torch.cat(vals_parts, dim=0)
            k = min(self.topk_per_family, fam.size)
            top_values, top_indices = torch.topk(values, k=k)
            fam_best = float(top_values[0].item())
            summaries[topology] = {
                "best_bank_support": fam_best,
                "indices": top_indices,
                "params": _slice_params(fam.params, top_indices),
            }
            if fam_best > best_value:
                best_value = fam_best
                best_psi = fam.psi[int(top_indices[0].item())]
        return summaries, best_psi.detach().clone(), best_value

    def __call__(self, residual: torch.Tensor) -> tuple[torch.Tensor, dict]:
        residual = (residual + torch.conj(residual.T)) / 2.0
        summaries, best_psi, bank_best_value = self._top_seeds(residual)

        ranked = sorted(
            summaries.keys(), key=lambda name: summaries[name]["best_bank_support"], reverse=True
        )
        chosen_topologies = ranked[:max(1, min(self.families_to_refine, len(ranked)))]
        refined_supports: dict[str, float] = {}
        best_value = bank_best_value
        best_topology = "basis"
        best_params: Optional[dict[str, torch.Tensor]] = None

        for topology in chosen_topologies:
            params = summaries[topology]["params"]
            if self.fresh_starts_per_family > 0:
                fresh = _sample_family(
                    topology,
                    self.fresh_starts_per_family,
                    self.bank.basis_psi.device,
                    self.bank.basis_psi.dtype,
                    self.generator,
                )
                params = _cat_params(params, fresh.params)
            psi, value, final_params = _refine_family(
                topology, params, residual, self.refinement_steps, self.learning_rate
            )
            refined_supports[topology] = value
            if value > best_value:
                best_value = value
                best_psi = psi
                best_topology = topology
                best_params = final_params

        if self.cache_refined_atoms and best_params is not None:
            self.bank.add_refined_seed(best_topology, best_psi, best_params)

        return _projector(best_psi), {
            "support": best_value,
            "bank_best_support": bank_best_value,
            "refinement_gain": best_value - bank_best_value,
            "topology": best_topology,
            "refined_topologies": chosen_topologies,
            "refined_supports": refined_supports,
            "bank_size": self.bank.size,
        }


# -----------------------------------------------------------------------------
# Gilbert line search and Gilbert-with-memory update
# -----------------------------------------------------------------------------


@torch.no_grad()
def _reduce_visible_3q(rho6: torch.Tensor) -> torch.Tensor:
    tensor = rho6.reshape((2,) * 12)
    tensor = tensor.permute(0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11).reshape(8, 8, 8, 8)
    return torch.einsum("iaja->ij", tensor)


@torch.no_grad()
def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    idx = torch.arange(1, v.numel() + 1, dtype=v.dtype, device=v.device)
    mask = u - cssv / idx > 0
    if not bool(torch.any(mask)):
        return torch.ones_like(v) / v.numel()
    last = torch.nonzero(mask, as_tuple=False)[-1, 0]
    theta = cssv[last] / idx[last]
    return torch.clamp(v - theta, min=0.0)


@torch.no_grad()
def _memory_update(
    target: torch.Tensor,
    states: Sequence[torch.Tensor],
    max_steps: int = 350,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Gilbert-with-memory simplex/QP update implementing supplemental Eq. (10)."""
    A = torch.stack([s.reshape(-1) for s in states], dim=1)
    b = target.reshape(-1)
    Q = torch.real(torch.conj(A).T @ A)
    c = torch.real(torch.conj(A).T @ b)
    Q = (Q + Q.T) / 2.0
    L = 2.0 * torch.linalg.eigvalsh(Q)[-1].clamp_min(1e-12)
    x = torch.ones(len(states), dtype=Q.dtype, device=target.device) / len(states)
    y = x.clone()
    t = torch.tensor(1.0, dtype=Q.dtype, device=target.device)
    for _ in range(int(max_steps)):
        x_new = _project_simplex(y - 2.0 * (Q @ y - c) / L)
        if float(torch.linalg.vector_norm(x_new - x, ord=1).item()) <= tol:
            x = x_new
            break
        t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return torch.einsum("m,mij->ij", x.to(target.dtype), torch.stack(list(states), dim=0))


@torch.no_grad()
def _line_search(target: torch.Tensor, sigma: torch.Tensor, atom: torch.Tensor) -> torch.Tensor:
    direction = atom - sigma
    denom = _hs_inner(direction, direction).clamp_min(1e-15)
    alpha = torch.clamp(_hs_inner(target - sigma, direction) / denom, 0.0, 1.0)
    return sigma + alpha.to(target.dtype) * direction


@torch.no_grad()
def _exact_diagonal_reduction(rho3: torch.Tensor, diagonal_tol: float) -> Optional[torch.Tensor]:
    off = rho3 - torch.diag(torch.diagonal(rho3))
    if float(torch.sqrt(_hs_inner(off, off)).item()) <= diagonal_tol:
        return rho3.clone()
    return None



class ProgressiveHybridContinuousSupportOracle3Q:
    """Accelerated adaptive UQN support oracle.

    A large parameterized atom bank supplies warm starts.  Most Gilbert
    iterations refine only a small number of the highest-scoring topology
    families for a short optimization; periodically, or after a stall, a
    fuller refinement is used.  Refined candidates are cached only inside one
    solve unless ``cache_refined_atoms`` is explicitly enabled.
    """

    def __init__(
        self,
        bank: HybridUQNBank3Q,
        *,
        allowed_topologies: Sequence[str] = TOPOLOGIES,
        topk_per_family: int = 2,
        fresh_starts_per_family: int = 0,
        fast_families_to_refine: int = 1,
        fast_refinement_steps: int = 8,
        full_families_to_refine: int = 4,
        full_refinement_steps: int = 24,
        learning_rate: float = 0.04,
        support_batch_size: int = 32768,
        local_cache_size: int = 24,
        cache_refined_atoms: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.bank = bank
        allowed = tuple(t for t in allowed_topologies if t in bank.families)
        if not allowed:
            raise ValueError('allowed_topologies contains no available atom-bank families.')
        self.allowed_topologies = allowed
        self.topk_per_family = max(1, int(topk_per_family))
        self.fresh_starts_per_family = max(0, int(fresh_starts_per_family))
        self.fast_families_to_refine = max(1, int(fast_families_to_refine))
        self.fast_refinement_steps = max(0, int(fast_refinement_steps))
        self.full_families_to_refine = max(1, int(full_families_to_refine))
        self.full_refinement_steps = max(0, int(full_refinement_steps))
        self.learning_rate = float(learning_rate)
        self.support_batch_size = max(1, int(support_batch_size))
        self.local_cache_size = max(0, int(local_cache_size))
        self.cache_refined_atoms = bool(cache_refined_atoms)
        self.local_cache: dict[str, FamilySeedBank] = {}
        self.generator = torch.Generator(device=bank.basis_psi.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def _summarize_family(self, fam: FamilySeedBank, residual: torch.Tensor) -> Optional[dict]:
        if fam.size == 0:
            return None
        values_parts: list[torch.Tensor] = []
        for begin in range(0, fam.size, self.support_batch_size):
            values_parts.append(_support_values(fam.psi[begin:begin + self.support_batch_size], residual))
        values = torch.cat(values_parts, dim=0)
        k = min(self.topk_per_family, fam.size)
        top_values, top_indices = torch.topk(values, k=k)
        return {
            'best_bank_support': float(top_values[0].item()),
            'psi': fam.psi[int(top_indices[0].item())].detach().clone(),
            'params': _slice_params(fam.params, top_indices),
        }

    @torch.no_grad()
    def _top_seeds(self, residual: torch.Tensor) -> tuple[dict[str, dict], torch.Tensor, float]:
        basis_values = torch.real(torch.diagonal(residual))
        basis_value, basis_idx = torch.max(basis_values, dim=0)
        best_value = float(basis_value.item())
        best_psi = self.bank.basis_psi[int(basis_idx.item())].detach().clone()
        summaries: dict[str, dict] = {}

        for topology in self.allowed_topologies:
            summary = self._summarize_family(self.bank.families[topology], residual)
            if summary is None:
                continue
            # Within-solve refined candidates are cheap, highly relevant warm starts.
            cached = self.local_cache.get(topology)
            if cached is not None and cached.size > 0:
                cached_summary = self._summarize_family(cached, residual)
                if cached_summary is not None:
                    if cached_summary['best_bank_support'] > summary['best_bank_support']:
                        summary['psi'] = cached_summary['psi']
                        summary['best_bank_support'] = cached_summary['best_bank_support']
                    summary['params'] = _cat_params(summary['params'], cached_summary['params'])
            summaries[topology] = summary
            if summary['best_bank_support'] > best_value:
                best_value = summary['best_bank_support']
                best_psi = summary['psi']
        return summaries, best_psi, best_value

    def _add_local_seed(self, topology: str, psi: torch.Tensor, params: dict[str, torch.Tensor]) -> None:
        if self.local_cache_size <= 0:
            return
        if topology not in self.local_cache:
            self.local_cache[topology] = FamilySeedBank(
                topology=topology,
                psi=psi.detach().reshape(1, 64).contiguous(),
                params={k: v.detach().reshape((1,) + tuple(v.shape)).contiguous() for k, v in params.items()},
            )
        else:
            self.local_cache[topology].append(psi, params)
        fam = self.local_cache[topology]
        if fam.size > self.local_cache_size:
            fam.psi = fam.psi[-self.local_cache_size:].contiguous()
            fam.params = {k: v[-self.local_cache_size:].contiguous() for k, v in fam.params.items()}

    def __call__(self, residual: torch.Tensor, *, full: bool = False) -> tuple[torch.Tensor, dict]:
        residual = (residual + torch.conj(residual.T)) / 2.0
        summaries, best_psi, warm_best = self._top_seeds(residual)
        ranked = sorted(summaries, key=lambda t: summaries[t]['best_bank_support'], reverse=True)
        count = self.full_families_to_refine if full else self.fast_families_to_refine
        steps = self.full_refinement_steps if full else self.fast_refinement_steps
        chosen = ranked[:max(1, min(count, len(ranked)))]
        best_value = warm_best
        best_topology = 'basis'
        best_params: Optional[dict[str, torch.Tensor]] = None
        refined_supports: dict[str, float] = {}

        for topology in chosen:
            params = summaries[topology]['params']
            if self.fresh_starts_per_family > 0:
                fresh = _sample_family(
                    topology, self.fresh_starts_per_family,
                    self.bank.basis_psi.device, self.bank.basis_psi.dtype, self.generator,
                )
                params = _cat_params(params, fresh.params)
            psi, value, final_params = _refine_family(
                topology, params, residual, steps, self.learning_rate
            )
            refined_supports[topology] = float(value)
            self._add_local_seed(topology, psi, final_params)
            if value > best_value:
                best_value = float(value)
                best_psi = psi
                best_topology = topology
                best_params = final_params

        if self.cache_refined_atoms and best_params is not None and best_topology != 'basis':
            self.bank.add_refined_seed(best_topology, best_psi, best_params)

        return _projector(best_psi), {
            'support': float(best_value),
            'warm_start_best_support': float(warm_best),
            'refinement_gain': float(best_value - warm_best),
            'topology': best_topology,
            'refined_topologies': chosen,
            'refined_supports': refined_supports,
            'full_refinement': bool(full),
            'allowed_topologies': list(self.allowed_topologies),
            'bank_size': self.bank.size,
        }


def gilbert_uqn_distance_3q(
    rho_like: Any,
    *,
    bank: Optional[HybridUQNBank3Q] = None,
    input_order: Literal['qiskit', 'internal'] = 'qiskit',
    qiskit_qubit_order: Sequence[int] = (0, 1, 2),
    device: str = 'auto',
    dtype: str = 'auto',
    max_iter: int = 36,
    memory: int = 24,
    topk_per_family: int = 2,
    fresh_starts_per_family: int = 0,
    fast_families_to_refine: int = 1,
    fast_refinement_steps: int = 8,
    families_to_refine: int = 4,
    refinement_steps: int = 24,
    full_refine_every: int = 8,
    memory_update_every: int = 4,
    local_cache_size: int = 24,
    learning_rate: float = 0.04,
    support_batch_size: int = 32768,
    cache_refined_atoms: bool = False,
    retry_full_refinement_on_stall: bool = True,
    skip_extended_biseparable_topologies: bool = True,
    gap_tol: float = 1e-8,
    distance_tol: float = 1e-8,
    qp_steps: int = 250,
    qp_tol: float = 1e-8,
    diagonal_tol: float = 1e-10,
    include_basis_active_set: bool = True,
    use_biseparable_baseline: bool = True,
    return_if_biseparable_within_tol: bool = True,
    bs_max_iter: int = 80,
    bs_memory: int = 40,
    bs_restarts: int = 4,
    bs_sweeps: int = 8,
    bs_initial_random_atoms: int = 8,
    bs_qp_steps: int = 500,
    bs_qp_tol: float = 1e-10,
    seed: Optional[int] = 1234,
) -> UQNGilbertHybridResult:
    """Return an upper-bound estimate for the UQN distance of a 3-qubit state.

    The function implements a fast progressive hybrid oracle.  A direct
    biseparable candidate is retained as a mandatory UQN subset bound.  If it
    does not settle the input, the six-qubit extension search refines only the
    chain/triangle topologies by default, because a direct visible-space
    biseparable candidate is already available.  The full memory/simplex QP is
    evaluated periodically and at the final iterate.
    """
    started = time.perf_counter()
    rho_np = as_density_matrix_3q(rho_like, input_order, qiskit_qubit_order)
    identity_bound_np = hs_distance_np(rho_np, np.eye(8, dtype=np.complex128) / 8.0)

    # Exact product-basis mixture path: no bank or baseline solve is necessary.
    offdiag = rho_np - np.diag(np.diag(rho_np))
    if hs_distance_np(offdiag, np.zeros_like(offdiag)) <= diagonal_tol:
        return UQNGilbertHybridResult(
            distance=0.0, squared_distance=0.0, sigma=rho_np.copy(),
            extended_distance=0.0, identity_bound=identity_bound_np,
            initial_reduced_distance=0.0, iterations=0, selected_iteration=0,
            status='exact_diagonal_uqn_decomposition', reduced_history=[0.0],
            extended_history=[0.0], gaps=[], oracle_history=[],
            elapsed_seconds=time.perf_counter() - started,
            bank_size=0 if bank is None else bank.size, refined_atoms_added=0,
            device='cpu', biseparable_distance=0.0,
            biseparable_status='diagonal_product_mixture', selected_source='exact_diagonal',
        )

    bs_result: Optional[BiseparableGilbertResult] = None
    if use_biseparable_baseline:
        bs_result = gilbert_biseparable_distance_3q(
            rho_np, input_order='internal', max_iter=bs_max_iter, memory=bs_memory,
            restarts=bs_restarts, sweeps=bs_sweeps,
            initial_random_atoms=bs_initial_random_atoms,
            include_basis_active_set=True, gap_tol=min(gap_tol, 1e-10),
            distance_tol=distance_tol, qp_steps=bs_qp_steps, qp_tol=bs_qp_tol, seed=seed,
        )
        if return_if_biseparable_within_tol and bs_result.distance <= distance_tol:
            return UQNGilbertHybridResult(
                distance=float(bs_result.distance), squared_distance=float(bs_result.squared_distance),
                sigma=bs_result.sigma, extended_distance=float(bs_result.distance / np.sqrt(8.0)),
                identity_bound=identity_bound_np, initial_reduced_distance=float(bs_result.distance),
                iterations=bs_result.iterations, selected_iteration=0,
                status='biseparable_uqn_bound_reached', reduced_history=list(bs_result.history),
                extended_history=[float(bs_result.distance / np.sqrt(8.0))],
                gaps=list(bs_result.gaps),
                oracle_history=[{'baseline': 'biseparable', 'status': bs_result.status}],
                elapsed_seconds=time.perf_counter() - started,
                bank_size=0 if bank is None else bank.size, refined_atoms_added=0,
                device='cpu', biseparable_distance=float(bs_result.distance),
                biseparable_status=bs_result.status, selected_source='biseparable_baseline',
            )

    if bank is None:
        bank = build_hybrid_uqn_bank_3q(device=device, dtype=dtype, seed=seed)
    dev = bank.basis_psi.device
    cdtype = bank.basis_psi.dtype
    rho3 = torch.as_tensor(rho_np, device=dev, dtype=cdtype)
    target = torch.as_tensor(extend_3q_state(rho_np), device=dev, dtype=cdtype)
    identity3 = torch.eye(8, device=dev, dtype=cdtype) / 8.0
    identity_bound = float(torch.sqrt(_hs_sq(rho3, identity3)).item())

    basis_atoms = [_projector(bank.basis_psi[i]) for i in range(64)]
    fixed_atoms: list[torch.Tensor] = basis_atoms if include_basis_active_set else []
    start_candidates: list[tuple[str, torch.Tensor]] = [
        ('identity', torch.eye(64, device=dev, dtype=cdtype) / 64.0)
    ]
    if include_basis_active_set:
        basis_sigma = _memory_update(target, basis_atoms, max_steps=qp_steps, tol=qp_tol)
        start_candidates.append(('basis_active_set', basis_sigma))

    baseline_fixed_atoms: list[torch.Tensor] = []
    if bs_result is not None:
        bs_ext = torch.as_tensor(extend_3q_state(bs_result.sigma), device=dev, dtype=cdtype)
        baseline_fixed_atoms.append(bs_ext)
        start_candidates.append(('biseparable_baseline', bs_ext))

    start_source, sigma = min(start_candidates, key=lambda item: float(_hs_sq(target, item[1]).item()))
    sigma = (sigma + torch.conj(sigma.T)) / 2.0
    sigma3 = _reduce_visible_3q(sigma)
    initial_reduced = float(torch.sqrt(_hs_sq(rho3, sigma3)).item())
    initial_extended = float(torch.sqrt(_hs_sq(target, sigma)).item())

    best_distance = initial_reduced
    best_sigma3 = sigma3.detach().clone()
    selected_source = start_source
    selected_iteration = 0
    if bs_result is not None and bs_result.distance < best_distance:
        best_distance = float(bs_result.distance)
        best_sigma3 = torch.as_tensor(bs_result.sigma, device=dev, dtype=cdtype)
        selected_source = 'biseparable_baseline'

    reduced_history = [float(best_distance)]
    extended_history = [initial_extended]
    dynamic_atoms: list[torch.Tensor] = []
    gaps: list[float] = []
    oracle_history: list[dict] = []
    status = 'max_iter_reached'
    iterations_done = 0
    start_added = bank.refined_atoms_added

    if bs_result is not None and skip_extended_biseparable_topologies:
        allowed_topologies = tuple(t for t in TOPOLOGIES if not t.startswith('biseparable:'))
    else:
        allowed_topologies = TOPOLOGIES

    oracle = ProgressiveHybridContinuousSupportOracle3Q(
        bank, allowed_topologies=allowed_topologies,
        topk_per_family=topk_per_family,
        fresh_starts_per_family=fresh_starts_per_family,
        fast_families_to_refine=fast_families_to_refine,
        fast_refinement_steps=fast_refinement_steps,
        full_families_to_refine=families_to_refine,
        full_refinement_steps=refinement_steps,
        learning_rate=learning_rate, support_batch_size=support_batch_size,
        local_cache_size=local_cache_size, cache_refined_atoms=cache_refined_atoms, seed=seed,
    )

    for iteration in range(1, int(max_iter) + 1):
        iterations_done = iteration
        residual = target - sigma
        do_full = full_refine_every <= 1 or iteration == 1 or (
            full_refine_every > 0 and iteration % int(full_refine_every) == 0
        )
        atom, info = oracle(residual, full=do_full)
        gap = float(_hs_inner(residual, atom - sigma).item())

        if gap <= gap_tol and retry_full_refinement_on_stall and not do_full:
            retry_atom, retry_info = oracle(residual, full=True)
            retry_gap = float(_hs_inner(residual, retry_atom - sigma).item())
            if retry_gap > gap:
                atom, info, gap = retry_atom, retry_info, retry_gap
            info = dict(info)
            info['full_refinement_retry'] = True
        else:
            info = dict(info)
            info['full_refinement_retry'] = False

        info.update({'iteration': iteration, 'gap': float(gap)})
        gaps.append(float(gap))
        oracle_history.append(info)
        if gap <= gap_tol:
            status = 'oracle_stalled_inconclusive'
            break

        dynamic_atoms.append(atom.detach())
        candidates: list[torch.Tensor] = [sigma, _line_search(target, sigma, atom)]
        do_memory = memory > 0 and (
            memory_update_every <= 1 or iteration % int(memory_update_every) == 0
        )
        if do_memory:
            states = fixed_atoms + baseline_fixed_atoms + [sigma] + dynamic_atoms[-int(memory):]
            candidates.append(_memory_update(target, states, max_steps=qp_steps, tol=qp_tol))

        sigma_new = min(candidates, key=lambda s: float(_hs_sq(target, s).item()))
        sigma = (sigma_new + torch.conj(sigma_new.T)) / 2.0
        sigma3 = _reduce_visible_3q(sigma)
        reduced = float(torch.sqrt(_hs_sq(rho3, sigma3)).item())
        extended = float(torch.sqrt(_hs_sq(target, sigma)).item())
        reduced_history.append(reduced)
        extended_history.append(extended)
        if reduced < best_distance:
            best_distance = reduced
            best_sigma3 = sigma3.detach().clone()
            selected_iteration = iteration
            selected_source = 'extended_uqn'
        if best_distance <= distance_tol:
            status = 'distance_tolerance_reached'
            break

    # One final memory projection keeps page-9 refinement present even when it
    # was skipped on intervening fast iterations.
    if memory > 0 and dynamic_atoms:
        states = fixed_atoms + baseline_fixed_atoms + [sigma] + dynamic_atoms[-int(memory):]
        final_sigma = _memory_update(target, states, max_steps=qp_steps, tol=qp_tol)
        if float(_hs_sq(target, final_sigma).item()) < float(_hs_sq(target, sigma).item()):
            sigma = (final_sigma + torch.conj(final_sigma.T)) / 2.0
            sigma3 = _reduce_visible_3q(sigma)
            reduced = float(torch.sqrt(_hs_sq(rho3, sigma3)).item())
            extended = float(torch.sqrt(_hs_sq(target, sigma)).item())
            reduced_history.append(reduced)
            extended_history.append(extended)
            if reduced < best_distance:
                best_distance = reduced
                best_sigma3 = sigma3.detach().clone()
                selected_iteration = iterations_done
                selected_source = 'extended_uqn_final_memory'

    if selected_source == 'biseparable_baseline' and status == 'max_iter_reached':
        status = 'biseparable_baseline_best_max_iter_reached'
    elif selected_source == 'biseparable_baseline' and status == 'oracle_stalled_inconclusive':
        status = 'biseparable_baseline_best_oracle_stalled'

    if bs_result is not None and best_distance > bs_result.distance + 5e-7:
        raise RuntimeError('Internal error: UQN result is worse than retained biseparable baseline.')

    return UQNGilbertHybridResult(
        distance=float(best_distance), squared_distance=float(best_distance * best_distance),
        sigma=best_sigma3.detach().cpu().numpy().astype(np.complex128),
        extended_distance=float(extended_history[-1]), identity_bound=identity_bound,
        initial_reduced_distance=float(initial_reduced), iterations=iterations_done,
        selected_iteration=selected_iteration, status=status, reduced_history=reduced_history,
        extended_history=extended_history, gaps=gaps, oracle_history=oracle_history,
        elapsed_seconds=time.perf_counter() - started, bank_size=bank.size,
        refined_atoms_added=bank.refined_atoms_added - start_added, device=str(dev),
        biseparable_distance=None if bs_result is None else float(bs_result.distance),
        biseparable_status=None if bs_result is None else bs_result.status,
        selected_source=selected_source,
    )

# -----------------------------------------------------------------------------
# Benchmark helpers
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


def run_white_noise_benchmark(
    bank: HybridUQNBank3Q,
    family: Literal["GHZ3", "W3"] = "GHZ3",
    p_values: Sequence[float] = (0.0, 0.2, 0.4, 0.5, 0.55, 0.57, 0.6),
    **kwargs,
) -> list[UQNGilbertHybridResult]:
    base = ghz3_density() if family == "GHZ3" else w3_density()
    results: list[UQNGilbertHybridResult] = []
    for i, p in enumerate(p_values):
        call_kwargs = dict(kwargs)
        if call_kwargs.get("seed") is not None:
            call_kwargs["seed"] = int(call_kwargs["seed"]) + i
        result = gilbert_uqn_distance_3q(
            white_noise_mixture(base, float(p)), bank=bank, input_order="internal", **call_kwargs
        )
        print(
            f"{family} p={p:.4g}: D={result.distance:.10g}, "
            f"status={result.status}, iters={result.iterations}, "
            f"refined={result.refined_atoms_added}, time={result.elapsed_seconds:.3g}s"
        )
        results.append(result)
    return results

# -----------------------------------------------------------------------------
# Direct 3-qubit biseparable baseline / safeguard
# -----------------------------------------------------------------------------


@dataclass
class BiseparableGilbertResult:
    """Upper-bound estimate for distance to the 3-qubit biseparable set."""
    distance: float
    squared_distance: float
    sigma: Array
    iterations: int
    converged: bool
    status: str
    history: list[float]
    gaps: list[float]
    last_oracle_info: dict
    elapsed_seconds: float


def _bs_hs_inner(a: Array, b: Array) -> float:
    return float(np.real(np.vdot(np.asarray(a), np.asarray(b))))


def _bs_product_state(
    a: Array,
    cut_a: tuple[int, ...],
    b: Array,
    cut_b: tuple[int, ...],
) -> Array:
    """Construct |a>_A tensor |b>_B in internal |A B C> qubit order."""
    order = cut_a + cut_b
    tensor = np.tensordot(
        np.asarray(a, dtype=np.complex128).reshape((2,) * len(cut_a)),
        np.asarray(b, dtype=np.complex128).reshape((2,) * len(cut_b)),
        axes=0,
    )
    perm = [order.index(q) for q in range(3)]
    psi = np.transpose(tensor, perm).reshape(8)
    return psi / np.linalg.norm(psi)


def _bs_embedding(
    free_cut: tuple[int, ...],
    fixed_cut: tuple[int, ...],
    fixed_vec: Array,
) -> Array:
    """Return V such that V @ x embeds |x>_free tensor |fixed>_fixed."""
    dim_free = 2 ** len(free_cut)
    V = np.empty((8, dim_free), dtype=np.complex128)
    for k in range(dim_free):
        basis = np.zeros(dim_free, dtype=np.complex128)
        basis[k] = 1.0
        V[:, k] = _bs_product_state(basis, free_cut, fixed_vec, fixed_cut)
    return V


def _bs_top_eigen_schmidt_seed(
    G: Array,
    cut_a: tuple[int, ...],
    cut_b: tuple[int, ...],
) -> tuple[Array, Array]:
    """Seed alternating optimization from the leading eigenvector of G."""
    _, vecs = np.linalg.eigh((G + G.conj().T) / 2.0)
    phi = vecs[:, -1].reshape((2, 2, 2))
    matrix = np.transpose(phi, cut_a + cut_b).reshape(2 ** len(cut_a), 2 ** len(cut_b))
    U, _, Vh = np.linalg.svd(matrix, full_matrices=False)
    # Product amplitudes satisfy M_ij = a_i b_j; use Vh row directly.
    a = U[:, 0].astype(np.complex128)
    b = Vh[0, :].astype(np.complex128)
    return a, b


def _bs_random_vector(dim: int, rng: np.random.Generator) -> Array:
    z = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return (z / np.linalg.norm(z)).astype(np.complex128)


def _bs_random_atom(rng: np.random.Generator) -> Array:
    cuts = (((0,), (1, 2)), ((1,), (0, 2)), ((2,), (0, 1)))
    cut_a, cut_b = cuts[int(rng.integers(0, len(cuts)))]
    a = _bs_random_vector(2 ** len(cut_a), rng)
    b = _bs_random_vector(2 ** len(cut_b), rng)
    return ketbra_np(_bs_product_state(a, cut_a, b, cut_b))


def _bs_computational_basis_atoms() -> list[Array]:
    atoms: list[Array] = []
    for k in range(8):
        v = np.zeros(8, dtype=np.complex128)
        v[k] = 1.0
        atoms.append(ketbra_np(v))
    return atoms


def _bs_project_simplex(v: NDArray[np.float64]) -> NDArray[np.float64]:
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.arange(1, len(v) + 1)
    cond = u - cssv / idx > 0
    if not np.any(cond):
        return np.ones_like(v) / len(v)
    rho_idx = idx[cond][-1]
    theta = cssv[cond][-1] / rho_idx
    return np.maximum(v - theta, 0.0)


def _bs_memory_update(
    target: Array,
    states: Sequence[Array],
    max_steps: int = 1200,
    tol: float = 1e-12,
) -> Array:
    """Simplex/QP projection used by Gilbert with memory in direct 8x8 space."""
    A = np.stack([np.asarray(s, dtype=np.complex128).reshape(-1) for s in states], axis=1)
    b = np.asarray(target, dtype=np.complex128).reshape(-1)
    Q = np.real(A.conj().T @ A)
    c = np.real(A.conj().T @ b)
    Q = (Q + Q.T) / 2.0
    L = 2.0 * max(float(np.linalg.eigvalsh(Q)[-1]), 1e-15)
    x = np.ones(len(states), dtype=float) / len(states)
    y = x.copy()
    t = 1.0
    for _ in range(int(max_steps)):
        grad = 2.0 * (Q @ y - c)
        x_new = _bs_project_simplex(y - grad / L)
        if np.linalg.norm(x_new - x, ord=1) <= tol:
            x = x_new
            break
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    sigma = sum(float(w) * s for w, s in zip(x, states))
    sigma = (sigma + sigma.conj().T) / 2.0
    return (sigma / np.trace(sigma)).astype(np.complex128)


def _bs_line_search(target: Array, sigma: Array, atom: Array) -> Array:
    direction = atom - sigma
    denom = _bs_hs_inner(direction, direction)
    if denom <= 1e-15:
        return sigma
    alpha = np.clip(_bs_hs_inner(target - sigma, direction) / denom, 0.0, 1.0)
    out = sigma + float(alpha) * direction
    out = (out + out.conj().T) / 2.0
    return (out / np.trace(out)).astype(np.complex128)


class BiseparableSupportOracle3Q:
    """
    Fast support oracle over pure 3-qubit biseparable states.

    For each cut A|BC, B|AC and C|AB it uses alternating exact eigenvector
    maximizations from deterministic and random initializations.
    """

    def __init__(self, restarts: int = 16, sweeps: int = 16, seed: Optional[int] = 1234) -> None:
        self.cuts = (((0,), (1, 2)), ((1,), (0, 2)), ((2,), (0, 1)))
        self.restarts = int(restarts)
        self.sweeps = int(sweeps)
        self.rng = np.random.default_rng(seed)

    def __call__(self, G: Array) -> tuple[Array, dict]:
        G = (np.asarray(G, dtype=np.complex128) + np.asarray(G, dtype=np.complex128).conj().T) / 2.0
        best_value = -np.inf
        best_atom: Optional[Array] = None
        best_cut: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None

        for cut_a, cut_b in self.cuts:
            dim_a = 2 ** len(cut_a)
            dim_b = 2 ** len(cut_b)
            starts: list[tuple[Array, Array]] = [_bs_top_eigen_schmidt_seed(G, cut_a, cut_b)]

            # Basis starts make diagonal/classical directions exact and cost almost nothing.
            for ia in range(dim_a):
                for ib in range(dim_b):
                    a = np.zeros(dim_a, dtype=np.complex128)
                    b = np.zeros(dim_b, dtype=np.complex128)
                    a[ia] = 1.0
                    b[ib] = 1.0
                    starts.append((a, b))

            for _ in range(self.restarts):
                starts.append((_bs_random_vector(dim_a, self.rng), _bs_random_vector(dim_b, self.rng)))

            for a, b in starts:
                for _ in range(self.sweeps):
                    Vb = _bs_embedding(cut_a, cut_b, b)
                    GA = (Vb.conj().T @ G @ Vb)
                    GA = (GA + GA.conj().T) / 2.0
                    _, va = np.linalg.eigh(GA)
                    a = va[:, -1]

                    Va = _bs_embedding(cut_b, cut_a, a)
                    GB = (Va.conj().T @ G @ Va)
                    GB = (GB + GB.conj().T) / 2.0
                    _, vb = np.linalg.eigh(GB)
                    b = vb[:, -1]

                psi = _bs_product_state(a, cut_a, b, cut_b)
                value = float(np.real(np.vdot(psi, G @ psi)))
                if value > best_value:
                    best_value = value
                    best_atom = ketbra_np(psi)
                    best_cut = (cut_a, cut_b)

        if best_atom is None:
            raise RuntimeError("Biseparable support oracle failed to produce an atom.")

        return best_atom, {
            "oracle": "BiseparableSupportOracle3Q",
            "expectation": float(best_value),
            "cut": best_cut,
            "restarts": self.restarts,
            "sweeps": self.sweeps,
        }


def gilbert_biseparable_distance_3q(
    rho_like: Any,
    *,
    input_order: Literal["qiskit", "internal"] = "qiskit",
    qiskit_qubit_order: Sequence[int] = (0, 1, 2),
    max_iter: int = 200,
    memory: int = 80,
    restarts: int = 16,
    sweeps: int = 16,
    initial_random_atoms: int = 24,
    include_basis_active_set: bool = True,
    gap_tol: float = 1e-10,
    distance_tol: float = 1e-10,
    qp_steps: int = 1200,
    qp_tol: float = 1e-12,
    seed: Optional[int] = 1234,
) -> BiseparableGilbertResult:
    """
    Estimate D from a 3-qubit state to the biseparable convex set.

    This is not the full UQN distance. It is a fast valid network-state
    baseline, because biseparable states form a subset of UQN/network states.
    Consequently a full UQN routine should not report a worse upper bound than
    this candidate when both are evaluated on the same input matrix.
    """
    started = time.perf_counter()
    rho = as_density_matrix_3q(rho_like, input_order, qiskit_qubit_order)
    oracle = BiseparableSupportOracle3Q(restarts=restarts, sweeps=sweeps, seed=seed)
    rng = np.random.default_rng(seed)

    fixed_atoms = _bs_computational_basis_atoms() if include_basis_active_set else []
    dynamic_atoms = [_bs_random_atom(rng) for _ in range(int(initial_random_atoms))]
    initial_states: list[Array] = [np.eye(8, dtype=np.complex128) / 8.0] + fixed_atoms + dynamic_atoms
    sigma = _bs_memory_update(rho, initial_states, max_steps=qp_steps, tol=qp_tol)

    history = [hs_distance_np(rho, sigma)]
    best_distance = history[0]
    best_sigma = sigma.copy()
    gaps: list[float] = []
    last_info: dict = {}
    status = "max_iter_reached"
    converged = False

    if best_distance <= distance_tol:
        return BiseparableGilbertResult(
            distance=0.0,
            squared_distance=0.0,
            sigma=best_sigma,
            iterations=0,
            converged=True,
            status="distance_tolerance_reached",
            history=history,
            gaps=gaps,
            last_oracle_info={"status": "initial_active_set_reached_distance_tol"},
            elapsed_seconds=time.perf_counter() - started,
        )

    for iteration in range(1, int(max_iter) + 1):
        residual = rho - sigma
        atom, info = oracle(residual)
        gap = _bs_hs_inner(residual, atom - sigma)
        gaps.append(float(gap))
        last_info = dict(info)
        last_info.update({"iteration": iteration, "gap": float(gap), "distance_before": history[-1]})

        if gap <= gap_tol:
            status = "oracle_stalled_or_converged"
            converged = True
            break

        dynamic_atoms.append(atom)
        line_candidate = _bs_line_search(rho, sigma, atom)

        if memory > 1:
            states = fixed_atoms + [sigma] + dynamic_atoms[-int(memory):]
            memory_candidate = _bs_memory_update(rho, states, max_steps=qp_steps, tol=qp_tol)
            # Numerical safeguard: never reject a better elementary Gilbert update.
            sigma_new = min(
                (line_candidate, memory_candidate),
                key=lambda x: hs_distance_np(rho, x),
            )
        else:
            sigma_new = line_candidate

        new_distance = hs_distance_np(rho, sigma_new)
        history.append(float(new_distance))
        sigma = sigma_new

        last_info["distance_after"] = float(new_distance)
        last_info["distance_improvement"] = float(history[-2] - history[-1])

        if new_distance < best_distance:
            best_distance = float(new_distance)
            best_sigma = sigma.copy()

        if best_distance <= distance_tol:
            status = "distance_tolerance_reached"
            converged = True
            break

    return BiseparableGilbertResult(
        distance=float(best_distance),
        squared_distance=float(best_distance * best_distance),
        sigma=best_sigma,
        iterations=len(history) - 1,
        converged=converged,
        status=status,
        history=history,
        gaps=gaps,
        last_oracle_info=last_info,
        elapsed_seconds=time.perf_counter() - started,
    )

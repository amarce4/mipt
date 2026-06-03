"""
Local dimension-reduction channels for six-qubit tripartite MIPT states.

Input partition:
    A = (q0, q1), B = (q2, q3), C = (q4, q5)

The useful routine is:
    compress_six_qubit_to_qutrits(rho, qiskit_order=True)

It applies, independently at A, B and C, the unital CPTP 4 -> 3 channel
described in Eqs. (15)-(17) of arXiv:2512.11118, using each party's local
density-matrix eigenbasis. The output is a 27 x 27 tripartite-qutrit state
appropriate for:
    ring_inflation_tmax(rho_qutrit, party_dims=(3, 3, 3), ...)

Because the channel is local and maps I_4/4 -> I_3/3, any inflation GNME
certificate for the qutrit output is a valid certificate for the original
six-qubit state and its white-noise robustness line.
"""

from __future__ import annotations

from itertools import product
from math import prod, sqrt
from typing import Sequence

import numpy as np


def as_density_matrix(rho: np.ndarray, dimension: int = 64) -> np.ndarray:
    """Return a normalized Hermitian density matrix from vector or matrix input."""
    arr = np.asarray(rho, dtype=np.complex128)
    if arr.ndim == 1:
        if arr.shape != (dimension,):
            raise ValueError(f"Expected statevector shape {(dimension,)}, got {arr.shape}.")
        norm = float(np.vdot(arr, arr).real)
        if norm <= 0.0:
            raise ValueError("The statevector has zero norm.")
        arr = arr / np.sqrt(norm)
        arr = np.outer(arr, arr.conj())
    elif arr.shape != (dimension, dimension):
        raise ValueError(
            f"Expected density matrix shape {(dimension, dimension)}, got {arr.shape}."
        )

    arr = (arr + arr.conj().T) / 2
    tr = np.trace(arr).real
    if tr <= 0:
        raise ValueError("The density matrix has non-positive trace.")
    return arr / tr


def qiskit_to_q0_first(rho: np.ndarray, n_qubits: int = 6) -> np.ndarray:
    """
    Convert Qiskit's |q_(n-1)...q_0> array ordering to |q_0...q_(n-1)>.

    Accepts a statevector or density matrix.
    """
    arr = np.asarray(rho, dtype=np.complex128)
    reverse = tuple(reversed(range(n_qubits)))

    if arr.ndim == 1:
        if arr.size != 2**n_qubits:
            raise ValueError("Statevector dimension does not match n_qubits.")
        return arr.reshape((2,) * n_qubits).transpose(reverse).reshape(-1)

    if arr.ndim == 2:
        if arr.shape != (2**n_qubits, 2**n_qubits):
            raise ValueError("Density-matrix dimension does not match n_qubits.")
        axes = reverse + tuple(n_qubits + i for i in reverse)
        return arr.reshape((2,) * (2 * n_qubits)).transpose(axes).reshape(arr.shape)

    raise ValueError("rho must be a statevector or density matrix.")


def partial_trace_keep(
    rho: np.ndarray,
    dims: Sequence[int],
    keep: Sequence[int],
) -> np.ndarray:
    """Partial trace, keeping subsystems in the order given by keep."""
    dims = list(int(d) for d in dims)
    keep = list(int(k) for k in keep)
    labels = list(range(len(dims)))
    tensor = np.asarray(rho).reshape(tuple(dims) + tuple(dims))

    for label in reversed(labels.copy()):
        if label not in keep:
            axis = labels.index(label)
            n_current = len(labels)
            tensor = np.trace(tensor, axis1=axis, axis2=axis + n_current)
            dims.pop(axis)
            labels.pop(axis)

    if labels != keep:
        perm = [labels.index(k) for k in keep]
        n_current = len(labels)
        tensor = tensor.transpose(tuple(perm) + tuple(n_current + p for p in perm))
        dims = [dims[p] for p in perm]

    d = prod(dims)
    return tensor.reshape((d, d))


def spectral_unital_kraus(
    local_rho: np.ndarray,
    output_dim: int = 3,
) -> list[np.ndarray]:
    """
    Construct the paper-style d_in -> output_dim unital CPTP channel.

    The most-occupied local eigenvectors are retained coherently by K.
    Every discarded eigenvector is redistributed uniformly over the output
    basis through additional Kraus operators.
    """
    local_rho = np.asarray(local_rho, dtype=np.complex128)
    d_in = local_rho.shape[0]
    if local_rho.shape != (d_in, d_in):
        raise ValueError("local_rho must be square.")
    if not (1 <= output_dim <= d_in):
        raise ValueError("output_dim must be between 1 and local input dimension.")

    eigvals, eigvecs = np.linalg.eigh((local_rho + local_rho.conj().T) / 2)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # K = sum_i |i><psi_i| for the retained local eigenspace.
    K = eigvecs[:, :output_dim].conj().T
    kraus = [K]

    # For each discarded |psi_k>, redistribute it uniformly over output basis:
    # Q_(j,k) = |j><psi_k| / sqrt(output_dim).
    for k in range(output_dim, d_in):
        bra = eigvecs[:, k].conj()
        for j in range(output_dim):
            Q = np.zeros((output_dim, d_in), dtype=np.complex128)
            Q[j, :] = bra / sqrt(output_dim)
            kraus.append(Q)

    return kraus


def compress_six_qubit_to_qutrits(
    rho: np.ndarray,
    *,
    qiskit_order: bool = True,
) -> np.ndarray:
    """
    Map a six-qubit state to three qutrits via local 4 -> 3 channels.

    The channel is chosen from the input state's three local RDMs. The input
    may be a length-64 statevector or a 64 x 64 density matrix.

    Parameters
    ----------
    qiskit_order:
        Set True when rho is obtained from Qiskit/Aer. Set False only when
        the array already uses tensor order |q0 q1 q2 q3 q4 q5>.
    """
    arr = np.asarray(rho, dtype=np.complex128)
    if qiskit_order:
        arr = qiskit_to_q0_first(arr, n_qubits=6)
    rho6 = as_density_matrix(arr, dimension=64)

    local_rdms = [
        partial_trace_keep(rho6, (4, 4, 4), (party,))
        for party in range(3)
    ]
    local_kraus = [spectral_unital_kraus(rdm, output_dim=3) for rdm in local_rdms]

    out = np.zeros((27, 27), dtype=np.complex128)
    for ka, kb, kc in product(*local_kraus):
        M = np.kron(np.kron(ka, kb), kc)
        out += M @ rho6 @ M.conj().T

    out = (out + out.conj().T) / 2
    out /= np.trace(out).real
    return out


def validate_qutrit_compression(rho: np.ndarray, *, qiskit_order: bool = True) -> dict[str, float]:
    """Return simple trace, PSD and normalized-white-noise preservation checks."""
    arr = np.asarray(rho, dtype=np.complex128)
    if qiskit_order:
        arr = qiskit_to_q0_first(arr, n_qubits=6)
    rho6 = as_density_matrix(arr, dimension=64)

    local_rdms = [partial_trace_keep(rho6, (4, 4, 4), (party,)) for party in range(3)]
    local_kraus = [spectral_unital_kraus(rdm, output_dim=3) for rdm in local_rdms]

    def apply_fixed_channels(state: np.ndarray) -> np.ndarray:
        result = np.zeros((27, 27), dtype=np.complex128)
        for ka, kb, kc in product(*local_kraus):
            M = np.kron(np.kron(ka, kb), kc)
            result += M @ state @ M.conj().T
        return (result + result.conj().T) / 2

    out = apply_fixed_channels(rho6)
    white_out = apply_fixed_channels(np.eye(64) / 64)
    return {
        "output_trace_error": float(abs(np.trace(out) - 1)),
        "output_min_eigenvalue": float(np.linalg.eigvalsh(out).min()),
        "white_noise_mapping_error": float(np.linalg.norm(white_out - np.eye(27) / 27)),
    }

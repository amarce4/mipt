"""
Practical local-decoding route for certifying GNME of six-qubit states.

A direct level-2 ring-inflation SDP for three two-qubit parties A=(q0,q1),
B=(q2,q3), C=(q4,q5) has inflated matrix dimension 4096 and is generally too
large for a naive CVXPY construction.

This module applies a local CPTP map at each two-qubit party:
    CNOT(first qubit -> second qubit), then discard the second qubit.

For |GHZ_6> = (|000000> + |111111>)/sqrt(2), the output is exactly
|GHZ_3> = (|000> + |111>)/sqrt(2). Because the operation is local, any GNME
certificate for the decoded three-qubit output also certifies the original
six-qubit input.

The same map is valid for any mixed or pure six-qubit rho, but it may lose
detectable GNME.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np

from ring_inflation import RingInflationResult, ring_inflation_tmax


def _density_matrix(rho: np.ndarray, dimension: int) -> np.ndarray:
    """Convert normalized vector/density-matrix input to a density matrix."""
    arr = np.asarray(rho, dtype=np.complex128)
    if arr.ndim == 1:
        if arr.shape != (dimension,):
            raise ValueError(f"Expected a statevector of length {dimension}; got {arr.shape}.")
        norm = np.vdot(arr, arr).real
        if norm <= 0:
            raise ValueError("Statevector has zero norm.")
        arr = arr / np.sqrt(norm)
        return np.outer(arr, arr.conj())
    if arr.shape != (dimension, dimension):
        raise ValueError(f"Expected a {dimension} x {dimension} density matrix; got {arr.shape}.")
    return arr


def _pairwise_cnot_permutation() -> np.ndarray:
    """
    Dense permutation unitary implementing CNOT on (q0->q1), (q2->q3),
    and (q4->q5), using the computational-basis ordering |q0 q1 ... q5>.
    """
    unitary = np.zeros((64, 64), dtype=np.float64)
    for old_index in range(64):
        bits = [int(x) for x in f"{old_index:06b}"]
        for control, target in ((0, 1), (2, 3), (4, 5)):
            bits[target] ^= bits[control]
        new_index = int("".join(str(b) for b in bits), 2)
        unitary[new_index, old_index] = 1.0
    return unitary


_LOCAL_DECODER_UNITARY = _pairwise_cnot_permutation()


def _partial_trace_discard_odd_qubits(rho: np.ndarray) -> np.ndarray:
    """Keep q0, q2, q4 and trace out q1, q3, q5 from a six-qubit state."""
    tensor = rho.reshape((2,) * 12)
    current_qubits = list(range(6))
    # Trace out in descending original-label order so axis indices stay simple.
    for discard in (5, 3, 1):
        axis = current_qubits.index(discard)
        n = len(current_qubits)
        tensor = np.trace(tensor, axis1=axis, axis2=axis + n)
        current_qubits.pop(axis)
    return tensor.reshape(8, 8)


def decode_paired_six_qubit_state(rho: np.ndarray) -> np.ndarray:
    """
    Apply local coherent repetition-code decoding and return rho_ABC on 3 qubits.

    Parties are assumed to be A=(q0,q1), B=(q2,q3), C=(q4,q5).
    """
    rho6 = _density_matrix(rho, 64)
    decoded_before_trace = _LOCAL_DECODER_UNITARY @ rho6 @ _LOCAL_DECODER_UNITARY.T
    rho3 = _partial_trace_discard_odd_qubits(decoded_before_trace)
    # Remove roundoff asymmetry.
    return (rho3 + rho3.conj().T) / 2.0


def decoded_ring_inflation_tmax(
    rho_six_qubit: np.ndarray,
    *,
    solver: str = cp.SCS,
    symmetry_reduced: bool = True,
    verbose: bool = False,
    **solver_options: Any,
) -> RingInflationResult:
    """
    Run the tractable three-qubit level-2 inflation test after local decoding.

    Certification logic:
        decoded output is GNME  =>  original six-qubit input is GNME.

    The returned t_max applies to the decoded state and not to the direct
    six-qubit white-noise robustness program.
    """
    decoded = decode_paired_six_qubit_state(rho_six_qubit)

    mosek_params = solver_options.pop("mosek_params", None)
    if solver_options:
        # ring_inflation_fixed currently exposes only mosek_params as solver
        # options. Keep failure explicit rather than silently ignoring options.
        unexpected = ", ".join(sorted(solver_options))
        raise TypeError(f"Unsupported solver options in this wrapper: {unexpected}")

    return ring_inflation_tmax(
        decoded,
        party_dims=(2, 2, 2),
        symmetry_reduced=symmetry_reduced,
        solver=solver,
        verbose=verbose,
        mosek_params=mosek_params,
    )


def ghz_state(n_qubits: int) -> np.ndarray:
    """Return the n-qubit GHZ statevector."""
    psi = np.zeros(2**n_qubits, dtype=np.complex128)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)
    return psi


def white_noise_mixture(psi: np.ndarray, p: float) -> np.ndarray:
    """Return (1-p)|psi><psi| + p*I/D."""
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    dimension = psi.size
    return (1 - p) * np.outer(psi, psi.conj()) + p * np.eye(dimension) / dimension

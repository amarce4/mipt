from __future__ import annotations
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import random_unitary, Statevector, partial_trace, entropy, mutual_information
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
import random
import numpy as np
import matplotlib.pyplot as plt

# USE_GPU = False

# if (USE_GPU):
#     backend = AerSimulator(device='GPU', method='matrix_product_state')
# else:
#     backend = AerSimulator(device='CPU', method='matrix_product_state')

from typing import Literal, Sequence

# Haar random unitary MIPT circuit in 1D with L=n qubits and t=d timesteps at measurement rate p.
def random_mipt_1d(n, d, p, closed=True):
    """n must be even. p must be between 0 and 1."""
    if n % 2 != 0:
        raise ValueError("n must be even")
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")
    
    qc = QuantumCircuit(n, n)
    for layer in range(d):

        if (layer % 2 == 0):
            # even layer
            for qubit in range(0, n, 2):
                U = UnitaryGate(random_unitary(4))
                qc.append(U, [qubit, (qubit + 1)])
        else:
            # odd layer
            U = UnitaryGate(random_unitary(4))
            if closed:
                qc.append(U, [0, -1])
            for qubit in range(1, n - 1, 2):
                U = UnitaryGate(random_unitary(4))
                qc.append(U, [qubit, (qubit + 1)])
        
        # random measurements with probability p
        for qubit in range(n):
            if random.random() < p:
                qc.measure(qubit, qubit)

    # No 50% chance of an extra even layer for Haar random circuits.
    # if (random.random() < 0.5): 
    # # 50% chance of an extra even layer at the end
    #     for qubit in range(0, n, 2):
    #         U = UnitaryGate(random_unitary(4))
    #         qc.append(U, [qubit, (qubit + 1)])
    #     for qubit in range(n):
    #         if random.random() < p:
    #             qc.measure(qubit, qubit)

    return qc

def _tmi_block_subsystems_1d(n: int) -> list[tuple[int, ...]]:
    """Return TMI storage subsystems in AB/AC/BC/D order."""
    n = int(n)
    if n < 4 or (n % 4) != 0:
        raise ValueError("TMI matrix writing requires n >= 4 and divisible by 4.")

    block = n // 4
    A = tuple(range(0, block))
    B = tuple(range(block, 2 * block))
    C = tuple(range(2 * block, 3 * block))
    D = tuple(range(3 * block, 4 * block))
    return [A + B, A + C, B + C, D]



def _validate_qubit_subsystems(n: int, subsystems: Sequence[Sequence[int]]) -> list[list[int]]:
    """Normalize and validate qubit subsystem lists."""
    out: list[list[int]] = []
    for i, subsystem in enumerate(subsystems):
        sub = [int(q) for q in subsystem]
        if not sub:
            raise ValueError(f"subsystem {i} is empty")
        if len(set(sub)) != len(sub):
            raise ValueError(f"subsystem {i} contains duplicate qubits: {sub}")
        bad = [q for q in sub if q < 0 or q >= int(n)]
        if bad:
            raise ValueError(f"subsystem {i} contains out-of-range qubits {bad}; n={n}")
        out.append(sub)
    return out


def _as_complex_array(obj) -> np.ndarray:
    """Return the complex ndarray behind a Qiskit state/density object."""
    return np.asarray(getattr(obj, "data", obj), dtype=np.complex128)


def _extract_single_shot_statevector(saved) -> Statevector:
    """
    Extract the unique full-system statevector saved by qc.save_statevector.

    With ``pershot=True`` and ``shots=1``, Aer normally returns a one-element
    list.  The dict branch is retained defensively for simulator/version
    variants that wrap saved data by classical keys.
    """
    candidates = []

    if isinstance(saved, dict):
        for value in saved.values():
            if isinstance(value, (list, tuple)):
                candidates.extend(value)
            else:
                candidates.append(value)
    elif isinstance(saved, (list, tuple)):
        candidates.extend(saved)
    else:
        candidates.append(saved)

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one saved statevector for shots=1, got {len(candidates)} entries."
        )

    arr = _as_complex_array(candidates[0])
    if arr.ndim != 1:
        raise RuntimeError(f"Saved statevector has shape {arr.shape}; expected a one-dimensional vector.")

    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Saved statevector has zero or non-finite norm.")

    return Statevector(arr / norm)


def _qubit_reduced_density_from_statevector(
    psi,
    keep: Sequence[int],
    n_qubits: int,
) -> np.ndarray:
    """
    Ordinary qubit partial trace of a full pure statevector, keeping ``keep``.

    This uses Qiskit's ``partial_trace`` on the full saved statevector.  The
    circuit simulator remains responsible for sampling and applying the
    mid-circuit measurement record; reduction happens only after the final
    conditional trajectory state has been saved.
    """
    n = int(n_qubits)
    keep_list = _validate_qubit_subsystems(n, [keep])[0]
    keep_set = set(keep_list)
    trace_out = [q for q in range(n) if q not in keep_set]
    rho = partial_trace(psi, trace_out)
    arr = _as_complex_array(rho)
    return np.asarray(0.5 * (arr + arr.conjugate().T), dtype=np.complex128)


def qubit_partial_trace_keep_density_matrix(
    rho: np.ndarray,
    keep: Sequence[int],
    n_qubits: int,
) -> np.ndarray:
    """
    Ordinary qubit partial trace of a density matrix, keeping local qubits.

    This helper is used by csim_csv.py for secondary reductions such as
    AB -> A/B and AC -> C.  It uses the same Qiskit partial-trace convention as
    ``get_mipt_rho_1d``.
    """
    n = int(n_qubits)
    keep_list = _validate_qubit_subsystems(n, [keep])[0]
    keep_set = set(keep_list)
    trace_out = [q for q in range(n) if q not in keep_set]
    rho_dm = rho if hasattr(rho, "data") else np.asarray(rho, dtype=np.complex128)
    out = partial_trace(rho_dm, trace_out)
    arr = _as_complex_array(out)
    return np.asarray(0.5 * (arr + arr.conjugate().T), dtype=np.complex128)


def get_mipt_rho_1d(
    n,
    d,
    p,
    subsyst=3,
    all_matrices=False,
    closed=True,
    gpu=False,
    tmi: bool = False,
    transpile_optimization_level: int = 0,
    subsystem_qubits: Sequence[Sequence[int]] | None = None,
):
    """
    Return one trajectory's reduced density matrix/matrices for the qubit MIPT
    circuit.

    This Qiskit-backed implementation intentionally avoids
    ``save_density_matrix`` after mid-circuit measurements.  In the tested
    Aer/MPS path, saved density matrices can represent the nonselective
    post-measurement average even when ``conditional=True`` and ``shots=1``.

    Instead, the circuit is simulated with its measurements intact and the full
    final state is saved with ``qc.save_statevector(pershot=True)``.  The saved
    statevector is then reduced with Qiskit's ordinary qubit ``partial_trace``.
    This keeps the Aer backend in place while computing entropies/GMN from a
    single sampled monitored trajectory.

    tmi=False preserves the previous rho3 behavior.
    tmi=True returns four matrices in the C++ TMI storage order AB, AC, BC, D.
    """

    n = int(n)
    if subsystem_qubits is not None:
        subsystems = _validate_qubit_subsystems(n, subsystem_qubits)
    elif tmi:
        subsystems = _validate_qubit_subsystems(n, _tmi_block_subsystems_1d(n))
    elif all_matrices and closed:
        subsystems = _validate_qubit_subsystems(
            n,
            [[(i + j) % n for j in range(int(subsyst))] for i in range(n)],
        )
    else:
        subsystems = _validate_qubit_subsystems(n, [list(range(int(subsyst)))])

    if gpu is False:
        backend = AerSimulator(device="CPU", method="matrix_product_state")
    else:
        backend = AerSimulator(device="GPU", method="matrix_product_state")

    qc = random_mipt_1d(n=n, d=int(d), p=float(p), closed=closed)
    qc.save_statevector(label="final_state", pershot=True)

    tqc = transpile(qc, backend, optimization_level=transpile_optimization_level)
    result = backend.run(tqc, shots=1).result()
    psi = _extract_single_shot_statevector(result.data(0)["final_state"])

    rhos = [
        _qubit_reduced_density_from_statevector(psi, subsystem, n)
        for subsystem in subsystems
    ]

    if len(rhos) == 1:
        return rhos[0]

    return rhos

def random_mipt_2d(x, y, d, p):
    """p must be between 0 and 1."""
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")
    
    n = x*y
    x,y = y,x

    qregs = [QuantumRegister(1, name=f"x_{j}y_{i}") for i in range(x) for j in range(y)]

    
    qc = QuantumCircuit(*qregs, ClassicalRegister(1, name="c"))

    for layer in range(d):
        if (layer % 4 == 0):
            # horiz even
            for i in range(y):
                if (i % 2 == 0):
                    for j in range(0, x, 2):
                        U = UnitaryGate(random_unitary(4))
                        qc.append(U, [i*x + j, i*x + j + 1])
                else:
                    U = UnitaryGate(random_unitary(4))
                    qc.append(U, [i*x, i*x + x - 1])
                    for j in range(1, x-1, 2):
                        U = UnitaryGate(random_unitary(4))
                        qc.append(U, [i*x + j, i*x + j + 1])
        elif (layer % 4 == 1):
            # horiz odd
            for i in range(y):
                if (i % 2 == 1):
                    for j in range(0, x, 2):
                        U = UnitaryGate(random_unitary(4))
                        qc.append(U, [i*x + j, i*x + j + 1])
                else:
                    U = UnitaryGate(random_unitary(4))
                    qc.append(U, [i*x, i*x + x - 1])
                    for j in range(1, x-1, 2):
                        U = UnitaryGate(random_unitary(4))
                        qc.append(U, [i*x + j, i*x + j + 1])
        elif (layer % 4 == 2):
            # vert even
            for i in range(x):
                if (i % 2 == 0):
                    for j in range(0, y, 2):
                        U = UnitaryGate(random_unitary(4))
                        qc.append(U, [j*y + i, ((j+1)*y)% (x*y) + i])
                else:
                    U = UnitaryGate(random_unitary(4))
                    qc.append(U, [i, (y-1)*y + i])
                    for j in range(1, y-1, 2):
                        U = UnitaryGate(random_unitary(4))
                        qc.append(U, [j*y + i, (j+1)*y + i])
        elif (layer % 4 == 3):
            # vert odd
            for i in range(x):
                if (i % 2 == 1):
                    for j in range(0, y, 2):
                        U = UnitaryGate(random_unitary(4))
                        qc.append(U, [j*y + i, ((j+1)*y)% (x*y) + i])
                else:
                    U = UnitaryGate(random_unitary(4))
                    qc.append(U, [i, (y-1)*y + i])
                    for j in range(1, y-1, 2):
                        U = UnitaryGate(random_unitary(4))
                        qc.append(U, [j*y + i, (j+1)*y + i])
        for qubit in range(n):
            if random.random() < p:
                qc.measure(qubit, 0)

    
    return qc
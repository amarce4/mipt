from __future__ import annotations
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import random_unitary, Statevector, partial_trace, entropy, mutual_information
from qiskit.circuit.library import UnitaryGate
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography
from qiskit_aer import AerSimulator
import random
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.sparse as sp
from itertools import combinations

# USE_GPU = False

# if (USE_GPU):
#     backend = AerSimulator(device='GPU', method='matrix_product_state')
# else:
#     backend = AerSimulator(device='CPU', method='matrix_product_state')

from functools import lru_cache
from itertools import combinations
from typing import Literal, Sequence



@lru_cache(maxsize=None)
def _partial_transpose_permutation(
    dims: tuple[int, ...],
    subsys: tuple[int, ...],
) -> sp.csr_matrix:
    """
    Return sparse T satisfying vec(X^{T_M}) = T @ vec(X),
    using column-major vectorization.
    """
    d = int(np.prod(dims))

    rows = np.empty(d * d, dtype=np.int64)
    cols = np.arange(d * d, dtype=np.int64)
    data = np.ones(d * d, dtype=np.float64)

    for i in range(d):
        ii = list(np.unravel_index(i, dims))

        for j in range(d):
            jj = list(np.unravel_index(j, dims))

            ii_pt = ii.copy()
            jj_pt = jj.copy()

            for s in subsys:
                ii_pt[s], jj_pt[s] = jj_pt[s], ii_pt[s]

            ip = np.ravel_multi_index(tuple(ii_pt), dims)
            jp = np.ravel_multi_index(tuple(jj_pt), dims)

            src = i + j * d
            dst = ip + jp * d

            rows[src] = dst

    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(d * d, d * d),
    )


def partial_transpose_expr(
    X: cp.Expression,
    dims: Sequence[int],
    subsys: Sequence[int],
) -> cp.Expression:
    """
    Partial transpose of a CVXPY matrix expression over the selected
    subsystems.
    """
    dims = tuple(int(x) for x in dims)
    subsys = tuple(sorted(set(int(x) for x in subsys)))

    n = len(dims)
    if any(s < 0 or s >= n for s in subsys):
        raise ValueError(f"Invalid subsystem indices: {subsys}")

    d = int(np.prod(dims))
    T = _partial_transpose_permutation(dims, subsys)

    vec_X = cp.reshape(X, (d * d, 1), order="F")
    vec_X_pt = T @ vec_X

    return cp.reshape(vec_X_pt, (d, d), order="F")


def inequivalent_bipartitions(parties: int) -> list[tuple[int, ...]]:
    """
    Generate one representative M from every nontrivial bipartition
    M | complement(M).

    Examples
    --------
    parties=3:
        [(0,), (1,), (2,)]

    parties=4:
        [(0,), (1,), (2,), (3,),
         (0, 1), (0, 2), (0, 3)]
    """
    if parties < 2:
        raise ValueError("At least two parties are required.")

    cuts: list[tuple[int, ...]] = []

    for size in range(1, parties // 2 + 1):
        for subset in combinations(range(parties), size):
            # For equal-size complementary cuts, retain only one
            # representative by requiring party 0 to be in M.
            if 2 * size == parties and 0 not in subset:
                continue

            cuts.append(subset)

    expected = 2 ** (parties - 1) - 1
    if len(cuts) != expected:
        raise RuntimeError(
            f"Generated {len(cuts)} cuts, expected {expected}."
        )

    return cuts


def gmn(
    rho: np.ndarray,
    parties: int,
    *,
    formulation: Literal["monotone", "paper_witness"] = "monotone",
    solver: str = cp.MOSEK,
    solver_options: dict | None = None,
    zero_tol: float = 1e-8,
    return_problem: bool = False,
):
    """
    Compute the genuine multipartite negativity SDP score.

    Parameters
    ----------
    rho
        Density matrix of shape (2**parties, 2**parties).

    parties
        Number of qubit parties.

    formulation
        "monotone":
            Uses 0 <= P_M <= I and 0 <= Q_M <= I.
            This is the GMN monotone formulation used in your current code.
            The returned value is clamped to zero within `zero_tol`.

        "paper_witness":
            Uses Tr(W) = 1 and P_M, Q_M >= 0.
            This matches Eq. (4) used for the Table I white-noise benchmark.
            A strictly positive returned score indicates detection.

    solver
        CVXPY solver identifier, e.g. cp.MOSEK or cp.SCS.

    solver_options
        Optional keyword arguments passed to problem.solve().

    zero_tol
        Numerical tolerance used to suppress solver-scale residual values
        in the monotone formulation.

    return_problem
        When True, return (score, raw_score, problem, bipartitions).

    Returns
    -------
    float or tuple
        GMN score, or additional solver details when return_problem=True.
    """
    if parties < 2:
        raise ValueError("At least two parties are required.")

    dims = [2] * parties
    d = 2 ** parties

    rho = np.asarray(rho, dtype=np.complex128)

    if rho.shape != (d, d):
        raise ValueError(
            f"rho has shape {rho.shape}, expected {(d, d)} "
            f"for parties={parties}."
        )

    if not np.allclose(rho, rho.conj().T, atol=1e-10):
        raise ValueError("rho must be Hermitian.")

    # Remove insignificant numerical anti-Hermitian noise.
    rho = 0.5 * (rho + rho.conj().T)

    if formulation not in {"monotone", "paper_witness"}:
        raise ValueError(
            "formulation must be 'monotone' or 'paper_witness'."
        )

    I = np.eye(d, dtype=np.complex128)
    cuts = inequivalent_bipartitions(parties)

    W = cp.Variable((d, d), hermitian=True, name="W")

    P: dict[tuple[int, ...], cp.Variable] = {}
    Q: dict[tuple[int, ...], cp.Variable] = {}

    constraints: list[cp.Constraint] = []

    for cut in cuts:
        label = "_".join(str(i) for i in cut)

        P[cut] = cp.Variable((d, d), hermitian=True, name=f"P_{label}")
        Q[cut] = cp.Variable((d, d), hermitian=True, name=f"Q_{label}")

        Q_pt = partial_transpose_expr(Q[cut], dims, cut)

        constraints.extend(
            [
                W == P[cut] + Q_pt,
                P[cut] >> 0,
                Q[cut] >> 0,
            ]
        )

        if formulation == "monotone":
            constraints.extend(
                [
                    I - P[cut] >> 0,
                    I - Q[cut] >> 0,
                ]
            )

    if formulation == "paper_witness":
        constraints.append(cp.trace(W) == 1)

    objective = cp.Minimize(cp.real(cp.trace(rho @ W)))
    problem = cp.Problem(objective, constraints)

    solve_kwargs = {"warm_start": True}
    if solver_options is not None:
        solve_kwargs.update(solver_options)

    problem.solve(solver=solver, **solve_kwargs)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(
            f"GMN SDP did not solve successfully: status={problem.status}"
        )

    raw_score = -float(problem.value)

    if formulation == "monotone":
        score = raw_score if raw_score > zero_tol else 0.0
    else:
        score = raw_score

    if return_problem:
        return score, raw_score, problem, cuts

    return score

def tmi(dm):
    rho_A = partial_trace(dm, [1, 2])
    rho_B = partial_trace(dm, [0, 2])
    rho_C = partial_trace(dm, [0, 1])

    rho_AB = partial_trace(dm, [2])
    rho_AC = partial_trace(dm, [1])
    rho_BC = partial_trace(dm, [0])

    S_A   = entropy(rho_A, base=2)
    S_B   = entropy(rho_B, base=2)
    S_C   = entropy(rho_C, base=2)

    S_AB  = entropy(rho_AB, base=2)
    S_AC  = entropy(rho_AC, base=2)
    S_BC  = entropy(rho_BC, base=2)

    S_ABC = entropy(dm, base=2)

    I3 = (
        S_A + S_B + S_C
        - S_AB - S_AC - S_BC
        + S_ABC
    )
    return I3

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

    if (random.random() < 0.5): 
    # 50% chance of an extra even layer at the end
        for qubit in range(0, n, 2):
            U = UnitaryGate(random_unitary(4))
            qc.append(U, [qubit, (qubit + 1)])
        for qubit in range(n):
            if random.random() < p:
                qc.measure(qubit, qubit)

    return qc

def get_mipt_rho_1d(n, d, p, subsyst=3, closed=True):

    backend = AerSimulator(device="CPU", method="matrix_product_state")
    qc = random_mipt_1d(n=n, d=d, p=float(p), closed=closed)

    qc.save_density_matrix(
        qubits=list(range(subsyst)),
        label="final_state",
        pershot=True,
    )

    tqc = transpile(qc, backend)

    # One conditional outcome trajectory for this independently drawn circuit.
    result = backend.run(tqc, shots=1).result()
    return result.data(0)["final_state"][0]

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

def run_1d_p_scan(n, depth, res, realisations, p_min=0, p_max=1, use_gpu=False, d=1):

    if (d*2 > n):
        raise ValueError(f"N = {n} not sufficient for distance d = {d}")

    subsystem = [i for i in range(0, 2*d+1, d)]
    
    if (use_gpu):
        backend = AerSimulator(device='GPU', method='matrix_product_state')
    else:
        backend = AerSimulator(device='CPU', method='matrix_product_state')

    ps = np.linspace(p_min, p_max, res)
    ents = []
    ent_stds = []
    mis = []
    mi_stds = []
    gmns = []
    gmn_stds = []
    for p in ps:
        print("Simulating p =", p)
        ent = []
        mi = []
        this_gmn = []
    
        for _ in range(realisations):
            qc = random_mipt_1d(n, depth, p, closed=True)
            # qc.save_statevector(label='final_state')
            qc.save_density_matrix(qubits=subsystem, label='final_state')
            transpiled_qc = transpile(qc, backend)

            result = backend.run(transpiled_qc).result()

            sub_m = result.data(0)['final_state']
            ent.append(entropy(sub_m))
            mi.append(tmi(sub_m))
            this_gmn.append(gmn(sub_m))

        ents.append(np.mean(ent))
        ent_stds.append(np.std(ent)/np.sqrt(realisations))
        mis.append(np.mean(mi))
        mi_stds.append(np.std(mi)/np.sqrt(realisations))
        gmns.append(np.mean(this_gmn))
        gmn_stds.append(np.std(this_gmn)/np.sqrt(realisations))

    return ps, ents, ent_stds, mis, mi_stds, gmns, gmn_stds

def run_1d_distance_scan(n, depth, p, realisations, d_min=1, d_max=None, use_gpu=False):

    if d_max is None:
        d_max = n // 2

    if (use_gpu):
        backend = AerSimulator(device='GPU', method='matrix_product_state')
    else:
        backend = AerSimulator(device='CPU', method='matrix_product_state')

    d_values = range(d_min, d_max + 1)
    ents = []
    ent_stds = []
    mis = []
    mi_stds = []
    gmns = []
    gmn_stds = []

    for d in d_values:
        print("Simulating d =", d)
        subsystem = [i for i in range(0, 2*d+1, d)]
        ent = []
        mi = []
        this_gmn = []
        for _ in range(realisations):
            qc = random_mipt_1d(n, depth, p, closed=True)
            # qc.save_statevector(label='final_state')
            qc.save_density_matrix(qubits=subsystem, label='final_state')
            transpiled_qc = transpile(qc, backend)

            result = backend.run(transpiled_qc).result()

            sub_m = result.data(0)['final_state']
            ent.append(entropy(sub_m))
            mi.append(tmi(sub_m))
            this_gmn.append(gmn(sub_m))


        ents.append(np.mean(ent))
        ent_stds.append(np.std(ent)/np.sqrt(realisations))
        mis.append(np.mean(mi))
        mi_stds.append(np.std(mi)/np.sqrt(realisations))
        gmns.append(np.mean(this_gmn))
        gmn_stds.append(np.std(this_gmn)/np.sqrt(realisations))

    return d_values, ents, ent_stds, mis, mi_stds, gmns, gmn_stds

def run_2d(x, y, d, res, realisations, p_min=0, p_max=1, use_gpu=False):
    
    if (use_gpu):
        backend = AerSimulator(device='GPU')
    else:
        backend = AerSimulator(device='CPU')

    ps = np.linspace(p_min, p_max, res)
    ents = []
    ent_stds = []
    mis = []
    mi_stds = []
    gmns = []
    gmn_stds = []
    for p in ps:
        print("Simulating p =", p)
        ent = []
        mi = []
        gmnn = []
        for _ in range(realisations):
            qc = random_mipt_2d(x, y, d, p, closed=True)
            qc.save_statevector(label='final_state')
            transpiled_qc = transpile(qc, backend)

            result = backend.run(transpiled_qc).result()

            saved_state = result.data(0)['final_state']

            sub_m = partial_trace(saved_state, [i for i in range(x*y-3)])
            ent.append(entropy(sub_m))
            mi.append(tmi(sub_m))
            gmnn.append(gmn(sub_m))

        ents.append(np.mean(ent))
        ent_stds.append(np.std(ent)/np.sqrt(realisations))
        mis.append(np.mean(mi))
        mi_stds.append(np.std(mi)/np.sqrt(realisations))
        gmns.append(np.mean(gmnn))
        gmn_stds.append(np.std(gmnn)/np.sqrt(realisations))

    return ps, ents, ent_stds, mis, mi_stds, gmns, gmn_stds
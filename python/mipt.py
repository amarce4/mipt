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

USE_GPU = False

if (USE_GPU):
    backend = AerSimulator(device='GPU', method='matrix_product_state')
else:
    backend = AerSimulator(device='CPU', method='matrix_product_state')


# ------------------------------------------------------------
# Convert Qiskit DensityMatrix / Statevector to numpy matrix
# ------------------------------------------------------------
def to_numpy_density_matrix(rho):
    rho = np.asarray(getattr(rho, "data", rho), dtype=complex)

    if rho.ndim == 1:  # statevector
        rho = np.outer(rho, rho.conj())

    return rho


# ------------------------------------------------------------
# Build partial transpose as a LINEAR MAP on vec(W)
# ------------------------------------------------------------
def partial_transpose_map(dims, sys):
    """
    Returns matrix PT such that:
        vec(W^T_sys) = PT @ vec(W)
    """
    d = int(np.prod(dims))
    dim2 = d * d

    PT = np.zeros((dim2, dim2), dtype=float)

    # basis element E_ij
    for i in range(d):
        for j in range(d):

            E = np.zeros((d, d))
            E[i, j] = 1.0

            # reshape into tensor form
            tensor = E.reshape(dims + dims)

            N = len(dims)
            axes = list(range(2 * N))

            for s in sys:
                axes[s], axes[s + N] = axes[s + N], axes[s]

            pt = tensor.transpose(axes).reshape(d, d)

            col = i * d + j
            PT[:, col] = pt.flatten()

    return PT


# ------------------------------------------------------------
# GMNN SDP
# ------------------------------------------------------------
def gmnn_sdp(rho, dims, k_minus_1=2):
    """
    GMNN witness SDP relaxation.

    Parameters:
    - rho: density matrix (Qiskit or numpy)
    - dims: subsystem dimensions, e.g. [2,2,2]
    - k_minus_1: max hyperedge size (2 = bipartite resources)
    """

    rho = to_numpy_density_matrix(rho)
    d = rho.shape[0]

    parties = list(range(len(dims)))

    # --------------------------------------------------------
    # Build allowed subsystem structure (hyperedges)
    # --------------------------------------------------------
    edges = []
    for r in range(1, k_minus_1 + 1):
        edges += list(combinations(parties, r))

    # --------------------------------------------------------
    # SDP variable: entanglement witness
    # --------------------------------------------------------
    W = cp.Variable((d, d), hermitian=True)

    constraints = []

    # --------------------------------------------------------
    # Constraint 1: upper bound (stability / normalization)
    # --------------------------------------------------------
    constraints += [W << np.eye(d)]

    # --------------------------------------------------------
    # Constraint 2: positivity of partial transposes
    # (network-relaxed PPT constraints)
    # --------------------------------------------------------
    W_vec = cp.vec(W, order="C")

    for S in edges:
        PT = partial_transpose_map(dims, S)

        W_pt = cp.reshape(PT @ W_vec, (d, d), order="C")

        constraints += [W_pt >> 0]

    # --------------------------------------------------------
    # Objective: GMNN witness value
    # --------------------------------------------------------
    objective = cp.Minimize(cp.real(cp.trace(rho @ W)))

    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.MOSEK)

    gmnn_value = -prob.value

    return gmnn_value, W.value

# ============================================================
# Partial transpose
# ============================================================

def partial_transpose_expr(X, dims, subsys):
    """
    Sparse linear-operator implementation of the partial transpose.

    Parameters
    ----------
    X : cvxpy matrix expression
        Hermitian matrix variable/expression.
    dims : list[int]
        Local subsystem dimensions, e.g. [2,2,2].
    subsys : list[int]
        Subsystems to transpose, e.g. [0] for T_A.

    Returns
    -------
    cvxpy expression
        Partial transpose of X over subsys.
    """

    d = int(np.prod(dims))
    n = len(dims)

    rows = []
    cols = []
    data = []

    # Build permutation matrix T such that:
    # vec(X^{T_M}) = T vec(X)

    for i in range(d):
        ii = np.unravel_index(i, dims)

        for j in range(d):
            jj = np.unravel_index(j, dims)

            ii2 = list(ii)
            jj2 = list(jj)

            # swap bra/ket indices on chosen subsystems
            for s in subsys:
                ii2[s], jj2[s] = jj2[s], ii2[s]

            ip = np.ravel_multi_index(ii2, dims)
            jp = np.ravel_multi_index(jj2, dims)

            # vec indexing (column-major)
            src = i + j*d
            dst = ip + jp*d

            rows.append(dst)
            cols.append(src)
            data.append(1.0)

    T = sp.coo_matrix((data, (rows, cols)),
                      shape=(d*d, d*d)).tocsr()

    # vec(X)
    vecX = cp.reshape(X, (d*d, 1), order='F')

    # Apply permutation
    vecPT = T @ vecX

    # reshape back into matrix
    return cp.reshape(vecPT, (d, d), order='F')

def gmn(rho, parties=3):

    # ============================================================
    # Problem dimensions
    # ============================================================

    dims = [2 for _ in range(parties)]
    d = 2 ** parties

    # ============================================================
    # SDP variables
    # ============================================================

    W = cp.Variable((d,d), hermitian=True)

    P = {}
    Q = {}

    partitions = {
        "A": [0],
        "B": [1],
        "C": [2]
    }

    constraints = []

    I = np.eye(d)

    for name, subsys in partitions.items():

        P[name] = cp.Variable((d,d), hermitian=True)
        Q[name] = cp.Variable((d,d), hermitian=True)

        QT = partial_transpose_expr(Q[name], dims, subsys)

        constraints += [
            cp.trace(W) == 1
        ]

        # W = P + Q^{T_M}
        constraints += [
            W == P[name] + QT
        ]

        # 0 <= P <= I
        constraints += [
            P[name] >> 0,
            I - P[name] >> 0
        ]

        # 0 <= Q <= I
        constraints += [
            Q[name] >> 0,
            I - Q[name] >> 0
        ]

    # ============================================================
    # Objective
    # ============================================================

    

    objective = cp.Minimize(cp.real(cp.trace(rho @ W)))

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.MOSEK, warm_start=True)
    return -problem.value

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

def run_1d(n, d, res, realisations, p_min=0, p_max=1, use_gpu=False):
    
    if (use_gpu):
        backend = AerSimulator(device='GPU', method='matrix_product_state', target_gpus=[1])
    else:
        backend = AerSimulator(device='CPU', method='matrix_product_state')

    ps = np.linspace(p_min, p_max, res)
    ents = []
    ent_stds = []
    mis = []
    mi_stds = []
    gmns = []
    gmn_stds = []
    # gmnns = []
    # gmnn_stds = []
    for p in ps:
        print("Simulating p =", p)
        ent = []
        mi = []
        this_gmn = []
        # this_gmnn = []
    
        for _ in range(realisations):
            qc = random_mipt_1d(n, d, p, closed=True)
            # qc.save_statevector(label='final_state')
            qc.save_density_matrix(qubits=[0,1,2], label='final_state')
            transpiled_qc = transpile(qc, backend)

            result = backend.run(transpiled_qc).result()

            # saved_state = result.data(0)['final_state']

            # sub_m = partial_trace(saved_state, [i for i in range(n-3)])
            sub_m = result.data(0)['final_state']
            ent.append(entropy(sub_m))
            mi.append(tmi(sub_m))
            this_gmn.append(gmn(sub_m))
            # this_gmnn.append(gmnn_sdp(sub_m, dims=[2,2,2], k_minus_1=2)[0])

        ents.append(np.mean(ent))
        ent_stds.append(np.std(ent)/np.sqrt(realisations))
        mis.append(np.mean(mi))
        mi_stds.append(np.std(mi)/np.sqrt(realisations))
        gmns.append(np.mean(this_gmn))
        gmn_stds.append(np.std(this_gmn)/np.sqrt(realisations))
        # gmnns.append(np.mean(this_gmnn))
        # gmnn_stds.append(np.std(this_gmnn)/np.sqrt(realisations))

    return ps, ents, ent_stds, mis, mi_stds, gmns, gmn_stds#, gmnns, gmnn_stds

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
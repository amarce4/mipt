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


class _GMN3CachedProblem:
    """
    Reusable fixed-size 3-qubit GMN SDP.

    This avoids rebuilding the CVXPY graph for every 8x8 density matrix.  The
    density matrix is a CVXPY Parameter; each call only updates that parameter
    and resolves the same problem.  This is the biggest GMN-side speedup for
    scans where thousands of 3-qubit density matrices are solved.
    """

    dims: tuple[int, int, int] = (2, 2, 2)
    d: int = 8
    cuts: tuple[tuple[int, ...], ...] = ((0,), (1,), (2,))

    def __init__(self, formulation: Literal["monotone", "paper_witness"] = "monotone"):
        if formulation not in {"monotone", "paper_witness"}:
            raise ValueError("formulation must be 'monotone' or 'paper_witness'.")

        self.formulation = formulation
        self.rho_param = cp.Parameter((self.d, self.d), complex=True, name="rho")
        self.W = cp.Variable((self.d, self.d), hermitian=True, name="W")

        I = np.eye(self.d, dtype=np.complex128)
        constraints: list[cp.Constraint] = []

        self.P: dict[tuple[int, ...], cp.Variable] = {}
        self.Q: dict[tuple[int, ...], cp.Variable] = {}

        for cut in self.cuts:
            label = "_".join(str(i) for i in cut)
            P_cut = cp.Variable((self.d, self.d), hermitian=True, name=f"P_{label}")
            Q_cut = cp.Variable((self.d, self.d), hermitian=True, name=f"Q_{label}")

            self.P[cut] = P_cut
            self.Q[cut] = Q_cut

            Q_pt = partial_transpose_expr(Q_cut, self.dims, cut)

            constraints.extend(
                [
                    self.W == P_cut + Q_pt,
                    P_cut >> 0,
                    Q_cut >> 0,
                ]
            )

            if formulation == "monotone":
                constraints.extend(
                    [
                        I - P_cut >> 0,
                        I - Q_cut >> 0,
                    ]
                )

        if formulation == "paper_witness":
            constraints.append(cp.trace(self.W) == 1)

        objective = cp.Minimize(cp.real(cp.trace(self.rho_param @ self.W)))
        self.problem = cp.Problem(objective, constraints)

    @staticmethod
    def _clean_rho(rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=np.complex128)

        if rho.shape != (8, 8):
            raise ValueError(f"rho has shape {rho.shape}; this optimized GMN expects an 8x8 matrix.")

        if not np.allclose(rho, rho.conj().T, atol=1e-10):
            raise ValueError("rho must be Hermitian.")

        return 0.5 * (rho + rho.conj().T)

    def solve(
        self,
        rho: np.ndarray,
        *,
        solver: str = cp.MOSEK,
        solver_options: dict | None = None,
        zero_tol: float = 1e-8,
        return_problem: bool = False,
    ):
        self.rho_param.value = self._clean_rho(rho)

        solve_kwargs = {"warm_start": True}
        if solver_options is not None:
            solve_kwargs.update(solver_options)

        self.problem.solve(solver=solver, **solve_kwargs)

        if self.problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(
                f"GMN SDP did not solve successfully: status={self.problem.status}"
            )

        raw_score = -float(self.problem.value)

        if self.formulation == "monotone":
            score = raw_score if raw_score > zero_tol else 0.0
        else:
            score = raw_score

        if return_problem:
            return score, raw_score, self.problem, list(self.cuts)

        return score


_GMN3_CACHE: dict[str, _GMN3CachedProblem] = {}


def _get_gmn3_problem(formulation: Literal["monotone", "paper_witness"]) -> _GMN3CachedProblem:
    problem = _GMN3_CACHE.get(formulation)
    if problem is None:
        problem = _GMN3CachedProblem(formulation=formulation)
        _GMN3_CACHE[formulation] = problem
    return problem


def gmn(
    rho: np.ndarray,
    *,
    parties: int = 3,
    formulation: Literal["monotone", "paper_witness"] = "monotone",
    solver: str = cp.MOSEK,
    solver_options: dict | None = None,
    zero_tol: float = 1e-8,
    return_problem: bool = False,
):
    """
    Optimized 3-qubit genuine multipartite negativity SDP score.

    This version intentionally supports only the 3-party / 8x8 case.  It reuses
    a cached CVXPY problem with the density matrix supplied as a Parameter,
    which removes the repeated problem-construction cost from large MIPT scans.
    The call signature is kept compatible with the previous ``gmn(rho)`` usage.
    """
    if parties != 3:
        raise ValueError(
            "This optimized gmn implementation only supports parties=3 and 8x8 rho."
        )

    problem = _get_gmn3_problem(formulation)
    return problem.solve(
        rho,
        solver=solver,
        solver_options=solver_options,
        zero_tol=zero_tol,
        return_problem=return_problem,
    )

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

def run_1d_p_scan(n, depth, res, realisations, p_min=0, p_max=1, use_gpu=False, d=1):

    if (d*2 > n):
        raise ValueError(f"N = {n} not sufficient for distance d = {d}")

    subsystem = [i for i in range(0, 2*d+1, d)]
    
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
            sub_m = get_mipt_rho_1d(
                n,
                depth,
                p,
                closed=True,
                gpu=use_gpu,
                subsystem_qubits=[subsystem],
            )
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
            sub_m = get_mipt_rho_1d(
                n,
                depth,
                p,
                closed=True,
                gpu=use_gpu,
                subsystem_qubits=[subsystem],
            )
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
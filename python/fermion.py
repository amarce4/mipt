from mipt import *

def fermionic_swap_gate() -> UnitaryGate:
    """
    Adjacent fermionic SWAP.

    Basis order:
        |00>, |01>, |10>, |11>

    Action:
        |00> -> |00>
        |01> -> |10>
        |10> -> |01>
        |11> -> -|11>

    The minus sign on |11> is the fermionic exchange sign.
    """
    fswap = np.array(
        [
            [1, 0, 0,  0],
            [0, 0, 1,  0],
            [0, 1, 0,  0],
            [0, 0, 0, -1],
        ],
        dtype=np.complex128,
    )

    return UnitaryGate(fswap, label="FSWAP")


def append_exact_periodic_jw_boundary_gate(qc, gate_0_last, n: int):
    """
    Apply an exact periodic-boundary fermionic two-mode gate between modes 0 and n-1
    under Jordan-Wigner ordering.

    Instead of applying the gate directly to qubits [0, n-1], this uses a fermionic
    swap network:

        [0, 1, 2, ..., n-2, n-1]
            -> [0, n-1, 1, 2, ..., n-2]

    Then it applies the desired two-mode gate to adjacent modes [0, 1], and swaps
    the modes back.

    Parameters
    ----------
    qc:
        QuantumCircuit.

    gate_0_last:
        Two-qubit parity-preserving UnitaryGate representing the desired fermionic
        gate between modes 0 and n-1, in the local occupation basis.

    n:
        Number of fermionic modes / qubits.
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    fswap = fermionic_swap_gate()

    # Move mode n-1 leftward until it sits next to mode 0.
    #
    # Mode ordering changes as:
    #   [0, 1, 2, ..., n-2, n-1]
    #   [0, n-1, 1, 2, ..., n-2]
    for k in range(n - 2, 0, -1):
        qc.append(fswap, [k, k + 1])

    # Now physical modes 0 and n-1 occupy adjacent JW positions 0 and 1.
    qc.append(gate_0_last, [0, 1])

    # Swap mode n-1 back to its original JW position.
    for k in range(1, n - 1):
        qc.append(fswap, [k, k + 1])

def random_parity_preserving_unitary_4(seed=None) -> np.ndarray:
    """
    Random two-mode fermionic parity-preserving unitary.

    Basis order:
        |00>, |01>, |10>, |11>

    Even local parity sector:
        |00>, |11>

    Odd local parity sector:
        |01>, |10>

    Returns
    -------
    U : np.ndarray, shape (4, 4)
        Block-diagonal parity-preserving two-mode unitary.
    """
    rng = np.random.default_rng(seed)

    seed_even = int(rng.integers(0, 2**32 - 1))
    seed_odd = int(rng.integers(0, 2**32 - 1))

    U_even = random_unitary(2, seed=seed_even).data
    U_odd = random_unitary(2, seed=seed_odd).data

    U = np.zeros((4, 4), dtype=np.complex128)

    even = [0, 3]  # |00>, |11>
    odd = [1, 2]   # |01>, |10>

    U[np.ix_(even, even)] = U_even
    U[np.ix_(odd, odd)] = U_odd

    return U

def random_parity_preserving_gate(seed=None) -> UnitaryGate:
    return UnitaryGate(random_parity_preserving_unitary_4(seed=seed))

def random_mipt_1d_ferm(n, d, p, closed=True):
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
                U = random_parity_preserving_gate()
                qc.append(U, [qubit, (qubit + 1)])
        else:
            # odd layer
            if closed:
                U = random_parity_preserving_gate()
                append_exact_periodic_jw_boundary_gate(
                    qc=qc,
                    gate_0_last=U,
                    n=n,
                )
            for qubit in range(1, n - 1, 2):
                U = random_parity_preserving_gate()
                qc.append(U, [qubit, (qubit + 1)])
        
        # random measurements with probability p
        for qubit in range(n):
            if random.random() < p:
                qc.measure(qubit, qubit)

    if (random.random() < 0.5): 
    # 50% chance of an extra even layer at the end
        for qubit in range(0, n, 2):
            U = random_parity_preserving_gate()
            qc.append(U, [qubit, (qubit + 1)])
        for qubit in range(n):
            if random.random() < p:
                qc.measure(qubit, qubit)

    return qc

def get_mipt_rho_1d_ferm(n, d, p, subsyst=3, all_matrices=False, closed=True, gpu=False):

    subsystems = []
    if all_matrices and closed:
        subsystems = [[ (i + j) % n for j in range(subsyst) ] for i in range(n)]
    else:
        subsystems = [list(range(subsyst))]

    backend: AerSimulator
    if gpu == False:
        backend = AerSimulator(device="CPU", method="matrix_product_state")
    else:
        backend = AerSimulator(device="GPU", method="matrix_product_state")
        
    qc = random_mipt_1d_ferm(n=n, d=d, p=float(p), closed=closed)

    for i in range(len(subsystems)):
        qc.save_density_matrix(
            qubits=subsystems[i],
            label=f"final_state_{i}",
            pershot=True,
        )

    tqc = transpile(qc, backend)

    # One conditional outcome trajectory for this independently drawn circuit.
    result = backend.run(tqc, shots=1).result()

    if len(subsystems) == 1:
        return result.data(0)["final_state_0"][0]

    data = []
    for i in range(len(subsystems)):
        data.append(result.data(0)[f"final_state_{i}"][0])

    return data
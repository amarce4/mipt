from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import os
from typing import Iterable, Sequence

from mipt import *
from fermionic_partial_trace import *


_UINT32_MAX = 2**32 - 1


def _child_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, _UINT32_MAX))


def _haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Haar-random unitary using QR decomposition.

    This avoids calling qiskit's random_unitary for every 2x2 block, which is
    useful in large random-circuit scans.
    """
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z.astype(np.complex128))

    diag = np.diag(r)
    phases = np.ones(dim, dtype=np.complex128)
    nonzero = np.abs(diag) > 0.0
    phases[nonzero] = diag[nonzero] / np.abs(diag[nonzero])

    return q * phases


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


_FSWAP_GATE = fermionic_swap_gate()


def append_exact_periodic_jw_boundary_gate(qc, gate_0_last, n: int):
    """
    Apply an exact periodic-boundary fermionic two-mode gate between modes 0 and
    n-1 under Jordan-Wigner ordering.

    The boundary mode is brought next to mode 0 using adjacent fermionic swaps,
    the two-mode gate is applied, and the mode is swapped back.  This includes
    the fermionic exchange signs that are missing from a direct qubit gate on
    [0, n - 1].
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    # Move mode n-1 leftward until it sits next to mode 0.
    for k in range(n - 2, 0, -1):
        qc.append(_FSWAP_GATE, [k, k + 1])

    # Now physical modes 0 and n-1 occupy adjacent JW positions 0 and 1.
    qc.append(gate_0_last, [0, 1])

    # Swap mode n-1 back to its original JW position.
    for k in range(1, n - 1):
        qc.append(_FSWAP_GATE, [k, k + 1])


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

    U_even = _haar_unitary(2, rng)
    U_odd = _haar_unitary(2, rng)

    U = np.zeros((4, 4), dtype=np.complex128)

    even = [0, 3]  # |00>, |11>
    odd = [1, 2]   # |01>, |10>

    U[np.ix_(even, even)] = U_even
    U[np.ix_(odd, odd)] = U_odd

    return U


def random_parity_preserving_gate(seed=None) -> UnitaryGate:
    return UnitaryGate(random_parity_preserving_unitary_4(seed=seed), label="Upp")


def random_mipt_1d_ferm(n, d, p, closed=True, seed=None):
    """n must be even. p must be between 0 and 1."""
    if n % 2 != 0:
        raise ValueError("n must be even")
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, n)

    for layer in range(d):
        if layer % 2 == 0:
            # even layer: (0,1), (2,3), ...
            for qubit in range(0, n, 2):
                U = random_parity_preserving_gate(seed=_child_seed(rng))
                qc.append(U, [qubit, qubit + 1])
        else:
            # odd layer: exact periodic JW boundary plus (1,2), (3,4), ...
            if closed:
                U = random_parity_preserving_gate(seed=_child_seed(rng))
                append_exact_periodic_jw_boundary_gate(
                    qc=qc,
                    gate_0_last=U,
                    n=n,
                )

            for qubit in range(1, n - 1, 2):
                U = random_parity_preserving_gate(seed=_child_seed(rng))
                qc.append(U, [qubit, qubit + 1])

        # local occupation measurements with probability p
        for qubit in range(n):
            if rng.random() < p:
                qc.measure(qubit, qubit)

    if rng.random() < 0.5:
        # 50% chance of an extra even layer at the end.
        for qubit in range(0, n, 2):
            U = random_parity_preserving_gate(seed=_child_seed(rng))
            qc.append(U, [qubit, qubit + 1])

        for qubit in range(n):
            if rng.random() < p:
                qc.measure(qubit, qubit)

    return qc


def get_mipt_rho_1d_ferm(
    n,
    d,
    p,
    plans=None,
    subsyst=3,
    all_matrices=False,
    closed=True,
    gpu=False,
    seed=None,
    transpile_optimization_level: int = 0,
    tmi: bool = False,
):
    """
    Return one trajectory's reduced density matrix/matrices for the fermionic
    parity-preserving MIPT circuit.

    tmi=False preserves the previous rho3/ring-neighborhood behavior.
    tmi=True returns four fermionically reduced matrices in the C++ TMI storage
    order AB, AC, BC, D.

    For repeated scans, pass precomputed ``plans``. If omitted, plans are built
    automatically for the selected mode.
    """
    if tmi:
        subsystems = [list(x) for x in tmi_block_subsystems(int(n))]
        if plans is None:
            plans = make_tmi_block_plans(int(n), basis_order="little")
    elif all_matrices and closed:
        subsystems = [[(i + j) % n for j in range(subsyst)] for i in range(n)]
        if plans is None:
            plans = make_ring_3mode_plans(int(n), closed=True, basis_order="little")
    else:
        subsystems = [list(range(subsyst))]
        if plans is None:
            plans = [FermionicPartialTracePlan(int(n), keep=subsystems[0], basis_order="little")]

    if len(plans) != len(subsystems):
        raise ValueError(
            f"Number of fermionic trace plans ({len(plans)}) does not match "
            f"number of requested subsystems ({len(subsystems)})."
        )

    if gpu:
        backend = AerSimulator(device="GPU", method="matrix_product_state")
    else:
        backend = AerSimulator(device="CPU", method="matrix_product_state")

    qc = random_mipt_1d_ferm(n=n, d=d, p=float(p), closed=closed, seed=seed)

    qc.save_statevector(pershot=True, label="final_state")

    tqc = transpile(qc, backend, optimization_level=transpile_optimization_level)

    # One conditional outcome trajectory for this independently drawn circuit.
    result = backend.run(tqc, shots=1).result()
    psi = result.data(0)["final_state"][0]

    rhos = [plan.trace_statevector(psi) for plan in plans]
    if len(rhos) == 1:
        return rhos[0]
    return rhos

def _merge_mosek_thread_option(solver_options: dict | None, mosek_threads: int | None) -> dict | None:
    if solver_options is None and mosek_threads is None:
        return None

    out = {} if solver_options is None else dict(solver_options)

    if mosek_threads is not None:
        mosek_params = dict(out.get("mosek_params", {}))
        mosek_params["MSK_IPAR_NUM_THREADS"] = int(mosek_threads)
        out["mosek_params"] = mosek_params

    return out


def _gmn_worker(payload):
    rho, gmn_kwargs = payload
    return float(gmn(np.asarray(rho, dtype=np.complex128), **gmn_kwargs))


def gmn_many(
    rhos: Sequence[np.ndarray],
    *,
    max_workers: int | None = None,
    chunksize: int = 1,
    mosek_threads: int | None = 1,
    **gmn_kwargs,
) -> list[float]:
    """
    Compute GMN for many 8x8 matrices, optionally in parallel.

    By default, each MOSEK solve is limited to one thread when running through
    this helper.  That avoids oversubscription when several worker processes are
    active.  Set mosek_threads=None to leave solver threading untouched.
    """
    rhos = list(rhos)
    if not rhos:
        return []

    gmn_kwargs = dict(gmn_kwargs)
    gmn_kwargs["solver_options"] = _merge_mosek_thread_option(
        gmn_kwargs.get("solver_options"),
        mosek_threads,
    )

    if max_workers is None:
        max_workers = max(1, min(len(rhos), (os.cpu_count() or 1)))

    if max_workers <= 1:
        return [_gmn_worker((rho, gmn_kwargs)) for rho in rhos]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_gmn_worker, [(rho, gmn_kwargs) for rho in rhos], chunksize=chunksize))


def _fermion_realization_gmns_worker(payload):
    (
        n,
        d,
        p,
        subsyst,
        all_matrices,
        closed,
        gpu,
        seed,
        transpile_optimization_level,
        gmn_kwargs,
    ) = payload

    rhos = get_mipt_rho_1d_ferm(
        n=n,
        d=d,
        p=p,
        subsyst=subsyst,
        all_matrices=all_matrices,
        closed=closed,
        gpu=gpu,
        seed=seed,
        transpile_optimization_level=transpile_optimization_level,
    )

    if isinstance(rhos, list):
        return [float(gmn(rho, **gmn_kwargs)) for rho in rhos]

    return [float(gmn(rhos, **gmn_kwargs))]


def run_fermion_gmn_scan(
    n: int,
    d: int,
    ps: Sequence[float],
    reps: int,
    *,
    subsyst: int = 3,
    all_matrices: bool = True,
    closed: bool = True,
    gpu: bool = False,
    seed: int | None = None,
    circuit_workers: int | None = None,
    gmn_workers: int | None = None,
    chunksize: int = 1,
    mosek_threads: int | None = 1,
    progress: bool = True,
    transpile_optimization_level: int = 0,
    return_all: bool = False,
    **gmn_kwargs,
):
    """
    Fast scan helper for the workflow:

        rhos = get_mipt_rho_1d_ferm(..., all_matrices=True)
        gmns = [gmn(rho) for rho in rhos]

    Parameters
    ----------
    circuit_workers:
        Number of worker processes used to parallelize whole realizations.
        For GPU simulation, the safe default is 1 so multiple processes do not
        contend for the same GPU.

    gmn_workers:
        Number of worker processes used to parallelize the GMN solves after each
        p's density matrices are generated serially.  For gpu=True, this is the
        default parallelism mode.

    mosek_threads:
        Thread count passed to each MOSEK worker solve via solver_options.  The
        default is 1, which is usually best when using process parallelism.
        Use None to leave MOSEK's own threading untouched.

    Returns
    -------
    means, stderrs
        Arrays with one entry per p.  If return_all=True, also returns the raw
        GMN values per p.
    """
    ps = np.asarray(ps, dtype=float)
    rng = np.random.default_rng(seed)

    if circuit_workers is None:
        # Do not launch several GPU-backed Aer processes by default.
        circuit_workers = 1 if gpu else max(1, min(int(reps), os.cpu_count() or 1))

    if gmn_workers is None:
        # With gpu=True, generate circuits serially on the GPU, then parallelize
        # the CPU-bound GMN solves.  With CPU simulation, parallelize by whole
        # realizations instead and keep GMN serial inside each worker.
        gmn_workers = max(1, (os.cpu_count() or 2) - 1) if gpu else 1

    gmn_kwargs = dict(gmn_kwargs)
    gmn_kwargs["solver_options"] = _merge_mosek_thread_option(
        gmn_kwargs.get("solver_options"),
        mosek_threads,
    )

    means: list[float] = []
    stderrs: list[float] = []
    all_values: list[list[float]] = []

    for p_index, p in enumerate(ps):
        if progress:
            print(f"Simulating p = {p}...")

        rep_seeds = [_child_seed(rng) for _ in range(reps)]
        gmns: list[float] = []

        if circuit_workers > 1:
            if gpu:
                raise ValueError(
                    "circuit_workers > 1 with gpu=True would launch multiple GPU Aer processes. "
                    "Use circuit_workers=1 and gmn_workers>1 instead."
                )

            payloads = [
                (
                    n,
                    d,
                    float(p),
                    subsyst,
                    all_matrices,
                    closed,
                    gpu,
                    rep_seed,
                    transpile_optimization_level,
                    gmn_kwargs,
                )
                for rep_seed in rep_seeds
            ]

            with ProcessPoolExecutor(max_workers=circuit_workers) as ex:
                for r, values in enumerate(ex.map(_fermion_realization_gmns_worker, payloads, chunksize=chunksize), start=1):
                    gmns.extend(values)
                    if progress:
                        print(f"Realisations: {r}/{reps}", end="\r", flush=True)
        else:
            rhos_for_gmn: list[np.ndarray] = []

            for r, rep_seed in enumerate(rep_seeds, start=1):
                rhos = get_mipt_rho_1d_ferm(
                    n=n,
                    d=d,
                    p=float(p),
                    subsyst=subsyst,
                    all_matrices=all_matrices,
                    closed=closed,
                    gpu=gpu,
                    seed=rep_seed,
                    transpile_optimization_level=transpile_optimization_level,
                )

                if isinstance(rhos, list):
                    if gmn_workers > 1:
                        rhos_for_gmn.extend(rhos)
                    else:
                        gmns.extend(float(gmn(rho, **gmn_kwargs)) for rho in rhos)
                else:
                    if gmn_workers > 1:
                        rhos_for_gmn.append(rhos)
                    else:
                        gmns.append(float(gmn(rhos, **gmn_kwargs)))

                if progress:
                    print(f"Realisations: {r}/{reps}", end="\r", flush=True)

            if gmn_workers > 1:
                if progress:
                    print(f"\nSolving {len(rhos_for_gmn)} GMN SDPs with {gmn_workers} workers...")

                gmns.extend(
                    gmn_many(
                        rhos_for_gmn,
                        max_workers=gmn_workers,
                        chunksize=chunksize,
                        mosek_threads=mosek_threads,
                        **{k: v for k, v in gmn_kwargs.items() if k != "solver_options"},
                        solver_options=gmn_kwargs.get("solver_options"),
                    )
                )

        mean = float(np.mean(gmns)) if gmns else float("nan")
        stderr = float(np.std(gmns, ddof=1) / np.sqrt(len(gmns))) if len(gmns) > 1 else 0.0

        means.append(mean)
        stderrs.append(stderr)
        all_values.append(gmns)

        if progress:
            print(f"\nGMN: {mean:.8g} ± {stderr:.3g}")

    means_arr = np.asarray(means, dtype=float)
    stderrs_arr = np.asarray(stderrs, dtype=float)

    if return_all:
        return means_arr, stderrs_arr, all_values

    return means_arr, stderrs_arr

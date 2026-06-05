from __future__ import annotations
from mipt import *

@lru_cache(maxsize=None)
def _partial_transpose_permutation_matrix(
    dims: tuple[int, ...],
    subsys: tuple[int, ...],
    *,
    order: str = "C",
) -> sp.csr_matrix:
    """
    Sparse permutation matrix M such that:

        vec(PT_subsys(X)) = M @ vec(X)

    using the same vectorization order as cp.vec(..., order=order).
    """
    if order not in {"C", "F"}:
        raise ValueError("order must be 'C' or 'F'.")

    dims = tuple(int(x) for x in dims)
    subsys = tuple(int(x) for x in subsys)

    d = int(np.prod(dims))
    nnz = d * d

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    k = 0

    for r in range(d):
        r_multi = list(np.unravel_index(r, dims, order=order))

        for c in range(d):
            c_multi = list(np.unravel_index(c, dims, order=order))

            r_pt = r_multi.copy()
            c_pt = c_multi.copy()

            for s in subsys:
                r_pt[s], c_pt[s] = c_pt[s], r_pt[s]

            rp = np.ravel_multi_index(tuple(r_pt), dims, order=order)
            cp_ = np.ravel_multi_index(tuple(c_pt), dims, order=order)

            if order == "C":
                out_index = rp * d + cp_
                in_index = r * d + c
            else:
                out_index = rp + d * cp_
                in_index = r + d * c

            rows[k] = out_index
            cols[k] = in_index
            k += 1

    data = np.ones(nnz, dtype=np.float64)

    return sp.csr_matrix((data, (rows, cols)), shape=(d * d, d * d))


def partial_transpose_expr_fast(
    X,
    dims: Sequence[int],
    subsys: Sequence[int],
    *,
    order: str = "C",
):
    """
    Fast CVXPY partial transpose using a cached sparse permutation matrix.
    """
    dims = tuple(int(x) for x in dims)
    subsys = tuple(int(x) for x in subsys)
    d = int(np.prod(dims))

    M = _partial_transpose_permutation_matrix(dims, subsys, order=order)

    return cp.reshape(
        M @ cp.vec(X, order=order),
        (d, d),
        order=order,
    )



def _num_qubits_from_dim(d: int) -> int:
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(
            f"rho dimension must be a power of two when party_dims is not given; got d={d}."
        )
    return d.bit_length() - 1


def _balanced_contiguous_qubit_partition(
    n_qubits: int,
    parties: int,
) -> list[tuple[int, ...]]:
    if parties < 3:
        raise ValueError("At least three effective parties are required for GMN.")

    if n_qubits < 3:
        raise ValueError("The input matrix must represent more than two qubits.")

    if parties > n_qubits:
        raise ValueError(
            f"Cannot split {n_qubits} qubits into {parties} nonempty parties."
        )

    base, rem = divmod(n_qubits, parties)

    sizes = [
        base + (1 if i < rem else 0)
        for i in range(parties)
    ]

    partition: list[tuple[int, ...]] = []
    start = 0
    for size in sizes:
        partition.append(tuple(range(start, start + size)))
        start += size

    return partition


def _permute_qubit_density_matrix(
    rho: np.ndarray,
    *,
    n_qubits: int,
    order: Sequence[int],
) -> np.ndarray:
    """
    Reorders tensor axes of a qubit density matrix.

    If rho is ordered as q0, q1, ..., qN-1, then order=[0,2,1]
    maps the tensor-factor order to q0, q2, q1.

    This is needed before collapsing qubits into higher-dimensional
    effective parties.
    """
    order = list(order)

    if order == list(range(n_qubits)):
        return rho

    if sorted(order) != list(range(n_qubits)):
        raise ValueError(
            f"Invalid qubit order {order}; expected a permutation of 0..{n_qubits - 1}."
        )

    tensor = rho.reshape((2,) * (2 * n_qubits))

    row_axes = order
    col_axes = [n_qubits + q for q in order]

    return np.transpose(tensor, row_axes + col_axes).reshape(rho.shape)


def _resolve_party_dims_and_rho(
    rho: np.ndarray,
    *,
    parties: int | None,
    party_dims: Sequence[int] | None,
    qubit_partition: Sequence[Sequence[int]] | None,
) -> tuple[np.ndarray, list[int]]:
    d = rho.shape[0]

    if party_dims is not None and qubit_partition is not None:
        raise ValueError("Specify only one of party_dims or qubit_partition, not both.")

    if party_dims is not None:
        dims = [int(x) for x in party_dims]

        if len(dims) < 3:
            raise ValueError("At least three effective parties are required for GMN.")

        if any(x < 2 for x in dims):
            raise ValueError(f"All party dimensions must be >= 2; got {dims}.")

        if int(np.prod(dims)) != d:
            raise ValueError(
                f"party_dims product is {int(np.prod(dims))}, but rho dimension is {d}."
            )

        if parties is not None and parties != len(dims):
            raise ValueError(
                f"parties={parties} disagrees with len(party_dims)={len(dims)}."
            )

        return rho, dims

    n_qubits = _num_qubits_from_dim(d)

    if n_qubits < 3:
        raise ValueError("The input matrix must represent more than two qubits.")

    if qubit_partition is None:
        if parties is None:
            parties = n_qubits

        partition = _balanced_contiguous_qubit_partition(n_qubits, parties)
        dims = [2 ** len(group) for group in partition]
        return rho, dims

    partition = [tuple(int(q) for q in group) for group in qubit_partition]

    if len(partition) < 3:
        raise ValueError("At least three effective parties are required for GMN.")

    if parties is not None and parties != len(partition):
        raise ValueError(
            f"parties={parties} disagrees with len(qubit_partition)={len(partition)}."
        )

    if any(len(group) == 0 for group in partition):
        raise ValueError("Each effective party must contain at least one qubit.")

    flat = [q for group in partition for q in group]

    if sorted(flat) != list(range(n_qubits)):
        raise ValueError(
            "qubit_partition must contain each qubit index exactly once. "
            f"Expected 0..{n_qubits - 1}, got {flat}."
        )

    rho = _permute_qubit_density_matrix(
        rho,
        n_qubits=n_qubits,
        order=flat,
    )

    dims = [2 ** len(group) for group in partition]

    return rho, dims

def gmn_fast(
    rho: np.ndarray,
    parties: int | None = None,
    *,
    party_dims: Sequence[int] | None = None,
    qubit_partition: Sequence[Sequence[int]] | None = None,
    formulation: Literal["monotone", "paper_witness"] = "monotone",
    solver: str = cp.MOSEK,
    solver_options: dict | None = None,
    zero_tol: float = 1e-8,
    real_tol: float = 1e-12,
    return_problem: bool = False,
):
    """
    Faster GMN SDP.

    Requires your existing helper functions:

        inequivalent_bipartitions(...)
        _resolve_party_dims_and_rho(...)

    from the previous version.
    """
    rho = np.asarray(rho, dtype=np.complex128)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho must be a square matrix; got shape {rho.shape}.")

    if not np.allclose(rho, rho.conj().T, atol=1e-10):
        raise ValueError("rho must be Hermitian.")

    if formulation not in {"monotone", "paper_witness"}:
        raise ValueError("formulation must be 'monotone' or 'paper_witness'.")

    rho = 0.5 * (rho + rho.conj().T)

    rho, dims = _resolve_party_dims_and_rho(
        rho,
        parties=parties,
        party_dims=party_dims,
        qubit_partition=qubit_partition,
    )

    d = rho.shape[0]
    m = len(dims)

    cuts = inequivalent_bipartitions(m)

    use_real_sdp = np.max(np.abs(rho.imag)) <= real_tol

    if use_real_sdp:
        rho_sdp = rho.real
        W = cp.Variable((d, d), symmetric=True, name="W")
    else:
        rho_sdp = rho
        W = cp.Variable((d, d), hermitian=True, name="W")

    I = np.eye(d)

    constraints: list[cp.Constraint] = []

    for cut in cuts:
        label = "_".join(str(i) for i in cut)

        if use_real_sdp:
            Q = cp.Variable((d, d), symmetric=True, name=f"Q_{label}")
        else:
            Q = cp.Variable((d, d), hermitian=True, name=f"Q_{label}")

        Q_pt = partial_transpose_expr_fast(Q, dims, cut)

        # P_M has been eliminated:
        #
        #     P_M = W - Q_M^{T_M}
        #
        # so P_M >= 0 becomes W - Q_pt >= 0.
        constraints.extend(
            [
                Q >> 0,
                W - Q_pt >> 0,
            ]
        )

        if formulation == "monotone":
            constraints.extend(
                [
                    I - Q >> 0,
                    I - W + Q_pt >> 0,
                ]
            )

    if formulation == "paper_witness":
        constraints.append(cp.trace(W) == 1)

    if use_real_sdp:
        objective = cp.Minimize(cp.trace(rho_sdp @ W))
    else:
        objective = cp.Minimize(cp.real(cp.trace(rho_sdp @ W)))

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

class CompiledGMNSDP:
    def __init__(
        self,
        dims: Sequence[int],
        *,
        formulation: Literal["monotone", "paper_witness"] = "monotone",
        real: bool = True,
    ):
        self.dims = tuple(int(x) for x in dims)
        self.formulation = formulation
        self.real = bool(real)

        d = int(np.prod(self.dims))
        m = len(self.dims)

        self.d = d
        self.cuts = inequivalent_bipartitions(m)

        I = np.eye(d)

        if self.real:
            self.rho_param = cp.Parameter((d, d), name="rho")
            W = cp.Variable((d, d), symmetric=True, name="W")
        else:
            self.rho_param = cp.Parameter((d, d), complex=True, name="rho")
            W = cp.Variable((d, d), hermitian=True, name="W")

        constraints: list[cp.Constraint] = []

        for cut in self.cuts:
            label = "_".join(str(i) for i in cut)

            if self.real:
                Q = cp.Variable((d, d), symmetric=True, name=f"Q_{label}")
            else:
                Q = cp.Variable((d, d), hermitian=True, name=f"Q_{label}")

            Q_pt = partial_transpose_expr_fast(Q, self.dims, cut)

            constraints.extend(
                [
                    Q >> 0,
                    W - Q_pt >> 0,
                ]
            )

            if formulation == "monotone":
                constraints.extend(
                    [
                        I - Q >> 0,
                        I - W + Q_pt >> 0,
                    ]
                )

        if formulation == "paper_witness":
            constraints.append(cp.trace(W) == 1)

        if self.real:
            objective = cp.Minimize(cp.trace(self.rho_param @ W))
        else:
            objective = cp.Minimize(cp.real(cp.trace(self.rho_param @ W)))

        self.problem = cp.Problem(objective, constraints)

    def solve(
        self,
        rho: np.ndarray,
        *,
        solver: str = cp.MOSEK,
        solver_options: dict | None = None,
        zero_tol: float = 1e-8,
    ) -> float:
        rho = np.asarray(rho, dtype=np.complex128)
        rho = 0.5 * (rho + rho.conj().T)

        if rho.shape != (self.d, self.d):
            raise ValueError(
                f"rho has shape {rho.shape}, expected {(self.d, self.d)}."
            )

        if self.real:
            self.rho_param.value = rho.real
        else:
            self.rho_param.value = rho

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
            return raw_score if raw_score > zero_tol else 0.0

        return raw_score
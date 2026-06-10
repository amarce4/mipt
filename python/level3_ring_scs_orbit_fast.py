from __future__ import annotations

"""
Exact level-3 ring-inflation SDP feasibility solver using direct SCS data
assembly, exact permutation-symmetry orbit parametrization, and direct marginal
map construction.

This is the intended replacement for the earlier full-coordinate direct-SCS
builder.  It avoids both CVXPY ConeMatrixStuffing and scipy sparse products of
large full-coordinate maps.

Mathematics:
  - Same level-3 SDP constraints as the debug.py transcription.
  - Still an SDP: full PSD and PPT cones are passed to SCS.
  - No relaxation from symmetry reduction.  Each 512x512 Hermitian variable is
    parametrized exactly in the fixed-point subspace of its copy-permutation
    symmetry group.

Subsystem order:
    0=A1, 1=B1, 2=C1, 3=A2, 4=B2, 5=C2, 6=A3, 7=B3, 8=C3.

Requirements:
    pip install numpy scipy scs
"""

from dataclasses import dataclass
import re
from functools import lru_cache
from itertools import product
from math import prod, sqrt
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]

_DIMS_9 = (2, 2, 2, 2, 2, 2, 2, 2, 2)
_D_FULL = 512
_RHO_DIM = 8
_SQRT2 = sqrt(2.0)


def _basis_index(indices: Sequence[int], dims: Sequence[int]) -> int:
    out = 0
    for i, d in zip(indices, dims):
        out = out * int(d) + int(i)
    return out


@lru_cache(maxsize=None)
def _basis_permutation_for_keep(dims: tuple[int, ...], keep: tuple[int, ...]) -> np.ndarray:
    """perm such that X_keep[a,b] = X[perm[a], perm[b]]."""
    dims = tuple(int(x) for x in dims)
    keep = tuple(int(x) for x in keep)
    n = len(dims)
    if len(keep) != n or sorted(keep) != list(range(n)):
        raise ValueError(f"keep={keep} is not a full subsystem permutation")
    d = prod(dims)
    out_dims = tuple(dims[i] for i in keep)
    perm = np.empty(d, dtype=np.int64)
    for a_flat in range(d):
        a_multi = np.unravel_index(a_flat, out_dims, order="C")
        ket = [0] * n
        for pos, subsystem in enumerate(keep):
            ket[subsystem] = int(a_multi[pos])
        perm[a_flat] = _basis_index(ket, dims)
    return perm


def _lower_index(i: int, j: int, n: int) -> int:
    """Lower-triangular F-order index for i >= j."""
    return j * n - (j * (j + 1)) // 2 + i


class _SignedDSU:
    """Signed union-find: value[x] = sign * value[root]."""

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.sign = np.ones(n, dtype=np.int8)
        self.rank = np.zeros(n, dtype=np.int8)
        self.forced_zero = np.zeros(n, dtype=bool)

    def find(self, x: int) -> tuple[int, int]:
        p = int(self.parent[x])
        if p == x:
            return x, 1
        r, s = self.find(p)
        self.parent[x] = r
        self.sign[x] = np.int8(int(self.sign[x]) * s)
        return r, int(self.sign[x])

    def union(self, x: int, y: int, relation_sign: int) -> None:
        # impose value[y] = relation_sign * value[x]
        relation_sign = 1 if relation_sign >= 0 else -1
        rx, sx = self.find(x)
        ry, sy = self.find(y)
        if rx == ry:
            if sy != relation_sign * sx:
                self.forced_zero[rx] = True
            return
        rel = relation_sign * sx * sy  # value[ry] = rel * value[rx]
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
            self.sign[rx] = np.int8(rel)
            self.forced_zero[ry] = bool(self.forced_zero[ry] or self.forced_zero[rx])
        else:
            self.parent[ry] = rx
            self.sign[ry] = np.int8(rel)
            self.forced_zero[rx] = bool(self.forced_zero[rx] or self.forced_zero[ry])
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1

    def force_zero(self, x: int) -> None:
        r, _ = self.find(x)
        self.forced_zero[r] = True


@dataclass(frozen=True)
class _ReducedMapArrays:
    local_size: int
    real_orbits: int
    imag_orbits: int
    real_col: np.ndarray       # length 512^2, local column for Re(X_flat)
    imag_col: np.ndarray       # length 512^2, local column for Im(X_flat), -1 if zero/real-only
    imag_coeff: np.ndarray     # length 512^2, +/-1 coefficient for imag_col


@lru_cache(maxsize=None)
def _reduced_arrays_for_generators(
    generator_keeps: tuple[tuple[int, ...], ...],
    complex_problem: bool,
) -> _ReducedMapArrays:
    """Exact fixed-point parametrization arrays for one 512x512 Hermitian variable."""
    n = _D_FULL
    n_lower = n * (n + 1) // 2
    dsu = _SignedDSU(n_lower)

    for i in range(n):
        dsu.force_zero(_lower_index(i, i, n))

    perms = [_basis_permutation_for_keep(_DIMS_9, keep) for keep in generator_keeps]
    for perm in perms:
        for j in range(n):
            pj = int(perm[j])
            for i in range(j, n):
                pi = int(perm[i])
                k1 = _lower_index(i, j, n)
                if pi >= pj:
                    k2 = _lower_index(pi, pj, n)
                    s = 1
                else:
                    k2 = _lower_index(pj, pi, n)
                    s = -1
                dsu.union(k1, k2, s)

    roots = np.empty(n_lower, dtype=np.int64)
    for k in range(n_lower):
        r, _ = dsu.find(k)
        roots[k] = r
    for k in range(n_lower):
        r, _ = dsu.find(k)
        rr, _ = dsu.find(r)
        if dsu.forced_zero[r]:
            dsu.forced_zero[rr] = True

    unique_roots = sorted(set(int(r) for r in roots))
    real_index = {r: idx for idx, r in enumerate(unique_roots)}
    real_count = len(real_index)

    if complex_problem:
        imag_roots = [r for r in unique_roots if not bool(dsu.forced_zero[dsu.find(r)[0]])]
        imag_index = {r: idx for idx, r in enumerate(imag_roots)}
        imag_count = len(imag_index)
    else:
        imag_index = {}
        imag_count = 0

    total = real_count + imag_count
    real_col = np.empty(n * n, dtype=np.int32)
    imag_col = np.full(n * n, -1, dtype=np.int32)
    imag_coeff = np.zeros(n * n, dtype=np.int8)

    for col_b in range(n):
        for row_a in range(n):
            flat = row_a + col_b * n
            if row_a >= col_b:
                k = _lower_index(row_a, col_b, n)
                conj_sign = 1
            else:
                k = _lower_index(col_b, row_a, n)
                conj_sign = -1
            root, s_to_root = dsu.find(k)
            root = int(root)
            real_col[flat] = real_index[root]
            if complex_problem and root in imag_index:
                imag_col[flat] = real_count + imag_index[root]
                imag_coeff[flat] = np.int8(conj_sign * s_to_root)

    return _ReducedMapArrays(
        local_size=total,
        real_orbits=real_count,
        imag_orbits=imag_count,
        real_col=real_col,
        imag_col=imag_col,
        imag_coeff=imag_coeff,
    )


@dataclass(frozen=True)
class _ReducedHermitianVar:
    name: str
    offset: int
    complex_problem: bool
    local_size: int
    arrays: _ReducedMapArrays

    @property
    def size(self) -> int:
        return self.local_size

    @property
    def real_orbits(self) -> int:
        return self.arrays.real_orbits

    @property
    def imag_orbits(self) -> int:
        return self.arrays.imag_orbits


def _make_reduced_var(
    name: str,
    offset: int,
    *,
    generator_keeps: Sequence[Sequence[int]],
    complex_problem: bool,
) -> _ReducedHermitianVar:
    key = tuple(tuple(int(x) for x in keep) for keep in generator_keeps)
    arrays = _reduced_arrays_for_generators(key, bool(complex_problem))
    return _ReducedHermitianVar(
        name=name,
        offset=offset,
        complex_problem=bool(complex_problem),
        local_size=arrays.local_size,
        arrays=arrays,
    )


def _target_unscaled_lower(mat: np.ndarray, complex_problem: bool) -> np.ndarray:
    n = mat.shape[0]
    H = np.asarray(mat, dtype=np.complex128)
    vals: list[float] = []
    for j in range(n):
        for i in range(j, n):
            if not complex_problem:
                vals.append(float(np.real(H[i, j])))
            elif i == j:
                vals.append(float(np.real(H[i, j])))
            else:
                vals.append(float(np.real(H[i, j])))
                vals.append(float(np.imag(H[i, j])))
    return np.asarray(vals, dtype=np.float64)


@lru_cache(maxsize=None)
def _entry_data_for_keep_pt(
    keep: tuple[int, ...],
    pt: tuple[int, ...],
    complex_problem: bool,
) -> tuple[int, list[tuple[int, int, int, bool]]]:
    """
    Return output dimension and rows for lower-Hermitian/cvec construction.

    Each tuple is (row, full_flat, imag_row, offdiag), where imag_row=-1 for
    real-only/diagonal rows.  Rows are unscaled; PSD scaling is added later.
    """
    keep = tuple(int(k) for k in keep)
    pt = tuple(int(s) for s in pt)
    nsys = len(_DIMS_9)
    trace = tuple(i for i in range(nsys) if i not in keep)
    out_dims = tuple(_DIMS_9[i] for i in keep)
    d_out = prod(out_dims)
    trace_ranges = [range(_DIMS_9[i]) for i in trace]

    entries: list[tuple[int, int, int, bool]] = []
    row = 0
    for j in range(d_out):
        bra0 = list(np.unravel_index(j, out_dims, order="C"))
        for i in range(j, d_out):
            ket0 = list(np.unravel_index(i, out_dims, order="C"))
            offdiag = i != j
            real_row = row
            imag_row = row + 1 if (complex_problem and offdiag) else -1
            row += 2 if (complex_problem and offdiag) else 1

            # F = PT(M).  Therefore F[i,j] = M[i_pt,j_pt].
            ket = ket0.copy()
            bra = bra0.copy()
            for s in pt:
                ket[s], bra[s] = bra[s], ket[s]

            for traced_vals in (product(*trace_ranges) if trace else [()]):
                full_ket = [0] * nsys
                full_bra = [0] * nsys
                for pos, subsystem in enumerate(keep):
                    full_ket[subsystem] = int(ket[pos])
                    full_bra[subsystem] = int(bra[pos])
                for pos, subsystem in enumerate(trace):
                    val = int(traced_vals[pos])
                    full_ket[subsystem] = val
                    full_bra[subsystem] = val
                I = _basis_index(full_ket, _DIMS_9)
                J = _basis_index(full_bra, _DIMS_9)
                entries.append((real_row, I + J * _D_FULL, imag_row, offdiag))
    return d_out, entries


def _direct_lower_map(
    var: _ReducedHermitianVar,
    total_vars: int,
    *,
    keep: Sequence[int] | None,
    pt: Sequence[int] | None = None,
    scaled_psd: bool = False,
) -> tuple[sp.csc_matrix, int]:
    keep_t = tuple(range(9)) if keep is None else tuple(int(k) for k in keep)
    pt_t = tuple() if pt is None else tuple(int(s) for s in pt)
    d_out, entries = _entry_data_for_keep_pt(keep_t, pt_t, var.complex_problem)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rc = var.arrays.real_col
    ic = var.arrays.imag_col
    isgn = var.arrays.imag_coeff
    off = var.offset

    for real_row, flat, imag_row, offdiag in entries:
        scale = _SQRT2 if scaled_psd and offdiag else 1.0
        rows.append(real_row)
        cols.append(off + int(rc[flat]))
        data.append(scale)
        if var.complex_problem and imag_row >= 0:
            c = int(ic[flat])
            if c >= 0:
                rows.append(imag_row)
                cols.append(off + c)
                data.append(scale * float(isgn[flat]))

    n_rows = d_out * d_out if var.complex_problem else d_out * (d_out + 1) // 2
    M = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, total_vars)).tocsc()
    return M, d_out


def _direct_trace_map(var: _ReducedHermitianVar, total_vars: int) -> sp.csc_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(_D_FULL):
        flat = i + i * _D_FULL
        rows.append(0)
        cols.append(var.offset + int(var.arrays.real_col[flat]))
        data.append(1.0)
    return sp.coo_matrix((data, (rows, cols)), shape=(1, total_vars)).tocsc()


@dataclass
class PreparedLevel3SCSData:
    data: dict[str, Any]
    cone: dict[str, Any]
    complex_problem: bool
    total_vars: int
    total_rows: int
    nnz_A: int
    target_slice: slice
    b_base: np.ndarray
    sigma_size: int
    tau_size: int
    gamma_size: int
    psd_dims: list[int]

    def b_for(self, rho: ComplexMatrix, t_value: float) -> np.ndarray:
        rho = _validate_rho(rho)
        if not self.complex_problem:
            max_imag = float(np.max(np.abs(np.imag(rho))))
            if max_imag > 1e-10:
                raise ValueError(f"complex_problem=False but rho has max imaginary part {max_imag:g}")
            rho_use = np.real(rho)
        else:
            rho_use = rho
        target = float(t_value) * rho_use + (1.0 - float(t_value)) * np.eye(8, dtype=np.complex128) / 8.0
        b = self.b_base.copy()
        b[self.target_slice] = _target_unscaled_lower(target, self.complex_problem)
        return b


class Level3RingInflationOrbitSCSBuilder:
    """Build direct SCS feasibility data once; update b for each t/rho."""

    def __init__(self, *, complex_problem: bool = False, verbose: bool = True) -> None:
        self.complex_problem = bool(complex_problem)
        self.verbose = bool(verbose)
        off = 0
        self.gamma = _make_reduced_var(
            "gamma", off,
            generator_keeps=[[3, 4, 5, 6, 7, 8, 0, 1, 2]],
            complex_problem=self.complex_problem,
        )
        off += self.gamma.size
        self.tau = _make_reduced_var(
            "tau", off,
            generator_keeps=[[3, 4, 5, 0, 1, 2, 6, 7, 8]],
            complex_problem=self.complex_problem,
        )
        off += self.tau.size
        self.sigma = _make_reduced_var(
            "sigma", off,
            generator_keeps=[
                [3, 4, 5, 0, 1, 2, 6, 7, 8],
                [0, 1, 2, 6, 7, 8, 3, 4, 5],
            ],
            complex_problem=self.complex_problem,
        )
        off += self.sigma.size
        self.total_vars = off

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _eq_map(self, var: _ReducedHermitianVar, keep: Sequence[int] | None) -> sp.csc_matrix:
        M, _ = _direct_lower_map(var, self.total_vars, keep=keep, scaled_psd=False)
        return M

    def _psd_map(
        self,
        var: _ReducedHermitianVar,
        keep: Sequence[int] | None = None,
        pt: Sequence[int] | None = None,
    ) -> tuple[sp.csc_matrix, int]:
        return _direct_lower_map(var, self.total_vars, keep=keep, pt=pt, scaled_psd=True)

    def build(self) -> PreparedLevel3SCSData:
        eq_blocks: list[sp.csc_matrix] = []
        eq_b: list[np.ndarray] = []
        psd_blocks: list[sp.csc_matrix] = []
        psd_dims: list[int] = []
        eq_row_cursor = 0

        def add_eq(A: sp.spmatrix, b: np.ndarray | None = None, label: str = "") -> slice:
            nonlocal eq_row_cursor
            A = A.tocsc()
            self._log(f"  equality {label}: rows={A.shape[0]:,}, nnz={A.nnz:,}")
            eq_blocks.append(A)
            bb = np.zeros(A.shape[0], dtype=np.float64) if b is None else np.asarray(b, dtype=np.float64)
            eq_b.append(bb)
            sl = slice(eq_row_cursor, eq_row_cursor + A.shape[0])
            eq_row_cursor += A.shape[0]
            return sl

        def add_psd(M: sp.spmatrix, dim: int, label: str) -> None:
            M = M.tocsc()
            self._log(f"  PSD {label}: dim={dim}, rows={M.shape[0]:,}, nnz={M.nnz:,}")
            psd_blocks.append((-M).tocsc())
            psd_dims.append(int(dim))

        self._log(
            "Building orbit-reduced/direct-map SCS structure, "
            f"complex={self.complex_problem}, vars={self.total_vars:,} "
            f"[gamma={self.gamma.size:,}, tau={self.tau.size:,}, sigma={self.sigma.size:,}]"
        )
        self._log(
            f"  orbit counts: gamma(real={self.gamma.real_orbits:,}, imag={self.gamma.imag_orbits:,}), "
            f"tau(real={self.tau.real_orbits:,}, imag={self.tau.imag_orbits:,}), "
            f"sigma(real={self.sigma.real_orbits:,}, imag={self.sigma.imag_orbits:,})"
        )

        n_target = _RHO_DIM * _RHO_DIM if self.complex_problem else _RHO_DIM * (_RHO_DIM + 1) // 2
        target_slice = add_eq(
            self._eq_map(self.sigma, [0, 1, 2]),
            np.zeros(n_target, dtype=np.float64),
            "sigma_012 = target",
        )

        for var in (self.sigma, self.tau, self.gamma):
            add_eq(_direct_trace_map(var, self.total_vars), np.array([1.0]), f"Tr({var.name})=1")

        marginal_pairs = [
            (self.gamma, [7, 8, 0, 1, 2, 4, 5], self.tau,   [4, 5, 0, 1, 2, 7, 8], "gamma_7801245 = tau_4501278"),
            (self.tau,   [1, 2, 4, 5, 6, 7, 8], self.sigma, [1, 2, 4, 5, 6, 7, 8], "tau_1245678 = sigma_1245678"),
            (self.gamma, [0, 1, 2, 3, 4, 6, 7], self.tau,   [0, 1, 2, 3, 4, 6, 7], "gamma_0123467 = tau_0123467"),
            (self.tau,   [0, 1, 3, 4, 6, 7, 8], self.sigma, [0, 1, 3, 4, 6, 7, 8], "tau_0134678 = sigma_0134678"),
            (self.gamma, [8, 0, 1, 2, 3, 5, 6], self.tau,   [5, 0, 1, 2, 3, 8, 6], "gamma_8012356 = tau_5012386"),
            (self.tau,   [2, 3, 5, 0, 6, 7, 8], self.sigma, [2, 0, 5, 3, 6, 7, 8], "tau_2350678 = sigma_2053678"),
        ]
        for v1, k1, v2, k2, label in marginal_pairs:
            add_eq(self._eq_map(v1, k1) - self._eq_map(v2, k2), label=label)

        full_keep = list(range(9))
        for var in (self.sigma, self.tau, self.gamma):
            M, dim = self._psd_map(var, keep=full_keep)
            add_psd(M, dim, f"{var.name} >= 0")

        for var, pt, label in [
            (self.tau, [6, 7, 8], "tau^T678 >= 0"),
            (self.sigma, [6, 7, 8], "sigma^T678 >= 0"),
        ]:
            M, dim = self._psd_map(var, keep=full_keep, pt=pt)
            add_psd(M, dim, label)

        gamma_ppts = [
            ([0, 2, 3, 4, 5, 6, 7], [0],       "gamma_0234567^T0"),
            ([0, 1, 2, 4, 5, 6, 7], [0, 1, 2], "gamma_0124567^T012"),
            ([1, 3, 4, 5, 6, 7, 8], [0],       "gamma_1345678^T1"),
            ([1, 2, 3, 5, 6, 7, 8], [0, 1, 2], "gamma_1235678^T123"),
            ([2, 4, 5, 6, 7, 8, 0], [0],       "gamma_2456780^T2"),
            ([2, 3, 4, 6, 7, 8, 0], [0, 1, 2], "gamma_2346780^T234"),
        ]
        for keep, pt, label in gamma_ppts:
            M, dim = self._psd_map(self.gamma, keep=keep, pt=pt)
            add_psd(M, dim, label)

        tau_ppts = [
            ([0, 2, 3, 4, 6, 7, 8], [0],       "tau_0234678^T1"),
            ([0, 2, 3, 4, 6, 7, 8], [1, 2, 3], "tau_0234678^T234"),
            ([1, 3, 4, 5, 6, 7, 8], [0],       "tau_1345678^T2"),
            ([1, 3, 4, 5, 6, 7, 8], [1, 2, 3], "tau_1345678^T345"),
            ([2, 4, 5, 0, 6, 7, 8], [0],       "tau_2450678^T3"),
            ([2, 4, 5, 0, 6, 7, 8], [1, 2, 3], "tau_2450678^T450"),
        ]
        for keep, pt, label in tau_ppts:
            M, dim = self._psd_map(self.tau, keep=keep, pt=pt)
            add_psd(M, dim, label)

        A = sp.vstack(eq_blocks + psd_blocks, format="csc")
        b_base = np.concatenate(eq_b + [np.zeros(B.shape[0], dtype=np.float64) for B in psd_blocks])
        c = np.zeros(self.total_vars, dtype=np.float64)
        cone: dict[str, Any] = {"z": int(sum(B.shape[0] for B in eq_blocks))}
        if self.complex_problem:
            cone["cs"] = psd_dims
        else:
            cone["s"] = psd_dims
        data = {"A": A, "b": b_base.copy(), "c": c}
        self._log(f"Built A: rows={A.shape[0]:,}, cols={A.shape[1]:,}, nnz={A.nnz:,}, zero_rows={cone['z']:,}")
        return PreparedLevel3SCSData(
            data=data,
            cone=cone,
            complex_problem=self.complex_problem,
            total_vars=self.total_vars,
            total_rows=A.shape[0],
            nnz_A=A.nnz,
            target_slice=target_slice,
            b_base=b_base,
            sigma_size=self.sigma.size,
            tau_size=self.tau.size,
            gamma_size=self.gamma.size,
            psd_dims=psd_dims,
        )


def _validate_rho(rho: ComplexMatrix) -> ComplexMatrix:
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (8, 8):
        raise ValueError(f"rho must be 8x8, got {rho.shape}")
    rho = (rho + rho.conj().T) / 2
    tr = np.trace(rho)
    if abs(tr) == 0:
        raise ValueError("rho has zero trace")
    return rho / tr


_PREPARED_CACHE: dict[bool, PreparedLevel3SCSData] = {}


def prepare_level3_ring_scs(
    *,
    complex_problem: bool,
    verbose_build: bool = True,
    rebuild: bool = False,
) -> PreparedLevel3SCSData:
    key = bool(complex_problem)
    if rebuild or key not in _PREPARED_CACHE:
        builder = Level3RingInflationOrbitSCSBuilder(complex_problem=key, verbose=verbose_build)
        _PREPARED_CACHE[key] = builder.build()
    return _PREPARED_CACHE[key]



def _scs_solve_compat(scs_module: Any, data: dict[str, Any], cone: dict[str, Any], settings: dict[str, Any], *, verbose: bool) -> dict[str, Any]:
    """Call scs.solve while tolerating backend-specific unsupported settings.

    scs-python has multiple compiled backends.  Depending on the installed wheel,
    a setting documented for one backend can be rejected by the low-level
    constructor before the solve begins.  This helper removes only the rejected
    keyword and retries; it does not change the cone data or the SDP.
    """
    remaining = dict(settings)
    dropped: list[str] = []
    while True:
        try:
            return scs_module.solve(data, cone, **remaining)
        except TypeError as exc:
            message = str(exc)
            m = re.search(r"'([^']+)' is an invalid keyword argument", message)
            if not m:
                raise
            key = m.group(1)
            if key not in remaining:
                raise
            dropped.append(key)
            remaining.pop(key)
            if verbose:
                print(f"SCS backend rejected setting {key!r}; retrying without it.")
            # If the GPU flag itself is rejected, there is no GPU-enabled SCS
            # backend in this environment.  Retrying on CPU is mathematically the
            # same SDP, only a different linear-system backend.
            continue

def solve_feasibility_at_t(
    rho: ComplexMatrix,
    t_value: float,
    *,
    complex_problem: bool | None = None,
    prepared: PreparedLevel3SCSData | None = None,
    verbose_build: bool = True,
    verbose_scs: bool = True,
    max_iters: int = 20_000,
    eps_abs: float = 2e-4,
    eps_rel: float = 2e-4,
    normalize: bool = True,
    scale: float = 1.0,
    acceleration_lookback: int = 0,
    use_indirect: bool = False,
    gpu: bool = False,
    **scs_options: Any,
) -> tuple[bool, dict[str, Any], PreparedLevel3SCSData]:
    rho = _validate_rho(rho)
    if complex_problem is None:
        complex_problem = bool(np.max(np.abs(np.imag(rho))) > 1e-10)
    if prepared is None:
        prepared = prepare_level3_ring_scs(complex_problem=bool(complex_problem), verbose_build=verbose_build)
    if prepared.complex_problem != bool(complex_problem):
        raise ValueError("prepared data complex_problem does not match requested complex_problem")

    try:
        import scs
    except ImportError as exc:
        raise RuntimeError("This direct implementation requires scs: pip install scs") from exc

    data = {"A": prepared.data["A"], "b": prepared.b_for(rho, t_value), "c": prepared.data["c"]}

    # Only pass non-default backend-selection knobs when explicitly requested.
    # Some scs-python wheels expose these through CVXPY but not through the raw
    # low-level SCS constructor, so passing use_indirect=False can raise
    # TypeError before solving starts.
    settings = dict(
        verbose=verbose_scs,
        max_iters=int(max_iters),
        eps_abs=float(eps_abs),
        eps_rel=float(eps_rel),
        normalize=bool(normalize),
        scale=float(scale),
        acceleration_lookback=int(acceleration_lookback),
    )
    if use_indirect:
        settings["use_indirect"] = True
    if gpu:
        settings["gpu"] = True
    settings.update(scs_options)
    sol = _scs_solve_compat(scs, data, prepared.cone, settings, verbose=verbose_scs)
    status = str(sol.get("info", {}).get("status", "")).lower()
    feasible = ("solved" in status) or ("optimal" in status)
    return feasible, sol, prepared


@dataclass
class Level3SCSResult:
    score: float
    lower_bound: float
    upper_bound: float
    iterations: int
    last_status: str
    complex_problem: bool
    rows: int
    cols: int
    nnz: int
    sigma_vars: int
    tau_vars: int
    gamma_vars: int


def solve_level3_ring_inflation_scs(
    rho: ComplexMatrix,
    *,
    complex_problem: bool | None = None,
    bisect_tol: float = 5e-3,
    max_bisect_steps: int = 8,
    start_low: float = 0.0,
    start_high: float = 1.0,
    verbose_build: bool = True,
    verbose_scs: bool = True,
    rebuild: bool = False,
    **scs_options: Any,
) -> Level3SCSResult:
    rho = _validate_rho(rho)
    if complex_problem is None:
        complex_problem = bool(np.max(np.abs(np.imag(rho))) > 1e-10)
    prepared = prepare_level3_ring_scs(
        complex_problem=bool(complex_problem),
        verbose_build=verbose_build,
        rebuild=rebuild,
    )

    lo = float(start_low)
    hi = float(start_high)
    last_status = "not run"
    steps = 0
    while steps < int(max_bisect_steps) and hi - lo > float(bisect_tol):
        mid = 0.5 * (lo + hi)
        feasible, sol, _ = solve_feasibility_at_t(
            rho,
            mid,
            complex_problem=bool(complex_problem),
            prepared=prepared,
            verbose_build=False,
            verbose_scs=verbose_scs,
            **scs_options,
        )
        last_status = str(sol.get("info", {}).get("status", "unknown"))
        if feasible:
            lo = mid
        else:
            hi = mid
        steps += 1
        print(
            f"bisection step {steps}: t={mid:.8f}, feasible={feasible}, "
            f"status={last_status}, bracket=[{lo:.8f}, {hi:.8f}]",
            flush=True,
        )

    return Level3SCSResult(
        score=lo,
        lower_bound=lo,
        upper_bound=hi,
        iterations=steps,
        last_status=last_status,
        complex_problem=bool(complex_problem),
        rows=prepared.total_rows,
        cols=prepared.total_vars,
        nnz=prepared.nnz_A,
        sigma_vars=prepared.sigma_size,
        tau_vars=prepared.tau_size,
        gamma_vars=prepared.gamma_size,
    )


def reset_cache() -> None:
    _PREPARED_CACHE.clear()
    _basis_permutation_for_keep.cache_clear()
    _reduced_arrays_for_generators.cache_clear()
    _entry_data_for_keep_pt.cache_clear()


if __name__ == "__main__":
    v = np.zeros(8, dtype=np.complex128)
    v[0] = 1 / np.sqrt(2)
    v[7] = 1 / np.sqrt(2)
    rho = np.outer(v, v.conj())
    prepared = prepare_level3_ring_scs(complex_problem=False, verbose_build=True, rebuild=True)
    print(prepared)

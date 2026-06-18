"""
Write MIPT reduced-density-matrix .bin files from Python.

Two binary formats are supported:

1. MIPTRHO2 (legacy rho3 / GMN format)
   - fixed 3-qubit 8x8 matrices
   - one record contains all requested 3-qubit subsystems

2. MIPTTMI1 (four-block TMI format)
   - same layout as the C++ TMI writer/readers
   - one record contains exactly four matrices, ordered:
       AB, AC, BC, D
   - for total_qubits=N, block=N/4
       A=[0, ..., block-1]
       B=[block, ..., 2*block-1]
       C=[2*block, ..., 3*block-1]
       D=[3*block, ..., 4*block-1]
     so AB/AC/BC have N/2 modes and D has N/4 modes.

The TMI format is intended to be consumed by the patched C++ tmi.exe:

    ./tmi.exe 0 rho_tmi_python.bin tmi_qubit.csv
    ./tmi.exe 1 rho_tmi_python.bin tmi_fermion.csv
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Legacy MIPTRHO2 rho3 format.
# ---------------------------------------------------------------------------

_MAGIC = b"MIPTRHO2"
_ENDIAN_MARKER = 0x01020304
_VERSION = 2
_HEADER_STRUCT = struct.Struct("<10IQdd")
_RECORD_HEADER_STRUCT = struct.Struct("<IId")
_KEPT_QUBITS = 3
_MATRIX_DIMENSION = 8

# ---------------------------------------------------------------------------
# Four-block TMI format; mirrors the C++ TmiMatrixWriter/TmiMatrixReader.
# ---------------------------------------------------------------------------

_TMI_MAGIC = b"MIPTTMI1"
_TMI_ENDIAN_MARKER = 0x01020304
_TMI_VERSION = 1
_TMI_HEADER_STRUCT = struct.Struct("<9IQdd")
_TMI_RECORD_HEADER_STRUCT = struct.Struct("<IId")
_TMI_TERM_NAME_BYTES = 8
_TMI_TERM_HEADER_STRUCT = struct.Struct("<8sII")
_TMI_TERM_ORDER = ("AB", "AC", "BC", "D")

ComplexMatrix = NDArray[np.complex128]
QubitTriple = tuple[int, int, int]
TrajectoryRecord = tuple[int, int, float, Sequence[ArrayLike]]


@dataclass(frozen=True)
class TmiTerm:
    name: str
    modes: tuple[int, ...]

    @property
    def kept_qubits(self) -> int:
        return len(self.modes)

    @property
    def matrix_dimension(self) -> int:
        return 1 << self.kept_qubits


TmiTrajectoryRecord = tuple[int, int, float, Sequence[ArrayLike]]


def ring_subsystem_qubits(
    total_qubits: int,
    *,
    kept_qubits: int = 3,
    closed: bool = True,
) -> list[QubitTriple]:
    """
    Return consecutive 3-qubit subsystem labels for a 1D chain/ring.

    For total_qubits=8 and closed=True:
        [(0,1,2), (1,2,3), ..., (7,0,1)]

    For closed=False:
        [(0,1,2), ..., (5,6,7)]
    """
    if kept_qubits != 3:
        raise ValueError("This rho3 writer is fixed to three-qubit 8x8 RDMs; kept_qubits must be 3.")
    if total_qubits < kept_qubits:
        raise ValueError("total_qubits must be at least kept_qubits.")

    if closed:
        return [
            tuple((start + offset) % total_qubits for offset in range(kept_qubits))  # type: ignore[misc]
            for start in range(total_qubits)
        ]

    return [
        tuple(start + offset for offset in range(kept_qubits))  # type: ignore[misc]
        for start in range(total_qubits - kept_qubits + 1)
    ]


def tmi_subsystem_terms(total_qubits: int) -> list[TmiTerm]:
    """
    Return the four TMI terms expected by the C++ MIPTTMI1 reader.

    Stored order is exactly: AB, AC, BC, D.
    """
    n = int(total_qubits)
    if n < 4 or (n % 4) != 0:
        raise ValueError("TMI matrix writing requires total_qubits >= 4 and divisible by 4.")

    block = n // 4
    A = tuple(range(0, block))
    B = tuple(range(block, 2 * block))
    C = tuple(range(2 * block, 3 * block))
    D = tuple(range(3 * block, 4 * block))

    return [
        TmiTerm("AB", A + B),
        TmiTerm("AC", A + C),
        TmiTerm("BC", B + C),
        TmiTerm("D", D),
    ]


def tmi_subsystem_qubits(total_qubits: int) -> list[tuple[int, ...]]:
    """Return only the mode lists for the four TMI terms, in AB/AC/BC/D order."""
    return [term.modes for term in tmi_subsystem_terms(total_qubits)]


def _as_uint32(value: int, name: str) -> int:
    value = int(value)
    if value < 0 or value > np.iinfo(np.uint32).max:
        raise ValueError(f"{name}={value} is outside uint32 range.")
    return value


def _as_uint64(value: int, name: str) -> int:
    value = int(value)
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise ValueError(f"{name}={value} is outside uint64 range.")
    return value


def _normalise_subsystem_qubits(subsystem_qubits: Sequence[Sequence[int]]) -> list[QubitTriple]:
    out: list[QubitTriple] = []
    for i, triple in enumerate(subsystem_qubits):
        if len(triple) != _KEPT_QUBITS:
            raise ValueError(f"subsystem_qubits[{i}] must contain exactly 3 qubit indices.")
        out.append(tuple(_as_uint32(q, f"subsystem_qubits[{i}]") for q in triple))  # type: ignore[arg-type]
    if not out:
        raise ValueError("subsystem_qubits must contain at least one subsystem.")
    return out


def _normalise_tmi_terms(total_qubits: int, terms: Optional[Sequence[TmiTerm | tuple[str, Sequence[int]]]]) -> list[TmiTerm]:
    if terms is None:
        out = tmi_subsystem_terms(total_qubits)
    else:
        out = []
        for item in terms:
            if isinstance(item, TmiTerm):
                out.append(item)
            else:
                name, modes = item
                out.append(TmiTerm(str(name), tuple(int(q) for q in modes)))

    if len(out) != 4:
        raise ValueError("TMI files must contain exactly four terms: AB, AC, BC, D.")
    names = tuple(term.name for term in out)
    if names != _TMI_TERM_ORDER:
        raise ValueError(f"TMI term order must be {_TMI_TERM_ORDER}, got {names}.")

    for term in out:
        if not term.modes:
            raise ValueError(f"TMI term {term.name} has no modes.")
        if len(set(term.modes)) != len(term.modes):
            raise ValueError(f"TMI term {term.name} contains duplicate modes.")
        for q in term.modes:
            if q < 0 or q >= int(total_qubits):
                raise ValueError(f"TMI term {term.name} contains out-of-range mode {q}.")

    block = int(total_qubits) // 4
    expected_sizes = (2 * block, 2 * block, 2 * block, block)
    got_sizes = tuple(term.kept_qubits for term in out)
    if got_sizes != expected_sizes:
        raise ValueError(f"TMI term sizes must be {expected_sizes}, got {got_sizes}.")

    return out


def _prepare_rho_fixed_dim(
    rho: ArrayLike,
    *,
    matrix_dimension: int,
    hermitize: bool,
    normalize_trace: bool,
    check_finite: bool,
) -> ComplexMatrix:
    arr = np.asarray(rho, dtype=np.complex128)
    expected_shape = (int(matrix_dimension), int(matrix_dimension))
    if arr.shape != expected_shape:
        raise ValueError(f"Each rho must have shape {expected_shape}, got {arr.shape}.")
    if check_finite and not np.all(np.isfinite(arr)):
        raise ValueError("rho contains NaN or infinite values.")

    arr = np.array(arr, dtype=np.complex128, copy=True)

    if hermitize:
        arr = 0.5 * (arr + arr.conj().T)

    if normalize_trace:
        tr = np.trace(arr)
        if abs(tr) == 0.0:
            raise ValueError("Cannot normalize rho with zero trace.")
        arr = arr / tr

    return arr


def _prepare_rho(
    rho: ArrayLike,
    *,
    hermitize: bool,
    normalize_trace: bool,
    check_finite: bool,
) -> ComplexMatrix:
    return _prepare_rho_fixed_dim(
        rho,
        matrix_dimension=_MATRIX_DIMENSION,
        hermitize=hermitize,
        normalize_trace=normalize_trace,
        check_finite=check_finite,
    )


def _pack_payload(
    rhos: Sequence[ArrayLike],
    *,
    subsystem_count: int,
    hermitize: bool,
    normalize_trace: bool,
    check_finite: bool,
) -> bytes:
    if len(rhos) != subsystem_count:
        raise ValueError(
            f"Expected {subsystem_count} subsystem matrices, got {len(rhos)}. "
            "The subsystem count must match subsystem_qubits."
        )

    payload = np.empty((subsystem_count, _MATRIX_DIMENSION, _MATRIX_DIMENSION, 2), dtype="<f8")
    for subsystem_index, rho in enumerate(rhos):
        arr = _prepare_rho(
            rho,
            hermitize=hermitize,
            normalize_trace=normalize_trace,
            check_finite=check_finite,
        )
        payload[subsystem_index, :, :, 0] = arr.real
        payload[subsystem_index, :, :, 1] = arr.imag

    return payload.tobytes(order="C")


def _pack_tmi_payload(
    rhos: Sequence[ArrayLike],
    *,
    terms: Sequence[TmiTerm],
    hermitize: bool,
    normalize_trace: bool,
    check_finite: bool,
) -> bytes:
    if len(rhos) != len(terms):
        raise ValueError(f"Expected {len(terms)} TMI matrices, got {len(rhos)}.")

    chunks: list[bytes] = []
    for term, rho in zip(terms, rhos):
        dim = term.matrix_dimension
        arr = _prepare_rho_fixed_dim(
            rho,
            matrix_dimension=dim,
            hermitize=hermitize,
            normalize_trace=normalize_trace,
            check_finite=check_finite,
        )
        payload = np.empty((dim, dim, 2), dtype="<f8")
        payload[:, :, 0] = arr.real
        payload[:, :, 1] = arr.imag
        chunks.append(payload.tobytes(order="C"))

    return b"".join(chunks)


class TmiBinWriter:
    """Streaming writer for C++-readable MIPTTMI1 four-block TMI files."""

    def __init__(
        self,
        file: str | os.PathLike[str],
        *,
        total_qubits: int,
        terms: Optional[Sequence[TmiTerm | tuple[str, Sequence[int]]]] = None,
        spatial_dimension: int = 1,
        periods: int = 0,
        realizations: int = 0,
        resolution: int = 0,
        grid_x: int = 0,
        grid_y: int = 1,
        p_min: float = 0.0,
        p_max: float = 0.0,
        p_vals: Optional[Sequence[float]] = None,
        hermitize: bool = False,
        normalize_trace: bool = False,
        check_finite: bool = True,
    ) -> None:
        self.path = Path(file)
        self.total_qubits = _as_uint32(total_qubits, "total_qubits")
        if self.total_qubits < 4 or (self.total_qubits % 4) != 0:
            raise ValueError("TMI matrix writing requires total_qubits >= 4 and divisible by 4.")
        self.block_qubits = self.total_qubits // 4
        self.terms = _normalise_tmi_terms(self.total_qubits, terms)
        self.term_count = len(self.terms)

        if p_vals is not None:
            p_arr = np.asarray(p_vals, dtype=float)
            if p_arr.ndim != 1 or p_arr.size == 0:
                raise ValueError("p_vals must be a non-empty 1D sequence when provided.")
            p_min = float(np.min(p_arr))
            p_max = float(np.max(p_arr))
            if resolution == 0:
                resolution = int(p_arr.size)
            if grid_x == 0:
                grid_x = int(p_arr.size)

        self.spatial_dimension = _as_uint32(spatial_dimension, "spatial_dimension")
        self.periods = _as_uint32(periods, "periods")
        self.realizations = _as_uint32(realizations, "realizations")
        self.resolution = _as_uint32(resolution, "resolution")
        self.grid_x = _as_uint32(grid_x, "grid_x")
        self.grid_y = _as_uint32(grid_y, "grid_y")
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        self.hermitize = bool(hermitize)
        self.normalize_trace = bool(normalize_trace)
        self.check_finite = bool(check_finite)

        self._f = open(self.path, "w+b")
        self._record_count = 0
        self._closed = False
        self._finalized = False

        self._write_prefix_and_placeholder_header()
        self._write_term_metadata()

    @property
    def record_count(self) -> int:
        return self._record_count

    def __enter__(self) -> "TmiBinWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close(finalize=(exc_type is None))

    def _write_prefix_and_placeholder_header(self) -> None:
        self._f.write(_TMI_MAGIC)
        self._f.write(struct.pack("<II", _TMI_ENDIAN_MARKER, _TMI_VERSION))
        self._write_header(record_count=0)

    def _write_header(self, *, record_count: int) -> None:
        self._f.write(
            _TMI_HEADER_STRUCT.pack(
                self.spatial_dimension,
                self.total_qubits,
                self.block_qubits,
                self.periods,
                self.realizations,
                self.resolution,
                self.grid_x,
                self.grid_y,
                self.term_count,
                _as_uint64(record_count, "record_count"),
                self.p_min,
                self.p_max,
            )
        )

    def _write_term_metadata(self) -> None:
        for term in self.terms:
            raw_name = term.name.encode("ascii")
            if len(raw_name) > _TMI_TERM_NAME_BYTES:
                raise ValueError(f"TMI term name {term.name!r} is too long.")
            name_bytes = raw_name + b"\0" * (_TMI_TERM_NAME_BYTES - len(raw_name))
            self._f.write(
                _TMI_TERM_HEADER_STRUCT.pack(
                    name_bytes,
                    _as_uint32(term.kept_qubits, f"{term.name}.kept_qubits"),
                    _as_uint32(term.matrix_dimension, f"{term.name}.matrix_dimension"),
                )
            )
            self._f.write(struct.pack(f"<{term.kept_qubits}I", *term.modes))
        if not self._f:
            raise RuntimeError("Failed while writing TMI term metadata.")

    def write_record(
        self,
        *,
        p_index: int,
        realization: int,
        p: float,
        rhos: Sequence[ArrayLike],
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed TmiBinWriter.")

        payload = _pack_tmi_payload(
            rhos,
            terms=self.terms,
            hermitize=self.hermitize,
            normalize_trace=self.normalize_trace,
            check_finite=self.check_finite,
        )

        self._f.write(_TMI_RECORD_HEADER_STRUCT.pack(_as_uint32(p_index, "p_index"), _as_uint32(realization, "realization"), float(p)))
        self._f.write(payload)
        if not self._f:
            raise RuntimeError("Failed while writing TMI record payload.")
        self._record_count += 1

    def finalize(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot finalize a closed TmiBinWriter.")
        if self._finalized:
            return

        current = self._f.tell()
        self._f.seek(len(_TMI_MAGIC) + 8)
        self._write_header(record_count=self._record_count)
        self._f.seek(current)
        self._f.flush()
        os.fsync(self._f.fileno())
        self._finalized = True

    def close(self, *, finalize: bool = True) -> None:
        if self._closed:
            return
        try:
            if finalize:
                self.finalize()
        finally:
            self._f.close()
            self._closed = True


class Rho3BinWriter:
    """
    Streaming writer for C++-readable MIPTRHO2 rho3 files.

    Passing ``tmi=True`` delegates to an internal TmiBinWriter while preserving
    the same ``write_record(...)``/``close(...)`` interface.
    """

    def __init__(
        self,
        file: str | os.PathLike[str],
        *,
        total_qubits: int,
        subsystem_qubits: Optional[Sequence[Sequence[int]]] = None,
        spatial_dimension: int = 1,
        periods: int = 0,
        realizations: int = 0,
        resolution: int = 0,
        grid_x: int = 0,
        grid_y: int = 1,
        p_min: float = 0.0,
        p_max: float = 0.0,
        p_vals: Optional[Sequence[float]] = None,
        hermitize: bool = False,
        normalize_trace: bool = False,
        check_finite: bool = True,
        tmi: bool = False,
        tmi_terms: Optional[Sequence[TmiTerm | tuple[str, Sequence[int]]]] = None,
    ) -> None:
        self._impl: Optional[TmiBinWriter] = None
        if tmi:
            self._impl = TmiBinWriter(
                file,
                total_qubits=total_qubits,
                terms=tmi_terms,
                spatial_dimension=spatial_dimension,
                periods=periods,
                realizations=realizations,
                resolution=resolution,
                grid_x=grid_x,
                grid_y=grid_y,
                p_min=p_min,
                p_max=p_max,
                p_vals=p_vals,
                hermitize=hermitize,
                normalize_trace=normalize_trace,
                check_finite=check_finite,
            )
            self._closed = False
            return

        if subsystem_qubits is None:
            raise ValueError("subsystem_qubits is required unless tmi=True.")

        self.path = Path(file)
        self.total_qubits = _as_uint32(total_qubits, "total_qubits")
        self.subsystem_qubits = _normalise_subsystem_qubits(subsystem_qubits)
        self.subsystem_count = len(self.subsystem_qubits)

        if p_vals is not None:
            p_arr = np.asarray(p_vals, dtype=float)
            if p_arr.ndim != 1 or p_arr.size == 0:
                raise ValueError("p_vals must be a non-empty 1D sequence when provided.")
            p_min = float(np.min(p_arr))
            p_max = float(np.max(p_arr))
            if resolution == 0:
                resolution = int(p_arr.size)
            if grid_x == 0:
                grid_x = int(p_arr.size)

        self.spatial_dimension = _as_uint32(spatial_dimension, "spatial_dimension")
        self.periods = _as_uint32(periods, "periods")
        self.realizations = _as_uint32(realizations, "realizations")
        self.resolution = _as_uint32(resolution, "resolution")
        self.grid_x = _as_uint32(grid_x, "grid_x")
        self.grid_y = _as_uint32(grid_y, "grid_y")
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        self.hermitize = bool(hermitize)
        self.normalize_trace = bool(normalize_trace)
        self.check_finite = bool(check_finite)

        self._f = open(self.path, "w+b")
        self._record_count = 0
        self._closed = False
        self._finalized = False

        self._write_prefix_and_placeholder_header()
        self._write_subsystem_metadata()

    @property
    def record_count(self) -> int:
        if self._impl is not None:
            return self._impl.record_count
        return self._record_count

    def __enter__(self) -> "Rho3BinWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close(finalize=(exc_type is None))

    def _write_prefix_and_placeholder_header(self) -> None:
        self._f.write(_MAGIC)
        self._f.write(struct.pack("<II", _ENDIAN_MARKER, _VERSION))
        self._write_header(record_count=0)

    def _write_header(self, *, record_count: int) -> None:
        self._f.write(
            _HEADER_STRUCT.pack(
                self.spatial_dimension,
                self.total_qubits,
                _KEPT_QUBITS,
                _MATRIX_DIMENSION,
                self.periods,
                self.realizations,
                self.resolution,
                self.grid_x,
                self.grid_y,
                self.subsystem_count,
                _as_uint64(record_count, "record_count"),
                self.p_min,
                self.p_max,
            )
        )

    def _write_subsystem_metadata(self) -> None:
        flat = [q for triple in self.subsystem_qubits for q in triple]
        self._f.write(struct.pack(f"<{len(flat)}I", *flat))

    def write_record(
        self,
        *,
        p_index: int,
        realization: int,
        p: float,
        rhos: Sequence[ArrayLike],
    ) -> None:
        if self._impl is not None:
            self._impl.write_record(p_index=p_index, realization=realization, p=p, rhos=rhos)
            return

        if self._closed:
            raise RuntimeError("Cannot write to a closed Rho3BinWriter.")

        payload = _pack_payload(
            rhos,
            subsystem_count=self.subsystem_count,
            hermitize=self.hermitize,
            normalize_trace=self.normalize_trace,
            check_finite=self.check_finite,
        )

        self._f.write(_RECORD_HEADER_STRUCT.pack(_as_uint32(p_index, "p_index"), _as_uint32(realization, "realization"), float(p)))
        self._f.write(payload)
        self._record_count += 1

    def finalize(self) -> None:
        if self._impl is not None:
            self._impl.finalize()
            return
        if self._closed:
            raise RuntimeError("Cannot finalize a closed Rho3BinWriter.")
        if self._finalized:
            return

        current = self._f.tell()
        self._f.seek(len(_MAGIC) + 8)
        self._write_header(record_count=self._record_count)
        self._f.seek(current)
        self._f.flush()
        os.fsync(self._f.fileno())
        self._finalized = True

    def close(self, *, finalize: bool = True) -> None:
        if self._impl is not None:
            self._impl.close(finalize=finalize)
            self._closed = True
            return
        if self._closed:
            return
        try:
            if finalize:
                self.finalize()
        finally:
            self._f.close()
            self._closed = True


def write_rho3_bin(
    file: str | os.PathLike[str],
    records: Iterable[TrajectoryRecord],
    *,
    total_qubits: int,
    subsystem_qubits: Optional[Sequence[Sequence[int]]] = None,
    spatial_dimension: int = 1,
    periods: int = 0,
    realizations: int = 0,
    resolution: int = 0,
    grid_x: int = 0,
    grid_y: int = 1,
    p_min: float = 0.0,
    p_max: float = 0.0,
    p_vals: Optional[Sequence[float]] = None,
    hermitize: bool = False,
    normalize_trace: bool = False,
    check_finite: bool = True,
    tmi: bool = False,
) -> int:
    """
    Write precomputed trajectory records.

    With tmi=False, records contain legacy rho3 8x8 matrices.
    With tmi=True, records contain four matrices ordered AB, AC, BC, D and the
    output is MIPTTMI1.
    """
    with Rho3BinWriter(
        file,
        total_qubits=total_qubits,
        subsystem_qubits=subsystem_qubits,
        spatial_dimension=spatial_dimension,
        periods=periods,
        realizations=realizations,
        resolution=resolution,
        grid_x=grid_x,
        grid_y=grid_y,
        p_min=p_min,
        p_max=p_max,
        p_vals=p_vals,
        hermitize=hermitize,
        normalize_trace=normalize_trace,
        check_finite=check_finite,
        tmi=tmi,
    ) as writer:
        for p_index, realization, p, rhos in records:
            writer.write_record(p_index=p_index, realization=realization, p=p, rhos=rhos)
        return writer.record_count


def write_simulated_rho3_bin(
    file: str | os.PathLike[str],
    simulator,
    *,
    n: int,
    d: int,
    p_vals: Sequence[float],
    realisations: int,
    subsystem_qubits: Optional[Sequence[Sequence[int]]] = None,
    closed: bool = True,
    spatial_dimension: int = 1,
    progress: bool = True,
    hermitize: bool = False,
    normalize_trace: bool = False,
    check_finite: bool = True,
    progress_file: Optional[TextIO] = None,
    tmi: bool = False,
) -> int:
    """
    Run a simulator and stream produced matrices to .bin.

    The simulator callable must accept ``(n, d, p)`` and return the full list of
    matrices for one trajectory. With ``tmi=True`` it must return four matrices
    in AB/AC/BC/D order.
    """
    if progress_file is None:
        import sys

        progress_file = sys.stdout

    p_arr = np.asarray(p_vals, dtype=float)
    if p_arr.ndim != 1 or p_arr.size == 0:
        raise ValueError("p_vals must be a non-empty 1D sequence.")

    if not tmi and subsystem_qubits is None:
        subsystem_qubits = ring_subsystem_qubits(n, closed=closed)

    with Rho3BinWriter(
        file,
        total_qubits=n,
        subsystem_qubits=subsystem_qubits,
        spatial_dimension=spatial_dimension,
        periods=d,
        realizations=realisations,
        p_vals=p_arr,
        grid_y=1,
        hermitize=hermitize,
        normalize_trace=normalize_trace,
        check_finite=check_finite,
        tmi=tmi,
    ) as writer:
        for p_index, p in enumerate(p_arr):
            if progress:
                print(f"Simulating p = {p}...", file=progress_file, flush=True)
            for realization in range(int(realisations)):
                rhos = simulator(int(n), int(d), float(p))
                writer.write_record(p_index=p_index, realization=realization, p=float(p), rhos=rhos)
                if progress:
                    print(
                        f"Realisations: {realization + 1}/{realisations}",
                        end="\r",
                        file=progress_file,
                        flush=True,
                    )
            if progress:
                print("", file=progress_file, flush=True)

        return writer.record_count


# Backward-compatible alias spelling for Canadian/British codebases.
write_simulated_rho3_bin_realisations = write_simulated_rho3_bin


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="This module is mainly intended to be imported. Use Rho3BinWriter/TmiBinWriter from Python."
    )
    parser.add_argument("--print-format", action="store_true", help="Print a short format description.")
    args = parser.parse_args()

    if args.print_format:
        print(__doc__)
    else:
        parser.print_help()

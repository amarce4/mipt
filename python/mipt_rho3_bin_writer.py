"""
Write MIPTRHO2 three-qubit reduced-density-matrix .bin files from Python.

The output matches the binary layout expected by the current C++ GMN program:

    ./gmn.exe 0 rho3_python.bin data_real.csv
    ./gmn.exe 1 rho3_python.bin data_complex.csv

Format summary
--------------
File prefix:
    magic              8 bytes: b"MIPTRHO2"
    endian_marker      uint32 little-endian, 0x01020304
    version            uint32 little-endian, 2

Metadata header, little-endian struct "<10IQdd":
    spatial_dimension  uint32
    total_qubits       uint32
    kept_qubits        uint32, always 3 here
    matrix_dimension   uint32, always 8 here
    periods            uint32
    realizations       uint32
    resolution         uint32
    grid_x             uint32
    grid_y             uint32
    subsystem_count    uint32
    record_count       uint64
    p_min              float64
    p_max              float64

Subsystem metadata:
    subsystem_count * kept_qubits uint32 values giving the retained qubits
    for each 8x8 RDM.

Each trajectory/circuit record:
    p_index            uint32
    realization        uint32
    p                  float64
    payload            subsystem_count * 8 * 8 complex128 entries, stored as
                       interleaved little-endian float64 real/imag values in
                       row-major order:
                           Re(rho[0,0]), Im(rho[0,0]),
                           Re(rho[0,1]), Im(rho[0,1]), ...

Important: one binary record contains all retained 3-qubit subsystem matrices
for one circuit trajectory. The C++ GMN program then expands that one record
into subsystem-level GMN jobs.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO

import numpy as np
from numpy.typing import ArrayLike, NDArray


_MAGIC = b"MIPTRHO2"
_ENDIAN_MARKER = 0x01020304
_VERSION = 2
_HEADER_STRUCT = struct.Struct("<10IQdd")
_RECORD_HEADER_STRUCT = struct.Struct("<IId")
_KEPT_QUBITS = 3
_MATRIX_DIMENSION = 8

ComplexMatrix = NDArray[np.complex128]
QubitTriple = tuple[int, int, int]
TrajectoryRecord = tuple[int, int, float, Sequence[ArrayLike]]


def ring_subsystem_qubits(
    total_qubits: int,
    *,
    kept_qubits: int = 3,
    closed: bool = True,
) -> list[QubitTriple]:
    """
    Return consecutive 3-qubit subsystem labels for a 1D chain/ring.

    For total_qubits=8 and closed=True, this returns:
        [(0,1,2), (1,2,3), ..., (7,0,1)]

    For closed=False, this returns only non-wrapping windows:
        [(0,1,2), (1,2,3), ..., (5,6,7)]
    """
    if kept_qubits != 3:
        raise ValueError("This writer is fixed to three-qubit 8x8 RDMs; kept_qubits must be 3.")
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


def _prepare_rho(
    rho: ArrayLike,
    *,
    hermitize: bool,
    normalize_trace: bool,
    check_finite: bool,
) -> ComplexMatrix:
    arr = np.asarray(rho, dtype=np.complex128)
    if arr.shape != (_MATRIX_DIMENSION, _MATRIX_DIMENSION):
        raise ValueError(f"Each rho must have shape (8, 8), got {arr.shape}.")
    if check_finite and not np.all(np.isfinite(arr)):
        raise ValueError("rho contains NaN or infinite values.")

    # Copy to avoid writing views of mutable simulator buffers.
    arr = np.array(arr, dtype=np.complex128, copy=True)

    if hermitize:
        arr = 0.5 * (arr + arr.conj().T)

    if normalize_trace:
        tr = np.trace(arr)
        if abs(tr) == 0.0:
            raise ValueError("Cannot normalize rho with zero trace.")
        arr = arr / tr

    return arr


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


class Rho3BinWriter:
    """
    Streaming writer for C++-readable MIPTRHO2 rho3 binary files.

    Use this class when matrices are generated inside a simulation loop and you
    do not want to keep every trajectory in RAM.
    """

    def __init__(
        self,
        file: str | os.PathLike[str],
        *,
        total_qubits: int,
        subsystem_qubits: Sequence[Sequence[int]],
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
        return self._record_count

    def __enter__(self) -> "Rho3BinWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        # If the simulation failed, do not pretend the partial file is complete.
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
        """
        Append one trajectory/circuit record containing all retained subsystems.

        Parameters
        ----------
        p_index
            Integer index of the measurement rate p in the p scan.

        realization
            Integer circuit trajectory index for this p value.

        p
            Measurement rate value.

        rhos
            Sequence of subsystem matrices. Its length must equal
            len(subsystem_qubits). For n=8 closed 1D triples, this is 8.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed Rho3BinWriter.")

        p_index = _as_uint32(p_index, "p_index")
        realization = _as_uint32(realization, "realization")
        p = float(p)

        payload = _pack_payload(
            rhos,
            subsystem_count=self.subsystem_count,
            hermitize=self.hermitize,
            normalize_trace=self.normalize_trace,
            check_finite=self.check_finite,
        )

        self._f.write(_RECORD_HEADER_STRUCT.pack(p_index, realization, p))
        self._f.write(payload)
        self._record_count += 1

    def finalize(self) -> None:
        """Patch the final record_count into the header."""
        if self._closed:
            raise RuntimeError("Cannot finalize a closed Rho3BinWriter.")
        if self._finalized:
            return

        current = self._f.tell()
        self._f.seek(len(_MAGIC) + 8)  # after magic, endian marker, and version
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


def write_rho3_bin(
    file: str | os.PathLike[str],
    records: Iterable[TrajectoryRecord],
    *,
    total_qubits: int,
    subsystem_qubits: Sequence[Sequence[int]],
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
) -> int:
    """
    Write an iterable of precomputed trajectory records to a MIPTRHO2 .bin file.

    Each record must be:
        (p_index, realization, p, rhos)

    where rhos is the full list of retained subsystem 8x8 matrices for that
    trajectory.

    Returns
    -------
    int
        Number of trajectory/circuit records written.
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
    ) as writer:
        for p_index, realization, p, rhos in records:
            writer.write_record(
                p_index=p_index,
                realization=realization,
                p=p,
                rhos=rhos,
            )
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
) -> int:
    """
    Run a Python simulator and stream all produced 3-qubit RDMs to .bin.

    The simulator callable must accept ``(n, d, p)`` and return the full list of
    retained 8x8 subsystem matrices for one circuit trajectory. For your current
    code, pass a lambda such as:

        lambda n, d, p: get_mipt_rho_1d(
            n, d, p, subsyst=3, all_matrices=True, closed=True
        )

    Returns the number of trajectory/circuit records written.
    """
    if progress_file is None:
        import sys

        progress_file = sys.stdout

    p_arr = np.asarray(p_vals, dtype=float)
    if p_arr.ndim != 1 or p_arr.size == 0:
        raise ValueError("p_vals must be a non-empty 1D sequence.")

    if subsystem_qubits is None:
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
    ) as writer:
        for p_index, p in enumerate(p_arr):
            if progress:
                print(f"Simulating p = {p}...", file=progress_file, flush=True)
            for realization in range(int(realisations)):
                rhos = simulator(int(n), int(d), float(p))
                writer.write_record(
                    p_index=p_index,
                    realization=realization,
                    p=float(p),
                    rhos=rhos,
                )
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
        description=(
            "This module is mainly intended to be imported. "
            "Use Rho3BinWriter or write_simulated_rho3_bin from Python."
        )
    )
    parser.add_argument(
        "--print-format",
        action="store_true",
        help="Print a short description of the MIPTRHO2 format.",
    )
    args = parser.parse_args()

    if args.print_format:
        print(__doc__)
    else:
        parser.print_help()

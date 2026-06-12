"""
Read MIPT three-qubit reduced-density-matrix .bin files.

This reader targets the MIPTRHO2 binary format produced by the revised
mipt.exe, where each circuit trajectory can contain multiple 3-qubit
8 x 8 complex reduced density matrices.

Main function
-------------
read_rho3_bin(
    file: str,
    p_vals: list | None = None,
    limit: int | None = None,
) -> list[list[dict]]

The returned value is grouped by circuit trajectory / realization:

    records[record_index][subsystem_index]["rho"]

For example, for a file with 8 retained subsystems:

    rhos_for_first_circuit = [records[0][i]["rho"] for i in range(8)]

Each subsystem dictionary contains:
    rho              np.ndarray, shape (8, 8), dtype complex128
    p                float
    p_index          int
    realization      int
    subsystem_index  int
    qubits           tuple[int, int, int]
    metadata         dict

The optional ``limit`` caps the number of circuit records loaded per
measurement rate p. It does not count individual subsystem matrices. If a
file has 11 p values and ``limit=250``, at most 250 grouped circuit records
are returned for each p value, i.e. at most 250 * 11 outer records.

Example
-------
    from read_mipt_rho3 import read_rho3_bin

    records = read_rho3_bin("rho3.bin", p_vals=[0.0, 0.5, 1.0], limit=250)
    first_circuit_rhos = [records[0][i]["rho"] for i in range(len(records[0]))]

    print(records[0][0]["p"], records[0][0]["qubits"])
    print(records[0][0]["rho"].shape)
"""

from __future__ import annotations

import os
import struct
from typing import Any, Optional, Sequence

import numpy as np


_MAGIC = b"MIPTRHO2"
_ENDIAN_MARKER = 0x01020304
_VERSION = 2
_HEADER_STRUCT = struct.Struct("<10IQdd")
_RECORD_HEADER_STRUCT = struct.Struct("<IId")


def _matches_p_value(p: float, p_vals: Optional[Sequence[float]]) -> bool:
    if p_vals is None:
        return True
    return any(np.isclose(p, float(target), rtol=1.0e-9, atol=1.0e-12) for target in p_vals)


def _validate_limit(limit: Optional[int]) -> None:
    if limit is None:
        return
    if not isinstance(limit, int):
        raise TypeError("limit must be an int or None.")
    if limit < 0:
        raise ValueError("limit must be non-negative or None.")


def read_rho3_bin(
    file: str,
    p_vals: Optional[Sequence[float]] = None,
    limit: Optional[int] = None,
) -> list[list[dict[str, Any]]]:
    """
    Read a MIPTRHO2 .bin file into grouped 8 x 8 complex NumPy arrays.

    Parameters
    ----------
    file
        Path to the .bin file, usually "rho3.bin" or "rho3_2d.bin".

    p_vals
        Optional sequence of p values to keep. Matching uses np.isclose with
        rtol=1e-9 and atol=1e-12 to avoid exact floating-point comparison
        problems. If None, all p values are loaded.

    limit
        Optional maximum number of circuit records to load per measurement
        rate p. This cap is applied after p_vals filtering. The cap is per
        p_index, so repeated circuit records at the same p are counted
        together. Subsystems inside a kept circuit record are not counted
        against this limit individually.

    Returns
    -------
    list[list[dict]]
        One outer list entry per kept circuit trajectory / realization.
        Each outer entry is an inner list of subsystem dictionaries ordered
        by subsystem_index. Therefore, records[0][i]["rho"] accesses the
        i-th retained subsystem matrix for the first kept circuit record.
    """
    _validate_limit(limit)

    if not os.path.isfile(file):
        raise FileNotFoundError(file)

    records: list[list[dict[str, Any]]] = []
    kept_per_p_index: dict[int, int] = {}

    with open(file, "rb") as f:
        magic = f.read(8)
        if magic != _MAGIC:
            raise ValueError(
                f"{file!r} is not a MIPTRHO2 density-matrix file. "
                f"Read magic {magic!r}."
            )

        endian_marker, version = struct.unpack("<II", f.read(8))
        if endian_marker != _ENDIAN_MARKER:
            raise ValueError(
                "Unsupported file endianness. Expected little-endian "
                "MIPTRHO2 data."
            )
        if version != _VERSION:
            raise ValueError(f"Unsupported MIPTRHO version {version}; expected {_VERSION}.")

        header_bytes = f.read(_HEADER_STRUCT.size)
        if len(header_bytes) != _HEADER_STRUCT.size:
            raise EOFError("Incomplete MIPTRHO2 header.")

        (
            spatial_dimension,
            total_qubits,
            kept_qubits,
            matrix_dimension,
            periods,
            realizations,
            resolution,
            grid_x,
            grid_y,
            subsystem_count,
            record_count,
            p_min,
            p_max,
        ) = _HEADER_STRUCT.unpack(header_bytes)

        if kept_qubits != 3:
            raise ValueError(f"Expected kept_qubits=3, got {kept_qubits}.")
        if matrix_dimension != 8:
            raise ValueError(f"Expected matrix_dimension=8, got {matrix_dimension}.")
        if subsystem_count == 0:
            raise ValueError("File contains zero retained subsystems.")

        metadata = {
            "spatial_dimension": int(spatial_dimension),
            "total_qubits": int(total_qubits),
            "kept_qubits": int(kept_qubits),
            "matrix_dimension": int(matrix_dimension),
            "periods": int(periods),
            "realizations": int(realizations),
            "resolution": int(resolution),
            "grid_x": int(grid_x),
            "grid_y": int(grid_y),
            "subsystem_count": int(subsystem_count),
            "record_count": int(record_count),
            "p_min": float(p_min),
            "p_max": float(p_max),
        }

        subsystem_index_count = subsystem_count * kept_qubits
        subsystem_bytes = f.read(4 * subsystem_index_count)
        if len(subsystem_bytes) != 4 * subsystem_index_count:
            raise EOFError("Incomplete subsystem-qubit metadata.")

        subsystem_flat = struct.unpack(f"<{subsystem_index_count}I", subsystem_bytes)
        subsystem_qubits = [
            tuple(int(q) for q in subsystem_flat[s * 3 : (s + 1) * 3])
            for s in range(subsystem_count)
        ]

        matrix_value_count = 2 * matrix_dimension * matrix_dimension
        payload_value_count = subsystem_count * matrix_value_count
        payload_byte_count = 8 * payload_value_count

        for _ in range(record_count):
            record_header = f.read(_RECORD_HEADER_STRUCT.size)
            if len(record_header) != _RECORD_HEADER_STRUCT.size:
                raise EOFError("Incomplete density-matrix record header.")

            p_index, realization, p = _RECORD_HEADER_STRUCT.unpack(record_header)
            p_index = int(p_index)
            realization = int(realization)
            p = float(p)

            payload_bytes = f.read(payload_byte_count)
            if len(payload_bytes) != payload_byte_count:
                raise EOFError("Incomplete density-matrix record payload.")

            if not _matches_p_value(p, p_vals):
                continue

            if limit is not None and kept_per_p_index.get(p_index, 0) >= limit:
                continue

            payload = np.frombuffer(payload_bytes, dtype="<f8")
            payload = payload.reshape((subsystem_count, matrix_dimension, matrix_dimension, 2))

            circuit_record: list[dict[str, Any]] = []
            for subsystem_index in range(subsystem_count):
                rho_ri = payload[subsystem_index]
                rho = rho_ri[..., 0] + 1j * rho_ri[..., 1]
                rho = np.asarray(rho, dtype=np.complex128)

                circuit_record.append(
                    {
                        "rho": rho,
                        "p": p,
                        "p_index": p_index,
                        "realization": realization,
                        "subsystem_index": int(subsystem_index),
                        "qubits": subsystem_qubits[subsystem_index],
                        "metadata": metadata,
                    }
                )

            records.append(circuit_record)
            kept_per_p_index[p_index] = kept_per_p_index.get(p_index, 0) + 1

        trailing = f.read(1)
        if trailing:
            raise ValueError("File contains unexpected trailing bytes.")

    return records


def flatten_rho3_records(
    records: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Flatten grouped records into the older one-dictionary-per-subsystem shape.

    This is useful for older analysis code that expects a simple list of RDM
    dictionaries rather than records[record_index][subsystem_index].
    """
    return [subsystem for circuit_record in records for subsystem in circuit_record]


# Alias with a more explicit name, if preferred in analysis scripts.
read_mipt_rho3_bin = read_rho3_bin


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read a MIPTRHO2 rho3.bin file.")
    parser.add_argument("file", help="Path to rho3.bin or rho3_2d.bin")
    parser.add_argument(
        "--p",
        dest="p_vals",
        type=float,
        nargs="*",
        default=None,
        help="Optional p values to keep, e.g. --p 0.0 0.5 1.0",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=None,
        help="Optional maximum number of circuit records to load per p value.",
    )
    args = parser.parse_args()

    loaded = read_rho3_bin(args.file, args.p_vals, limit=args.limit)
    subsystem_total = sum(len(record) for record in loaded)
    print(f"Loaded {len(loaded)} circuit records containing {subsystem_total} three-qubit RDMs.")
    if loaded:
        first = loaded[0][0]
        rho0 = first["rho"]
        print(
            "First subsystem of first circuit record: "
            f"p={first['p']}, "
            f"p_index={first['p_index']}, "
            f"realization={first['realization']}, "
            f"subsystem_index={first['subsystem_index']}, "
            f"qubits={first['qubits']}, "
            f"shape={rho0.shape}, "
            f"trace={np.trace(rho0)}"
        )

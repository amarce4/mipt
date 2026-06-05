"""
Read MIPT three-qubit reduced-density-matrix .bin files.

This reader targets the MIPTRHO2 binary format produced by the revised
mipt.exe, where each circuit trajectory can contain multiple 3-qubit
8 x 8 complex reduced density matrices.

Main function
-------------
read_rho3_bin(file: str, p_vals: list | None = None) -> list[dict]

Each returned dict contains:
    rho              np.ndarray, shape (8, 8), dtype complex128
    p                float
    p_index          int
    realization      int
    subsystem_index  int
    qubits           tuple[int, int, int]

Example
-------
    from read_mipt_rho3 import read_rho3_bin

    records = read_rho3_bin("rho3.bin", p_vals=[0.0, 0.5, 1.0])
    rhos = [rec["rho"] for rec in records]

    print(records[0]["p"], records[0]["qubits"])
    print(records[0]["rho"].shape)
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


def read_rho3_bin(file: str, p_vals: Optional[list] = None) -> list[dict[str, Any]]:
    """
    Read a MIPTRHO2 .bin file into 8 x 8 complex NumPy arrays.

    Parameters
    ----------
    file
        Path to the .bin file, usually "rho3.bin" or "rho3_2d.bin".

    p_vals
        Optional list of p values to keep. Matching uses np.isclose with
        rtol=1e-9 and atol=1e-12 to avoid exact floating-point comparison
        problems. If None, all p values are loaded.

    Returns
    -------
    list[dict]
        One dictionary per stored 3-qubit RDM. The key "rho" contains the
        actual 8 x 8 complex128 NumPy array.
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(file)

    records: list[dict[str, Any]] = []

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

            payload_bytes = f.read(payload_byte_count)
            if len(payload_bytes) != payload_byte_count:
                raise EOFError("Incomplete density-matrix record payload.")

            if not _matches_p_value(p, p_vals):
                continue

            payload = np.frombuffer(payload_bytes, dtype="<f8")
            payload = payload.reshape((subsystem_count, matrix_dimension, matrix_dimension, 2))

            for subsystem_index in range(subsystem_count):
                rho_ri = payload[subsystem_index]
                rho = rho_ri[..., 0] + 1j * rho_ri[..., 1]
                rho = np.asarray(rho, dtype=np.complex128)

                records.append(
                    {
                        "rho": rho,
                        "p": float(p),
                        "p_index": int(p_index),
                        "realization": int(realization),
                        "subsystem_index": int(subsystem_index),
                        "qubits": subsystem_qubits[subsystem_index],
                        "metadata": {
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
                        },
                    }
                )

        trailing = f.read(1)
        if trailing:
            raise ValueError("File contains unexpected trailing bytes.")

    return records


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
    args = parser.parse_args()

    loaded = read_rho3_bin(args.file, args.p_vals)
    print(f"Loaded {len(loaded)} three-qubit RDMs.")
    if loaded:
        first = loaded[0]
        rho0 = first["rho"]
        print(
            "First record: "
            f"p={first['p']}, "
            f"p_index={first['p_index']}, "
            f"realization={first['realization']}, "
            f"subsystem_index={first['subsystem_index']}, "
            f"qubits={first['qubits']}, "
            f"shape={rho0.shape}, "
            f"trace={np.trace(rho0)}"
        )

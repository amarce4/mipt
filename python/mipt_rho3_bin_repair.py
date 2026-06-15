"""
Inspect and repair/finish partial MIPTRHO2 rho3 .bin files.

Use this if a Python writer was interrupted before it patched the final
record_count into the file header. The tool computes the number of physically
complete trajectory records from the file size, optionally truncates an
incomplete trailing record, and patches the header record_count.

Examples
--------
    python mipt_rho3_bin_repair.py inspect rho3_python_archA.bin
    python mipt_rho3_bin_repair.py repair rho3_python_archA.bin
"""

from __future__ import annotations

import argparse
import os
import struct
from dataclasses import dataclass
from pathlib import Path

_MAGIC = b"MIPTRHO2"
_ENDIAN_MARKER = 0x01020304
_VERSION = 2
_HEADER_STRUCT = struct.Struct("<10IQdd")


@dataclass(frozen=True)
class MiptRho3BinInfo:
    path: str
    file_size_bytes: int
    header_record_count: int
    physical_complete_records: int
    trailing_partial_bytes: int
    subsystem_count: int
    header_and_metadata_bytes: int
    record_bytes: int
    matrix_dimension: int
    kept_qubits: int
    total_qubits: int
    realizations: int
    resolution: int
    p_min: float
    p_max: float


def inspect_mipt_rho3_bin(path: str | os.PathLike[str]) -> MiptRho3BinInfo:
    path = str(path)
    size = os.path.getsize(path)

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _MAGIC:
            raise ValueError(f"{path!r} is not a MIPTRHO2 file; magic={magic!r}")

        endian_marker, version = struct.unpack("<II", f.read(8))
        if endian_marker != _ENDIAN_MARKER:
            raise ValueError("Unsupported endian marker; expected little-endian MIPTRHO2.")
        if version != _VERSION:
            raise ValueError(f"Unsupported MIPTRHO version {version}; expected {_VERSION}.")

        header_offset = 16
        header = f.read(_HEADER_STRUCT.size)
        if len(header) != _HEADER_STRUCT.size:
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
        ) = _HEADER_STRUCT.unpack(header)

    metadata_bytes = int(subsystem_count) * int(kept_qubits) * 4
    header_and_metadata_bytes = 16 + _HEADER_STRUCT.size + metadata_bytes
    record_bytes = 4 + 4 + 8 + int(subsystem_count) * int(matrix_dimension) * int(matrix_dimension) * 2 * 8

    if size < header_and_metadata_bytes:
        physical_complete_records = 0
        trailing = max(0, size - header_and_metadata_bytes)
    else:
        payload_size = size - header_and_metadata_bytes
        physical_complete_records = payload_size // record_bytes
        trailing = payload_size % record_bytes

    return MiptRho3BinInfo(
        path=path,
        file_size_bytes=size,
        header_record_count=int(record_count),
        physical_complete_records=int(physical_complete_records),
        trailing_partial_bytes=int(trailing),
        subsystem_count=int(subsystem_count),
        header_and_metadata_bytes=int(header_and_metadata_bytes),
        record_bytes=int(record_bytes),
        matrix_dimension=int(matrix_dimension),
        kept_qubits=int(kept_qubits),
        total_qubits=int(total_qubits),
        realizations=int(realizations),
        resolution=int(resolution),
        p_min=float(p_min),
        p_max=float(p_max),
    )


def patch_mipt_rho3_record_count(
    path: str | os.PathLike[str],
    *,
    record_count: int | None = None,
    truncate_incomplete_tail: bool = True,
) -> MiptRho3BinInfo:
    """
    Patch the header record_count to match complete physical records.

    If record_count is None, the count is inferred from file size. If the file
    has a partial trailing record and truncate_incomplete_tail=True, the file is
    truncated to the last complete record before patching.
    """
    path = Path(path)
    info = inspect_mipt_rho3_bin(path)

    if record_count is None:
        record_count = info.physical_complete_records
    record_count = int(record_count)
    if record_count < 0:
        raise ValueError("record_count must be non-negative.")
    if record_count > info.physical_complete_records:
        raise ValueError(
            f"Requested record_count={record_count}, but the file only contains "
            f"{info.physical_complete_records} complete physical records."
        )

    final_size = info.header_and_metadata_bytes + record_count * info.record_bytes
    if truncate_incomplete_tail or final_size < info.file_size_bytes:
        with open(path, "r+b") as f:
            f.truncate(final_size)

    # record_count is the 11th field in <10IQdd>, after ten uint32 values.
    record_count_offset = 16 + 10 * 4
    with open(path, "r+b") as f:
        f.seek(record_count_offset)
        f.write(struct.pack("<Q", record_count))
        f.flush()
        os.fsync(f.fileno())

    return inspect_mipt_rho3_bin(path)


def _print_info(info: MiptRho3BinInfo) -> None:
    print(f"file: {info.path}")
    print(f"file_size_bytes: {info.file_size_bytes}")
    print(f"header_record_count: {info.header_record_count}")
    print(f"physical_complete_records: {info.physical_complete_records}")
    print(f"trailing_partial_bytes: {info.trailing_partial_bytes}")
    print(f"subsystem_count: {info.subsystem_count}")
    print(f"record_bytes: {info.record_bytes}")
    print(f"header_and_metadata_bytes: {info.header_and_metadata_bytes}")
    print(f"matrix_dimension: {info.matrix_dimension}")
    print(f"kept_qubits: {info.kept_qubits}")
    print(f"total_qubits: {info.total_qubits}")
    print(f"realizations metadata: {info.realizations}")
    print(f"resolution metadata: {info.resolution}")
    print(f"p range metadata: [{info.p_min}, {info.p_max}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or repair a partial MIPTRHO2 rho3 .bin file.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Show header and physical record-count information.")
    p_inspect.add_argument("file")

    p_repair = sub.add_parser("repair", help="Patch header record_count to complete physical records.")
    p_repair.add_argument("file")
    p_repair.add_argument("--record-count", type=int, default=None)
    p_repair.add_argument("--keep-tail", action="store_true", help="Do not truncate an incomplete trailing record.")

    args = parser.parse_args()
    if args.cmd == "inspect":
        _print_info(inspect_mipt_rho3_bin(args.file))
    elif args.cmd == "repair":
        info = patch_mipt_rho3_record_count(
            args.file,
            record_count=args.record_count,
            truncate_incomplete_tail=not args.keep_tail,
        )
        _print_info(info)


if __name__ == "__main__":
    main()

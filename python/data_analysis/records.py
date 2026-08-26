"""Reading ``dist_scaling.exe``'s per-record binary output.

The aggregated CSV keeps one mean and one standard error per geometry, and
that standard error treats records as independent. They are not: every
geometry is measured inside the *same* trajectory, so the points of a decay
curve covary and an exponent fitted to the whole curve carries a much larger
error than propagating the per-point stderrs suggests. ``<stem>_records.bin``
keeps the records themselves, which makes both a trajectory-clustered
bootstrap and the value *distributions* recoverable.

Format
------
A text header followed by a flat fixed-width record array -- see the header
comment of ``cpp/include/mipt/dist_records.hpp`` for why that rather than
Parquet or HDF5. The short version: both are dependencies that buy nothing on
a fixed-width table of two integers and two float64s, and neither appends
safely, which an interruptible run needs.

::

    bytes  0.. 7  magic "MIPTDREC"
    bytes  8..11  uint32 format version
    bytes 12..15  uint32 record bytes
    bytes 16..19  uint32 header text bytes
    bytes 20..23  uint32 reserved
    then          header text: `key=value` lines, then the geometry table
    then          records

Each record is ``realization_id`` (u4), ``geometry_id`` (u2),
``embedding_id`` (u2), then the observables as float64 -- ``I_k`` and ``N_k``,
which are ``(mi, mn)`` at k=2 and ``(tmi, gmn)`` at k=3, followed by the
fermionic pair for parity-preserving ensembles. The header's ``fields`` line
is authoritative; nothing here assumes a particular observable set.

``d`` is deliberately *not* in the records. It lives once in the header's
geometry table, keyed by ``geometry_id``, which is what keeps a record down to
its two identifiers and its two float64s.

NaN is not zero. A GMN the schedule never selected, and one the solver failed
on, are both NaN; ``records()`` leaves them as NaN and every aggregation here
drops them from that column while keeping the row's other observables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAGIC = b"MIPTDREC"
PREAMBLE_BYTES = 24
FORMAT_VERSION = 1

_ID_DTYPE = [
    ("realization_id", "<u4"),
    ("geometry_id", "<u2"),
    ("embedding_id", "<u2"),
]

_NUMERIC_HEADER_KEYS = {
    "version": int,
    "k": int,
    "N": int,
    "circ_type": int,
    "periods": int,
    "realizations": int,
    "geometry_rows": int,
    "embeddings_per_geometry": int,
    "p": float,
    "b_min": float,
}


def is_record_file(path: str | Path) -> bool:
    """Whether ``path`` is a dist_scaling record binary, by its magic."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            return handle.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def _parse_header_text(text: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Split the header into its ``key=value`` block and its geometry table."""
    lines = text.splitlines()
    header: dict[str, Any] = {}
    geometry_lines: list[str] = []
    in_table = False
    for line in lines:
        if line == "[geometry]":
            in_table = True
            continue
        if line == "[end]":
            in_table = False
            continue
        if in_table:
            geometry_lines.append(line)
            continue
        key, separator, value = line.partition("=")
        if separator:
            header[key] = value

    for key, caster in _NUMERIC_HEADER_KEYS.items():
        if key in header:
            try:
                header[key] = caster(header[key])
            except ValueError:
                pass

    if not geometry_lines:
        raise ValueError("A dist_scaling record file must carry a geometry table.")
    columns = geometry_lines[0].split(",")
    rows = [line.split(",") for line in geometry_lines[1:] if line]
    geometries = pd.DataFrame(rows, columns=columns)
    for column in geometries.columns:
        geometries[column] = pd.to_numeric(geometries[column], errors="coerce")
    if "d" not in geometries.columns:
        raise ValueError(
            "A dist_scaling record file's geometry table must carry the "
            "effective chord distance column 'd'."
        )
    return header, geometries


def _record_dtype(fields: str) -> tuple[np.dtype, tuple[str, ...]]:
    """The numpy dtype the ``fields`` header line describes.

    The identifiers are fixed but the observables are not -- a qubit ensemble
    writes two and a parity-preserving one writes four -- so the layout is read
    off the file rather than assumed.
    """
    kinds = {"u4": "<u4", "u2": "<u2", "f8": "<f8"}
    entries: list[tuple[str, str]] = []
    observables: list[str] = []
    for field in fields.split(","):
        name, _, kind = field.partition(":")
        if kind not in kinds:
            raise ValueError(f"Unsupported record field type {kind!r} in {field!r}.")
        entries.append((name, kinds[kind]))
        if kind == "f8":
            observables.append(name)
    if [entry[0] for entry in entries[: len(_ID_DTYPE)]] != [
        name for name, _ in _ID_DTYPE
    ]:
        raise ValueError(
            "A dist_scaling record must begin with realization_id, geometry_id "
            f"and embedding_id; found {[entry[0] for entry in entries]}."
        )
    return np.dtype(entries), tuple(observables)


def read_records(path: str | Path) -> dict[str, Any]:
    """Read one record file into its header, geometry table and records.

    Returns ``{"path", "header", "geometries", "records", "observables", "k",
    "L", "circuit_type", "truncated"}``. ``records`` is a DataFrame carrying
    the three identifiers, every observable, and ``d`` joined in from the
    geometry table.

    A partial trailing record -- what a killed run leaves -- is discarded and
    reported in ``truncated``, which is exactly why the format is a flat array
    with a fixed record size rather than a container with a footer.
    """
    path = Path(path)
    with path.open("rb") as handle:
        preamble = handle.read(PREAMBLE_BYTES)
        if len(preamble) < PREAMBLE_BYTES or preamble[: len(MAGIC)] != MAGIC:
            raise ValueError(f"{path} is not a dist_scaling record file.")
        version, record_bytes, header_bytes = np.frombuffer(
            preamble, dtype="<u4", count=3, offset=8
        )
        if int(version) != FORMAT_VERSION:
            raise ValueError(
                f"{path} is record format version {int(version)}; this reader "
                f"understands version {FORMAT_VERSION}."
            )
        header_text = handle.read(int(header_bytes)).decode("utf-8")

    header, geometries = _parse_header_text(header_text)
    dtype, observables = _record_dtype(header["fields"])
    if dtype.itemsize != int(record_bytes):
        raise ValueError(
            f"{path}: the header describes a {dtype.itemsize}-byte record but "
            f"the file declares {int(record_bytes)}."
        )

    payload_offset = PREAMBLE_BYTES + int(header_bytes)
    payload_bytes = path.stat().st_size - payload_offset
    count = payload_bytes // dtype.itemsize
    truncated = payload_bytes - count * dtype.itemsize
    raw = np.fromfile(path, dtype=dtype, count=int(count), offset=payload_offset)

    frame = pd.DataFrame(
        {
            "realization_id": raw["realization_id"].astype(np.int64),
            "geometry_id": raw["geometry_id"].astype(np.int64),
            "embedding_id": raw["embedding_id"].astype(np.int64),
            **{name: raw[name].astype(float) for name in observables},
        }
    )
    lookup = geometries.set_index("geometry_id")["d"]
    frame["d"] = frame["geometry_id"].map(lookup).astype(float)

    return {
        "path": str(path),
        "header": header,
        "geometries": geometries,
        "records": frame,
        "observables": observables,
        "k": int(header.get("k", 2)),
        "L": int(header["N"]) if "N" in header else None,
        "circuit_type": header.get("circ_type"),
        "entropy_units": header.get("entropy_units", "bits"),
        "circuit_name": header.get("circuit_name"),
        "truncated": int(truncated),
    }


def aggregate_records(bundle: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Reduce records to the ``{metric: DataFrame[d, mean, stderr, count]}``
    structure the CSV path produces, so the two formats fit the same fitting
    and plotting code.

    NaNs are dropped per column, not per row: a record whose GMN was never
    requested still contributes its TMI. Every negativity is additionally
    decomposed the way the aggregate CSV's ``<metric>_positive_count`` column
    allows -- ``P_ent`` and the conditional magnitude -- since a record file
    knows the individual values and needs no such column.
    """
    frame = bundle["records"]
    curves: dict[str, pd.DataFrame] = {}
    for metric in bundle["observables"]:
        values = frame[["d", metric]].dropna()
        if values.empty:
            continue
        grouped = values.groupby("d")[metric]
        count = grouped.count()
        mean = grouped.mean()
        # ddof=1 matches the sample standard deviation the C++ RunningStats
        # reports, so a curve built here and one read from the CSV agree.
        stderr = grouped.std(ddof=1) / np.sqrt(count)
        curve = pd.DataFrame(
            {
                "d": count.index.to_numpy(dtype=float),
                "mean": mean.to_numpy(dtype=float),
                "stderr": stderr.to_numpy(dtype=float),
                "count": count.to_numpy(dtype=np.int64),
            }
        ).sort_values("d").reset_index(drop=True)
        curves[metric] = curve

        if metric in {"mn", "fmn", "gmn", "fgmn"}:
            positive = grouped.apply(lambda column: float((column > 0.0).sum()))
            fraction = positive / count
            with np.errstate(invalid="ignore", divide="ignore"):
                magnitude = mean / fraction
                fraction_stderr = np.sqrt(
                    np.maximum(fraction * (1.0 - fraction), 0.0) / count
                )
            curves[f"{metric}_ent_fraction"] = pd.DataFrame(
                {
                    "d": count.index.to_numpy(dtype=float),
                    "mean": fraction.to_numpy(dtype=float),
                    "stderr": fraction_stderr.to_numpy(dtype=float),
                    "count": count.to_numpy(dtype=np.int64),
                }
            ).sort_values("d").reset_index(drop=True)
            entangled = values.loc[values[metric] > 0.0].groupby("d")[metric]
            magnitude_frame = pd.DataFrame(
                {
                    "d": magnitude.index.to_numpy(dtype=float),
                    "mean": magnitude.to_numpy(dtype=float),
                    "stderr": (
                        entangled.std(ddof=1) / np.sqrt(entangled.count())
                    ).reindex(magnitude.index).to_numpy(dtype=float),
                    "count": positive.reindex(magnitude.index)
                    .fillna(0)
                    .to_numpy(dtype=np.int64),
                }
            ).sort_values("d").reset_index(drop=True)
            curves[f"{metric}_ent_magnitude"] = magnitude_frame
    return curves

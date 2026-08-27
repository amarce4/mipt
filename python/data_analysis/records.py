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
geometry table, keyed by ``geometry_id``.

NaN is not zero. A GMN the schedule never selected, and one the solver failed
on, are both NaN; they are counted separately and excluded from every mean.

Reading these files without running out of memory
-------------------------------------------------
A production file is *large*: a k=2 run at N=24 writes 276 records per
trajectory, so 243k trajectories is 67 million records and 2.7 GB on disk --
and turning that into a pandas frame costs another 4.3 GB, because the
identifiers widen to int64 and ``d`` is materialised per row. On a machine
with a few gigabytes that is not a slow read, it is an out-of-memory kill.

So the path everything here takes is **streaming**: ``summarize`` makes one
pass over the file in bounded chunks and reduces it to per-geometry moments
and per-geometry logarithmic histograms, which together are a few thousand
numbers. From that summary the decay curves come back *exactly* -- the mean
and standard error are reconstructed from the summed values and squares, not
approximated -- and the value distributions come back at the resolution of
the histogram grid, which is what they are drawn at anyway.

``write_summary_csv`` puts that summary in a small CSV so the 2.7 GB pass
happens once rather than on every plot; ``python -m
data_analysis.summarize_records`` (or ``./summarize_records`` at the repo
root) is the command-line front end. ``data_analysis.dist_scaling`` picks up
a ``<stem>_summary.csv`` automatically if one sits beside the binary.

``read_records`` still exists and still returns every record as a frame, for
the small files where that is convenient, but it refuses by default above a
memory budget rather than taking the machine down with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

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

# ---------------------------------------------------------------------------
# The histogram grid
#
# Fixed rather than derived from the data, so that two summaries are directly
# comparable and can be added together, and so that a summary does not depend
# on the zero tolerance a later plot happens to choose. Every power of ten is
# an edge, which is what lets a tolerance like 1e-10 be honoured exactly
# rather than snapped to the nearest bin boundary.
#
# The values live between about 1e-16 (floating-point noise around a
# vanishing mutual information) and 2 (the largest a two-qubit MI in bits can
# be), so the grid is generous at both ends; 20 bins per decade is five times
# finer than anything is drawn at, so the display rebins by grouping adjacent
# bins rather than by interpolating.
# ---------------------------------------------------------------------------

HIST_LOG_MIN = -18
HIST_LOG_MAX = 2
HIST_BINS_PER_DECADE = 20
HIST_BIN_COUNT = (HIST_LOG_MAX - HIST_LOG_MIN) * HIST_BINS_PER_DECADE


def histogram_edges() -> np.ndarray:
    """The fixed logarithmic bin edges every summary is built on."""
    return np.logspace(HIST_LOG_MIN, HIST_LOG_MAX, HIST_BIN_COUNT + 1)


# The measures whose mean splits into an entangled fraction and a conditional
# magnitude, and the threshold each is counted positive at.
#
# These mirror `dist_scaling.exe` exactly -- 1e-12 for the pair negativities
# and MIPT_DIST_GMN_ZERO_TOL (1e-10) for the SDP family -- so a decomposition
# built from the records reproduces the `<metric>_positive_count` column of
# the aggregate CSV rather than merely resembling it. Both are exact edges of
# the histogram grid, so the count is a sum of whole bins.
_NEGATIVITY_POSITIVE_TOL = {
    "mn": 1.0e-12,
    "fmn": 1.0e-12,
    "gmn": 1.0e-10,
    "fgmn": 1.0e-10,
}
_NEGATIVITY_METRICS = frozenset(_NEGATIVITY_POSITIVE_TOL)

# Reconstructed rather than stored, so a summary carries the same metadata
# block every other file in the stack does.
SUMMARY_METADATA_COLUMNS = (
    "N",
    "circ_type",
    "circuit_name",
    "realizations",
    "p",
    "periods",
    "k",
    "entropy_units",
)


def is_record_file(path: str | Path) -> bool:
    """Whether ``path`` is a dist_scaling record binary, by its magic."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            return handle.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def is_summary_file(path: str | Path) -> bool:
    """Whether ``path`` is a record summary CSV, by its column set."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return False
    try:
        # The provenance block at the top is comment lines; without skipping
        # them the first one is read as the header and nothing matches.
        header = pd.read_csv(path, nrows=0, comment="#")
    except Exception:
        return False
    columns = {str(column).strip().lower() for column in header.columns}
    return {"metric", "bin_kind", "count"}.issubset(columns)


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


def record_file_info(path: str | Path) -> dict[str, Any]:
    """Everything about a record file except its records.

    Reads only the preamble and header, so this is instant on a file of any
    size and is what every streaming reader starts from.
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
    count = int(payload_bytes // dtype.itemsize)
    return {
        "path": str(path),
        "header": header,
        "geometries": geometries,
        "dtype": dtype,
        "observables": observables,
        "payload_offset": payload_offset,
        "record_bytes": int(record_bytes),
        "record_count": count,
        # What a killed run left mid-write. The record size is fixed, so this
        # is the whole reason the format is a flat array and not a container
        # with a footer.
        "truncated": int(payload_bytes - count * dtype.itemsize),
        "k": int(header.get("k", 2)),
        "L": int(header["N"]) if "N" in header else None,
        "circuit_type": header.get("circ_type"),
        "entropy_units": header.get("entropy_units", "bits"),
        "circuit_name": header.get("circuit_name"),
    }


def iter_record_chunks(
    path: str | Path,
    *,
    chunk_records: int = 1_000_000,
    info: dict[str, Any] | None = None,
) -> Iterator[np.ndarray]:
    """Yield the records in fixed-size blocks, as raw structured arrays.

    One chunk of the default size is 40 MB for a four-observable record, which
    is what keeps a pass over a multi-gigabyte file inside a small machine's
    memory. Nothing is widened and nothing is joined -- the caller reduces each
    chunk and drops it.
    """
    if chunk_records <= 0:
        raise ValueError("chunk_records must be positive.")
    info = info or record_file_info(path)
    dtype: np.dtype = info["dtype"]
    remaining = info["record_count"]
    offset = info["payload_offset"]
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        while remaining > 0:
            take = min(chunk_records, remaining)
            block = np.fromfile(handle, dtype=dtype, count=take)
            if block.size == 0:
                break
            yield block
            remaining -= block.size


# ---------------------------------------------------------------------------
# The streaming summary
# ---------------------------------------------------------------------------


class RecordSummary:
    """A record file reduced to per-geometry moments and histograms.

    Three arrays, all indexed ``[metric, geometry]``:

    ``counts``  finite records folded in
    ``sums``    their sum, and ``squares`` their sum of squares -- so the mean
                and the sample standard error come back exactly, not
                approximately
    ``nans``    records whose value was NaN: a GMN the schedule never asked
                for, or one the solver failed on. Neither a value nor a zero,
                so it is in no other count
    ``zeros``   records whose value was exactly 0.0

    plus ``hist[metric, geometry, bin]``, the counts of ``|value|`` on the
    fixed logarithmic grid. The magnitude is what is binned because the
    tripartite information is negative throughout the monitored phase; the
    *signed* value is what goes into ``sums``, so the moments still match the
    aggregate CSV term for term.
    """

    def __init__(self, info: dict[str, Any]):
        self.info = info
        self.observables: tuple[str, ...] = tuple(info["observables"])
        self.geometries: pd.DataFrame = info["geometries"]
        self.geometry_count = int(self.geometries["geometry_id"].max()) + 1
        shape = (len(self.observables), self.geometry_count)
        self.counts = np.zeros(shape, dtype=np.int64)
        self.nans = np.zeros(shape, dtype=np.int64)
        self.zeros = np.zeros(shape, dtype=np.int64)
        self.sums = np.zeros(shape, dtype=np.float64)
        self.squares = np.zeros(shape, dtype=np.float64)
        self.hist = np.zeros(shape + (HIST_BIN_COUNT,), dtype=np.int64)
        self.records_total = 0
        self.realization_max = -1
        self.edges = histogram_edges()

    # Convenience accessors -------------------------------------------------

    @property
    def k(self) -> int:
        return int(self.info["k"])

    @property
    def L(self) -> int | None:
        return self.info["L"]

    @property
    def trajectories(self) -> int:
        """Trajectories present. Ids are contiguous from zero."""
        return self.realization_max + 1

    def distance_of(self) -> np.ndarray:
        """``d`` by geometry id, NaN for ids the table does not name."""
        distance = np.full(self.geometry_count, np.nan)
        ids = self.geometries["geometry_id"].to_numpy(dtype=int)
        distance[ids] = self.geometries["d"].to_numpy(dtype=float)
        return distance

    def absorb(self, block: np.ndarray) -> None:
        """Fold one chunk of raw records in."""
        geometry = block["geometry_id"].astype(np.intp)
        self.records_total += block.size
        if block.size:
            self.realization_max = max(
                self.realization_max, int(block["realization_id"].max())
            )
        length = self.geometry_count
        for index, metric in enumerate(self.observables):
            value = block[metric]
            finite = np.isfinite(value)
            if not finite.all():
                self.nans[index] += np.bincount(
                    geometry[~finite], minlength=length
                )
            good_geometry = geometry[finite]
            good = value[finite]
            self.counts[index] += np.bincount(good_geometry, minlength=length)
            self.sums[index] += np.bincount(
                good_geometry, weights=good, minlength=length
            )
            self.squares[index] += np.bincount(
                good_geometry, weights=good * good, minlength=length
            )

            magnitude = np.abs(good)
            positive = magnitude > 0.0
            if not positive.all():
                self.zeros[index] += np.bincount(
                    good_geometry[~positive], minlength=length
                )
            magnitude = magnitude[positive]
            if magnitude.size == 0:
                continue
            # The grid is logarithmic with a fixed number of bins per decade,
            # so the bin index is arithmetic rather than a search -- which
            # matters at 10^8 values, where searchsorted's binary search over
            # 400 edges dominates the pass.
            bin_index = np.floor(
                (np.log10(magnitude) - HIST_LOG_MIN) * HIST_BINS_PER_DECADE
            ).astype(np.intp)
            np.clip(bin_index, 0, HIST_BIN_COUNT - 1, out=bin_index)
            flat = good_geometry[positive] * HIST_BIN_COUNT + bin_index
            self.hist[index] += np.bincount(
                flat, minlength=length * HIST_BIN_COUNT
            ).reshape(length, HIST_BIN_COUNT)

    # Reductions ------------------------------------------------------------

    def curves(
        self, *, positive_tol: dict[str, float] | float | None = None
    ) -> dict[str, pd.DataFrame]:
        """The ``{metric: DataFrame[d, mean, stderr, count]}`` the CSV path
        produces, plus the two decomposition factors for every negativity.

        The mean and standard error are reconstructed from the summed values
        and squares, so they equal what a pass over the individual records
        would give -- this is a lossless reduction of the first two moments,
        not a binned approximation.
        """
        distance = self.distance_of()
        curves: dict[str, pd.DataFrame] = {}
        edges = self.edges
        for index, metric in enumerate(self.observables):
            count = self.counts[index].astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(count > 0, self.sums[index] / count, np.nan)
                # The usual m2 = sum(x^2) - n*mean^2, then the sample variance.
                m2 = self.squares[index] - count * mean**2
                variance = np.where(count > 1, np.maximum(m2, 0.0) / (count - 1.0), np.nan)
                stderr = np.where(count > 1, np.sqrt(variance / count), np.nan)
            frame = _curve_frame(distance, mean, stderr, self.counts[index])
            if frame is None:
                continue
            curves[metric] = frame

            if metric not in _NEGATIVITY_METRICS:
                continue
            # Positive means "above the tolerance", the same test the C++
            # positivity counts use -- and the same per-metric value, so this
            # reproduces the aggregate CSV's `<metric>_positive_count`. Every
            # power of ten is a bin edge, so the count is a sum of whole bins.
            if isinstance(positive_tol, dict):
                tolerance = positive_tol.get(metric, _NEGATIVITY_POSITIVE_TOL[metric])
            elif positive_tol is None:
                tolerance = _NEGATIVITY_POSITIVE_TOL[metric]
            else:
                tolerance = float(positive_tol)
            above = edges[:-1] >= tolerance
            positive = self.hist[index][:, above].sum(axis=1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                fraction = np.where(count > 0, positive / count, np.nan)
                fraction_stderr = np.where(
                    count > 0,
                    np.sqrt(np.maximum(fraction * (1.0 - fraction), 0.0) / count),
                    np.nan,
                )
                magnitude = np.where(positive > 0, mean / fraction, np.nan)
                # The conditional variance follows from the same identity the
                # aggregate path uses: the zero records contribute nothing to
                # either sum, so the entangled records' own m2 is
                # m2 + n*mean^2 - n^2*mean^2/m.
                m2_entangled = m2 + count * mean**2 - np.square(count) * mean**2 / positive
                magnitude_stderr = np.where(
                    positive > 1,
                    np.sqrt(np.maximum(m2_entangled, 0.0) / (positive * (positive - 1.0))),
                    np.nan,
                )
            fraction_frame = _curve_frame(
                distance, fraction, fraction_stderr, self.counts[index]
            )
            magnitude_frame = _curve_frame(
                distance, magnitude, magnitude_stderr, positive.astype(np.int64)
            )
            if fraction_frame is not None:
                curves[f"{metric}_ent_fraction"] = fraction_frame
            if magnitude_frame is not None:
                curves[f"{metric}_ent_magnitude"] = magnitude_frame
        return curves

    def distribution(
        self, metric: str, *, d_min: float, zero_tol: float
    ) -> dict[str, Any] | None:
        """The histogram of one metric over the geometries at ``d >= d_min``.

        Returns the occupied bin edges and counts, together with how many
        records went into the panel and how many of them were zero -- where
        "zero" is at or below ``zero_tol``, folding in the floating-point
        noise a vanishing measure leaves behind.
        """
        if metric not in self.observables:
            return None
        index = self.observables.index(metric)
        distance = self.distance_of()
        selected = np.isfinite(distance) & (distance >= d_min)
        if not selected.any():
            return None
        counts = self.hist[index][selected].sum(axis=0)
        records = int(self.counts[index][selected].sum())
        if records == 0:
            return None
        above = self.edges[:-1] >= zero_tol
        positive = int(counts[above].sum())
        return {
            "edges": self.edges,
            "counts": counts,
            "above": above,
            "records": records,
            "positive": positive,
            "zeros": records - positive,
            "d_min": d_min,
            "zero_tol": zero_tol,
        }


def _curve_frame(
    distance: np.ndarray,
    mean: np.ndarray,
    stderr: np.ndarray,
    count: np.ndarray,
) -> pd.DataFrame | None:
    """One measure's per-geometry values as a curve, pooled by distance."""
    frame = pd.DataFrame(
        {
            "d": distance,
            "mean": mean,
            "stderr": stderr,
            "count": np.asarray(count, dtype=np.int64),
        }
    )
    frame = frame.loc[np.isfinite(frame["d"]) & (frame["count"] > 0)]
    if frame.empty:
        return None
    # Two geometries can share an effective distance -- two triangles with
    # different sides but the same geometric mean -- so they are pooled by
    # sample count rather than plotted as coincident points.
    if frame["d"].duplicated().any():
        rows = []
        for value, group in frame.groupby("d", sort=True):
            weight = group["count"].to_numpy(dtype=float)
            group_mean = group["mean"].to_numpy(dtype=float)
            group_stderr = np.nan_to_num(group["stderr"].to_numpy(dtype=float))
            usable = np.isfinite(group_mean) & (weight > 0)
            if not usable.any():
                continue
            weight, group_mean = weight[usable], group_mean[usable]
            group_stderr = group_stderr[usable]
            total = weight.sum()
            pooled = float((weight * group_mean).sum() / total)
            within = float(((weight / total) ** 2 * group_stderr**2).sum())
            between = (
                float(
                    (weight * (group_mean - pooled) ** 2).sum()
                    / (total * (len(weight) - 1))
                )
                if len(weight) > 1
                else 0.0
            )
            rows.append(
                {
                    "d": float(value),
                    "mean": pooled,
                    "stderr": float(np.sqrt(max(0.0, within + between / total))),
                    "count": int(total),
                }
            )
        frame = pd.DataFrame(rows, columns=["d", "mean", "stderr", "count"])
    return frame.sort_values("d").reset_index(drop=True)


def summarize(
    path: str | Path,
    *,
    chunk_records: int = 1_000_000,
    progress: bool = False,
) -> RecordSummary:
    """Stream a record file once and reduce it to a :class:`RecordSummary`.

    Memory is bounded by ``chunk_records`` regardless of file size. Time is
    one sequential read: a 2.7 GB file takes tens of seconds.
    """
    info = record_file_info(path)
    summary = RecordSummary(info)
    done = 0
    for block in iter_record_chunks(path, chunk_records=chunk_records, info=info):
        summary.absorb(block)
        done += block.size
        if progress:
            share = 100.0 * done / max(1, info["record_count"])
            print(
                f"\r  {done:,} / {info['record_count']:,} records ({share:5.1f}%)",
                end="",
                flush=True,
            )
    if progress:
        print()
    return summary


# ---------------------------------------------------------------------------
# The summary CSV
# ---------------------------------------------------------------------------


def _metadata_values(info: dict[str, Any]) -> dict[str, Any]:
    header = info["header"]
    return {
        "N": header.get("N"),
        "circ_type": header.get("circ_type"),
        "circuit_name": header.get("circuit_name"),
        "realizations": header.get("realizations"),
        "p": header.get("p"),
        "periods": header.get("periods"),
        "k": header.get("k"),
        "entropy_units": header.get("entropy_units", "bits"),
    }


def summary_to_frame(summary: RecordSummary) -> pd.DataFrame:
    """The summary as a long-format table, one row per occupied bin.

    Three kinds of row, told apart by ``bin_kind``:

    ``nan``    records with no value at all -- a GMN the schedule never asked
               for, or one the solver failed on
    ``zero``   records whose value was exactly 0.0
    ``value``  one row per occupied logarithmic bin of ``|value|``

    ``sum`` and ``sumsq`` are of the *signed* values in the row, so adding
    them over a geometry's rows reproduces that geometry's mean and standard
    error exactly. Empty bins are dropped, which is what keeps the file a few
    thousand rows rather than a few tens of thousands.
    """
    metadata = _metadata_values(summary.info)
    distance = summary.distance_of()
    edges = summary.edges
    rows: list[dict[str, Any]] = []

    for index, metric in enumerate(summary.observables):
        for geometry in range(summary.geometry_count):
            base = {
                **metadata,
                "metric": metric,
                "geometry_id": geometry,
                "d": distance[geometry],
            }
            if summary.nans[index, geometry]:
                rows.append(
                    {
                        **base,
                        "bin_kind": "nan",
                        "bin_low": np.nan,
                        "bin_high": np.nan,
                        "count": int(summary.nans[index, geometry]),
                        "sum": np.nan,
                        "sumsq": np.nan,
                    }
                )
            # The moments are tracked per geometry, not per bin, so they ride
            # on the first row that geometry emits and are NaN on the rest.
            # Summing the column over a geometry therefore still gives that
            # geometry's exact totals, which is what makes the reconstructed
            # mean and standard error equal the ones a pass over the
            # individual records would produce.
            pending = (
                float(summary.sums[index, geometry]),
                float(summary.squares[index, geometry]),
            )
            if summary.zeros[index, geometry]:
                rows.append(
                    {
                        **base,
                        "bin_kind": "zero",
                        "bin_low": 0.0,
                        "bin_high": 0.0,
                        "count": int(summary.zeros[index, geometry]),
                        "sum": pending[0],
                        "sumsq": pending[1],
                    }
                )
                pending = (np.nan, np.nan)
            counts = summary.hist[index, geometry]
            for bin_index in np.flatnonzero(counts):
                rows.append(
                    {
                        **base,
                        "bin_kind": "value",
                        "bin_low": edges[bin_index],
                        "bin_high": edges[bin_index + 1],
                        "count": int(counts[bin_index]),
                        "sum": pending[0],
                        "sumsq": pending[1],
                    }
                )
                pending = (np.nan, np.nan)

    return pd.DataFrame(rows)


def write_summary_csv(summary: RecordSummary, path: str | Path) -> Path:
    """Write the summary CSV, with the run parameters in its first columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = summary_to_frame(summary)
    columns = list(SUMMARY_METADATA_COLUMNS) + [
        "metric",
        "geometry_id",
        "d",
        "bin_kind",
        "bin_low",
        "bin_high",
        "count",
        "sum",
        "sumsq",
    ]
    frame = frame[columns]
    with path.open("w", newline="") as handle:
        # A short provenance block, ignored by the reader (pandas skips '#'
        # lines) but the first thing a person opening the file will want.
        handle.write(
            f"# dist_scaling record summary of {Path(summary.info['path']).name}\n"
            f"# records={summary.records_total} trajectories={summary.trajectories} "
            f"histogram={HIST_BINS_PER_DECADE}/decade over "
            f"1e{HIST_LOG_MIN}..1e{HIST_LOG_MAX}\n"
        )
        frame.to_csv(handle, index=False, float_format="%.17g")
    return path


def read_summary_csv(path: str | Path) -> RecordSummary:
    """Rebuild a :class:`RecordSummary` from its CSV."""
    path = Path(path)
    frame = pd.read_csv(path, comment="#")
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"metric", "geometry_id", "d", "bin_kind", "count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is not a record summary CSV; missing {sorted(missing)}.")

    observables = list(dict.fromkeys(frame["metric"].astype(str)))
    geometry_ids = frame["geometry_id"].astype(int)
    geometry_count = int(geometry_ids.max()) + 1
    distance = np.full(geometry_count, np.nan)
    for geometry, group in frame.groupby("geometry_id"):
        values = pd.to_numeric(group["d"], errors="coerce").dropna().unique()
        if len(values):
            distance[int(geometry)] = float(values[0])

    def scalar(column: str) -> Any:
        if column not in frame.columns:
            return None
        values = frame[column].dropna().unique()
        return values[0] if len(values) else None

    header = {
        "N": scalar("N"),
        "circ_type": scalar("circ_type"),
        "circuit_name": scalar("circuit_name"),
        "realizations": scalar("realizations"),
        "p": scalar("p"),
        "periods": scalar("periods"),
        "k": scalar("k"),
        "entropy_units": scalar("entropy_units") or "bits",
    }
    for key in ("N", "circ_type", "periods", "realizations", "k"):
        if header[key] is not None:
            header[key] = int(header[key])

    info = {
        "path": str(path),
        "header": header,
        "geometries": pd.DataFrame(
            {"geometry_id": np.arange(geometry_count), "d": distance}
        ),
        "observables": observables,
        "k": int(header["k"]) if header["k"] is not None else 2,
        "L": header["N"],
        "circuit_type": header["circ_type"],
        "entropy_units": header["entropy_units"],
        "circuit_name": header["circuit_name"],
        "record_count": int(frame["count"].sum()),
        "truncated": 0,
        "dtype": None,
        "payload_offset": 0,
        "record_bytes": 0,
    }
    summary = RecordSummary(info)

    metric_index = {metric: index for index, metric in enumerate(observables)}
    index_of = frame["metric"].astype(str).map(metric_index).to_numpy(dtype=np.intp)
    geometry_of = geometry_ids.to_numpy(dtype=np.intp)
    count_of = pd.to_numeric(frame["count"], errors="coerce").to_numpy(dtype=np.int64)
    kind_of = frame["bin_kind"].astype(str).to_numpy()
    shape = (len(observables), geometry_count)

    def scatter(target: np.ndarray, mask: np.ndarray, values: np.ndarray) -> None:
        if not mask.any():
            return
        flat = index_of[mask] * geometry_count + geometry_of[mask]
        target += np.bincount(
            flat, weights=values[mask], minlength=shape[0] * shape[1]
        ).reshape(shape).astype(target.dtype)

    is_nan = kind_of == "nan"
    scatter(summary.nans, is_nan, count_of.astype(float))
    scatter(summary.zeros, kind_of == "zero", count_of.astype(float))
    scatter(summary.counts, ~is_nan, count_of.astype(float))

    for column, target in (("sum", summary.sums), ("sumsq", summary.squares)):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        scatter(target, np.isfinite(values), values)

    # Bin edges round-trip through 17 significant digits, so a value bin is
    # matched to its slot by rounding on the logarithmic grid rather than by
    # comparing floats for equality.
    is_value = kind_of == "value"
    if is_value.any():
        bin_low = pd.to_numeric(frame["bin_low"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            slots = np.floor(
                (np.log10(bin_low[is_value]) - HIST_LOG_MIN) * HIST_BINS_PER_DECADE + 0.5
            )
        slots = np.clip(np.nan_to_num(slots), 0, HIST_BIN_COUNT - 1).astype(np.intp)
        flat = (
            index_of[is_value] * geometry_count + geometry_of[is_value]
        ) * HIST_BIN_COUNT + slots
        summary.hist += (
            np.bincount(
                flat,
                weights=count_of[is_value].astype(float),
                minlength=shape[0] * shape[1] * HIST_BIN_COUNT,
            )
            .reshape(shape + (HIST_BIN_COUNT,))
            .astype(np.int64)
        )
    summary.records_total = int(count_of.sum())
    return summary


def default_summary_path(record_path: str | Path) -> Path:
    """``<stem>_summary.csv`` beside a ``<stem>_records.bin``."""
    path = Path(record_path)
    stem = path.with_suffix("")
    name = stem.name
    if name.endswith("_records"):
        name = name[: -len("_records")]
    return stem.with_name(name + "_summary.csv")


# ---------------------------------------------------------------------------
# The eager reader, for files small enough to hold
# ---------------------------------------------------------------------------

DEFAULT_MAX_FRAME_BYTES = 1 << 30  # 1 GiB


def read_records(
    path: str | Path,
    *,
    max_frame_bytes: int | None = DEFAULT_MAX_FRAME_BYTES,
) -> dict[str, Any]:
    """Read every record into a frame. Only for files that fit in memory.

    Returns ``{"path", "header", "geometries", "records", "observables", "k",
    "L", "circuit_type", "truncated"}``, with ``d`` joined in from the
    geometry table.

    A production file does not fit: at N=24 a k=2 run writes 276 records per
    trajectory, so 243k trajectories is 67 million of them -- 2.7 GB on disk
    and 4.3 GB once pandas has widened the identifiers and materialised ``d``
    per row. This therefore refuses above ``max_frame_bytes`` and points at
    :func:`summarize`, which needs no more memory for a large file than a
    small one. Pass ``max_frame_bytes=None`` to override, knowing what it
    costs.
    """
    info = record_file_info(path)
    estimate = info["record_count"] * (8 * 3 + 8 * len(info["observables"]) + 8)
    if max_frame_bytes is not None and estimate > max_frame_bytes:
        raise MemoryError(
            f"{path} holds {info['record_count']:,} records, which would take "
            f"about {estimate / 1e9:.1f} GB as a frame -- more than the "
            f"{max_frame_bytes / 1e9:.1f} GB budget. Use "
            "data_analysis.records.summarize(path) to stream it in bounded "
            "memory, or `python -m data_analysis.summarize_records <path>` to "
            "write the summary CSV once and read that instead. Pass "
            "max_frame_bytes=None to force the read."
        )

    dtype: np.dtype = info["dtype"]
    raw = np.fromfile(
        path, dtype=dtype, count=info["record_count"], offset=info["payload_offset"]
    )
    frame = pd.DataFrame(
        {
            "realization_id": raw["realization_id"].astype(np.int64),
            "geometry_id": raw["geometry_id"].astype(np.int64),
            "embedding_id": raw["embedding_id"].astype(np.int64),
            **{name: raw[name].astype(float) for name in info["observables"]},
        }
    )
    lookup = info["geometries"].set_index("geometry_id")["d"]
    frame["d"] = frame["geometry_id"].map(lookup).astype(float)

    bundle = {key: info[key] for key in ("path", "header", "geometries", "k", "L")}
    bundle.update(
        {
            "records": frame,
            "observables": info["observables"],
            "circuit_type": info["circuit_type"],
            "entropy_units": info["entropy_units"],
            "circuit_name": info["circuit_name"],
            "truncated": info["truncated"],
        }
    )
    return bundle


def aggregate_records(bundle: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Reduce an in-memory :func:`read_records` bundle to decay curves.

    Equivalent to ``summarize(path).curves()`` and kept for callers that
    already hold the records; the streaming path is what large files use.
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
        curves[metric] = (
            pd.DataFrame(
                {
                    "d": count.index.to_numpy(dtype=float),
                    "mean": mean.to_numpy(dtype=float),
                    "stderr": stderr.to_numpy(dtype=float),
                    "count": count.to_numpy(dtype=np.int64),
                }
            )
            .sort_values("d")
            .reset_index(drop=True)
        )

        if metric not in _NEGATIVITY_METRICS:
            continue
        # The same per-metric threshold the streaming path and the C++ use, so
        # the three agree on what counts as entangled.
        tolerance = _NEGATIVITY_POSITIVE_TOL[metric]
        positive = grouped.apply(lambda column: float((column > tolerance).sum()))
        fraction = positive / count
        with np.errstate(invalid="ignore", divide="ignore"):
            magnitude = mean / fraction
            fraction_stderr = np.sqrt(
                np.maximum(fraction * (1.0 - fraction), 0.0) / count
            )
        curves[f"{metric}_ent_fraction"] = (
            pd.DataFrame(
                {
                    "d": count.index.to_numpy(dtype=float),
                    "mean": fraction.to_numpy(dtype=float),
                    "stderr": fraction_stderr.to_numpy(dtype=float),
                    "count": count.to_numpy(dtype=np.int64),
                }
            )
            .sort_values("d")
            .reset_index(drop=True)
        )
        entangled = values.loc[values[metric] > tolerance].groupby("d")[metric]
        curves[f"{metric}_ent_magnitude"] = (
            pd.DataFrame(
                {
                    "d": magnitude.index.to_numpy(dtype=float),
                    "mean": magnitude.to_numpy(dtype=float),
                    "stderr": (entangled.std(ddof=1) / np.sqrt(entangled.count()))
                    .reindex(magnitude.index)
                    .to_numpy(dtype=float),
                    "count": positive.reindex(magnitude.index)
                    .fillna(0)
                    .to_numpy(dtype=np.int64),
                }
            )
            .sort_values("d")
            .reset_index(drop=True)
        )
    return curves

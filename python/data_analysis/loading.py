"""Input-file discovery, column lookup, and run-metadata resolution.

Every analysis function funnels its file handling through this module, so a
new naming convention or a new metadata column only has to be taught here.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _as_paths(files: Sequence[str | Path] | str | Path | None) -> list[Path]:
    if files is None:
        return []
    if isinstance(files, (str, Path)):
        return [Path(files)]
    return [Path(path) for path in files]


def _resolve_files(
    files: Sequence[str | Path] | str | Path | None,
    file_glob: str | Path | None,
) -> list[Path]:
    paths = _as_paths(files)
    if not paths and file_glob is not None:
        paths = [Path(path) for path in sorted(glob(str(file_glob)))]
    if not paths:
        raise FileNotFoundError("No input files were found.")
    return paths


def _find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    lowered = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise KeyError(
            f"Could not find any of {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _constant_csv_value(
    df: pd.DataFrame,
    candidates: Sequence[str],
    path: Path,
    *,
    integer: bool = False,
) -> float | int:
    column = _find_column(df, candidates)
    values = pd.to_numeric(df[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"{path}: expected one constant value in {column!r}, found {values}."
        )
    value = float(values[0])
    if integer:
        rounded = int(round(value))
        if not np.isclose(value, rounded, rtol=0.0, atol=1e-9):
            raise ValueError(f"{path}: {column!r} must contain an integer.")
        return rounded
    return value


def _parse_size(
    path: str | Path,
    overrides: Mapping[str, int] | None = None,
) -> int:
    path = Path(path)
    for pattern in (
        r"(?:^|[_-])n[_-]?(\d+)(?:[_-]|$)",
        r"(?:^|[_-])N[_-]?(\d+)(?:[_-]|$)",
        r"(?:^|[_-])l[_-]?(\d+)(?:[_-]|$)",
        r"(?:^|[_-])L[_-]?(\d+)(?:[_-]|$)",
    ):
        match = re.search(pattern, path.name)
        if match:
            return int(match.group(1))

    overrides = overrides or {}
    for key in (path.name, str(path)):
        if key in overrides:
            return int(overrides[key])

    raise ValueError(
        f"Could not parse L from {path.name!r}. Add it to l_by_file."
    )


def _parse_p(path: str | Path) -> float:
    match = re.search(
        r"(?:^|[_-])p[_-]?"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"(?:[_-]|$)",
        Path(path).name,
    )
    return float(match.group(1)) if match else np.nan


def _parse_circuit_name(path: str | Path) -> str:
    """Return the display name of a circuit encoded in an output filename."""
    aliases = {
        "mms": "MMS",
        "haar": "Haar U(4)",
        "rppu": "RPPU",
        "rfgs": "RFGS",
        "qrppu": "qRPPU",
    }
    match = re.search(
        r"(?:^|[_-])(qrppu|rppu|rfgs|haar|mms)(?:[_-]|$)",
        Path(path).name,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Could not parse the circuit type from {Path(path).name!r}. "
            "Expected one of: mms, haar, rppu, rfgs, qrppu."
        )
    return aliases[match.group(1).lower()]


def _chunked_group_summary(
    path: str | Path,
    value_columns: Sequence[str],
    *,
    p_min: float | None,
    p_max: float | None,
    chunksize: int,
) -> dict[str, pd.DataFrame]:
    """Calculate count, mean, and standard error by p in one CSV pass."""
    partials: dict[str, list[pd.DataFrame]] = {
        column: [] for column in value_columns
    }
    usecols = ["p", *value_columns]

    try:
        reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    except ValueError as exc:
        raise ValueError(
            f"{path!r} does not contain the expected columns {usecols}."
        ) from exc

    for chunk in reader:
        chunk["p"] = pd.to_numeric(chunk["p"], errors="coerce")
        for column in value_columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.replace([np.inf, -np.inf], np.nan)

        if p_min is not None:
            chunk = chunk.loc[chunk["p"] >= p_min]
        if p_max is not None:
            chunk = chunk.loc[chunk["p"] <= p_max]

        for column in value_columns:
            clean = chunk.dropna(subset=["p", column])
            if clean.empty:
                continue
            grouped = clean.groupby("p", sort=False)[column].agg(
                count="count",
                total="sum",
                sumsq=lambda values: np.square(
                    values.to_numpy(dtype=float)
                ).sum(),
            )
            partials[column].append(grouped)

    output: dict[str, pd.DataFrame] = {}
    for column, pieces in partials.items():
        if not pieces:
            output[column] = pd.DataFrame(
                columns=["p", "mean", "stderr", "count"]
            )
            continue

        combined = pd.concat(pieces).groupby(level=0).sum().sort_index()
        count = combined["count"].to_numpy(dtype=float)
        total = combined["total"].to_numpy(dtype=float)
        sumsq = combined["sumsq"].to_numpy(dtype=float)
        mean = total / count

        variance = np.full_like(mean, np.nan)
        valid = count > 1
        variance[valid] = (
            sumsq[valid] - total[valid] ** 2 / count[valid]
        ) / (count[valid] - 1)
        variance[valid] = np.maximum(variance[valid], 0.0)

        output[column] = pd.DataFrame(
            {
                "p": combined.index.to_numpy(dtype=float),
                "mean": mean,
                "stderr": np.sqrt(variance) / np.sqrt(count),
                "count": count.astype(int),
            }
        )

    return output


# ---------------------------------------------------------------------------
# Run metadata
#
# The C++ applications increasingly write their run parameters as real CSV
# columns (N, p, circ_type, circuit_name, realizations). Prefer those; fall
# back to the filename conventions above for the archive of older files.
# ---------------------------------------------------------------------------


CIRCUIT_DISPLAY_NAMES = {
    0: "MMS",
    1: "Haar U(4)",
    2: "RPPU",
    3: "RFGS",
    4: "qRPPU",
}

_SIZE_COLUMNS = ("N", "L", "n", "l", "system_size", "sites")
_P_COLUMNS = ("p", "p_value", "measurement_rate")
_CIRCUIT_NAME_COLUMNS = ("circuit_name", "circuit")
_CIRCUIT_TYPE_COLUMNS = ("circ_type", "circuit_type")
_REALIZATION_COLUMNS = (
    "realizations",
    "completed_realizations",
    "requested_realizations",
    "trajectories",
)


def _constant_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    integer: bool = False,
) -> float | int | None:
    """Return a column's single constant value, or None if absent or varying.

    Unlike :func:`_constant_csv_value` this never raises: a missing column or
    one that legitimately varies within the file (a p scan, a time series)
    simply means the caller should look elsewhere.
    """
    column = _find_column(df, candidates, required=False)
    if column is None:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        return None
    value = float(values[0])
    if not np.isfinite(value):
        return None
    if integer:
        rounded = int(round(value))
        return rounded if np.isclose(value, rounded, rtol=0.0, atol=1e-9) else None
    return value


def _constant_text_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    """Return a text column's single constant value, or None."""
    column = _find_column(df, candidates, required=False)
    if column is None:
        return None
    values = df[column].dropna().astype(str).str.strip().unique()
    if len(values) != 1 or not values[0]:
        return None
    return str(values[0])


def _optional_circuit_name(path: str | Path) -> str | None:
    """Filename circuit lookup that returns None instead of raising."""
    try:
        return _parse_circuit_name(path)
    except ValueError:
        return None


def resolve_metadata(
    df: pd.DataFrame,
    path: str | Path,
    *,
    l_by_file: Mapping[str, int] | None = None,
    require_size: bool = True,
) -> dict[str, Any]:
    """Resolve L, p, circuit, and realizations for one output file.

    CSV metadata columns win over the filename. ``source`` records where each
    field came from, which makes a surprising fit easy to diagnose.
    """
    source: dict[str, str] = {}

    size = _constant_column(df, _SIZE_COLUMNS, integer=True)
    if size is not None:
        source["L"] = "column"
    else:
        try:
            size = _parse_size(path, l_by_file)
            source["L"] = "filename"
        except ValueError:
            if require_size:
                raise
            source["L"] = "missing"

    p = _constant_column(df, _P_COLUMNS)
    if p is not None:
        source["p"] = "column"
    else:
        p = _parse_p(path)
        source["p"] = "filename" if np.isfinite(p) else "missing"

    circuit = _constant_text_column(df, _CIRCUIT_NAME_COLUMNS)
    if circuit is not None:
        source["circuit"] = "column"
    else:
        circ_type = _constant_column(df, _CIRCUIT_TYPE_COLUMNS, integer=True)
        if circ_type is not None and circ_type in CIRCUIT_DISPLAY_NAMES:
            circuit = CIRCUIT_DISPLAY_NAMES[circ_type]
            source["circuit"] = "column"
        else:
            circuit = _optional_circuit_name(path)
            source["circuit"] = "filename" if circuit else "missing"

    realizations = _constant_column(df, _REALIZATION_COLUMNS, integer=True)
    source["realizations"] = "column" if realizations is not None else "missing"

    return {
        "L": size,
        "p": p,
        "circuit": circuit,
        "realizations": realizations,
        "source": source,
    }


def metadata_sidecar_path(path: str | Path) -> Path:
    """Companion metadata file written beside a row-level solver CSV."""
    path = Path(path)
    return path.with_name(f"{path.stem}_meta.csv")


def read_metadata_sidecar(path: str | Path) -> dict[str, Any] | None:
    """Read the ``*_meta.csv`` written by gmn.exe/fgmn.exe, or None.

    The row-level GMN outputs run to millions of lines, so their run
    parameters live in a one-row sidecar instead of being repeated on every
    row. Older outputs have no sidecar; callers fall back to the filename.
    """
    sidecar = metadata_sidecar_path(path)
    if not sidecar.is_file():
        return None
    frame = pd.read_csv(sidecar)
    if frame.empty:
        return None
    row = frame.iloc[0]
    return {str(key): row[key] for key in frame.columns}

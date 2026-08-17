"""Distance scaling of two- and three-party entanglement measures.

Reads both output formats of ``dist_scaling.exe``:

* the **aggregated** format it writes now -- one row per geometry carrying
  ``<metric>_mean``, ``<metric>_stderr`` and ``<metric>_samples``, for pairs
  (``k=2``) or balanced triangles (``k=3``);
* the older **row-level** format -- one ``d,mi,mn`` (or ``d,mi,fmi,mn,fmn``)
  row per pair per trajectory.

Both end up as the same ``{metric: DataFrame[d, mean, stderr, count]}``
structure, so the fitting and plotting below never has to know which one it
came from.

Files at different party counts may be plotted together -- a ``k=0`` run of
``dist_scaling.exe`` writes one of each from the same trajectories -- so the
loaded curves are keyed by ``(k, L)`` and the panels are named by the quantity
they hold rather than by how many parties measured it.
"""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fitting import _weighted_linear_fit
from .loading import (
    CIRCUIT_DISPLAY_NAMES,
    _parse_circuit_name,
    _parse_size,
    _resolve_files,
    resolve_metadata,
)
from .plotting import (
    _color_map_by_size,
    _mi_unit_spec,
    _positive_log_errorbar,
    _show,
)

# Distinguishes "argument not supplied" from an explicit None, which for the
# `fit_last` family means "use every point" rather than "use the default".
_UNSET = object()


# ---------------------------------------------------------------------------
# Metric catalogue
#
# `entropy` marks the measures that carry entropy units and therefore follow
# `mi_units`; `magnitude` marks the ones whose sign is fixed by physics and
# whose decay is read off the absolute value. The tripartite information of a
# monitored circuit is negative, so a log-log fit needs |I_3|.
# ---------------------------------------------------------------------------

_METRIC_SPECS: dict[str, dict[str, Any]] = {
    "mi": {"exponent": "MI", "entropy": True},
    "fmi": {"exponent": r"\mathrm{fMI}", "entropy": True},
    "mn": {"exponent": "MN"},
    "fmn": {"exponent": r"\mathrm{fMN}"},
    "tmi": {"exponent": r"\mathrm{TMI}", "entropy": True, "magnitude": True},
    "ftmi": {"exponent": r"\mathrm{fTMI}", "entropy": True, "magnitude": True},
    "gmn": {"exponent": r"\mathrm{GMN}"},
    "fgmn": {"exponent": r"\mathrm{fGMN}"},
    "average_mi": {"exponent": r"\overline{\mathrm{MI}}", "entropy": True},
    "faverage_mi": {"exponent": r"\overline{\mathrm{fMI}}", "entropy": True},
    "min_bipneg": {"exponent": r"\mathrm{minN}"},
    "min_fbipneg": {"exponent": r"\mathrm{minN}^f"},
}

# ---------------------------------------------------------------------------
# Panel layout
#
# Rows are the two quantity families (information, then negativity) and
# columns the two trace conventions (ordinary, then fermionic), so a panel is
# named by *what* it measures rather than by how many parties measured it.
# Each slot lists the metric that realizes it at each party count, which is
# what lets one panel carry k=2 and k=3 together: the ordinary-information
# panel holds I_2 at k=2 and |I_3| at k=3.
#
# Everything else a file carries -- the supporting mean-MI, minimum-negativity
# and purity columns -- is still loaded and still fitted, it just takes up no
# subplot.
# ---------------------------------------------------------------------------

_PANEL_GRID: tuple[tuple[dict[str, Any], ...], ...] = (
    (
        {
            "label": "Mutual Information",
            "entropy": True,
            "metrics": {2: "mi", 3: "tmi"},
        },
        {
            "label": "Fermionic Mutual Information",
            "entropy": True,
            "metrics": {2: "fmi", 3: "ftmi"},
        },
    ),
    (
        {"label": "Negativity", "entropy": False, "metrics": {2: "mn", 3: "gmn"}},
        {
            "label": "Fermionic Negativity",
            "entropy": False,
            "metrics": {2: "fmn", 3: "fgmn"},
        },
    ),
)

# Every metric that has a panel, and the slot it belongs to.
_PANEL_SLOT_OF_METRIC: dict[str, tuple[int, int]] = {
    metric: (row_index, column_index)
    for row_index, row in enumerate(_PANEL_GRID)
    for column_index, slot in enumerate(row)
    for metric in slot["metrics"].values()
}

# k=2 and k=3 share a panel, so they are told apart by marker and by the
# line style of their fit, with colour left to carry the system size.
_K_MARKERS = {2: "o", 3: "s"}
_K_LINESTYLES = {2: "--", 3: ":"}

# Loaded and returned, but never power-law fitted: a purity tends to a
# constant with distance, so an exponent for it would be noise dressed as a
# number.
_NON_SCALING_METRICS = frozenset(
    {
        "joint_purity",
        "mean_single_purity",
        "fjoint_purity",
        "fmean_single_purity",
    }
)


def _metric_spec(metric: str) -> dict[str, Any]:
    return _METRIC_SPECS.get(metric, {"exponent": metric.replace("_", r"\_")})


def _marker_for_k(k: int) -> str:
    return _K_MARKERS.get(k, "^")


def _linestyle_for_k(k: int) -> str:
    return _K_LINESTYLES.get(k, "-.")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _chunked_distance_summary(
    path: str | Path,
    *,
    metrics: Sequence[str],
    chunksize: int,
    distance_round: int,
) -> dict[str, pd.DataFrame]:
    """Calculate count, mean, and standard error by chord distance.

    This is the row-level path: the older CSVs carry one row per pair per
    trajectory and run to millions of lines, so they are streamed rather than
    loaded.
    """
    metrics = tuple(metrics)
    partials: dict[str, list[pd.DataFrame]] = {
        metric: [] for metric in metrics
    }
    required_columns = ["d", *metrics]

    try:
        reader = pd.read_csv(
            path,
            usecols=required_columns,
            chunksize=chunksize,
        )
    except ValueError as exc:
        raise ValueError(
            f"{path!r} must contain the columns "
            + ", ".join(required_columns)
            + "."
        ) from exc

    for chunk in reader:
        for column in required_columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.loc[np.isfinite(chunk["d"]) & (chunk["d"] > 0.0)]
        if chunk.empty:
            continue
        chunk["_d"] = chunk["d"].round(distance_round)

        for metric in metrics:
            clean = chunk.dropna(subset=[metric])
            if clean.empty:
                continue
            grouped = clean.groupby("_d", sort=False)[metric].agg(
                count="count",
                total="sum",
                sumsq=lambda values: np.square(
                    values.to_numpy(dtype=float)
                ).sum(),
            )
            partials[metric].append(grouped)

    output: dict[str, pd.DataFrame] = {}
    for metric, pieces in partials.items():
        if not pieces:
            output[metric] = pd.DataFrame(
                columns=["d", "mean", "stderr", "count"]
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

        output[metric] = pd.DataFrame(
            {
                "d": combined.index.to_numpy(dtype=float),
                "mean": mean,
                "stderr": np.sqrt(variance) / np.sqrt(count),
                "count": count.astype(np.int64),
            }
        )

    return output


def _aggregate_metric_names(columns: Sequence[str]) -> list[str]:
    """Metric stems of an aggregated file, in column order."""
    lowered = {str(column).strip().lower() for column in columns}
    return [
        column[: -len("_mean")]
        for column in columns
        if str(column).strip().lower().endswith("_mean")
        and f"{str(column).strip().lower()[:-len('_mean')]}_stderr" in lowered
    ]


def _read_aggregate_file(path: str | Path) -> dict[str, Any]:
    """Read one aggregated dist_scaling CSV."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"No rows were found in {path}.")
    if "d" not in {str(column).strip().lower() for column in frame.columns}:
        raise ValueError(
            f"{path}: an aggregated dist_scaling CSV must carry an effective "
            "chord-distance column 'd'."
        )
    frame.columns = [str(column).strip() for column in frame.columns]

    distance = pd.to_numeric(frame["d"], errors="coerce")
    data: dict[str, pd.DataFrame] = {}
    for metric in _aggregate_metric_names(frame.columns):
        samples_column = f"{metric}_samples"
        count = (
            pd.to_numeric(frame[samples_column], errors="coerce")
            if samples_column in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        curve = pd.DataFrame(
            {
                "d": distance,
                "mean": pd.to_numeric(frame[f"{metric}_mean"], errors="coerce"),
                "stderr": pd.to_numeric(
                    frame[f"{metric}_stderr"], errors="coerce"
                ),
                "count": count.fillna(0).astype(np.int64),
            }
        )
        curve = curve.replace([np.inf, -np.inf], np.nan)
        curve = curve.loc[np.isfinite(curve["d"]) & (curve["d"] > 0.0)]
        # Several rows can share one effective distance -- two triangle
        # geometries with different side lengths but the same geometric mean --
        # so pool them by their sample counts rather than plotting a column of
        # coincident points.
        data[metric] = _pool_by_distance(curve)

    k_column = next(
        (column for column in frame.columns if column.strip().lower() == "k"),
        None,
    )
    if k_column is not None:
        k_values = pd.to_numeric(frame[k_column], errors="coerce").dropna().unique()
        k = int(round(float(k_values[0]))) if len(k_values) == 1 else None
    else:
        k = 3 if "tmi" in data else 2

    units_column = next(
        (
            column
            for column in frame.columns
            if column.strip().lower() in {"entropy_units", "mi_units", "units"}
        ),
        None,
    )
    stored_units = None
    if units_column is not None:
        values = frame[units_column].dropna().astype(str).str.strip().unique()
        if len(values) == 1 and values[0]:
            stored_units = values[0].lower()

    # `circuit_name` in the CSV is the long display name; the integer
    # `circ_type` maps onto the same short labels the filename convention
    # uses, so a run given as an aggregate file and one given as a row-level
    # file compare equal.
    circuit = None
    type_column = next(
        (
            column
            for column in frame.columns
            if column.strip().lower() in {"circ_type", "circuit_type"}
        ),
        None,
    )
    if type_column is not None:
        values = pd.to_numeric(frame[type_column], errors="coerce").dropna().unique()
        if len(values) == 1:
            circuit = CIRCUIT_DISPLAY_NAMES.get(int(round(float(values[0]))))

    return {
        "data": data,
        "k": k,
        "stored_units": stored_units,
        "circuit_from_type": circuit,
        "frame": frame,
    }


def _pool_by_distance(curve: pd.DataFrame) -> pd.DataFrame:
    """Combine rows sharing one effective distance into one weighted point."""
    if curve.empty or not curve["d"].duplicated().any():
        return curve.sort_values("d").reset_index(drop=True)

    rows: list[dict[str, float]] = []
    for distance, group in curve.groupby("d", sort=True):
        weight = group["count"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        stderr = group["stderr"].to_numpy(dtype=float)
        usable = np.isfinite(mean) & (weight > 0)
        if not usable.any():
            continue
        weight, mean = weight[usable], mean[usable]
        stderr = np.where(np.isfinite(stderr[usable]), stderr[usable], 0.0)
        total = weight.sum()
        pooled_mean = float((weight * mean).sum() / total)
        # Each group already reports the standard error of its own mean, so
        # the pooled error is the error of a weighted average of independent
        # estimates plus the spread between the groups themselves.
        within = float(((weight / total) ** 2 * stderr**2).sum())
        between = (
            float((weight * (mean - pooled_mean) ** 2).sum() / (total * (len(weight) - 1)))
            if len(weight) > 1
            else 0.0
        )
        rows.append(
            {
                "d": float(distance),
                "mean": pooled_mean,
                "stderr": float(np.sqrt(max(0.0, within + between / total))),
                "count": int(total),
            }
        )
    return pd.DataFrame(rows, columns=["d", "mean", "stderr", "count"])


def _is_aggregate_file(path: str | Path) -> bool:
    header = pd.read_csv(path, nrows=0)
    return bool(_aggregate_metric_names(header.columns))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _fit_range_for_size(
    fit_range: (
        tuple[float | None, float | None]
        | Mapping[int, tuple[float | None, float | None]]
        | None
    ),
    size: int,
) -> tuple[float | None, float | None] | None:
    if fit_range is None:
        return None
    if isinstance(fit_range, Mapping):
        return fit_range.get(size)
    return fit_range


def _normalize_fit_sizes(
    requested: int | Sequence[int] | None,
    parsed_sizes: Sequence[int],
    metric: str,
) -> list[int]:
    """Resolve a ``fit_size`` entry to a sorted list of system sizes."""
    if requested is None:
        return [max(parsed_sizes)]
    if isinstance(requested, (int, np.integer)):
        candidates = [int(requested)]
    else:
        candidates = [int(size) for size in requested]
        if not candidates:
            raise ValueError(f"fit_size[{metric!r}] must not be empty.")
    unknown = sorted(set(candidates) - set(parsed_sizes))
    if unknown:
        raise ValueError(
            f"fit_size[{metric!r}]={unknown if len(unknown) > 1 else unknown[0]} "
            f"is not among {list(parsed_sizes)}."
        )
    return sorted(set(candidates))


def _log_log_power_law_fit(
    selected: pd.DataFrame,
    *,
    min_relative_error: float,
) -> dict[str, float]:
    """Weighted least squares for mean = prefactor * d**(-alpha)."""
    if len(selected) < 3:
        raise ValueError(
            f"Only {len(selected)} positive points remain in the fit window; "
            "at least three are required."
        )

    relative_error = (
        selected["stderr"].to_numpy(dtype=float)
        / selected["mean"].to_numpy(dtype=float)
    )
    relative_error = np.where(
        np.isfinite(relative_error) & (relative_error > 0.0),
        relative_error,
        min_relative_error,
    )
    relative_error = np.maximum(relative_error, min_relative_error)

    linear = _weighted_linear_fit(
        np.log(selected["d"].to_numpy(dtype=float)),
        np.log(selected["mean"].to_numpy(dtype=float)),
        relative_error,
    )
    return {
        "alpha": -linear["slope"],
        "alpha_stderr": linear["slope_stderr"],
        "prefactor": float(np.exp(linear["intercept"])),
        "log_prefactor": linear["intercept"],
        "log_prefactor_stderr": linear["intercept_stderr"],
        "reduced_chi2": linear["reduced_chi2"],
        "d_min": float(selected["d"].min()),
        "d_max": float(selected["d"].max()),
        "points": int(len(selected)),
    }


def _distance_power_law_fit(
    curve: pd.DataFrame,
    *,
    fit_range: tuple[float | None, float | None] | None,
    fit_last: int | None,
    min_relative_error: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit mean = prefactor * d**(-alpha) in logarithmic coordinates."""
    selected = curve.loc[
        np.isfinite(curve["d"])
        & np.isfinite(curve["mean"])
        & (curve["d"] > 0.0)
        & (curve["mean"] > 0.0)
    ].copy()

    if fit_range is not None:
        lower, upper = fit_range
        if lower is not None:
            selected = selected.loc[selected["d"] >= lower]
        if upper is not None:
            selected = selected.loc[selected["d"] <= upper]
    elif fit_last is not None:
        if fit_last < 3:
            raise ValueError("fit_last must be at least 3 or None.")
        selected = selected.tail(fit_last)

    fit = _log_log_power_law_fit(
        selected, min_relative_error=min_relative_error
    )
    return fit, selected


def _merge_per_metric(
    shared: Any,
    named: Mapping[str, Any],
    metrics: Sequence[str],
    default: Any,
) -> dict[str, Any]:
    """Resolve a per-metric setting from a dict, the legacy kwargs, or a default.

    ``shared`` may be a mapping keyed by metric or one value applied to every
    metric; the ``fit_*_mi``-style keyword arguments win over it, and anything
    still unset falls back to ``default``. An explicit ``None`` is a value, not
    an omission -- ``fit_last={"tmi": None}`` asks for every point.
    """
    shared_map: Mapping[str, Any] | None = None
    shared_value: Any = _UNSET
    if (
        isinstance(shared, Mapping)
        and shared
        and set(shared) <= set(_METRIC_SPECS) | set(metrics)
    ):
        shared_map = shared
    elif shared is not _UNSET:
        # An explicit None reaches here and stays a value: `fit_last=None`
        # asks for every point, it does not ask for the default.
        shared_value = shared

    resolved: dict[str, Any] = {}
    for metric in metrics:
        value = named.get(metric, _UNSET)
        if value is _UNSET and shared_map is not None:
            value = shared_map.get(metric, _UNSET)
        if value is _UNSET:
            value = shared_value
        resolved[metric] = default if value is _UNSET else value
    return resolved


def dist_scaling(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    sizes: Sequence[int] | None = None,
    l_by_file: Mapping[str, int] | None = None,
    metrics: Sequence[str] | None = None,
    fit_size: Any = _UNSET,
    fit_range: Any = _UNSET,
    fit_last: Any = _UNSET,
    fit_size_mi: Any = _UNSET,
    fit_size_fmi: Any = _UNSET,
    fit_size_mn: Any = _UNSET,
    fit_size_fmn: Any = _UNSET,
    joint_fit: bool = False,
    fit_range_mi: Any = _UNSET,
    fit_range_fmi: Any = _UNSET,
    fit_range_mn: Any = _UNSET,
    fit_range_fmn: Any = _UNSET,
    fit_last_mi: Any = _UNSET,
    fit_last_fmi: Any = _UNSET,
    fit_last_mn: Any = _UNSET,
    fit_last_fmn: Any = _UNSET,
    min_relative_error: float = 0.03,
    chunksize: int = 1_000_000,
    distance_round: int = 12,
    mi_units: str = "bits",
    legacy_entropy_units: str = "bits",
    show_errorbars: bool = True,
    capsize: float = 2.0,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    dpi: int = 130,
    title: str | None = None,
    show_summary: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    r"""Plot Fig.-4-style distance scaling for two- and three-party measures.

    Accepts either output format of ``dist_scaling.exe``. The aggregated
    format has one row per geometry with ``<metric>_mean``/``_stderr``/
    ``_samples`` columns; the older row-level format has one ``d,mi,mn`` (or
    ``d,mi,fmi,mn,fmn``) row per pair per trajectory and is streamed in
    ``chunksize`` blocks. Files of both kinds may be mixed in one call as long
    as they share a circuit; **party counts may be mixed too**, which is what a
    ``k=0`` run of ``dist_scaling.exe`` produces -- a pair file and a triangle
    file measured on the same trajectories.

    Panels are named by the quantity they carry rather than by the number of
    parties that measured it, so both party counts share one panel:

    ============================ =========== ============
    panel                        ``k=2``     ``k=3``
    ============================ =========== ============
    Mutual Information           ``mi``      ``tmi``
    Fermionic Mutual Information ``fmi``     ``ftmi``
    Negativity                   ``mn``      ``gmn``
    Fermionic Negativity         ``fmn``     ``fgmn``
    ============================ =========== ============

    Within a panel the party count is carried by the marker (``o`` for
    ``k=2``, ``s`` for ``k=3``) and by the fit line style, and named in every
    legend entry; colour stays with the system size. The fermionic column is
    dropped for qubit ensembles, leaving mutual information above negativity.

    Every other measure the file carries -- ``average_mi``, ``min_bipneg``,
    the purities -- is still loaded into ``data`` and still fitted into
    ``fits``; it just gets no subplot. Pass ``metrics`` to override the panel
    set.

    ``data`` and ``selected_fit_points`` are keyed by ``(k, L)`` and
    ``(k, L, metric)`` respectively, and ``fits`` carries a ``k`` column,
    since one system size can now appear at both party counts.

    Rows are aggregated at equal chord distance, including zero-negativity
    events. All panels use logarithmic axes and are fitted with power laws

    .. math::

        O(d)=A_O\,d^{-\alpha_k^O}

    by weighted least squares in log coordinates. As in Fig. 4 of
    arXiv:2602.04969, the relative uncertainty used by the fit is floored at
    ``min_relative_error`` (default 0.03).

    The tripartite information is negative throughout the monitored phase, so
    ``tmi``/``ftmi`` are plotted and fitted as :math:`|I_3|`. Their signed
    means are kept in the ``signed_mean`` column of ``data``.

    ``fit_size``, ``fit_range``, and ``fit_last`` take a dictionary keyed by
    metric -- ``fit_last={"tmi": 5, "gmn": 3}`` -- or one value applied to
    every metric. The older ``fit_size_mi``/``fit_range_mn``/``fit_last_fmi``
    style keywords still work and win over the dictionary. Fit results for
    every size are returned in ``fits``, while dashed lines are drawn only for
    the corresponding ``fit_size`` entry (largest size by default). With a
    list of sizes, ``joint_fit=True`` pools their selected points into one
    weighted fit with a shared exponent, returned in ``joint_fits``.

    ``mi_units`` selects the unit of every entropy-valued measure (``mi``,
    ``fmi``, ``tmi``, ``ftmi``, ``average_mi``) and of their fitted
    prefactors; it does not change their exponents. ``dist_scaling.exe``
    writes those in **bits** and says so in its ``entropy_units`` column;
    ``legacy_entropy_units`` supplies the same information for the older
    row-level files, which carry no such column and are also log2.
    """
    if min_relative_error <= 0.0:
        raise ValueError("min_relative_error must be positive.")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    if distance_round < 0:
        raise ValueError("distance_round must be non-negative.")
    mi_units, mi_scale_from_nats = _mi_unit_spec(mi_units)
    _, legacy_scale_from_nats = _mi_unit_spec(legacy_entropy_units)

    paths = _resolve_files(files, file_glob)

    # --- read every file, then reconcile what they say about themselves -----
    loaded: list[dict[str, Any]] = []
    for path in paths:
        aggregated = _is_aggregate_file(path)
        entry: dict[str, Any] = {"path": path, "aggregated": aggregated}
        if aggregated:
            entry.update(_read_aggregate_file(path))
            metadata = resolve_metadata(
                entry["frame"], path, l_by_file=l_by_file, require_size=False
            )
            entry["size"] = metadata["L"]
            entry["circuit"] = entry["circuit_from_type"] or metadata["circuit"]
        else:
            entry["k"] = 2
            entry["stored_units"] = None
            entry["size"] = None
            entry["circuit"] = _parse_circuit_name(path)
        loaded.append(entry)

    circuit_names = sorted(
        {entry["circuit"] for entry in loaded if entry["circuit"]}
    )
    if len(circuit_names) != 1:
        raise ValueError(
            "Distance-scaling files must belong to one circuit type; found "
            f"{circuit_names}."
        )
    circuit_name = circuit_names[0]

    # Party counts may be mixed. A k=0 run of dist_scaling.exe writes one pair
    # file and one triangle file from the same trajectories, and the point of
    # plotting them together is to read the two-party and three-party
    # exponents off one ensemble.
    for entry in loaded:
        if entry["k"] is None:
            entry["k"] = 3 if "tmi" in entry.get("data", {}) else 2
    k_values = sorted({int(entry["k"]) for entry in loaded})

    if sizes is None:
        parsed_sizes = [
            entry["size"] if entry["size"] is not None else _parse_size(entry["path"], l_by_file)
            for entry in loaded
        ]
    else:
        parsed_sizes = [int(size) for size in sizes]
        if len(parsed_sizes) != len(paths):
            raise ValueError("sizes must match the number of input files.")

    keys = [(int(entry["k"]), int(size)) for entry, size in zip(loaded, parsed_sizes)]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(
            "Each (k, L) may appear once; duplicated "
            + ", ".join(f"k={k}, L={size}" for k, size in duplicates)
            + "."
        )

    order = sorted(range(len(keys)), key=lambda index: keys[index])
    loaded = [loaded[index] for index in order]
    parsed_sizes = [parsed_sizes[index] for index in order]
    keys = [keys[index] for index in order]
    unique_sizes = sorted(set(parsed_sizes))

    # The row-level format only exists for k=2 and only ever carried these
    # four columns; which of them are present is what says whether the run
    # reported the fermionic trace.
    legacy_metrics = (
        ("mi", "fmi", "mn", "fmn")
        if circuit_name in {"RPPU", "RFGS"}
        else ("mi", "mn")
    )

    data: dict[tuple[int, int], dict[str, pd.DataFrame]] = {}
    for key, entry in zip(keys, loaded):
        k_of_file, size = key
        print(f"Importing k={k_of_file}, L={size}: {entry['path']}")
        if entry["aggregated"]:
            curves = entry["data"]
            stored_scale = _mi_unit_spec(entry["stored_units"] or "bits")[1]
        else:
            curves = _chunked_distance_summary(
                entry["path"],
                metrics=legacy_metrics,
                chunksize=chunksize,
                distance_round=distance_round,
            )
            stored_scale = legacy_scale_from_nats

        # Stored values are in `stored_scale` units per nat; convert through
        # nats into whatever `mi_units` asked for.
        entropy_scale = mi_scale_from_nats / stored_scale
        for metric, curve in curves.items():
            spec = _metric_spec(metric)
            if spec.get("entropy") and entropy_scale != 1.0:
                curve.loc[:, ["mean", "stderr"]] *= entropy_scale
            if spec.get("magnitude"):
                curve["signed_mean"] = curve["mean"]
                curve["mean"] = curve["mean"].abs()
        if all(curve.empty for curve in curves.values()):
            raise ValueError(
                f"No valid distance-scaling rows were found in {entry['path']}."
            )
        data[key] = curves

    available = list(dict.fromkeys(name for key in keys for name in data[key]))
    is_fermionic = any(
        metric in available for metric in ("fmi", "fmn", "ftmi", "fgmn")
    )

    # A metric name belongs to exactly one party count -- `mi` only ever comes
    # from a k=2 file and `tmi` only from a k=3 one -- so the mapping is read
    # off the data rather than declared.
    k_of_metric: dict[str, int] = {}
    for k_of_file, size in keys:
        for metric in data[(k_of_file, size)]:
            previous = k_of_metric.setdefault(metric, k_of_file)
            if previous != k_of_file:
                raise ValueError(
                    f"Metric {metric!r} appears under both k={previous} and "
                    f"k={k_of_file}; the two cannot share a fit."
                )

    if metrics is not None:
        panel_metrics = tuple(metrics)
        missing = [name for name in panel_metrics if name not in available]
        if missing:
            raise ValueError(
                f"Requested metric(s) {missing} are not present; available: {available}."
            )
        # The panel grid is fixed, so a supporting measure has nowhere to be
        # drawn. Say so rather than dropping it silently -- it is still loaded
        # and still fitted, just not plotted.
        off_grid = [
            name for name in panel_metrics if name not in _PANEL_SLOT_OF_METRIC
        ]
        if off_grid:
            raise ValueError(
                f"Metric(s) {off_grid} have no panel; the grid holds "
                f"{sorted(_PANEL_SLOT_OF_METRIC)}. They are still returned in "
                "`data` and `fits`."
            )
    else:
        panel_metrics = tuple(
            metric
            for row in _PANEL_GRID
            for slot in row
            for k_of_slot in sorted(slot["metrics"])
            for metric in (slot["metrics"][k_of_slot],)
            if metric in available
        )
        if not panel_metrics:
            raise ValueError(
                "None of the panel measures were found; available: "
                f"{available}."
            )

    # Which grid slots earn a subplot, and which of the fixed rows and columns
    # therefore survive. Dropping an empty column is what turns a purely
    # ordinary-trace run into a one-column figure with mutual information above
    # negativity, rather than a 2x2 with two blank panels.
    panel_slots = {
        _PANEL_SLOT_OF_METRIC[metric]
        for metric in panel_metrics
        if metric in _PANEL_SLOT_OF_METRIC
    }
    if not panel_slots:
        raise ValueError(
            f"None of the requested metrics {list(panel_metrics)} has a panel; "
            "the panel grid holds mi/tmi, fmi/ftmi, mn/gmn and fmn/fgmn."
        )
    panel_rows = sorted({row for row, _ in panel_slots})
    panel_columns = sorted({column for _, column in panel_slots})

    # --- fit every loaded measure, draw only the panel ones ----------------
    fit_settings_range = _merge_per_metric(
        fit_range,
        {
            "mi": fit_range_mi,
            "fmi": fit_range_fmi,
            "mn": fit_range_mn,
            "fmn": fit_range_fmn,
        },
        available,
        None,
    )
    fit_settings_last = _merge_per_metric(
        fit_last,
        {
            "mi": fit_last_mi,
            "fmi": fit_last_fmi,
            "mn": fit_last_mn,
            "fmn": fit_last_fmn,
        },
        available,
        4,
    )

    fit_rows: list[dict[str, Any]] = []
    selected_points: dict[tuple[int, int, str], pd.DataFrame] = {}
    fitted_metrics = [
        metric
        for metric in available
        if metric not in _NON_SCALING_METRICS or metric in panel_metrics
    ]
    for k_of_file, size in keys:
        for metric in fitted_metrics:
            if metric not in data[(k_of_file, size)]:
                continue
            try:
                fit, selected = _distance_power_law_fit(
                    data[(k_of_file, size)][metric],
                    fit_range=_fit_range_for_size(
                        fit_settings_range[metric], size
                    ),
                    fit_last=fit_settings_last[metric],
                    min_relative_error=min_relative_error,
                )
                fit_rows.append(
                    {"k": k_of_file, "L": size, "metric": metric, **fit}
                )
                selected_points[(k_of_file, size, metric)] = selected
            except ValueError as exc:
                fit_rows.append(
                    {
                        "k": k_of_file,
                        "L": size,
                        "metric": metric,
                        "alpha": np.nan,
                        "alpha_stderr": np.nan,
                        "prefactor": np.nan,
                        "log_prefactor": np.nan,
                        "log_prefactor_stderr": np.nan,
                        "reduced_chi2": np.nan,
                        "d_min": np.nan,
                        "d_max": np.nan,
                        "points": 0,
                        "error": str(exc),
                    }
                )
                if metric in panel_metrics:
                    warnings.warn(
                        f"k={k_of_file}, L={size}, {metric.upper()} fit "
                        f"unavailable: {exc}"
                    )

    fits = pd.DataFrame(fit_rows)
    # A fit_size entry names system sizes, and the sizes on offer are the ones
    # measured at that metric's own party count.
    sizes_by_k: dict[int, list[int]] = {}
    for k_of_file, size in keys:
        sizes_by_k.setdefault(k_of_file, []).append(size)
    fit_sizes = {
        metric: _normalize_fit_sizes(
            _merge_per_metric(
                fit_size,
                {
                    "mi": fit_size_mi,
                    "fmi": fit_size_fmi,
                    "mn": fit_size_mn,
                    "fmn": fit_size_fmn,
                },
                panel_metrics,
                None,
            )[metric],
            sizes_by_k.get(k_of_metric[metric], unique_sizes),
            metric,
        )
        for metric in panel_metrics
    }

    colors = _color_map_by_size(unique_sizes, cmap)
    if figsize is None:
        figsize = (6.25 * len(panel_columns), 5.0 * len(panel_rows))
    fig, axis_grid = plt.subplots(
        len(panel_rows),
        len(panel_columns),
        figsize=figsize,
        dpi=dpi,
        constrained_layout=True,
    )
    axis_array = np.atleast_1d(axis_grid).reshape(
        len(panel_rows), len(panel_columns)
    )
    flat_axes = tuple(axis_array.flat)
    slot_axes = {
        (row, column): axis_array[row_index, column_index]
        for row_index, row in enumerate(panel_rows)
        for column_index, column in enumerate(panel_columns)
    }
    for slot, ax in slot_axes.items():
        if slot not in panel_slots:
            ax.set_visible(False)
    axes = {
        metric: slot_axes[_PANEL_SLOT_OF_METRIC[metric]]
        for metric in panel_metrics
        if metric in _PANEL_SLOT_OF_METRIC
    }

    for k_of_file, size in keys:
        color = colors[size]
        for metric in panel_metrics:
            ax = axes.get(metric)
            if ax is None or k_of_metric.get(metric) != k_of_file:
                continue
            curve = data[(k_of_file, size)].get(metric)
            if curve is None:
                continue
            positive = curve.loc[
                np.isfinite(curve["d"])
                & np.isfinite(curve["mean"])
                & (curve["d"] > 0.0)
                & (curve["mean"] > 0.0)
            ]
            omitted = len(curve) - len(positive)
            if omitted:
                warnings.warn(
                    f"k={k_of_file}, L={size}, {metric.upper()}: omitted "
                    f"{omitted} non-positive mean point(s) from the "
                    "logarithmic plot."
                )
            if positive.empty:
                continue

            x = positive["d"].to_numpy(dtype=float)
            y = positive["mean"].to_numpy(dtype=float)
            dy = positive["stderr"].to_numpy(dtype=float)
            dy = np.where(np.isfinite(dy), dy, 0.0)
            # The panel can hold both party counts, so the annotation has to
            # say which one each series is.
            label = rf"$k={k_of_file}$, $L={size}$"
            marker = _marker_for_k(k_of_file)
            if show_errorbars:
                _positive_log_errorbar(
                    ax,
                    x,
                    y,
                    dy,
                    fmt=marker,
                    markersize=4.0,
                    color=color,
                    ecolor=color,
                    markeredgecolor=color,
                    capsize=capsize,
                    elinewidth=0.9,
                    linestyle="none",
                    label=label,
                    alpha=0.95,
                )
            else:
                ax.plot(
                    x,
                    y,
                    linestyle="none",
                    marker=marker,
                    markersize=4.0,
                    color=color,
                    markeredgecolor=color,
                    label=label,
                    alpha=0.95,
                )

    joint_rows: list[dict[str, Any]] = []
    for metric, metric_fit_sizes in fit_sizes.items():
        if metric not in axes:
            continue
        exponent = _metric_spec(metric)["exponent"]
        metric_k = k_of_metric[metric]
        if joint_fit:
            usable = [
                size
                for size in metric_fit_sizes
                if (metric_k, size, metric) in selected_points
            ]
            missing = sorted(set(metric_fit_sizes) - set(usable))
            if missing:
                warnings.warn(
                    f"k={metric_k}, {metric.upper()} joint fit excludes "
                    f"L={missing}: no usable per-size fit window."
                )
            if not usable:
                continue
            pooled = pd.concat(
                [selected_points[(metric_k, size, metric)] for size in usable]
            ).sort_values("d")
            try:
                joint = _log_log_power_law_fit(
                    pooled, min_relative_error=min_relative_error
                )
            except ValueError as exc:
                warnings.warn(
                    f"k={metric_k}, {metric.upper()} joint fit unavailable: {exc}"
                )
                continue
            joint_rows.append(
                {"k": metric_k, "metric": metric, "sizes": tuple(usable), **joint}
            )
            if len(usable) == 1:
                prefix = rf"$k={metric_k}$, $L={usable[0]}$ fit: "
            else:
                sizes_text = ",".join(str(size) for size in usable)
                prefix = rf"$k={metric_k}$, $L\in\{{{sizes_text}\}}$ joint fit: "
            x_line = np.geomspace(joint["d_min"], joint["d_max"], 200)
            axes[metric].plot(
                x_line,
                joint["prefactor"] * x_line ** (-joint["alpha"]),
                color="black",
                linestyle=_linestyle_for_k(metric_k),
                linewidth=1.5,
                label=(
                    prefix
                    + rf"$\alpha_{metric_k}^{{{exponent}}}="
                    rf"{joint['alpha']:.3g}\pm {joint['alpha_stderr']:.2g}$"
                ),
                zorder=5,
            )
            continue

        for size in metric_fit_sizes:
            row = fits.loc[(fits["L"] == size) & (fits["metric"] == metric)]
            if row.empty or not np.isfinite(row.iloc[0]["alpha"]):
                continue
            row = row.iloc[0]
            selected = selected_points[(metric_k, size, metric)]
            x_line = np.geomspace(
                selected["d"].min(), selected["d"].max(), 200
            )
            axes[metric].plot(
                x_line,
                row["prefactor"] * x_line ** (-row["alpha"]),
                color=colors[size],
                linestyle=_linestyle_for_k(metric_k),
                linewidth=1.5,
                label=(
                    rf"$k={metric_k}$, $L={size}$ fit: "
                    rf"$\alpha_{metric_k}^{{{exponent}}}="
                    rf"{row['alpha']:.3g}\pm {row['alpha_stderr']:.2g}$"
                ),
                zorder=5,
            )

    joint_fits = pd.DataFrame(joint_rows) if joint_fit else None

    from matplotlib.ticker import FuncFormatter

    plain_distance = FuncFormatter(
        lambda value, _: f"{value:g}" if value > 0 else ""
    )
    for slot in panel_slots:
        ax = slot_axes[slot]
        descriptor = _PANEL_GRID[slot[0]][slot[1]]
        ax.set_xlabel(r"Effective chord distance $d$")
        # A panel can hold both party counts, so it is named by the quantity
        # it measures rather than by any one of them; the unit is the only
        # thing the party count does not change.
        label = str(descriptor["label"])
        if descriptor["entropy"]:
            label += f" [{mi_units}]"
        ax.set_ylabel(label)
        ax.set_xscale("log")
        # Plain-number formatting for logarithmic distance axis.
        ax.xaxis.set_major_formatter(plain_distance)
        ax.xaxis.set_minor_formatter(plain_distance)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.20)
        ax.legend(fontsize=8)

    if title is not None:
        fig.suptitle(title, fontsize=14)
    _show(fig, show)

    summary = pd.DataFrame(
        [
            {
                "k": key[0],
                "L": key[1],
                "file": str(entry["path"]),
                "circuit": circuit_name,
                "format": "aggregate" if entry["aggregated"] else "row-level",
                **{
                    f"distance_points_{metric}": len(data[key][metric])
                    for metric in panel_metrics
                    if metric in data[key]
                },
                **{
                    f"samples_{metric}": int(data[key][metric]["count"].sum())
                    for metric in panel_metrics
                    if metric in data[key]
                },
            }
            for key, entry in zip(keys, loaded)
        ]
    )
    if show_summary:
        try:
            from IPython.display import display

            display(summary)
            display(fits)
            if joint_fits is not None and not joint_fits.empty:
                display(joint_fits)
        except ImportError:
            print(summary.to_string(index=False))
            print(fits.to_string(index=False))
            if joint_fits is not None and not joint_fits.empty:
                print(joint_fits.to_string(index=False))

    return {
        "figure": fig,
        "axes": flat_axes,
        "axes_by_metric": axes,
        # `k` stays a plain number for the single-party-count case every caller
        # before mixed-k had; it is None when both are present, and
        # `k_values` is the general answer.
        "k": k_values[0] if len(k_values) == 1 else None,
        "k_values": tuple(k_values),
        "metrics": panel_metrics,
        "metric_party_counts": dict(k_of_metric),
        "available_metrics": tuple(available),
        "circuit_name": circuit_name,
        "is_fermionic": is_fermionic,
        "data": data,
        "summary": summary,
        "fits": fits,
        "joint_fit": joint_fit,
        "joint_fits": joint_fits,
        "fit_sizes": fit_sizes,
        "mi_units": mi_units,
        "selected_fit_points": selected_points,
    }

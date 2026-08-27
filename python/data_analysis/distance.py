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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .fitting import _weighted_linear_fit
from .records import (
    default_summary_path,
    is_record_file,
    is_summary_file,
    read_summary_csv,
    record_file_info,
    summarize,
)
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
    # Two-point correlators, mean of |.|^2 over records. They are amplitudes
    # squared rather than entropies, so no unit conversion applies.
    "g2": {"exponent": r"|G|^{2}"},
    "f2": {"exponent": r"|F|^{2}"},
    "fg2": {"exponent": r"|G^{f}|^{2}"},
    "ff2": {"exponent": r"|F^{f}|^{2}"},
    "min_bipneg": {"exponent": r"\mathrm{minN}"},
    "min_fbipneg": {"exponent": r"\mathrm{minN}^f"},
    # The two factors of the entanglement decomposition (see below). Derived,
    # not read: nothing in the CSV carries them directly.
    "mn_ent_fraction": {"exponent": r"P_{\mathrm{ent}}"},
    "fmn_ent_fraction": {"exponent": r"P^{f}_{\mathrm{ent}}"},
    "mn_ent_magnitude": {"exponent": r"\mathcal{N}_{\mathrm{ent}}"},
    "fmn_ent_magnitude": {"exponent": r"\mathcal{N}^{f}_{\mathrm{ent}}"},
    "gmn_ent_fraction": {"exponent": r"P^{(3)}_{\mathrm{ent}}"},
    "fgmn_ent_fraction": {"exponent": r"P^{(3)f}_{\mathrm{ent}}"},
    "gmn_ent_magnitude": {"exponent": r"\mathcal{G}_{\mathrm{ent}}"},
    "fgmn_ent_magnitude": {"exponent": r"\mathcal{G}^{f}_{\mathrm{ent}}"},
}

# ---------------------------------------------------------------------------
# Entanglement decomposition
#
# A bipartite negativity is exactly zero on most records -- monogamy plus
# entanglement sudden death -- so its mean over all records confounds two
# quantities that need not share an exponent:
#
#     <N>(d) = P_ent(d) * <N>_ent(d)
#
# the probability that a pair is entangled at all, and the typical magnitude
# when it is. `dist_scaling.exe` writes `<metric>_positive_count` beside the
# mean, which is what makes both recoverable from the aggregate alone.
# ---------------------------------------------------------------------------

_DECOMPOSITION_SUFFIXES = ("_ent_fraction", "_ent_magnitude")


def _is_decomposition_metric(metric: str) -> bool:
    return metric.endswith(_DECOMPOSITION_SUFFIXES)


def _decomposition_source(metric: str) -> str:
    """The measure a decomposition factor was derived from, or the metric."""
    for suffix in _DECOMPOSITION_SUFFIXES:
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric


# ---------------------------------------------------------------------------
# The published fit protocol (`paper_fit_ranges`)
#
# Section C of arXiv:2602.04969 does not fit every two-party measure over one
# window. The negativities are read off the four largest distances at every
# size, and the mutual informations are too at the sizes where the power law
# has not yet opened up a usable window; only the larger chains get a fitted
# range. Reproducing a published exponent means reproducing that, so the rule
# is spelled out here rather than left to a caller to hand-assemble.
#
# The paper names L=18 and L=20 for the tail-fitted mutual information and
# L>=22 for the ranged one, so the boundary sits between them; smaller chains
# have fewer distance points still and follow the tail rule.
# ---------------------------------------------------------------------------

_PAPER_TAIL_POINTS = 4
_PAPER_MI_TAIL_MAX_SIZE = 20


def _paper_tail_points(metric: str, size: int, k: int) -> int | None:
    """How many largest-distance points the paper's protocol fits, if any.

    ``None`` means "use the range this call was given". A decomposition factor
    follows its source measure, so the two factors and the mean they multiply
    to are always fitted over the same points.
    """
    if k != 2:
        return None
    stem = _decomposition_source(metric)
    if stem in ("mn", "fmn"):
        return _PAPER_TAIL_POINTS
    if stem in ("mi", "fmi") and size <= _PAPER_MI_TAIL_MAX_SIZE:
        return _PAPER_TAIL_POINTS
    return None

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

# The `ent_decomp=True` layout, a recreation of Fig. 8 of arXiv:2602.04969.
# Here the rows are the trace conventions and the columns the two factors of
# the decomposition, so reading a row left to right multiplies out to the
# negativity panel of the default figure.
_ENT_DECOMP_GRID: tuple[tuple[dict[str, Any], ...], ...] = (
    (
        {
            "label": r"Entangled fraction $P_{\mathrm{ent}}$",
            "entropy": False,
            "metrics": {2: "mn_ent_fraction", 3: "gmn_ent_fraction"},
        },
        {
            "label": r"Conditional negativity "
            r"$\langle\mathcal{N}\rangle_{\mathrm{ent}}$",
            "entropy": False,
            "metrics": {2: "mn_ent_magnitude", 3: "gmn_ent_magnitude"},
        },
    ),
    (
        {
            "label": r"Fermionic entangled fraction $P^{f}_{\mathrm{ent}}$",
            "entropy": False,
            "metrics": {2: "fmn_ent_fraction", 3: "fgmn_ent_fraction"},
        },
        {
            "label": r"Conditional fermionic negativity "
            r"$\langle\mathcal{N}^{f}\rangle_{\mathrm{ent}}$",
            "entropy": False,
            "metrics": {2: "fmn_ent_magnitude", 3: "fgmn_ent_magnitude"},
        },
    ),
)


# The `correlators=True` layout: the two fermionic two-point functions by
# column, trace convention by row. Only the bottom row is G and F proper --
# the top row comes from the ordinary trace, which drops the Jordan-Wigner
# string between the two modes, so it holds the spin correlators instead. The
# two rows coincide identically at separation 1, where the string is either
# empty or fixed by parity conservation.
#
# k=2 only: there is no three-party analogue of a two-point function.
_CORRELATOR_GRID: tuple[tuple[dict[str, Any], ...], ...] = (
    (
        {
            "label": r"Hopping $\overline{|\langle S^{+}_i S^{-}_j\rangle|^{2}}$",
            "entropy": False,
            "metrics": {2: "g2"},
        },
        {
            "label": r"Pairing $\overline{|\langle S^{-}_i S^{-}_j\rangle|^{2}}$",
            "entropy": False,
            "metrics": {2: "f2"},
        },
    ),
    (
        {
            "label": r"Fermionic hopping "
            r"$\overline{|\langle c^{\dagger}_i c_j\rangle|^{2}}$",
            "entropy": False,
            "metrics": {2: "fg2"},
        },
        {
            "label": r"Fermionic pairing "
            r"$\overline{|\langle c_i c_j\rangle|^{2}}$",
            "entropy": False,
            "metrics": {2: "ff2"},
        },
    ),
)


def _panel_slot_of_metric(
    grid: tuple[tuple[dict[str, Any], ...], ...],
) -> dict[str, tuple[int, int]]:
    """Every metric that has a panel in ``grid``, and the slot it belongs to."""
    return {
        metric: (row_index, column_index)
        for row_index, row in enumerate(grid)
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


def _decomposition_curves(
    frame: pd.DataFrame,
    distance: pd.Series,
    stem: str,
) -> dict[str, pd.DataFrame]:
    """Split ``<stem>_mean`` into an entangled fraction and a magnitude.

    Writing ``n`` for the records folded into the bin, ``m`` for those with a
    strictly positive value, and ``mu`` for the mean over all ``n``:

        P_ent = m / n            <N>_ent = n*mu / m = mu / P_ent

    The second identity is exact, not an approximation -- the zero records
    contribute nothing to the sum, so ``n*mu`` *is* the sum over the entangled
    ones. Both errors follow from the file's own ``m2 = stderr^2 * n * (n-1)``,
    and the population-level identity

        var(mu) = P_ent^2 var(<N>_ent) + <N>_ent^2 var(P_ent)

    says the split loses nothing: the mean's error bar is exactly the two
    factors' error bars recombined. (Both inherit the caveat the mean already
    carries -- every geometry is measured inside one trajectory, so records in
    a bin are correlated and every stderr here is optimistic.)
    """
    count_column = f"{stem}_positive_count"
    if count_column not in frame.columns or f"{stem}_mean" not in frame.columns:
        return {}

    n = pd.to_numeric(frame[f"{stem}_samples"], errors="coerce").to_numpy(dtype=float)
    m = pd.to_numeric(frame[count_column], errors="coerce").to_numpy(dtype=float)
    mu = pd.to_numeric(frame[f"{stem}_mean"], errors="coerce").to_numpy(dtype=float)
    stderr = pd.to_numeric(frame[f"{stem}_stderr"], errors="coerce").to_numpy(
        dtype=float
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(n > 0, m / n, np.nan)
        fraction_stderr = np.where(
            n > 0, np.sqrt(np.maximum(fraction * (1.0 - fraction), 0.0) / n), np.nan
        )

        magnitude = np.where(m > 0, mu / fraction, np.nan)
        # m2_Y = m2 + n*mu^2 - n^2*mu^2/m is the sum of squared deviations of
        # the entangled records about their own mean; the conditional standard
        # error is its usual sqrt(m2_Y / (m * (m - 1))).
        m2 = stderr**2 * n * (n - 1.0)
        m2_entangled = m2 + n * mu**2 - np.square(n) * mu**2 / m
        magnitude_stderr = np.where(
            m > 1,
            np.sqrt(np.maximum(m2_entangled, 0.0) / (m * (m - 1.0))),
            np.nan,
        )

    return {
        f"{stem}_ent_fraction": pd.DataFrame(
            {
                "d": distance,
                "mean": fraction,
                "stderr": fraction_stderr,
                "count": pd.Series(n, index=frame.index).fillna(0).astype(np.int64),
            }
        ),
        f"{stem}_ent_magnitude": pd.DataFrame(
            {
                "d": distance,
                "mean": magnitude,
                "stderr": magnitude_stderr,
                "count": pd.Series(m, index=frame.index).fillna(0).astype(np.int64),
            }
        ),
    }


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

    # Any measure that reports how often it was strictly positive can be split
    # into its two factors: the pair negativities always have, and the
    # GMN family since 2026-08-19. Older k=3 files carry only
    # `gmn_zero_prefilter_count` -- the zeros a vanishing bipartite negativity
    # settled without a solve, a lower bound on the zeros rather than their
    # count -- and so yield no decomposition.
    for column in frame.columns:
        if not column.strip().lower().endswith("_positive_count"):
            continue
        stem = column.strip()[: -len("_positive_count")]
        for name, curve in _decomposition_curves(frame, distance, stem).items():
            curve = curve.replace([np.inf, -np.inf], np.nan)
            curve = curve.loc[np.isfinite(curve["d"]) & (curve["d"] > 0.0)]
            data[name] = _pool_by_distance(curve)

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


def _records_sidecar(path: str | Path) -> Path | None:
    """The per-record binary beside a dist_scaling CSV, if one was written.

    ``dist_scaling.exe`` names it after the CSV's own stem, so a caller who
    named the CSV never has to know the binary exists.
    """
    candidate = Path(path).with_suffix("")
    candidate = candidate.with_name(candidate.name + "_records.bin")
    return candidate if candidate.exists() and is_record_file(candidate) else None


def _load_record_summary(path: str | Path) -> Any:
    """The summary of a record binary, preferring a pre-computed one.

    A production record file is far larger than the machine plotting it -- 2.7
    GB for a k=2 N=24 run at 243k trajectories -- so it is never held whole.
    Either a ``<stem>_summary.csv`` written by ``./summarize_records`` is read
    (instant), or the binary is streamed in bounded chunks (one sequential
    pass, tens of seconds). Both give the same numbers; the summary CSV just
    stops the pass happening again on every plot.
    """
    path = Path(path)
    if is_summary_file(path):
        return read_summary_csv(path)
    summary_path = default_summary_path(path)
    if summary_path.exists() and summary_path.stat().st_mtime >= path.stat().st_mtime:
        return read_summary_csv(summary_path)
    info = record_file_info(path)
    if info["record_count"] > 5_000_000:
        print(
            f"Streaming {info['record_count']:,} records from {path.name} "
            f"({path.stat().st_size / 1e9:.2f} GB). Run "
            f"`./summarize_records {path}` once to skip this next time."
        )
    return summarize(path)


# ---------------------------------------------------------------------------
# The `show_dist` supplemental figure
#
# A recreation of Fig. 9 of arXiv:2602.04969: what the *distribution* of a
# measure looks like at fixed size, over every record beyond some distance,
# rather than what its mean does with distance. Only the non-zero records are
# shown -- most triangles are not genuinely entangled at all, and a spike at
# zero holding 90% of the weight would leave nothing visible of the part that
# carries the physics. The zero fraction is stated in each panel instead, so
# nothing is hidden by the cut.
#
# "Non-zero" has to mean "above a tolerance", not "> 0". A vanishing mutual
# information comes back as +1e-16 and a vanishing TMI as -2e-16, so a literal
# `> 0` test keeps a cloud of floating-point noise sixteen decades below the
# physics and, on a shared logarithmic axis, squashes everything real into the
# right quarter of the figure. The default tolerance is the same 1e-10 the C++
# positivity counts use (MIPT_DIST_GMN_ZERO_TOL), which is measured to sit
# seven orders of magnitude clear of the solver's floor: a genuinely entangled
# record comes back at 1e-3 or above.
#
# This needs the individual values, so it needs the record binary -- but not
# all of them at once. The binary is reduced to per-geometry counts on a fixed
# logarithmic grid (20 bins per decade, five times finer than anything is
# drawn at), and the panel is a sum of those counts over the geometries at
# d >= d_min. That is what keeps a 2.7 GB file plottable on a small machine,
# and it is why the display bins are always groups of whole grid bins rather
# than edges fitted to the data.
# ---------------------------------------------------------------------------

_DISTRIBUTION_PANELS: tuple[tuple[str, int, str], ...] = (
    ("mi", 2, r"$\mathcal{I}_2$"),
    ("fmi", 2, r"$\mathcal{I}^{f}_2$"),
    ("gmn", 3, r"$\mathcal{N}_3$"),
    ("fgmn", 3, r"$\mathcal{N}^{f}_3$"),
)


def _normalize_show_dist(show_dist: Any) -> tuple[int, float] | None:
    if show_dist is None:
        return None
    try:
        size, d_min = show_dist
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "show_dist must be None or a (L, d_min) pair: the system size to "
            "draw and the smallest effective chord distance to include."
        ) from exc
    return int(size), float(d_min)


def _distribution_values(
    summary: Any,
    metric: str,
    d_min: float,
    entropy_scale: float,
    zero_tol: float,
) -> dict[str, Any] | None:
    """One panel's binned records, and what was cut to get them.

    ``entropy_scale`` rescales the bin *edges* rather than any value, which is
    exactly equivalent for a change of entropy unit: a constant factor slides
    a logarithmic histogram along its axis without redistributing anything.
    """
    panel = summary.distribution(metric, d_min=d_min, zero_tol=zero_tol)
    if panel is None or panel["positive"] == 0:
        return None
    edges = panel["edges"]
    if _metric_spec(metric).get("entropy") and entropy_scale != 1.0:
        edges = edges * entropy_scale
    return {
        "edges": edges,
        "counts": panel["counts"],
        "above": panel["above"],
        "records": panel["records"],
        "positive": panel["positive"],
        "d_min": d_min,
        "zero_tol": zero_tol,
    }


def _draw_value_distributions(
    panels: dict[str, dict[str, Any]],
    *,
    size: int,
    d_min: float,
    bins: int,
    mi_units: str,
    circuit_name: str,
    zero_tol: float,
    dpi: int,
    figsize: tuple[float, float] | None,
) -> Any:
    """The stacked histogram: one measure per row, flush, on a shared x axis."""
    order = [metric for metric, _, _ in _DISTRIBUTION_PANELS if metric in panels]
    if figsize is None:
        figsize = (6.5, 2.05 * len(order) + 0.9)
    # constrained_layout would override the zero spacing that makes the panel
    # edges touch, so the margins are set by hand instead.
    fig, axis_grid = plt.subplots(
        len(order), 1, figsize=figsize, dpi=dpi, sharex=True
    )
    axes = np.atleast_1d(axis_grid)
    fig.subplots_adjust(hspace=0.0, left=0.14, right=0.97, top=0.94, bottom=0.11)

    # The panels share an x axis, so a bar at a given position has to mean the
    # same interval in every one of them. The counts arrive on the summary's
    # fixed logarithmic grid; the display bins are contiguous *groups* of grid
    # bins, so rebinning is a reshape-and-sum with no interpolation anywhere.
    mask = np.zeros_like(panels[order[0]]["counts"], dtype=bool)
    for metric in order:
        mask |= (panels[metric]["counts"] > 0) & panels[metric]["above"]
    occupied = np.flatnonzero(mask)
    if occupied.size == 0:
        raise ValueError(
            f"No records above {zero_tol:g} survive d >= {d_min} at L={size}."
        )
    first, last = int(occupied[0]), int(occupied[-1]) + 1
    group = max(1, int(round((last - first) / max(1, bins))))
    # Extend the window so it divides evenly into groups.
    last = first + int(np.ceil((last - first) / group)) * group

    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.8, len(order)))
    labels = {metric: label for metric, _, label in _DISTRIBUTION_PANELS}
    party = {metric: k for metric, k, _ in _DISTRIBUTION_PANELS}
    for index, (metric, ax) in enumerate(zip(order, axes)):
        panel = panels[metric]
        counts = np.where(panel["above"], panel["counts"], 0)[first:last]
        edges = panel["edges"][first : last + 1 : group]
        binned = counts.reshape(-1, group).sum(axis=1).astype(float)
        # Each panel integrates to one over its own non-zero records, so rows
        # measuring different things stay comparable in shape.
        total = binned.sum()
        ax.stairs(
            binned / total if total > 0 else binned,
            edges,
            fill=True,
            color=colors[index],
            edgecolor="0.25",
            linewidth=0.4,
        )
        ax.set_xscale("log")
        unit = f" [{mi_units}]" if _metric_spec(metric).get("entropy") else ""
        # The non-zero share rather than the zero share: a genuinely
        # multipartite-entangled triangle is rare enough that the zero
        # fraction rounds to 100% and reads as "all of them", right beside a
        # panel drawn from thousands of records that are not.
        share = 100.0 * panel["positive"] / panel["records"]
        # Upper left: the distributions run to the right of the axis, so this
        # is the corner that stays clear as a tail grows.
        ax.text(
            0.022,
            0.88,
            rf"{labels[metric]}  ($k={party[metric]}$){unit}"
            + "\n"
            + rf"$n_{{>0}}={panel['positive']:,}$ of {panel['records']:,}"
            + rf"  (${share:.3g}\%$ non-zero)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.8},
        )
        ax.set_ylabel("fraction")
        ax.grid(True, which="both", axis="x", alpha=0.18)
        # Flush edges mean a panel's top tick label sits on top of the one
        # below it and its bottom label under the one above, so every panel
        # but the last drops both ends; the last keeps its zero, which has
        # nothing beneath it to collide with.
        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=4, prune="upper" if index == len(order) - 1 else "both")
        )
        ax.margins(y=0.0)

    axes[-1].set_xlabel(f"Value  (records at or below {zero_tol:g} counted as zero)")
    axes[0].set_title(
        f"{circuit_name}, $L={size}$, records with $d \geq {d_min:g}$",
        fontsize=11,
    )
    return fig


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _normalize_fit_range(
    fit_range: tuple[float | None, float | None] | None,
    name: str,
) -> tuple[float | None, float | None] | None:
    """Validate one ``(d_min, d_max)`` fit window, either bound optional."""
    if fit_range is None:
        return None
    try:
        lower, upper = fit_range
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a (d_min, d_max) pair, either bound None; "
            f"got {fit_range!r}."
        ) from None
    lower = None if lower is None else float(lower)
    upper = None if upper is None else float(upper)
    if lower is not None and lower <= 0.0:
        raise ValueError(
            f"{name} lower bound must be positive; the distance axis is "
            "logarithmic."
        )
    if lower is not None and upper is not None and upper <= lower:
        raise ValueError(
            f"{name}=({lower}, {upper}) is empty; the upper bound must exceed "
            "the lower one."
        )
    return (lower, upper)


def _fit_range_text(
    fit_range: tuple[float | None, float | None] | None,
) -> str:
    if fit_range is None:
        return "every positive point"
    lower, upper = fit_range
    return (
        f"d in [{'0' if lower is None else format(lower, 'g')}, "
        f"{'inf' if upper is None else format(upper, 'g')}]"
    )


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
    tail_points: int | None = None,
    min_relative_error: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit mean = prefactor * d**(-alpha) in logarithmic coordinates.

    ``tail_points`` selects that many largest-distance points and **replaces**
    ``fit_range`` rather than narrowing it -- the published protocol asks for
    the four largest distances outright, not for the four largest inside some
    other window.
    """
    selected = curve.loc[
        np.isfinite(curve["d"])
        & np.isfinite(curve["mean"])
        & (curve["d"] > 0.0)
        & (curve["mean"] > 0.0)
    ].copy()

    if tail_points is not None:
        selected = selected.sort_values("d", kind="stable").tail(tail_points)
    elif fit_range is not None:
        lower, upper = fit_range
        if lower is not None:
            selected = selected.loc[selected["d"] >= lower]
        if upper is not None:
            selected = selected.loc[selected["d"] <= upper]

    fit = _log_log_power_law_fit(
        selected, min_relative_error=min_relative_error
    )
    return fit, selected


def _print_distance_fits(
    fits: pd.DataFrame,
    metrics: Sequence[str],
    k_of_metric: Mapping[str, int],
    fit_ranges: Mapping[int, tuple[float | None, float | None] | None],
    plotted: Mapping[str, int],
    circuit_name: str,
    paper_fit_ranges: bool = False,
) -> None:
    """Report every fit, including the sizes the figure does not draw.

    Only the largest size gets a line in the panel -- the smaller ones lie
    almost on top of it -- so the console is where the size dependence of an
    exponent is actually read.
    """
    if fits.empty:
        return
    print(f"Distance-scaling power-law fits ({circuit_name}):")
    party_counts = sorted(
        {int(k_of_metric[metric]) for metric in metrics if metric in k_of_metric}
    )
    for k in party_counts:
        header = f"  k={k}, fit window: {_fit_range_text(fit_ranges.get(k))}"
        if paper_fit_ranges and k == 2:
            header += (
                f"; paper_fit_ranges overrides the rows marked "
                f"[tail {_PAPER_TAIL_POINTS}]"
            )
        print(header)
        for metric in metrics:
            if k_of_metric.get(metric) != k:
                continue
            rows = fits.loc[fits["metric"] == metric].sort_values("L")
            name = f"alpha_{k}^{metric.upper()}"
            for _, row in rows.iterrows():
                head = f"    L={int(row['L']):<3d} {name:<20s}"
                if not np.isfinite(row["alpha"]):
                    detail = (
                        row["error"]
                        if "error" in row.index and isinstance(row["error"], str)
                        else "no usable fit window"
                    )
                    print(f"{head} unavailable: {detail}")
                    continue
                window = f"[{row['d_min']:.4g}, {row['d_max']:.4g}]"
                selection = str(row.get("fit_selection", "range"))
                marker = (
                    f"  [tail {selection.split(':')[1]}]"
                    if selection.startswith("tail:")
                    else ""
                )
                print(
                    f"{head} = {row['alpha']:7.4f} +/- {row['alpha_stderr']:.4f}"
                    f"   d in {window:<18s}"
                    f"{int(row['points']):3d} pts"
                    f"   chi2/dof = {row['reduced_chi2']:6.2f}"
                    + marker
                    + ("   <- plotted" if plotted.get(metric) == int(row["L"]) else "")
                )


def dist_scaling(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    sizes: Sequence[int] | None = None,
    l_by_file: Mapping[str, int] | None = None,
    metrics: Sequence[str] | None = None,
    ent_decomp: bool = False,
    correlators: bool = False,
    show_dist: tuple[int, float] | None = None,
    dist_bins: int = 40,
    dist_zero_tol: float = 1.0e-10,
    fit_range_2: tuple[float | None, float | None] | None = None,
    fit_range_3: tuple[float | None, float | None] | None = None,
    paper_fit_ranges: bool = False,
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

    Accepts any output format of ``dist_scaling.exe``. The aggregated CSV has
    one row per geometry with ``<metric>_mean``/``_stderr``/``_samples``
    columns; the older row-level CSV has one ``d,mi,mn`` (or
    ``d,mi,fmi,mn,fmn``) row per pair per trajectory and is streamed in
    ``chunksize`` blocks; the **per-record binary** ``<stem>_records.bin`` has
    one fixed-width record per (trajectory, geometry, embedding); and the
    **summary CSV** ``<stem>_summary.csv`` that ``./summarize_records``
    reduces such a binary to. All four end up as the same curves. Files of
    every kind may be mixed in one call as long as they share a circuit;
    **party counts may be mixed too**, which is what a ``k=0`` run of
    ``dist_scaling.exe`` produces -- a pair file and a triangle file measured
    on the same trajectories.

    A record binary is **never held whole**: a k=2 run at N=24 writes 67
    million records, 2.7 GB on disk and 4.3 GB more as a frame, which is an
    out-of-memory kill rather than a slow read on an ordinary machine. It is
    streamed in bounded chunks instead, and if a ``<stem>_summary.csv`` sits
    beside it and is no older, that is read instead so the pass happens once.
    Run ``./summarize_records <path>`` to make one. The reduction is lossless
    for the curves -- means and standard errors come back from summed values
    and squares, not from bins -- so a summary and its binary give identical
    exponents.

    A record file carries only :math:`I_k` and :math:`N_k`, so a run given
    that way has no ``g2``/``f2`` and no purities; pass the CSV for those. It
    does carry the individual values, so its negativity decompositions come
    from the records themselves rather than from a ``_positive_count`` column,
    using the same per-metric threshold ``dist_scaling.exe`` does (1e-12 for
    the pair negativities, 1e-10 for the GMN family) so the two agree
    exactly. A record binary beside a named CSV is located but not read
    unless ``show_dist`` asks for it.

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

    Within a panel the two annotations are orthogonal and each says one
    thing: **colour is the system size** and **marker shape is the party
    count** (``o`` for ``k=2``, ``s`` for ``k=3``, and the fit line style to
    match). The legend therefore holds one entry per size and one per party
    count instead of one per series. The fermionic column is dropped for qubit
    ensembles, leaving mutual information above negativity.

    Every other measure the file carries -- ``average_mi``, ``min_bipneg``,
    the purities -- is still loaded into ``data`` and still fitted into
    ``fits``; it just gets no subplot. Pass ``metrics`` to override the panel
    set.

    ``correlators=True`` replaces the whole figure with the two fermionic
    two-point functions: columns are the hopping correlator
    :math:`G_{ij} = \langle c^\dagger_i c_j\rangle` and the pairing
    correlator :math:`F_{ij} = \langle c_i c_j\rangle`, rows are the trace
    convention. What is plotted is
    :math:`\overline{|G_{ij}|^2}` -- the record-wise square, averaged -- not
    :math:`|\overline{G_{ij}}|^2`, whose phases average away. Only the bottom
    row is G and F proper; the top row comes from the ordinary trace, which
    drops the Jordan-Wigner string between the two modes. k=2 only.

    ``ent_decomp=True`` replaces the whole figure with the **entanglement
    decomposition**, a recreation of Fig. 8 of arXiv:2602.04969. A bipartite
    negativity is exactly zero on most records, so its mean confounds two
    quantities that need not share an exponent:

    .. math::

        \langle\mathcal{N}\rangle(d)
            = P_{\mathrm{ent}}(d)\,
              \langle\mathcal{N}\rangle_{\mathrm{ent}}(d)

    -- how often a pair is entangled at all, and how much when it is. The
    columns are those two factors and the rows the two trace conventions, so
    the top row multiplies out to the Negativity panel of the default figure
    and the bottom row to the Fermionic Negativity one:

    =============== ============================ =============================
    row              left column                  right column
    =============== ============================ =============================
    ordinary trace  ``mn_ent_fraction`` (k=2)    ``mn_ent_magnitude`` (k=2)
                    ``gmn_ent_fraction`` (k=3)   ``gmn_ent_magnitude`` (k=3)
    fermionic trace ``fmn_ent_fraction`` (k=2)   ``fmn_ent_magnitude`` (k=2)
                    ``fgmn_ent_fraction`` (k=3)  ``fgmn_ent_magnitude`` (k=3)
    =============== ============================ =============================

    Both factors are derived from the ``<metric>_positive_count`` column, so
    this needs the **aggregated** format. The row-level archive carries no such
    count at all, and k=3 files written before 2026-08-19 carry only the zeros
    the GMN prefilter settled, a lower bound on the zeros rather than their
    count; either way those files contribute no panel and are reported. The
    derived curves are loaded into ``data`` whether or not ``ent_decomp`` is
    set; they are only fitted when they are on a panel. ``metrics`` cannot be
    combined with ``ent_decomp``.

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

    ``fit_range_2`` and ``fit_range_3`` are the ``(d_min, d_max)`` windows
    the power laws are fitted over, one per party count and applied to every
    measure at that count; either bound may be ``None`` for unbounded, and a
    range left ``None`` fits every positive point. **Every** size is fitted,
    every fit is printed, and all of them are returned in ``fits``; only the
    largest size per measure gets a line in the panel, since the smaller ones
    lie nearly on top of it and say nothing the printout does not.

    ``paper_fit_ranges=True`` overrides those windows where Section C of
    arXiv:2602.04969 uses a tail fit instead, which is what reproducing a
    published two-party exponent needs:

    ===================== ================= ==================================
    measure               sizes             window
    ===================== ================= ==================================
    ``mn``, ``fmn``       every ``L``       the 4 largest distances
    ``mi``, ``fmi``       ``L <= 20``       the 4 largest distances
    ``mi``, ``fmi``       ``L >= 22``       ``fit_range_2``
    everything at k=3     every ``L``       ``fit_range_3``
    ===================== ================= ==================================

    The paper names L=18 and L=20 for the tail-fitted mutual information and
    L>=22 for the ranged one, so the boundary sits between them and smaller
    chains -- which have fewer distance points still -- follow the tail rule.
    A tail **replaces** the range rather than narrowing it. The decomposition
    factors follow their source measure, so ``ent_decomp`` panels stay
    consistent with the negativity they multiply to. Every fit records which
    rule produced it in the ``fit_selection`` column of ``fits`` and in the
    printed report.

    ``show_dist`` draws a **supplemental** second figure, a recreation of
    Fig. 9 of arXiv:2602.04969: the distribution of the non-zero records of
    ``mi``, ``fmi``, ``gmn`` and ``fgmn``, one measure per row, stacked flush
    on a shared logarithmic x axis. It takes ``(L, d_min)`` --
    the system size to draw and the smallest effective chord distance to
    include -- or ``None`` (the default) for no such figure. Because a
    distribution cannot be recovered from a mean, this reads the **record
    binaries**: ``L`` must be a size for which one exists, either passed
    directly or sitting beside a passed CSV. Both party counts are needed for
    all four rows, which is what a ``k=0`` run writes; a missing row is warned
    about rather than fatal. The zeros are excluded because they are the
    majority and would leave nothing else visible; each panel states what
    fraction was cut. ``dist_bins`` sets the number of logarithmic bins. The
    figure is returned as ``dist_figure`` and the binned counts behind it as
    ``dist_panels``. The counts come from the summary's fixed logarithmic
    grid -- 20 bins per decade, five times finer than the figure is drawn at
    -- so ``dist_bins`` sets how many *groups* of grid bins a bar spans and
    the edges are always real bin boundaries rather than a fit to the data.

    "Non-zero" means above ``dist_zero_tol`` (default ``1e-10``, the same
    ``MIPT_DIST_GMN_ZERO_TOL`` the C++ positivity counts use), not literally
    ``> 0``. A vanishing mutual information comes back as ``+1e-16``, so a
    literal test would keep a cloud of floating-point noise sixteen decades
    below the physics and squash everything real into the right quarter of a
    shared logarithmic axis. A ``NaN`` -- a GMN the schedule never requested
    -- is neither a value nor a zero and is left out of both counts.

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
    if ent_decomp and metrics is not None:
        raise ValueError(
            "ent_decomp=True fixes the panel set to the two decomposition "
            "factors; metrics= cannot also be given."
        )
    if correlators and (metrics is not None or ent_decomp):
        raise ValueError(
            "correlators=True fixes the panel set to the two-point functions; "
            "metrics= and ent_decomp= cannot also be given."
        )
    if correlators:
        panel_grid = _CORRELATOR_GRID
    elif ent_decomp:
        panel_grid = _ENT_DECOMP_GRID
    else:
        panel_grid = _PANEL_GRID
    slot_of_metric = _panel_slot_of_metric(panel_grid)
    fit_ranges = {
        2: _normalize_fit_range(fit_range_2, "fit_range_2"),
        3: _normalize_fit_range(fit_range_3, "fit_range_3"),
    }

    def resolve_selection(metric: str, size: int, k: int) -> int | None:
        return _paper_tail_points(metric, size, k) if paper_fit_ranges else None
    mi_units, mi_scale_from_nats = _mi_unit_spec(mi_units)
    _, legacy_scale_from_nats = _mi_unit_spec(legacy_entropy_units)

    wanted_distribution = _normalize_show_dist(show_dist)
    if dist_bins < 2:
        raise ValueError("dist_bins must be at least 2.")
    if dist_zero_tol < 0.0:
        raise ValueError("dist_zero_tol must be non-negative.")

    paths = _resolve_files(files, file_glob)

    # --- read every file, then reconcile what they say about themselves -----
    #
    # Three input formats reach the same `{metric: DataFrame[d, mean, stderr,
    # count]}` structure: the aggregated CSV, the older row-level CSV, and the
    # per-record binary. A record file is reduced here to the same curves the
    # CSV states outright, so nothing downstream has to know which it came
    # from -- but it carries only I_k and N_k, so a run given as a record file
    # has no g2/f2 and no purities.
    loaded: list[dict[str, Any]] = []
    for path in paths:
        entry: dict[str, Any] = {"path": path}
        if is_record_file(path) or is_summary_file(path):
            summary = _load_record_summary(path)
            truncated = summary.info.get("truncated", 0)
            if truncated:
                warnings.warn(
                    f"{path}: {truncated} trailing byte(s) do not form a whole "
                    "record and were ignored; the run was probably killed "
                    "mid-write."
                )
            entry["format"] = "summary" if is_summary_file(path) else "records"
            entry["aggregated"] = True
            entry["data"] = summary.curves()
            entry["k"] = summary.k
            entry["stored_units"] = summary.info.get("entropy_units", "bits")
            entry["size"] = summary.L
            circuit_type = summary.info.get("circuit_type")
            entry["circuit"] = (
                CIRCUIT_DISPLAY_NAMES.get(int(circuit_type))
                if circuit_type is not None
                else _parse_circuit_name(path)
            )
            entry["record_summary"] = summary
            entry["records_path"] = Path(path)
            loaded.append(entry)
            continue

        aggregated = _is_aggregate_file(path)
        entry["aggregated"] = aggregated
        entry["format"] = "aggregate" if aggregated else "row-level"
        # The sidecar is located but not read: it is only needed for
        # `show_dist`, and it is far larger than the CSV beside it.
        entry["records_path"] = _records_sidecar(path)
        entry["record_summary"] = None
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
    if paper_fit_ranges and 2 not in k_values:
        warnings.warn(
            "paper_fit_ranges=True only overrides two-party measures, and no "
            f"k=2 file was given (found k={k_values}); every fit uses "
            "fit_range_3."
        )

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
        print(f"Importing k={k_of_file}, L={size} [{entry['format']}]: {entry['path']}")
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
        off_grid = [name for name in panel_metrics if name not in slot_of_metric]
        if off_grid:
            raise ValueError(
                f"Metric(s) {off_grid} have no panel; the grid holds "
                f"{sorted(slot_of_metric)}. They are still returned in "
                "`data` and `fits`."
            )
    else:
        panel_metrics = tuple(
            metric
            for row in panel_grid
            for slot in row
            for k_of_slot in sorted(slot["metrics"])
            for metric in (slot["metrics"][k_of_slot],)
            if metric in available
        )
        if not panel_metrics:
            if ent_decomp:
                raise ValueError(
                    "ent_decomp=True needs a negativity that reports how often "
                    "it was positive -- a `<metric>_positive_count` column, "
                    "which the row-level format never wrote and which k=3 "
                    "files only gained on 2026-08-19. "
                    f"Available here: {available}."
                )
            if correlators and any(
                entry["format"] == "records" for entry in loaded
            ):
                raise ValueError(
                    "correlators=True needs g2/f2 (and fg2/ff2), which the "
                    "per-record binary does not carry -- a record holds only "
                    "I_k and N_k. Pass the CSV instead. "
                    f"Available here: {available}."
                )
            raise ValueError(
                "None of the panel measures were found; available: "
                f"{available}."
            )
        if ent_decomp:
            silent = sorted(
                {k_of_file for k_of_file, _ in keys}
                - {k_of_metric[metric] for metric in panel_metrics}
            )
            if silent:
                warnings.warn(
                    f"ent_decomp=True: the k={silent} file(s) carry no "
                    "positive-count column -- k=3 files written before "
                    "2026-08-19 do not -- so they are neither drawn nor "
                    "fitted."
                )

    # Which grid slots earn a subplot, and which of the fixed rows and columns
    # therefore survive. Dropping an empty column is what turns a purely
    # ordinary-trace run into a one-column figure with mutual information above
    # negativity, rather than a 2x2 with two blank panels.
    panel_slots = {
        slot_of_metric[metric]
        for metric in panel_metrics
        if metric in slot_of_metric
    }
    if not panel_slots:
        raise ValueError(
            f"None of the requested metrics {list(panel_metrics)} has a panel; "
            f"the panel grid holds {sorted(slot_of_metric)}."
        )
    panel_rows = sorted({row for row, _ in panel_slots})
    panel_columns = sorted({column for _, column in panel_slots})

    # --- fit every loaded measure, draw only the panel ones ----------------
    fit_rows: list[dict[str, Any]] = []
    selected_points: dict[tuple[int, int, str], pd.DataFrame] = {}
    # A purity is constant with distance and the decomposition factors are
    # only wanted when they are the subject of the figure, so neither is fitted
    # unless it earns a panel. In `ent_decomp` mode a party count with no panel
    # is dropped whole: a k=3 file passed along with the pair files would
    # otherwise contribute sixteen fits to a figure it does not appear in.
    panel_party_counts = {k_of_metric[metric] for metric in panel_metrics}
    fitted_metrics = [
        metric
        for metric in available
        if (not ent_decomp or k_of_metric[metric] in panel_party_counts)
        and (
            metric in panel_metrics
            or not (
                metric in _NON_SCALING_METRICS or _is_decomposition_metric(metric)
            )
        )
    ]
    for k_of_file, size in keys:
        for metric in fitted_metrics:
            if metric not in data[(k_of_file, size)]:
                continue
            tail_points = resolve_selection(metric, size, k_of_file)
            selection = (
                f"tail:{tail_points}" if tail_points is not None else "range"
            )
            try:
                fit, selected = _distance_power_law_fit(
                    data[(k_of_file, size)][metric],
                    fit_range=fit_ranges.get(k_of_file),
                    tail_points=tail_points,
                    min_relative_error=min_relative_error,
                )
                fit_rows.append(
                    {
                        "k": k_of_file,
                        "L": size,
                        "metric": metric,
                        "fit_selection": selection,
                        **fit,
                    }
                )
                selected_points[(k_of_file, size, metric)] = selected
            except ValueError as exc:
                fit_rows.append(
                    {
                        "k": k_of_file,
                        "L": size,
                        "metric": metric,
                        "fit_selection": selection,
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

    # One line per panel, drawn for the largest size that produced a usable
    # fit at that measure's own party count. Every other size is still fitted
    # and reported -- the panel just stops carrying a stack of nearly parallel
    # dashed lines that no reader can tell apart.
    plotted_fit_size: dict[str, int] = {}
    for metric in panel_metrics:
        usable = [
            size
            for k_of_file, size in keys
            if k_of_file == k_of_metric[metric]
            and (k_of_file, size, metric) in selected_points
        ]
        if usable:
            plotted_fit_size[metric] = max(usable)
    if show_summary:
        _print_distance_fits(
            fits,
            fitted_metrics,
            k_of_metric,
            fit_ranges,
            plotted_fit_size,
            circuit_name,
            paper_fit_ranges,
        )

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
        metric: slot_axes[slot_of_metric[metric]]
        for metric in panel_metrics
        if metric in slot_of_metric
    }

    # Colour and marker are the whole legend, so each panel only needs to
    # know which sizes and which party counts actually reached it.
    drawn_sizes: dict[tuple[int, int], set[int]] = {}
    drawn_party_counts: dict[tuple[int, int], set[int]] = {}
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
            # The series carries no legend entry of its own: colour already
            # names the size and the marker already names the party count, and
            # one entry per (k, L) pair says both twice.
            slot = slot_of_metric[metric]
            drawn_sizes.setdefault(slot, set()).add(size)
            drawn_party_counts.setdefault(slot, set()).add(k_of_file)
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
                    label="_nolegend_",
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
                    label="_nolegend_",
                    alpha=0.95,
                )

    # The exponent's own subscript names the party count, so the fit needs no
    # `k=..., L=...` prefix; its colour is the size it belongs to and its line
    # style matches that party count's marker.
    for metric, size in plotted_fit_size.items():
        ax = axes.get(metric)
        if ax is None:
            continue
        metric_k = k_of_metric[metric]
        row = fits.loc[
            (fits["k"] == metric_k)
            & (fits["L"] == size)
            & (fits["metric"] == metric)
        ]
        if row.empty or not np.isfinite(row.iloc[0]["alpha"]):
            continue
        row = row.iloc[0]
        exponent = _metric_spec(metric)["exponent"]
        selected = selected_points[(metric_k, size, metric)]
        x_line = np.geomspace(selected["d"].min(), selected["d"].max(), 200)
        ax.plot(
            x_line,
            row["prefactor"] * x_line ** (-row["alpha"]),
            color=colors[size],
            linestyle=_linestyle_for_k(metric_k),
            linewidth=1.5,
            label=(
                rf"$\alpha_{metric_k}^{{{exponent}}}="
                rf"{row['alpha']:.3g}\pm {row['alpha_stderr']:.2g}$"
            ),
            zorder=5,
        )

    from matplotlib.ticker import FuncFormatter

    plain_distance = FuncFormatter(
        lambda value, _: f"{value:g}" if value > 0 else ""
    )
    for slot in panel_slots:
        ax = slot_axes[slot]
        descriptor = panel_grid[slot[0]][slot[1]]
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
        # Two orthogonal keys plus the fits, rather than one entry per series.
        # The size swatches carry no marker at all so that shape is left to
        # mean only the party count.
        handles: list[Any] = [
            Patch(facecolor=colors[size], edgecolor="none", label=rf"$L={size}$")
            for size in sorted(drawn_sizes.get(slot, ()))
        ]
        handles += [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=_marker_for_k(k_of_slot),
                color="0.35",
                markersize=4.5,
                label=rf"$k={k_of_slot}$",
            )
            for k_of_slot in sorted(drawn_party_counts.get(slot, ()))
        ]
        # Only the fit lines were given a real label, so this returns them and
        # nothing else.
        handles += ax.get_legend_handles_labels()[0]
        if handles:
            ax.legend(handles=handles, fontsize=8)

    if title is not None:
        fig.suptitle(title, fontsize=14)
    _show(fig, show)

    # --- the supplemental distribution figure ------------------------------
    #
    # Drawn from the record binaries rather than from anything above: the
    # aggregate keeps means, and a distribution cannot be recovered from a
    # mean. A file passed as a CSV contributes through its sidecar.
    distribution_figure = None
    distribution_panels: dict[str, dict[str, Any]] = {}
    if wanted_distribution is not None:
        dist_size, dist_d_min = wanted_distribution
        bundles: dict[int, Any] = {}
        for key, entry in zip(keys, loaded):
            k_of_file, size = key
            if size != dist_size or entry.get("records_path") is None:
                continue
            summary = entry.get("record_summary")
            if summary is None:
                summary = _load_record_summary(entry["records_path"])
                entry["record_summary"] = summary
            bundles[k_of_file] = summary

        if not bundles:
            available_sizes = sorted(
                {
                    size
                    for (_, size), entry in zip(keys, loaded)
                    if entry.get("records_path") is not None
                }
            )
            raise ValueError(
                f"show_dist=({dist_size}, {dist_d_min}) needs per-record output "
                f"for L={dist_size}, and none of the given files has a record "
                "binary at that size. dist_scaling.exe writes "
                "`<stem>_records.bin` beside its CSV unless MIPT_DIST_RECORDS=0. "
                + (
                    f"Record files were found at L={available_sizes}."
                    if available_sizes
                    else "No record file was found beside any given file."
                )
            )

        for metric, k_of_panel, _ in _DISTRIBUTION_PANELS:
            bundle = bundles.get(k_of_panel)
            if bundle is None:
                continue
            scale = mi_scale_from_nats / _mi_unit_spec(
                bundle.info.get("entropy_units") or "bits"
            )[1]
            panel = _distribution_values(
                bundle, metric, dist_d_min, scale, dist_zero_tol
            )
            if panel is not None and panel["positive"] > 0:
                distribution_panels[metric] = panel

        missing = [
            metric
            for metric, _, _ in _DISTRIBUTION_PANELS
            if metric not in distribution_panels
        ]
        if missing:
            warnings.warn(
                f"show_dist: no records above {dist_zero_tol:g} for {missing} at "
                f"L={dist_size}, d >= {dist_d_min}; those rows are omitted. A qubit "
                "ensemble has no fermionic measures, and the k=3 rows need a "
                "triangle record file at that size."
            )
        if not distribution_panels:
            raise ValueError(
                f"show_dist: nothing to draw at L={dist_size}, d >= {dist_d_min}."
            )
        distribution_figure = _draw_value_distributions(
            distribution_panels,
            size=dist_size,
            d_min=dist_d_min,
            bins=dist_bins,
            mi_units=mi_units,
            circuit_name=circuit_name,
            zero_tol=dist_zero_tol,
            dpi=dpi,
            figsize=None,
        )
        _show(distribution_figure, show)

    summary = pd.DataFrame(
        [
            {
                "k": key[0],
                "L": key[1],
                "file": str(entry["path"]),
                "circuit": circuit_name,
                "format": entry["format"],
                "records": (
                    str(entry["records_path"])
                    if entry.get("records_path") is not None
                    else ""
                ),
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
        except ImportError:
            print(summary.to_string(index=False))
            print(fits.to_string(index=False))

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
        "ent_decomp": ent_decomp,
        "correlators": correlators,
        # The supplemental Fig.-9-style figure and the positive records behind
        # it; both None/empty unless `show_dist` asked for them.
        "dist_figure": distribution_figure,
        "dist_panels": distribution_panels,
        "show_dist": wanted_distribution,
        "metric_party_counts": dict(k_of_metric),
        "available_metrics": tuple(available),
        "circuit_name": circuit_name,
        "is_fermionic": is_fermionic,
        "data": data,
        "summary": summary,
        "fits": fits,
        "fit_ranges": dict(fit_ranges),
        "paper_fit_ranges": paper_fit_ranges,
        "plotted_fit_sizes": dict(plotted_fit_size),
        "mi_units": mi_units,
        "selected_fit_points": selected_points,
    }

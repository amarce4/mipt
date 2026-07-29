"""Two-site mutual-information and negativity distance scaling."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fitting import _weighted_linear_fit
from .loading import _parse_circuit_name, _parse_size, _resolve_files
from .plotting import (
    _color_map_by_size,
    _mi_unit_spec,
    _positive_log_errorbar,
    _show,
)


def _chunked_distance_summary(
    path: str | Path,
    *,
    metrics: Sequence[str],
    chunksize: int,
    distance_round: int,
) -> dict[str, pd.DataFrame]:
    """Calculate count, mean, and standard error by chord distance."""
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
    fit = {
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
    return fit, selected


def dist_scaling(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    sizes: Sequence[int] | None = None,
    l_by_file: Mapping[str, int] | None = None,
    fit_size_mi: int | None = None,
    fit_size_fmi: int | None = None,
    fit_size_mn: int | None = None,
    fit_size_fmn: int | None = None,
    fit_range_mi: (
        tuple[float | None, float | None]
        | Mapping[int, tuple[float | None, float | None]]
        | None
    ) = None,
    fit_range_fmi: (
        tuple[float | None, float | None]
        | Mapping[int, tuple[float | None, float | None]]
        | None
    ) = None,
    fit_range_mn: (
        tuple[float | None, float | None]
        | Mapping[int, tuple[float | None, float | None]]
        | None
    ) = None,
    fit_range_fmn: (
        tuple[float | None, float | None]
        | Mapping[int, tuple[float | None, float | None]]
        | None
    ) = None,
    fit_last_mi: int | None = 4,
    fit_last_fmi: int | None = 4,
    fit_last_mn: int | None = 4,
    fit_last_fmn: int | None = 4,
    min_relative_error: float = 0.03,
    chunksize: int = 1_000_000,
    distance_round: int = 12,
    mi_units: str = "nats",
    show_errorbars: bool = True,
    capsize: float = 2.0,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    dpi: int = 130,
    title: str | None = None,
    show_summary: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    r"""Plot Fig.-4-style two-site distance scaling for MI and negativity.

    Filenames must contain a circuit tag: ``mms``, ``haar``, ``rppu``,
    ``rfgs``, or ``qrppu``. Qubit-system CSVs use ``d,mi,mn`` and produce two
    panels. Fermionic-system CSVs use ``d,mi,fmi,mn,fmn`` and produce four
    panels, comparing ordinary and fermionic partial trace/transposition
    conventions. Rows are aggregated at equal chord distance, including zero
    negativity events. All panels use logarithmic axes. Power laws

    .. math::

        O(d)=A_O\,d^{-\alpha_2^O}

    are fitted by weighted least squares in log coordinates. As in Fig. 4 of
    arXiv:2602.04969, the relative uncertainty used by the fit is floored at
    ``min_relative_error`` (default 0.03).

    Each ``fit_range_*`` may be one ``(d_min, d_max)`` tuple applied to every
    size, or a dictionary keyed by system size. When no range is supplied, the
    largest ``fit_last_*`` positive-distance points are used. A dashed line is
    drawn only for the corresponding ``fit_size_*`` (largest size by default),
    while fit results for every supplied size are returned in ``fits``.
    ``mi_units`` converts both MI conventions and their fitted prefactors
    between ``"nats"`` and ``"bits"``; it does not change their exponents.
    """
    if min_relative_error <= 0.0:
        raise ValueError("min_relative_error must be positive.")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    if distance_round < 0:
        raise ValueError("distance_round must be non-negative.")
    mi_units, mi_scale = _mi_unit_spec(mi_units)

    paths = _resolve_files(files, file_glob)
    circuit_names = [_parse_circuit_name(path) for path in paths]
    unique_circuits = sorted(set(circuit_names))
    if len(unique_circuits) != 1:
        raise ValueError(
            "Distance-scaling files must belong to one circuit type; found "
            f"{unique_circuits}."
        )
    circuit_name = unique_circuits[0]
    is_fermionic = circuit_name in {"RPPU", "RFGS"}
    metrics = ("mi", "fmi", "mn", "fmn") if is_fermionic else ("mi", "mn")

    if sizes is None:
        parsed_sizes = [_parse_size(path, l_by_file) for path in paths]
    else:
        parsed_sizes = [int(size) for size in sizes]
        if len(parsed_sizes) != len(paths):
            raise ValueError("sizes must match the number of input files.")

    if len(set(parsed_sizes)) != len(parsed_sizes):
        raise ValueError(f"Duplicate system sizes found: {parsed_sizes}")

    order = np.argsort(parsed_sizes)
    paths = [paths[index] for index in order]
    parsed_sizes = [parsed_sizes[index] for index in order]

    data: dict[int, dict[str, pd.DataFrame]] = {}
    for size, path in zip(parsed_sizes, paths):
        print(f"Importing L={size}: {path}")
        data[size] = _chunked_distance_summary(
            path,
            metrics=metrics,
            chunksize=chunksize,
            distance_round=distance_round,
        )
        for metric in ("mi", "fmi"):
            if metric in data[size]:
                data[size][metric].loc[:, ["mean", "stderr"]] *= mi_scale
        if all(data[size][metric].empty for metric in metrics):
            raise ValueError(f"No valid distance-scaling rows were found in {path}.")

    fit_rows: list[dict[str, Any]] = []
    selected_points: dict[tuple[int, str], pd.DataFrame] = {}
    fit_settings = {
        "mi": (fit_range_mi, fit_last_mi),
        "fmi": (fit_range_fmi, fit_last_fmi),
        "mn": (fit_range_mn, fit_last_mn),
        "fmn": (fit_range_fmn, fit_last_fmn),
    }
    for size in parsed_sizes:
        for metric in metrics:
            range_spec, last_points = fit_settings[metric]
            try:
                fit, selected = _distance_power_law_fit(
                    data[size][metric],
                    fit_range=_fit_range_for_size(range_spec, size),
                    fit_last=last_points,
                    min_relative_error=min_relative_error,
                )
                fit_rows.append({"L": size, "metric": metric, **fit})
                selected_points[(size, metric)] = selected
            except ValueError as exc:
                fit_rows.append(
                    {
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
                warnings.warn(f"L={size}, {metric.upper()} fit unavailable: {exc}")

    fits = pd.DataFrame(fit_rows)
    requested_fit_sizes = {
        "mi": fit_size_mi,
        "fmi": fit_size_fmi,
        "mn": fit_size_mn,
        "fmn": fit_size_fmn,
    }
    fit_sizes = {
        metric: (
            max(parsed_sizes)
            if requested_fit_sizes[metric] is None
            else int(requested_fit_sizes[metric])
        )
        for metric in metrics
    }
    for metric, fit_size in fit_sizes.items():
        if fit_size not in data:
            raise ValueError(
                f"fit_size_{metric}={fit_size} is not among {parsed_sizes}."
            )

    colors = _color_map_by_size(parsed_sizes, cmap)
    if figsize is None:
        figsize = (12.5, 10.0) if is_fermionic else (12.5, 5.2)
    if is_fermionic:
        fig, axis_grid = plt.subplots(
            2, 2, figsize=figsize, dpi=dpi, constrained_layout=True
        )
        flat_axes = tuple(axis_grid.flat)
    else:
        fig, axis_grid = plt.subplots(
            1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
        )
        flat_axes = tuple(np.atleast_1d(axis_grid).flat)
    axes = dict(zip(metrics, flat_axes))

    for size in parsed_sizes:
        color = colors[size]
        for metric in metrics:
            ax = axes[metric]
            curve = data[size][metric]
            positive = curve.loc[
                np.isfinite(curve["d"])
                & np.isfinite(curve["mean"])
                & (curve["d"] > 0.0)
                & (curve["mean"] > 0.0)
            ]
            omitted = len(curve) - len(positive)
            if omitted:
                warnings.warn(
                    f"L={size}, {metric.upper()}: omitted {omitted} non-positive "
                    "mean point(s) from the logarithmic plot."
                )
            if positive.empty:
                continue

            x = positive["d"].to_numpy(dtype=float)
            y = positive["mean"].to_numpy(dtype=float)
            dy = positive["stderr"].to_numpy(dtype=float)
            dy = np.where(np.isfinite(dy), dy, 0.0)
            if show_errorbars:
                _positive_log_errorbar(
                    ax,
                    x,
                    y,
                    dy,
                    fmt="o",
                    markersize=4.0,
                    color=color,
                    ecolor=color,
                    markeredgecolor=color,
                    capsize=capsize,
                    elinewidth=0.9,
                    linestyle="none",
                    label=rf"$L={size}$",
                    alpha=0.95,
                )
            else:
                ax.plot(
                    x,
                    y,
                    linestyle="none",
                    marker="o",
                    markersize=4.0,
                    color=color,
                    markeredgecolor=color,
                    label=rf"$L={size}$",
                    alpha=0.95,
                )

    metric_specs = {
        "mi": {
            "exponent": "MI",
            "ylabel": rf"Two-party mutual information $I_2$ [{mi_units}]",
            "title": "Ordinary mutual information",
        },
        "fmi": {
            "exponent": r"\mathrm{fMI}",
            "ylabel": rf"Fermionic mutual information $I_2^f$ [{mi_units}]",
            "title": "Fermionic mutual information",
        },
        "mn": {
            "exponent": "MN",
            "ylabel": r"Bipartite negativity $\mathcal{N}_2$",
            "title": "Ordinary negativity",
        },
        "fmn": {
            "exponent": r"\mathrm{fMN}",
            "ylabel": r"Fermionic negativity $\mathcal{N}_2^f$",
            "title": "Fermionic negativity",
        },
    }

    for metric, fit_size in fit_sizes.items():
        row = fits.loc[
            (fits["L"] == fit_size) & (fits["metric"] == metric)
        ].iloc[0]
        if not np.isfinite(row["alpha"]):
            continue
        selected = selected_points[(fit_size, metric)]
        x_line = np.geomspace(
            selected["d"].min(), selected["d"].max(), 200
        )
        y_line = row["prefactor"] * x_line ** (-row["alpha"])
        axes[metric].plot(
            x_line,
            y_line,
            color=colors[fit_size],
            linestyle="--",
            linewidth=1.5,
            label=(
                rf"$L={fit_size}$ fit: "
                rf"$\alpha_2^{{{metric_specs[metric]['exponent']}}}="
                rf"{row['alpha']:.3g}\pm {row['alpha_stderr']:.2g}$"
            ),
            zorder=5,
        )

    for metric in metrics:
        ax = axes[metric]
        ax.set_xlabel(r"Effective chord distance $d$")
        ax.set_ylabel(metric_specs[metric]["ylabel"])
        ax.set_title(metric_specs[metric]["title"])
        ax.set_xscale("log")
        # Plain-number formatting for logarithmic distance axis.
        from matplotlib.ticker import FuncFormatter
        _plain_distance = FuncFormatter(
            lambda value, _: f"{value:g}" if value > 0 else ""
        )
        ax.xaxis.set_major_formatter(_plain_distance)
        ax.xaxis.set_minor_formatter(_plain_distance)
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
                "L": size,
                "file": str(path),
                "circuit": circuit_name,
                **{
                    f"distance_points_{metric}": len(data[size][metric])
                    for metric in metrics
                },
                **{
                    f"rows_{metric}": int(data[size][metric]["count"].sum())
                    for metric in metrics
                },
            }
            for size, path in zip(parsed_sizes, paths)
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
        "metrics": metrics,
        "circuit_name": circuit_name,
        "is_fermionic": is_fermionic,
        "data": data,
        "summary": summary,
        "fits": fits,
        "fit_sizes": fit_sizes,
        "fit_size_mi": fit_sizes["mi"],
        "fit_size_fmi": fit_sizes.get("fmi"),
        "fit_size_mn": fit_sizes["mn"],
        "fit_size_fmn": fit_sizes.get("fmn"),
        "mi_units": mi_units,
        "selected_fit_points": selected_points,
    }

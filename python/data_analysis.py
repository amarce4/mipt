"""Reusable analysis and plotting tools for the MIPT simulation outputs.

Public functions
----------------
gmn(...)               Plot GMN/fGMN and optional minimum bipartite negativity.
entropy(...)           Plot Rényi-2 entropy scaling curves and weighted fits.
dist_scaling(...)      Plot two-site MI/negativity distance scaling and power-law fits.
expvals(...)           Plot aggregate fermionic observables and Wick residuals.
tmi_collapse(...)      Fit and plot the TMI finite-size scaling collapse.
probe1_collapse(...)   Fit and plot the one-ancilla dynamical collapse.
probe2_collapse(...)   Fit and plot the two-ancilla bulk-exponent collapse.

All plotting functions return a dictionary containing the figure, axes, loaded
or summarized data, and fitted parameters. They show the plot by default; pass
``show=False`` to suppress ``plt.show()``.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
import re
import warnings
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

__all__ = [
    "gmn",
    "entropy",
    "dist_scaling",
    "expvals",
    "tmi_collapse",
    "probe1_collapse",
    "probe2_collapse",
]


DEFAULT_EXPVAL_OBSERVABLES = {
    "hop_r1_sq": r"Squared hopping, $r=1$",
    "hop_r2_sq": r"Squared hopping, $r=2$",
    "pair_r1_sq": r"Squared pairing, $r=1$",
    "pair_r2_sq": r"Squared pairing, $r=2$",
    "density_r1_sq": r"Squared connected density, $r=1$",
    "density_r2_sq": r"Squared connected density, $r=2$",
    "wick4": r"Four-Majorana Wick residual $W_4$",
    "wick6": r"Six-Majorana Wick residual $W_6$",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _show(fig, show: bool) -> None:
    if show:
        plt.show()


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


def _color_map_by_size(sizes: Sequence[int], cmap_name: str) -> dict[int, Any]:
    cmap = plt.get_cmap(cmap_name)
    positions = (
        np.array([0.15])
        if len(sizes) == 1
        else np.linspace(0.10, 0.90, len(sizes))
    )
    return {size: cmap(position) for size, position in zip(sizes, positions)}


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


def _weighted_linear_fit(
    x: Iterable[float],
    y: Iterable[float],
    sigma: Iterable[float],
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sigma)
        & (sigma > 0.0)
    )
    x, y, sigma = x[valid], y[valid], sigma[valid]
    if x.size < 3:
        raise ValueError("At least three valid points are required for the fit.")

    design = np.column_stack((x, np.ones_like(x)))
    weights = 1.0 / sigma**2
    normal = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Linear-fit covariance matrix is singular.") from exc

    slope, intercept = covariance @ (design.T @ (weights * y))
    fitted = slope * x + intercept
    reduced_chi2 = np.sum(((y - fitted) / sigma) ** 2) / (x.size - 2)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_stderr": float(np.sqrt(covariance[0, 0])),
        "intercept_stderr": float(np.sqrt(covariance[1, 1])),
        "reduced_chi2": float(reduced_chi2),
    }


# ---------------------------------------------------------------------------
# GMN / fGMN
# ---------------------------------------------------------------------------


def gmn(
    files: Sequence[str | Path] | str | Path,
    sizes: Sequence[int],
    *,
    metric: str = "gmn",
    p_range: tuple[float | None, float | None] = (0.0, 1.0),
    chunksize: int = 1_000_000,
    plot_bipneg: bool = True,
    periods: int | None = None,
    realizations: int | None = None,
    colors: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (10, 4.5),
    show: bool = True,
) -> dict[str, Any]:
    """Plot GMN or fGMN means and standard errors versus measurement rate."""
    metric = metric.lower()
    if metric not in {"gmn", "fgmn"}:
        raise ValueError("metric must be 'gmn' or 'fgmn'.")

    paths = _as_paths(files)
    sizes = list(sizes)
    if len(paths) != len(sizes):
        raise ValueError("files and sizes must have equal lengths.")

    p_min, p_max = p_range
    data: dict[int, dict[str, pd.DataFrame]] = {}
    columns = [metric] + (["bipneg"] if plot_bipneg else [])

    for size, path in zip(sizes, paths):
        print(f"Importing L={size}: {path}")
        data[size] = _chunked_group_summary(
            path,
            columns,
            p_min=p_min,
            p_max=p_max,
            chunksize=chunksize,
        )

    fig, ax = plt.subplots(figsize=figsize)
    colors = list(colors) if colors is not None else [
        plt.get_cmap("tab10")(i % 10) for i in range(len(sizes))
    ]
    metric_name = "GMN" if metric == "gmn" else "fGMN"

    for i, size in enumerate(sizes):
        color = colors[i % len(colors)]
        curve = data[size][metric]
        if curve.empty:
            warnings.warn(f"No valid {metric} data found for L={size}.")
            continue
        ax.errorbar(
            curve["p"],
            curve["mean"],
            yerr=curve["stderr"],
            fmt=".",
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1,
            label=rf"{metric_name}, $L={size}$",
        )
        ax.plot(curve["p"], curve["mean"], color=color, linewidth=1)

        if plot_bipneg:
            bip = data[size]["bipneg"]
            if not bip.empty:
                ax.errorbar(
                    bip["p"],
                    bip["mean"],
                    yerr=bip["stderr"],
                    fmt="x",
                    color=color,
                    ecolor=color,
                    capsize=2,
                    elinewidth=0.8,
                    alpha=0.75,
                    label=rf"Min. bipartite negativity, $L={size}$",
                )
                ax.plot(
                    bip["p"],
                    bip["mean"],
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.75,
                )

    ax.set_xlabel(r"Measurement rate $p$")
    ax.set_ylabel(metric_name)
    if p_min is not None and p_max is not None:
        ax.set_xlim(p_min, p_max)
    ax.grid(alpha=0.2)
    ax.legend()

    metadata: list[str] = []
    if len(sizes) == 1:
        metadata.append(rf"$L={sizes[0]}$")
    if periods is not None:
        metadata.append(rf"$t={periods}$")
    if realizations is not None:
        metadata.append(f"Realisations per p: {realizations:,}")
    if len(sizes) == 1 and not data[sizes[0]][metric].empty:
        counts = data[sizes[0]][metric]["count"].to_numpy(dtype=int)
        if counts.min() == counts.max():
            metadata.append(f"Rows per p: {counts.min():,}")
        else:
            metadata.append(f"Rows per p: {counts.min():,}–{counts.max():,}")
    if metadata:
        ax.text(
            0.98,
            0.96,
            "\n".join(metadata),
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="top",
            bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.7},
        )

    fig.tight_layout()
    _show(fig, show)
    return {"figure": fig, "axis": ax, "data": data}


# ---------------------------------------------------------------------------
# Rényi-2 entropy
# ---------------------------------------------------------------------------


def entropy(
    files: Sequence[str | Path] | str | Path,
    *,
    fit_above_p: float = 0.3,
    exclude_first: int = 1,
    uncertainty: str = "S2_stderr",
    xlim: tuple[float, float] | None = (0.5, 2.05),
    ylim: tuple[float, float] | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 6),
    show: bool = True,
) -> dict[str, Any]:
    """Plot entropy versus conformal distance and fit selected curves."""
    paths = _as_paths(files)
    if not paths:
        raise FileNotFoundError("No entropy CSVs were provided.")
    if exclude_first < 0:
        raise ValueError("exclude_first must be non-negative.")

    curves: list[dict[str, Any]] = []
    for path in paths:
        df = pd.read_csv(path)
        required = {
            "N",
            "periods",
            "p",
            "completed_realizations",
            "circ_type",
            "circuit_name",
            "L_A",
            "ln_x",
            "S2_mean",
            uncertainty,
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df = df.sort_values("L_A").reset_index(drop=True)
        curves.append(
            {
                "path": path,
                "df": df,
                "L": int(df["N"].iloc[0]),
                "periods": int(df["periods"].iloc[0]),
                "p": float(df["p"].iloc[0]),
                "realizations": int(df["completed_realizations"].iloc[0]),
                "circuit_type": int(df["circ_type"].iloc[0]),
                "circuit_name": str(df["circuit_name"].iloc[0]),
                "x": df["ln_x"].to_numpy(dtype=float),
                "y": df["S2_mean"].to_numpy(dtype=float),
                "dy": df[uncertainty].to_numpy(dtype=float),
            }
        )

    unique_sizes = sorted({curve["L"] for curve in curves})
    unique_periods = sorted({curve["periods"] for curve in curves})
    if len(unique_sizes) != 1:
        raise ValueError(f"Input CSVs use multiple L values: {unique_sizes}")
    if len(unique_periods) != 1:
        raise ValueError(f"Input CSVs use multiple period counts: {unique_periods}")

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, len(curves)))
    fit_rows: list[dict[str, Any]] = []

    for color, curve in zip(colors, curves):
        x, y, dy, p = curve["x"], curve["y"], curve["dy"], curve["p"]
        ax.errorbar(
            x,
            y,
            yerr=dy,
            fmt=".",
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1,
            label="_nolegend_",
        )
        data_line, = ax.plot(x, y, color=color, linewidth=1, label="_nolegend_")

        fit_record = {"file": str(curve["path"]), "p": p}
        if p > fit_above_p:
            try:
                fit = _weighted_linear_fit(
                    x[exclude_first:],
                    y[exclude_first:],
                    dy[exclude_first:],
                )
                fit_x = x[exclude_first:]
                fit_x_line = np.linspace(np.min(fit_x), np.max(fit_x), 200)
                fit_y_line = fit["slope"] * fit_x_line + fit["intercept"]
                slope_error = fit["slope_stderr"]
                precision = (
                    max(0, int(np.ceil(-np.log10(abs(slope_error)))))
                    if slope_error > 0 and np.isfinite(slope_error)
                    else 3
                )
                ax.plot(
                    fit_x_line,
                    fit_y_line,
                    linestyle=":",
                    linewidth=2,
                    color=color,
                    label=(
                        rf"$p={p:g}$: $\alpha=${fit['slope']:.{precision}f}"
                        rf"$\pm${slope_error:.{precision}f}, "
                        rf"$\chi_\nu^2=${fit['reduced_chi2']:.1g}"
                    ),
                )
                fit_record.update(fit)
            except ValueError as exc:
                data_line.set_label(rf"$p={p:g}$: fit unavailable ({exc})")
                fit_record["error"] = str(exc)
        else:
            data_line.set_label(rf"$p={p:g}$")
        fit_rows.append(fit_record)

    ax.text(
        0.99,
        0.02,
        rf"$L={unique_sizes[0]}$" + "\n" + rf"$t={unique_periods[0]}$",
        transform=ax.transAxes,
        fontsize=12,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.7},
    )
    ax.set_xlabel(
        r"$\ln x$, $x=\frac{L}{\pi}\sin\left(\frac{\pi L_A}{L}\right)$"
    )
    ax.set_ylabel(r"$\overline{S_A^{(2)}}$")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.grid(alpha=0.25)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    _show(fig, show)

    metadata = pd.DataFrame(
        [
            {
                "file": str(curve["path"]),
                "L": curve["L"],
                "periods": curve["periods"],
                "p": curve["p"],
                "realizations": curve["realizations"],
                "circuit_type": curve["circuit_type"],
                "circuit_name": curve["circuit_name"],
            }
            for curve in curves
        ]
    )
    return {
        "figure": fig,
        "axis": ax,
        "fits": pd.DataFrame(fit_rows),
        "metadata": metadata,
        "curves": curves,
    }



# ---------------------------------------------------------------------------
# Two-site distance scaling
# ---------------------------------------------------------------------------


def _chunked_distance_summary(
    path: str | Path,
    *,
    chunksize: int,
    distance_round: int,
) -> dict[str, pd.DataFrame]:
    """Calculate count, mean, and standard error by chord distance."""
    partials: dict[str, list[pd.DataFrame]] = {"mi": [], "mn": []}

    try:
        reader = pd.read_csv(
            path,
            usecols=["d", "mi", "mn"],
            chunksize=chunksize,
        )
    except ValueError as exc:
        raise ValueError(
            f"{path!r} must contain the columns d, mi, and mn."
        ) from exc

    for chunk in reader:
        for column in ("d", "mi", "mn"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.loc[np.isfinite(chunk["d"]) & (chunk["d"] > 0.0)]
        if chunk.empty:
            continue
        chunk["_d"] = chunk["d"].round(distance_round)

        for metric in ("mi", "mn"):
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


def _positive_log_errorbar(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    **kwargs,
):
    """Draw log-compatible error bars without extending below zero."""
    lower = np.minimum(np.maximum(dy, 0.0), 0.999999 * y)
    upper = np.maximum(dy, 0.0)
    return ax.errorbar(x, y, yerr=np.vstack((lower, upper)), **kwargs)


def dist_scaling(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    sizes: Sequence[int] | None = None,
    l_by_file: Mapping[str, int] | None = None,
    fit_size_mi: int | None = None,
    fit_size_mn: int | None = None,
    fit_range_mi: (
        tuple[float | None, float | None]
        | Mapping[int, tuple[float | None, float | None]]
        | None
    ) = None,
    fit_range_mn: (
        tuple[float | None, float | None]
        | Mapping[int, tuple[float | None, float | None]]
        | None
    ) = None,
    fit_last_mi: int | None = 4,
    fit_last_mn: int | None = 4,
    min_relative_error: float = 0.03,
    chunksize: int = 1_000_000,
    distance_round: int = 12,
    show_errorbars: bool = True,
    capsize: float = 2.0,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (12.5, 5.2),
    dpi: int = 130,
    title: str | None = None,
    show_summary: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    r"""Plot Fig.-4-style two-site distance scaling for MI and negativity.

    Each input CSV must contain one raw row per two-site RDM with columns
    ``d,mi,mn``. Rows are aggregated at equal chord distance, including zero
    negativity events. Both panels use logarithmic axes. Power laws

    .. math::

        I_2(d)=A_{MI}d^{-\alpha_2^{MI}},\qquad
        \mathcal{N}_2(d)=A_{MN}d^{-\alpha_2^{MN}}

    are fitted by weighted least squares in log coordinates. As in Fig. 4 of
    arXiv:2602.04969, the relative uncertainty used by the fit is floored at
    ``min_relative_error`` (default 0.03).

    ``fit_range_mi`` and ``fit_range_mn`` may be one ``(d_min, d_max)`` tuple
    applied to every size, or dictionaries keyed by system size. When no range
    is supplied, the largest ``fit_last_*`` positive-distance points are used.
    The dashed line is drawn only for ``fit_size_*`` (largest size by default),
    while fit results for every supplied size are returned in ``fits``.
    """
    if min_relative_error <= 0.0:
        raise ValueError("min_relative_error must be positive.")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    if distance_round < 0:
        raise ValueError("distance_round must be non-negative.")

    paths = _resolve_files(files, file_glob)
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
            chunksize=chunksize,
            distance_round=distance_round,
        )
        if data[size]["mi"].empty and data[size]["mn"].empty:
            raise ValueError(f"No valid distance-scaling rows were found in {path}.")

    fit_rows: list[dict[str, Any]] = []
    selected_points: dict[tuple[int, str], pd.DataFrame] = {}
    fit_settings = {
        "mi": (fit_range_mi, fit_last_mi),
        "mn": (fit_range_mn, fit_last_mn),
    }
    for size in parsed_sizes:
        for metric, (range_spec, last_points) in fit_settings.items():
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
    fit_size_mi = max(parsed_sizes) if fit_size_mi is None else int(fit_size_mi)
    fit_size_mn = max(parsed_sizes) if fit_size_mn is None else int(fit_size_mn)
    for metric, fit_size in (("mi", fit_size_mi), ("mn", fit_size_mn)):
        if fit_size not in data:
            raise ValueError(
                f"fit_size_{metric}={fit_size} is not among {parsed_sizes}."
            )

    colors = _color_map_by_size(parsed_sizes, cmap)
    fig, (ax_mi, ax_mn) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )
    axes = {"mi": ax_mi, "mn": ax_mn}

    for size in parsed_sizes:
        color = colors[size]
        for metric in ("mi", "mn"):
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

    for metric, fit_size in (("mi", fit_size_mi), ("mn", fit_size_mn)):
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
        exponent = "MI" if metric == "mi" else "MN"
        axes[metric].plot(
            x_line,
            y_line,
            color=colors[fit_size],
            linestyle="--",
            linewidth=1.5,
            label=(
                rf"$L={fit_size}$ fit: "
                rf"$\alpha_2^{{{exponent}}}="
                rf"{row['alpha']:.3g}\pm{row['alpha_stderr']:.2g}$"
            ),
            zorder=5,
        )

    ax_mi.set_xlabel(r"Effective chord distance $d$")
    ax_mi.set_ylabel(r"Two-party mutual information $I_2$")
    ax_mi.set_title("Two-party mutual information")
    ax_mn.set_xlabel(r"Effective chord distance $d$")
    ax_mn.set_ylabel(r"Bipartite negativity $\mathcal{N}_2$")
    ax_mn.set_title("Bipartite negativity")

    for ax in (ax_mi, ax_mn):
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
                "distance_points_mi": len(data[size]["mi"]),
                "distance_points_mn": len(data[size]["mn"]),
                "rows_mi": int(data[size]["mi"]["count"].sum()),
                "rows_mn": int(data[size]["mn"]["count"].sum()),
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
        "axes": (ax_mi, ax_mn),
        "data": data,
        "summary": summary,
        "fits": fits,
        "fit_size_mi": fit_size_mi,
        "fit_size_mn": fit_size_mn,
        "selected_fit_points": selected_points,
    }


# ---------------------------------------------------------------------------
# Expectation values
# ---------------------------------------------------------------------------


def expvals(
    file: str | Path,
    *,
    p_range: tuple[float | None, float | None] = (0.0, 1.0),
    observables: Mapping[str, str] | None = None,
    normalize: bool = False,
    figsize: tuple[float, float] = (19, 11),
    capsize: float = 2,
    alpha: float = 0.9,
    show_table: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    """Plot aggregate three-mode fermionic observables versus p."""
    file = Path(file)
    observables = dict(observables or DEFAULT_EXPVAL_OBSERVABLES)
    df = pd.read_csv(file)
    p_min, p_max = p_range
    if p_min is not None:
        df = df.loc[df["p"] >= p_min]
    if p_max is not None:
        df = df.loc[df["p"] <= p_max]
    df = df.sort_values("p").copy()
    if df.empty:
        raise ValueError(f"No data remain in the selected p range {p_range}.")

    required = {"p", "realizations"}
    for base in observables:
        required.update({f"{base}_mean", f"{base}_stderr"})
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=figsize)
    for base, label in observables.items():
        mean = df[f"{base}_mean"].to_numpy(dtype=float)
        stderr = df[f"{base}_stderr"].to_numpy(dtype=float)
        if normalize:
            scale = np.nanmax(np.abs(mean))
            if np.isfinite(scale) and scale > 0.0:
                mean, stderr = mean / scale, stderr / scale
        ax.errorbar(
            df["p"],
            mean,
            yerr=stderr,
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=capsize,
            alpha=alpha,
            label=label,
        )

    ax.set_xlabel(r"Measurement probability $p$")
    ax.set_ylabel("Mean observable / maximum mean" if normalize else "Observable")
    n_min, n_max = int(df["realizations"].min()), int(df["realizations"].max())
    realization_text = (
        f"{n_min:,} trajectories per $p$"
        if n_min == n_max
        else f"{n_min:,}–{n_max:,} trajectories per $p$"
    )
    title = (
        "Three-mode fermionic expectation values and Wick residuals\n"
        f"{file.name}; {realization_text}; "
        "error bars are trajectory-level standard errors"
    )
    if normalize:
        title += "; curves normalized independently"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    _show(fig, show)

    if show_table:
        try:
            from IPython.display import display

            display(df)
        except ImportError:
            print(df.to_string(index=False))

    return {"figure": fig, "axis": ax, "data": df}


# ---------------------------------------------------------------------------
# TMI finite-size scaling collapse
# ---------------------------------------------------------------------------


def _summarize_tmi(
    path: str | Path,
    *,
    p_min: float | None,
    p_max: float | None,
    chunksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summary = _chunked_group_summary(
        path,
        ["tmi"],
        p_min=p_min,
        p_max=p_max,
        chunksize=chunksize,
    )["tmi"]
    return (
        summary["p"].to_numpy(dtype=float),
        summary["mean"].to_numpy(dtype=float),
        summary["stderr"].to_numpy(dtype=float),
    )


def _prepare_sorted_arrays(
    ps_arr: Sequence[np.ndarray],
    mean_arr: Sequence[np.ndarray],
    stderr_arr: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ps_out, mean_out, stderr_out = [], [], []
    for ps, mean, stderr in zip(ps_arr, mean_arr, stderr_arr):
        ps = np.asarray(ps, dtype=float)
        order = np.argsort(ps)
        ps_out.append(ps[order])
        mean_out.append(np.asarray(mean, dtype=float)[order])
        stderr_out.append(np.asarray(stderr, dtype=float)[order])
    return ps_out, mean_out, stderr_out


def _align_tmi_grids(
    ps_arr: Sequence[np.ndarray],
    mean_arr: Sequence[np.ndarray],
    stderr_arr: Sequence[np.ndarray],
    *,
    p_tol: float,
    min_points: int,
    allow_interpolation: bool,
    error_floor: float,
    sizes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    ps_arr, mean_arr, stderr_arr = _prepare_sorted_arrays(
        ps_arr, mean_arr, stderr_arr
    )
    matches: list[list[int]] = []
    for i_ref, p_ref in enumerate(ps_arr[0]):
        indices = [i_ref]
        for ps in ps_arr[1:]:
            k = int(np.argmin(np.abs(ps - p_ref)))
            if abs(ps[k] - p_ref) > p_tol:
                break
            indices.append(k)
        if len(indices) == len(ps_arr):
            matches.append(indices)

    if len(matches) >= min_points:
        rho = np.array(
            [np.mean([ps_arr[j][idx] for j, idx in enumerate(row)]) for row in matches]
        )
        a = np.array(
            [[mean_arr[j][row[j]] for row in matches] for j in range(len(ps_arr))]
        )
        da = np.array(
            [[stderr_arr[j][row[j]] for row in matches] for j in range(len(ps_arr))]
        )
        da = np.where(np.isfinite(da) & (da > 0.0), da, error_floor)
        return rho, a, da, "matched"

    if not allow_interpolation:
        raise RuntimeError(
            f"Fewer than {min_points} common p values were found within p_tol={p_tol:g}."
        )

    overlap_min = max(ps.min() for ps in ps_arr)
    overlap_max = min(ps.max() for ps in ps_arr)
    if overlap_min >= overlap_max:
        ranges = ", ".join(
            f"L={size}: [{ps.min():g}, {ps.max():g}]"
            for size, ps in zip(sizes, ps_arr)
        )
        raise RuntimeError(f"No common p range across datasets ({ranges}).")

    rho = ps_arr[0][(ps_arr[0] >= overlap_min) & (ps_arr[0] <= overlap_max)]
    if rho.size < min_points:
        native_count = min(
            np.count_nonzero((ps >= overlap_min) & (ps <= overlap_max))
            for ps in ps_arr
        )
        rho = np.linspace(overlap_min, overlap_max, max(native_count, min_points))
    if rho.size < min_points:
        raise RuntimeError(f"Only {rho.size} p points are available in the overlap.")

    a = np.asarray([np.interp(rho, ps, mean) for ps, mean in zip(ps_arr, mean_arr)])
    da = np.asarray(
        [np.interp(rho, ps, stderr) for ps, stderr in zip(ps_arr, stderr_arr)]
    )
    da = np.where(np.isfinite(da) & (da > 0.0), da, error_floor)
    return rho, a, da, "interpolated"


def _curvature_errors(
    objective,
    theta_hat: Sequence[float],
    *,
    f_min: float | None = None,
    rel_step: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_hat = np.asarray(theta_hat, dtype=float)
    if f_min is None:
        f_min = objective(theta_hat)
    steps = rel_step * np.maximum(np.abs(theta_hat), 1.0)
    hessian = np.zeros((theta_hat.size, theta_hat.size), dtype=float)

    def safe_eval(x):
        value = objective(x)
        return value if np.isfinite(value) else np.inf

    for i in range(theta_hat.size):
        step = steps[i]
        for _ in range(12):
            xp, xm = theta_hat.copy(), theta_hat.copy()
            xp[i] += step
            xm[i] -= step
            fp, fm = safe_eval(xp), safe_eval(xm)
            if np.isfinite(fp) and np.isfinite(fm):
                break
            step *= 0.5
        hessian[i, i] = (
            (fp - 2.0 * f_min + fm) / step**2
            if np.isfinite(fp) and np.isfinite(fm)
            else np.nan
        )

    for i in range(theta_hat.size):
        for j in range(i + 1, theta_hat.size):
            hi, hj = steps[i], steps[j]
            values = [np.inf] * 4
            for _ in range(12):
                points = []
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    x = theta_hat.copy()
                    x[i] += si * hi
                    x[j] += sj * hj
                    points.append(x)
                values = [safe_eval(point) for point in points]
                if all(np.isfinite(value) for value in values):
                    break
                hi *= 0.5
                hj *= 0.5
            if all(np.isfinite(value) for value in values):
                fpp, fpm, fmp, fmm = values
                value = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj)
            else:
                value = np.nan
            hessian[i, j] = hessian[j, i] = value

    if not np.all(np.isfinite(hessian)):
        return (
            np.full(theta_hat.size, np.nan),
            np.full_like(hessian, np.nan),
            hessian,
        )
    try:
        covariance = 2.0 * f_min * np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = 2.0 * f_min * np.linalg.pinv(hessian)
    errors = np.sqrt(np.where(np.diag(covariance) >= 0.0, np.diag(covariance), np.nan))
    return errors, covariance, hessian


def _collapse_quality_text(score: float) -> str:
    if not np.isfinite(score):
        return "The collapse fit failed or produced invalid scaled data."
    if score < 0.5:
        return "Suspiciously small S; errors may be overestimated or data correlated."
    if score < 2.0:
        return "S is near unity; the collapse is statistically reasonable."
    if score < 5.0:
        return "The collapse is marginal; corrections to scaling may be relevant."
    if score < 10.0:
        return "The collapse is poor; vary the fit window or system sizes."
    return "The collapse is very poor under the assumed scaling ansatz."


def tmi_collapse(
    files: Sequence[str | Path] | str | Path,
    sizes: Sequence[int],
    *,
    realizations: Sequence[int] | None = None,
    p_range: tuple[float | None, float | None] = (0.0, 1.0),
    pc0: float = 0.33,
    nu0: float = 2.0,
    zeta0: float = 0.0,
    fit_zeta: bool = False,
    x_bounds: tuple[float, float] | None = None,
    p_tol: float = 1e-6,
    min_common_p: int = 4,
    interpolate: bool = True,
    chunksize: int = 1_000_000,
    error_floor: float = 1e-12,
    curvature_step: float = 1e-3,
    colors: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (10, 8),
    show: bool = True,
) -> dict[str, Any]:
    """Fit p_c and nu for the TMI crossing using pyfssa."""
    try:
        import fssa
    except ImportError as exc:
        raise ImportError(
            "tmi_collapse requires pyfssa (the import name is 'fssa')."
        ) from exc

    # Older pyfssa releases still reference np.int.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]

    paths = _as_paths(files)
    sizes = list(sizes)
    if len(paths) != len(sizes):
        raise ValueError("files and sizes must have equal lengths.")
    if realizations is not None and len(realizations) != len(paths):
        raise ValueError("realizations must match files when provided.")

    p_min, p_max = p_range
    ps_arr, mean_arr, stderr_arr = [], [], []
    for size, path in zip(sizes, paths):
        print(f"Importing L={size}: {path}")
        ps, mean, stderr = _summarize_tmi(
            path,
            p_min=p_min,
            p_max=p_max,
            chunksize=chunksize,
        )
        if ps.size == 0:
            raise RuntimeError(f"No usable TMI data were loaded from {path}.")
        ps_arr.append(ps)
        mean_arr.append(mean)
        stderr_arr.append(stderr)

    rho, a, da, alignment = _align_tmi_grids(
        ps_arr,
        mean_arr,
        stderr_arr,
        p_tol=p_tol,
        min_points=min_common_p,
        allow_interpolation=interpolate,
        error_floor=error_floor,
        sizes=sizes,
    )
    print(f"p-grid alignment: {alignment}, {rho.size} points")
    l = np.asarray(sizes, dtype=float)

    if fit_zeta:
        fit = fssa.autoscale(
            l=l,
            rho=rho,
            a=a,
            da=da,
            rho_c0=pc0,
            nu0=nu0,
            zeta0=zeta0,
            x_bounds=x_bounds,
        )
        pc, nu, zeta = float(fit["rho"]), float(fit["nu"]), float(fit["zeta"])
        dpc, dnu, dzeta = (
            float(fit["drho"]),
            float(fit["dnu"]),
            float(fit["dzeta"]),
        )
        success = bool(fit["success"])
        raw_fit = fit
    else:
        def objective(params):
            pc_trial, nu_trial = params
            if (
                not np.isfinite(pc_trial)
                or not np.isfinite(nu_trial)
                or nu_trial <= 0.0
                or pc_trial < rho.min()
                or pc_trial > rho.max()
            ):
                return np.inf
            scaled_trial = fssa.scaledata(
                l=l,
                rho=rho,
                a=a,
                da=da,
                rho_c=pc_trial,
                nu=nu_trial,
                zeta=zeta0,
            )
            quality = fssa.quality(
                scaled_trial.x,
                scaled_trial.y,
                scaled_trial.dy,
                x_bounds=x_bounds,
            )
            return float(quality) if np.isfinite(quality) else np.inf

        raw_fit = minimize(
            objective,
            x0=np.array([pc0, nu0], dtype=float),
            method="Nelder-Mead",
            options={
                "xatol": 1e-8,
                "fatol": 1e-8,
                "maxiter": 20_000,
                "maxfev": 20_000,
            },
        )
        pc, nu, zeta = float(raw_fit.x[0]), float(raw_fit.x[1]), float(zeta0)
        quality = float(objective(raw_fit.x))
        errors, covariance, hessian = _curvature_errors(
            objective,
            raw_fit.x,
            f_min=quality,
            rel_step=curvature_step,
        )
        dpc, dnu, dzeta = float(errors[0]), float(errors[1]), 0.0
        success = bool(raw_fit.success)

    scaled = fssa.scaledata(
        l=l,
        rho=rho,
        a=a,
        da=da,
        rho_c=pc,
        nu=nu,
        zeta=zeta,
    )
    quality = float(
        fssa.quality(scaled.x, scaled.y, scaled.dy, x_bounds=x_bounds)
    )
    print(f"p_c = {pc:.8g} ± {dpc:.2g}")
    print(f"nu  = {nu:.8g} ± {dnu:.2g}")
    print(
        f"zeta = {zeta:.8g} ± {dzeta:.2g}"
        if fit_zeta
        else f"zeta = {zeta:.8g} fixed"
    )
    print(f"collapse quality S = {quality:.8g}")
    print(_collapse_quality_text(quality))

    fig, (ax_raw, ax_collapse) = plt.subplots(2, 1, figsize=figsize)
    colors = list(colors) if colors is not None else [
        plt.get_cmap("tab10")(i % 10) for i in range(len(sizes))
    ]
    for i, size in enumerate(sizes):
        color = colors[i % len(colors)]
        label = rf"$L={size}$"
        if realizations is not None:
            label += f", Reals={realizations[i]:,}"
        ax_raw.errorbar(
            ps_arr[i],
            mean_arr[i],
            yerr=stderr_arr[i],
            fmt=".",
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1,
        )
        ax_raw.plot(ps_arr[i], mean_arr[i], color=color, label=label)
        ax_collapse.errorbar(
            scaled.x[i],
            scaled.y[i],
            yerr=scaled.dy[i],
            fmt=".",
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1,
            label=rf"$L={size}$",
        )
        ax_collapse.plot(scaled.x[i], scaled.y[i], color=color, alpha=0.8)

    ax_raw.axvline(pc, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_raw.set_xlabel(r"Measurement rate $p$")
    ax_raw.set_ylabel(r"TMI $\overline{I}_3$")
    ax_raw.set_title("Raw TMI crossing")
    ax_raw.legend()

    ax_collapse.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_collapse.set_xlabel(r"Scaled variable $(p-p_c)L^{1/\nu}$")
    ax_collapse.set_ylabel(
        r"Scaled TMI $L^{-\zeta/\nu}\overline{I}_3$"
        if fit_zeta
        else r"TMI $\overline{I}_3$"
    )
    ax_collapse.set_title("Finite-size scaling collapse")
    ax_collapse.legend()

    text = (
        rf"$p_c={pc:.4g}\pm{dpc:.1g}$" "\n"
        rf"$\nu={nu:.3g}\pm{dnu:.1g}$" "\n"
        rf"$S={quality:.3g}$" "\n"
        + (
            rf"$\zeta={zeta:.3g}\pm{dzeta:.1g}$"
            if fit_zeta
            else rf"$\zeta={zeta:g}$ fixed"
        )
    )
    ax_collapse.text(
        0.03,
        0.97,
        text,
        transform=ax_collapse.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_raw, ax_collapse),
        "pc": pc,
        "pc_stderr": dpc,
        "nu": nu,
        "nu_stderr": dnu,
        "zeta": zeta,
        "zeta_stderr": dzeta,
        "quality": quality,
        "quality_interpretation": _collapse_quality_text(quality),
        "success": success,
        "alignment": alignment,
        "p_grid": rho,
        "scaled": scaled,
        "raw_fit": raw_fit,
        "raw_data": {
            size: {"p": ps, "mean": mean, "stderr": stderr}
            for size, ps, mean, stderr in zip(sizes, ps_arr, mean_arr, stderr_arr)
        },
    }


# ---------------------------------------------------------------------------
# One-probe dynamical collapse
# ---------------------------------------------------------------------------


def _load_probe1_curve(
    path: Path,
    *,
    l_by_file: Mapping[str, int] | None,
    t_max: float | None,
) -> dict[str, Any]:
    df = pd.read_csv(path)
    t_col = _find_column(df, ["t", "time", "timestep"])
    mean_col = _find_column(
        df, ["S_mean", "s_mean", "entropy_mean", "ancilla_entropy_mean"]
    )
    stderr_col = _find_column(
        df, ["S_stderr", "s_stderr", "stderr", "entropy_stderr"]
    )
    clean = pd.DataFrame(
        {
            "t": pd.to_numeric(df[t_col], errors="coerce"),
            "S": pd.to_numeric(df[mean_col], errors="coerce"),
            "dS": pd.to_numeric(df[stderr_col], errors="coerce"),
        }
    ).dropna()
    clean = clean.sort_values("t").drop_duplicates("t", keep="last")
    if t_max is not None:
        if t_max < 0:
            raise ValueError("t_max must be non-negative or None.")
        clean = clean.loc[clean["t"] <= t_max]
    if clean.empty:
        raise ValueError(f"No valid rows remain in {path}.")
    if np.any(clean["dS"] < 0):
        warnings.warn(f"{path}: negative standard errors found; using absolute values.")
        clean["dS"] = np.abs(clean["dS"])
    return {
        "path": path,
        "L": _parse_size(path, l_by_file),
        "t": clean["t"].to_numpy(dtype=float),
        "S": clean["S"].to_numpy(dtype=float),
        "dS": clean["dS"].to_numpy(dtype=float),
    }


def probe1_collapse(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    l_by_file: Mapping[str, int] | None = None,
    t_max: float | None = 150,
    fit_t: tuple[float, float | None] = (2.0, None),
    fit_s_min: float = 1e-6,
    z_bounds: tuple[float, float] = (0.4, 2.5),
    x_a_bounds: tuple[float, float] = (0.0, 1.0),
    fixed_x_a: float | None = 0.0,
    interpolation_points: int = 200,
    relative_error_floor: float = 0.01,
    absolute_error_floor: float = 1e-6,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 24680,
    bootstrap_maxiter: int = 500,
    raw_xscale: str = "linear",
    collapse_xscale: str = "linear",
    show_errorbars: bool = True,
    capsize: float = 2,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (13, 5),
    dpi: int = 130,
    show: bool = True,
) -> dict[str, Any]:
    """Fit S_A(L,t)=L^(-x_A)F(t/L^z) for one ancilla probe."""
    paths = _resolve_files(files, file_glob)
    curves = [
        _load_probe1_curve(path, l_by_file=l_by_file, t_max=t_max)
        for path in paths
    ]
    curves.sort(key=lambda curve: curve["L"])
    sizes = [curve["L"] for curve in curves]
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"Duplicate system sizes found: {sizes}")
    if len(curves) < 3:
        warnings.warn("A collapse with fewer than three sizes is weakly constrained.")

    fit_t_min, fit_t_max = fit_t

    def fit_mask(curve):
        mask = (
            np.isfinite(curve["t"])
            & np.isfinite(curve["S"])
            & np.isfinite(curve["dS"])
            & (curve["t"] >= fit_t_min)
            & (curve["S"] >= fit_s_min)
        )
        if fit_t_max is not None:
            mask &= curve["t"] <= fit_t_max
        return mask

    def transformed(curve, z, x_a):
        mask = fit_mask(curve)
        scale = float(curve["L"]) ** x_a
        x = curve["t"][mask] / float(curve["L"]) ** z
        order = np.argsort(x)
        return x[order], (curve["S"][mask] * scale)[order], (curve["dS"][mask] * scale)[order]

    def pair_score(curve_a, curve_b, z, x_a):
        xa, ya, dya = transformed(curve_a, z, x_a)
        xb, yb, dyb = transformed(curve_b, z, x_a)
        if len(xa) < 3 or len(xb) < 3:
            return np.inf
        lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
        if hi <= lo:
            return np.inf
        grid = (
            np.geomspace(lo, hi, interpolation_points)
            if lo > 0
            else np.linspace(lo, hi, interpolation_points)
        )
        ya_g, yb_g = np.interp(grid, xa, ya), np.interp(grid, xb, yb)
        dya_g, dyb_g = np.interp(grid, xa, dya), np.interp(grid, xb, dyb)
        typical = max(
            np.nanmedian(np.abs(ya_g)),
            np.nanmedian(np.abs(yb_g)),
            absolute_error_floor,
        )
        floor = max(absolute_error_floor, relative_error_floor * typical)
        return float(np.mean((ya_g - yb_g) ** 2 / (dya_g**2 + dyb_g**2 + floor**2)))

    def score(z, x_a, curve_set=curves):
        scores = [
            pair_score(curve_set[i], curve_set[j], z, x_a)
            for i in range(len(curve_set))
            for j in range(i + 1, len(curve_set))
        ]
        finite = [value for value in scores if np.isfinite(value)]
        return float(np.mean(finite)) if finite else np.inf

    bounds = [z_bounds, x_a_bounds] if fixed_x_a is None else [z_bounds]

    def objective(params, curve_set=curves):
        if fixed_x_a is None:
            return score(float(params[0]), float(params[1]), curve_set)
        return score(float(params[0]), float(fixed_x_a), curve_set)

    global_fit = differential_evolution(
        objective,
        bounds=bounds,
        seed=12345,
        polish=False,
        updating="immediate",
        workers=1,
    )
    local_fit = minimize(
        objective,
        global_fit.x,
        method="L-BFGS-B",
        bounds=bounds,
    )
    best = (
        local_fit.x
        if local_fit.success and local_fit.fun <= global_fit.fun
        else global_fit.x
    )
    best_z = float(best[0])
    best_x_a = float(best[1]) if fixed_x_a is None else float(fixed_x_a)
    best_score = score(best_z, best_x_a)

    rng = np.random.default_rng(bootstrap_seed)
    estimates: list[np.ndarray] = []
    if bootstrap_samples >= 2:
        initial = np.array([best_z, best_x_a]) if fixed_x_a is None else np.array([best_z])
        for index in range(bootstrap_samples):
            synthetic_curves = []
            for curve in curves:
                synthetic = dict(curve)
                synthetic["S"] = np.clip(
                    rng.normal(curve["S"], np.maximum(curve["dS"], 0.0)),
                    0.0,
                    np.log(2.0),
                )
                synthetic_curves.append(synthetic)
            result = minimize(
                lambda params: objective(params, synthetic_curves),
                initial,
                method="Powell",
                bounds=bounds,
                options={
                    "maxiter": bootstrap_maxiter,
                    "xtol": 1e-4,
                    "ftol": 1e-4,
                },
            )
            if result.success and np.all(np.isfinite(result.x)):
                estimates.append(np.asarray(result.x, dtype=float))
            if (index + 1) % 25 == 0 or index + 1 == bootstrap_samples:
                print(f"Bootstrap fits: {index + 1}/{bootstrap_samples}")

    if len(estimates) >= 2:
        estimates_array = np.vstack(estimates)
        z_stderr = float(np.std(estimates_array[:, 0], ddof=1))
        x_a_stderr = (
            float(np.std(estimates_array[:, 1], ddof=1))
            if fixed_x_a is None
            else 0.0
        )
    else:
        z_stderr = x_a_stderr = np.nan
        if bootstrap_samples >= 2:
            warnings.warn("Too few successful bootstrap fits to estimate errors.")

    print(f"z   = {best_z:.6f} ± {z_stderr:.6f}")
    print(
        f"x_A = {best_x_a:.6f} ± {x_a_stderr:.6f}"
        if fixed_x_a is None
        else f"x_A = {best_x_a:.6f} fixed"
    )
    print(f"collapse score = {best_score:.6g}")

    colors = _color_map_by_size(sizes, cmap)
    fig, (ax_raw, ax_collapse) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )
    loaded_t_max = max(np.max(curve["t"]) for curve in curves)
    fit_upper = min(fit_t_max, loaded_t_max) if fit_t_max is not None else loaded_t_max
    ax_raw.axvspan(
        fit_t_min,
        fit_upper,
        facecolor="lightsteelblue",
        alpha=0.16,
        edgecolor="none",
        label="fit window",
        zorder=0,
    )

    for curve in curves:
        color, size = colors[curve["L"]], curve["L"]
        kwargs = dict(
            color=color,
            marker="o",
            markersize=3.8,
            linewidth=1.25,
            label=rf"$L={size}$",
            zorder=2,
        )
        if show_errorbars:
            ax_raw.errorbar(
                curve["t"],
                curve["S"],
                yerr=curve["dS"],
                elinewidth=1.0,
                capsize=capsize,
                **kwargs,
            )
        else:
            ax_raw.plot(curve["t"], curve["S"], **kwargs)

        mask = fit_mask(curve)
        scale = float(size) ** best_x_a
        excluded = ~mask
        if np.any(excluded):
            ax_collapse.plot(
                curve["t"][excluded] / float(size) ** best_z,
                curve["S"][excluded] * scale,
                linestyle="none",
                marker="o",
                markersize=3,
                color=color,
                alpha=0.20,
            )
        kwargs = dict(
            color=color,
            marker="o",
            markersize=3.8,
            linewidth=1.25,
            label=rf"$L={size}$",
        )
        if show_errorbars:
            ax_collapse.errorbar(
                curve["t"][mask] / float(size) ** best_z,
                curve["S"][mask] * scale,
                yerr=curve["dS"][mask] * scale,
                elinewidth=1.0,
                capsize=capsize,
                **kwargs,
            )
        else:
            ax_collapse.plot(
                curve["t"][mask] / float(size) ** best_z,
                curve["S"][mask] * scale,
                **kwargs,
            )

    ax_raw.set_xlabel(r"Time $t$")
    ax_raw.set_ylabel(r"Ancilla entropy $\overline{S_A}$")
    ax_raw.set_title("Critical ancilla dynamics")
    ax_raw.set_xscale(raw_xscale)
    ax_raw.grid(alpha=0.25)
    handles, labels = ax_raw.get_legend_handles_labels()
    order = [i for i, label in enumerate(labels) if label == "fit window"] + [
        i for i, label in enumerate(labels) if label != "fit window"
    ]
    ax_raw.legend([handles[i] for i in order], [labels[i] for i in order])

    ax_collapse.set_xlabel(r"Scaled time $t/L^z$")
    ax_collapse.set_ylabel(
        r"$\overline{S_A}$"
        if abs(best_x_a) < 1e-12
        else r"$L^{x_A}\overline{S_A}$"
    )
    ax_collapse.set_title("Finite-size scaling collapse")
    ax_collapse.set_xscale(collapse_xscale)
    ax_collapse.grid(alpha=0.25)
    ax_collapse.legend()
    x_a_text = (
        rf"$x_A={best_x_a:.4f}\pm{x_a_stderr:.4f}$"
        if fixed_x_a is None
        else rf"$x_A={best_x_a:.4f}$ fixed"
    )
    ax_collapse.text(
        0.97,
        0.97,
        r"$\overline{S_A}=L^{-x_A}F(t/L^z)$"
        + "\n"
        + rf"$z={best_z:.4f}\pm{z_stderr:.4f}$"
        + "\n"
        + x_a_text
        + "\n"
        + rf"score $={best_score:.3g}$",
        transform=ax_collapse.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    fig.suptitle("Ancilla probe at the measurement-induced critical point", fontsize=14)
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_raw, ax_collapse),
        "z": best_z,
        "z_stderr": z_stderr,
        "x_a": best_x_a,
        "x_a_stderr": x_a_stderr,
        "score": best_score,
        "bootstrap_successes": len(estimates),
        "curves": curves,
        "global_fit": global_fit,
        "local_fit": local_fit,
    }


# ---------------------------------------------------------------------------
# Two-probe bulk exponent collapse
# ---------------------------------------------------------------------------


def _probe2_metric_spec(metric: str) -> dict[str, Any]:
    """Resolve a two-probe observable name and its CSV/plot metadata."""
    key = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "i": "mi",
        "mutual_information": "mi",
        "mutualinfo": "mi",
        "n": "negativity",
        "neg": "negativity",
        "mn": "negativity",
        "log_neg": "log_negativity",
        "logneg": "log_negativity",
        "logarithmic_negativity": "log_negativity",
        "ln_negativity": "log_negativity",
    }
    key = aliases.get(key, key)

    specs = {
        "mi": {
            "mean_columns": [
                "I_mean",
                "i_mean",
                "mi_mean",
                "mutual_information_mean",
            ],
            "stderr_columns": [
                "I_stderr",
                "i_stderr",
                "mi_stderr",
                "mutual_information_stderr",
                "stderr",
            ],
            "summary_name": "I",
            "raw_ylabel": r"Mutual information $I(A:B)$ [nats]",
            "raw_title": "Two-probe mutual information",
            "collapse_ylabel": r"$L^{\eta}I(A:B)$",
            "ansatz": r"$I(A:B)=L^{-\eta}g((t-2L)/L)$",
            "suptitle": (
                "Bulk two-probe mutual information at fixed measurement rate"
            ),
            "bootstrap_upper": 2.0 * np.log(2.0),
        },
        "negativity": {
            "mean_columns": [
                "negativity_mean",
                "neg_mean",
                "N_mean",
                "n_mean",
                "mn_mean",
            ],
            "stderr_columns": [
                "negativity_stderr",
                "neg_stderr",
                "N_stderr",
                "n_stderr",
                "mn_stderr",
            ],
            "summary_name": "negativity",
            "raw_ylabel": r"Negativity $\mathcal{N}(A:B)$",
            "raw_title": "Two-probe negativity",
            "collapse_ylabel": r"$L^{\eta}\mathcal{N}(A:B)$",
            "ansatz": r"$\mathcal{N}(A:B)=L^{-\eta}g((t-2L)/L)$",
            "suptitle": "Bulk two-probe negativity at fixed measurement rate",
            "bootstrap_upper": 0.5,
        },
        "log_negativity": {
            "mean_columns": [
                "log_negativity_mean",
                "log_neg_mean",
                "logarithmic_negativity_mean",
                "E_N_mean",
                "en_mean",
            ],
            "stderr_columns": [
                "log_negativity_stderr",
                "log_neg_stderr",
                "logarithmic_negativity_stderr",
                "E_N_stderr",
                "en_stderr",
            ],
            "summary_name": "log_negativity",
            "raw_ylabel": (
                r"Logarithmic negativity $E_{\mathcal{N}}(A:B)$ [nats]"
            ),
            "raw_title": "Two-probe logarithmic negativity",
            "collapse_ylabel": r"$L^{\eta}E_{\mathcal{N}}(A:B)$",
            "ansatz": (
                r"$E_{\mathcal{N}}(A:B)="
                r"L^{-\eta}g((t-2L)/L)$"
            ),
            "suptitle": (
                "Bulk two-probe logarithmic negativity at fixed measurement rate"
            ),
            "bootstrap_upper": np.log(2.0),
        },
    }
    if key not in specs:
        raise ValueError(
            "metric must be 'mi', 'negativity', or 'log_negativity'."
        )
    return {"key": key, **specs[key]}


def _load_probe2_curve(
    path: Path,
    *,
    l_by_file: Mapping[str, int] | None,
    x_load: tuple[float | None, float | None],
    metric_spec: Mapping[str, Any],
) -> dict[str, Any]:
    size = _parse_size(path, l_by_file)
    df = pd.read_csv(path)
    t_col = _find_column(df, ["t", "time", "timestep"])
    mean_col = _find_column(df, metric_spec["mean_columns"])
    stderr_col = _find_column(df, metric_spec["stderr_columns"])
    scaled_col = _find_column(df, ["scaled_time", "scaled_t", "x"], required=False)
    clean = pd.DataFrame(
        {
            "t": pd.to_numeric(df[t_col], errors="coerce"),
            "value": pd.to_numeric(df[mean_col], errors="coerce"),
            "dvalue": pd.to_numeric(df[stderr_col], errors="coerce"),
        }
    )
    if scaled_col is not None:
        clean["x_csv"] = pd.to_numeric(df[scaled_col], errors="coerce")
    clean = clean.dropna(subset=["t", "value", "dvalue"])
    clean = clean.sort_values("t").drop_duplicates("t", keep="last")
    clean["x"] = (clean["t"] - 2.0 * size) / float(size)

    if "x_csv" in clean:
        valid = np.isfinite(clean["x_csv"])
        if np.any(valid):
            discrepancy = np.max(
                np.abs(clean.loc[valid, "x_csv"] - clean.loc[valid, "x"])
            )
            if discrepancy > 1e-9:
                warnings.warn(
                    f"{path}: scaled_time differs from (t-2L)/L by "
                    f"{discrepancy:.3g}; using the recomputed value."
                )

    x_min, x_max = x_load
    if x_min is not None:
        clean = clean.loc[clean["x"] >= x_min]
    if x_max is not None:
        clean = clean.loc[clean["x"] <= x_max]
    if clean.empty:
        raise ValueError(f"No valid rows remain in {path} after load cutoffs.")
    if np.any(clean["dvalue"] < 0):
        warnings.warn(
            f"{path}: negative standard errors found; using absolute values."
        )
        clean["dvalue"] = np.abs(clean["dvalue"])
    if np.any(clean["x"] < -1e-12):
        warnings.warn(f"{path}: contains points before t0=2L.")

    return {
        "path": path,
        "L": size,
        "p": _parse_p(path),
        "metric": metric_spec["key"],
        "t": clean["t"].to_numpy(dtype=float),
        "x": clean["x"].to_numpy(dtype=float),
        "value": clean["value"].to_numpy(dtype=float),
        "dvalue": clean["dvalue"].to_numpy(dtype=float),
    }


def probe2_collapse(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    l_by_file: Mapping[str, int] | None = None,
    metric: str = "mi",
    x_load: tuple[float | None, float | None] = (None, None),
    fit_x: tuple[float, float | None] = (0.25, 8.0),
    fit_i_min: float = 1e-8,
    eta_bounds: tuple[float, float] = (0.0, 2.0),
    interpolation_points: int = 250,
    relative_error_floor: float = 0.01,
    absolute_error_floor: float = 1e-8,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 24680,
    bootstrap_xtol: float = 1e-5,
    raw_yscale: str = "linear",
    collapse_yscale: str = "linear",
    show_errorbars: bool = True,
    capsize: float = 2,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (13, 5),
    dpi: int = 130,
    show_summary: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    r"""Fit ``O(L,t)=L^(-eta)g((t-2L)/L)`` for two ancilla probes.

    ``metric`` selects ``"mi"``, ``"negativity"``, or
    ``"log_negativity"``. The same one-parameter bulk-exponent ansatz and
    collapse objective are used for all three observables. ``fit_i_min`` is
    retained for notebook compatibility and acts as the minimum selected
    observable value, regardless of metric.
    """
    metric_spec = _probe2_metric_spec(metric)
    paths = _resolve_files(files, file_glob)
    curves = [
        _load_probe2_curve(
            path,
            l_by_file=l_by_file,
            x_load=x_load,
            metric_spec=metric_spec,
        )
        for path in paths
    ]
    curves.sort(key=lambda curve: curve["L"])
    sizes = [curve["L"] for curve in curves]
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"Duplicate system sizes found: {sizes}")
    if len(curves) < 3:
        warnings.warn("A collapse with fewer than three sizes is weakly constrained.")

    finite_ps = [curve["p"] for curve in curves if np.isfinite(curve["p"])]
    if finite_ps and not np.allclose(finite_ps, finite_ps[0], rtol=0, atol=1e-12):
        warnings.warn(f"The files contain multiple p values: {sorted(set(finite_ps))}")

    summary_name = metric_spec["summary_name"]
    summary = pd.DataFrame(
        {
            "L": sizes,
            "metric": metric_spec["key"],
            "p_from_filename": [curve["p"] for curve in curves],
            "points": [len(curve["x"]) for curve in curves],
            "x_min": [np.min(curve["x"]) for curve in curves],
            "x_max": [np.max(curve["x"]) for curve in curves],
            f"{summary_name}_max": [np.max(curve["value"]) for curve in curves],
            f"x_at_{summary_name}_max": [
                curve["x"][np.argmax(curve["value"])] for curve in curves
            ],
            "file": [str(curve["path"]) for curve in curves],
        }
    )
    if show_summary:
        try:
            from IPython.display import display

            display(summary)
        except ImportError:
            print(summary.to_string(index=False))

    fit_x_min, fit_x_max = fit_x

    def fit_mask(curve):
        mask = (
            np.isfinite(curve["x"])
            & np.isfinite(curve["value"])
            & np.isfinite(curve["dvalue"])
            & (curve["x"] >= fit_x_min)
            & (curve["value"] > fit_i_min)
        )
        if fit_x_max is not None:
            mask &= curve["x"] <= fit_x_max
        return mask

    def pair_score(curve_a, curve_b, eta):
        def transformed(curve):
            mask = fit_mask(curve)
            scale = float(curve["L"]) ** eta
            order = np.argsort(curve["x"][mask])
            return (
                curve["x"][mask][order],
                (curve["value"][mask] * scale)[order],
                (curve["dvalue"][mask] * scale)[order],
            )

        xa, ya, dya = transformed(curve_a)
        xb, yb, dyb = transformed(curve_b)
        if len(xa) < 3 or len(xb) < 3:
            return np.inf
        lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
        if hi <= lo:
            return np.inf
        grid = np.linspace(lo, hi, interpolation_points)
        ya_g, yb_g = np.interp(grid, xa, ya), np.interp(grid, xb, yb)
        dya_g, dyb_g = np.interp(grid, xa, dya), np.interp(grid, xb, dyb)
        typical = max(
            np.nanmedian(np.abs(ya_g)),
            np.nanmedian(np.abs(yb_g)),
            absolute_error_floor,
        )
        floor = max(absolute_error_floor, relative_error_floor * typical)
        return float(
            np.mean((ya_g - yb_g) ** 2 / (dya_g**2 + dyb_g**2 + floor**2))
        )

    def score(eta, curve_set=curves):
        values = [
            pair_score(curve_set[i], curve_set[j], eta)
            for i in range(len(curve_set))
            for j in range(i + 1, len(curve_set))
        ]
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else np.inf

    fit_result = minimize_scalar(
        score,
        bounds=eta_bounds,
        method="bounded",
        options={"xatol": 1e-7},
    )
    if not fit_result.success or not np.isfinite(fit_result.fun):
        raise RuntimeError(f"Eta fit failed: {fit_result.message}")
    eta = float(fit_result.x)
    best_score = float(fit_result.fun)
    if min(eta - eta_bounds[0], eta_bounds[1] - eta) < 0.01 * (
        eta_bounds[1] - eta_bounds[0]
    ):
        warnings.warn("eta is close to an eta_bounds boundary; widen the bounds.")

    rng = np.random.default_rng(bootstrap_seed)
    estimates: list[float] = []
    if bootstrap_samples >= 2:
        for index in range(bootstrap_samples):
            synthetic_curves = []
            for curve in curves:
                synthetic = dict(curve)
                synthetic["value"] = np.clip(
                    rng.normal(
                        curve["value"],
                        np.maximum(curve["dvalue"], 0.0),
                    ),
                    0.0,
                    metric_spec["bootstrap_upper"],
                )
                synthetic_curves.append(synthetic)
            result = minimize_scalar(
                lambda value: score(value, synthetic_curves),
                bounds=eta_bounds,
                method="bounded",
                options={"xatol": bootstrap_xtol},
            )
            if result.success and np.isfinite(result.x) and np.isfinite(result.fun):
                estimates.append(float(result.x))
            if (index + 1) % 25 == 0 or index + 1 == bootstrap_samples:
                print(f"Bootstrap fits: {index + 1}/{bootstrap_samples}")

    if len(estimates) >= 2:
        estimate_array = np.asarray(estimates)
        eta_stderr = float(np.std(estimate_array, ddof=1))
        eta_p16, eta_p84 = map(float, np.percentile(estimate_array, [16, 84]))
    else:
        eta_stderr = eta_p16 = eta_p84 = np.nan
        if bootstrap_samples >= 2:
            warnings.warn("Too few successful bootstrap fits to estimate eta uncertainty.")

    print(f"metric = {metric_spec['key']}")
    print(f"eta = {eta:.6f} ± {eta_stderr:.6f}")
    if np.isfinite(eta_p16):
        print(f"bootstrap 68% interval = [{eta_p16:.6f}, {eta_p84:.6f}]")
    print(f"collapse score = {best_score:.6g}")

    colors = _color_map_by_size(sizes, cmap)
    fig, (ax_raw, ax_collapse) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )
    for curve in curves:
        size, color = curve["L"], colors[curve["L"]]
        kwargs = dict(
            color=color,
            marker="o",
            markersize=3.6,
            linewidth=1.2,
            label=rf"$L={size}$",
        )
        if show_errorbars:
            ax_raw.errorbar(
                curve["x"],
                curve["value"],
                yerr=curve["dvalue"],
                elinewidth=0.9,
                capsize=capsize,
                **kwargs,
            )
        else:
            ax_raw.plot(curve["x"], curve["value"], **kwargs)

        mask = fit_mask(curve)
        scale = float(size) ** eta
        excluded = ~mask
        if np.any(excluded):
            ax_collapse.plot(
                curve["x"][excluded],
                curve["value"][excluded] * scale,
                linestyle="none",
                marker="o",
                markersize=3,
                color=color,
                alpha=0.20,
            )
        if show_errorbars:
            ax_collapse.errorbar(
                curve["x"][mask],
                curve["value"][mask] * scale,
                yerr=curve["dvalue"][mask] * scale,
                elinewidth=0.9,
                capsize=capsize,
                **kwargs,
            )
        else:
            ax_collapse.plot(
                curve["x"][mask],
                curve["value"][mask] * scale,
                **kwargs,
            )

    ax_raw.axvline(0.0, linewidth=1, linestyle="--", alpha=0.6)
    fit_upper = (
        fit_x_max
        if fit_x_max is not None
        else max(curve["x"].max() for curve in curves)
    )
    ax_raw.axvspan(fit_x_min, fit_upper, alpha=0.08, label="fit window")
    ax_raw.set_xlabel(r"Scaled time $x=(t-2L)/L$")
    ax_raw.set_ylabel(metric_spec["raw_ylabel"])
    ax_raw.set_title(metric_spec["raw_title"])
    ax_raw.set_yscale(raw_yscale)
    ax_raw.grid(alpha=0.25)
    ax_raw.legend()

    ax_collapse.set_xlabel(r"Scaled time $x=(t-2L)/L$")
    ax_collapse.set_ylabel(metric_spec["collapse_ylabel"])
    ax_collapse.set_title("Bulk-exponent scaling collapse")
    ax_collapse.set_yscale(collapse_yscale)
    ax_collapse.grid(alpha=0.25)
    ax_collapse.legend()
    eta_text = (
        rf"$\eta={eta:.4f}\pm{eta_stderr:.4f}$"
        if np.isfinite(eta_stderr)
        else rf"$\eta={eta:.4f}$"
    )
    ax_collapse.text(
        0.97,
        0.02,
        metric_spec["ansatz"]
        + "\n"
        + eta_text
        + "\n"
        + rf"score $={best_score:.3g}$",
        transform=ax_collapse.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    fig.suptitle(metric_spec["suptitle"], fontsize=14)
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_raw, ax_collapse),
        "metric": metric_spec["key"],
        "eta": eta,
        "eta_stderr": eta_stderr,
        "eta_p16": eta_p16,
        "eta_p84": eta_p84,
        "score": best_score,
        "bootstrap_successes": len(estimates),
        "summary": summary,
        "curves": curves,
        "fit_result": fit_result,
    }

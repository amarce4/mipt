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
probe_distance_collapse(...) Plot and collapse mode-4 distance-resolved probes.
probe_anisotropy(...)  Plot mode-3 space/time correlators and estimate alpha.
free_energy_ceff(...)  Recreate Fig. 1(b) and extract the effective central charge.
free_energy_equilibration(...) Recreate Supplementary Fig. S1(a,b).
probe4_collapse(...)   Fit and plot four-ancilla I2/I3/I4 collapses.
probe_pc_collapse(...) Fit fixed-t p scans for one/two/four probes.
probe_entropy_map(...) Plot the p-t S_Q heatmap and selected-p time curves.

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
    "probe_distance_collapse",
    "probe_anisotropy",
    "free_energy_ceff",
    "free_energy_equilibration",
    "probe4_collapse",
    "probe_pc_collapse",
    "probe_entropy_map",
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


def _mi_unit_spec(mi_units: str) -> tuple[str, float]:
    """Return the canonical MI unit and the nats-to-unit scale factor."""
    key = str(mi_units).strip().lower()
    key = {"nat": "nats", "bit": "bits"}.get(key, key)
    if key not in {"nats", "bits"}:
        raise ValueError("mi_units must be 'nats' or 'bits'.")
    return key, 1.0 if key == "nats" else 1.0 / np.log(2.0)


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
                        rf"$\pm ${slope_error:.{precision}f}, "
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
    mi_units: str = "nats",
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
    ``mi_units`` converts the MI panel and fitted prefactor between ``"nats"``
    and ``"bits"``; it does not change the fitted exponent.
    """
    if min_relative_error <= 0.0:
        raise ValueError("min_relative_error must be positive.")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    if distance_round < 0:
        raise ValueError("distance_round must be non-negative.")
    mi_units, mi_scale = _mi_unit_spec(mi_units)

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
        data[size]["mi"].loc[:, ["mean", "stderr"]] *= mi_scale
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
                rf"{row['alpha']:.3g}\pm {row['alpha_stderr']:.2g}$"
            ),
            zorder=5,
        )

    ax_mi.set_xlabel(r"Effective chord distance $d$")
    ax_mi.set_ylabel(rf"Two-party mutual information $I_2$ [{mi_units}]")
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
        "mi_units": mi_units,
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
    mi_units: str = "nats",
    error_floor: float = 1e-12,
    curvature_step: float = 1e-3,
    colors: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (10, 8),
    show: bool = True,
) -> dict[str, Any]:
    """Fit p_c and nu for the TMI crossing using pyfssa.

    ``mi_units`` selects ``"nats"`` or ``"bits"`` for TMI values and errors.
    """
    try:
        import fssa
    except ImportError as exc:
        raise ImportError(
            "tmi_collapse requires pyfssa (the import name is 'fssa')."
        ) from exc

    # Older pyfssa releases still reference np.int.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    mi_units, mi_scale = _mi_unit_spec(mi_units)

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
        mean_arr.append(mean * mi_scale)
        stderr_arr.append(stderr * mi_scale)

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
    ax_raw.set_ylabel(rf"TMI $\overline{{I}}_3$ [{mi_units}]")
    ax_raw.set_title("Raw TMI crossing")
    ax_raw.legend()

    ax_collapse.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_collapse.set_xlabel(r"Scaled variable $(p-p_c)L^{1/\nu}$")
    ax_collapse.set_ylabel(
        rf"Scaled TMI $L^{{-\zeta/\nu}}\overline{{I}}_3$ [{mi_units}]"
        if fit_zeta
        else rf"TMI $\overline{{I}}_3$ [{mi_units}]"
    )
    ax_collapse.set_title("Finite-size scaling collapse")
    ax_collapse.legend()

    text = (
        rf"$p_c={pc:.4g}\pm {dpc:.1g}$" "\n"
        rf"$\nu={nu:.3g}\pm {dnu:.1g}$" "\n"
        rf"$S={quality:.3g}$" "\n"
        + (
            rf"$\zeta={zeta:.3g}\pm {dzeta:.1g}$"
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
        "mi_units": mi_units,
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
        rf"$x_A={best_x_a:.4f}\pm {x_a_stderr:.4f}$"
        if fixed_x_a is None
        else rf"$x_A={best_x_a:.4f}$ fixed"
    )
    ax_collapse.text(
        0.97,
        0.97,
        r"$\overline{S_A}=L^{-x_A}F(t/L^z)$"
        + "\n"
        + rf"$z={best_z:.4f}\pm {z_stderr:.4f}$"
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
    mi_units: str = "nats",
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
    observable value, regardless of metric. ``mi_units`` applies only when
    ``metric="mi"``.
    """
    metric_spec = _probe2_metric_spec(metric)
    mi_units, mi_scale = _mi_unit_spec(mi_units)
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
    if metric_spec["key"] == "mi":
        metric_spec = dict(metric_spec)
        metric_spec["bootstrap_upper"] *= mi_scale
        for curve in curves:
            curve["value"] *= mi_scale
            curve["dvalue"] *= mi_scale
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
    ax_raw.set_ylabel(
        rf"Mutual information $I(A:B)$ [{mi_units}]"
        if metric_spec["key"] == "mi"
        else metric_spec["raw_ylabel"]
    )
    ax_raw.set_title(metric_spec["raw_title"])
    ax_raw.set_yscale(raw_yscale)
    ax_raw.grid(alpha=0.25)
    ax_raw.legend()

    ax_collapse.set_xlabel(r"Scaled time $x=(t-2L)/L$")
    ax_collapse.set_ylabel(
        metric_spec["collapse_ylabel"] + f" [{mi_units}]"
        if metric_spec["key"] == "mi"
        else metric_spec["collapse_ylabel"]
    )
    ax_collapse.set_title("Bulk-exponent scaling collapse")
    ax_collapse.set_yscale(collapse_yscale)
    ax_collapse.grid(alpha=0.25)
    ax_collapse.legend()
    eta_text = (
        rf"$\eta={eta:.4f}\pm {eta_stderr:.4f}$"
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
        "mi_units": mi_units,
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


# ---------------------------------------------------------------------------
# Distance-resolved two-probe dynamics
# ---------------------------------------------------------------------------


def _load_probe_distance_curves(
    path: Path,
    *,
    metric_spec: Mapping[str, Any],
    distance_load: tuple[int, int | None],
) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path}: mode-4 CSV is empty.")

    size = int(_constant_csv_value(df, ["N", "L", "system_N"], path, integer=True))
    p = float(_constant_csv_value(df, ["p"], path))
    t_eq = int(
        _constant_csv_value(df, ["t_eq", "teq"], path, integer=True)
    )
    distance_col = _find_column(df, ["distance", "r", "delta_x"])
    tau_col = _find_column(df, ["tau", "elapsed_time", "readout_time"])
    chord_col = _find_column(
        df, ["chord_length", "ell_r", "chord"], required=False
    )
    mean_col = _find_column(df, metric_spec["mean_columns"])
    stderr_col = _find_column(df, metric_spec["stderr_columns"])
    positive_col = _find_column(
        df,
        ["negativity_positive_fraction", "positive_negativity_fraction"],
        required=False,
    )

    clean = pd.DataFrame(
        {
            "distance": pd.to_numeric(df[distance_col], errors="coerce"),
            "tau": pd.to_numeric(df[tau_col], errors="coerce"),
            "value": pd.to_numeric(df[mean_col], errors="coerce"),
            "dvalue": pd.to_numeric(df[stderr_col], errors="coerce"),
        }
    )
    if chord_col is not None:
        clean["chord_csv"] = pd.to_numeric(df[chord_col], errors="coerce")
    if positive_col is not None:
        clean["positive_fraction"] = pd.to_numeric(
            df[positive_col], errors="coerce"
        )
    clean = clean.dropna(subset=["distance", "tau", "value", "dvalue"])
    if clean.empty:
        raise ValueError(f"{path}: no finite mode-4 rows were found.")

    rounded_distance = np.rint(clean["distance"])
    if not np.allclose(clean["distance"], rounded_distance, rtol=0.0, atol=1e-9):
        raise ValueError(f"{path}: distance must be integer-valued.")
    clean["distance"] = rounded_distance.astype(int)
    if np.any((clean["distance"] < 1) | (clean["distance"] > size // 2)):
        raise ValueError(f"{path}: distance lies outside 1,...,floor(N/2).")
    if np.any(clean["tau"] < 0):
        raise ValueError(f"{path}: tau must be non-negative.")
    if np.any(clean["dvalue"] < 0):
        warnings.warn(f"{path}: negative standard errors found; using magnitudes.")
        clean["dvalue"] = np.abs(clean["dvalue"])

    distance_min, distance_max = distance_load
    if distance_min < 1:
        raise ValueError("distance_load minimum must be at least 1.")
    clean = clean.loc[clean["distance"] >= distance_min]
    if distance_max is not None:
        clean = clean.loc[clean["distance"] <= distance_max]
    if clean.empty:
        raise ValueError(f"{path}: no rows remain after distance_load.")

    curves: list[dict[str, Any]] = []
    for distance, group in clean.groupby("distance", sort=True):
        group = group.sort_values("tau").drop_duplicates("tau", keep="last")
        chord = size / np.pi * np.sin(np.pi * distance / size)
        if "chord_csv" in group:
            supplied = group["chord_csv"].dropna().to_numpy(dtype=float)
            if supplied.size and not np.allclose(
                supplied, chord, rtol=1e-10, atol=1e-12
            ):
                warnings.warn(
                    f"{path}: stored chord length for r={distance} differs "
                    "from L/pi sin(pi r/L); using the recomputed value."
                )
        curve = {
            "path": path,
            "L": size,
            "p": p,
            "t_eq": t_eq,
            "distance": int(distance),
            "chord": float(chord),
            "tau": group["tau"].to_numpy(dtype=float),
            "value": group["value"].to_numpy(dtype=float),
            "dvalue": group["dvalue"].to_numpy(dtype=float),
        }
        if "positive_fraction" in group:
            curve["positive_fraction"] = group[
                "positive_fraction"
            ].to_numpy(dtype=float)
        curves.append(curve)
    return curves


def probe_distance_collapse(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    metric: str = "mi",
    mi_units: str = "nats",
    eta: float | None = None,
    z: float | None = 1.0,
    alpha: float = 1.0,
    eta_bounds: tuple[float, float] = (0.0, 2.0),
    z_bounds: tuple[float, float] = (0.5, 2.0),
    distance_load: tuple[int, int | None] = (1, None),
    fit_distance: tuple[int, int | None] = (2, None),
    fit_tau: tuple[float, float | None] = (1.0, None),
    fit_x: tuple[float | None, float | None] = (None, None),
    fit_value_min: float = 1e-10,
    interpolation_points: int = 250,
    relative_error_floor: float = 0.01,
    absolute_error_floor: float = 1e-10,
    bootstrap_samples: int = 100,
    bootstrap_seed: int = 97531,
    raw_yscale: str = "linear",
    collapse_yscale: str = "linear",
    show_errorbars: bool = True,
    errorbar_points: int = 80,
    capsize: float = 2,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (13, 5),
    dpi: int = 130,
    show_summary: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    r"""Plot and fit mode-4 distance-resolved two-probe data.

    The collapse uses

    ``O(tau,r,L) = ell_r**(-eta) G(alpha*tau/ell_r**z)``,

    with ``ell_r=(L/pi) sin(pi*r/L)``. A numeric ``eta`` or ``z`` fixes that
    parameter; ``None`` fits it. The CFT default is therefore ``z=1`` with
    ``eta`` fitted. ``alpha`` is always supplied/fixed because a free global
    rescaling of the unknown scaling function's argument is not identifiable.
    ``fit_distance=(2,None)`` leaves the microscopic ``r=1`` curve visible but
    excludes it from the default fit.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    if eta is not None and (not np.isfinite(eta) or eta < 0):
        raise ValueError("eta must be non-negative or None.")
    if z is not None and (not np.isfinite(z) or z <= 0):
        raise ValueError("z must be positive or None.")
    if interpolation_points < 3:
        raise ValueError("interpolation_points must be at least 3.")
    if errorbar_points < 1:
        raise ValueError("errorbar_points must be positive.")

    metric_spec = _probe2_metric_spec(metric)
    mi_units, mi_scale = _mi_unit_spec(mi_units)
    paths = _resolve_files(files, file_glob)
    curves = [
        curve
        for path in paths
        for curve in _load_probe_distance_curves(
            path,
            metric_spec=metric_spec,
            distance_load=distance_load,
        )
    ]
    if metric_spec["key"] == "mi":
        metric_spec = dict(metric_spec)
        metric_spec["bootstrap_upper"] *= mi_scale
        for curve in curves:
            curve["value"] *= mi_scale
            curve["dvalue"] *= mi_scale
    curves.sort(key=lambda curve: (curve["L"], curve["distance"], str(curve["path"])))
    if len(curves) < 2:
        raise ValueError("At least two distance curves are required for a collapse.")

    finite_ps = sorted({curve["p"] for curve in curves if np.isfinite(curve["p"])})
    if len(finite_ps) > 1:
        warnings.warn(f"The files contain multiple p values: {finite_ps}")
    fit_distance_min, fit_distance_max = fit_distance
    fit_tau_min, fit_tau_max = fit_tau
    fit_x_min, fit_x_max = fit_x
    if fit_distance_min < 1:
        raise ValueError("fit_distance minimum must be at least 1.")
    if fit_tau_min < 0:
        raise ValueError("fit_tau minimum must be non-negative.")

    summary = pd.DataFrame(
        {
            "L": [curve["L"] for curve in curves],
            "r": [curve["distance"] for curve in curves],
            "chord_length": [curve["chord"] for curve in curves],
            "p": [curve["p"] for curve in curves],
            "t_eq": [curve["t_eq"] for curve in curves],
            "tau_max": [np.max(curve["tau"]) for curve in curves],
            "points": [len(curve["tau"]) for curve in curves],
            "file": [str(curve["path"]) for curve in curves],
        }
    )
    if show_summary:
        try:
            from IPython.display import display

            display(summary)
        except ImportError:
            print(summary.to_string(index=False))

    def transformed(curve, eta_value, z_value):
        x = alpha * curve["tau"] / curve["chord"] ** z_value
        mask = (
            np.isfinite(x)
            & np.isfinite(curve["value"])
            & np.isfinite(curve["dvalue"])
            & (curve["tau"] >= fit_tau_min)
            & (curve["value"] > fit_value_min)
            & (curve["distance"] >= fit_distance_min)
        )
        if fit_tau_max is not None:
            mask &= curve["tau"] <= fit_tau_max
        if fit_distance_max is not None:
            mask &= curve["distance"] <= fit_distance_max
        if fit_x_min is not None:
            mask &= x >= fit_x_min
        if fit_x_max is not None:
            mask &= x <= fit_x_max
        order = np.argsort(x[mask])
        scale = curve["chord"] ** eta_value
        return (
            x[mask][order],
            (curve["value"][mask] * scale)[order],
            (curve["dvalue"][mask] * scale)[order],
            mask,
        )

    def pair_score(curve_a, curve_b, eta_value, z_value):
        xa, ya, dya, _ = transformed(curve_a, eta_value, z_value)
        xb, yb, dyb, _ = transformed(curve_b, eta_value, z_value)
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

    def score_values(eta_value, z_value, curve_set=curves):
        scores = [
            pair_score(curve_set[i], curve_set[j], eta_value, z_value)
            for i in range(len(curve_set))
            for j in range(i + 1, len(curve_set))
        ]
        finite = [value for value in scores if np.isfinite(value)]
        return float(np.mean(finite)) if finite else np.inf

    eta_fixed, z_fixed = eta is not None, z is not None

    def fit_parameters(curve_set, *, bootstrap=False):
        if eta_fixed and z_fixed:
            return float(eta), float(z), score_values(float(eta), float(z), curve_set)
        if not eta_fixed and z_fixed:
            result = minimize_scalar(
                lambda trial: score_values(trial, float(z), curve_set),
                bounds=eta_bounds,
                method="bounded",
                options={"xatol": 1e-5 if bootstrap else 1e-7},
            )
            return float(result.x), float(z), float(result.fun)
        if eta_fixed and not z_fixed:
            result = minimize_scalar(
                lambda trial: score_values(float(eta), trial, curve_set),
                bounds=z_bounds,
                method="bounded",
                options={"xatol": 1e-5 if bootstrap else 1e-7},
            )
            return float(eta), float(result.x), float(result.fun)
        if bootstrap:
            result = minimize(
                lambda values: score_values(values[0], values[1], curve_set),
                x0=np.array([best_eta, best_z]),
                method="Powell",
                bounds=[eta_bounds, z_bounds],
                options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 400},
            )
            return float(result.x[0]), float(result.x[1]), float(result.fun)
        result = differential_evolution(
            lambda values: score_values(values[0], values[1], curve_set),
            bounds=[eta_bounds, z_bounds],
            seed=bootstrap_seed,
            polish=True,
            tol=1e-7,
        )
        return float(result.x[0]), float(result.x[1]), float(result.fun)

    best_eta, best_z, best_score = fit_parameters(curves)
    if not np.isfinite(best_score):
        raise RuntimeError(
            "No fitted curve pair has at least three overlapping scaled-time "
            "points. Widen fit_x/fit_tau or include more distances."
        )

    estimates: list[tuple[float, float]] = []
    if bootstrap_samples >= 2 and (not eta_fixed or not z_fixed):
        rng = np.random.default_rng(bootstrap_seed)
        for index in range(bootstrap_samples):
            synthetic_curves = []
            for curve in curves:
                synthetic = dict(curve)
                synthetic["value"] = np.clip(
                    rng.normal(curve["value"], np.maximum(curve["dvalue"], 0.0)),
                    0.0,
                    metric_spec["bootstrap_upper"],
                )
                synthetic_curves.append(synthetic)
            trial_eta, trial_z, trial_score = fit_parameters(
                synthetic_curves, bootstrap=True
            )
            if np.all(np.isfinite([trial_eta, trial_z, trial_score])):
                estimates.append((trial_eta, trial_z))
            if (index + 1) % 25 == 0 or index + 1 == bootstrap_samples:
                print(f"Bootstrap fits: {index + 1}/{bootstrap_samples}")

    estimate_array = np.asarray(estimates, dtype=float)
    eta_stderr = (
        0.0
        if eta_fixed
        else float(np.std(estimate_array[:, 0], ddof=1))
        if len(estimates) >= 2
        else np.nan
    )
    z_stderr = (
        0.0
        if z_fixed
        else float(np.std(estimate_array[:, 1], ddof=1))
        if len(estimates) >= 2
        else np.nan
    )
    print(f"metric = {metric_spec['key']}")
    print(
        f"eta = {best_eta:.6f}"
        + (" (fixed)" if eta_fixed else f" ± {eta_stderr:.6f}")
    )
    print(
        f"z = {best_z:.6f}"
        + (" (fixed)" if z_fixed else f" ± {z_stderr:.6f}")
    )
    print(f"alpha = {alpha:.6f} (fixed)")
    print(f"collapse score = {best_score:.6g}")

    chord_values = np.asarray([curve["chord"] for curve in curves], dtype=float)
    if np.ptp(chord_values) > 0:
        color_values = 0.08 + 0.84 * (
            chord_values - np.min(chord_values)
        ) / np.ptp(chord_values)
    else:
        color_values = np.full(len(curves), 0.5)
    colors = [plt.get_cmap(cmap)(value) for value in color_values]
    one_size = len({curve["L"] for curve in curves}) == 1
    fig, (ax_raw, ax_collapse) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )
    for curve, color in zip(curves, colors):
        label = (
            rf"$r={curve['distance']},\ \ell_r={curve['chord']:.3g}$"
            if one_size
            else rf"$L={curve['L']},\ r={curve['distance']}$"
        )
        every = max(1, int(np.ceil(len(curve["tau"]) / errorbar_points)))
        plot_kwargs = {
            "color": color,
            "marker": "o",
            "markersize": 3.2,
            "linewidth": 1.1,
            "label": label,
        }
        if show_errorbars:
            ax_raw.errorbar(
                curve["tau"],
                curve["value"],
                yerr=curve["dvalue"],
                errorevery=every,
                elinewidth=0.8,
                capsize=capsize,
                **plot_kwargs,
            )
        else:
            ax_raw.plot(curve["tau"], curve["value"], **plot_kwargs)

        x, y, dy, mask = transformed(curve, best_eta, best_z)
        all_x = alpha * curve["tau"] / curve["chord"] ** best_z
        excluded = ~mask
        if np.any(excluded):
            ax_collapse.plot(
                all_x[excluded],
                curve["value"][excluded] * curve["chord"] ** best_eta,
                linestyle="none",
                marker="o",
                markersize=2.8,
                color=color,
                alpha=0.18,
            )
        collapse_kwargs = dict(plot_kwargs)
        collapse_kwargs["label"] = label
        if show_errorbars:
            collapse_every = max(1, int(np.ceil(len(x) / errorbar_points)))
            ax_collapse.errorbar(
                x,
                y,
                yerr=dy,
                errorevery=collapse_every,
                elinewidth=0.8,
                capsize=capsize,
                **collapse_kwargs,
            )
        else:
            ax_collapse.plot(x, y, **collapse_kwargs)

    observable_label = (
        rf"Mutual information $I(A:B)$ [{mi_units}]"
        if metric_spec["key"] == "mi"
        else metric_spec["raw_ylabel"]
    )
    ax_raw.set_xlabel(r"Elapsed time $\tau$")
    ax_raw.set_ylabel(observable_label)
    ax_raw.set_title("Distance-resolved probe dynamics")
    ax_raw.set_yscale(raw_yscale)
    ax_raw.grid(alpha=0.25)
    ax_raw.legend(fontsize=8, ncols=2 if len(curves) > 8 else 1)

    ax_collapse.set_xlabel(r"$\alpha\tau/\ell_r^z$")
    collapse_symbol = {
        "mi": r"I(A:B)",
        "negativity": r"\mathcal{N}(A:B)",
        "log_negativity": r"E_{\mathcal{N}}(A:B)",
    }[metric_spec["key"]]
    unit_suffix = f" [{mi_units}]" if metric_spec["key"] == "mi" else ""
    ax_collapse.set_ylabel(
        rf"$\ell_r^{{\eta}}{collapse_symbol}$" + unit_suffix
    )
    ax_collapse.set_title("Chord-length scaling collapse")
    ax_collapse.set_yscale(collapse_yscale)
    ax_collapse.grid(alpha=0.25)
    ax_collapse.legend(fontsize=8, ncols=2 if len(curves) > 8 else 1)
    eta_text = rf"$\eta={best_eta:.4f}$" + (
        " fixed"
        if eta_fixed
        else rf" $\pm {eta_stderr:.4f}$"
        if np.isfinite(eta_stderr)
        else ""
    )
    z_text = rf"$z={best_z:.4f}$" + (
        " fixed"
        if z_fixed
        else rf" $\pm {z_stderr:.4f}$"
        if np.isfinite(z_stderr)
        else ""
    )
    parameter_text = (
        eta_text
        + "\n"
        + z_text
        + "\n"
        + rf"$\alpha={alpha:.4g}$ fixed"
        + "\n"
        + rf"score $={best_score:.3g}$"
    )
    ax_collapse.text(
        0.97,
        0.03,
        parameter_text,
        transform=ax_collapse.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    fig.suptitle(
        r"$O(\tau,r,L)=\ell_r^{-\eta}"
        r"G(\alpha\tau/\ell_r^z)$",
        fontsize=14,
    )
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_raw, ax_collapse),
        "metric": metric_spec["key"],
        "mi_units": mi_units,
        "eta": best_eta,
        "eta_stderr": eta_stderr,
        "eta_fixed": eta_fixed,
        "z": best_z,
        "z_stderr": z_stderr,
        "z_fixed": z_fixed,
        "alpha": alpha,
        "score": best_score,
        "bootstrap_successes": len(estimates),
        "summary": summary,
        "curves": curves,
    }


# ---------------------------------------------------------------------------
# Two-probe spacetime anisotropy
# ---------------------------------------------------------------------------


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


def _load_probe_anisotropy_curve(
    path: Path,
    l_by_file: Mapping[str, int] | None,
) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path}: anisotropy CSV is empty.")

    n_col = _find_column(df, ["N", "L", "system_N"], required=False)
    if n_col is None:
        size = _parse_size(path, l_by_file)
    else:
        size = int(_constant_csv_value(df, [n_col], path, integer=True))
        try:
            parsed_size = _parse_size(path, l_by_file)
        except ValueError:
            parsed_size = size
        if parsed_size != size:
            raise ValueError(
                f"{path}: filename gives L={parsed_size}, but CSV gives N={size}."
            )

    delta_x = int(
        _constant_csv_value(df, ["delta_x", "dx"], path, integer=True)
    )
    delta_t = int(
        _constant_csv_value(df, ["delta_t", "dt"], path, integer=True)
    )
    p = float(_constant_csv_value(df, ["p", "measurement_rate"], path))
    tau_col = _find_column(df, ["tau", "readout_time", "elapsed_time"])
    scaled_col = _find_column(
        df,
        ["tau_over_L", "tau_over_l", "scaled_time", "scaled_tau"],
        required=False,
    )
    mean_col = _find_column(
        df, ["I_mean", "i_mean", "I2_mean", "mi_mean"]
    )
    stderr_col = _find_column(
        df, ["I_stderr", "i_stderr", "I2_stderr", "mi_stderr"]
    )
    clean = pd.DataFrame(
        {
            "tau": pd.to_numeric(df[tau_col], errors="coerce"),
            "I": pd.to_numeric(df[mean_col], errors="coerce"),
            "dI": pd.to_numeric(df[stderr_col], errors="coerce"),
        }
    )
    if scaled_col is not None:
        clean["x_csv"] = pd.to_numeric(df[scaled_col], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["tau", "I", "dI"]
    )
    clean = clean.sort_values("tau").drop_duplicates("tau", keep="last")
    if clean.empty:
        raise ValueError(f"{path}: no finite anisotropy rows were found.")
    if np.any(clean["dI"] < 0):
        warnings.warn(f"{path}: negative standard errors found; taking abs().")
        clean["dI"] = np.abs(clean["dI"])
    clean["x"] = clean["tau"] / float(size)
    if "x_csv" in clean:
        valid = np.isfinite(clean["x_csv"])
        if np.any(valid):
            discrepancy = np.max(
                np.abs(clean.loc[valid, "x_csv"] - clean.loc[valid, "x"])
            )
            if discrepancy > 1e-9:
                warnings.warn(
                    f"{path}: tau_over_L differs from tau/L by {discrepancy:.3g}; "
                    "using the recomputed value."
                )

    spatial = delta_t == 0 and delta_x * 2 == size
    temporal = delta_x == 0 and delta_t > 0
    if not (spatial or temporal):
        raise ValueError(
            f"{path}: expected spatial (delta_x=L/2, delta_t=0) or temporal "
            "(delta_x=0, delta_t>0) mode-3 geometry."
        )
    return {
        "path": path,
        "L": size,
        "p": p,
        "delta_x": delta_x,
        "delta_t": delta_t,
        "branch": "spatial" if spatial else "temporal",
        "tau": clean["tau"].to_numpy(dtype=float),
        "x": clean["x"].to_numpy(dtype=float),
        "I": clean["I"].to_numpy(dtype=float),
        "dI": clean["dI"].to_numpy(dtype=float),
    }


def _anisotropy_late_estimate(
    curve: Mapping[str, Any],
    window: tuple[float, float],
    method: str,
) -> tuple[float, float, int]:
    lo, hi = window
    mask = (
        np.isfinite(curve["x"])
        & np.isfinite(curve["I"])
        & np.isfinite(curve["dI"])
        & (curve["x"] >= lo)
        & (curve["x"] <= hi)
    )
    if not np.any(mask):
        raise ValueError(
            f"{curve['path']}: no points lie in estimate_window={window}."
        )
    x = curve["x"][mask]
    values = curve["I"][mask]
    errors = np.maximum(curve["dI"][mask], 0.0)
    if method == "last":
        index = int(np.argmax(x))
        return float(values[index]), float(errors[index]), len(values)
    if method != "mean":
        raise ValueError("estimate_method must be 'mean' or 'last'.")

    positive = errors > 0
    if np.any(positive):
        floor = max(float(np.median(errors[positive])) * 1e-6, 1e-15)
        weights = 1.0 / np.maximum(errors, floor) ** 2
        value = float(np.average(values, weights=weights))
    else:
        value = float(np.mean(values))
    # Readout times within one trajectory are correlated. Keep the typical
    # per-time standard error instead of reducing it by sqrt(number of times).
    error = float(np.sqrt(np.mean(errors**2)))
    return value, error, len(values)


def probe_anisotropy(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    l_by_file: Mapping[str, int] | None = None,
    mi_units: str = "nats",
    estimate_window: tuple[float, float] = (7.0, 8.0),
    estimate_method: str = "mean",
    full_xlim: tuple[float, float] = (0.0, 8.0),
    zoom_xlim: tuple[float, float] = (7.0, 8.0),
    full_ylim: tuple[float, float] | None = None,
    zoom_ylim: tuple[float, float] | None = None,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 24680,
    show_errorbars: bool = True,
    errorbar_points: int = 48,
    colors: Sequence[Any] = ("tab:blue", "tab:green", "tab:red"),
    figsize: tuple[float, float] = (13, 4.8),
    dpi: int = 130,
    show_summary: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    r"""Reproduce Fig. S8(b,c) and estimate the spacetime anisotropy.

    Supply one mode-3 spatial file, ``(delta_x,delta_t)=(L/2,0)``, and at
    least two temporal files, ``(0,delta_t)``, whose late-time mutual
    information brackets the spatial value. The crossing ``delta_t_star`` is
    found by linear interpolation, then

    ``alpha = log(1+sqrt(2)) / (pi * (delta_t_star/L))``.

    If the supplied temporal curves do not bracket the spatial value, the
    curves are still plotted and all crossing-derived results are returned as
    ``NaN``.

    ``estimate_method="mean"`` averages the selected late-time window while
    conservatively retaining a typical per-time standard error because points
    from the same trajectories are correlated. Use ``"last"`` to reproduce a
    final-readout-only estimate. ``mi_units`` changes the displayed MI units
    but leaves the crossing and anisotropy unchanged.
    """
    mi_units, mi_scale = _mi_unit_spec(mi_units)
    paths = _resolve_files(files, file_glob)
    curves = [_load_probe_anisotropy_curve(path, l_by_file) for path in paths]
    for curve in curves:
        curve["I"] *= mi_scale
        curve["dI"] *= mi_scale
    sizes = {curve["L"] for curve in curves}
    if len(sizes) != 1:
        raise ValueError(f"Anisotropy files must have one system size; found {sizes}.")
    size = sizes.pop()
    probabilities = np.asarray([curve["p"] for curve in curves], dtype=float)
    if not np.allclose(probabilities, probabilities[0], rtol=0.0, atol=1e-12):
        raise ValueError(
            "All anisotropy branches must use the same critical measurement rate."
        )
    spatial_curves = [curve for curve in curves if curve["branch"] == "spatial"]
    temporal_curves = sorted(
        (curve for curve in curves if curve["branch"] == "temporal"),
        key=lambda curve: curve["delta_t"],
    )
    if len(spatial_curves) != 1:
        raise ValueError(
            f"Expected exactly one spatial branch, found {len(spatial_curves)}."
        )
    if len(temporal_curves) < 2:
        raise ValueError("At least two temporal branches are required.")
    if len({curve["delta_t"] for curve in temporal_curves}) != len(temporal_curves):
        raise ValueError("Temporal files contain duplicate delta_t values.")
    if not estimate_window[0] < estimate_window[1]:
        raise ValueError("estimate_window must be strictly increasing.")

    ordered_curves = [spatial_curves[0], *temporal_curves]
    estimates = []
    for curve in ordered_curves:
        value, error, point_count = _anisotropy_late_estimate(
            curve, estimate_window, estimate_method
        )
        estimates.append(
            {
                "curve": curve,
                "value": value,
                "error": error,
                "points": point_count,
            }
        )
    spatial_estimate = estimates[0]
    temporal_estimates = estimates[1:]
    candidates = []
    for left, right in zip(temporal_estimates[:-1], temporal_estimates[1:]):
        left_difference = left["value"] - spatial_estimate["value"]
        right_difference = right["value"] - spatial_estimate["value"]
        if left_difference == 0.0 or right_difference == 0.0 or (
            left_difference * right_difference < 0.0
        ):
            span = right["curve"]["delta_t"] - left["curve"]["delta_t"]
            proximity = abs(left_difference) + abs(right_difference)
            candidates.append((span, proximity, left, right))
    alpha_constant = float(np.log1p(np.sqrt(2.0)) / np.pi)
    lower = upper = None
    delta_t_star_over_l = np.nan
    delta_t_stderr = np.nan
    alpha = np.nan
    alpha_stderr = np.nan
    alpha_interval = (np.nan, np.nan)
    alpha_bracket = (np.nan, np.nan)
    bootstrap_alpha = []
    bootstrap_delta_t = []
    if not candidates:
        values = ", ".join(
            f"delta_t/L={item['curve']['delta_t']/size:g}: I={item['value']:.6g}"
            for item in temporal_estimates
        )
        warnings.warn(
            "Temporal late-time values do not bracket the spatial value; "
            "delta_t*/L and alpha are NaN. "
            f"I_space={spatial_estimate['value']:.6g}. Temporal values: {values}."
        )
    else:
        _, _, lower, upper = min(candidates, key=lambda item: (item[0], item[1]))
        x_lower = lower["curve"]["delta_t"] / float(size)
        x_upper = upper["curve"]["delta_t"] / float(size)
        denominator = upper["value"] - lower["value"]
        if denominator == 0.0:
            raise ValueError("The bracketing temporal estimates are equal.")
        fraction = (spatial_estimate["value"] - lower["value"]) / denominator
        delta_t_star_over_l = float(x_lower + fraction * (x_upper - x_lower))
        if delta_t_star_over_l <= 0.0:
            raise ValueError("The interpolated delta_t_star/L is not positive.")
        alpha = alpha_constant / delta_t_star_over_l
        alpha_bracket = tuple(
            sorted((alpha_constant / x_upper, alpha_constant / x_lower))
        )

        rng = np.random.default_rng(bootstrap_seed)
        if bootstrap_samples >= 2:
            for _ in range(bootstrap_samples):
                sampled_space = rng.normal(
                    spatial_estimate["value"], spatial_estimate["error"]
                )
                sampled_lower = rng.normal(lower["value"], lower["error"])
                sampled_upper = rng.normal(upper["value"], upper["error"])
                sampled_denominator = sampled_upper - sampled_lower
                if abs(sampled_denominator) < 1e-15:
                    continue
                sampled_fraction = (
                    sampled_space - sampled_lower
                ) / sampled_denominator
                sampled_x = x_lower + sampled_fraction * (x_upper - x_lower)
                if x_lower <= sampled_x <= x_upper and sampled_x > 0.0:
                    bootstrap_delta_t.append(sampled_x)
                    bootstrap_alpha.append(alpha_constant / sampled_x)
        if len(bootstrap_alpha) >= 2:
            alpha_samples = np.asarray(bootstrap_alpha)
            delta_t_samples = np.asarray(bootstrap_delta_t)
            alpha_stderr = float(np.std(alpha_samples, ddof=1))
            alpha_interval = tuple(
                map(float, np.percentile(alpha_samples, [16, 84]))
            )
            delta_t_stderr = float(np.std(delta_t_samples, ddof=1))
        elif bootstrap_samples >= 2:
            warnings.warn(
                "Too few bootstrap crossings remained inside the interpolation bracket."
            )

    summary = pd.DataFrame(
        [
            {
                "branch": item["curve"]["branch"],
                "delta_x": item["curve"]["delta_x"],
                "delta_t": item["curve"]["delta_t"],
                "delta_t_over_L": item["curve"]["delta_t"] / float(size),
                "I_late": item["value"],
                "I_late_stderr": item["error"],
                "mi_units": mi_units,
                "late_points": item["points"],
                "file": str(item["curve"]["path"]),
            }
            for item in estimates
        ]
    )
    print(
        f"delta_t*/L = {delta_t_star_over_l:.7g} ± {delta_t_stderr:.2g}\n"
        f"alpha = {alpha:.7g} ± {alpha_stderr:.2g} (stat.), "
        f"bracket [{alpha_bracket[0]:.7g}, {alpha_bracket[1]:.7g}]"
    )
    if show_summary:
        try:
            from IPython.display import display

            display(summary)
        except ImportError:
            print(summary.to_string(index=False))

    # palette = list(colors)
    # if not palette:
    #     raise ValueError("colors must contain at least one color.")
    # if len(palette) < len(ordered_curves):
    palette = list(
        plt.get_cmap("viridis")(
            np.linspace(0.08, 0.92, len(ordered_curves))
        )
    )
    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )
    for color, curve in zip(palette, ordered_curves):
        if curve["branch"] == "spatial":
            label = r"$(\delta x,\delta t)=(L/2,0)$"
        else:
            label = (
                rf"$(\delta x,\delta t)="
                rf"(0,{curve['delta_t']}L/{size})$"
            )
        every = max(1, int(np.ceil(len(curve["x"]) / max(1, errorbar_points))))
        for axis in (ax_full, ax_zoom):
            if show_errorbars:
                axis.errorbar(
                    curve["x"],
                    curve["I"],
                    yerr=curve["dI"],
                    color=color,
                    marker="o",
                    markersize=2.8,
                    linewidth=1.15,
                    elinewidth=0.75,
                    capsize=1.5,
                    errorevery=every,
                    label=label,
                )
            else:
                axis.plot(
                    curve["x"], curve["I"], color=color, linewidth=1.3, label=label
                )
    for axis, limits, title in (
        (ax_full, full_xlim, "Full evolution"),
        (ax_zoom, zoom_xlim, "Late-time zoom"),
    ):
        axis.set_xlim(*limits)
        axis.set_xlabel(r"Readout time $\tau/L$")
        axis.set_ylabel(rf"Mutual information $I_{{12}}$ [{mi_units}]")
        axis.set_title(title)
        axis.grid(alpha=0.22)
    if full_ylim is not None:
        ax_full.set_ylim(*full_ylim)
    if zoom_ylim is None:
        visible_lower = []
        visible_upper = []
        for curve in ordered_curves:
            mask = (curve["x"] >= zoom_xlim[0]) & (curve["x"] <= zoom_xlim[1])
            if np.any(mask):
                visible_lower.extend((curve["I"][mask] - curve["dI"][mask]).tolist())
                visible_upper.extend((curve["I"][mask] + curve["dI"][mask]).tolist())
        if visible_lower:
            lower_y, upper_y = min(visible_lower), max(visible_upper)
            margin = max(0.08 * (upper_y - lower_y), 1e-6)
            ax_zoom.set_ylim(lower_y - margin, upper_y + margin)
    else:
        ax_zoom.set_ylim(*zoom_ylim)
    ax_full.legend()
    ax_zoom.axvspan(
        estimate_window[0], estimate_window[1], color="black", alpha=0.045
    )
    ax_zoom.text(
        0.97,
        0.04,
        rf"$\delta t_*/L={delta_t_star_over_l:.4g}$" + "\n"
        + rf"$\alpha={alpha:.4g}\pm {alpha_stderr:.2g}$ (stat.)",
        transform=ax_zoom.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    fig.suptitle(
        rf"Two-probe anisotropy correlators ($L={size}$, $p={probabilities[0]:g}$)"
    )
    _show(fig, show)
    return {
        "figure": fig,
        "axes": (ax_full, ax_zoom),
        "L": size,
        "p": float(probabilities[0]),
        "mi_units": mi_units,
        "summary": summary,
        "curves": ordered_curves,
        "spatial_estimate": spatial_estimate,
        "bracketing_temporal_estimates": (lower, upper),
        "delta_t_star_over_L": delta_t_star_over_l,
        "delta_t_star": delta_t_star_over_l * size,
        "delta_t_star_stderr": delta_t_stderr * size,
        "alpha": alpha,
        "alpha_stderr": alpha_stderr,
        "alpha_interval_68": alpha_interval,
        "alpha_bracket": alpha_bracket,
        "bootstrap_successes": len(bootstrap_alpha),
    }


# ---------------------------------------------------------------------------
# Unified probe p scans and p-t entropy maps
# ---------------------------------------------------------------------------


_PROBE_PC_SPECS = {
    "sq": {
        "mean": ["S_Q_mean", "SQ_mean", "S_mean", "entropy_mean"],
        "stderr": ["S_Q_stderr", "SQ_stderr", "S_stderr", "entropy_stderr"],
        "ylabel": r"Reference entropy $\overline{S_Q}$ [nats]",
        "scaled_ylabel": r"$\overline{S_Q}$",
        "title": "One-probe Gullans--Huse order parameter",
        "x_symbol": "0",
    },
    "i2": {
        "mean": ["I2_mean", "I_mean", "i2_mean", "mi_mean"],
        "stderr": ["I2_stderr", "I_stderr", "i2_stderr", "mi_stderr"],
        "ylabel": r"$\overline{I_2}$ [nats]",
        "scaled_ylabel": r"$L^{x_2}\overline{I_2}$",
        "title": "Two-point probe information",
        "x_symbol": r"x_2",
    },
    "i3": {
        "mean": ["I3_mean", "i3_mean", "tmi_mean"],
        "stderr": ["I3_stderr", "i3_stderr", "tmi_stderr"],
        "ylabel": r"$\overline{I_3}$ [nats]",
        "scaled_ylabel": r"$L^{x_3}\overline{I_3}$",
        "title": "Three-point probe information",
        "x_symbol": r"x_3",
    },
    "i4": {
        "mean": ["I4_mean", "i4_mean"],
        "stderr": ["I4_stderr", "i4_stderr"],
        "ylabel": r"$I_4$ [nats]",
        "scaled_ylabel": r"$L^{x_4}I_4$",
        "title": "Four-point probe information",
        "x_symbol": r"x_4",
    },
}


def _infer_probe_count(path: Path, df: pd.DataFrame) -> int:
    match = re.search(r"(?:^|[_-])probed[_-]([124])(?:[_-]|$)", path.name)
    if match:
        return int(match.group(1))
    lowered = {str(column).lower() for column in df.columns}
    if "i4_mean" in lowered:
        return 4
    if "i2_mean" in lowered or "i_mean" in lowered:
        return 2
    return 1


def _load_probe_pc_curve(
    path: Path,
    metric: str,
    l_by_file: Mapping[str, int] | None,
    p_range: tuple[float | None, float | None],
) -> dict[str, Any]:
    spec = _PROBE_PC_SPECS[metric]
    df = pd.read_csv(path)
    p_col = _find_column(df, ["p", "measurement_rate"])
    mean_col = _find_column(df, spec["mean"])
    stderr_col = _find_column(df, spec["stderr"])
    clean = pd.DataFrame(
        {
            "p": pd.to_numeric(df[p_col], errors="coerce"),
            "value": pd.to_numeric(df[mean_col], errors="coerce"),
            "dvalue": pd.to_numeric(df[stderr_col], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean.sort_values("p").drop_duplicates("p", keep="last")
    p_min, p_max = p_range
    if p_min is not None:
        clean = clean.loc[clean["p"] >= p_min]
    if p_max is not None:
        clean = clean.loc[clean["p"] <= p_max]
    if clean.empty:
        raise ValueError(f"No usable {metric} rows remain in {path}.")
    if np.any(clean["dvalue"] < 0):
        warnings.warn(f"{path}: negative standard errors found; taking abs().")
        clean["dvalue"] = np.abs(clean["dvalue"])
    return {
        "path": path,
        "L": _parse_size(path, l_by_file),
        "p": clean["p"].to_numpy(dtype=float),
        "value": clean["value"].to_numpy(dtype=float),
        "dvalue": clean["dvalue"].to_numpy(dtype=float),
    }


def _fit_probe_pc_metric(
    curves: Sequence[dict[str, Any]],
    metric: str,
    *,
    pc_bounds: tuple[float, float],
    nu_bounds: tuple[float, float],
    x_bounds: tuple[float, float],
    fixed_pc: float | None,
    fixed_nu: float | None,
    fixed_x: float | None,
    interpolation_points: int,
    relative_error_floor: float,
    absolute_error_floor: float,
    seed: int,
) -> dict[str, Any]:
    for name, value, bound in (
        ("fixed_pc", fixed_pc, pc_bounds),
        ("fixed_nu", fixed_nu, nu_bounds),
        ("fixed_x", fixed_x, x_bounds),
    ):
        if value is not None and not bound[0] <= value <= bound[1]:
            raise ValueError(f"{name}={value} lies outside bounds {bound}.")
    names, bounds, fixed = [], [], {}
    for name, value, bound in (
        ("pc", fixed_pc, pc_bounds),
        ("nu", fixed_nu, nu_bounds),
        ("x", fixed_x, x_bounds),
    ):
        if value is None:
            names.append(name)
            bounds.append(bound)
        else:
            fixed[name] = float(value)

    def unpack(theta):
        values = dict(fixed)
        values.update({name: float(value) for name, value in zip(names, theta)})
        return values["pc"], values["nu"], values["x"]

    def transformed(curve, pc, nu, exponent):
        scale = float(curve["L"]) ** exponent
        x = (curve["p"] - pc) * float(curve["L"]) ** (1.0 / nu)
        order = np.argsort(x)
        return x[order], (curve["value"] * scale)[order], (
            curve["dvalue"] * scale
        )[order]

    def objective(theta, curve_set=curves):
        pc, nu, exponent = unpack(theta)
        if (
            not np.isfinite(pc + nu + exponent)
            or not pc_bounds[0] <= pc <= pc_bounds[1]
            or not nu_bounds[0] <= nu <= nu_bounds[1]
            or not x_bounds[0] <= exponent <= x_bounds[1]
        ):
            return np.inf
        scores = []
        for i in range(len(curve_set)):
            for j in range(i + 1, len(curve_set)):
                xa, ya, dya = transformed(curve_set[i], pc, nu, exponent)
                xb, yb, dyb = transformed(curve_set[j], pc, nu, exponent)
                lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
                if hi <= lo:
                    continue
                grid = np.linspace(lo, hi, interpolation_points)
                ya_grid, yb_grid = np.interp(grid, xa, ya), np.interp(grid, xb, yb)
                dya_grid = np.interp(grid, xa, dya)
                dyb_grid = np.interp(grid, xb, dyb)
                typical = max(
                    np.nanmedian(np.abs(ya_grid)),
                    np.nanmedian(np.abs(yb_grid)),
                    absolute_error_floor,
                )
                floor = max(absolute_error_floor, relative_error_floor * typical)
                scores.append(
                    np.mean(
                        (ya_grid - yb_grid) ** 2
                        / (dya_grid**2 + dyb_grid**2 + floor**2)
                    )
                )
        return float(np.mean(scores)) if scores else np.inf

    if bounds:
        global_fit = differential_evolution(
            objective,
            bounds=bounds,
            seed=seed,
            polish=False,
            updating="immediate",
            workers=1,
        )
        local_fit = minimize(
            objective,
            global_fit.x,
            method="Nelder-Mead",
            options={"maxiter": 20_000, "xatol": 1e-8, "fatol": 1e-8},
        )
        theta = local_fit.x if np.isfinite(local_fit.fun) else global_fit.x
        score = float(objective(theta))
        errors_free, covariance, hessian = _curvature_errors(
            objective, theta, f_min=score
        )
    else:
        global_fit = local_fit = None
        theta = np.empty(0)
        score = float(objective(theta))
        errors_free = np.empty(0)
        covariance = hessian = np.empty((0, 0))

    pc, nu, exponent = unpack(theta)
    errors = {"pc": 0.0, "nu": 0.0, "x": 0.0}
    for name, error in zip(names, errors_free):
        errors[name] = float(error)
    return {
        "metric": metric,
        "pc": pc,
        "nu": nu,
        "x": exponent,
        "pc_stderr": errors["pc"],
        "nu_stderr": errors["nu"],
        "x_stderr": errors["x"],
        "score": score,
        "transform": transformed,
        "global_fit": global_fit,
        "local_fit": local_fit,
        "covariance": covariance,
        "hessian": hessian,
    }


def probe_pc_collapse(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    l_by_file: Mapping[str, int] | None = None,
    metrics: Sequence[str] | str | None = None,
    p_range: tuple[float | None, float | None] = (None, None),
    pc_bounds: tuple[float, float] | None = None,
    nu_bounds: tuple[float, float] = (0.3, 5.0),
    x_bounds: tuple[float, float] = (0.0, 8.0),
    fixed_pc: float | None = None,
    fixed_nu: float | None = None,
    fixed_x: float | Mapping[str, float] | None = None,
    sq_units: str = "bits",
    mi_units: str = "nats",
    interpolation_points: int = 250,
    relative_error_floor: float = 0.02,
    absolute_error_floor: float = 1e-10,
    seed: int = 24680,
    cmap: str = "viridis",
    show_errorbars: bool = True,
    capsize: float = 2,
    figsize: tuple[float, float] | None = None,
    dpi: int = 130,
    show: bool = True,
) -> dict[str, Any]:
    r"""Plot fixed-``t=4L`` probe scans and fit
    ``I_k=L^{-x_k}f_k((p-p_c)L^{1/nu})``.

    A one-probe input uses ``S_Q`` with ``x=0`` and reproduces the Fig. 1(b)
    crossing/collapse protocol of Gullans and Huse. Two-probe inputs plot
    ``I2``; four-probe inputs produce stacked ``I2``, ``I3``, and ``I4`` raw
    and collapse pairs. Supply ``fixed_pc`` and/or ``fixed_nu`` to stabilize
    the noisier three-parameter multiprobe fits. ``mi_units`` controls all
    multiprobe information metrics independently of ``sq_units``.
    """
    sq_units = str(sq_units).strip().lower()
    if sq_units not in {"bits", "nats"}:
        raise ValueError("sq_units must be 'bits' or 'nats'.")
    mi_units, mi_scale = _mi_unit_spec(mi_units)
    paths = _resolve_files(files, file_glob)
    first_df = pd.read_csv(paths[0], nrows=2)
    probes = _infer_probe_count(paths[0], first_df)
    inferred_metrics = {1: ["sq"], 2: ["i2"], 4: ["i2", "i3", "i4"]}[probes]
    if metrics is None:
        selected_metrics = inferred_metrics
    elif isinstance(metrics, str):
        selected_metrics = [metrics.lower()]
    else:
        selected_metrics = [str(metric).lower() for metric in metrics]
    invalid = [metric for metric in selected_metrics if metric not in inferred_metrics]
    if invalid:
        raise ValueError(
            f"Metrics {invalid} are unavailable for probes={probes}; "
            f"choose from {inferred_metrics}."
        )

    curves_by_metric = {}
    all_p = []
    for metric in selected_metrics:
        curves = [
            _load_probe_pc_curve(path, metric, l_by_file, p_range)
            for path in paths
        ]
        curves.sort(key=lambda curve: curve["L"])
        if metric == "sq" and sq_units == "bits":
            for curve in curves:
                curve["value"] = curve["value"] / np.log(2.0)
                curve["dvalue"] = curve["dvalue"] / np.log(2.0)
        elif metric != "sq":
            for curve in curves:
                curve["value"] *= mi_scale
                curve["dvalue"] *= mi_scale
        sizes = [curve["L"] for curve in curves]
        if len(set(sizes)) != len(sizes):
            raise ValueError(f"Duplicate system sizes found: {sizes}")
        if len(curves) < 3:
            warnings.warn("Critical collapse has fewer than three system sizes.")
        curves_by_metric[metric] = curves
        all_p.extend(np.concatenate([curve["p"] for curve in curves]))

    if pc_bounds is None:
        pc_bounds = (float(np.min(all_p)), float(np.max(all_p)))
    if not pc_bounds[0] < pc_bounds[1]:
        raise ValueError("pc_bounds must be strictly increasing.")

    fits = {}
    for index, metric in enumerate(selected_metrics):
        if isinstance(fixed_x, Mapping):
            metric_fixed_x = fixed_x.get(metric)
        elif fixed_x is not None:
            metric_fixed_x = float(fixed_x)
        else:
            metric_fixed_x = 0.0 if metric == "sq" else None
        fits[metric] = _fit_probe_pc_metric(
            curves_by_metric[metric],
            metric,
            pc_bounds=pc_bounds,
            nu_bounds=nu_bounds,
            x_bounds=x_bounds,
            fixed_pc=fixed_pc,
            fixed_nu=fixed_nu,
            fixed_x=metric_fixed_x,
            interpolation_points=interpolation_points,
            relative_error_floor=relative_error_floor,
            absolute_error_floor=absolute_error_floor,
            seed=seed + index,
        )
        fit = fits[metric]
        print(
            f"{metric}: p_c={fit['pc']:.6g} ± {fit['pc_stderr']:.3g}, "
            f"nu={fit['nu']:.6g} ± {fit['nu_stderr']:.3g}, "
            f"x={fit['x']:.6g} ± {fit['x_stderr']:.3g}, "
            f"score={fit['score']:.6g}"
        )

    nrows = len(selected_metrics)
    if figsize is None:
        figsize = (13, 4.5 * nrows)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        constrained_layout=True,
    )
    sizes = sorted({curve["L"] for curves in curves_by_metric.values() for curve in curves})
    colors = _color_map_by_size(sizes, cmap)
    for row, metric in enumerate(selected_metrics):
        spec = _PROBE_PC_SPECS[metric]
        fit = fits[metric]
        ax_raw, ax_scaled = axes[row]
        for curve in curves_by_metric[metric]:
            size = curve["L"]
            kwargs = {
                "color": colors[size],
                "marker": "o",
                "markersize": 3.8,
                "linewidth": 1.2,
                "label": rf"$L={size}$",
            }
            if show_errorbars:
                ax_raw.errorbar(
                    curve["p"], curve["value"], yerr=curve["dvalue"],
                    capsize=capsize, elinewidth=0.9, **kwargs
                )
            else:
                ax_raw.plot(curve["p"], curve["value"], **kwargs)
            x_scaled, y_scaled, dy_scaled = fit["transform"](
                curve, fit["pc"], fit["nu"], fit["x"]
            )
            if show_errorbars:
                ax_scaled.errorbar(
                    x_scaled, y_scaled, yerr=dy_scaled,
                    capsize=capsize, elinewidth=0.9, **kwargs
                )
            else:
                ax_scaled.plot(x_scaled, y_scaled, **kwargs)

        ax_raw.axvline(fit["pc"], color="black", linestyle="--", linewidth=1)
        ax_raw.axhline(0.0, color="black", linewidth=0.7, alpha=0.3)
        ax_raw.set_xlabel(r"Measurement rate $p$")
        ax_raw.set_ylabel(
            r"Reference entropy $\overline{S_Q}$ [bits]"
            if metric == "sq" and sq_units == "bits"
            else (
                spec["ylabel"].replace("[nats]", f"[{mi_units}]")
                if metric != "sq"
                else spec["ylabel"]
            )
        )
        ax_raw.set_title(spec["title"] + " — raw")
        ax_raw.grid(alpha=0.25)
        ax_raw.legend()

        ax_scaled.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax_scaled.axhline(0.0, color="black", linewidth=0.7, alpha=0.3)
        ax_scaled.set_xlabel(r"$(p-p_c)L^{1/\nu}$")
        ax_scaled.set_ylabel(
            spec["scaled_ylabel"] + f" [{mi_units}]"
            if metric != "sq"
            else spec["scaled_ylabel"]
        )
        ax_scaled.set_title("Finite-size scaling collapse")
        ax_scaled.grid(alpha=0.25)
        ax_scaled.legend()
        x_text = (
            r"$x=0$ fixed"
            if metric == "sq" and abs(fit["x"]) < 1e-14
            else rf"${spec['x_symbol']}={fit['x']:.4g}\pm {fit['x_stderr']:.2g}$"
        )
        ax_scaled.text(
            0.97,
            0.03,
            rf"$p_c={fit['pc']:.5g}\pm {fit['pc_stderr']:.2g}$" + "\n"
            + rf"$\nu={fit['nu']:.4g}\pm {fit['nu_stderr']:.2g}$" + "\n"
            + x_text + "\n" + rf"score $={fit['score']:.3g}$",
            transform=ax_scaled.transAxes,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
        )
    fig.suptitle(f"Fixed-time critical probe scaling (probes={probes}, t=4L)")
    _show(fig, show)
    return {
        "figure": fig,
        "axes": axes,
        "probes": probes,
        "metrics": selected_metrics,
        "sq_units": sq_units,
        "mi_units": mi_units,
        "fits": fits,
        "curves": curves_by_metric,
    }


def probe_entropy_map(
    file: str | Path,
    *,
    p_values: Sequence[float] | None = None,
    line_count: int = 5,
    cmap: str = "magma",
    capsize: float = 2,
    show_errorbars: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (13, 5),
    dpi: int = 130,
    show: bool = True,
) -> dict[str, Any]:
    """Plot mode-2 ``S_Q(p,t)`` as a p-x/t-y heatmap and selected-p curves.

    ``p_values`` is the user control for the right-hand plot. Requested values
    are matched to the nearest simulated p value and duplicates are removed.
    """
    path = Path(file)
    df = pd.read_csv(path)
    probes = _infer_probe_count(path, df)
    p_col = _find_column(df, ["p", "measurement_rate"])
    t_col = _find_column(df, ["t", "time", "timestep"])
    mean_col = _find_column(df, _PROBE_PC_SPECS["sq"]["mean"])
    stderr_col = _find_column(df, _PROBE_PC_SPECS["sq"]["stderr"])
    clean = pd.DataFrame(
        {
            "p": pd.to_numeric(df[p_col], errors="coerce"),
            "t": pd.to_numeric(df[t_col], errors="coerce"),
            "S_Q": pd.to_numeric(df[mean_col], errors="coerce"),
            "dS_Q": pd.to_numeric(df[stderr_col], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError(f"No valid p-t entropy rows were loaded from {path}.")
    clean = clean.sort_values(["p", "t"]).drop_duplicates(["p", "t"], keep="last")
    simulated_p = np.sort(clean["p"].unique())
    simulated_t = np.sort(clean["t"].unique())
    expected = len(simulated_p) * len(simulated_t)
    if len(clean) != expected:
        raise ValueError(
            "The mode-2 CSV is not a complete rectangular p-t grid: "
            f"found {len(clean)} of {expected} rows."
        )
    pivot = clean.pivot(index="t", columns="p", values="S_Q").reindex(
        index=simulated_t, columns=simulated_p
    )

    if p_values is None:
        count = min(max(1, line_count), len(simulated_p))
        indices = np.unique(
            np.rint(np.linspace(0, len(simulated_p) - 1, count)).astype(int)
        )
        selected_p = simulated_p[indices]
    else:
        selected = []
        for requested in p_values:
            nearest = float(simulated_p[np.argmin(np.abs(simulated_p - requested))])
            if nearest not in selected:
                selected.append(nearest)
            if not np.isclose(nearest, requested, rtol=0.0, atol=1e-12):
                warnings.warn(
                    f"Requested p={requested:g}; using nearest simulated p={nearest:g}."
                )
        selected_p = np.asarray(selected, dtype=float)

    fig, (ax_map, ax_lines) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )
    mesh = ax_map.pcolormesh(
        simulated_p,
        simulated_t,
        pivot.to_numpy(dtype=float),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = fig.colorbar(mesh, ax=ax_map)
    colorbar.set_label(r"$\overline{S_Q}$ [nats]")
    ax_map.set_xlabel(r"Measurement rate $p$")
    ax_map.set_ylabel(r"Time $t$")
    ax_map.set_title(r"Joint-reference entropy $\overline{S_Q}(p,t)$")

    colors = plt.get_cmap("viridis")(
        np.linspace(0.1, 0.9, max(1, len(selected_p)))
    )
    for color, p in zip(colors, selected_p):
        curve = clean.loc[np.isclose(clean["p"], p)].sort_values("t")
        kwargs = {
            "color": color,
            "marker": "o",
            "markersize": 3.8,
            "linewidth": 1.2,
            "label": rf"$p={p:g}$",
        }
        if show_errorbars:
            ax_lines.errorbar(
                curve["t"], curve["S_Q"], yerr=curve["dS_Q"],
                capsize=capsize, elinewidth=0.9, **kwargs
            )
        else:
            ax_lines.plot(curve["t"], curve["S_Q"], **kwargs)
    ax_lines.set_xlabel(r"Time $t$")
    ax_lines.set_ylabel(r"Reference entropy $\overline{S_Q}$ [nats]")
    ax_lines.set_title("Entropy dynamics at selected measurement rates")
    ax_lines.grid(alpha=0.25)
    ax_lines.legend()
    fig.suptitle(f"Probe-entropy p-t scan (probes={probes})")
    _show(fig, show)
    return {
        "figure": fig,
        "axes": (ax_map, ax_lines),
        "colorbar": colorbar,
        "probes": probes,
        "selected_p": selected_p,
        "p": simulated_p,
        "t": simulated_t,
        "entropy": pivot.to_numpy(dtype=float),
        "data": clean,
    }


# ---------------------------------------------------------------------------
# Measurement-record free energy / effective central charge
# ---------------------------------------------------------------------------


def _free_energy_linear_fit(
    x: Iterable[float],
    y: Iterable[float],
    sigma: Iterable[float] | None = None,
) -> dict[str, float]:
    """Fit y=slope*x+intercept, using uncertainties when they are usable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma_array = np.asarray(sigma, dtype=float)
        valid &= np.isfinite(sigma_array)
    else:
        sigma_array = np.ones_like(x)
    x, y, sigma_array = x[valid], y[valid], sigma_array[valid]
    if x.size < 2:
        raise ValueError("At least two finite points are required for a line fit.")

    weighted = np.count_nonzero(sigma_array > 0.0) == sigma_array.size
    weights = 1.0 / sigma_array**2 if weighted else np.ones_like(x)
    design = np.column_stack((x, np.ones_like(x)))
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(normal)
    slope, intercept = covariance @ (design.T @ (weights * y))
    residual = y - (slope * x + intercept)
    if not weighted and x.size > 2:
        covariance *= np.sum(residual**2) / (x.size - 2)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_stderr": float(np.sqrt(max(0.0, covariance[0, 0]))),
        "intercept_stderr": float(np.sqrt(max(0.0, covariance[1, 1]))),
    }


def _load_free_energy_samples(
    paths: Sequence[Path],
) -> tuple[pd.DataFrame, dict[tuple[int, int], np.ndarray]]:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {
            "N",
            "p",
            "circ_type",
            "init_state",
            "free_energy_density_tilde",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}.")
        frame = frame.copy()
        frame["source"] = str(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for column in (
        "N",
        "p",
        "circ_type",
        "init_state",
        "free_energy_density_tilde",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "N",
            "p",
            "circ_type",
            "init_state",
            "free_energy_density_tilde",
        ]
    )
    data["N"] = data["N"].astype(int)
    data["init_state"] = data["init_state"].astype(int)

    samples = {
        (int(size), int(initial)): group[
            "free_energy_density_tilde"
        ].to_numpy(dtype=float)
        for (size, initial), group in data.groupby(["N", "init_state"])
    }
    return data, samples


def free_energy_ceff(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    alpha: float,
    alpha_stderr: float = 0.0,
    fit_sizes: Sequence[int] | None = None,
    l_min_values: Sequence[int] | None = None,
    require_both_initial_states: bool = True,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 24_680,
    cmap: str = "Blues",
    figsize: tuple[float, float] = (11.5, 5.3),
    show: bool = True,
) -> dict[str, Any]:
    """Recreate Zabalo et al. Fig. 1(b) and extract ``c_eff``.

    The input is the ``*_samples.csv`` output from ``free_energy.exe`` for
    several system sizes and both ``init_state=0`` and ``init_state=1``.  The
    simulator column ``free_energy_density_tilde`` is
    ``F_production/(L*t_production)`` as obtained from the production-window
    slope.  As in the paper, the plotted and fitted quantity is
    ``f_tilde(L)=F/(L*t)``; ``alpha`` is applied when converting its asymptotic
    slope to ``c_eff``.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")
    if not np.isfinite(alpha_stderr) or alpha_stderr < 0.0:
        raise ValueError("alpha_stderr must be finite and non-negative.")

    paths = _resolve_files(files, file_glob)
    data, samples = _load_free_energy_samples(paths)
    if data["circ_type"].nunique() != 1:
        raise ValueError("All files must use the same circ_type.")
    probabilities = data["p"].to_numpy(dtype=float)
    if np.ptp(probabilities) > 1.0e-12 * max(1.0, np.max(np.abs(probabilities))):
        raise ValueError("All files must use the same measurement rate p.")

    available_sizes = sorted(data["N"].unique().astype(int))
    if fit_sizes is not None:
        requested = {int(size) for size in fit_sizes}
        available_sizes = [size for size in available_sizes if size in requested]
    if len(available_sizes) < 3:
        raise ValueError("At least three system sizes are required.")

    summary_rows = []
    grouped_samples: dict[int, dict[int, np.ndarray]] = {}
    for size in available_sizes:
        grouped_samples[size] = {}
        for initial in (0, 1):
            values = samples.get((size, initial), np.array([], dtype=float))
            if values.size:
                grouped_samples[size][initial] = values
        if require_both_initial_states and set(grouped_samples[size]) != {0, 1}:
            raise ValueError(
                f"L={size} needs both init_state=0 and init_state=1 files."
            )
        if not grouped_samples[size]:
            continue

        state_means = {
            initial: float(np.mean(values))
            for initial, values in grouped_samples[size].items()
        }
        state_errors = {
            initial: (
                float(np.std(values, ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else 0.0
            )
            for initial, values in grouped_samples[size].items()
        }
        initial_states = sorted(state_means)
        combined_mean = float(np.mean([state_means[key] for key in initial_states]))
        combined_stderr = float(
            np.sqrt(sum(state_errors[key] ** 2 for key in initial_states))
            / len(initial_states)
        )
        summary_rows.append(
            {
                "L": size,
                "f_tilde": combined_mean,
                "f_tilde_stderr": combined_stderr,
                "f": combined_mean / alpha,
                "f_stderr": combined_stderr / alpha,
                "product_count": len(grouped_samples[size].get(0, [])),
                "haar_count": len(grouped_samples[size].get(1, [])),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("L").reset_index(drop=True)
    sizes = summary["L"].to_numpy(dtype=int)

    eligible_lmins = [
        int(size) for size in sizes if np.count_nonzero(sizes >= size) >= 3
    ]
    if l_min_values is None:
        chosen_lmins = eligible_lmins
    else:
        chosen_lmins = [int(value) for value in l_min_values]
        invalid = set(chosen_lmins).difference(eligible_lmins)
        if invalid:
            raise ValueError(
                f"Each L_min must leave at least three sizes; invalid: {sorted(invalid)}."
            )
    if not chosen_lmins:
        raise ValueError("No valid L_min values remain.")

    def first_stage(means: np.ndarray) -> list[dict[str, float]]:
        fitted = []
        errors = summary["f_tilde_stderr"].to_numpy(dtype=float)
        for minimum in chosen_lmins:
            mask = sizes >= minimum
            fit = _free_energy_linear_fit(
                1.0 / sizes[mask].astype(float) ** 2,
                means[mask],
                errors[mask],
            )
            fitted.append({"L_min": minimum, **fit})
        return fitted

    def second_stage(fitted: Sequence[Mapping[str, float]]) -> dict[str, float]:
        if len(fitted) == 1:
            only = fitted[0]
            return {
                "slope": np.nan,
                "intercept": float(only["slope"]),
                "slope_stderr": np.nan,
                "intercept_stderr": float(only["slope_stderr"]),
            }
        return _free_energy_linear_fit(
            [1.0 / float(row["L_min"]) ** 2 for row in fitted],
            [float(row["slope"]) for row in fitted],
            [float(row["slope_stderr"]) for row in fitted],
        )

    central_means = summary["f_tilde"].to_numpy(dtype=float)
    fits = first_stage(central_means)
    correction_fit = second_stage(fits)
    m_tilde_infinite = correction_fit["intercept"]
    c_eff = -6.0 * m_tilde_infinite / (np.pi * alpha)
    analytic_stderr = np.sqrt(
        (6.0 * correction_fit["intercept_stderr"] / (np.pi * alpha)) ** 2
        + (c_eff * alpha_stderr / alpha) ** 2
    )

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_ceff = []
    if bootstrap_samples >= 2:
        for _ in range(bootstrap_samples):
            sampled_means = []
            for size in sizes:
                by_initial = grouped_samples[int(size)]
                means = [
                    np.mean(rng.choice(values, size=values.size, replace=True))
                    for values in by_initial.values()
                ]
                sampled_means.append(float(np.mean(means)))
            sampled_alpha = (
                rng.normal(alpha, alpha_stderr)
                if alpha_stderr > 0.0
                else alpha
            )
            if sampled_alpha <= 0.0:
                continue
            try:
                sampled_fits = first_stage(np.asarray(sampled_means, dtype=float))
                sampled_correction = second_stage(sampled_fits)
                value = (
                    -6.0 * sampled_correction["intercept"]
                    / (np.pi * sampled_alpha)
                )
                if np.isfinite(value):
                    bootstrap_ceff.append(value)
            except (ValueError, np.linalg.LinAlgError):
                continue

    if len(bootstrap_ceff) >= 10:
        bootstrap_array = np.asarray(bootstrap_ceff, dtype=float)
        c_eff_stderr = float(np.std(bootstrap_array, ddof=1))
        c_eff_p16, c_eff_p84 = np.percentile(bootstrap_array, [16.0, 84.0])
    else:
        c_eff_stderr = float(analytic_stderr)
        c_eff_p16 = c_eff - c_eff_stderr
        c_eff_p84 = c_eff + c_eff_stderr
        if bootstrap_samples >= 2:
            warnings.warn("Too few successful c_eff bootstrap fits; using fit errors.")

    fig, ax = plt.subplots(figsize=figsize)
    # Reserve the extra horizontal space for the finite-L_min inset so it does
    # not obscure the main finite-size data or fits.
    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.15, top=0.91)
    colors = plt.get_cmap(cmap)(
        np.linspace(0.35, 0.90, max(1, len(fits)))
    )
    x_all = 1.0 / sizes.astype(float) ** 2
    ax.errorbar(
        x_all,
        summary["f_tilde"],
        yerr=summary["f_tilde_stderr"],
        linestyle="none",
        marker="o",
        markersize=3,
        markeredgewidth=0.7,
        color="black",
        capsize=1.5,
        elinewidth=0.75,
        capthick=0.75,
        label="product/Haar average",
        zorder=5,
    )
    for color, fit in zip(colors, fits):
        maximum_x = 1.0 / float(fit["L_min"]) ** 2
        x_line = np.linspace(0.0, maximum_x * 1.03, 200)
        ax.plot(
            x_line,
            fit["slope"] * x_line + fit["intercept"],
            color=color,
            label=rf"$L_{{\min}}={int(fit['L_min'])}$",
        )
    tick_order = np.argsort(x_all)
    ax.set_xticks(x_all[tick_order])
    ax.set_xticklabels(
        [rf"$1/{int(size)}^2$" for size in sizes[tick_order]]
    )
    ax.set_xlabel(r"$1/L^2$")
    ax.set_ylabel(r"$\widetilde{f}(L)=F/(L t)$")
    ax.set_title("Measurement-record free-energy density")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(
        0.03,
        0.04,
        rf"$\alpha={alpha:.5g}$" + "\n" +
        rf"$c_{{\mathrm{{eff}}}}={c_eff:.5f}\pm{c_eff_stderr:.5f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    inset = fig.add_axes([0.77, 0.55, 0.20, 0.34])
    inset_x = np.asarray(
        [1.0 / float(row["L_min"]) ** 2 for row in fits]
    )
    inset_y = np.asarray([float(row["slope"]) for row in fits])
    inset_error = np.asarray([float(row["slope_stderr"]) for row in fits])
    inset.errorbar(
        inset_x,
        inset_y,
        yerr=inset_error,
        linestyle="none",
        marker="o",
        markersize=3,
        markeredgewidth=0.7,
        color="black",
        capsize=1,
        elinewidth=0.7,
        capthick=0.7,
    )
    if len(fits) >= 2:
        inset_line_x = np.linspace(0.0, 1.05 * np.max(inset_x), 100)
        inset.plot(
            inset_line_x,
            correction_fit["slope"] * inset_line_x + m_tilde_infinite,
            linestyle=":",
            color="tab:blue",
        )
    inset.set_xlabel(r"$1/L_{\min}^2$", fontsize=8)
    inset.set_ylabel(r"$m_0(L_{\min})$", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.2)
    _show(fig, show)

    return {
        "figure": fig,
        "axis": ax,
        "inset": inset,
        "summary": summary,
        "fits": pd.DataFrame(fits),
        "correction_fit": correction_fit,
        "alpha": alpha,
        "alpha_stderr": alpha_stderr,
        "c_eff": float(c_eff),
        "c_eff_stderr": float(c_eff_stderr),
        "c_eff_p16": float(c_eff_p16),
        "c_eff_p84": float(c_eff_p84),
        "bootstrap_successes": len(bootstrap_ceff),
        "samples": data,
    }


def free_energy_equilibration(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    t_over_l_range: tuple[float, float] = (0.0, 10.0),
    fit_t_over_l_min: float = 2.0,
    colors: tuple[Any, Any, Any] = ("tab:blue", "tab:orange", "tab:green"),
    figsize: tuple[float, float] = (11.5, 4.5),
    show: bool = True,
) -> dict[str, Any]:
    """Recreate Supplementary Fig. S1(a,b) from ``*_timeseries.csv``.

    Unlike the production-only estimator used by :func:`free_energy_ceff`,
    these curves use the cumulative measurement-record entropy from the first
    timestep.  ``init_state=0`` is the product ensemble and ``init_state=1`` is
    the Haar ensemble.  The half-chain von Neumann entropy inset is in bits.
    """
    paths = _resolve_files(files, file_glob)
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {
            "N",
            "p",
            "circ_type",
            "init_state",
            "completed_realizations",
            "t",
            "t_over_L",
            "F_mean",
            "F_stderr",
            "F_over_tL_mean",
            "F_over_tL_stderr",
            "S_half_mean",
            "S_half_stderr",
            "S_half_count",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}.")
        frame = frame.copy()
        frame["source"] = str(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    numeric = [
        "N",
        "p",
        "circ_type",
        "init_state",
        "completed_realizations",
        "t",
        "t_over_L",
        "F_mean",
        "F_stderr",
        "F_over_tL_mean",
        "F_over_tL_stderr",
        "S_half_mean",
        "S_half_stderr",
        "S_half_count",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["N", "circ_type", "init_state", "t"])
    if data["N"].nunique() != 1 or data["circ_type"].nunique() != 1:
        raise ValueError("Fig. S1 inputs must use one N and one circ_type.")
    probabilities = data["p"].dropna().to_numpy(dtype=float)
    if probabilities.size and np.ptp(probabilities) > 1.0e-12 * max(
        1.0, np.max(np.abs(probabilities))
    ):
        raise ValueError("Fig. S1 inputs must use one measurement rate p.")

    # A time-series file is a checkpoint that is rewritten after every
    # trajectory. If several checkpoints are supplied, retain the most complete
    # one for each initial-state/time pair rather than counting the same samples
    # more than once.
    data = data.sort_values("completed_realizations").drop_duplicates(
        ["init_state", "t"], keep="last"
    )
    initial_states = set(data["init_state"].astype(int).unique())
    if not {0, 1}.issubset(initial_states):
        raise ValueError("Provide both init_state=0 and init_state=1 time-series files.")

    product = data.loc[data["init_state"] == 0].copy()
    haar = data.loc[data["init_state"] == 1].copy()
    columns = [
        "t",
        "t_over_L",
        "F_mean",
        "F_stderr",
        "F_over_tL_mean",
        "F_over_tL_stderr",
        "S_half_mean",
        "S_half_stderr",
        "S_half_count",
    ]
    aligned = product[columns].merge(
        haar[columns], on=["t", "t_over_L"], suffixes=("_product", "_haar")
    ).sort_values("t")
    if aligned.empty:
        raise ValueError("The product and Haar time grids do not overlap.")

    aligned["F_over_tL_mean_average"] = 0.5 * (
        aligned["F_over_tL_mean_product"] + aligned["F_over_tL_mean_haar"]
    )
    aligned["F_over_tL_stderr_average"] = 0.5 * np.sqrt(
        aligned["F_over_tL_stderr_product"] ** 2
        + aligned["F_over_tL_stderr_haar"] ** 2
    )
    aligned["S_half_mean_average"] = 0.5 * (
        aligned["S_half_mean_product"] + aligned["S_half_mean_haar"]
    )
    aligned["S_half_stderr_average"] = 0.5 * np.sqrt(
        aligned["S_half_stderr_product"] ** 2
        + aligned["S_half_stderr_haar"] ** 2
    )
    size = int(data["N"].iloc[0])
    aligned["delta_F_over_L"] = (
        aligned["F_mean_haar"] - aligned["F_mean_product"]
    ) / size
    aligned["delta_F_over_L_stderr"] = np.sqrt(
        aligned["F_stderr_haar"] ** 2 + aligned["F_stderr_product"] ** 2
    ) / size

    fig, (ax_density, ax_delta) = plt.subplots(1, 2, figsize=figsize)
    curve_specs = (
        ("haar", "random Haar state", colors[0]),
        ("product", "random product state", colors[1]),
        ("average", "average", colors[2]),
    )
    boundary_fits = {}
    for key, label, color in curve_specs:
        y = aligned[f"F_over_tL_mean_{key}"].to_numpy(dtype=float)
        error = aligned[f"F_over_tL_stderr_{key}"].to_numpy(dtype=float)
        x = aligned["t_over_L"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax_density.plot(x[mask], y[mask], color=color, label=label)
        ax_density.fill_between(
            x[mask], y[mask] - error[mask], y[mask] + error[mask],
            color=color, alpha=0.16,
        )

        fit_mask = mask & (x >= fit_t_over_l_min) & (x <= t_over_l_range[1])
        if np.count_nonzero(fit_mask) >= 2:
            fit = _free_energy_linear_fit(
                1.0 / aligned.loc[fit_mask, "t"].to_numpy(dtype=float),
                y[fit_mask],
                error[fit_mask],
            )
            boundary_fits[key] = fit
            fit_x = x[fit_mask]
            fit_t = aligned.loc[fit_mask, "t"].to_numpy(dtype=float)
            ax_density.plot(
                fit_x,
                fit["intercept"] + fit["slope"] / fit_t,
                color=color,
                linestyle=":",
                linewidth=1.1,
            )

    ax_density.set_xlim(*t_over_l_range)
    ax_density.set_xlabel(r"$t/L$")
    ax_density.set_ylabel(r"$F(t)/(tL)$")
    ax_density.set_title("(a) Record-entropy density")
    ax_density.grid(alpha=0.25)
    ax_density.legend(fontsize=8)

    # The record-density curves live in the upper half of this panel after
    # equilibration; the lower-right corner keeps the entropy inset clear.
    inset = ax_density.inset_axes([0.50, 0.06, 0.47, 0.38])
    for key, label, color in curve_specs:
        y = aligned[f"S_half_mean_{key}"].to_numpy(dtype=float)
        x = aligned["t_over_L"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            inset.plot(x[mask], y[mask], color=color, label=label)
    inset.set_xlabel(r"$t/L$", fontsize=8)
    inset.set_ylabel(r"$S_1(t)$ [bits]", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.2)

    x = aligned["t_over_L"].to_numpy(dtype=float)
    delta = aligned["delta_F_over_L"].to_numpy(dtype=float)
    delta_error = aligned["delta_F_over_L_stderr"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(delta)
    ax_delta.plot(x[mask], delta[mask], color="tab:purple")
    ax_delta.fill_between(
        x[mask],
        delta[mask] - delta_error[mask],
        delta[mask] + delta_error[mask],
        color="tab:purple",
        alpha=0.18,
    )
    ax_delta.set_xlim(*t_over_l_range)
    ax_delta.set_xlabel(r"$t/L$")
    ax_delta.set_ylabel(r"$\Delta F(t)/L$")
    ax_delta.set_title("(b) Initial-state boundary contribution")
    ax_delta.grid(alpha=0.25)
    fig.tight_layout()
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_density, ax_delta),
        "entropy_inset": inset,
        "data": aligned,
        "boundary_fits": boundary_fits,
        "N": size,
        "p": float(probabilities[0]) if probabilities.size else np.nan,
    }


# ---------------------------------------------------------------------------
# Four-probe multipartite-information collapse
# ---------------------------------------------------------------------------


def _probe4_metric_spec(metric: str) -> dict[str, Any]:
    """Resolve a four-probe information metric and its CSV/plot metadata."""
    key = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "2": "i2",
        "mi2": "i2",
        "pair": "i2",
        "pairwise": "i2",
        "pairwise_mi": "i2",
        "3": "i3",
        "tmi": "i3",
        "tripartite": "i3",
        "tripartite_information": "i3",
        "4": "i4",
        "fourpartite": "i4",
        "four_party": "i4",
        "four_party_information": "i4",
    }
    key = aliases.get(key, key)

    specs = {
        "i2": {
            "mean_columns": ["I2_mean", "i2_mean", "I_2_mean"],
            "stderr_columns": ["I2_stderr", "i2_stderr", "I_2_stderr"],
            "raw_ylabel": r"Mean pair information $\overline{I_2}$ [nats]",
            "raw_title": "Four-probe mean pair information",
            "collapse_ylabel": r"$L^{\eta_2}\overline{I_2}$",
            "ansatz": r"$\overline{I_2}=L^{-\eta_2}g_2((t-2L)/L)$",
            "suptitle": "Four-probe pair-information scaling",
            "eta_symbol": r"\eta_2",
            "nonnegative": True,
        },
        "i3": {
            "mean_columns": ["I3_mean", "i3_mean", "I_3_mean", "tmi_mean"],
            "stderr_columns": [
                "I3_stderr",
                "i3_stderr",
                "I_3_stderr",
                "tmi_stderr",
            ],
            "raw_ylabel": r"Mean tripartite information $\overline{I_3}$ [nats]",
            "raw_title": "Four-probe mean tripartite information",
            "collapse_ylabel": r"$L^{\eta_3}\overline{I_3}$",
            "ansatz": r"$\overline{I_3}=L^{-\eta_3}g_3((t-2L)/L)$",
            "suptitle": "Four-probe tripartite-information scaling",
            "eta_symbol": r"\eta_3",
            "nonnegative": False,
        },
        "i4": {
            "mean_columns": ["I4_mean", "i4_mean", "I_4_mean"],
            "stderr_columns": ["I4_stderr", "i4_stderr", "I_4_stderr"],
            "raw_ylabel": r"Four-party information $I_4$ [nats]",
            "raw_title": "Four-probe four-party information",
            "collapse_ylabel": r"$L^{\eta_4}I_4$",
            "ansatz": r"$I_4=L^{-\eta_4}g_4((t-2L)/L)$",
            "suptitle": "Four-probe four-party-information scaling",
            "eta_symbol": r"\eta_4",
            "nonnegative": False,
        },
    }
    if key not in specs:
        raise ValueError("metric must be 'i2', 'i3', or 'i4'.")
    return {"key": key, **specs[key]}


def _load_probe4_curve(
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


def probe4_collapse(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    l_by_file: Mapping[str, int] | None = None,
    metric: str = "i2",
    mi_units: str = "nats",
    x_load: tuple[float | None, float | None] = (None, None),
    fit_x: tuple[float, float | None] = (0.25, 8.0),
    fit_abs_min: float = 1e-8,
    eta_bounds: tuple[float, float] = (0.0, 8.0),
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
    r"""Fit a four-probe ``I_k=L^(-eta_k)g_k((t-2L)/L)`` collapse.

    ``metric`` selects ``"i2"``, ``"i3"``, or ``"i4"``. The simulator's
    ``I2`` is the mean over all six probe pairs, ``I3`` is the mean over all
    four probe triplets, and ``I4`` is the four-party information. Because
    ``I3`` and ``I4`` may be signed, the fit cutoff is applied to the absolute
    value through ``fit_abs_min`` and bootstrap samples are not clipped.
    ``mi_units`` selects ``"nats"`` or ``"bits"`` for every information metric.
    """
    if fit_abs_min < 0.0:
        raise ValueError("fit_abs_min must be non-negative.")
    mi_units, mi_scale = _mi_unit_spec(mi_units)
    metric_spec = _probe4_metric_spec(metric)
    paths = _resolve_files(files, file_glob)
    curves = [
        _load_probe4_curve(
            path,
            l_by_file=l_by_file,
            x_load=x_load,
            metric_spec=metric_spec,
        )
        for path in paths
    ]
    for curve in curves:
        curve["value"] *= mi_scale
        curve["dvalue"] *= mi_scale
    curves.sort(key=lambda curve: curve["L"])
    sizes = [curve["L"] for curve in curves]
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"Duplicate system sizes found: {sizes}")
    if len(curves) < 3:
        warnings.warn("A collapse with fewer than three sizes is weakly constrained.")

    finite_ps = [curve["p"] for curve in curves if np.isfinite(curve["p"])]
    if finite_ps and not np.allclose(finite_ps, finite_ps[0], rtol=0, atol=1e-12):
        warnings.warn(f"The files contain multiple p values: {sorted(set(finite_ps))}")

    summary = pd.DataFrame(
        {
            "L": sizes,
            "metric": metric_spec["key"],
            "p_from_filename": [curve["p"] for curve in curves],
            "points": [len(curve["x"]) for curve in curves],
            "x_min": [np.min(curve["x"]) for curve in curves],
            "x_max": [np.max(curve["x"]) for curve in curves],
            "value_min": [np.min(curve["value"]) for curve in curves],
            "value_max": [np.max(curve["value"]) for curve in curves],
            "abs_max": [np.max(np.abs(curve["value"])) for curve in curves],
            "x_at_abs_max": [
                curve["x"][np.argmax(np.abs(curve["value"]))]
                for curve in curves
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
            & (np.abs(curve["value"]) > fit_abs_min)
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
                synthetic_values = rng.normal(
                    curve["value"],
                    np.maximum(curve["dvalue"], 0.0),
                )
                synthetic["value"] = (
                    np.maximum(synthetic_values, 0.0)
                    if metric_spec["nonnegative"]
                    else synthetic_values
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

    ax_raw.axhline(0.0, linewidth=1, alpha=0.35)
    ax_raw.axvline(0.0, linewidth=1, linestyle="--", alpha=0.6)
    fit_upper = (
        fit_x_max
        if fit_x_max is not None
        else max(curve["x"].max() for curve in curves)
    )
    ax_raw.axvspan(fit_x_min, fit_upper, alpha=0.08, label="fit window")
    ax_raw.set_xlabel(r"Scaled time $x=(t-2L)/L$")
    ax_raw.set_ylabel(
        metric_spec["raw_ylabel"].replace("[nats]", f"[{mi_units}]")
    )
    ax_raw.set_title(metric_spec["raw_title"])
    ax_raw.set_yscale(raw_yscale)
    ax_raw.grid(alpha=0.25)
    ax_raw.legend()

    ax_collapse.axhline(0.0, linewidth=1, alpha=0.35)
    ax_collapse.set_xlabel(r"Scaled time $x=(t-2L)/L$")
    ax_collapse.set_ylabel(
        metric_spec["collapse_ylabel"] + f" [{mi_units}]"
    )
    ax_collapse.set_title("Bulk-exponent scaling collapse")
    ax_collapse.set_yscale(collapse_yscale)
    ax_collapse.grid(alpha=0.25)
    ax_collapse.legend()
    eta_text = (
        rf"${metric_spec['eta_symbol']}={eta:.4f}\pm {eta_stderr:.4f}$"
        if np.isfinite(eta_stderr)
        else rf"${metric_spec['eta_symbol']}={eta:.4f}$"
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
        "mi_units": mi_units,
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

"""Renyi-2 entropy scaling against conformal chord distance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fitting import _weighted_linear_fit
from .loading import _as_paths
from .plotting import _show


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

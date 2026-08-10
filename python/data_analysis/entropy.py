"""Entanglement-entropy scaling against conformal chord distance.

``entropy.exe`` reports both the von Neumann entropy S1 = -Tr(rho ln rho) and the
second Renyi entropy S2 = -ln Tr(rho^2) for every block, so ``order`` selects
which one is plotted and fitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fitting import _weighted_linear_fit
from .loading import _resolve_files
from .plotting import _show

# Renyi index -> (name, label for the y axis).
_ORDERS = {
    1: ("von Neumann", r"$\overline{S_A^{(1)}}$"),
    2: ("Renyi-2", r"$\overline{S_A^{(2)}}$"),
}

# Curve-separating parameter -> (curve key, formatter for legend/annotation).
_VARY_KEYS = {
    "p": ("p", lambda value: rf"$p={value:g}$"),
    "L": ("L", lambda value: rf"$L={value:g}$"),
}


def _resolve_vary(vary: str | None, curves: Sequence[dict[str, Any]]) -> str:
    """Decide which of L and p separates the curves, and check the other is fixed.

    A single file leaves both constant; that is a p scan of one point, which
    keeps the historical annotation (L in the box, p in the legend).
    """
    spread = {
        key: [f"{value:g}" for value in sorted({curve[key] for curve in curves})]
        for key in _VARY_KEYS
    }
    if vary is None:
        if len(spread["L"]) == 1 and len(spread["p"]) > 1:
            vary = "p"
        elif len(spread["p"]) == 1 and len(spread["L"]) > 1:
            vary = "L"
        elif len(spread["L"]) == 1 and len(spread["p"]) == 1:
            vary = "p"
        else:
            raise ValueError(
                f"Both L and p vary across the inputs "
                f"(L = {', '.join(spread['L'])}; p = {', '.join(spread['p'])}), "
                f"so neither can label the curves. Narrow the file selection to "
                f"one size or one measurement rate."
            )
    fixed = "L" if vary == "p" else "p"
    if len(spread[fixed]) != 1:
        raise ValueError(
            f"vary={vary!r} needs a single {fixed}, but the inputs use "
            f"{fixed} = {', '.join(spread[fixed])}."
        )
    return vary


def _resolve_uncertainty(uncertainty: str, order: int) -> str:
    """Map ``uncertainty`` onto a column name for the requested Renyi index."""
    if uncertainty in ("stderr", "stddev"):
        return f"S{order}_{uncertainty}"
    # A literal column name. Reject one that belongs to the *other* order rather
    # than silently pairing S1 means with S2 error bars -- the historical
    # default was the literal "S2_stderr", so this is a real mixture to hit.
    for other in _ORDERS:
        if other != order and uncertainty.startswith(f"S{other}_"):
            raise ValueError(
                f"uncertainty={uncertainty!r} belongs to Renyi index {other}, but "
                f"order={order} was requested. Pass 'stderr' or 'stddev' to track "
                f"the selected order automatically."
            )
    return uncertainty


def entropy(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    vary: str | None = None,
    order: int = 2,
    fit_above_p: float = 0.3,
    exclude_first: int = 1,
    uncertainty: str = "stderr",
    xlim: tuple[float, float] | None = (0.5, 2.05),
    ylim: tuple[float, float] | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 6),
    show: bool = True,
) -> dict[str, Any]:
    """Plot entropy versus conformal distance and fit selected curves.

    Parameters
    ----------
    files:
        One path, or a sequence of them. Any entry may be a glob pattern
        (``"cpp/csv/entropy/*_haar_*.csv"``); patterns expand in sorted order
        and duplicates across patterns are dropped. An entry that names an
        existing file is always taken literally.
    file_glob:
        A single pattern used only when ``files`` is empty, matching the
        convention of the probe and free-energy plotters.
    vary:
        Which parameter separates the curves: ``"p"`` for a measurement-rate
        scan at one system size, ``"L"`` for a size scan at one measurement
        rate. The varied parameter labels the legend and the fits, the fixed
        one goes in the annotation box. ``None`` (the default) infers it from
        whichever of the two is constant across the inputs.
    order:
        Renyi index to plot: ``1`` for the von Neumann entropy, ``2`` for the
        second Renyi entropy (the default, and the only one older CSVs carry).
    uncertainty:
        ``"stderr"`` or ``"stddev"``, which follow ``order``, or an explicit
        column name.
    """
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {sorted(_ORDERS)}, got {order!r}.")
    if vary is not None and vary not in _VARY_KEYS:
        raise ValueError(f"vary must be one of {sorted(_VARY_KEYS)} or None, got {vary!r}.")
    paths = _resolve_files(files, file_glob, description="entropy CSVs")
    if exclude_first < 0:
        raise ValueError("exclude_first must be non-negative.")

    order_name, order_label = _ORDERS[order]
    mean_column = f"S{order}_mean"
    uncertainty_column = _resolve_uncertainty(uncertainty, order)

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
            mean_column,
            uncertainty_column,
        }
        missing = required.difference(df.columns)
        if missing:
            if order == 1 and {mean_column, uncertainty_column} & missing:
                raise ValueError(
                    f"{path} has no von Neumann columns. Files written before "
                    f"entropy.exe reported S1 carry S2 only; re-run the scan or "
                    f"pass order=2."
                )
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df = df.sort_values("L_A").reset_index(drop=True)

        x = df["ln_x"].to_numpy(dtype=float)
        y = df[mean_column].to_numpy(dtype=float)
        dy = df[uncertainty_column].to_numpy(dtype=float)
        # Blocks excluded by MIPT_ENTROPY_S1_MAX_LA report NaN, never 0. Drop
        # them so a capped run still plots and fits over the range it covered.
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(dy)
        if not keep.any():
            raise ValueError(
                f"{path} has no finite {mean_column} values. A run with "
                f"MIPT_ENTROPY_S1=0 reports S1 as NaN for every block."
            )
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
                "order": order,
                "x": x[keep],
                "y": y[keep],
                "dy": dy[keep],
            }
        )

    vary = _resolve_vary(vary, curves)
    fixed_key = "L" if vary == "p" else "p"
    if fixed_key == "L":
        # At one size the depth should match too, or the curves are different
        # runs pooled by accident. A size scan is exempt: t = 2L by convention,
        # so periods necessarily moves with L.
        unique_periods = sorted({curve["periods"] for curve in curves})
        if len(unique_periods) != 1:
            raise ValueError(
                f"Input CSVs share L={curves[0]['L']} but use multiple period "
                f"counts: {unique_periods}."
            )
    _, vary_label = _VARY_KEYS[vary]
    _, fixed_label = _VARY_KEYS[fixed_key]
    fixed_value = curves[0][fixed_key]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, len(curves)))
    fit_rows: list[dict[str, Any]] = []

    for color, curve in zip(colors, curves):
        x, y, dy, p = curve["x"], curve["y"], curve["dy"], curve["p"]
        curve_label = vary_label(curve[vary])
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

        fit_record = {
            "file": str(curve["path"]),
            "L": curve["L"],
            "p": p,
            "order": order,
        }
        if p > fit_above_p and len(x) > exclude_first + 1:
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
                        rf"{curve_label}: $\alpha=${fit['slope']:.{precision}f}"
                        rf"$\pm ${slope_error:.{precision}f}, "
                        rf"$\chi_\nu^2=${fit['reduced_chi2']:.1g}"
                    ),
                )
                fit_record.update(fit)
            except ValueError as exc:
                data_line.set_label(rf"{curve_label}: fit unavailable ({exc})")
                fit_record["error"] = str(exc)
        else:
            data_line.set_label(curve_label)
        fit_rows.append(fit_record)

    ax.text(
        0.99,
        0.02,
        fixed_label(fixed_value),# + "\n" + order_name,
        transform=ax.transAxes,
        fontsize=12,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.7},
    )
    ax.set_xlabel(
        r"$\ln x$, $x=\frac{L}{\pi}\sin\left(\frac{\pi L_A}{L}\right)$"
    )
    ax.set_ylabel(order_label)
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
                "order": order,
                "blocks_plotted": len(curve["x"]),
            }
            for curve in curves
        ]
    )
    return {
        "figure": fig,
        "axis": ax,
        "vary": vary,
        "fixed": {fixed_key: fixed_value},
        "order": order,
        "mean_column": mean_column,
        "uncertainty_column": uncertainty_column,
        "fits": pd.DataFrame(fit_rows),
        "metadata": metadata,
        "curves": curves,
    }

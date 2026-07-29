"""Aggregate fermionic three-mode expectation values."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import _show


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

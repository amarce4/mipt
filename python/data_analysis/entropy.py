"""Entanglement-entropy scaling against conformal chord distance.

``entropy.exe`` reports both the von Neumann entropy S1 = -Tr(rho ln rho) and the
second Renyi entropy S2 = -ln Tr(rho^2) for every block, so ``order`` selects
which one is plotted and fitted.

A size scan at fixed measurement rate also gets a secondary panel -- an inset
by default -- carrying the fitted log coefficient against ``1/L^2`` and its
``alpha(L) = alpha(inf) + a L^-2`` extrapolation.  Thus the thermodynamic limit
is the plotted intercept at ``1/L^2 = 0``, rather than an off-axis implication.

``entropy.exe`` writes both entropies in nats; ``units`` converts them, and
defaults to bits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fitting import _format_with_uncertainty, _weighted_linear_fit
from .loading import _resolve_files
from .plotting import (
    _annotate_axes,
    _font_kwargs,
    _inset_overlay_defaults,
    _mi_unit_spec,
    _paired_axes,
    _reserve_axis_space,
    _set_secondary_title,
    _show,
)

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


def _extrapolate_log_coefficient(
    sizes: Sequence[float],
    alphas: Sequence[float],
    alpha_errors: Sequence[float],
) -> dict[str, float]:
    """Fit ``alpha(L) = alpha(inf) + a L^-2`` to the per-size log coefficients.

    Leading finite-size corrections to the conformal log coefficient are taken
    to be ``L^-2``, so the fit is linear in ``L^-2`` and the extrapolated
    coefficient is its intercept. ``_weighted_linear_fit`` needs three valid
    points, which is also the minimum that leaves the fit any residual.
    """
    fit = _weighted_linear_fit(
        np.asarray(sizes, dtype=float) ** -2.0, alphas, alpha_errors
    )
    return {
        "alpha_inf": fit["intercept"],
        "alpha_inf_stderr": fit["intercept_stderr"],
        "a": fit["slope"],
        "a_stderr": fit["slope_stderr"],
        "reduced_chi2": fit["reduced_chi2"],
        "points": int(np.size(sizes)),
    }


def _draw_extrapolation_panel(
    ax,
    points: Sequence[dict[str, Any]],
    fit: dict[str, float] | None,
    *,
    inset: bool,
    fontsize: float,
    facecolor: str,
    alpha: float,
    annotation_text: str,
    annotation_loc: str,
    annotation_fontsize: float | None,
    annotation_headroom: float,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    units: str,
    capsize: float,
) -> None:
    """Draw ``alpha`` against ``1/L^2`` with its thermodynamic extrapolation.

    The markers reuse each curve's colour from the primary panel, and the fit
    is a thin black dashed line drawn over the measured range.
    """
    font = _font_kwargs(inset, fontsize)
    for point in points:
        inverse_square = float(point["L"]) ** -2.0
        ax.errorbar(
            inverse_square,
            point["alpha"],
            yerr=point["alpha_stderr"],
            fmt="o",
            markersize=3.6,
            color=point["color"],
            ecolor=point["color"],
            elinewidth=0.9,
            capsize=capsize,
        )
    if fit is not None:
        sizes = np.asarray([point["L"] for point in points], dtype=float)
        inverse_square = sizes**-2.0
        grid = np.linspace(0.0, float(np.max(inverse_square)) * 1.03, 200)
        ax.plot(
            grid,
            fit["alpha_inf"] + fit["a"] * grid,
            linestyle="--",
            linewidth=0.9,
            color="black",
        )
        # The paper's large-size panels make the extrapolated limit a datum in
        # the figure.  It is visually distinct from the measured, size-coloured
        # points and sits exactly on the fitted intercept.
        ax.plot(
            0.0,
            fit["alpha_inf"],
            marker="o",
            markersize=3.8,
            color="black",
            clip_on=False,
            zorder=6,
        )
    ax.set_xlabel(xlabel if xlabel is not None else r"$1/L^2$", **font)
    # alpha is the coefficient of a plain logarithm, so it carries the entropy's
    # units: alpha[bits] = alpha[nats]/ln 2.
    ax.set_ylabel(
        ylabel if ylabel is not None else rf"$\alpha(L)$ [{units}]", **font
    )
    if inset:
        # The x label has the gap below the inset to live in; the y label sits
        # outside the left spine, which for a right-flush inset is over the
        # parent's data. The parent's curves rise diagonally, so they cross
        # that edge somewhere in the inset's y span whatever the headroom --
        # give the label the inset's own face to read against.
        ax.yaxis.label.set_bbox(
            {
                "boxstyle": "square,pad=0.15",
                "facecolor": facecolor,
                "edgecolor": "none",
                "alpha": alpha,
            }
        )
    ax.grid(alpha=0.25)
    inverse_squares = np.asarray(
        [float(point["L"]) ** -2.0 for point in points], dtype=float
    )
    if inverse_squares.size:
        ax.set_xlim(0.0, float(np.max(inverse_squares)) * 1.08)
    # Room at the bottom for the annotation box, which sits over the corner
    # the smallest sizes leave empty.
    _reserve_axis_space(ax, "bottom", annotation_headroom)
    annotation_kwargs: dict[str, Any] = {}
    if annotation_fontsize is not None:
        annotation_kwargs["fontsize"] = annotation_fontsize
    elif inset:
        annotation_kwargs["fontsize"] = fontsize
    _annotate_axes(ax, annotation_text, annotation_loc, **annotation_kwargs)
    # After _set_secondary_title, which reinstalls the inset tick locators,
    # keep scientific notation compact (normally a few times 10^-3) and make
    # the thermodynamic intercept an explicit tick.
    _set_secondary_title(ax, title, inset=inset, fontsize=fontsize + 1.0)
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(formatter)


def entropy(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    vary: str | None = None,
    order: int = 2,
    units: str = "bits",
    fit_above_p: float = 0.3,
    exclude_first: int = 1,
    uncertainty: str = "stderr",
    xlim: tuple[float, float] | None = (0.5, 2.05),
    ylim: tuple[float, float] | None = None,
    cmap: str = "viridis",
    extrapolate: bool = True,
    extrapolation_inset: bool = True,
    inset_corner: str = "lower right",
    inset_size: tuple[float, float] = (0.42, 0.40),
    inset_pad: tuple[float, float] = (0.0, 0.155),
    inset_fontsize: float = 7.5,
    inset_facecolor: str = "white",
    inset_alpha: float = 1.0,
    # Larger than the probe figures use: this panel's curves are straight
    # lines rising into the bottom-right corner, so they need more room than
    # a saturating one to clear the inset.
    inset_headroom: float = 0.38,
    inset_title: str | None = None,
    inset_xlabel: str | None = None,
    inset_ylabel: str | None = None,
    inset_capsize: float = 2.0,
    annotation_loc: str = "lower right",
    annotation_fontsize: float | None = None,
    annotation_headroom: float = 0.30,
    legend_loc: str | None = None,
    legend_fontsize: float = 8.0,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
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
    units:
        ``"bits"`` (the default) or ``"nats"``. ``entropy.exe`` writes nats, so
        bits divides through by ``ln 2``. The unit is appended to the y label,
        and the fitted log coefficient ``alpha`` -- being the coefficient of a
        plain logarithm -- is quoted in it too, in the legend, in the
        extrapolation panel and in ``fits``. Note that
        :func:`free_energy_ceff` reads ``alpha`` in **nats** unless told
        otherwise; ``fits`` also carries ``slope_nats`` for that.
    uncertainty:
        ``"stderr"`` or ``"stddev"``, which follow ``order``, or an explicit
        column name.
    extrapolate:
        Draw a secondary panel of the fitted log coefficient against
        ``1/L^2`` and extrapolate it with
        ``alpha(L) = alpha(inf) + a L^-2``. It
        needs a size scan at fixed ``p`` (``vary="L"``) with at least three
        fitted curves, and is skipped silently otherwise.
    extrapolation_inset:
        Draw that panel as an inset in the bottom-right corner of the main
        axes, lifted clear of the bottom edge so both x axes stay readable.
        ``False`` puts it beside the main panel instead. ``inset_size`` and
        ``inset_pad`` are fractions of the parent axes.
    annotation_loc, annotation_headroom:
        Where the parameter box -- the fixed measurement rate and the
        extrapolated ``alpha(inf)`` -- sits inside the extrapolation panel,
        and how much of that panel's y range is reserved for it. Without an
        extrapolation panel the box keeps its historical place in the
        bottom-right corner of the main axes and carries the fixed parameter
        only.
    """
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {sorted(_ORDERS)}, got {order!r}.")
    if vary is not None and vary not in _VARY_KEYS:
        raise ValueError(f"vary must be one of {sorted(_VARY_KEYS)} or None, got {vary!r}.")
    units, unit_scale = _mi_unit_spec(units, name="units")
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
        # The stored entropies are nats. Both the mean and its spread scale
        # linearly, so the fit and its chi^2 are unchanged by the conversion.
        y = df[mean_column].to_numpy(dtype=float) * unit_scale
        dy = df[uncertainty_column].to_numpy(dtype=float) * unit_scale
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

    # The extrapolation needs one alpha per size, so it is only defined for a
    # size scan. Whether it survives is decided after the fits are in hand --
    # a curve can still fail its own fit -- but the axes have to exist first.
    want_extrapolation = extrapolate and vary == "L" and len(curves) >= 3
    if figsize is None:
        figsize = (10, 6) if extrapolation_inset or not want_extrapolation else (15, 6)
    if dpi is None:
        dpi = plt.rcParams["figure.dpi"]
    auto_legend_loc, _, reserve_side = _inset_overlay_defaults(inset_corner)
    if want_extrapolation:
        fig, panels = _paired_axes(
            inset=extrapolation_inset,
            figsize=figsize,
            dpi=dpi,
            inset_corner=inset_corner,
            inset_size=inset_size,
            inset_pad=inset_pad,
            inset_fontsize=inset_fontsize,
            inset_facecolor=inset_facecolor,
            inset_alpha=inset_alpha,
        )
        ax, ax_extrapolation = panels[0]
    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax_extrapolation = None
    if legend_loc is None:
        legend_loc = (
            auto_legend_loc
            if want_extrapolation and extrapolation_inset
            else "upper left"
        )
    colors = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, len(curves)))
    fit_rows: list[dict[str, Any]] = []
    extrapolation_points: list[dict[str, Any]] = []

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
                # free_energy_ceff's c_eff formula is written in nats; keep the
                # unconverted slope so a bits figure can still feed it.
                fit_record["slope_nats"] = fit["slope"] / unit_scale
                fit_record["slope_nats_stderr"] = slope_error / unit_scale
                extrapolation_points.append(
                    {
                        "L": float(curve["L"]),
                        "alpha": fit["slope"],
                        "alpha_stderr": slope_error,
                        "color": color,
                    }
                )
            except ValueError as exc:
                data_line.set_label(rf"{curve_label}: fit unavailable ({exc})")
                fit_record["error"] = str(exc)
        else:
            data_line.set_label(curve_label)
        fit_rows.append(fit_record)

    extrapolation_points.sort(key=lambda point: point["L"])
    extrapolation: dict[str, float] | None = None
    if ax_extrapolation is not None and len(extrapolation_points) >= 3:
        try:
            extrapolation = _extrapolate_log_coefficient(
                [point["L"] for point in extrapolation_points],
                [point["alpha"] for point in extrapolation_points],
                [point["alpha_stderr"] for point in extrapolation_points],
            )
        except ValueError:
            extrapolation = None
    elif ax_extrapolation is not None:
        # Too few fitted curves to extrapolate: drop the panel rather than
        # leave an empty frame over the data.
        ax_extrapolation.remove()
        ax_extrapolation = None

    annotation_text = fixed_label(fixed_value)  # + "\n" + order_name
    if extrapolation is not None:
        alpha_str, alpha_stderr_str = _format_with_uncertainty(
            extrapolation["alpha_inf"], extrapolation["alpha_inf_stderr"]
        )
        annotation_text += (
            "\n" + rf"$\alpha(\infty)={alpha_str}\pm {alpha_stderr_str}$"
        )
    if ax_extrapolation is None:
        ax.text(
            0.99,
            0.02,
            annotation_text,
            transform=ax.transAxes,
            fontsize=12,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.7},
        )
    ax.set_xlabel(
        r"$\ln x$, $x=\frac{L}{\pi}\sin\left(\frac{\pi L_A}{L}\right)$"
    )
    ax.set_ylabel(rf"{order_label} [{units}]")
    ax.legend(fontsize=legend_fontsize, framealpha=0.9, loc=legend_loc)
    ax.grid(alpha=0.25)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if ax_extrapolation is not None:
        if extrapolation_inset and ylim is None:
            _reserve_axis_space(ax, reserve_side, inset_headroom)
        _draw_extrapolation_panel(
            ax_extrapolation,
            extrapolation_points,
            extrapolation,
            inset=extrapolation_inset,
            fontsize=inset_fontsize,
            facecolor=inset_facecolor,
            alpha=inset_alpha,
            annotation_text=annotation_text,
            annotation_loc=annotation_loc,
            annotation_fontsize=annotation_fontsize,
            annotation_headroom=annotation_headroom,
            title=inset_title,
            xlabel=inset_xlabel,
            ylabel=inset_ylabel,
            units=units,
            capsize=inset_capsize,
        )
    elif not want_extrapolation:
        # _paired_axes lays its figures out with constrained_layout, which
        # tight_layout would fight even after the panel has been removed.
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
        "extrapolation_axis": ax_extrapolation,
        "extrapolation": extrapolation,
        "vary": vary,
        "fixed": {fixed_key: fixed_value},
        "order": order,
        "units": units,
        "mean_column": mean_column,
        "uncertainty_column": uncertainty_column,
        "fits": pd.DataFrame(fit_rows),
        "metadata": metadata,
        "curves": curves,
    }

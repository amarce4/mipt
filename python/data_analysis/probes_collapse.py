"""Ancilla-probe finite-size collapses for one, two, and four references."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .collapse import (
    _probe2_metric_spec,
    _probe4_metric_spec,
    bulk_exponent_collapse,
    load_scaled_time_curve,
)
from .loading import _find_column, _parse_size, _resolve_files
from .plotting import (
    _annotate_axes,
    _color_map_by_size,
    _font_kwargs,
    _inset_overlay_defaults,
    _set_secondary_title,
    _mi_unit_spec,
    _paired_axes,
    _reserve_axis_space,
    _show,
)


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
    collapse_inset: bool = True,
    show_fit_window: bool = False,
    inset_corner: str = "upper right",
    inset_size: tuple[float, float] = (0.40, 0.38),
    inset_pad: tuple[float, float] = (0.0, 0.0),
    inset_fontsize: float = 7.5,
    inset_facecolor: str = "white",
    inset_alpha: float = 1.0,
    inset_headroom: float = 0.20,
    annotation_loc: str | None = None,
    annotation_fontsize: float | None = None,
    legend_loc: str | None = None,
    legend_fontsize: float | None = None,
    legend_ncols: int = 1,
    raw_title: str | None = "Critical ancilla dynamics",
    collapse_title: str | None = "Finite-size scaling collapse",
    raw_xlabel: str | None = None,
    raw_ylabel: str | None = None,
    collapse_xlabel: str | None = None,
    collapse_ylabel: str | None = None,
    suptitle: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 130,
    show: bool = True,
) -> dict[str, Any]:
    """Fit S_A(L,t)=L^(-x_A)F(t/L^z) for one ancilla probe.

    The collapse is drawn by default as an inset in the top-right corner of
    the raw panel, sharing the raw panel's single legend. ``inset_size`` and
    ``inset_pad`` are fractions of the parent axes. ``collapse_inset=False``
    restores the side-by-side layout with a legend on each panel, and
    ``show_fit_window=True`` restores the shaded fit-window band.
    """
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
    if figsize is None:
        figsize = (7.6, 5.2) if collapse_inset else (13, 5)
    fig, panels = _paired_axes(
        inset=collapse_inset,
        figsize=figsize,
        dpi=dpi,
        inset_corner=inset_corner,
        inset_size=inset_size,
        inset_pad=inset_pad,
        inset_fontsize=inset_fontsize,
        inset_facecolor=inset_facecolor,
        inset_alpha=inset_alpha,
    )
    ax_raw, ax_collapse = panels[0]
    inset_font = _font_kwargs(collapse_inset, inset_fontsize)
    auto_legend_loc, auto_annotation_loc, reserve_side = _inset_overlay_defaults(
        inset_corner
    )
    if legend_loc is None:
        legend_loc = auto_legend_loc if collapse_inset else "best"
    if annotation_loc is None:
        annotation_loc = auto_annotation_loc if collapse_inset else "upper right"
    loaded_t_max = max(np.max(curve["t"]) for curve in curves)
    fit_upper = min(fit_t_max, loaded_t_max) if fit_t_max is not None else loaded_t_max
    if show_fit_window:
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

    ax_raw.set_xlabel(raw_xlabel if raw_xlabel is not None else r"Time $t$")
    ax_raw.set_ylabel(
        raw_ylabel
        if raw_ylabel is not None
        else r"Ancilla entropy $\overline{S_A}$"
    )
    if raw_title:
        ax_raw.set_title(raw_title)
    ax_raw.set_xscale(raw_xscale)
    ax_raw.grid(alpha=0.25)
    legend_kwargs: dict[str, Any] = {"loc": legend_loc, "ncols": legend_ncols}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    handles, labels = ax_raw.get_legend_handles_labels()
    order = [i for i, label in enumerate(labels) if label == "fit window"] + [
        i for i, label in enumerate(labels) if label != "fit window"
    ]
    ax_raw.legend(
        [handles[i] for i in order], [labels[i] for i in order], **legend_kwargs
    )

    ax_collapse.set_xlabel(
        collapse_xlabel if collapse_xlabel is not None else r"Scaled time $t/L^z$",
        **inset_font,
    )
    default_collapse_ylabel = (
        r"$\overline{S_A}$"
        if abs(best_x_a) < 1e-12
        else r"$L^{x_A}\overline{S_A}$"
    )
    if collapse_ylabel is not None or not collapse_inset:
        ax_collapse.set_ylabel(
            collapse_ylabel
            if collapse_ylabel is not None
            else default_collapse_ylabel,
            **inset_font,
        )
    ax_collapse.set_xscale(collapse_xscale)
    ax_collapse.grid(alpha=0.25)
    if not collapse_inset:
        ax_collapse.legend(**legend_kwargs)
    _set_secondary_title(
        ax_collapse,
        collapse_title,
        inset=collapse_inset,
        fontsize=inset_fontsize + 1.0,
    )
    x_a_text = (
        rf"$x_A={best_x_a:.4f}\pm {x_a_stderr:.4f}$"
        if fixed_x_a is None
        else rf"$x_A={best_x_a:.4f}$ fixed"
    )
    annotation_kwargs: dict[str, Any] = {}
    if annotation_fontsize is not None:
        annotation_kwargs["fontsize"] = annotation_fontsize
    _annotate_axes(
        ax_raw if collapse_inset else ax_collapse,
        r"$\overline{S_A}=L^{-x_A}F(t/L^z)$"
        + "\n"
        + rf"$z={best_z:.4f}\pm {z_stderr:.4f}$"
        + "\n"
        + x_a_text
        + "\n"
        + rf"score $={best_score:.3g}$",
        annotation_loc,
        **annotation_kwargs,
    )
    if collapse_inset:
        _reserve_axis_space(ax_raw, reserve_side, inset_headroom)
    fig.suptitle(
        suptitle
        if suptitle is not None
        else "Ancilla probe at the measurement-induced critical point",
        fontsize=14,
    )
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_raw, ax_collapse),
        "collapse_inset": collapse_inset,
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
# Two- and four-probe bulk-exponent collapses
#
# Both are the same one-parameter fit on the same file layout, so they only
# resolve their metric spec and hand the curves to the shared engine.
# ---------------------------------------------------------------------------


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
    collapse_inset: bool = True,
    show_fit_window: bool = False,
    inset_corner: str = "lower right",
    inset_size: tuple[float, float] = (0.42, 0.40),
    inset_pad: tuple[float, float] = (0.0, 0.185),
    inset_fontsize: float = 7.5,
    inset_facecolor: str = "white",
    inset_alpha: float = 1.0,
    inset_headroom: float = 0.20,
    annotation_loc: str | None = None,
    annotation_fontsize: float | None = None,
    legend_loc: str | None = None,
    legend_fontsize: float | None = None,
    legend_ncols: int = 1,
    raw_title: str | None = None,
    collapse_title: str | None = "Bulk-exponent scaling collapse",
    raw_xlabel: str | None = None,
    raw_ylabel: str | None = None,
    collapse_xlabel: str | None = None,
    collapse_ylabel: str | None = None,
    suptitle: str | None = None,
    figsize: tuple[float, float] | None = None,
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

    The collapse is drawn by default as an inset in the bottom-right corner
    of the raw panel, both panels sharing the raw panel's legend, and the
    fit-window band is off. ``collapse_inset=False`` restores the
    side-by-side layout; ``show_fit_window=True`` restores the band.
    """
    metric_spec = _probe2_metric_spec(metric)
    mi_units, mi_scale = _mi_unit_spec(mi_units)
    curves = [
        load_scaled_time_curve(
            path,
            l_by_file=l_by_file,
            x_load=x_load,
            metric_spec=metric_spec,
        )
        for path in _resolve_files(files, file_glob)
    ]
    return bulk_exponent_collapse(
        curves,
        metric_spec,
        mi_units=mi_units,
        mi_scale=mi_scale,
        fit_x=fit_x,
        fit_min=fit_i_min,
        eta_bounds=eta_bounds,
        interpolation_points=interpolation_points,
        relative_error_floor=relative_error_floor,
        absolute_error_floor=absolute_error_floor,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_xtol=bootstrap_xtol,
        raw_yscale=raw_yscale,
        collapse_yscale=collapse_yscale,
        show_errorbars=show_errorbars,
        capsize=capsize,
        cmap=cmap,
        figsize=figsize,
        dpi=dpi,
        show_summary=show_summary,
        show=show,
        collapse_inset=collapse_inset,
        show_fit_window=show_fit_window,
        inset_corner=inset_corner,
        inset_size=inset_size,
        inset_pad=inset_pad,
        inset_fontsize=inset_fontsize,
        inset_facecolor=inset_facecolor,
        inset_alpha=inset_alpha,
        inset_headroom=inset_headroom,
        annotation_loc=annotation_loc,
        annotation_fontsize=annotation_fontsize,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        legend_ncols=legend_ncols,
        raw_title=raw_title,
        collapse_title=collapse_title,
        raw_xlabel=raw_xlabel,
        raw_ylabel=raw_ylabel,
        collapse_xlabel=collapse_xlabel,
        collapse_ylabel=collapse_ylabel,
        suptitle=suptitle,
    )


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
    collapse_inset: bool = True,
    show_fit_window: bool = False,
    inset_corner: str = "lower right",
    inset_size: tuple[float, float] = (0.42, 0.40),
    inset_pad: tuple[float, float] = (0.0, 0.185),
    inset_fontsize: float = 7.5,
    inset_facecolor: str = "white",
    inset_alpha: float = 1.0,
    inset_headroom: float = 0.20,
    annotation_loc: str | None = None,
    annotation_fontsize: float | None = None,
    legend_loc: str | None = None,
    legend_fontsize: float | None = None,
    legend_ncols: int = 1,
    raw_title: str | None = None,
    collapse_title: str | None = "Bulk-exponent scaling collapse",
    raw_xlabel: str | None = None,
    raw_ylabel: str | None = None,
    collapse_xlabel: str | None = None,
    collapse_ylabel: str | None = None,
    suptitle: str | None = None,
    figsize: tuple[float, float] | None = None,
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

    The layout controls match :func:`probe2_collapse`: the collapse is an
    inset in the bottom-right corner of the raw panel by default, the two
    panels share one legend, and the fit-window band is off.
    """
    if fit_abs_min < 0.0:
        raise ValueError("fit_abs_min must be non-negative.")
    metric_spec = _probe4_metric_spec(metric)
    mi_units, mi_scale = _mi_unit_spec(mi_units)
    curves = [
        load_scaled_time_curve(
            path,
            l_by_file=l_by_file,
            x_load=x_load,
            metric_spec=metric_spec,
        )
        for path in _resolve_files(files, file_glob)
    ]
    return bulk_exponent_collapse(
        curves,
        metric_spec,
        mi_units=mi_units,
        mi_scale=mi_scale,
        fit_x=fit_x,
        fit_min=fit_abs_min,
        eta_bounds=eta_bounds,
        interpolation_points=interpolation_points,
        relative_error_floor=relative_error_floor,
        absolute_error_floor=absolute_error_floor,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_xtol=bootstrap_xtol,
        raw_yscale=raw_yscale,
        collapse_yscale=collapse_yscale,
        show_errorbars=show_errorbars,
        capsize=capsize,
        cmap=cmap,
        figsize=figsize,
        dpi=dpi,
        show_summary=show_summary,
        show=show,
        collapse_inset=collapse_inset,
        show_fit_window=show_fit_window,
        inset_corner=inset_corner,
        inset_size=inset_size,
        inset_pad=inset_pad,
        inset_fontsize=inset_fontsize,
        inset_facecolor=inset_facecolor,
        inset_alpha=inset_alpha,
        inset_headroom=inset_headroom,
        annotation_loc=annotation_loc,
        annotation_fontsize=annotation_fontsize,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        legend_ncols=legend_ncols,
        raw_title=raw_title,
        collapse_title=collapse_title,
        raw_xlabel=raw_xlabel,
        raw_ylabel=raw_ylabel,
        collapse_xlabel=collapse_xlabel,
        collapse_ylabel=collapse_ylabel,
        suptitle=suptitle,
    )

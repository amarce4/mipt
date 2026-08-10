"""One-parameter finite-size collapse of probe observables.

Every bulk-exponent collapse in this package fits the same ansatz

    O(L, t) = L^(-eta) g((t - 2L) / L),

and differs only in which CSV columns hold the observable, how the observable
is labelled, and whether it may be negative. Those differences live in the
metric specs below; the loader and the fit itself are shared.
"""

from __future__ import annotations

from collections import abc
from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .fitting import _format_with_uncertainty
from .loading import _find_column, resolve_metadata
from .plotting import (
    _annotate_axes,
    _color_map_by_size,
    _flush_stacked_axes,
    _font_kwargs,
    _inset_overlay_defaults,
    _prune_upper_yticks,
    _set_secondary_title,
    _paired_axes,
    _reserve_axis_space,
    _show,
)


# ---------------------------------------------------------------------------
# Metric specs
#
# Fields consumed by the engine:
#   key, mean_columns, stderr_columns   CSV lookup
#   raw_ylabel, raw_title,              labelling
#   collapse_ylabel, ansatz, suptitle, eta_symbol
#   summary_name                        summary column prefix (unsigned only)
#   signed                              observable may be negative: fit on
#                                       |O|, show a zero line, and report a
#                                       signed summary
#   scale_with_mi_units                 observable is an information measure,
#                                       so mi_units rescales it
#   bootstrap_lower / bootstrap_upper   resampling clip, None for unbounded
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
            "scale_with_mi_units": True,
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
            "scale_with_mi_units": False,
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
            "scale_with_mi_units": False,
        },
    }
    if key not in specs:
        raise ValueError(
            "metric must be 'mi', 'negativity', or 'log_negativity'."
        )
    return {
        "key": key,
        "signed": False,
        "eta_symbol": r"\eta",
        "bootstrap_lower": 0.0,
        **specs[key],
    }


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
            "bootstrap_lower": 0.0,
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
            "bootstrap_lower": None,
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
            "bootstrap_lower": None,
        },
    }
    if key not in specs:
        raise ValueError("metric must be 'i2', 'i3', or 'i4'.")
    return {
        "key": key,
        "signed": True,
        "summary_name": key,
        "scale_with_mi_units": True,
        "bootstrap_upper": None,
        **specs[key],
    }


# ---------------------------------------------------------------------------
# Shared loader
# ---------------------------------------------------------------------------


def load_scaled_time_curve(
    path: Path,
    *,
    l_by_file: Mapping[str, int] | None,
    x_load: tuple[float | None, float | None],
    metric_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one probe time series and attach the scaled time x=(t-2L)/L."""
    df = pd.read_csv(path)
    metadata = resolve_metadata(df, path, l_by_file=l_by_file)
    size = metadata["L"]

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
        "p": metadata["p"],
        "circuit": metadata["circuit"],
        "metadata_source": metadata["source"],
        "metric": metric_spec["key"],
        "t": clean["t"].to_numpy(dtype=float),
        "x": clean["x"].to_numpy(dtype=float),
        "value": clean["value"].to_numpy(dtype=float),
        "dvalue": clean["dvalue"].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------------------
# Shared collapse engine
# ---------------------------------------------------------------------------


def _collapse_summary(
    curves: Sequence[Mapping[str, Any]],
    metric_spec: Mapping[str, Any],
) -> pd.DataFrame:
    """Per-size overview of the loaded curves."""
    common = {
        "L": [curve["L"] for curve in curves],
        "metric": metric_spec["key"],
        "p_from_filename": [curve["p"] for curve in curves],
        "points": [len(curve["x"]) for curve in curves],
        "x_min": [np.min(curve["x"]) for curve in curves],
        "x_max": [np.max(curve["x"]) for curve in curves],
    }
    if metric_spec["signed"]:
        extrema = {
            "value_min": [np.min(curve["value"]) for curve in curves],
            "value_max": [np.max(curve["value"]) for curve in curves],
            "abs_max": [np.max(np.abs(curve["value"])) for curve in curves],
            "x_at_abs_max": [
                curve["x"][np.argmax(np.abs(curve["value"]))] for curve in curves
            ],
        }
    else:
        name = metric_spec["summary_name"]
        extrema = {
            f"{name}_max": [np.max(curve["value"]) for curve in curves],
            f"x_at_{name}_max": [
                curve["x"][np.argmax(curve["value"])] for curve in curves
            ],
        }
    return pd.DataFrame(
        {**common, **extrema, "file": [str(curve["path"]) for curve in curves]}
    )


def _fit_bulk_exponent(
    curves: list[dict[str, Any]],
    metric_spec: Mapping[str, Any],
    *,
    mi_scale: float,
    fit_x: tuple[float, float | None],
    fit_min: float,
    eta_bounds: tuple[float, float],
    interpolation_points: int,
    relative_error_floor: float,
    absolute_error_floor: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    bootstrap_xtol: float,
    show_summary: bool,
    progress_label: str | None = None,
) -> dict[str, Any]:
    """Fit ``eta`` in ``O(L,t)=L^(-eta)g((t-2L)/L)`` for one observable.

    ``curves`` come from :func:`load_scaled_time_curve` and are sorted and, for
    an information measure, rescaled by ``mi_scale`` in place. The returned
    dictionary carries everything the drawing stage needs, including the
    ``fit_mask`` closure that selects the fitted points.
    """
    signed = bool(metric_spec["signed"])
    metric_spec = dict(metric_spec)
    if metric_spec["scale_with_mi_units"]:
        for curve in curves:
            curve["value"] *= mi_scale
            curve["dvalue"] *= mi_scale
        if metric_spec["bootstrap_upper"] is not None:
            metric_spec["bootstrap_upper"] *= mi_scale

    curves.sort(key=lambda curve: curve["L"])
    sizes = [curve["L"] for curve in curves]
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"Duplicate system sizes found: {sizes}")
    if len(curves) < 3:
        warnings.warn("A collapse with fewer than three sizes is weakly constrained.")

    finite_ps = [curve["p"] for curve in curves if np.isfinite(curve["p"])]
    if finite_ps and not np.allclose(finite_ps, finite_ps[0], rtol=0, atol=1e-12):
        warnings.warn(f"The files contain multiple p values: {sorted(set(finite_ps))}")

    summary = _collapse_summary(curves, metric_spec)
    if show_summary:
        try:
            from IPython.display import display

            display(summary)
        except ImportError:
            print(summary.to_string(index=False))

    prefix = f"[{progress_label}] " if progress_label else ""
    fit_x_min, fit_x_max = fit_x

    def fit_mask(curve):
        selected = np.abs(curve["value"]) if signed else curve["value"]
        mask = (
            np.isfinite(curve["x"])
            & np.isfinite(curve["value"])
            & np.isfinite(curve["dvalue"])
            & (curve["x"] >= fit_x_min)
            & (selected > fit_min)
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

    lower, upper = metric_spec["bootstrap_lower"], metric_spec["bootstrap_upper"]
    rng = np.random.default_rng(bootstrap_seed)
    estimates: list[float] = []
    if bootstrap_samples >= 2:
        for index in range(bootstrap_samples):
            synthetic_curves = []
            for curve in curves:
                synthetic = dict(curve)
                values = rng.normal(
                    curve["value"], np.maximum(curve["dvalue"], 0.0)
                )
                if lower is not None:
                    values = np.maximum(values, lower)
                if upper is not None:
                    values = np.minimum(values, upper)
                synthetic["value"] = values
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
                print(f"{prefix}Bootstrap fits: {index + 1}/{bootstrap_samples}")

    if len(estimates) >= 2:
        estimate_array = np.asarray(estimates)
        eta_stderr = float(np.std(estimate_array, ddof=1))
        eta_p16, eta_p84 = map(float, np.percentile(estimate_array, [16, 84]))
    else:
        eta_stderr = eta_p16 = eta_p84 = np.nan
        if bootstrap_samples >= 2:
            warnings.warn("Too few successful bootstrap fits to estimate eta uncertainty.")

    print(f"{prefix}metric = {metric_spec['key']}")
    print(f"{prefix}eta = {eta:.6f} ± {eta_stderr:.6f}")
    if np.isfinite(eta_p16):
        print(f"{prefix}bootstrap 68% interval = [{eta_p16:.6f}, {eta_p84:.6f}]")
    print(f"{prefix}collapse score = {best_score:.6g}")

    return {
        "metric_spec": metric_spec,
        "curves": curves,
        "sizes": sizes,
        "signed": signed,
        "fit_mask": fit_mask,
        "summary": summary,
        "eta": eta,
        "eta_stderr": eta_stderr,
        "eta_p16": eta_p16,
        "eta_p84": eta_p84,
        "score": best_score,
        "bootstrap_successes": len(estimates),
        "fit_result": fit_result,
    }


def _draw_bulk_exponent_panel(
    ax_raw,
    ax_collapse,
    fit: Mapping[str, Any],
    *,
    colors: Mapping[int, Any],
    mi_units: str,
    fit_x: tuple[float, float | None],
    show_errorbars: bool,
    capsize: float,
    raw_yscale: str,
    collapse_yscale: str,
    collapse_inset: bool,
    show_fit_window: bool,
    inset_fontsize: float,
    inset_headroom: float,
    reserve_side: str,
    annotation_loc: str,
    annotation_fontsize: float | None,
    legend_loc: str,
    legend_fontsize: float | None,
    legend_ncols: int,
    draw_legend: bool,
    raw_title: str | None,
    collapse_title: str | None,
    raw_xlabel: str | None,
    raw_ylabel: str | None,
    collapse_xlabel: str | None,
    collapse_ylabel: str | None,
) -> None:
    """Draw one raw/collapse panel pair from a :func:`_fit_bulk_exponent` result.

    ``draw_legend=False`` leaves the panel bare so that a stack of them can
    share one legend; everything else is per panel, including the annotated
    exponent.
    """
    metric_spec = fit["metric_spec"]
    curves, signed = fit["curves"], fit["signed"]
    fit_mask, eta = fit["fit_mask"], fit["eta"]
    eta_stderr, best_score = fit["eta_stderr"], fit["score"]
    fit_x_min, fit_x_max = fit_x
    inset_font = _font_kwargs(collapse_inset, inset_fontsize)
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

    if signed:
        ax_raw.axhline(0.0, linewidth=1, alpha=0.35)
    ax_raw.axvline(0.0, linewidth=1, linestyle="--", alpha=0.6)
    fit_upper = (
        fit_x_max
        if fit_x_max is not None
        else max(curve["x"].max() for curve in curves)
    )
    if show_fit_window:
        ax_raw.axvspan(fit_x_min, fit_upper, alpha=0.08, label="fit window")
    default_raw_ylabel = metric_spec["raw_ylabel"]
    default_collapse_ylabel = metric_spec["collapse_ylabel"]
    if metric_spec["scale_with_mi_units"]:
        default_raw_ylabel = default_raw_ylabel.replace("[nats]", f"[{mi_units}]")
        default_collapse_ylabel = default_collapse_ylabel + f" [{mi_units}]"
    scaled_time_label = r"Scaled time $x=(t-2L)/L$"
    if raw_xlabel != "":
        ax_raw.set_xlabel(
            raw_xlabel if raw_xlabel is not None else scaled_time_label
        )
    ax_raw.set_ylabel(
        raw_ylabel if raw_ylabel is not None else default_raw_ylabel
    )
    # An empty title is a request for no title at all, not for a blank one:
    # a stacked figure names its metrics on the y axes and cannot afford the
    # vertical space a per-panel title would take.
    panel_title = raw_title if raw_title is not None else metric_spec["raw_title"]
    if panel_title:
        ax_raw.set_title(panel_title)
    ax_raw.set_yscale(raw_yscale)
    ax_raw.grid(alpha=0.25)
    legend_kwargs: dict[str, Any] = {"loc": legend_loc, "ncols": legend_ncols, "bbox_to_anchor": (0.45, 0.0)}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    if draw_legend:
        ax_raw.legend(**legend_kwargs)

    if signed:
        ax_collapse.axhline(0.0, linewidth=1, alpha=0.35)
    if collapse_xlabel != "":
        ax_collapse.set_xlabel(
            collapse_xlabel if collapse_xlabel is not None else scaled_time_label,
            **inset_font,
        )
    if collapse_ylabel is not None or not collapse_inset:
        ax_collapse.set_ylabel(
            collapse_ylabel
            if collapse_ylabel is not None
            else default_collapse_ylabel,
            **inset_font,
        )
    ax_collapse.set_yscale(collapse_yscale)
    ax_collapse.grid(alpha=0.25)
    if not collapse_inset and draw_legend:
        ax_collapse.legend(**legend_kwargs)
    _set_secondary_title(
        ax_collapse,
        collapse_title,
        inset=collapse_inset,
        fontsize=inset_fontsize + 1.0,
    )
    symbol = metric_spec["eta_symbol"]
    # The error bar is quoted to one significant figure and eta is quoted to
    # that same decimal place, rather than both to a fixed number of decimals.
    eta_str, eta_stderr_str = _format_with_uncertainty(eta, eta_stderr)
    eta_text = (
        rf"${symbol}={eta_str}\pm {eta_stderr_str}$"
        if np.isfinite(eta_stderr)
        else rf"${symbol}={eta_str}$"
    )
    annotation_kwargs: dict[str, Any] = {}
    if annotation_fontsize is not None:
        annotation_kwargs["fontsize"] = annotation_fontsize
    _annotate_axes(
        ax_raw if collapse_inset else ax_collapse,
        # metric_spec["ansatz"]
        # + "\n"
        eta_text
        + "\n"
        + rf"score $={best_score:.3g}$",
        annotation_loc,
        **annotation_kwargs,
    )
    if collapse_inset:
        _reserve_axis_space(ax_raw, reserve_side, inset_headroom)


def _panel_result(fit: Mapping[str, Any], mi_units: str) -> dict[str, Any]:
    """The public per-observable half of a collapse result."""
    return {
        "metric": fit["metric_spec"]["key"],
        "mi_units": mi_units,
        "eta": fit["eta"],
        "eta_stderr": fit["eta_stderr"],
        "eta_p16": fit["eta_p16"],
        "eta_p84": fit["eta_p84"],
        "score": fit["score"],
        "bootstrap_successes": fit["bootstrap_successes"],
        "summary": fit["summary"],
        "curves": fit["curves"],
        "fit_result": fit["fit_result"],
    }


def bulk_exponent_collapse(
    curves: list[dict[str, Any]],
    metric_spec: Mapping[str, Any],
    *,
    mi_units: str,
    mi_scale: float,
    fit_x: tuple[float, float | None],
    fit_min: float,
    eta_bounds: tuple[float, float],
    interpolation_points: int,
    relative_error_floor: float,
    absolute_error_floor: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    bootstrap_xtol: float,
    raw_yscale: str,
    collapse_yscale: str,
    show_errorbars: bool,
    capsize: float,
    cmap: str,
    figsize: tuple[float, float] | None,
    dpi: int,
    show_summary: bool,
    show: bool,
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
) -> dict[str, Any]:
    """Fit and plot ``O(L,t)=L^(-eta)g((t-2L)/L)`` for a set of curves.

    ``curves`` come from :func:`load_scaled_time_curve` and are modified in
    place when the metric is rescaled by ``mi_scale``.

    The collapse is drawn by default as an inset in the bottom-right corner
    of the raw panel, lifted clear of the bottom edge so both x axes stay
    readable, and the two panels share the raw panel's legend.
    """
    fit = _fit_bulk_exponent(
        curves,
        metric_spec,
        mi_scale=mi_scale,
        fit_x=fit_x,
        fit_min=fit_min,
        eta_bounds=eta_bounds,
        interpolation_points=interpolation_points,
        relative_error_floor=relative_error_floor,
        absolute_error_floor=absolute_error_floor,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_xtol=bootstrap_xtol,
        show_summary=show_summary,
    )
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
    auto_legend_loc, auto_annotation_loc, reserve_side = _inset_overlay_defaults(
        inset_corner
    )
    if legend_loc is None:
        legend_loc = auto_legend_loc if collapse_inset else "best"
    if annotation_loc is None:
        annotation_loc = auto_annotation_loc if collapse_inset else "lower right"
    _draw_bulk_exponent_panel(
        ax_raw,
        ax_collapse,
        fit,
        colors=_color_map_by_size(fit["sizes"], cmap),
        mi_units=mi_units,
        fit_x=fit_x,
        show_errorbars=show_errorbars,
        capsize=capsize,
        raw_yscale=raw_yscale,
        collapse_yscale=collapse_yscale,
        collapse_inset=collapse_inset,
        show_fit_window=show_fit_window,
        inset_fontsize=inset_fontsize,
        inset_headroom=inset_headroom,
        reserve_side=reserve_side,
        annotation_loc=annotation_loc,
        annotation_fontsize=annotation_fontsize,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        legend_ncols=legend_ncols,
        draw_legend=True,
        raw_title=raw_title,
        collapse_title=collapse_title,
        raw_xlabel=raw_xlabel,
        raw_ylabel=raw_ylabel,
        collapse_xlabel=collapse_xlabel,
        collapse_ylabel=collapse_ylabel,
    )
    fig.suptitle(
        suptitle if suptitle is not None else fit["metric_spec"]["suptitle"],
        fontsize=14,
    )
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_raw, ax_collapse),
        "collapse_inset": collapse_inset,
        **_panel_result(fit, mi_units),
    }


# ---------------------------------------------------------------------------
# Stacked variant
#
# Several observables of the same run, one per row, sharing a single x axis
# with the row frames flush against each other. Every row is an independent
# one-parameter fit; only the layout and the legend are shared.
# ---------------------------------------------------------------------------


def _per_metric(value: Any, key: str) -> Any:
    """Resolve a style override that may be given per metric key."""
    if isinstance(value, abc.Mapping):
        return value.get(key)
    return value


def stacked_bulk_exponent_collapse(
    curve_sets: Sequence[list[dict[str, Any]]],
    metric_specs: Sequence[Mapping[str, Any]],
    *,
    mi_units: str,
    mi_scale: float,
    fit_x: tuple[float, float | None],
    fit_min: float,
    eta_bounds: tuple[float, float],
    interpolation_points: int,
    relative_error_floor: float,
    absolute_error_floor: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    bootstrap_xtol: float,
    raw_yscale: str | Mapping[str, str],
    collapse_yscale: str | Mapping[str, str],
    show_errorbars: bool,
    capsize: float,
    cmap: str,
    figsize: tuple[float, float] | None,
    dpi: int,
    show_summary: bool,
    show: bool,
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
    raw_title: str | Mapping[str, str] | None = None,
    collapse_title: str | Mapping[str, str] | None = "Bulk-exponent scaling collapse",
    raw_xlabel: str | None = None,
    raw_ylabel: str | Mapping[str, str] | None = None,
    collapse_xlabel: str | None = None,
    collapse_ylabel: str | Mapping[str, str] | None = None,
    suptitle: str | None = None,
    prune_shared_yticks: bool = True,
) -> dict[str, Any]:
    """Stack one bulk-exponent panel per observable, flush on a shared x axis.

    ``curve_sets[i]`` are the curves for ``metric_specs[i]``, in top-to-bottom
    row order. Each row is fitted independently and annotated with its own
    exponent; the rows share one x axis, one legend on the top row, and their
    frames touch, so the stack reads as a single plot.

    The overrides that name an observable — ``raw_title``, ``raw_ylabel``,
    ``collapse_title``, ``collapse_ylabel``, ``raw_yscale``, ``collapse_yscale``
    — accept either one value for every row or a mapping keyed by metric.
    ``raw_title`` defaults to no per-row title (the y labels carry the metric
    names, and a title between two flush rows would break the stack); when
    given it goes on the top row only.
    """
    if len(curve_sets) != len(metric_specs):
        raise ValueError("curve_sets and metric_specs must have the same length.")
    if not metric_specs:
        raise ValueError("At least one metric is required.")
    rows = len(metric_specs)

    fits = [
        _fit_bulk_exponent(
            curves,
            metric_spec,
            mi_scale=mi_scale,
            fit_x=fit_x,
            fit_min=fit_min,
            eta_bounds=eta_bounds,
            interpolation_points=interpolation_points,
            relative_error_floor=relative_error_floor,
            absolute_error_floor=absolute_error_floor,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_xtol=bootstrap_xtol,
            show_summary=show_summary,
            progress_label=metric_spec["key"],
        )
        for curves, metric_spec in zip(curve_sets, metric_specs)
    ]
    sizes = [fit["sizes"] for fit in fits]
    if any(row != sizes[0] for row in sizes[1:]):
        warnings.warn(f"The stacked panels cover different system sizes: {sizes}")

    if figsize is None:
        figsize = (7.6, 3.6 * rows) if collapse_inset else (13, 4.4 * rows)
    fig, panels = _paired_axes(
        inset=collapse_inset,
        figsize=figsize,
        dpi=dpi,
        nrows=rows,
        sharex=True,
        inset_corner=inset_corner,
        inset_size=inset_size,
        inset_pad=inset_pad,
        inset_fontsize=inset_fontsize,
        inset_facecolor=inset_facecolor,
        inset_alpha=inset_alpha,
    )
    auto_legend_loc, auto_annotation_loc, reserve_side = _inset_overlay_defaults(
        inset_corner
    )
    if legend_loc is None:
        legend_loc = auto_legend_loc if collapse_inset else "best"
    if annotation_loc is None:
        annotation_loc = auto_annotation_loc if collapse_inset else "lower right"

    colors = _color_map_by_size(sizes[0], cmap)
    for index, (fit, (ax_raw, ax_collapse)) in enumerate(zip(fits, panels)):
        key = fit["metric_spec"]["key"]
        last = index == rows - 1
        # An inset titles and labels itself from the inside, so every row can
        # carry both. A side-by-side collapse panel is a stacked row like the
        # raw one: its title sits above its frame and its x label below, which
        # on any row but the first and last lands in the neighbouring row.
        second_column_stacked = not collapse_inset
        _draw_bulk_exponent_panel(
            ax_raw,
            ax_collapse,
            fit,
            colors=colors,
            mi_units=mi_units,
            fit_x=fit_x,
            show_errorbars=show_errorbars,
            capsize=capsize,
            raw_yscale=_per_metric(raw_yscale, key) or "linear",
            collapse_yscale=_per_metric(collapse_yscale, key) or "linear",
            collapse_inset=collapse_inset,
            show_fit_window=show_fit_window,
            inset_fontsize=inset_fontsize,
            inset_headroom=inset_headroom,
            reserve_side=reserve_side,
            annotation_loc=annotation_loc,
            annotation_fontsize=annotation_fontsize,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncols=legend_ncols,
            draw_legend=index == 0,
            # Only the bottom row carries the shared x label, and only the top
            # row can carry a title without splitting the stack. The per-metric
            # default title is suppressed: it would name one row's observable
            # over a figure that holds several.
            raw_title=("" if raw_title is None else _per_metric(raw_title, key))
            if index == 0
            else "",
            collapse_title=(
                ""
                if second_column_stacked and index > 0
                else _per_metric(collapse_title, key)
            ),
            raw_xlabel=raw_xlabel if last else "",
            raw_ylabel=_per_metric(raw_ylabel, key),
            collapse_xlabel=(
                "" if second_column_stacked and not last else collapse_xlabel
            ),
            collapse_ylabel=_per_metric(collapse_ylabel, key),
        )
        if prune_shared_yticks and index > 0:
            _prune_upper_yticks(ax_raw)
            if second_column_stacked:
                _prune_upper_yticks(ax_collapse)
    fig.suptitle(
        suptitle if suptitle is not None else metric_specs[0]["suptitle"],
        fontsize=14,
    )
    # An inset is positioned in its parent's axes fraction and follows it, so
    # only the gridspec axes of each row are re-laid.
    _flush_stacked_axes(
        fig, [list(pair[:1] if collapse_inset else pair) for pair in panels]
    )
    _show(fig, show)

    results = {fit["metric_spec"]["key"]: _panel_result(fit, mi_units) for fit in fits}
    return {
        "figure": fig,
        "axes": tuple(panels),
        "collapse_inset": collapse_inset,
        "stacked": True,
        "metrics": list(results),
        "mi_units": mi_units,
        "panels": results,
        "eta": {key: panel["eta"] for key, panel in results.items()},
        "eta_stderr": {key: panel["eta_stderr"] for key, panel in results.items()},
        "score": {key: panel["score"] for key, panel in results.items()},
        "summary": pd.concat(
            [panel["summary"] for panel in results.values()], ignore_index=True
        ),
        "curves": {key: panel["curves"] for key, panel in results.items()},
    }

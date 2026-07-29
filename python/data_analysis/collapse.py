"""One-parameter finite-size collapse of probe observables.

Every bulk-exponent collapse in this package fits the same ansatz

    O(L, t) = L^(-eta) g((t - 2L) / L),

and differs only in which CSV columns hold the observable, how the observable
is labelled, and whether it may be negative. Those differences live in the
metric specs below; the loader and the fit itself are shared.
"""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .loading import _find_column, resolve_metadata
from .plotting import _color_map_by_size, _show


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
    figsize: tuple[float, float],
    dpi: int,
    show_summary: bool,
    show: bool,
) -> dict[str, Any]:
    """Fit and plot ``O(L,t)=L^(-eta)g((t-2L)/L)`` for a set of curves.

    ``curves`` come from :func:`load_scaled_time_curve` and are modified in
    place when the metric is rescaled by ``mi_scale``.
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

    if signed:
        ax_raw.axhline(0.0, linewidth=1, alpha=0.35)
    ax_raw.axvline(0.0, linewidth=1, linestyle="--", alpha=0.6)
    fit_upper = (
        fit_x_max
        if fit_x_max is not None
        else max(curve["x"].max() for curve in curves)
    )
    ax_raw.axvspan(fit_x_min, fit_upper, alpha=0.08, label="fit window")
    ax_raw.set_xlabel(r"Scaled time $x=(t-2L)/L$")
    raw_ylabel = metric_spec["raw_ylabel"]
    collapse_ylabel = metric_spec["collapse_ylabel"]
    if metric_spec["scale_with_mi_units"]:
        raw_ylabel = raw_ylabel.replace("[nats]", f"[{mi_units}]")
        collapse_ylabel = collapse_ylabel + f" [{mi_units}]"
    ax_raw.set_ylabel(raw_ylabel)
    ax_raw.set_title(metric_spec["raw_title"])
    ax_raw.set_yscale(raw_yscale)
    ax_raw.grid(alpha=0.25)
    ax_raw.legend()

    if signed:
        ax_collapse.axhline(0.0, linewidth=1, alpha=0.35)
    ax_collapse.set_xlabel(r"Scaled time $x=(t-2L)/L$")
    ax_collapse.set_ylabel(collapse_ylabel)
    ax_collapse.set_title("Bulk-exponent scaling collapse")
    ax_collapse.set_yscale(collapse_yscale)
    ax_collapse.grid(alpha=0.25)
    ax_collapse.legend()
    symbol = metric_spec["eta_symbol"]
    eta_text = (
        rf"${symbol}={eta:.4f}\pm {eta_stderr:.4f}$"
        if np.isfinite(eta_stderr)
        else rf"${symbol}={eta:.4f}$"
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

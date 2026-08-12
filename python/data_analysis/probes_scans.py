"""Fixed-time p scans, entropy maps, and spacetime-anisotropy probes.

These cover ``mipt_probed.exe`` modes 1, 2, and 3.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .fitting import _curvature_errors
from .loading import (
    _constant_csv_value,
    _find_column,
    _parse_size,
    _resolve_files,
)
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
    mi_units: str = "bits",
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
    zoom_inset: bool = True,
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
    full_title: str | None = "Full evolution",
    zoom_title: str | None = "Late-time zoom",
    xlabel: str | None = None,
    ylabel: str | None = None,
    suptitle: str | None = None,
    figsize: tuple[float, float] | None = None,
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

    By default the late-time zoom is drawn as an inset in the bottom-right
    corner of the full-evolution panel, lifted clear of the bottom edge so
    both x axes stay readable. ``inset_size`` and ``inset_pad`` are fractions
    of the parent axes; ``zoom_inset=False`` restores the two-panel layout.
    The single legend lives on the full-evolution axes.
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
    if figsize is None:
        figsize = (7.6, 5.2) if zoom_inset else (13, 4.8)
    fig, panels = _paired_axes(
        inset=zoom_inset,
        figsize=figsize,
        dpi=dpi,
        inset_corner=inset_corner,
        inset_size=inset_size,
        inset_pad=inset_pad,
        inset_fontsize=inset_fontsize,
        inset_facecolor=inset_facecolor,
        inset_alpha=inset_alpha,
    )
    ax_full, ax_zoom = panels[0]
    inset_font = _font_kwargs(zoom_inset, inset_fontsize)
    auto_legend_loc, auto_annotation_loc, reserve_side = _inset_overlay_defaults(
        inset_corner
    )
    if legend_loc is None:
        legend_loc = auto_legend_loc if zoom_inset else "best"
    if annotation_loc is None:
        annotation_loc = auto_annotation_loc if zoom_inset else "lower right"
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
    x_text = xlabel if xlabel is not None else r"Readout time $\tau/L$"
    y_text = (
        ylabel
        if ylabel is not None
        else rf"Mutual information $I_{{12}}$ [{mi_units}]"
    )
    for axis, limits, title, is_inset in (
        (ax_full, full_xlim, full_title, False),
        (ax_zoom, zoom_xlim, zoom_title, zoom_inset),
    ):
        font = _font_kwargs(is_inset, inset_fontsize)
        axis.set_xlim(*limits)
        axis.set_xlabel(x_text, **font)
        if not is_inset:
            axis.set_ylabel(y_text)
        axis.grid(alpha=0.22)
    if full_title:
        ax_full.set_title(full_title)
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
    _set_secondary_title(
        ax_zoom,
        zoom_title,
        inset=zoom_inset,
        fontsize=inset_fontsize + 1.0,
    )
    legend_kwargs: dict[str, Any] = {"loc": legend_loc, "ncols": legend_ncols}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    ax_full.legend(**legend_kwargs)
    ax_zoom.axvspan(
        estimate_window[0], estimate_window[1], color="black", alpha=0.045
    )
    annotation_text = (
        rf"$\delta t_*/L={delta_t_star_over_l:.4g}$" + "\n"
        + rf"$a={alpha:.2g}\pm {alpha_stderr:.1g}$"
    )
    annotation_kwargs: dict[str, Any] = {}
    if annotation_fontsize is not None:
        annotation_kwargs["fontsize"] = annotation_fontsize
    _annotate_axes(
        ax_full if zoom_inset else ax_zoom,
        annotation_text,
        annotation_loc,
        **annotation_kwargs,
    )
    if zoom_inset and full_ylim is None:
        _reserve_axis_space(ax_full, reserve_side, inset_headroom)
    fig.suptitle(
        suptitle
        if suptitle is not None
        else rf"Two-probe anisotropy correlators ($L={size}$, $p={probabilities[0]:g}$)"
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


# -------------------------------------------------------------------------
# Fixed-t critical scans (mode 1)
# -------------------------------------------------------------------------


_PROBE_PC_SPECS = {
    "sq": {
        "mean": ["S_Q_mean", "SQ_mean", "S_mean", "entropy_mean"],
        "stderr": ["S_Q_stderr", "SQ_stderr", "S_stderr", "entropy_stderr"],
        "ylabel": r"Ancilla entropy $\overline{S_Q}$ [nats]",
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
    mi_units: str = "bits",
    interpolation_points: int = 250,
    relative_error_floor: float = 0.02,
    absolute_error_floor: float = 1e-10,
    seed: int = 24680,
    cmap: str = "viridis",
    show_errorbars: bool = True,
    capsize: float = 2,
    collapse_inset: bool = True,
    inset_corner: str = "upper right",
    inset_size: tuple[float, float] = (0.40, 0.38),
    inset_pad: tuple[float, float] = (0.0, 0.0),
    inset_fontsize: float = 7.5,
    inset_facecolor: str = "white",
    inset_alpha: float = 1.0,
    inset_headroom: float = 0.0,
    annotation_loc: str | None = None,
    annotation_fontsize: float | None = None,
    legend_loc: str | None = None,
    legend_fontsize: float | None = None,
    legend_ncols: int = 1,
    raw_title: str | None = None,
    collapse_title: str | None = "Finite-size scaling collapse",
    suptitle: str | None = None,
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

    Each collapse is drawn by default as an inset in the top-right corner of
    its raw panel, with one legend per row on the raw axes. ``inset_size``
    and ``inset_pad`` are fractions of the parent axes;
    ``collapse_inset=False`` restores the side-by-side layout, which keeps a
    legend on both panels. ``raw_title`` overrides the per-metric title;
    ``None`` keeps ``"<metric title> — raw"``.
    """
    # S_Q keeps its own knob because it is the order parameter and was quoted in
    # bits long before the information metrics were; both go through the one
    # unit helper so the aliases and the conversion factor agree.
    sq_units, sq_scale = _mi_unit_spec(sq_units, name="sq_units")
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
        scale = sq_scale if metric == "sq" else mi_scale
        for curve in curves:
            curve["value"] *= scale
            curve["dvalue"] *= scale
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
        figsize = (7.6, 5.2 * nrows) if collapse_inset else (13, 4.5 * nrows)
    fig, axes = _paired_axes(
        inset=collapse_inset,
        figsize=figsize,
        dpi=dpi,
        nrows=nrows,
        inset_corner=inset_corner,
        inset_size=inset_size,
        inset_pad=inset_pad,
        inset_fontsize=inset_fontsize,
        inset_facecolor=inset_facecolor,
        inset_alpha=inset_alpha,
    )
    inset_font = _font_kwargs(collapse_inset, inset_fontsize)
    _, _, reserve_side = _inset_overlay_defaults(inset_corner)
    # Crossing curves leave the left column free, so both overlays stack
    # there: the parameter box in the corner, the legend centred above it.
    if legend_loc is None:
        legend_loc = "center left" if collapse_inset else "best"
    if annotation_loc is None:
        annotation_loc = "lower left" if collapse_inset else "lower right"
    legend_kwargs: dict[str, Any] = {"loc": legend_loc, "ncols": legend_ncols}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    annotation_kwargs: dict[str, Any] = {}
    if annotation_fontsize is not None:
        annotation_kwargs["fontsize"] = annotation_fontsize
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
        metric_units = sq_units if metric == "sq" else mi_units
        ax_raw.set_ylabel(spec["ylabel"].replace("[nats]", f"[{metric_units}]"))
        ax_raw.set_title(
            raw_title if raw_title is not None else spec["title"] + " — raw"
        )
        ax_raw.grid(alpha=0.25)
        ax_raw.legend(**legend_kwargs)

        ax_scaled.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax_scaled.axhline(0.0, color="black", linewidth=0.7, alpha=0.3)
        ax_scaled.set_xlabel(r"$(p-p_c)L^{1/\nu}$", **inset_font)
        if not collapse_inset:
            ax_scaled.set_ylabel(spec["scaled_ylabel"] + f" [{metric_units}]")
            ax_scaled.legend(**legend_kwargs)
        ax_scaled.grid(alpha=0.25)
        _set_secondary_title(
            ax_scaled,
            collapse_title,
            inset=collapse_inset,
            fontsize=inset_fontsize + 1.0,
        )
        x_text = (
            r"$x=0$ fixed"
            if metric == "sq" and abs(fit["x"]) < 1e-14
            else rf"${spec['x_symbol']}={fit['x']:.4g}\pm {fit['x_stderr']:.2g}$"
        )
        _annotate_axes(
            ax_raw if collapse_inset else ax_scaled,
            rf"$p_c={fit['pc']:.5g}\pm {fit['pc_stderr']:.2g}$" + "\n"
            + rf"$\nu={fit['nu']:.4g}\pm {fit['nu_stderr']:.2g}$" + "\n"
            + rf"score $={fit['score']:.3g}$",
            annotation_loc,
            **annotation_kwargs,
        )
        if collapse_inset:
            _reserve_axis_space(ax_raw, reserve_side, inset_headroom)
    fig.suptitle(
        suptitle
        if suptitle is not None
        else f"Fixed-time critical probe scaling (probes={probes}, t=4L)"
    )
    _show(fig, show)
    return {
        "figure": fig,
        "axes": np.array(axes, dtype=object),
        "collapse_inset": collapse_inset,
        "probes": probes,
        "metrics": selected_metrics,
        "sq_units": sq_units,
        "mi_units": mi_units,
        "fits": fits,
        "curves": curves_by_metric,
    }


# -------------------------------------------------------------------------
# Entropy map (mode 2)
# -------------------------------------------------------------------------


def probe_entropy_map(
    file: str | Path,
    *,
    p_values: Sequence[float] | None = None,
    line_count: int = 5,
    entropy_units: str = "bits",
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
    ``entropy_units`` selects ``"bits"`` (the default) or ``"nats"``; the probe
    CSVs store nats.
    """
    entropy_units, entropy_scale = _mi_unit_spec(entropy_units, name="entropy_units")
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
    clean[["S_Q", "dS_Q"]] *= entropy_scale
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
    colorbar.set_label(rf"$\overline{{S_Q}}$ [{entropy_units}]")
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
    ax_lines.set_ylabel(rf"Ancilla entropy $\overline{{S_Q}}$ [{entropy_units}]")
    ax_lines.set_title("Entropy dynamics at selected measurement rates")
    ax_lines.grid(alpha=0.25)
    ax_lines.legend()
    fig.suptitle(f"Probe-entropy p-t scan (probes={probes})")
    _show(fig, show)
    return {
        "figure": fig,
        "axes": (ax_map, ax_lines),
        "colorbar": colorbar,
        "entropy_units": entropy_units,
        "probes": probes,
        "selected_p": selected_p,
        "p": simulated_p,
        "t": simulated_t,
        "entropy": pivot.to_numpy(dtype=float),
        "data": clean,
    }

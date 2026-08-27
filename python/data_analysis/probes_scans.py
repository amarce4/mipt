"""Fixed-time p scans, entropy maps, and spacetime-anisotropy probes.

These cover ``mipt_probed.exe`` modes 1, 2, and 3.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re
import warnings
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, differential_evolution, minimize

from .fitting import _curvature_errors
from .loading import (
    _REALIZATION_COLUMNS,
    _constant_column,
    _constant_csv_value,
    _find_column,
    _parse_size,
    _resolve_files,
)
from .plotting import (
    _annotate_axes,
    _color_map_by_size,
    _font_kwargs,
    _inset_rectangle,
    _inset_overlay_defaults,
    _set_secondary_title,
    _mi_unit_spec,
    _panel_label,
    _paired_axes,
    _reserve_axis_space,
    _show,
    _style_inset,
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


def _repaired_realization_counts(
    counts: pd.Series,
    path: Path,
    column: str,
) -> pd.Series | None:
    """Refill a per-row realization count that a CSV merge left blank.

    ``realizations`` is run metadata: constant for one scan file, repeated on
    every row only so that several files at one ``L`` can be pooled exactly.
    A file stitched together from two runs can therefore carry blank metadata
    cells on the rows contributed by the second one -- ``N``, ``circ_type``,
    ``circuit_name`` and ``realizations`` empty, the measurement columns
    intact. Those rows are complete data, so the count is refilled from the
    file's own constant value rather than the row being discarded.

    Returns ``None`` when the gap cannot be closed unambiguously (the file is
    not constant in the column, or carries no value at all), which sends the
    caller down the inverse-variance path it already uses for files with no
    realization column.
    """
    missing = int(counts.isna().sum())
    if missing == 0:
        return counts
    present = counts.dropna().unique()
    if len(present) == 1:
        warnings.warn(
            f"{path}: {missing} of {len(counts)} rows carry no "
            f"{column!r}; refilling from the file's constant value "
            f"{present[0]:,.0f}."
        )
        return counts.fillna(present[0])
    detail = (
        "the file carries no value for it"
        if len(present) == 0
        else f"the file is not constant in it ({sorted(present)})"
    )
    warnings.warn(
        f"{path}: {missing} of {len(counts)} rows carry no {column!r} and "
        f"{detail}; pooling shared p points in this file by inverse variance "
        "instead."
    )
    return None


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
    # The realization count rides along per row so that several scans at one L
    # can be pooled exactly (see `_merge_probe_pc_curves`). It is optional:
    # older files without the column pool by inverse variance instead.
    reals_col = _find_column(df, list(_REALIZATION_COLUMNS), required=False)
    columns = {
        "p": pd.to_numeric(df[p_col], errors="coerce"),
        "value": pd.to_numeric(df[mean_col], errors="coerce"),
        "dvalue": pd.to_numeric(df[stderr_col], errors="coerce"),
    }
    if reals_col is not None:
        repaired = _repaired_realization_counts(
            pd.to_numeric(df[reals_col], errors="coerce"), path, reals_col
        )
        if repaired is not None:
            columns["realizations"] = repaired
    # Only the measurement columns can invalidate a row. Dropping on the
    # metadata columns as well would discard every p point a merged file
    # carries with blank metadata cells -- silently, since the curve that
    # remains is still a valid curve, just a sparser one.
    clean = (
        pd.DataFrame(columns)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["p", "value", "dvalue"])
        .sort_values("p", kind="stable")
    )
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
    # One file can hold a p point twice, for the same reason two files can:
    # a rerun that added realizations was concatenated into it. Pool them the
    # way `_merge_probe_pc_curves` pools across files, rather than keeping
    # whichever row came last and throwing the other run's statistics away.
    pooled, shared = _pool_curve_by_p(
        clean["p"].to_numpy(dtype=float),
        clean["value"].to_numpy(dtype=float),
        clean["dvalue"].to_numpy(dtype=float),
        (
            clean["realizations"].to_numpy(dtype=float)
            if "realizations" in clean.columns
            else None
        ),
    )
    if shared:
        print(
            f"{metric}: {path.name} pooled {shared} repeated p "
            f"point{'s' if shared > 1 else ''} into "
            f"{pooled['p'].size} points."
        )
    return {
        "path": path,
        "paths": [path],
        "L": _parse_size(path, l_by_file),
        "p": pooled["p"],
        "value": pooled["value"],
        "dvalue": pooled["dvalue"],
        "realizations": pooled["realizations"],
        "readout_time": _constant_column(
            df, ["t", "time", "timestep"], integer=True
        ),
    }


# Two p values are the same grid point if they agree to this. The scan grids
# come from `checked_linspace` and are written at 17 digits, so a shared point
# round-trips to within an ulp; anything a scan actually resolves is thousands
# of times coarser, so there is a very wide margin either way.
_PROBE_PC_MERGE_P_TOL = 1e-9


def _pool_probe_pc_points(
    values: np.ndarray,
    errors: np.ndarray,
    counts: np.ndarray | None,
) -> tuple[float, float, float]:
    """Pool independent estimates of one ``(L, p)`` point.

    Returns ``(mean, stderr, realizations)``.

    With realization counts known this is Welford's parallel merge, so the
    result is exactly what one run over the concatenated trajectories would
    have reported: ``m2 = stderr**2 * n * (n - 1)`` recovers each group's sum
    of squares about its own mean -- the same inversion
    ``util/resume_csv.hpp`` uses to restart a checkpointed scan -- the groups
    are merged about the pooled mean, and the combined standard error is read
    back out. Note the mean is weighted by ``n``, not by ``1/stderr**2``:
    the former is the mean of the pooled sample, which is what the data would
    have given, and the two differ whenever the per-run variances differ.

    Without counts (a file carrying no realization column) there is no pooled
    sample to reconstruct, so the estimates are combined by inverse variance
    and the count is reported as ``nan``.
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if values.size == 1:
        return (
            float(values[0]),
            float(errors[0]),
            float("nan") if counts is None else float(counts[0]),
        )

    if counts is None:
        if not np.all(errors > 0.0):
            # An exact point cannot be inverse-variance weighted. Fall back to
            # the unweighted mean and keep the tightest error quoted.
            return float(np.mean(values)), float(np.min(errors)), float("nan")
        weights = 1.0 / errors**2
        mean = float(np.sum(weights * values) / np.sum(weights))
        return mean, float(1.0 / np.sqrt(np.sum(weights))), float("nan")

    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0:
        raise ValueError("Pooling requires positive realization counts.")
    mean = float(np.sum(counts * values) / total)
    # Each group's sum of squares about its own mean, then shifted onto the
    # pooled mean. stderr**2 * n * (n - 1) is m2; a single-realization group
    # contributes nothing, which is correct.
    m2 = float(
        np.sum(errors**2 * counts * (counts - 1.0))
        + np.sum(counts * (values - mean) ** 2)
    )
    if total <= 1.0:
        return mean, 0.0, total
    return mean, float(np.sqrt(max(m2, 0.0) / (total * (total - 1.0)))), total


def _pool_curve_by_p(
    p: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    counts: np.ndarray | None,
) -> tuple[dict[str, np.ndarray | None], int]:
    """Collapse repeated p values into one point each.

    Points within ``_PROBE_PC_MERGE_P_TOL`` of each other are one grid point
    and are pooled by :func:`_pool_probe_pc_points`. Returns the pooled curve
    arrays and how many grid points had more than one estimate. This is used
    both within a file (a rerun concatenated into it) and across the files at
    one ``L``, so the two cases cannot drift apart.
    """
    order = np.argsort(p, kind="stable")
    pooled: list[tuple[float, float, float, float]] = []
    shared = 0
    start = 0
    while start < order.size:
        stop = start + 1
        while (
            stop < order.size
            and p[order[stop]] - p[order[start]] <= _PROBE_PC_MERGE_P_TOL
        ):
            stop += 1
        index = order[start:stop]
        mean, stderr, count = _pool_probe_pc_points(
            values[index],
            errors[index],
            None if counts is None else counts[index],
        )
        pooled.append((float(np.mean(p[index])), mean, stderr, count))
        shared += index.size > 1
        start = stop

    arrays = {
        "p": np.array([point[0] for point in pooled], dtype=float),
        "value": np.array([point[1] for point in pooled], dtype=float),
        "dvalue": np.array([point[2] for point in pooled], dtype=float),
        "realizations": (
            None
            if counts is None
            else np.array([point[3] for point in pooled], dtype=float)
        ),
    }
    return arrays, shared


def _merge_probe_pc_curves(
    curves: Sequence[dict[str, Any]],
    metric: str,
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Combine several scans at the same ``L`` into one curve.

    Disjoint p grids interleave; a p point that several files share is pooled
    by :func:`_pool_probe_pc_points`, so its error reflects the combined
    realization count rather than one file's. Files at one ``L`` must agree on
    the readout time -- mode 1 reads out at a fixed ``t=4N``, and two different
    readout times are two different observables, not more statistics.
    """
    by_size: dict[int, list[dict[str, Any]]] = {}
    for curve in curves:
        by_size.setdefault(int(curve["L"]), []).append(curve)

    merged: list[dict[str, Any]] = []
    for size in sorted(by_size):
        group = by_size[size]
        if len(group) == 1:
            merged.append(group[0])
            continue

        times = {
            curve["readout_time"]
            for curve in group
            if curve["readout_time"] is not None
        }
        if len(times) > 1:
            names = ", ".join(Path(c["path"]).name for c in group)
            raise ValueError(
                f"L={size}: refusing to merge scans with different readout "
                f"times {sorted(times)}; these are different observables. "
                f"Files: {names}."
            )

        missing = [c for c in group if c["realizations"] is None]
        if missing and len(missing) != len(group):
            warnings.warn(
                f"L={size}: {len(missing)} of {len(group)} files carry no "
                "realization column; pooling shared p points by inverse "
                "variance instead of by realization count."
            )
        use_counts = not missing

        pooled, shared = _pool_curve_by_p(
            np.concatenate([curve["p"] for curve in group]),
            np.concatenate([curve["value"] for curve in group]),
            np.concatenate([curve["dvalue"] for curve in group]),
            (
                np.concatenate([curve["realizations"] for curve in group])
                if use_counts
                else None
            ),
        )

        combined = dict(group[0])
        combined["paths"] = [c["path"] for c in group]
        combined["p"] = pooled["p"]
        combined["value"] = pooled["value"]
        combined["dvalue"] = pooled["dvalue"]
        combined["realizations"] = pooled["realizations"]
        combined["merged_files"] = len(group)
        combined["shared_p_points"] = shared
        merged.append(combined)

        if verbose:
            # Realizations per p point, not summed over the scan: the pooled
            # count at a point is what its error bar is built from.
            if use_counts:
                low = int(np.min(combined["realizations"]))
                high = int(np.max(combined["realizations"]))
                span = f"{low:,}" if low == high else f"{low:,}-{high:,}"
                total = f", {span} realizations per p"
            else:
                total = ""
            print(
                f"{metric}: L={size} merged {len(group)} files into "
                f"{combined['p'].size} p points ({shared} shared{total})."
            )

    merged.sort(key=lambda curve: curve["L"])
    return merged


def _select_collapse_curves(
    curves: Sequence[dict[str, Any]],
    collapse_l_range: tuple[int | None, int | None],
) -> list[dict[str, Any]]:
    """Select the inclusive system-size range used by collapse fits."""
    if not isinstance(collapse_l_range, tuple) or len(collapse_l_range) != 2:
        raise ValueError(
            "collapse_l_range must be a two-item tuple (L_min, L_max)."
        )
    lower, upper = collapse_l_range
    for name, value in (("L_min", lower), ("L_max", upper)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"{name} in collapse_l_range must be an int or None.")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("collapse_l_range requires L_min <= L_max.")
    selected = [
        curve
        for curve in curves
        if (lower is None or curve["L"] >= lower)
        and (upper is None or curve["L"] <= upper)
    ]
    if len(selected) < 2:
        available = sorted(int(curve["L"]) for curve in curves)
        raise ValueError(
            "collapse_l_range must retain at least two system sizes; "
            f"available sizes are {available}."
        )
    return selected


def _zabalo_objective_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    error: np.ndarray,
    *,
    correction_basis: np.ndarray | None = None,
) -> dict[str, Any] | None:
    r"""Evaluate Eqs. (S1)--(S4) of Zabalo et al. exactly.

    The pooled data are sorted by their scaled coordinate.  Every interior
    point is compared with the straight line through its two neighbours, and
    all three quoted standard errors are propagated into the denominator.
    If ``correction_basis`` is supplied, its linear coefficients are profiled
    out before evaluating the same objective.  No empirical error floor or
    interpolation grid is introduced.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    error = np.asarray(error, dtype=float)
    if not (x.ndim == y.ndim == error.ndim == 1):
        raise ValueError("Zabalo objective arrays must be one-dimensional.")
    if not (x.size == y.size == error.size):
        raise ValueError("Zabalo objective arrays must have equal lengths.")
    if x.size < 3:
        return None
    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(error))
        and np.all(error > 0.0)
    ):
        return None

    order = np.argsort(x, kind="mergesort")
    x, y, error = x[order], y[order], error[order]
    basis = None
    if correction_basis is not None:
        basis = np.asarray(correction_basis, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != order.size:
            raise ValueError(
                "correction_basis must have shape (number of points, order)."
            )
        basis = basis[order]
        if not np.all(np.isfinite(basis)):
            return None

    span = x[2:] - x[:-2]
    # The paper assumes x_1 < ... < x_n.  Coincident triples can occur only
    # at isolated parameter values for shared p grids; the exact expression
    # is undefined there, so leave those values outside the objective domain.
    if np.any(span <= 0.0):
        return None
    left_weight = (x[2:] - x[1:-1]) / span
    right_weight = (x[1:-1] - x[:-2]) / span
    interpolated = left_weight * y[:-2] + right_weight * y[2:]
    residual = y[1:-1] - interpolated
    variance = (
        error[1:-1] ** 2
        + (left_weight * error[:-2]) ** 2
        + (right_weight * error[2:]) ** 2
    )
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        return None

    coefficients = None
    if basis is not None:
        residual_basis = (
            basis[1:-1]
            - left_weight[:, None] * basis[:-2]
            - right_weight[:, None] * basis[2:]
        )
        if residual.size <= residual_basis.shape[1]:
            return None
        sigma = np.sqrt(variance)
        try:
            coefficients, _, rank, _ = np.linalg.lstsq(
                residual_basis / sigma[:, None],
                residual / sigma,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            return None
        if rank < residual_basis.shape[1] or not np.all(
            np.isfinite(coefficients)
        ):
            return None
        residual = residual - residual_basis @ coefficients

    weights = residual**2 / variance
    if not np.all(np.isfinite(weights)):
        return None
    return {
        "score": float(np.sum(weights) / (x.size - 2)),
        "weights": weights,
        "coefficients": coefficients,
        "x": x,
        "y": y,
        "error": error,
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
    correction_to_scaling: bool,
    omega_bounds: tuple[float, float],
    fixed_omega: float | None,
    f1_order: int,
    seed: int,
) -> dict[str, Any]:
    if correction_to_scaling:
        if not 0.0 < omega_bounds[0] < omega_bounds[1]:
            raise ValueError(
                "omega_bounds must be strictly increasing and positive."
            )
        if len(curves) < 3:
            raise ValueError(
                "The correction-to-scaling fit requires at least three "
                "system sizes."
            )
        if (
            isinstance(f1_order, (bool, np.bool_))
            or not isinstance(f1_order, (int, np.integer))
            or f1_order < 0
        ):
            raise ValueError("f1_order must be a non-negative integer.")
    elif fixed_omega is not None:
        raise ValueError(
            "fixed_omega requires correction_to_scaling=True."
        )

    parameter_specs = [
        ("fixed_pc", fixed_pc, pc_bounds),
        ("fixed_nu", fixed_nu, nu_bounds),
        ("fixed_x", fixed_x, x_bounds),
    ]
    if correction_to_scaling:
        parameter_specs.append(("fixed_omega", fixed_omega, omega_bounds))
    for name, value, bound in parameter_specs:
        if value is not None and not bound[0] <= value <= bound[1]:
            raise ValueError(f"{name}={value} lies outside bounds {bound}.")

    names, bounds, fixed = [], [], {}
    fit_specs = [
        ("pc", fixed_pc, pc_bounds),
        ("nu", fixed_nu, nu_bounds),
        ("x", fixed_x, x_bounds),
    ]
    if correction_to_scaling:
        fit_specs.append(("omega", fixed_omega, omega_bounds))
    for name, value, bound in fit_specs:
        if value is None:
            names.append(name)
            bounds.append(bound)
        else:
            fixed[name] = float(value)

    def unpack(theta):
        values = dict(fixed)
        values.update({name: float(value) for name, value in zip(names, theta)})
        return (
            values["pc"],
            values["nu"],
            values["x"],
            values.get("omega"),
        )

    def raw_transformed(curve, pc, nu, exponent):
        scale = float(curve["L"]) ** exponent
        x = (curve["p"] - pc) * float(curve["L"]) ** (1.0 / nu)
        order = np.argsort(x)
        return x[order], (curve["value"] * scale)[order], (
            curve["dvalue"] * scale
        )[order]

    def pooled_transformed(pc, nu, exponent, curve_set=curves):
        transformed = [
            raw_transformed(curve, pc, nu, exponent) for curve in curve_set
        ]
        return tuple(
            np.concatenate([values[index] for values in transformed])
            for index in range(3)
        )

    def correction_model(pc, nu, exponent, omega, curve_set=curves):
        """Profile polynomial F1 inside the exact Zabalo objective."""
        x, y, error = pooled_transformed(
            pc, nu, exponent, curve_set=curve_set
        )
        x_scale = float(np.max(np.abs(x)))
        if not np.isfinite(x_scale):
            return None
        if x_scale == 0.0:
            x_scale = 1.0
        size_factors = np.concatenate(
            [
                np.full(len(curve["p"]), float(curve["L"]) ** (-omega))
                for curve in curve_set
            ]
        )
        basis = size_factors[:, None] * np.vander(
            x / x_scale, N=f1_order + 1, increasing=True
        )
        evaluated = _zabalo_objective_from_arrays(
            x, y, error, correction_basis=basis
        )
        if evaluated is None:
            return None
        return {
            "score": evaluated["score"],
            "x_scale": x_scale,
            "f1_coefficients": evaluated["coefficients"],
        }

    def objective(theta, curve_set=curves):
        pc, nu, exponent, omega = unpack(theta)
        omega_value = 0.0 if omega is None else omega
        if (
            not np.isfinite(pc + nu + exponent + omega_value)
            or not pc_bounds[0] <= pc <= pc_bounds[1]
            or not nu_bounds[0] <= nu <= nu_bounds[1]
            or not x_bounds[0] <= exponent <= x_bounds[1]
            or (
                correction_to_scaling
                and not omega_bounds[0] <= omega_value <= omega_bounds[1]
            )
        ):
            return np.inf
        if correction_to_scaling:
            model = correction_model(
                pc, nu, exponent, omega_value, curve_set=curve_set
            )
            return np.inf if model is None else model["score"]

        x, y, error = pooled_transformed(
            pc, nu, exponent, curve_set=curve_set
        )
        evaluated = _zabalo_objective_from_arrays(x, y, error)
        return np.inf if evaluated is None else evaluated["score"]

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

    pc, nu, exponent, omega = unpack(theta)
    errors = {"pc": 0.0, "nu": 0.0, "x": 0.0, "omega": 0.0}
    for name, error in zip(names, errors_free):
        errors[name] = float(error)

    fitted_correction = None
    f1_function = None
    if correction_to_scaling:
        fitted_correction = correction_model(pc, nu, exponent, omega)
        if fitted_correction is None:
            raise RuntimeError("Unable to construct the fitted correction model.")

        def f1_function(scaled_x):
            polynomial_x = np.asarray(scaled_x, dtype=float) / fitted_correction[
                "x_scale"
            ]
            return np.polynomial.polynomial.polyval(
                polynomial_x, fitted_correction["f1_coefficients"]
            )

    def transformed(curve, pc_value, nu_value, exponent_value):
        scaled_x, value, error = raw_transformed(
            curve, pc_value, nu_value, exponent_value
        )
        if correction_to_scaling:
            value = value - float(curve["L"]) ** (-omega) * f1_function(
                scaled_x
            )
        return scaled_x, value, error

    return {
        "metric": metric,
        "pc": pc,
        "nu": nu,
        "x": exponent,
        "omega": omega,
        "pc_stderr": errors["pc"],
        "nu_stderr": errors["nu"],
        "x_stderr": errors["x"],
        "omega_stderr": errors["omega"] if correction_to_scaling else None,
        "score": score,
        "objective_name": "Zabalo-Kawashima-Ito",
        "correction_to_scaling": correction_to_scaling,
        "f1_order": f1_order,
        "f1_coefficients": (
            None
            if fitted_correction is None
            else fitted_correction["f1_coefficients"].copy()
        ),
        "scaling_variable_scale": (
            None if fitted_correction is None else fitted_correction["x_scale"]
        ),
        "f1_function": f1_function,
        "transform": transformed,
        "raw_transform": raw_transformed,
        "fit_parameter_names": tuple(names),
        "fit_parameter_bounds": {
            name: tuple(bound) for name, bound in zip(names, bounds)
        },
        "fit_theta": np.asarray(theta, dtype=float).copy(),
        "objective": objective,
        "global_fit": global_fit,
        "local_fit": local_fit,
        "covariance": covariance,
        "hessian": hessian,
    }


def _objective_threshold_interval(
    fit: Mapping[str, Any],
    parameter: str,
    *,
    factor: float = 1.3,
    scan_points: int = 96,
) -> tuple[float, float]:
    """Return the fixed-other-parameters ``factor * O*`` interval.

    Lyu et al. use the two points satisfying ``O = 1.3 O*`` as the
    confidence limits for each pairwise collapse. If the threshold is not
    reached before a fit bound, that bound is returned as a one-sided limit.
    """
    names = tuple(fit["fit_parameter_names"])
    value = float(fit[parameter])
    if parameter not in names:
        return value, value

    index = names.index(parameter)
    lower_bound, upper_bound = fit["fit_parameter_bounds"][parameter]
    theta_hat = np.asarray(fit["fit_theta"], dtype=float)
    objective = fit["objective"]
    target = factor * float(fit["score"])

    def threshold_difference(candidate: float) -> float:
        theta = theta_hat.copy()
        theta[index] = candidate
        result = float(objective(theta))
        return result - target if np.isfinite(result) else np.inf

    def find_limit(bound: float) -> float:
        if np.isclose(bound, value):
            return value
        grid = np.linspace(value, bound, max(3, int(scan_points)))
        previous_x = value
        previous_y = threshold_difference(value)
        for candidate in grid[1:]:
            current_y = threshold_difference(float(candidate))
            if current_y >= 0.0:
                if np.isfinite(previous_y) and previous_y <= 0.0:
                    left, right = sorted((previous_x, float(candidate)))
                    try:
                        return float(
                            brentq(
                                threshold_difference,
                                left,
                                right,
                                xtol=1e-10,
                                rtol=1e-10,
                            )
                        )
                    except (ValueError, RuntimeError):
                        pass
                return float(candidate)
            previous_x, previous_y = float(candidate), current_y
        return float(bound)

    return find_limit(float(lower_bound)), find_limit(float(upper_bound))


def _apply_objective_threshold_errors(
    fit: dict[str, Any],
    parameters: Sequence[str] = ("pc", "nu"),
) -> None:
    """Store paper-style 1.3 O* intervals and use their larger half-width."""
    for parameter in parameters:
        if parameter not in fit["fit_parameter_names"]:
            continue
        interval = _objective_threshold_interval(fit, parameter)
        value = float(fit[parameter])
        lower_error = max(0.0, value - interval[0])
        upper_error = max(0.0, interval[1] - value)
        fit[f"{parameter}_curvature_stderr"] = fit[f"{parameter}_stderr"]
        fit[f"{parameter}_interval_1p3"] = interval
        fit[f"{parameter}_stderr_minus"] = lower_error
        fit[f"{parameter}_stderr_plus"] = upper_error
        fit[f"{parameter}_stderr"] = max(lower_error, upper_error)


def _curve_crossing(
    curve_a: Mapping[str, Any],
    curve_b: Mapping[str, Any],
    *,
    method: str,
    reference: float | None = None,
    values_a: np.ndarray | None = None,
    values_b: np.ndarray | None = None,
) -> float:
    """Find a pairwise curve crossing, selecting the root nearest reference."""
    p_a = np.asarray(curve_a["p"], dtype=float)
    p_b = np.asarray(curve_b["p"], dtype=float)
    y_a = np.asarray(
        curve_a["value"] if values_a is None else values_a, dtype=float
    )
    y_b = np.asarray(
        curve_b["value"] if values_b is None else values_b, dtype=float
    )
    lo, hi = max(float(p_a.min()), float(p_b.min())), min(
        float(p_a.max()), float(p_b.max())
    )
    if hi <= lo:
        raise ValueError(
            f"L={curve_a['L']} and L={curve_b['L']} have no overlapping p range."
        )

    if method == "linear":
        evaluate_a = lambda p: np.interp(p, p_a, y_a)
        evaluate_b = lambda p: np.interp(p, p_b, y_b)
    elif method == "spline":
        if min(p_a.size, p_b.size) < 3:
            raise ValueError("Spline crossings require at least three p points.")
        spline_a = CubicSpline(p_a, y_a, extrapolate=False)
        spline_b = CubicSpline(p_b, y_b, extrapolate=False)
        evaluate_a, evaluate_b = spline_a, spline_b
    else:
        raise ValueError("method must be 'linear' or 'spline'.")

    def difference(p):
        return evaluate_a(p) - evaluate_b(p)

    grid = np.unique(
        np.concatenate(
            (
                p_a[(p_a >= lo) & (p_a <= hi)],
                p_b[(p_b >= lo) & (p_b <= hi)],
                np.linspace(lo, hi, 2049),
            )
        )
    )
    differences = np.asarray(difference(grid), dtype=float)
    roots: list[float] = []
    zero_tolerance = 1e-12 * max(1.0, float(np.nanmax(np.abs(differences))))
    for index in range(grid.size - 1):
        left_y, right_y = differences[index], differences[index + 1]
        if not np.isfinite(left_y + right_y):
            continue
        if abs(left_y) <= zero_tolerance:
            roots.append(float(grid[index]))
            continue
        if left_y * right_y < 0.0:
            roots.append(
                float(brentq(difference, grid[index], grid[index + 1]))
            )
    if abs(differences[-1]) <= zero_tolerance:
        roots.append(float(grid[-1]))
    if not roots:
        raise ValueError(
            f"No {method} crossing was found for L={curve_a['L']} and "
            f"L={curve_b['L']} inside p=[{lo:g}, {hi:g}]."
        )

    roots = sorted(set(round(root, 13) for root in roots))
    if reference is None or not np.isfinite(reference):
        return float(roots[len(roots) // 2])
    return float(min(roots, key=lambda root: abs(root - reference)))


def _pair_crossing_summary(
    curve_a: Mapping[str, Any],
    curve_b: Mapping[str, Any],
    *,
    reference: float,
) -> dict[str, float]:
    """Spline crossing with Lyu-style interpolation and statistical errors."""
    linear = _curve_crossing(
        curve_a, curve_b, method="linear", reference=reference
    )
    try:
        spline = _curve_crossing(
            curve_a, curve_b, method="spline", reference=reference
        )
    except ValueError:
        warnings.warn(
            f"L={curve_a['L']}, {curve_b['L']}: spline crossing failed; "
            "using the linear crossing."
        )
        spline = linear

    shifted_roots = []
    for sign in (-1.0, 1.0):
        try:
            shifted_roots.append(
                _curve_crossing(
                    curve_a,
                    curve_b,
                    method="spline",
                    reference=spline,
                    values_a=np.asarray(curve_a["value"])
                    + sign * np.asarray(curve_a["dvalue"]),
                    values_b=np.asarray(curve_b["value"])
                    - sign * np.asarray(curve_b["dvalue"]),
                )
            )
        except ValueError:
            continue
    statistical = max(
        (abs(root - spline) for root in shifted_roots), default=0.0
    )
    systematic = abs(spline - linear)
    return {
        "pc": float(spline),
        "stderr": float(statistical + systematic),
        "statistical_error": float(statistical),
        "interpolation_error": float(systematic),
        "linear_pc": float(linear),
        "spline_pc": float(spline),
    }


def _equal_weight_linear_fit(
    x: Sequence[float],
    y: Sequence[float],
) -> dict[str, float]:
    """Fit ``y = slope*x + intercept`` with the equal weights used by Lyu."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        raise ValueError("At least three finite pairwise points are required.")
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (slope * x + intercept)
    residual_variance = float(np.sum(residual**2) / (x.size - 2))
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    total_variance = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = (
        1.0 - float(np.sum(residual**2)) / total_variance
        if total_variance > 0.0
        else np.nan
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_stderr": float(np.sqrt(max(0.0, covariance[0, 0]))),
        "intercept_stderr": float(np.sqrt(max(0.0, covariance[1, 1]))),
        "r_squared": r_squared,
        "points": int(x.size),
    }


def _fit_lyu_pairwise_metric(
    curves: Sequence[dict[str, Any]],
    metric: str,
    *,
    pc_bounds: tuple[float, float],
    nu_bounds: tuple[float, float],
    x_bounds: tuple[float, float],
    fixed_x: float | None,
    seed: int,
) -> dict[str, Any]:
    """Fit all size pairs with the original one-parameter scaling ansatz."""
    pair_results = []
    for pair_index, (curve_a, curve_b) in enumerate(combinations(curves, 2)):
        fit = _fit_probe_pc_metric(
            [curve_a, curve_b],
            metric,
            pc_bounds=pc_bounds,
            nu_bounds=nu_bounds,
            x_bounds=x_bounds,
            fixed_pc=None,
            fixed_nu=None,
            fixed_x=fixed_x,
            correction_to_scaling=False,
            omega_bounds=(0.1, 4.0),
            fixed_omega=None,
            f1_order=0,
            seed=seed + pair_index,
        )
        _apply_objective_threshold_errors(fit)
        pc_interval = fit["pc_interval_1p3"]
        nu_interval = fit["nu_interval_1p3"]
        crossing = _pair_crossing_summary(
            curve_a, curve_b, reference=float(fit["pc"])
        )
        pair_results.append(
            {
                "L1": int(curve_a["L"]),
                "L2": int(curve_b["L"]),
                "inverse_product": 1.0
                / (float(curve_a["L"]) * float(curve_b["L"])),
                "crossing": crossing,
                "collapse": fit,
            }
        )

    excluded = max(pair_results, key=lambda item: item["inverse_product"])
    for result in pair_results:
        result["included_in_extrapolation"] = result is not excluded
    included = [
        result for result in pair_results if result["included_in_extrapolation"]
    ]
    inverse_products = [result["inverse_product"] for result in included]
    pc_crossing_fit = _equal_weight_linear_fit(
        inverse_products, [result["crossing"]["pc"] for result in included]
    )
    nu_fit = _equal_weight_linear_fit(
        inverse_products, [result["collapse"]["nu"] for result in included]
    )
    return {
        "metric": metric,
        "pairs": pair_results,
        "excluded_pair": (excluded["L1"], excluded["L2"]),
        "pc_crossing_extrapolation": pc_crossing_fit,
        "nu_extrapolation": nu_fit,
        "confidence_objective_factor": 1.3,
        "ansatz": r"L^x I(p,L)=F((p-p_c)L^{1/nu})",
    }


def _draw_lyu_extrapolation_row(
    ax_pc,
    ax_nu,
    summary: Mapping[str, Any],
    metric: str,
    *,
    cmap: str,
    show_errorbars: bool,
    capsize: float,
    legend_loc: str | None,
    legend_fontsize: float | None,
    annotation_loc: str | None,
    annotation_fontsize: float | None,
    title_prefix: bool,
) -> None:
    """Draw one pair of Lyu-style large-size extrapolation axes."""
    color_map = plt.get_cmap(cmap)
    crossing_color = color_map(0.18)
    collapse_color = color_map(0.72)
    nu_color = color_map(0.46)
    legend_kwargs: dict[str, Any] = {"loc": legend_loc or "best"}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    annotation_kwargs: dict[str, Any] = {}
    if annotation_fontsize is not None:
        annotation_kwargs["fontsize"] = annotation_fontsize

    pairs = sorted(summary["pairs"], key=lambda item: item["inverse_product"])
    x = np.asarray([item["inverse_product"] for item in pairs])
    crossing_pc = np.asarray([item["crossing"]["pc"] for item in pairs])
    crossing_error = np.asarray([item["crossing"]["stderr"] for item in pairs])
    collapse_pc = np.asarray([item["collapse"]["pc"] for item in pairs])
    collapse_pc_error = np.asarray(
        [
            [
                item["collapse"]["pc"]
                - item["collapse"]["pc_interval_1p3"][0]
                for item in pairs
            ],
            [
                item["collapse"]["pc_interval_1p3"][1]
                - item["collapse"]["pc"]
                for item in pairs
            ],
        ]
    )
    nu = np.asarray([item["collapse"]["nu"] for item in pairs])
    nu_error = np.asarray(
        [
            [
                item["collapse"]["nu"]
                - item["collapse"]["nu_interval_1p3"][0]
                for item in pairs
            ],
            [
                item["collapse"]["nu_interval_1p3"][1]
                - item["collapse"]["nu"]
                for item in pairs
            ],
        ]
    )
    excluded_mask = np.asarray(
        [not item["included_in_extrapolation"] for item in pairs]
    )
    ax_pc.errorbar(
        x,
        crossing_pc,
        yerr=crossing_error if show_errorbars else None,
        fmt="o",
        color=crossing_color,
        ecolor=crossing_color,
        markersize=4.5,
        linewidth=1.1,
        elinewidth=0.9,
        capsize=capsize,
        label=r"$p_c$ (crossing)",
    )
    ax_pc.errorbar(
        x,
        collapse_pc,
        yerr=collapse_pc_error if show_errorbars else None,
        fmt="s",
        color=collapse_color,
        ecolor=collapse_color,
        markersize=4.2,
        linewidth=1.1,
        elinewidth=0.9,
        capsize=capsize,
        label=r"$p_c$ (collapse)",
    )
    ax_nu.errorbar(
        x,
        nu,
        yerr=nu_error if show_errorbars else None,
        fmt="o",
        color=nu_color,
        ecolor=nu_color,
        markersize=4.5,
        linewidth=1.1,
        elinewidth=0.9,
        capsize=capsize,
        label="_nolegend_",
    )

    # Keep the rightmost (smallest-size) pair visible but mark its omission
    # from the equal-weight extrapolation, matching Lyu Fig. 3.
    if np.any(excluded_mask):
        ax_pc.plot(
            x[excluded_mask],
            crossing_pc[excluded_mask],
            "o",
            markerfacecolor="white",
            markeredgecolor=crossing_color,
            markersize=4.7,
        )
        ax_pc.plot(
            x[excluded_mask],
            collapse_pc[excluded_mask],
            "s",
            markerfacecolor="white",
            markeredgecolor=collapse_color,
            markersize=4.5,
        )
        ax_nu.plot(
            x[excluded_mask],
            nu[excluded_mask],
            "o",
            markerfacecolor="white",
            markeredgecolor=nu_color,
            markersize=4.7,
            label="_nolegend_",
        )

    line_x = np.linspace(0.0, float(np.max(x)) * 1.03, 250)
    pc_line = summary["pc_crossing_extrapolation"]
    nu_line = summary["nu_extrapolation"]
    ax_pc.plot(
        line_x,
        pc_line["intercept"] + pc_line["slope"] * line_x,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="_nolegend_",
    )
    ax_nu.plot(
        line_x,
        nu_line["intercept"] + nu_line["slope"] * line_x,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="_nolegend_",
    )

    # Make the N -> infinity limit a visible point at the fitted intercept,
    # matching the construction of the paper's Fig. 3 rather than leaving it
    # to be inferred from the dashed line.
    ax_pc.plot(
        0.0,
        pc_line["intercept"],
        marker="o",
        markersize=4.0,
        color="black",
        clip_on=False,
        zorder=8,
    )
    ax_nu.plot(
        0.0,
        nu_line["intercept"],
        marker="o",
        markersize=4.0,
        color="black",
        clip_on=False,
        zorder=8,
    )

    for axis in (ax_pc, ax_nu):
        axis.set_xlabel(r"$(L_1L_2)^{-1}$")
        # Leave enough vertical headroom for the in-panel letter even when
        # the thermodynamic-limit intercept is the extremal data point.
        axis.margins(y=0.12)
        axis.grid(alpha=0.25)
        from matplotlib.ticker import ScalarFormatter

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, -3))
        axis.xaxis.set_major_formatter(formatter)
    ax_pc.legend(**legend_kwargs)
    ax_pc.set_ylabel(r"$p_c(L_1,L_2)$")
    ax_nu.set_ylabel(r"$\nu(L_1,L_2)$")
    prefix = _PROBE_PC_SPECS[metric]["title"] + " — " if title_prefix else ""
    ax_pc.set_title(prefix + "Critical measurement rate")
    ax_nu.set_title(prefix + "Correlation-length exponent")
    _annotate_axes(
        ax_pc,
        rf"$p_c(\infty)={pc_line['intercept']:.5g}\pm "
        rf"{pc_line['intercept_stderr']:.2g}$",
        annotation_loc or "lower left",
        bbox=None,
        **annotation_kwargs,
    )
    _annotate_axes(
        ax_nu,
        rf"$\nu(\infty)={nu_line['intercept']:.4g}\pm "
        rf"{nu_line['intercept_stderr']:.2g}$",
        annotation_loc or "lower left",
        bbox=None,
        **annotation_kwargs,
    )


def _pairwise_collapse_difference(
    curve_a: Mapping[str, Any],
    curve_b: Mapping[str, Any],
    fit: Mapping[str, Any],
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate two rescaled curves and return their signed difference."""
    transformed = []
    for curve in (curve_a, curve_b):
        x, y, _ = fit["transform"](curve, fit["pc"], nu, fit["x"])
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        order = np.argsort(x)
        transformed.append((x[order], y[order]))
    (x_a, y_a), (x_b, y_b) = transformed
    if x_a.size < 2 or x_b.size < 2:
        return np.array([]), np.array([])
    lower = max(float(x_a[0]), float(x_b[0]))
    upper = min(float(x_a[-1]), float(x_b[-1]))
    if lower >= upper:
        return np.array([]), np.array([])
    sample = np.unique(
        np.concatenate(
            (
                x_a[(x_a >= lower) & (x_a <= upper)],
                x_b[(x_b >= lower) & (x_b <= upper)],
            )
        )
    )
    if sample.size < 3:
        sample = np.linspace(lower, upper, 20)
    difference = np.interp(sample, x_a, y_a) - np.interp(sample, x_b, y_b)
    return sample, difference


def _draw_pairwise_collapse_figures(
    pairwise_fits: Mapping[str, Mapping[str, Any]],
    curves_by_metric: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_metrics: Sequence[str],
    *,
    sq_units: str,
    mi_units: str,
    cmap: str,
    show_errorbars: bool,
    capsize: float,
    dpi: int,
) -> dict[str, dict[str, Any]]:
    """Recreate the paper's Fig. 7 pair-by-pair collapse diagnostics.

    Each pair gets one collapse panel and a lower-right inset showing the
    signed curve difference at ``nu* - dnu``, ``nu*`` and ``nu* + dnu``.
    ``dnu`` is the larger side of the fitted ``O <= 1.3 O*`` interval.
    """
    from matplotlib.lines import Line2D

    output: dict[str, dict[str, Any]] = {}
    for metric in selected_metrics:
        summary = pairwise_fits[metric]
        pairs = sorted(
            summary["pairs"], key=lambda item: (item["L1"], item["L2"])
        )
        ncols = min(3, len(pairs))
        nrows = int(np.ceil(len(pairs) / ncols))
        fig, raw_grid = plt.subplots(
            nrows,
            ncols,
            figsize=(4.0 * ncols, 3.45 * nrows),
            dpi=dpi,
            squeeze=False,
            constrained_layout=True,
        )
        axes = np.asarray(raw_grid, dtype=object)
        insets: list[Any] = []
        curve_lookup = {
            int(curve["L"]): curve for curve in curves_by_metric[metric]
        }
        sizes = sorted(curve_lookup)
        colors = _color_map_by_size(sizes, cmap)
        units = sq_units if metric == "sq" else mi_units
        ylabel = _PROBE_PC_SPECS[metric]["scaled_ylabel"] + f" [{units}]"

        for index, item in enumerate(pairs):
            row, column = divmod(index, ncols)
            ax = axes[row, column]
            fit = item["collapse"]
            curve_a = curve_lookup[int(item["L1"])]
            curve_b = curve_lookup[int(item["L2"])]
            for curve in (curve_a, curve_b):
                size = int(curve["L"])
                x_scaled, y_scaled, dy_scaled = fit["transform"](
                    curve, fit["pc"], fit["nu"], fit["x"]
                )
                kwargs = {
                    "color": colors[size],
                    "marker": "o",
                    "markersize": 3.2,
                    "linewidth": 1.0,
                    "label": "_nolegend_",
                }
                if show_errorbars:
                    ax.errorbar(
                        x_scaled,
                        y_scaled,
                        yerr=dy_scaled,
                        capsize=capsize,
                        elinewidth=0.7,
                        **kwargs,
                    )
                else:
                    ax.plot(x_scaled, y_scaled, **kwargs)

            interval = fit.get("nu_interval_1p3", (fit["nu"], fit["nu"]))
            delta_nu = max(
                float(fit["nu"] - interval[0]),
                float(interval[1] - fit["nu"]),
                0.015 * float(fit["nu"]),
            )
            diagnostic = ax.inset_axes([0.55, 0.08, 0.42, 0.38])
            _style_inset(
                diagnostic,
                6.8,
                "white",
                1.0,
                (0.55, 0.08, 0.42, 0.38),
            )
            diagnostic.set_zorder(8)
            for offset, color, label in (
                (-delta_nu, "red", r"$\nu_*-\Delta\nu$"),
                (0.0, "black", r"$\nu_*$"),
                (delta_nu, "blue", r"$\nu_*+\Delta\nu$"),
            ):
                sample, difference = _pairwise_collapse_difference(
                    curve_a, curve_b, fit, float(fit["nu"] + offset)
                )
                if sample.size:
                    diagnostic.plot(
                        sample,
                        difference,
                        color=color,
                        linewidth=0.9,
                        label=label,
                    )
            diagnostic.axhline(0.0, color="0.45", linewidth=0.6)
            diagnostic.set_title(r"$\Delta S$", fontsize=7.0, pad=1.0)
            diagnostic.grid(alpha=0.20)
            insets.append(diagnostic)

            ax.set_title(
                rf"$L_1={item['L1']},\ L_2={item['L2']},\ "
                rf"\nu={fit['nu']:.3g}\pm{fit['nu_stderr']:.2g}$",
                pad=4.0,
            )
            ax.grid(alpha=0.22)
            _panel_label(ax, f"{chr(ord('a') + index)})")
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == nrows - 1:
                ax.set_xlabel(r"$(p-p_c)L^{1/\nu}$")

        for index in range(len(pairs), nrows * ncols):
            row, column = divmod(index, ncols)
            axes[row, column].set_visible(False)

        size_handles = [
            Line2D(
                [],
                [],
                color=colors[size],
                marker="o",
                linewidth=1.0,
                markersize=3.5,
                label=rf"$L={size}$",
            )
            for size in sizes
        ]
        axes[0, 0].legend(
            handles=size_handles,
            loc="upper left",
            bbox_to_anchor=(0.075, 1.0),
            fontsize=7.2,
        )
        output[metric] = {
            "figure": fig,
            "axes": axes,
            "insets": tuple(insets),
        }
    return output


def _objective_plot_range(
    fit: Mapping[str, Any],
    parameter: str,
    requested: tuple[float, float] | None,
) -> tuple[float, float]:
    """Choose a useful view around the 1.3 O* interval."""
    bound = tuple(fit["fit_parameter_bounds"][parameter])
    if requested is not None:
        if not requested[0] < requested[1]:
            raise ValueError(f"{parameter}_range must be strictly increasing.")
        lower = max(bound[0], requested[0])
        upper = min(bound[1], requested[1])
        if lower >= upper:
            raise ValueError(
                f"{parameter}_range does not overlap the fitted bounds {bound}."
            )
        return lower, upper
    value = float(fit[parameter])
    lower, upper = _objective_threshold_interval(fit, parameter)
    bound_span = float(bound[1] - bound[0])
    half_width = max(
        value - lower,
        upper - value,
        0.04 * bound_span,
    )
    if lower <= bound[0] or upper >= bound[1]:
        half_width = max(half_width, 0.12 * bound_span)
    return (
        max(float(bound[0]), value - 1.5 * half_width),
        min(float(bound[1]), value + 1.5 * half_width),
    )


def probe_pc_objective(
    collapse_result: Mapping[str, Any],
    *,
    pc_range: tuple[float, float] | None = None,
    nu_range: tuple[float, float] | None = None,
    grid_points: int | tuple[int, int] = 151,
    contour_factor: float = 1.3,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    dpi: int = 130,
    show: bool = True,
) -> dict[str, Any]:
    r"""Recreate Zabalo et al. (2020) Fig. S4 for a collapse result.

    Pass the dictionary returned by :func:`probe_pc_collapse`.  The heatmap is
    the exact objective ``O(p_c, nu)`` from Eqs. (S1)--(S4), with the white
    contour at ``contour_factor * O*``.  Any additional fitted parameters are
    held at their optimum; for the one-probe ``S_Q`` analysis, ``x=0`` is
    fixed and the surface is exactly two-dimensional as in the paper.
    """
    if not np.isfinite(contour_factor) or contour_factor <= 1.0:
        raise ValueError("contour_factor must be finite and greater than one.")
    if isinstance(grid_points, (int, np.integer)) and not isinstance(
        grid_points, (bool, np.bool_)
    ):
        pc_points = nu_points = int(grid_points)
    elif (
        isinstance(grid_points, tuple)
        and len(grid_points) == 2
        and all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in grid_points
        )
    ):
        pc_points, nu_points = map(int, grid_points)
    else:
        raise ValueError("grid_points must be an int or a two-int tuple.")
    if min(pc_points, nu_points) < 20:
        raise ValueError("grid_points must provide at least 20 points per axis.")

    fits = collapse_result.get("global_fits", collapse_result.get("fits"))
    if not isinstance(fits, Mapping):
        raise ValueError("collapse_result does not contain global collapse fits.")
    metrics = list(collapse_result.get("metrics", fits.keys()))
    if not metrics:
        raise ValueError("collapse_result contains no fitted metrics.")
    if figsize is None:
        figsize = (6.4 * len(metrics), 5.1)
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        constrained_layout=True,
    )
    surfaces = {}
    for column, metric in enumerate(metrics):
        fit = fits[metric]
        names = tuple(fit["fit_parameter_names"])
        if "pc" not in names or "nu" not in names:
            raise ValueError(
                "Zabalo Fig. S4 requires fitted p_c and nu; neither may be fixed."
            )
        pc_values = np.linspace(
            *_objective_plot_range(fit, "pc", pc_range), pc_points
        )
        nu_values = np.linspace(
            *_objective_plot_range(fit, "nu", nu_range), nu_points
        )
        theta = np.asarray(fit["fit_theta"], dtype=float)
        pc_index, nu_index = names.index("pc"), names.index("nu")
        surface = np.full((nu_points, pc_points), np.nan)
        for nu_index_grid, nu_value in enumerate(nu_values):
            for pc_index_grid, pc_value in enumerate(pc_values):
                candidate = theta.copy()
                candidate[pc_index] = pc_value
                candidate[nu_index] = nu_value
                score = float(fit["objective"](candidate))
                if np.isfinite(score):
                    surface[nu_index_grid, pc_index_grid] = score
        finite = surface[np.isfinite(surface)]
        if finite.size == 0:
            raise RuntimeError(f"No finite objective values were found for {metric}.")
        color_max = float(np.percentile(finite, 98.0))
        color_min = float(np.min(finite))
        if color_max <= color_min:
            color_max = color_min + max(abs(color_min), 1.0) * 1e-6
        axis = axes[0, column]
        mesh = axis.pcolormesh(
            pc_values,
            nu_values,
            surface,
            shading="auto",
            cmap=cmap,
            vmin=color_min,
            vmax=color_max,
        )
        threshold = contour_factor * float(fit["score"])
        if color_min <= threshold <= float(np.nanmax(surface)):
            axis.contour(
                pc_values,
                nu_values,
                surface,
                levels=[threshold],
                colors="white",
                linewidths=1.35,
            )
        axis.plot(
            fit["pc"],
            fit["nu"],
            marker="*",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.7,
            linestyle="none",
        )
        axis.set_xlabel(r"$p_c$")
        axis.set_ylabel(r"$\nu$")
        axis.set_title(
            "Collapse objective"
            if len(metrics) == 1
            else _PROBE_PC_SPECS[metric]["title"]
        )
        colorbar = fig.colorbar(mesh, ax=axis)
        colorbar.set_label(r"$O(p_c,\nu)$")
        confidence = np.isfinite(surface) & (surface <= threshold)
        if np.any(confidence):
            pc_mask = np.any(confidence, axis=0)
            nu_mask = np.any(confidence, axis=1)
            pc_interval = (
                float(pc_values[pc_mask][0]),
                float(pc_values[pc_mask][-1]),
            )
            nu_interval = (
                float(nu_values[nu_mask][0]),
                float(nu_values[nu_mask][-1]),
            )
        else:
            pc_interval = (np.nan, np.nan)
            nu_interval = (np.nan, np.nan)
        surfaces[metric] = {
            "pc": pc_values,
            "nu": nu_values,
            "objective": surface,
            "minimum": float(fit["score"]),
            "threshold": threshold,
            "pc_interval": pc_interval,
            "nu_interval": nu_interval,
            "mesh": mesh,
            "colorbar": colorbar,
        }

    fig.suptitle("Zabalo et al. (2020) Fig. S4 collapse objective")
    _show(fig, show)
    return {
        "figure": fig,
        "axes": axes,
        "surfaces": surfaces,
        "contour_factor": contour_factor,
    }


def _resolve_raw_ylim(
    value: Any,
    metric: str,
) -> tuple[float | None, float | None] | None:
    """Pick and validate this metric's explicit raw-panel y limits.

    Accepts one ``(lower, upper)`` pair shared by every metric, or a mapping
    from metric name to such a pair -- a four-probe figure stacks ``I2``,
    ``I3``, and ``I4``, whose scales have nothing to do with each other.
    Either endpoint may be ``None`` to leave that side autoscaled.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get(metric)
        if value is None:
            return None
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(item, (bool, np.bool_)) for item in value)
    ):
        raise ValueError(
            "raw_ylim must be a (lower, upper) pair, or a mapping from metric "
            f"name to one; got {value!r}."
        )
    limits: list[float | None] = []
    for name, item in zip(("lower", "upper"), value):
        if item is None:
            limits.append(None)
            continue
        item = float(item)
        if not np.isfinite(item):
            raise ValueError(f"raw_ylim {name} bound must be finite; got {item!r}.")
        limits.append(item)
    lower, upper = limits
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(f"raw_ylim must be increasing; got {value!r}.")
    return lower, upper


def _draw_probe_pc_metric(
    ax_raw,
    ax_scaled,
    *,
    curves: Sequence[Mapping[str, Any]],
    collapse_curves: Sequence[Mapping[str, Any]],
    metric: str,
    fit: Mapping[str, Any],
    colors: Mapping[int, Any],
    sq_units: str,
    mi_units: str,
    show_errorbars: bool,
    capsize: float,
    collapse_inset: bool,
    inset_fontsize: float,
    inset_headroom: float,
    inset_corner: str,
    legend_loc: str,
    legend_fontsize: float | None,
    legend_ncols: int,
    annotation_loc: str,
    annotation_fontsize: float | None,
    raw_title: str | None,
    collapse_title: str | None,
    raw_ylim: tuple[float | None, float | None] | None = None,
) -> None:
    """Draw one raw/collapse pair using a pre-existing axes pair."""
    spec = _PROBE_PC_SPECS[metric]
    inset_font = _font_kwargs(collapse_inset, inset_fontsize)
    _, _, reserve_side = _inset_overlay_defaults(inset_corner)
    legend_kwargs: dict[str, Any] = {"loc": legend_loc, "ncols": legend_ncols}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    annotation_kwargs: dict[str, Any] = {}
    if annotation_fontsize is not None:
        annotation_kwargs["fontsize"] = annotation_fontsize

    for curve in curves:
        size = int(curve["L"])
        kwargs = {
            "color": colors[size],
            "marker": "o",
            "markersize": 3.8,
            "linewidth": 1.2,
            "label": rf"$L={size}$",
        }
        if show_errorbars:
            ax_raw.errorbar(
                curve["p"],
                curve["value"],
                yerr=curve["dvalue"],
                capsize=capsize,
                elinewidth=0.9,
                **kwargs,
            )
        else:
            ax_raw.plot(curve["p"], curve["value"], **kwargs)
    for curve in collapse_curves:
        size = int(curve["L"])
        kwargs = {
            "color": colors[size],
            "marker": "o",
            "markersize": 3.8,
            "linewidth": 1.2,
            "label": rf"$L={size}$",
        }
        x_scaled, y_scaled, dy_scaled = fit["transform"](
            curve, fit["pc"], fit["nu"], fit["x"]
        )
        if show_errorbars:
            ax_scaled.errorbar(
                x_scaled,
                y_scaled,
                yerr=dy_scaled,
                capsize=capsize,
                elinewidth=0.9,
                **kwargs,
            )
        else:
            ax_scaled.plot(x_scaled, y_scaled, **kwargs)

    ax_raw.axvline(fit["pc"], color="black", linestyle="--", linewidth=1)
    ax_raw.set_xlabel(r"Measurement rate $p$")
    metric_units = sq_units if metric == "sq" else mi_units
    ax_raw.set_ylabel(spec["ylabel"].replace("[nats]", f"[{metric_units}]"))
    ax_raw.set_title(raw_title if raw_title is not None else spec["title"] + " — raw")
    ax_raw.grid(alpha=0.25)
    ax_raw.legend(**legend_kwargs)

    ax_scaled.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax_scaled.set_xlabel(r"$(p-p_c)L^{1/\nu}$", **inset_font)
    if not collapse_inset:
        scaled_ylabel = spec["scaled_ylabel"]
        if fit["correction_to_scaling"]:
            scaled_ylabel += r" (leading correction subtracted)"
        ax_scaled.set_ylabel(scaled_ylabel + f" [{metric_units}]")
        ax_scaled.legend(**legend_kwargs)
    ax_scaled.grid(alpha=0.25)
    _set_secondary_title(
        ax_scaled,
        collapse_title,
        inset=collapse_inset,
        fontsize=inset_fontsize + 1.0,
    )
    correction_text = (
        ""
        if not fit["correction_to_scaling"]
        else rf"$\omega={fit['omega']:.4g}\pm "
        rf"{fit['omega_stderr']:.2g}$" + "\n"
    )
    _annotate_axes(
        ax_raw if collapse_inset else ax_scaled,
        rf"$p_c={fit['pc']:.5g}\pm {fit['pc_stderr']:.2g}$" + "\n"
        + rf"$\nu={fit['nu']:.4g}\pm {fit['nu_stderr']:.2g}$" + "\n"
        + correction_text
        + rf"$O_*={fit['score']:.3g}$",
        annotation_loc,
        **annotation_kwargs,
    )
    if collapse_inset:
        _reserve_axis_space(ax_raw, reserve_side, inset_headroom)
    if raw_ylim is not None:
        # Applied last, so it wins over both autoscale and the inset headroom
        # reservation above. A `None` endpoint leaves that side as drawn, which
        # is what makes `(0, None)` a way to just pin the floor -- autoscale
        # retains Matplotlib's data-driven autoscaling on that side.
        lower, upper = raw_ylim
        current_lower, current_upper = ax_raw.get_ylim()
        ax_raw.set_ylim(
            current_lower if lower is None else lower,
            current_upper if upper is None else upper,
        )


# Inches of figure that the extrapolation layout spends on decorations rather
# than on axes: tick labels, axis labels, panel titles, and the suptitle. They
# turn a requested axes footprint into a figure size, so the panels come out at
# roughly the requested inches instead of the requested inches minus whatever
# the labels happened to need. The gutter is the inner decoration between the
# two extrapolation panels -- counting it is what keeps the bottom row snug, so
# that shrinking those panels moves them together instead of leaving them
# marooned at the edges of over-wide grid cells.
_EXTRAPOLATION_WIDTH_MARGIN = 1.00
_EXTRAPOLATION_GUTTER = 1.03
# Vertical decoration splits into a per-metric part (two rows of x labels and
# the extrapolation titles) and the one suptitle, so a four-probe figure with
# three metric rows is not sized as though it had one.
_EXTRAPOLATION_ROW_MARGIN = 1.05
_EXTRAPOLATION_SUPTITLE_MARGIN = 0.35


def _panel_size(value: Any, name: str) -> tuple[float, float]:
    """Validate a (width, height) axes footprint in inches."""
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(item, (bool, np.bool_)) for item in value)
    ):
        raise ValueError(f"{name} must be a (width, height) pair in inches.")
    width, height = (float(item) for item in value)
    if not (np.isfinite(width) and np.isfinite(height)) or width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive and finite; got {value!r}.")
    return width, height


def _draw_probe_pc_extrapolation(
    fits: Mapping[str, Mapping[str, Any]],
    pairwise_fits: Mapping[str, Mapping[str, Any]],
    curves_by_metric: Mapping[str, Sequence[Mapping[str, Any]]],
    collapse_curves_by_metric: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_metrics: Sequence[str],
    *,
    sq_units: str,
    mi_units: str,
    cmap: str,
    show_errorbars: bool,
    capsize: float,
    inset_corner: str,
    inset_size: tuple[float, float],
    inset_pad: tuple[float, float],
    inset_fontsize: float,
    inset_facecolor: str,
    inset_alpha: float,
    inset_headroom: float,
    annotation_loc: str | None,
    annotation_fontsize: float | None,
    legend_loc: str | None,
    legend_fontsize: float | None,
    legend_ncols: int,
    raw_title: str | None,
    collapse_title: str | None,
    suptitle: str | None,
    raw_ylim: Any,
    raw_panel_size: tuple[float, float],
    extrapolation_panel_size: tuple[float, float],
    figsize: tuple[float, float] | None,
    dpi: int,
) -> tuple[Any, np.ndarray]:
    """Draw the raw collapse above the two Lyu extrapolation panels."""
    metric_count = len(selected_metrics)
    raw_width, raw_height = _panel_size(raw_panel_size, "raw_panel_size")
    ext_width, ext_height = _panel_size(
        extrapolation_panel_size, "extrapolation_panel_size"
    )
    if figsize is None:
        figsize = (
            max(raw_width, 2.0 * ext_width + _EXTRAPOLATION_GUTTER)
            + _EXTRAPOLATION_WIDTH_MARGIN,
            metric_count
            * (raw_height + ext_height + _EXTRAPOLATION_ROW_MARGIN)
            + _EXTRAPOLATION_SUPTITLE_MARGIN,
        )
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    # Rows are shared out by requested axes height, and every panel is locked to
    # its requested aspect, so the realized footprints keep the requested ratio
    # even when an explicit `figsize` scales them all.
    height_ratios = [
        ratio for _ in selected_metrics for ratio in (raw_height, ext_height)
    ]
    grid = fig.add_gridspec(2 * metric_count, 2, height_ratios=height_ratios)
    all_sizes = sorted(
        {int(curve["L"]) for curves in curves_by_metric.values() for curve in curves}
    )
    colors = _color_map_by_size(all_sizes, cmap)
    axes = []
    rectangle = _inset_rectangle(inset_corner, inset_size, inset_pad)
    top_legend_loc = legend_loc or "center left"
    top_annotation_loc = annotation_loc or "lower left"
    for row, metric in enumerate(selected_metrics):
        ax_raw = fig.add_subplot(grid[2 * row, :])
        # The top panel spans both columns, so without an aspect lock it would
        # stretch to the full figure width while the two extrapolation panels
        # keep their own. Locking it to `raw_panel_size` is what lets the
        # default put the three panels on a common span.
        ax_raw.set_box_aspect(raw_height / raw_width)
        ax_scaled = ax_raw.inset_axes(rectangle)
        _style_inset(
            ax_scaled,
            inset_fontsize,
            inset_facecolor,
            inset_alpha,
            rectangle,
        )
        ax_scaled.set_zorder(5.0)
        _draw_probe_pc_metric(
            ax_raw,
            ax_scaled,
            curves=curves_by_metric[metric],
            collapse_curves=collapse_curves_by_metric[metric],
            metric=metric,
            fit=fits[metric],
            colors=colors,
            sq_units=sq_units,
            mi_units=mi_units,
            show_errorbars=show_errorbars,
            capsize=capsize,
            collapse_inset=True,
            inset_fontsize=inset_fontsize,
            inset_headroom=inset_headroom,
            inset_corner=inset_corner,
            legend_loc=top_legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncols=legend_ncols,
            annotation_loc=top_annotation_loc,
            annotation_fontsize=annotation_fontsize,
            raw_title=raw_title,
            collapse_title=collapse_title,
            raw_ylim=_resolve_raw_ylim(raw_ylim, metric),
        )
        ax_pc = fig.add_subplot(grid[2 * row + 1, 0])
        ax_nu = fig.add_subplot(grid[2 * row + 1, 1])
        for ax in (ax_pc, ax_nu):
            ax.set_box_aspect(ext_height / ext_width)
        _draw_lyu_extrapolation_row(
            ax_pc,
            ax_nu,
            pairwise_fits[metric],
            metric,
            cmap=cmap,
            show_errorbars=show_errorbars,
            capsize=capsize,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            annotation_loc=annotation_loc,
            annotation_fontsize=annotation_fontsize,
            title_prefix=metric_count > 1,
        )
        _panel_label(ax_raw, f"{chr(ord('a') + 3 * row)})")
        _panel_label(ax_pc, f"{chr(ord('b') + 3 * row)})")
        _panel_label(ax_nu, f"{chr(ord('c') + 3 * row)})")
        axes.append((ax_raw, ax_scaled, ax_pc, ax_nu))

    fig.suptitle(
        suptitle
        if suptitle is not None
        else "Fixed-time probe collapse and pairwise finite-size extrapolation"
    )
    return fig, np.asarray(axes, dtype=object)


def probe_pc_collapse(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    l_by_file: Mapping[str, int] | None = None,
    metrics: Sequence[str] | str | None = None,
    p_range: tuple[float | None, float | None] = (None, None),
    collapse_l_range: tuple[int | None, int | None] = (None, None),
    pc_bounds: tuple[float, float] | None = None,
    nu_bounds: tuple[float, float] = (0.3, 5.0),
    x_bounds: tuple[float, float] = (0.0, 8.0),
    fixed_pc: float | None = None,
    fixed_nu: float | None = None,
    fixed_x: float | Mapping[str, float] | None = None,
    correction_to_scaling: bool = False,
    extrapolate: bool = False,
    omega_bounds: tuple[float, float] = (0.1, 4.0),
    fixed_omega: float | Mapping[str, float] | None = None,
    f1_order: int = 2,
    sq_units: str = "bits",
    mi_units: str = "bits",
    objective_grid_points: int | tuple[int, int] = 151,
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
    raw_ylim: (
        tuple[float | None, float | None]
        | Mapping[str, tuple[float | None, float | None]]
        | None
    ) = None,
    raw_panel_size: tuple[float, float] = (9.4, 5.3),
    extrapolation_panel_size: tuple[float, float] = (4.15, 3.5),
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

    **Several files may target one ``L``** -- an interleaved refinement of the
    critical window, or a rerun adding realizations -- and they are pooled into
    a single curve. Disjoint p grids simply interleave. A p point that more
    than one file carries is pooled by realization count via Welford's
    parallel merge, so its mean and standard error are exactly what one run
    over the combined trajectories would have reported; the error therefore
    shrinks as it should rather than being taken from whichever file was read
    last. Files at one ``L`` must agree on the readout time ``t`` -- two
    readout times are two different observables, not more statistics -- and a
    file with no ``realizations`` column falls back to inverse-variance
    weighting with a warning. **A single file holding a p point twice is
    pooled the same way**, since a rerun is as often concatenated into an
    existing scan as left beside it. Such a merge also tends to leave the
    repeated metadata cells (``N``, ``circ_type``, ``circuit_name``,
    ``realizations``) blank on the rows it appended; those rows are kept and
    their realization count is refilled from the file's own constant value.

    Set ``correction_to_scaling=True`` to fit the leading irrelevant
    correction
    ``L^x I(p,L)=F_0(u)+L^{-omega}F_1(u)``, where
    ``u=(p-p_c)L^{1/nu}``. The collapse panel then subtracts the fitted
    ``L^{-omega}F_1(u)`` term. ``omega`` is fitted within ``omega_bounds``
    unless ``fixed_omega`` is supplied, and may be fixed separately for each
    metric with a mapping. The Zabalo neighbour objective leaves ``F_0``
    nonparametric, while ``F_1`` is profiled out as a weighted
    polynomial of order ``f1_order``. Restrict ``p_range`` to the scaling
    window when using this polynomial ansatz.

    Every collapse is scored with the exact Zabalo--Kawashima--Ito objective:
    the scaled points from all sizes are pooled and sorted, each interior
    point is compared with the line through its neighbours, and the three
    quoted standard errors are propagated exactly. No relative error floor or
    interpolation grid is used. Reported ``p_c`` and ``nu`` errors are the
    larger half-widths of the one-parameter ``O <= 1.3 O*`` slices; the old
    local-curvature estimates remain available as ``*_curvature_stderr``.

    ``collapse_l_range=(L_min, L_max)`` inclusively selects which sizes enter
    the fit and collapse inset; use ``None`` for either open endpoint. All
    loaded sizes remain visible in the raw panel.

    Set ``extrapolate=True`` for a three-panel figure: the ordinary raw scan
    and collapse inset on top, followed by the pairwise large-``L`` analysis
    of Fig. 3 in Lyu et al. (2026). Every ``(L1,L2)`` combination from the
    selected collapse-size range is fitted independently with the original
    ansatz. A Fig.-7-style supplemental grid shows every pairwise collapse
    with an inset of the curve difference at ``nu*`` and ``nu* +/- dnu``.
    A second supplemental figure recreates Zabalo et al. Fig. S4, including
    the white ``O=1.3 O*`` contour. This mode is deliberately
    incompatible with ``correction_to_scaling=True`` and fixed ``p_c`` or
    ``nu``.

    ``raw_ylim`` sets the raw panel's y limits explicitly, in both layouts. It
    is applied after autoscaling and after the inset headroom reservation, so
    it always wins; either endpoint may be ``None`` to leave that side alone.
    That is usually what you want, because the ``y=0`` reference line puts zero
    inside the autoscale range and matplotlib then adds its default margin
    below it -- ``raw_ylim=(0, None)`` pins the floor to zero and leaves the
    top free. Pass a mapping from metric name to a pair to set them per metric,
    which is what stacked four-probe figures need.

    ``raw_panel_size`` and ``extrapolation_panel_size`` size that figure's
    panels, as ``(width, height)`` axes footprints in inches; the second sets
    *each* of the two extrapolation panels, which are always equal. Both are
    ignored when ``extrapolate=False``. Every panel is locked to the requested
    aspect and the figure size is derived from them, so the realized footprints
    come out at the requested inches. The defaults are chosen so the top panel
    spans exactly the two extrapolation panels plus their gutter; changing one
    without the other breaks that alignment, and a top width of about
    ``2 * width + 1.03`` restores it to within half a percent (the gutter
    itself creeps up a little as the panels grow). Passing ``figsize``
    explicitly overrides the
    derived size -- the panels then keep their aspect ratios but scale to fit,
    so an under-sized ``figsize`` shrinks them rather than distorting them.

    Each collapse is drawn by default as an inset in the top-right corner of
    its raw panel, with one legend per row on the raw axes. ``inset_size``
    and ``inset_pad`` are fractions of the parent axes;
    ``collapse_inset=False`` restores the side-by-side layout, which keeps a
    legend on both panels. ``raw_title`` overrides the per-metric title;
    ``None`` keeps ``"<metric title> — raw"``.
    """
    if extrapolate and correction_to_scaling:
        raise ValueError(
            "extrapolate=True uses only the original scaling ansatz and "
            "cannot be combined with correction_to_scaling=True."
        )
    if extrapolate and (fixed_pc is not None or fixed_nu is not None):
        raise ValueError(
            "extrapolate=True must fit p_c and nu independently for every "
            "size pair; fixed_pc and fixed_nu must both be None."
        )
    if extrapolate and not collapse_inset:
        raise ValueError(
            "extrapolate=True requires collapse_inset=True so the top panel "
            "retains the original raw-plus-inset layout."
        )
    if fixed_omega is not None and not correction_to_scaling:
        raise ValueError(
            "fixed_omega requires correction_to_scaling=True."
        )
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
    collapse_curves_by_metric = {}
    all_p = []
    for metric in selected_metrics:
        curves = [
            _load_probe_pc_curve(path, metric, l_by_file, p_range)
            for path in paths
        ]
        # Several scans may target one L -- an interleaved refinement of the
        # critical window, or a rerun adding realizations. Pool them before
        # anything downstream sees a size twice.
        curves = _merge_probe_pc_curves(curves, metric)
        scale = sq_scale if metric == "sq" else mi_scale
        for curve in curves:
            curve["value"] *= scale
            curve["dvalue"] *= scale
        collapse_curves = _select_collapse_curves(curves, collapse_l_range)
        collapse_sizes = [int(curve["L"]) for curve in collapse_curves]
        for curve in collapse_curves:
            if np.any(np.asarray(curve["dvalue"], dtype=float) <= 0.0):
                raise ValueError(
                    "The exact Zabalo objective requires strictly positive "
                    f"standard errors; found a zero error for L={curve['L']}."
                )
        if len(collapse_curves) < 3:
            warnings.warn("Critical collapse has fewer than three system sizes.")
        if correction_to_scaling and len(collapse_curves) < 4:
            warnings.warn(
                "Correction-to-scaling fits are weakly constrained with fewer "
                "than four system sizes."
            )
        if extrapolate and len(collapse_curves) < 4:
            raise ValueError(
                "extrapolate=True requires at least four sizes inside "
                f"collapse_l_range; selected {collapse_sizes}."
            )
        curves_by_metric[metric] = curves
        collapse_curves_by_metric[metric] = collapse_curves
        all_p.extend(np.concatenate([curve["p"] for curve in collapse_curves]))

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
        if isinstance(fixed_omega, Mapping):
            metric_fixed_omega = fixed_omega.get(metric)
        elif fixed_omega is not None:
            metric_fixed_omega = float(fixed_omega)
        else:
            metric_fixed_omega = None
        fits[metric] = _fit_probe_pc_metric(
            collapse_curves_by_metric[metric],
            metric,
            pc_bounds=pc_bounds,
            nu_bounds=nu_bounds,
            x_bounds=x_bounds,
            fixed_pc=fixed_pc,
            fixed_nu=fixed_nu,
            fixed_x=metric_fixed_x,
            correction_to_scaling=correction_to_scaling,
            omega_bounds=omega_bounds,
            fixed_omega=metric_fixed_omega,
            f1_order=f1_order,
            seed=seed + index,
        )
        fit = fits[metric]
        _apply_objective_threshold_errors(fit)
        omega_text = (
            ""
            if not correction_to_scaling
            else (
                f", omega={fit['omega']:.6g} "
                f"± {fit['omega_stderr']:.3g}"
            )
        )
        print(
            f"{metric}: p_c={fit['pc']:.6g} ± {fit['pc_stderr']:.3g}, "
            f"nu={fit['nu']:.6g} ± {fit['nu_stderr']:.3g}, "
            f"x={fit['x']:.6g} ± {fit['x_stderr']:.3g}"
            f"{omega_text}, "
            f"score={fit['score']:.6g}"
        )

    pairwise_fits = None
    if extrapolate:
        pairwise_fits = {}
        for index, metric in enumerate(selected_metrics):
            if isinstance(fixed_x, Mapping):
                metric_fixed_x = fixed_x.get(metric)
            elif fixed_x is not None:
                metric_fixed_x = float(fixed_x)
            else:
                metric_fixed_x = 0.0 if metric == "sq" else None
            pairwise_fits[metric] = _fit_lyu_pairwise_metric(
                collapse_curves_by_metric[metric],
                metric,
                pc_bounds=pc_bounds,
                nu_bounds=nu_bounds,
                x_bounds=x_bounds,
                fixed_x=metric_fixed_x,
                seed=seed + 10_000 * index,
            )
            summary = pairwise_fits[metric]
            for pair in summary["pairs"]:
                collapse = pair["collapse"]
                print(
                    f"{metric} (L1,L2)=({pair['L1']},{pair['L2']}): "
                    f"p_c(cross)={pair['crossing']['pc']:.6g} ± "
                    f"{pair['crossing']['stderr']:.3g}, "
                    f"p_c(collapse)={collapse['pc']:.6g}, "
                    f"nu={collapse['nu']:.6g}, O*={collapse['score']:.6g}"
                )
            pc_line = summary["pc_crossing_extrapolation"]
            nu_line = summary["nu_extrapolation"]
            print(
                f"{metric}: p_c(inf)={pc_line['intercept']:.6g} ± "
                f"{pc_line['intercept_stderr']:.3g}, "
                f"nu(inf)={nu_line['intercept']:.6g} ± "
                f"{nu_line['intercept_stderr']:.3g}; excluded pair="
                f"{summary['excluded_pair']}"
            )

        fig, axes = _draw_probe_pc_extrapolation(
            fits,
            pairwise_fits,
            curves_by_metric,
            collapse_curves_by_metric,
            selected_metrics,
            sq_units=sq_units,
            mi_units=mi_units,
            cmap=cmap,
            show_errorbars=show_errorbars,
            capsize=capsize,
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
            suptitle=suptitle,
            raw_ylim=raw_ylim,
            raw_panel_size=raw_panel_size,
            extrapolation_panel_size=extrapolation_panel_size,
            figsize=figsize,
            dpi=dpi,
        )
        collapse_sizes_by_metric = {
            metric: tuple(int(curve["L"]) for curve in curves)
            for metric, curves in collapse_curves_by_metric.items()
        }
        result = {
            "figure": fig,
            "axes": axes,
            "collapse_inset": True,
            "extrapolate": True,
            "probes": probes,
            "metrics": selected_metrics,
            "correction_to_scaling": False,
            "sq_units": sq_units,
            "mi_units": mi_units,
            "fits": fits,
            "global_fits": fits,
            "pairwise_fits": pairwise_fits,
            "curves": curves_by_metric,
            "collapse_curves": collapse_curves_by_metric,
            "collapse_l_range": collapse_l_range,
            "collapse_sizes": collapse_sizes_by_metric,
        }
        # Show the main composite first, then the pair-by-pair Fig.-7-style
        # collapses and finally the objective surface.  Notebook output thus
        # follows the analysis from result to diagnostic.
        _show(fig, show)
        pairwise_supplemental = _draw_pairwise_collapse_figures(
            pairwise_fits,
            collapse_curves_by_metric,
            selected_metrics,
            sq_units=sq_units,
            mi_units=mi_units,
            cmap=cmap,
            show_errorbars=show_errorbars,
            capsize=capsize,
            dpi=dpi,
        )
        for panel in pairwise_supplemental.values():
            _show(panel["figure"], show)
        result["pairwise_supplemental"] = pairwise_supplemental
        result["pairwise_figures"] = {
            metric: panel["figure"]
            for metric, panel in pairwise_supplemental.items()
        }
        if len(selected_metrics) == 1:
            only_panel = pairwise_supplemental[selected_metrics[0]]
            result["pairwise_figure"] = only_panel["figure"]
            result["pairwise_axes"] = only_panel["axes"]
        objective_result = probe_pc_objective(
            result,
            grid_points=objective_grid_points,
            cmap=cmap,
            dpi=dpi,
            show=show,
        )
        result["objective_figure"] = objective_result["figure"]
        result["objective_axes"] = objective_result["axes"]
        result["objective_surfaces"] = objective_result["surfaces"]
        result["supplemental"] = objective_result
        return result

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
    resolved_legend_loc = legend_loc or (
        "center left" if collapse_inset else "best"
    )
    resolved_annotation_loc = annotation_loc or (
        "lower left" if collapse_inset else "lower right"
    )
    sizes = sorted(
        {curve["L"] for curves in curves_by_metric.values() for curve in curves}
    )
    colors = _color_map_by_size(sizes, cmap)
    for row, metric in enumerate(selected_metrics):
        ax_raw, ax_scaled = axes[row]
        _draw_probe_pc_metric(
            ax_raw,
            ax_scaled,
            curves=curves_by_metric[metric],
            collapse_curves=collapse_curves_by_metric[metric],
            metric=metric,
            fit=fits[metric],
            colors=colors,
            sq_units=sq_units,
            mi_units=mi_units,
            show_errorbars=show_errorbars,
            capsize=capsize,
            collapse_inset=collapse_inset,
            inset_fontsize=inset_fontsize,
            inset_headroom=inset_headroom,
            inset_corner=inset_corner,
            legend_loc=resolved_legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncols=legend_ncols,
            annotation_loc=resolved_annotation_loc,
            annotation_fontsize=annotation_fontsize,
            raw_title=raw_title,
            collapse_title=collapse_title,
            raw_ylim=_resolve_raw_ylim(raw_ylim, metric),
        )
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
        "extrapolate": False,
        "probes": probes,
        "metrics": selected_metrics,
        "correction_to_scaling": correction_to_scaling,
        "sq_units": sq_units,
        "mi_units": mi_units,
        "fits": fits,
        "global_fits": fits,
        "pairwise_fits": None,
        "curves": curves_by_metric,
        "collapse_curves": collapse_curves_by_metric,
        "collapse_l_range": collapse_l_range,
        "collapse_sizes": {
            metric: tuple(int(curve["L"]) for curve in curves)
            for metric, curves in collapse_curves_by_metric.items()
        },
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

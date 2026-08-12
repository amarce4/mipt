"""Mode-4 probe dynamics: distance-resolved pairs and probe triangles."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from .collapse import _probe2_metric_spec
from .loading import (
    _constant_csv_value,
    _find_column,
    _parse_circuit_name,
    _parse_p,
    _parse_size,
    _resolve_files,
)
from .plotting import (
    _annotate_axes,
    _font_kwargs,
    _inset_overlay_defaults,
    _set_secondary_title,
    _mi_unit_spec,
    _paired_axes,
    _reserve_axis_space,
    _show,
)


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


def _normalize_fit_parity(fit_parity: str) -> str:
    parity = str(fit_parity).strip().lower()
    if parity not in {"all", "odd", "even"}:
        raise ValueError("fit_parity must be 'all', 'odd', or 'even'.")
    return parity


def _distance_matches_parity(distance: int, fit_parity: str) -> bool:
    return (
        fit_parity == "all"
        or (fit_parity == "odd" and distance % 2 == 1)
        or (fit_parity == "even" and distance % 2 == 0)
    )


def probe_distance_collapse(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    metric: str = "mi",
    mi_units: str = "bits",
    ansatz: str = "lightcone",
    eta: float | None = None,
    z: float | None = 1.0,
    alpha: float = 1.0,
    delay: float | None = None,
    eta_bounds: tuple[float, float] = (0.0, 2.0),
    z_bounds: tuple[float, float] = (0.5, 2.0),
    delay_bounds: tuple[float, float] = (0.0, 4.0),
    distance_load: tuple[int, int | None] = (1, None),
    fit_distance: tuple[int, int | None] = (2, None),
    fit_parity: str = "all",
    fit_tau: tuple[float, float | None] = (1.0, None),
    fit_x: tuple[float | None, float | None] = (None, None),
    tau_bounds: tuple[float, float] | None = None,
    fit_value_min: float = 1e-10,
    interpolation_points: int = 250,
    minimum_pair_coverage: float = 0.8,
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
    collapse_inset: bool = True,
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
    legend_fontsize: float = 8,
    legend_ncols: int | None = None,
    raw_title: str | None = "Distance-resolved probe dynamics",
    collapse_title: str | None = None,
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
    r"""Plot and fit mode-4 distance-resolved two-probe data.

    The default finite-size light-cone ansatz is

    ``O(tau,r,L) = ell_r**(-eta)
    F(alpha*(tau-delay*ell_r**z)/L**z)``,

    with ``ell_r=(L/pi) sin(pi*r/L)``. It separates the distance-dependent
    signal-arrival time from the system-size relaxation time. A numeric
    ``eta``, ``z``, or ``delay`` fixes that parameter; ``None`` fits it.
    Fitting ``z`` requires at least two system sizes.

    ``ansatz="cylinder"`` uses the finite-cylinder coordinate

    ``x=log(1+sinh(pi*alpha*tau/L)/sin(pi*r/L))``

    and fits only the vertical exponent ``eta``. ``ansatz="legacy_local"``
    retains the earlier ``alpha*tau/ell_r**z`` form for explicit comparisons.
    ``alpha`` is supplied rather than fitted because an overall horizontal
    rescaling of an unknown scaling function is not identifiable.

    The mean fitted pairwise overlap must retain at least
    ``minimum_pair_coverage`` of the points. ``fit_distance=(2,None)`` leaves
    the microscopic ``r=1`` curve visible but excludes it from the default fit.
    ``fit_parity`` may be ``"all"``, ``"odd"``, or ``"even"``. Odd/even
    selection applies to the fit, bootstrap, score, and collapse panel; the raw
    dynamics panel continues to show every separation loaded by
    ``distance_load``.

    The collapse is drawn by default as an inset in the bottom-right corner
    of the raw panel, lifted clear of the bottom edge so both x axes stay
    readable, and both panels share the raw panel's legend. ``inset_size``
    and ``inset_pad`` are fractions of the parent axes;
    ``collapse_inset=False`` restores the side-by-side layout.
    """
    ansatz_key = str(ansatz).strip().lower().replace("-", "_")
    ansatz_key = {
        "light_cone": "lightcone",
        "finite_size_lightcone": "lightcone",
        "finite_size_light_cone": "lightcone",
        "finite_cylinder": "cylinder",
        "conformal": "cylinder",
        "local": "legacy_local",
        "legacy": "legacy_local",
        "original": "legacy_local",
    }.get(ansatz_key, ansatz_key)
    if ansatz_key not in {"lightcone", "cylinder", "legacy_local"}:
        raise ValueError(
            "ansatz must be 'lightcone', 'cylinder', or 'legacy_local'."
        )
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    if eta is not None and (not np.isfinite(eta) or eta < 0):
        raise ValueError("eta must be non-negative or None.")
    if z is not None and (not np.isfinite(z) or z <= 0):
        raise ValueError("z must be positive or None.")
    if delay is not None and (not np.isfinite(delay) or delay < 0):
        raise ValueError("delay must be non-negative or None.")
    if interpolation_points < 3:
        raise ValueError("interpolation_points must be at least 3.")
    if errorbar_points < 1:
        raise ValueError("errorbar_points must be positive.")
    if not 0 < minimum_pair_coverage <= 1:
        raise ValueError("minimum_pair_coverage must lie in (0,1].")
    for name, bounds in (
        ("eta_bounds", eta_bounds),
        ("z_bounds", z_bounds),
        ("delay_bounds", delay_bounds),
    ):
        if (
            len(bounds) != 2
            or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(f"{name} must contain two increasing finite values.")

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
    # Both the MI and the logarithmic negativity carry entropy units; the bare
    # negativity does not. The spec is the single place that records which.
    if metric_spec["scale_with_mi_units"]:
        metric_spec = dict(metric_spec)
        metric_spec["bootstrap_upper"] *= mi_scale
        for curve in curves:
            curve["value"] *= mi_scale
            curve["dvalue"] *= mi_scale
    curves.sort(key=lambda curve: (curve["L"], curve["distance"], str(curve["path"])))
    if len(curves) < 2:
        raise ValueError("At least two distance curves are required for a collapse.")
    unique_sizes = sorted({curve["L"] for curve in curves})
    if ansatz_key == "lightcone" and z is None and len(unique_sizes) < 2:
        raise ValueError(
            "The light-cone exponent z cannot be fitted from one system size. "
            "Supply a numeric z (normally z=1) or include at least two L values."
        )

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
    fit_parity = _normalize_fit_parity(fit_parity)

    def matches_fit_parity(curve):
        return _distance_matches_parity(curve["distance"], fit_parity)

    summary = pd.DataFrame(
        {
            "L": [curve["L"] for curve in curves],
            "r": [curve["distance"] for curve in curves],
            "chord_length": [curve["chord"] for curve in curves],
            "p": [curve["p"] for curve in curves],
            "t_eq": [curve["t_eq"] for curve in curves],
            "tau_max": [np.max(curve["tau"]) for curve in curves],
            "points": [len(curve["tau"]) for curve in curves],
            "fit_parity_eligible": [
                matches_fit_parity(curve) for curve in curves
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

    eta_active = True
    z_active = ansatz_key in {"lightcone", "legacy_local"}
    delay_active = ansatz_key == "lightcone"

    def scaled_x(curve, z_value, delay_value):
        tau = curve["tau"]
        if ansatz_key == "lightcone":
            return (
                alpha
                * (tau - delay_value * curve["chord"] ** z_value)
                / float(curve["L"]) ** z_value
            )
        if ansatz_key == "legacy_local":
            return alpha * tau / curve["chord"] ** z_value

        argument = np.pi * alpha * tau / float(curve["L"])
        log_sinh = np.full_like(argument, -np.inf, dtype=float)
        positive = argument > 0
        small = positive & (argument < 20.0)
        log_sinh[small] = np.log(np.sinh(argument[small]))
        large = positive & ~small
        log_sinh[large] = (
            argument[large]
            - np.log(2.0)
            + np.log1p(-np.exp(-2.0 * argument[large]))
        )
        spatial = np.sin(np.pi * curve["distance"] / float(curve["L"]))
        return np.logaddexp(0.0, log_sinh - np.log(spatial))

    def transformed(curve, eta_value, z_value, delay_value):
        x = scaled_x(curve, z_value, delay_value)
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

    def pair_diagnostics(curve_a, curve_b, eta_value, z_value, delay_value):
        xa, ya, dya, _ = transformed(
            curve_a, eta_value, z_value, delay_value
        )
        xb, yb, dyb, _ = transformed(
            curve_b, eta_value, z_value, delay_value
        )
        if len(xa) < 3 or len(xb) < 3:
            return np.inf, 0.0
        lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
        if hi <= lo:
            return np.inf, 0.0
        points_in_overlap = np.count_nonzero((xa >= lo) & (xa <= hi))
        points_in_overlap += np.count_nonzero((xb >= lo) & (xb <= hi))
        coverage = points_in_overlap / float(len(xa) + len(xb))
        grid = np.linspace(lo, hi, interpolation_points)
        ya_g, yb_g = np.interp(grid, xa, ya), np.interp(grid, xb, yb)
        dya_g, dyb_g = np.interp(grid, xa, dya), np.interp(grid, xb, dyb)
        typical = max(
            np.nanmedian(np.abs(ya_g)),
            np.nanmedian(np.abs(yb_g)),
            absolute_error_floor,
        )
        floor = max(absolute_error_floor, relative_error_floor * typical)
        score = np.mean(
            (ya_g - yb_g) ** 2 / (dya_g**2 + dyb_g**2 + floor**2)
        )
        return float(score), float(coverage)

    def eligible_curves(curve_set):
        selected = [
            curve
            for curve in curve_set
            if curve["distance"] >= fit_distance_min
            and matches_fit_parity(curve)
            and (
                fit_distance_max is None
                or curve["distance"] <= fit_distance_max
            )
        ]
        return selected

    initially_fitted_curves = eligible_curves(curves)
    if len(initially_fitted_curves) < 2:
        raise ValueError(
            f"fit_parity={fit_parity!r} and fit_distance={fit_distance!r} "
            "retain fewer than two separation curves. Widen fit_distance, "
            "choose another parity, or load more separations."
        )

    def score_values(
        eta_value,
        z_value,
        delay_value,
        curve_set=curves,
    ):
        fitted_curve_set = eligible_curves(curve_set)
        diagnostics = [
            pair_diagnostics(
                fitted_curve_set[i],
                fitted_curve_set[j],
                eta_value,
                z_value,
                delay_value,
            )
            for i in range(len(fitted_curve_set))
            for j in range(i + 1, len(fitted_curve_set))
        ]
        finite = [score for score, _ in diagnostics if np.isfinite(score)]
        coverage = [value for score, value in diagnostics if np.isfinite(score)]
        return (
            float(np.mean(finite))
            if (
                len(finite) == len(diagnostics)
                and finite
                and float(np.mean(coverage)) >= minimum_pair_coverage
            )
            else np.inf
        )

    fixed_parameters = {
        "eta": None if eta is None else float(eta),
        "z": (
            None
            if z_active and z is None
            else float(z)
            if z_active
            else np.nan
        ),
        "delay": (
            None
            if delay_active and delay is None
            else float(delay)
            if delay_active
            else np.nan
        ),
    }
    free_names = [
        name
        for name, active in (
            ("eta", eta_active),
            ("z", z_active),
            ("delay", delay_active),
        )
        if active and fixed_parameters[name] is None
    ]
    parameter_bounds = {
        "eta": eta_bounds,
        "z": z_bounds,
        "delay": delay_bounds,
    }

    def unpack(values):
        parameters = dict(fixed_parameters)
        for name, value in zip(free_names, np.atleast_1d(values)):
            parameters[name] = float(value)
        return parameters

    def objective(values, curve_set):
        parameters = unpack(values)
        return score_values(
            parameters["eta"],
            parameters["z"],
            parameters["delay"],
            curve_set,
        )

    best_parameters: dict[str, float] | None = None

    def fit_parameters(curve_set, *, bootstrap=False):
        if not free_names:
            parameters = unpack([])
            return parameters, objective([], curve_set)
        bounds = [parameter_bounds[name] for name in free_names]
        if len(free_names) == 1:
            result = minimize_scalar(
                lambda trial: objective([trial], curve_set),
                bounds=bounds[0],
                method="bounded",
                options={"xatol": 1e-5 if bootstrap else 1e-7},
            )
            return unpack([result.x]), float(result.fun)
        if bootstrap:
            if best_parameters is None:
                raise RuntimeError("The main collapse fit must precede bootstrapping.")
            result = minimize(
                lambda values: objective(values, curve_set),
                x0=np.asarray(
                    [best_parameters[name] for name in free_names], dtype=float
                ),
                method="Powell",
                bounds=bounds,
                options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 400},
            )
            return unpack(result.x), float(result.fun)
        result = differential_evolution(
            lambda values: objective(values, curve_set),
            bounds=bounds,
            seed=bootstrap_seed,
            polish=True,
            tol=1e-7,
        )
        return unpack(result.x), float(result.fun)

    best_parameters, best_score = fit_parameters(curves)
    best_eta = float(best_parameters["eta"])
    best_z = float(best_parameters["z"])
    best_delay = float(best_parameters["delay"])
    if not np.isfinite(best_score):
        raise RuntimeError(
            "No fitted curve pair has at least three overlapping scaled-time "
            "points with the required coverage. Widen fit_x/fit_tau, lower "
            "minimum_pair_coverage, or include more distances."
        )

    estimates: list[dict[str, float]] = []
    if bootstrap_samples >= 2 and free_names:
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
            trial_parameters, trial_score = fit_parameters(
                synthetic_curves, bootstrap=True
            )
            if np.all(
                np.isfinite(
                    [trial_score] + [trial_parameters[name] for name in free_names]
                )
            ):
                estimates.append(trial_parameters)
            if (index + 1) % 25 == 0 or index + 1 == bootstrap_samples:
                print(f"Bootstrap fits: {index + 1}/{bootstrap_samples}")

    def parameter_stderr(name):
        if name not in free_names:
            return 0.0 if np.isfinite(best_parameters[name]) else np.nan
        if len(estimates) < 2:
            return np.nan
        return float(
            np.std([sample[name] for sample in estimates], ddof=1)
        )

    eta_fixed = "eta" not in free_names
    z_fixed = "z" not in free_names
    delay_fixed = "delay" not in free_names
    eta_stderr = parameter_stderr("eta")
    z_stderr = parameter_stderr("z")
    delay_stderr = parameter_stderr("delay")
    fitted_curves = eligible_curves(curves)
    best_diagnostics = [
        pair_diagnostics(
            fitted_curves[i],
            fitted_curves[j],
            best_eta,
            best_z,
            best_delay,
        )
        for i in range(len(fitted_curves))
        for j in range(i + 1, len(fitted_curves))
    ]
    finite_diagnostics = [
        (score, coverage)
        for score, coverage in best_diagnostics
        if np.isfinite(score)
    ]
    mean_pair_coverage = float(
        np.mean([coverage for _, coverage in finite_diagnostics])
    )
    minimum_fitted_pair_coverage = float(
        np.min([coverage for _, coverage in finite_diagnostics])
    )
    fitted_pairs = len(finite_diagnostics)
    possible_pairs = len(fitted_curves) * (len(fitted_curves) - 1) // 2

    print(f"ansatz = {ansatz_key}")
    print(f"metric = {metric_spec['key']}")
    print(f"fit parity = {fit_parity}")
    print(
        f"eta = {best_eta:.6f}"
        + (" (fixed)" if eta_fixed else f" ± {eta_stderr:.6f}")
    )
    if z_active:
        print(
            f"z = {best_z:.6f}"
            + (" (fixed)" if z_fixed else f" ± {z_stderr:.6f}")
        )
    if delay_active:
        print(
            f"delay = {best_delay:.6f}"
            + (" (fixed)" if delay_fixed else f" ± {delay_stderr:.6f}")
        )
    print(f"alpha = {alpha:.6f} (fixed)")
    print(f"collapse score = {best_score:.6g}")
    print(
        "pair coverage = "
        f"{mean_pair_coverage:.1%} mean, "
        f"{minimum_fitted_pair_coverage:.1%} minimum "
        f"({fitted_pairs}/{possible_pairs} pairs)"
    )

    chord_values = np.asarray([curve["chord"] for curve in curves], dtype=float)
    if np.ptp(chord_values) > 0:
        color_values = 0.08 + 0.84 * (
            chord_values - np.min(chord_values)
        ) / np.ptp(chord_values)
    else:
        color_values = np.full(len(curves), 0.5)
    colors = [plt.get_cmap(cmap)(value) for value in color_values]
    one_size = len({curve["L"] for curve in curves}) == 1
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
        annotation_loc = auto_annotation_loc if collapse_inset else "lower right"
    if legend_ncols is None:
        legend_ncols = 2 if len(curves) > 8 else 1
    legend_kwargs: dict[str, Any] = {
        "loc": legend_loc,
        "ncols": legend_ncols,
        "fontsize": legend_fontsize,
    }
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

        if not matches_fit_parity(curve):
            continue

        x, y, dy, mask = transformed(
            curve, best_eta, best_z, best_delay
        )
        all_x = scaled_x(curve, best_z, best_delay)
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
        metric_spec["raw_ylabel"].replace("[nats]", f"[{mi_units}]")
        if metric_spec["scale_with_mi_units"]
        else metric_spec["raw_ylabel"]
    )
    ax_raw.set_xlabel(
        raw_xlabel if raw_xlabel is not None else r"Elapsed time $\tau$"
    )
    ax_raw.set_ylabel(raw_ylabel if raw_ylabel is not None else observable_label)
    if raw_title:
        ax_raw.set_title(raw_title)
    ax_raw.set_yscale(raw_yscale)
    ax_raw.grid(alpha=0.25)
    ax_raw.legend(**legend_kwargs)
    if (tau_bounds is not None) and len(tau_bounds) == 2:
        ax_raw.set_xlim(tau_bounds)

    x_labels = {
        "lightcone": (
            r"$\alpha(\tau-\lambda\ell_r^z)/L^z$"
        ),
        "cylinder": (
            r"$\log[1+\sinh(\pi\alpha\tau/L)/\sin(\pi r/L)]$"
        ),
        "legacy_local": r"$\alpha\tau/\ell_r^z$",
    }
    ax_collapse.set_xlabel(
        collapse_xlabel if collapse_xlabel is not None else x_labels[ansatz_key],
        **inset_font,
    )
    collapse_symbol = {
        "mi": r"I(A:B)",
        "negativity": r"\mathcal{N}(A:B)",
        "log_negativity": r"E_{\mathcal{N}}(A:B)",
    }[metric_spec["key"]]
    unit_suffix = f" [{mi_units}]" if metric_spec["scale_with_mi_units"] else ""
    if collapse_ylabel is not None or not collapse_inset:
        ax_collapse.set_ylabel(
            collapse_ylabel
            if collapse_ylabel is not None
            else rf"$\ell_r^{{\eta}}{collapse_symbol}$" + unit_suffix,
            **inset_font,
        )
    collapse_titles = {
        "lightcone": "Finite-size light-cone collapse",
        "cylinder": "Finite-cylinder collapse",
        "legacy_local": "Legacy local-time collapse",
    }
    resolved_collapse_title = (
        collapse_title
        if collapse_title is not None
        else collapse_titles[ansatz_key]
    )
    if collapse_title is None and fit_parity != "all":
        resolved_collapse_title = (
            f"{fit_parity.capitalize()}-separation "
            + resolved_collapse_title.lower()
        )
    ax_collapse.set_yscale(collapse_yscale)
    ax_collapse.grid(alpha=0.25)
    if not collapse_inset:
        ax_collapse.legend(**legend_kwargs)
    _set_secondary_title(
        ax_collapse,
        resolved_collapse_title,
        inset=collapse_inset,
        fontsize=inset_fontsize + 1.0,
    )
    eta_text = rf"$\eta={best_eta:.4f}$" + (
        " fixed"
        if eta_fixed
        else rf" $\pm {eta_stderr:.4f}$"
        if np.isfinite(eta_stderr)
        else ""
    )
    parameter_lines = [eta_text]
    if fit_parity != "all":
        parameter_lines.append(f"separations = {fit_parity}")
    if z_active:
        parameter_lines.append(
            rf"$z={best_z:.4f}$"
            + (
                " fixed"
                if z_fixed
                else rf" $\pm {z_stderr:.4f}$"
                if np.isfinite(z_stderr)
                else ""
            )
        )
    if delay_active:
        parameter_lines.append(
            rf"$\lambda={best_delay:.4f}$"
            + (
                " fixed"
                if delay_fixed
                else rf" $\pm {delay_stderr:.4f}$"
                if np.isfinite(delay_stderr)
                else ""
            )
        )
    parameter_lines.extend(
        [
            rf"$\alpha={alpha:.4g}$ fixed",
            rf"score $={best_score:.3g}$",
            f"coverage = {mean_pair_coverage:.1%}",
        ]
    )
    parameter_text = "\n".join(parameter_lines)
    annotation_kwargs: dict[str, Any] = {}
    if annotation_fontsize is not None:
        annotation_kwargs["fontsize"] = annotation_fontsize
    _annotate_axes(
        ax_raw if collapse_inset else ax_collapse,
        parameter_text,
        annotation_loc,
        **annotation_kwargs,
    )
    if collapse_inset:
        _reserve_axis_space(ax_raw, reserve_side, inset_headroom)
    ansatz_titles = {
        "lightcone": (
            r"$O(\tau,r,L)=\ell_r^{-\eta}"
            r"F[\alpha(\tau-\lambda\ell_r^z)/L^z]$"
        ),
        "cylinder": (
            r"$O(\tau,r,L)=\ell_r^{-\eta}F(u_{\rm cyl})$"
        ),
        "legacy_local": (
            r"$O(\tau,r,L)=\ell_r^{-\eta}"
            r"G(\alpha\tau/\ell_r^z)$"
        ),
    }
    fig.suptitle(
        suptitle if suptitle is not None else ansatz_titles[ansatz_key],
        fontsize=14,
    )
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_raw, ax_collapse),
        "collapse_inset": collapse_inset,
        "metric": metric_spec["key"],
        "mi_units": mi_units,
        "ansatz": ansatz_key,
        "fit_parity": fit_parity,
        "eta": best_eta,
        "eta_stderr": eta_stderr,
        "eta_fixed": eta_fixed,
        "z": best_z if z_active else None,
        "z_stderr": z_stderr,
        "z_fixed": z_fixed,
        "alpha": alpha,
        "delay": best_delay if delay_active else None,
        "delay_stderr": delay_stderr,
        "delay_fixed": delay_fixed,
        "score": best_score,
        "mean_pair_coverage": mean_pair_coverage,
        "minimum_pair_coverage": minimum_fitted_pair_coverage,
        "fitted_pairs": fitted_pairs,
        "possible_pairs": possible_pairs,
        "bootstrap_successes": len(estimates),
        "summary": summary,
        "curves": curves,
        "collapsed_curves": fitted_curves,
    }


def probe3_dynamics(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    metrics: Sequence[str] = (
        "tmi",
        "average_mi",
        "min_negativity",
        "gmn",
        "joint_purity",
        "mean_single_purity",
    ),
    mi_units: str = "bits",
    geometry_ids: Sequence[int] | None = None,
    tau_range: tuple[float | None, float | None] = (None, None),
    time_axis: str = "tau",
    show_errorbands: bool = True,
    error_alpha: float = 0.16,
    linewidth: float = 1.5,
    cmap: str = "viridis",
    max_legend_entries: int = 16,
    figsize: tuple[float, float] = (15, 9),
    dpi: int = 130,
    show_summary: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    """Plot the three-probe mode-4 observables for every CFT triangle.

    Curves are colored by the geometric mean of their three chord lengths, but
    legend labels show only that effective CFT distance. This is a visualization
    rather than a one-variable collapse: inequivalent triangle shapes generally
    cannot be reduced completely to one scalar distance.

    ``time_axis="tau"`` plots post-attachment timesteps. ``"scaled"`` plots
    ``tau / (ell1*ell2*ell3)**(1/3)`` as an exploratory comparison.
    """
    metric_specs = {
        "tmi": ("TMI_mean", "TMI_stderr", "Tripartite mutual information"),
        "average_mi": (
            "average_MI_mean",
            "average_MI_stderr",
            "Average pairwise mutual information",
        ),
        "min_negativity": (
            "min_bipartite_negativity_mean",
            "min_bipartite_negativity_stderr",
            "Minimum bipartite negativity",
        ),
        "gmn": ("GMN_mean", "GMN_stderr", "GMN/fGMN"),
        "joint_purity": (
            "joint_purity_mean",
            "joint_purity_stderr",
            r"Joint purity $\mathrm{Tr}(\rho_{ABC}^2)$",
        ),
        "mean_single_purity": (
            "mean_single_purity_mean",
            "mean_single_purity_stderr",
            "Mean single-probe purity",
        ),
    }
    selected_metrics = [str(metric).strip().lower() for metric in metrics]
    if not selected_metrics:
        raise ValueError("metrics must contain at least one metric.")
    unknown = [metric for metric in selected_metrics if metric not in metric_specs]
    if unknown:
        raise ValueError(
            f"Unknown three-probe metric(s): {unknown}. "
            f"Choose from {sorted(metric_specs)}."
        )
    if time_axis not in {"tau", "scaled"}:
        raise ValueError("time_axis must be 'tau' or 'scaled'.")
    if max_legend_entries < 0:
        raise ValueError("max_legend_entries must be non-negative.")

    mi_units, mi_scale = _mi_unit_spec(mi_units)
    paths = _resolve_files(files, file_glob)
    filename_sizes = [_parse_size(path) for path in paths]
    filename_ps = [_parse_p(path) for path in paths]
    filename_circuits = [_parse_circuit_name(path) for path in paths]
    if any(not np.isfinite(value) for value in filename_ps):
        missing = [
            path.name
            for path, value in zip(paths, filename_ps)
            if not np.isfinite(value)
        ]
        raise ValueError(
            "Could not parse p from the following filename(s): "
            + ", ".join(repr(name) for name in missing)
        )
    if len(set(filename_sizes)) != 1:
        raise ValueError(
            f"Three-probe inputs contain multiple filename sizes: "
            f"{sorted(set(filename_sizes))}."
        )
    if not np.allclose(filename_ps, filename_ps[0], rtol=0.0, atol=1e-12):
        raise ValueError(
            "Three-probe inputs contain multiple filename measurement rates: "
            f"{sorted(set(filename_ps))}."
        )
    if len(set(filename_circuits)) != 1:
        raise ValueError(
            "Three-probe inputs contain multiple filename circuit types: "
            f"{sorted(set(filename_circuits))}."
        )
    filename_size = filename_sizes[0]
    filename_p = filename_ps[0]
    filename_circuit = filename_circuits[0]

    required = {
        "N",
        "p",
        "geometry_id",
        "distance_1",
        "distance_2",
        "distance_3",
        "chord_geometric_mean",
        "tau",
        "TMI_mean",
        "TMI_stderr",
        "average_MI_mean",
        "average_MI_stderr",
        "min_bipartite_negativity_mean",
        "min_bipartite_negativity_stderr",
        "GMN_mean",
        "GMN_stderr",
        "joint_purity_mean",
        "joint_purity_stderr",
        "mean_single_purity_mean",
        "mean_single_purity_stderr",
    }
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}.")
        frame = frame.copy()
        frame["source_file"] = str(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    numeric_columns = sorted(required - {"gmn_type"})
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(
        subset=[
            "N",
            "p",
            "geometry_id",
            "distance_1",
            "distance_2",
            "distance_3",
            "chord_geometric_mean",
            "tau",
        ]
    )
    if geometry_ids is not None:
        requested = {int(value) for value in geometry_ids}
        data = data.loc[data["geometry_id"].astype(int).isin(requested)]
    tau_min, tau_max = tau_range
    if tau_min is not None:
        data = data.loc[data["tau"] >= tau_min]
    if tau_max is not None:
        data = data.loc[data["tau"] <= tau_max]
    if data.empty:
        raise ValueError("No three-probe rows remain after filtering.")

    for column in ("TMI_mean", "TMI_stderr", "average_MI_mean", "average_MI_stderr"):
        data[column] *= mi_scale

    group_columns = ["source_file", "N", "p", "geometry_id"]
    groups = list(data.groupby(group_columns, sort=True))
    chord_values = np.asarray(
        [group["chord_geometric_mean"].iloc[0] for _, group in groups],
        dtype=float,
    )
    chord_min = float(np.nanmin(chord_values))
    chord_max = float(np.nanmax(chord_values))
    denominator = chord_max - chord_min
    color_map = plt.get_cmap(cmap)

    n_metrics = len(selected_metrics)
    ncols = min(3, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharex=False,
    )
    axes = list(axes_grid.flat)
    curve_records = []
    for key, group in groups:
        group = group.sort_values("tau").drop_duplicates("tau", keep="last")
        chord = float(group["chord_geometric_mean"].iloc[0])
        color_fraction = 0.5 if denominator <= 0 else (chord - chord_min) / denominator
        color = color_map(color_fraction)
        distances = tuple(
            int(group[f"distance_{index}"].iloc[0]) for index in (1, 2, 3)
        )
        size = int(group["N"].iloc[0])
        label = rf"$\ell_{{\mathrm{{eff}}}}={chord:.3g}$"
        x = group["tau"].to_numpy(dtype=float)
        if time_axis == "scaled":
            x = x / chord

        for axis, metric in zip(axes, selected_metrics):
            value_column, error_column, title = metric_specs[metric]
            y = group[value_column].to_numpy(dtype=float)
            dy = np.abs(group[error_column].to_numpy(dtype=float))
            finite = np.isfinite(x) & np.isfinite(y)
            axis.plot(
                x[finite],
                y[finite],
                color=color,
                linewidth=linewidth,
                label=label,
            )
            if show_errorbands:
                band = finite & np.isfinite(dy)
                axis.fill_between(
                    x[band],
                    y[band] - dy[band],
                    y[band] + dy[band],
                    color=color,
                    alpha=error_alpha,
                    linewidth=0,
                )
            axis.set_title(title)
        curve_records.append(
            {
                "file": key[0],
                "L": size,
                "p": float(group["p"].iloc[0]),
                "geometry_id": int(group["geometry_id"].iloc[0]),
                "distances": distances,
                "chord_geometric_mean": chord,
                "points": len(group),
            }
        )

    x_label = (
        r"Post-attachment time $\tau$"
        if time_axis == "tau"
        else r"$\tau/(\ell_{AB}\ell_{AC}\ell_{BC})^{1/3}$"
    )
    for axis, metric in zip(axes, selected_metrics):
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.25)
        if metric in {"tmi", "average_mi"}:
            # The panel title names the quantity; the label carries only its
            # unit, in the same [bits]/[nats] form the rest of the package uses.
            axis.set_ylabel(f"[{mi_units}]")
        elif metric in {"joint_purity", "mean_single_purity"}:
            axis.set_ylim(0.0, 1.02)
    for axis in axes[n_metrics:]:
        axis.set_visible(False)

    if len(groups) <= max_legend_entries and max_legend_entries > 0:
        axes[0].legend(fontsize=8, ncol=2)
    else:
        norm = plt.Normalize(chord_min, chord_max)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
        scalar.set_array([])
        fig.colorbar(
            scalar,
            ax=axes[:n_metrics],
            label=r"Geometric-mean chord length $(\ell_1\ell_2\ell_3)^{1/3}$",
            shrink=0.85,
        )

    summary = pd.DataFrame(curve_records)
    if show_summary:
        try:
            from IPython.display import display

            display(summary)
        except ImportError:
            print(summary.to_string(index=False))

    fig.suptitle(
        "Three-probe mode-4 dynamics: "
        rf"$p={filename_p:g}$, {filename_circuit}, $L={filename_size}$",
        fontsize=14,
    )
    fig.tight_layout()
    _show(fig, show)
    return {
        "figure": fig,
        "axes": tuple(axes[:n_metrics]),
        "data": data,
        "summary": summary,
        "metrics": tuple(selected_metrics),
        "mi_units": mi_units,
        "time_axis": time_axis,
        "filename_metadata": {
            "L": filename_size,
            "p": filename_p,
            "circuit": filename_circuit,
        },
    }

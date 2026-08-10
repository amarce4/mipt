"""Measurement-record free energy and the effective central charge."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .loading import _resolve_files
from .plotting import _show


def _free_energy_linear_fit(
    x: Iterable[float],
    y: Iterable[float],
    sigma: Iterable[float] | None = None,
) -> dict[str, float]:
    """Fit y=slope*x+intercept, using uncertainties when they are usable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma_array = np.asarray(sigma, dtype=float)
        valid &= np.isfinite(sigma_array)
    else:
        sigma_array = np.ones_like(x)
    x, y, sigma_array = x[valid], y[valid], sigma_array[valid]
    if x.size < 2:
        raise ValueError("At least two finite points are required for a line fit.")

    weighted = np.count_nonzero(sigma_array > 0.0) == sigma_array.size
    weights = 1.0 / sigma_array**2 if weighted else np.ones_like(x)
    design = np.column_stack((x, np.ones_like(x)))
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(normal)
    slope, intercept = covariance @ (design.T @ (weights * y))
    residual = y - (slope * x + intercept)
    if not weighted and x.size > 2:
        covariance *= np.sum(residual**2) / (x.size - 2)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_stderr": float(np.sqrt(max(0.0, covariance[0, 0]))),
        "intercept_stderr": float(np.sqrt(max(0.0, covariance[1, 1]))),
    }


def _load_free_energy_samples(
    paths: Sequence[Path],
) -> tuple[pd.DataFrame, dict[tuple[int, int], np.ndarray]]:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {
            "N",
            "p",
            "circ_type",
            "init_state",
            "free_energy_density_tilde",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}.")
        frame = frame.copy()
        frame["source"] = str(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for column in (
        "N",
        "p",
        "circ_type",
        "init_state",
        "free_energy_density_tilde",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "N",
            "p",
            "circ_type",
            "init_state",
            "free_energy_density_tilde",
        ]
    )
    data["N"] = data["N"].astype(int)
    data["init_state"] = data["init_state"].astype(int)

    samples = {
        (int(size), int(initial)): group[
            "free_energy_density_tilde"
        ].to_numpy(dtype=float)
        for (size, initial), group in data.groupby(["N", "init_state"])
    }
    return data, samples


def free_energy_ceff(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    alpha: float,
    alpha_stderr: float = 0.0,
    fit_sizes: Sequence[int] | None = None,
    l_min_values: Sequence[int] | None = None,
    require_both_initial_states: bool = True,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 24_680,
    cmap: str = "Blues",
    figsize: tuple[float, float] = (11.5, 5.3),
    show: bool = True,
) -> dict[str, Any]:
    """Recreate Zabalo et al. Fig. 1(b) and extract ``c_eff``.

    The input is the ``*_samples.csv`` output from ``free_energy.exe`` for
    several system sizes and both ``init_state=0`` and ``init_state=1``.  The
    simulator column ``free_energy_density_tilde`` is
    ``F_production/(L*t_production)`` as obtained from the production-window
    slope.  As in the paper, the plotted and fitted quantity is
    ``f_tilde(L)=F/(L*t)``; ``alpha`` is applied when converting its asymptotic
    slope to ``c_eff``.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")
    if not np.isfinite(alpha_stderr) or alpha_stderr < 0.0:
        raise ValueError("alpha_stderr must be finite and non-negative.")

    paths = _resolve_files(files, file_glob)
    data, samples = _load_free_energy_samples(paths)
    if data["circ_type"].nunique() != 1:
        raise ValueError("All files must use the same circ_type.")
    probabilities = data["p"].to_numpy(dtype=float)
    if np.ptp(probabilities) > 1.0e-12 * max(1.0, np.max(np.abs(probabilities))):
        raise ValueError("All files must use the same measurement rate p.")

    available_sizes = sorted(data["N"].unique().astype(int))
    if fit_sizes is not None:
        requested = {int(size) for size in fit_sizes}
        available_sizes = [size for size in available_sizes if size in requested]
    if len(available_sizes) < 3:
        raise ValueError("At least three system sizes are required.")

    summary_rows = []
    grouped_samples: dict[int, dict[int, np.ndarray]] = {}
    for size in available_sizes:
        grouped_samples[size] = {}
        for initial in (0, 1):
            values = samples.get((size, initial), np.array([], dtype=float))
            if values.size:
                grouped_samples[size][initial] = values
        if require_both_initial_states and set(grouped_samples[size]) != {0, 1}:
            raise ValueError(
                f"L={size} needs both init_state=0 and init_state=1 files."
            )
        if not grouped_samples[size]:
            continue

        state_means = {
            initial: float(np.mean(values))
            for initial, values in grouped_samples[size].items()
        }
        state_errors = {
            initial: (
                float(np.std(values, ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else 0.0
            )
            for initial, values in grouped_samples[size].items()
        }
        initial_states = sorted(state_means)
        combined_mean = float(np.mean([state_means[key] for key in initial_states]))
        combined_stderr = float(
            np.sqrt(sum(state_errors[key] ** 2 for key in initial_states))
            / len(initial_states)
        )
        summary_rows.append(
            {
                "L": size,
                "f_tilde": combined_mean,
                "f_tilde_stderr": combined_stderr,
                "f": combined_mean / alpha,
                "f_stderr": combined_stderr / alpha,
                "product_count": len(grouped_samples[size].get(0, [])),
                "haar_count": len(grouped_samples[size].get(1, [])),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("L").reset_index(drop=True)
    sizes = summary["L"].to_numpy(dtype=int)

    eligible_lmins = [
        int(size) for size in sizes if np.count_nonzero(sizes >= size) >= 3
    ]
    if l_min_values is None:
        chosen_lmins = eligible_lmins
    else:
        chosen_lmins = [int(value) for value in l_min_values]
        invalid = set(chosen_lmins).difference(eligible_lmins)
        if invalid:
            raise ValueError(
                f"Each L_min must leave at least three sizes; invalid: {sorted(invalid)}."
            )
    if not chosen_lmins:
        raise ValueError("No valid L_min values remain.")

    def first_stage(means: np.ndarray) -> list[dict[str, float]]:
        fitted = []
        errors = summary["f_tilde_stderr"].to_numpy(dtype=float)
        for minimum in chosen_lmins:
            mask = sizes >= minimum
            fit = _free_energy_linear_fit(
                1.0 / sizes[mask].astype(float) ** 2,
                means[mask],
                errors[mask],
            )
            fitted.append({"L_min": minimum, **fit})
        return fitted

    def second_stage(fitted: Sequence[Mapping[str, float]]) -> dict[str, float]:
        if len(fitted) == 1:
            only = fitted[0]
            return {
                "slope": np.nan,
                "intercept": float(only["slope"]),
                "slope_stderr": np.nan,
                "intercept_stderr": float(only["slope_stderr"]),
            }
        return _free_energy_linear_fit(
            [1.0 / float(row["L_min"]) ** 2 for row in fitted],
            [float(row["slope"]) for row in fitted],
            [float(row["slope_stderr"]) for row in fitted],
        )

    central_means = summary["f_tilde"].to_numpy(dtype=float)
    fits = first_stage(central_means)
    correction_fit = second_stage(fits)
    m_tilde_infinite = correction_fit["intercept"]
    c_eff = -6.0 * m_tilde_infinite / (np.pi * alpha)
    analytic_stderr = np.sqrt(
        (6.0 * correction_fit["intercept_stderr"] / (np.pi * alpha)) ** 2
        + (c_eff * alpha_stderr / alpha) ** 2
    )

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_ceff = []
    if bootstrap_samples >= 2:
        for _ in range(bootstrap_samples):
            sampled_means = []
            for size in sizes:
                by_initial = grouped_samples[int(size)]
                means = [
                    np.mean(rng.choice(values, size=values.size, replace=True))
                    for values in by_initial.values()
                ]
                sampled_means.append(float(np.mean(means)))
            sampled_alpha = (
                rng.normal(alpha, alpha_stderr)
                if alpha_stderr > 0.0
                else alpha
            )
            if sampled_alpha <= 0.0:
                continue
            try:
                sampled_fits = first_stage(np.asarray(sampled_means, dtype=float))
                sampled_correction = second_stage(sampled_fits)
                value = (
                    -6.0 * sampled_correction["intercept"]
                    / (np.pi * sampled_alpha)
                )
                if np.isfinite(value):
                    bootstrap_ceff.append(value)
            except (ValueError, np.linalg.LinAlgError):
                continue

    if len(bootstrap_ceff) >= 10:
        bootstrap_array = np.asarray(bootstrap_ceff, dtype=float)
        c_eff_stderr = float(np.std(bootstrap_array, ddof=1))
        c_eff_p16, c_eff_p84 = np.percentile(bootstrap_array, [16.0, 84.0])
    else:
        c_eff_stderr = float(analytic_stderr)
        c_eff_p16 = c_eff - c_eff_stderr
        c_eff_p84 = c_eff + c_eff_stderr
        if bootstrap_samples >= 2:
            warnings.warn("Too few successful c_eff bootstrap fits; using fit errors.")

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.91)
    colors = plt.get_cmap(cmap)(
        np.linspace(0.35, 0.90, max(1, len(fits)))
    )
    x_all = 1.0 / sizes.astype(float) ** 2
    ax.errorbar(
        x_all,
        summary["f_tilde"],
        yerr=summary["f_tilde_stderr"],
        linestyle="none",
        marker="o",
        markersize=3,
        markeredgewidth=0.7,
        color="black",
        capsize=1.5,
        elinewidth=0.75,
        capthick=0.75,
        label="product/Haar average",
        zorder=5,
    )
    for color, fit in zip(colors, fits):
        maximum_x = 1.0 / float(fit["L_min"]) ** 2
        x_line = np.linspace(0.0, maximum_x * 1.03, 200)
        ax.plot(
            x_line,
            fit["slope"] * x_line + fit["intercept"],
            color=color,
            label=rf"$L_{{\min}}={int(fit['L_min'])}$",
        )
    tick_order = np.argsort(x_all)
    ax.set_xticks(x_all[tick_order])
    ax.set_xticklabels(
        [rf"$1/{int(size)}^2$" for size in sizes[tick_order]]
    )
    ax.set_xlabel(r"$1/L^2$")
    ax.set_ylabel(r"$f(L)$")
    # ax.set_title("Measurement-record free-energy density")
    ax.grid(alpha=0.25)
    information = ax.text(
        0.025,
        0.025,
        rf"$\alpha={alpha:.5g}$" + "\n" +
        rf"$c_{{\mathrm{{eff}}}}={c_eff:.5f}\pm{c_eff_stderr:.5f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
        zorder=10,
    )
    # Stack the legend directly above the information box in the lower-left
    # corner.  Using its rendered height prevents overlap if fonts or values
    # change.
    fig.canvas.draw()
    information_bbox = information.get_window_extent(
        renderer=fig.canvas.get_renderer()
    ).transformed(ax.transAxes.inverted())
    ax.legend(
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0.025, information_bbox.y1 + 0.018),
        borderaxespad=0.0,
    )

    # Keep the inset's top and right spines exactly aligned with those of the
    # main axes.
    inset = ax.inset_axes([0.685, 0.57, 0.315, 0.43])
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    inset.set_zorder(10)
    inset_x = np.asarray(
        [int(row["L_min"]) for row in fits],
        dtype=int,
    )
    inset_y = np.asarray([float(row["slope"]) for row in fits])
    inset_error = np.asarray([float(row["slope_stderr"]) for row in fits])
    inset.errorbar(
        inset_x,
        inset_y,
        yerr=inset_error,
        linestyle="none",
        marker="o",
        markersize=3,
        markeredgewidth=0.7,
        color="black",
        capsize=1,
        elinewidth=0.7,
        capthick=0.7,
    )
    if len(fits) >= 2:
        inset_line_x = np.linspace(
            float(np.min(inset_x)),
            float(np.max(inset_x)),
            200,
        )
        inset.plot(
            inset_line_x,
            correction_fit["slope"] / inset_line_x**2 + m_tilde_infinite,
            linestyle=":",
            color="tab:blue",
        )
    inset.set_xlabel(r"$L_{\min}$", fontsize=8)
    inset.set_ylabel(r"$m_0(L_{\min})$", fontsize=8)
    inset.set_xticks(inset_x)
    inset.set_xticklabels([str(value) for value in inset_x])
    if len(inset_x) == 1:
        inset.set_xlim(inset_x[0] - 0.5, inset_x[0] + 0.5)
    else:
        inset_padding = max(0.25, 0.04 * float(np.ptp(inset_x)))
        inset.set_xlim(
            float(np.min(inset_x)) - inset_padding,
            float(np.max(inset_x)) + inset_padding,
        )
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.2)
    _show(fig, show)

    return {
        "figure": fig,
        "axis": ax,
        "inset": inset,
        "summary": summary,
        "fits": pd.DataFrame(fits),
        "correction_fit": correction_fit,
        "alpha": alpha,
        "alpha_stderr": alpha_stderr,
        "c_eff": float(c_eff),
        "c_eff_stderr": float(c_eff_stderr),
        "c_eff_p16": float(c_eff_p16),
        "c_eff_p84": float(c_eff_p84),
        "bootstrap_successes": len(bootstrap_ceff),
        "samples": data,
    }


def free_energy_equilibration(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    t_over_l_range: tuple[float, float] = (0.0, 10.0),
    fit_t_over_l_min: float = 2.0,
    colors: tuple[Any, Any, Any] = ("tab:blue", "tab:orange", "tab:green"),
    figsize: tuple[float, float] = (11.5, 4.5),
    show: bool = True,
) -> dict[str, Any]:
    """Recreate Supplementary Fig. S1(a,b) from ``*_timeseries.csv``.

    Unlike the production-only estimator used by :func:`free_energy_ceff`,
    these curves use the cumulative measurement-record entropy from the first
    timestep.  ``init_state=0`` is the product ensemble and ``init_state=1`` is
    the Haar ensemble.  The half-chain von Neumann entropy inset is in bits.
    """
    paths = _resolve_files(files, file_glob)
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {
            "N",
            "p",
            "circ_type",
            "init_state",
            "completed_realizations",
            "t",
            "t_over_L",
            "F_mean",
            "F_stderr",
            "F_over_tL_mean",
            "F_over_tL_stderr",
            "S_half_mean",
            "S_half_stderr",
            "S_half_count",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}.")
        frame = frame.copy()
        frame["source"] = str(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    numeric = [
        "N",
        "p",
        "circ_type",
        "init_state",
        "completed_realizations",
        "t",
        "t_over_L",
        "F_mean",
        "F_stderr",
        "F_over_tL_mean",
        "F_over_tL_stderr",
        "S_half_mean",
        "S_half_stderr",
        "S_half_count",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["N", "circ_type", "init_state", "t"])
    if data["N"].nunique() != 1 or data["circ_type"].nunique() != 1:
        raise ValueError("Fig. S1 inputs must use one N and one circ_type.")
    probabilities = data["p"].dropna().to_numpy(dtype=float)
    if probabilities.size and np.ptp(probabilities) > 1.0e-12 * max(
        1.0, np.max(np.abs(probabilities))
    ):
        raise ValueError("Fig. S1 inputs must use one measurement rate p.")

    # A time-series file is a checkpoint that is rewritten after every
    # trajectory. If several checkpoints are supplied, retain the most complete
    # one for each initial-state/time pair rather than counting the same samples
    # more than once.
    data = data.sort_values("completed_realizations").drop_duplicates(
        ["init_state", "t"], keep="last"
    )
    initial_states = set(data["init_state"].astype(int).unique())
    if not {0, 1}.issubset(initial_states):
        raise ValueError("Provide both init_state=0 and init_state=1 time-series files.")

    product = data.loc[data["init_state"] == 0].copy()
    haar = data.loc[data["init_state"] == 1].copy()
    columns = [
        "t",
        "t_over_L",
        "F_mean",
        "F_stderr",
        "F_over_tL_mean",
        "F_over_tL_stderr",
        "S_half_mean",
        "S_half_stderr",
        "S_half_count",
    ]
    aligned = product[columns].merge(
        haar[columns], on=["t", "t_over_L"], suffixes=("_product", "_haar")
    ).sort_values("t")
    if aligned.empty:
        raise ValueError("The product and Haar time grids do not overlap.")

    aligned["F_over_tL_mean_average"] = 0.5 * (
        aligned["F_over_tL_mean_product"] + aligned["F_over_tL_mean_haar"]
    )
    aligned["F_over_tL_stderr_average"] = 0.5 * np.sqrt(
        aligned["F_over_tL_stderr_product"] ** 2
        + aligned["F_over_tL_stderr_haar"] ** 2
    )
    aligned["S_half_mean_average"] = 0.5 * (
        aligned["S_half_mean_product"] + aligned["S_half_mean_haar"]
    )
    aligned["S_half_stderr_average"] = 0.5 * np.sqrt(
        aligned["S_half_stderr_product"] ** 2
        + aligned["S_half_stderr_haar"] ** 2
    )
    size = int(data["N"].iloc[0])
    aligned["delta_F_over_L"] = (
        aligned["F_mean_haar"] - aligned["F_mean_product"]
    ) / size
    aligned["delta_F_over_L_stderr"] = np.sqrt(
        aligned["F_stderr_haar"] ** 2 + aligned["F_stderr_product"] ** 2
    ) / size

    fig, (ax_density, ax_delta) = plt.subplots(1, 2, figsize=figsize)
    curve_specs = (
        ("haar", "random Haar state", colors[0]),
        ("product", "random product state", colors[1]),
        ("average", "average", colors[2]),
    )
    boundary_fits = {}
    for key, label, color in curve_specs:
        y = aligned[f"F_over_tL_mean_{key}"].to_numpy(dtype=float)
        error = aligned[f"F_over_tL_stderr_{key}"].to_numpy(dtype=float)
        x = aligned["t_over_L"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax_density.plot(x[mask], y[mask], color=color, label=label)
        ax_density.fill_between(
            x[mask], y[mask] - error[mask], y[mask] + error[mask],
            color=color, alpha=0.16,
        )

        fit_mask = mask & (x >= fit_t_over_l_min) & (x <= t_over_l_range[1])
        if np.count_nonzero(fit_mask) >= 2:
            fit = _free_energy_linear_fit(
                1.0 / aligned.loc[fit_mask, "t"].to_numpy(dtype=float),
                y[fit_mask],
                error[fit_mask],
            )
            boundary_fits[key] = fit
            fit_x = x[fit_mask]
            fit_t = aligned.loc[fit_mask, "t"].to_numpy(dtype=float)
            ax_density.plot(
                fit_x,
                fit["intercept"] + fit["slope"] / fit_t,
                color=color,
                linestyle=":",
                linewidth=1.1,
            )

    ax_density.set_xlim(*t_over_l_range)
    ax_density.set_xlabel(r"$t/L$")
    ax_density.set_ylabel(r"$F(t)/(tL)$")
    ax_density.set_title("(a) Record-entropy density")
    ax_density.grid(alpha=0.25)
    ax_density.legend(fontsize=8)

    # The record-density curves live in the upper half of this panel after
    # equilibration; the lower-right corner keeps the entropy inset clear.
    inset = ax_density.inset_axes([0.50, 0.06, 0.47, 0.38])
    for key, label, color in curve_specs:
        y = aligned[f"S_half_mean_{key}"].to_numpy(dtype=float)
        x = aligned["t_over_L"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            inset.plot(x[mask], y[mask], color=color, label=label)
    inset.set_xlabel(r"$t/L$", fontsize=8)
    inset.set_ylabel(r"$S_1(t)$ [bits]", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.2)

    x = aligned["t_over_L"].to_numpy(dtype=float)
    delta = aligned["delta_F_over_L"].to_numpy(dtype=float)
    delta_error = aligned["delta_F_over_L_stderr"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(delta)
    ax_delta.plot(x[mask], delta[mask], color="tab:purple")
    ax_delta.fill_between(
        x[mask],
        delta[mask] - delta_error[mask],
        delta[mask] + delta_error[mask],
        color="tab:purple",
        alpha=0.18,
    )
    ax_delta.set_xlim(*t_over_l_range)
    ax_delta.set_xlabel(r"$t/L$")
    ax_delta.set_ylabel(r"$\Delta F(t)/L$")
    ax_delta.set_title("(b) Initial-state boundary contribution")
    ax_delta.grid(alpha=0.25)
    fig.tight_layout()
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_density, ax_delta),
        "entropy_inset": inset,
        "data": aligned,
        "boundary_fits": boundary_fits,
        "N": size,
        "p": float(probabilities[0]) if probabilities.size else np.nan,
    }

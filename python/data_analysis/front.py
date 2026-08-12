"""Mode-5 information fronts: how fast classical and quantum signals spread.

``mipt_probed.exe`` mode 5 injects one Bell-entangled reference at a random site
after equilibration and then, at every elapsed timestep, measures what has
reached the receiver blocks ``B_w(x)`` sitting at arc distance ``x``. This module
turns that ``(width, x, tau)`` grid into the two things the protocol asks for:

1. side-by-side heatmaps of a classical and a quantum signal over ``(x, tau)``,
   with the same equal-signal contours drawn on both; and
2. arrival time against distance, ``tau_q(x) = b + lambda x``, whose fitted slope
   gives the front velocity ``v = 1 / lambda``.

The interesting quantity is the *gap* between the two fronts. A band that grows
linearly with ``x`` means the classical front genuinely outruns entanglement
(``lambda_classical < lambda_entanglement``); a band of constant width means the
two velocities agree and entanglement merely takes a fixed time to form once the
classical correlation arrives. The fourth panel fits exactly that slope, so the
two hypotheses are separated by a number with an error bar rather than by eye.

Thresholds are stated as a *fraction of the injected signal* by default. At the
moment of injection the reference is maximally entangled with its partner site,
so every metric starts from a known ceiling -- ``log 2`` for the Holevo
informations, ``1/2`` for the negativity -- and normalizing by it puts the
classical and quantum channels on one comparable scale. That is what makes a
single threshold ``q`` meaningful for both, and therefore what makes the two
slopes comparable at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from .fitting import _weighted_linear_fit
from .loading import _find_column, _resolve_files, resolve_metadata
from .plotting import _annotate_axes, _mi_unit_spec, _show

LOG2 = float(np.log(2.0))

# Categorical slots 1 and 2 of the validated default palette, in fixed order:
# the classical front always takes slot 1 and the entanglement front slot 2, so
# a figure that drops one series never repaints the other.
_CLASSICAL_COLOR = "#2a78d6"
_QUANTUM_COLOR = "#eb6834"

# Every mode-5 observable, with the value it takes at the instant of injection.
#
# ``source`` is the maximum the metric can reach: a maximally entangled
# reference carries one full bit in each Pauli basis, has negativity 1/2, and
# has mutual information 2 log 2. ``entropy_like`` marks the quantities that
# carry entropy units and therefore respond to ``mi_units``.
_FRONT_METRICS: dict[str, dict[str, Any]] = {
    "C_X": {
        "mean": ("C_X_mean",), "stderr": ("C_X_stderr",),
        "label": r"$C_X$", "source": LOG2, "entropy_like": True, "kind": "classical",
    },
    "C_Y": {
        "mean": ("C_Y_mean",), "stderr": ("C_Y_stderr",),
        "label": r"$C_Y$", "source": LOG2, "entropy_like": True, "kind": "classical",
    },
    "C_Z": {
        "mean": ("C_Z_mean",), "stderr": ("C_Z_stderr",),
        "label": r"$C_Z$", "source": LOG2, "entropy_like": True, "kind": "classical",
    },
    "C_mean": {
        "mean": ("C_mean_mean",), "stderr": ("C_mean_stderr",),
        "label": r"$\overline{C}$", "source": LOG2, "entropy_like": True,
        "kind": "classical",
    },
    "C_max": {
        "mean": ("C_max_mean",), "stderr": ("C_max_stderr",),
        "label": r"$C_{\max}$", "source": LOG2, "entropy_like": True,
        "kind": "classical",
    },
    "I": {
        "mean": ("I_mean",), "stderr": ("I_stderr",),
        "label": r"$I(R{:}B)$", "source": 2.0 * LOG2, "entropy_like": True,
        "kind": "total",
    },
    "negativity": {
        "mean": ("negativity_mean",), "stderr": ("negativity_stderr",),
        "label": r"$\mathcal{N}$", "source": 0.5, "entropy_like": False,
        "kind": "quantum",
    },
    "log_negativity": {
        "mean": ("log_negativity_mean",), "stderr": ("log_negativity_stderr",),
        "label": r"$E_{\mathcal{N}}$", "source": LOG2, "entropy_like": True,
        "kind": "quantum",
    },
    "coherent_information": {
        "mean": ("coherent_information_mean",), "stderr": ("coherent_information_stderr",),
        "label": r"$I_c(R\rangle B)$", "source": LOG2, "entropy_like": True,
        "kind": "quantum",
    },
    # Far more sensitive than the mean negativity -- it fires as soon as any
    # trajectory has entanglement at all -- but it is NOT a safe front
    # detector. mipt_probed counts a trajectory as positive above 1e-12, while
    # an fp32 state vector produces round-off negativities around 1e-5, so
    # outside the light cone this saturates at a few percent rather than zero
    # and a low threshold will fit a superluminal "front". Use it to see where
    # entanglement is rare but present, and read velocities off the mean.
    "positive_fraction": {
        "mean": ("negativity_positive_fraction",), "stderr": (),
        "label": r"$P(\mathcal{N}>0)$", "source": 1.0, "entropy_like": False,
        "kind": "quantum",
    },
}


def _metric_spec(name: str) -> dict[str, Any]:
    if name not in _FRONT_METRICS:
        raise ValueError(
            f"Unknown mode-5 metric {name!r}. Choose one of {sorted(_FRONT_METRICS)}."
        )
    return _FRONT_METRICS[name]


def _load_front_file(
    path: Path,
    *,
    width: int,
    metrics: Sequence[str],
    mi_scale: float,
    l_by_file: Mapping[str, int] | None,
) -> dict[str, Any]:
    """Read one mode-5 CSV into rectangular ``(tau, x)`` grids per metric."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path}: mode-5 CSV is empty.")

    metadata = resolve_metadata(df, path, l_by_file=l_by_file)
    width_col = _find_column(df, ["width", "block_width", "w"])
    x_col = _find_column(df, ["x", "distance", "r"])
    tau_col = _find_column(df, ["tau", "elapsed_time", "readout_time"])
    t_eq_col = _find_column(df, ["t_eq", "teq"], required=False)

    selected = df.loc[pd.to_numeric(df[width_col], errors="coerce") == width]
    if selected.empty:
        available = sorted(pd.to_numeric(df[width_col], errors="coerce").dropna().unique())
        raise ValueError(
            f"{path}: no rows with receiver width {width}. Available widths: "
            f"{[int(value) for value in available]}."
        )

    columns = {
        "x": pd.to_numeric(selected[x_col], errors="coerce"),
        "tau": pd.to_numeric(selected[tau_col], errors="coerce"),
    }
    for name in metrics:
        spec = _metric_spec(name)
        mean_col = _find_column(selected, spec["mean"])
        scale = mi_scale if spec["entropy_like"] else 1.0
        columns[f"{name}_mean"] = pd.to_numeric(selected[mean_col], errors="coerce") * scale
        stderr_col = _find_column(selected, spec["stderr"], required=False) if spec["stderr"] else None
        columns[f"{name}_stderr"] = (
            pd.to_numeric(selected[stderr_col], errors="coerce") * scale
            if stderr_col is not None
            else pd.Series(np.nan, index=selected.index)
        )

    clean = pd.DataFrame(columns).replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "tau"])
    if clean.empty:
        raise ValueError(f"{path}: no usable rows at receiver width {width}.")
    clean = clean.sort_values(["x", "tau"]).drop_duplicates(["x", "tau"], keep="last")

    xs = np.sort(clean["x"].unique()).astype(int)
    taus = np.sort(clean["tau"].unique()).astype(int)
    grids: dict[str, dict[str, np.ndarray]] = {}
    for name in metrics:
        pivot_mean = clean.pivot(index="tau", columns="x", values=f"{name}_mean")
        pivot_stderr = clean.pivot(index="tau", columns="x", values=f"{name}_stderr")
        grids[name] = {
            "mean": pivot_mean.reindex(index=taus, columns=xs).to_numpy(dtype=float),
            "stderr": pivot_stderr.reindex(index=taus, columns=xs).to_numpy(dtype=float),
        }

    t_eq = None
    if t_eq_col is not None:
        values = pd.to_numeric(selected[t_eq_col], errors="coerce").dropna().unique()
        if len(values) == 1:
            t_eq = int(round(float(values[0])))

    return {
        "path": path,
        "L": int(metadata["L"]),
        "p": float(metadata["p"]),
        "circuit": metadata["circuit"],
        "realizations": metadata["realizations"],
        "t_eq": t_eq,
        "width": int(width),
        "x": xs,
        "tau": taus,
        "grids": grids,
    }


def _threshold_level(spec: Mapping[str, Any], threshold: float, normalize: bool, mi_scale: float) -> float:
    """Convert a fractional threshold into the metric's own units."""
    if not normalize:
        return float(threshold)
    scale = mi_scale if spec["entropy_like"] else 1.0
    return float(threshold) * float(spec["source"]) * scale


def _arrival_times(
    taus: np.ndarray,
    signal: np.ndarray,
    stderr: np.ndarray,
    level: float,
    *,
    persistence: int,
) -> tuple[float, float]:
    """First time the signal crosses ``level``, linearly interpolated.

    Returns ``(tau, sigma_tau)``, both NaN if the signal never crosses.

    The crossing is required to hold for ``persistence`` consecutive readout
    times. Statistical noise at large ``x`` produces isolated upcrossings well
    ahead of the real front, and because the fit only ever sees the *first*
    crossing, one such point biases the velocity directly rather than averaging
    away. The uncertainty is the signal error transported through the local
    slope, ``sigma_tau = sigma_S / |dS/dtau|``, which is what makes the
    subsequent fit weightable.
    """
    # Cells blanked by the wrap mask are NaN, and NaN >= level is False, so a
    # masked readout simply never counts as an arrival.
    above = signal >= level
    count = above.size
    for index in range(count):
        if not above[index]:
            continue
        window = min(persistence, count - index)
        if not np.all(above[index : index + window]):
            continue
        if index == 0:
            # Already above at the first readout time: no bracket to
            # interpolate in, so the crossing is not resolved beyond "<= tau_0".
            return float(taus[0]), float(taus[1] - taus[0]) if count > 1 else 0.0

        lower_tau, upper_tau = float(taus[index - 1]), float(taus[index])
        lower_value, upper_value = float(signal[index - 1]), float(signal[index])
        rise = upper_value - lower_value
        if not np.isfinite(rise) or rise <= 0.0:
            return upper_tau, float(upper_tau - lower_tau)

        span = upper_tau - lower_tau
        crossing = lower_tau + span * (level - lower_value) / rise
        slope = rise / span
        errors = np.array([stderr[index - 1], stderr[index]], dtype=float)
        errors = errors[np.isfinite(errors)]
        sigma_signal = float(np.sqrt(np.sum(errors**2))) if errors.size else np.nan
        sigma = sigma_signal / slope if np.isfinite(sigma_signal) and slope > 0.0 else np.nan
        # Never claim to resolve the crossing better than the readout spacing.
        floor = 0.25 * span
        sigma = floor if not np.isfinite(sigma) else max(sigma, floor)
        return float(crossing), float(sigma)
    return np.nan, np.nan


def _front_curve(
    dataset: Mapping[str, Any],
    metric: str,
    level: float,
    *,
    x_min: int,
    x_max: int | None,
    persistence: int,
) -> dict[str, np.ndarray]:
    """Arrival time against distance for one metric and one threshold."""
    xs = dataset["x"]
    taus = dataset["tau"]
    grid = dataset["grids"][metric]
    keep = xs >= x_min
    if x_max is not None:
        keep &= xs <= x_max

    distances, arrivals, errors = [], [], []
    for column, x in enumerate(xs):
        if not keep[column]:
            continue
        arrival, sigma = _arrival_times(
            taus, grid["mean"][:, column], grid["stderr"][:, column], level,
            persistence=persistence,
        )
        distances.append(float(x))
        arrivals.append(arrival)
        errors.append(sigma)
    return {
        "x": np.asarray(distances, dtype=float),
        "tau": np.asarray(arrivals, dtype=float),
        "tau_stderr": np.asarray(errors, dtype=float),
    }


def _fit_front(curve: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Fit ``tau = b + lambda x`` and invert the slope into a velocity."""
    valid = np.isfinite(curve["tau"]) & np.isfinite(curve["x"])
    x = curve["x"][valid]
    tau = curve["tau"][valid]
    sigma = curve["tau_stderr"][valid]
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 1.0)
    if x.size < 3:
        return {
            "slope": np.nan, "slope_stderr": np.nan, "intercept": np.nan,
            "intercept_stderr": np.nan, "reduced_chi2": np.nan,
            "velocity": np.nan, "velocity_stderr": np.nan, "points": int(x.size),
        }

    fit = _weighted_linear_fit(x, tau, sigma)
    slope = fit["slope"]
    velocity = 1.0 / slope if slope > 0.0 else np.nan
    velocity_stderr = fit["slope_stderr"] / slope**2 if slope > 0.0 else np.nan
    return {**fit, "velocity": velocity, "velocity_stderr": velocity_stderr, "points": int(x.size)}


def _mask_wrapped(dataset: dict[str, Any]) -> int:
    """Blank the readouts a signal could only have reached by going the long way.

    Both fronts leave ``x_0`` at no more than one site per timestep, so the one
    travelling away from a block at distance ``x`` needs ``L - x`` timesteps to
    come round the ring and reach it from behind. From then on the block is
    seeing both fronts and ``x`` no longer labels which. The cut is therefore
    per distance, not global: at ``tau = L/2`` only the antipodal blocks are
    compromised, while ``x = 1`` stays clean almost twice as long. Truncating
    the whole ``tau`` axis at ``L/2`` instead -- the naive reading of "retain
    only times before the fronts wrap" -- discards most of the usable data at
    exactly the small distances the fit relies on.

    Returns the number of grid cells blanked.
    """
    size = dataset["L"]
    taus = dataset["tau"][:, None]
    xs = dataset["x"][None, :]
    wrapped = taus >= (size - xs)
    for grid in dataset["grids"].values():
        grid["mean"] = np.where(wrapped, np.nan, grid["mean"])
        grid["stderr"] = np.where(wrapped, np.nan, grid["stderr"])
    return int(wrapped.sum())


def front_velocity(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    width: int = 1,
    classical_metric: str = "C_mean",
    quantum_metric: str = "negativity",
    thresholds: Sequence[float] = (0.01, 0.02),
    normalize: bool = True,
    x_min: int = 1,
    x_max: int | None = None,
    tau_max: int | None = None,
    drop_wrapped: bool = True,
    persistence: int = 2,
    heatmap_file: str | Path | None = None,
    mi_units: str = "bits",
    cmap: str = "magma",
    l_by_file: Mapping[str, int] | None = None,
    figsize: tuple[float, float] = (13.0, 9.5),
    dpi: int = 130,
    show: bool = True,
) -> dict[str, Any]:
    """Extract classical and entanglement front velocities from mode-5 output.

    Parameters
    ----------
    files, file_glob:
        One mode-5 CSV per system size. The heatmaps come from a single file
        (``heatmap_file``, or the largest ``L``); the front fits use all of them.
    width:
        Receiver-block width ``w`` to analyse. Wider blocks pick the signal up
        earlier, which shifts the intercept far more than the slope.
    classical_metric, quantum_metric:
        Which observable stands for each channel. ``C_Z`` is the
        superselection-safe choice for fermionic circuits; ``C_mean`` averages
        the three Pauli bases and is the better default for qubit circuits.
    thresholds:
        Signal levels ``q``. Fractions of the injected signal when ``normalize``
        is true (the default), otherwise absolute values in the metric's units.
    x_min, x_max:
        Distance window for the fits. ``x_min=1`` excludes the block that
        contains the injection site itself.
    tau_max, drop_wrapped:
        Readout times are cut at ``tau_max``. ``drop_wrapped`` additionally
        blanks each distance once a front could have reached it the long way
        round, at ``tau >= L - x``.
    persistence:
        Consecutive readout times a crossing must hold for to count as an
        arrival. See :func:`_arrival_times`.

    Returns a dictionary with the figure, its axes, the loaded grids, the
    per-(metric, threshold, L) front curves and fits, and the fitted width of
    the classical-only band.
    """
    paths = _resolve_files(files, file_glob)
    _, mi_scale = _mi_unit_spec(mi_units)
    metrics = [classical_metric, quantum_metric]
    for name in metrics:
        _metric_spec(name)
    if classical_metric == quantum_metric:
        raise ValueError("classical_metric and quantum_metric must differ.")
    thresholds = [float(value) for value in thresholds]
    if not thresholds:
        raise ValueError("At least one signal threshold is required.")
    if normalize and any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Fractional thresholds must lie strictly between 0 and 1.")
    if persistence < 1:
        raise ValueError("persistence must be at least 1.")

    datasets = [
        _load_front_file(path, width=width, metrics=metrics, mi_scale=mi_scale, l_by_file=l_by_file)
        for path in paths
    ]
    datasets.sort(key=lambda dataset: dataset["L"])

    # Trim the readout window before anything is fitted, so that the heatmaps
    # and the arrival times always describe the same data.
    for dataset in datasets:
        if tau_max is not None:
            keep = dataset["tau"] <= tau_max
            if not keep.any():
                raise ValueError(
                    f"{dataset['path']}: no readout times survive tau <= {tau_max}."
                )
            dataset["tau"] = dataset["tau"][keep]
            for grid in dataset["grids"].values():
                grid["mean"] = grid["mean"][keep, :]
                grid["stderr"] = grid["stderr"][keep, :]
        if drop_wrapped:
            _mask_wrapped(dataset)

    levels = {
        name: [_threshold_level(_metric_spec(name), value, normalize, mi_scale) for value in thresholds]
        for name in metrics
    }

    # A threshold below the sampling noise is not a front, it is a coin flip:
    # the "first crossing" would then be set by whichever bin fluctuated up.
    #
    # Cells outside the light cone are identically zero in every trajectory, so
    # their standard error is zero up to round-off. Including them would drag
    # the median down to ~1e-18 for any metric with a hard floor -- the
    # negativities, over most of the grid -- and the check would never fire.
    # Only cells that genuinely vary across trajectories say anything about the
    # sampling noise, so anything at round-off scale is dropped.
    noise_floor: dict[str, float] = {}
    for name in metrics:
        spec = _metric_spec(name)
        source = spec["source"] * (mi_scale if spec["entropy_like"] else 1.0)
        errors = np.concatenate([dataset["grids"][name]["stderr"].ravel() for dataset in datasets])
        errors = errors[np.isfinite(errors) & (errors > 1.0e-9 * source)]
        noise_floor[name] = float(np.median(errors)) if errors.size else np.nan

    curves: dict[tuple[str, float, int], dict[str, np.ndarray]] = {}
    fits: dict[tuple[str, float, int], dict[str, float]] = {}
    for dataset in datasets:
        for name in metrics:
            floor = noise_floor[name]
            for threshold, level in zip(thresholds, levels[name]):
                key = (name, threshold, dataset["L"])
                curve = _front_curve(
                    dataset, name, level, x_min=x_min, x_max=x_max, persistence=persistence
                )
                curves[key] = curve
                fits[key] = _fit_front(curve)
                resolved = int(np.isfinite(curve["tau"]).sum())
                if np.isfinite(floor) and level < 2.0 * floor:
                    warnings.warn(
                        f"L={dataset['L']}: the q={threshold:g} level for {name} "
                        f"({level:.3g}) is below twice the sampling noise ({floor:.3g}); "
                        "the arrival times are dominated by fluctuations. Raise the "
                        "threshold or the realization count."
                    )
                elif resolved < 3:
                    warnings.warn(
                        f"L={dataset['L']}: {name} reaches q={threshold:g} at only "
                        f"{resolved} distance(s), too few to fit. Lower the threshold, "
                        "extend tau_max, or widen the receiver block."
                    )

    # The classical-only band: how much later the entanglement front arrives.
    bands: dict[float, dict[str, np.ndarray]] = {}
    band_fits: dict[float, dict[str, float]] = {}
    for threshold in thresholds:
        distances, gaps, errors = [], [], []
        for dataset in datasets:
            classical = curves[(classical_metric, threshold, dataset["L"])]
            quantum = curves[(quantum_metric, threshold, dataset["L"])]
            shared = np.intersect1d(classical["x"], quantum["x"])
            for x in shared:
                c_index = int(np.flatnonzero(classical["x"] == x)[0])
                q_index = int(np.flatnonzero(quantum["x"] == x)[0])
                gap = quantum["tau"][q_index] - classical["tau"][c_index]
                if not np.isfinite(gap):
                    continue
                sigma = float(
                    np.sqrt(
                        np.nansum(
                            [classical["tau_stderr"][c_index] ** 2, quantum["tau_stderr"][q_index] ** 2]
                        )
                    )
                )
                distances.append(float(x))
                gaps.append(float(gap))
                errors.append(sigma if sigma > 0.0 else 1.0)
        band = {
            "x": np.asarray(distances, dtype=float),
            "delta_tau": np.asarray(gaps, dtype=float),
            "delta_tau_stderr": np.asarray(errors, dtype=float),
        }
        bands[threshold] = band
        band_fits[threshold] = _fit_front(
            {"x": band["x"], "tau": band["delta_tau"], "tau_stderr": band["delta_tau_stderr"]}
        )

    # ----------------------------------------------------------------- figure
    heatmap_dataset = datasets[-1]
    if heatmap_file is not None:
        wanted = Path(heatmap_file)
        matches = [dataset for dataset in datasets if dataset["path"] == wanted]
        if not matches:
            raise ValueError(f"{heatmap_file} is not among the loaded files.")
        heatmap_dataset = matches[0]

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    (ax_classical, ax_quantum), (ax_arrival, ax_band) = axes

    # Both panels share one normalization so the two channels can be compared
    # directly; a log scale keeps the decaying tail readable, since the signal
    # falls by orders of magnitude across the grid.
    #
    # The floor is the sampling noise, not the smallest number in the grid.
    # Ranging down to the smallest value spends most of the ramp resolving
    # differences between cells that are all statistically zero, which reads as
    # structure that is not there.
    floors, ceilings = [], []
    for name in metrics:
        spec = _metric_spec(name)
        source = spec["source"] * (mi_scale if spec["entropy_like"] else 1.0)
        scaled = heatmap_dataset["grids"][name]["mean"] / source
        positive = scaled[np.isfinite(scaled) & (scaled > 0.0)]
        if positive.size:
            ceilings.append(float(positive.max()))
        floor = noise_floor[name]
        if np.isfinite(floor) and floor > 0.0:
            floors.append(floor / source)
    vmin = max(min(floors) if floors else 1e-4, 1e-6)
    vmax = max(ceilings) if ceilings else 1.0
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0))

    mesh = None
    for ax, name, title in (
        (ax_classical, classical_metric, "Classical information"),
        (ax_quantum, quantum_metric, "Entanglement"),
    ):
        spec = _metric_spec(name)
        source = spec["source"] * (mi_scale if spec["entropy_like"] else 1.0)
        scaled = heatmap_dataset["grids"][name]["mean"] / source
        mesh = ax.pcolormesh(
            heatmap_dataset["x"], heatmap_dataset["tau"], scaled,
            shading="auto", cmap=cmap, norm=norm,
        )
        contour_levels = sorted(value for value in thresholds if vmin < value < vmax)
        if contour_levels and np.isfinite(scaled).any():
            contours = ax.contour(
                heatmap_dataset["x"], heatmap_dataset["tau"], scaled,
                levels=contour_levels, colors="white", linewidths=1.2,
            )
            ax.clabel(contours, fmt=lambda value: f"q={value:g}", fontsize=7.5, inline=True)
        # One brickwork layer per timestep couples nearest neighbours only, so
        # the *unitary* cone is tau = x, and at p=0 everything outside it is
        # zero to machine precision. It is not a strict bound once measurements
        # are on: projecting a site inside the cone leaves a conditional state
        # whose R-B correlations are not confined to it, and a faint tail of
        # order 1e-4 of the source shows up one or two sites early. That tail
        # sits far below any sensible threshold, so the line is still the right
        # sanity check -- a *fitted* front that crosses it is threshold noise,
        # not signal.
        cone = np.array([heatmap_dataset["x"].min(), heatmap_dataset["x"].max()], dtype=float)
        ax.plot(cone, cone, color="white", linewidth=1.0, linestyle="--", alpha=0.65)
        ax.set_xlabel(r"Distance from injection, $x$ (sites)")
        ax.set_ylabel(r"Elapsed time, $\tau$ (timesteps)")
        ax.set_title(f"{title}: {spec['label']} / {spec['label']}$(0,0)$")

    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=[ax_classical, ax_quantum], location="right", pad=0.015)
        colorbar.set_label("Signal, as a fraction of the injected value")

    markers = ["o", "s", "^", "D", "v", "P"]
    alphas = np.linspace(1.0, 0.45, max(len(thresholds), 1))
    for name, color in ((classical_metric, _CLASSICAL_COLOR), (quantum_metric, _QUANTUM_COLOR)):
        for threshold_index, threshold in enumerate(thresholds):
            for size_index, dataset in enumerate(datasets):
                key = (name, threshold, dataset["L"])
                curve, fit = curves[key], fits[key]
                valid = np.isfinite(curve["tau"])
                if not valid.any():
                    continue
                ax_arrival.errorbar(
                    curve["x"][valid], curve["tau"][valid], yerr=curve["tau_stderr"][valid],
                    color=color, alpha=alphas[threshold_index],
                    marker=markers[size_index % len(markers)], markersize=5.0,
                    linestyle="none", capsize=2, elinewidth=1.0,
                )
                if np.isfinite(fit["slope"]):
                    line_x = np.linspace(curve["x"][valid].min(), curve["x"][valid].max(), 64)
                    ax_arrival.plot(
                        line_x, fit["intercept"] + fit["slope"] * line_x,
                        color=color, alpha=alphas[threshold_index], linewidth=2.0,
                    )
    # Two series only, so identity is carried by the legend and by the direct
    # labels in the parameter box below rather than by colour alone.
    ax_arrival.plot([], [], color=_CLASSICAL_COLOR, linewidth=2.0,
                    label=f"classical, {_metric_spec(classical_metric)['label']}")
    ax_arrival.plot([], [], color=_QUANTUM_COLOR, linewidth=2.0,
                    label=f"entanglement, {_metric_spec(quantum_metric)['label']}")
    for size_index, dataset in enumerate(datasets):
        ax_arrival.plot(
            [], [], color="0.35", linestyle="none",
            marker=markers[size_index % len(markers)], markersize=5.0,
            label=rf"$L={dataset['L']}$",
        )
    cone_x = np.array([max(x_min, 0), max(dataset["x"].max() for dataset in datasets)], dtype=float)
    ax_arrival.plot(
        cone_x, cone_x, color="0.55", linewidth=1.0, linestyle="--",
        label=r"unitary cone $\tau=x$",
    )
    ax_arrival.set_xlabel(r"Distance from injection, $x$ (sites)")
    ax_arrival.set_ylabel(r"Arrival time, $\tau_{\mu,q}(x)$")
    ax_arrival.set_title(r"Front arrival: $\tau = b + \lambda x$")
    ax_arrival.legend(fontsize=8, ncol=2)

    velocity_lines = []
    for name, channel in ((classical_metric, "classical"), (quantum_metric, "entanglement")):
        for threshold in thresholds:
            estimates = [
                (fits[(name, threshold, dataset["L"])]["velocity"],
                 fits[(name, threshold, dataset["L"])]["velocity_stderr"])
                for dataset in datasets
            ]
            finite = [(value, error) for value, error in estimates if np.isfinite(value)]
            if not finite:
                continue
            values = np.array([value for value, _ in finite])
            spread = float(values.std(ddof=1)) if values.size > 1 else float(finite[0][1])
            velocity_lines.append(
                rf"$v_{{\rm {channel}}}(q={threshold:g}) = {values.mean():.2f} \pm {spread:.2f}$"
            )
    if velocity_lines:
        _annotate_axes(ax_arrival, "\n".join(velocity_lines), "upper left", fontsize=8)

    for threshold_index, threshold in enumerate(thresholds):
        band = bands[threshold]
        fit = band_fits[threshold]
        if band["x"].size == 0:
            continue
        ax_band.errorbar(
            band["x"], band["delta_tau"], yerr=band["delta_tau_stderr"],
            color=_QUANTUM_COLOR, alpha=alphas[threshold_index], marker="o", markersize=5.0,
            linestyle="none", capsize=2, elinewidth=1.0, label=rf"$q={threshold:g}$",
        )
        if np.isfinite(fit["slope"]):
            line_x = np.linspace(band["x"].min(), band["x"].max(), 64)
            ax_band.plot(
                line_x, fit["intercept"] + fit["slope"] * line_x,
                color=_QUANTUM_COLOR, alpha=alphas[threshold_index], linewidth=2.0,
            )
    ax_band.axhline(0.0, color="0.5", linewidth=1.0, linestyle=":")
    ax_band.set_xlabel(r"Distance from injection, $x$ (sites)")
    ax_band.set_ylabel(r"Band width, $\tau_{\mathcal{N}} - \tau_{C}$")
    ax_band.set_title("Classical-only band")
    if ax_band.get_legend_handles_labels()[0]:
        ax_band.legend(fontsize=8)

    band_text = []
    for threshold in thresholds:
        fit = band_fits[threshold]
        if not np.isfinite(fit["slope"]):
            continue
        significant = np.isfinite(fit["slope_stderr"]) and abs(fit["slope"]) > 2.0 * fit["slope_stderr"]
        verdict = "grows: $v_{\\rm cl} > v_{\\rm ent}$" if significant and fit["slope"] > 0.0 else (
            "shrinks" if significant else "flat: equal velocities"
        )
        band_text.append(
            rf"$q={threshold:g}$: $d\Delta\tau/dx = {fit['slope']:.3f} \pm {fit['slope_stderr']:.3f}$"
            f"  ({verdict})"
        )
    if band_text:
        _annotate_axes(ax_band, "\n".join(band_text), "upper left", fontsize=8)

    reference = datasets[0]
    fig.suptitle(
        f"Information fronts, {reference['circuit']}, "
        rf"$p={reference['p']:g}$, receiver width $w={width}$"
    )

    _show(fig, show)
    return {
        "figure": fig,
        "axes": axes,
        "datasets": datasets,
        "curves": curves,
        "fits": fits,
        "bands": bands,
        "band_fits": band_fits,
        "levels": levels,
        "noise_floor": noise_floor,
        "thresholds": thresholds,
        "width": width,
        "classical_metric": classical_metric,
        "quantum_metric": quantum_metric,
        "mi_units": _mi_unit_spec(mi_units)[0],
    }

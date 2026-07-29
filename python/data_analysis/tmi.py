"""Tripartite-information crossing and finite-size collapse."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .fitting import (
    _collapse_quality_text,
    _curvature_errors,
    _prepare_sorted_arrays,
)
from .loading import _as_paths, _chunked_group_summary
from .plotting import _mi_unit_spec, _show


def _summarize_tmi(
    path: str | Path,
    *,
    p_min: float | None,
    p_max: float | None,
    chunksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summary = _chunked_group_summary(
        path,
        ["tmi"],
        p_min=p_min,
        p_max=p_max,
        chunksize=chunksize,
    )["tmi"]
    return (
        summary["p"].to_numpy(dtype=float),
        summary["mean"].to_numpy(dtype=float),
        summary["stderr"].to_numpy(dtype=float),
    )


def _align_tmi_grids(
    ps_arr: Sequence[np.ndarray],
    mean_arr: Sequence[np.ndarray],
    stderr_arr: Sequence[np.ndarray],
    *,
    p_tol: float,
    min_points: int,
    allow_interpolation: bool,
    error_floor: float,
    sizes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    ps_arr, mean_arr, stderr_arr = _prepare_sorted_arrays(
        ps_arr, mean_arr, stderr_arr
    )
    matches: list[list[int]] = []
    for i_ref, p_ref in enumerate(ps_arr[0]):
        indices = [i_ref]
        for ps in ps_arr[1:]:
            k = int(np.argmin(np.abs(ps - p_ref)))
            if abs(ps[k] - p_ref) > p_tol:
                break
            indices.append(k)
        if len(indices) == len(ps_arr):
            matches.append(indices)

    if len(matches) >= min_points:
        rho = np.array(
            [np.mean([ps_arr[j][idx] for j, idx in enumerate(row)]) for row in matches]
        )
        a = np.array(
            [[mean_arr[j][row[j]] for row in matches] for j in range(len(ps_arr))]
        )
        da = np.array(
            [[stderr_arr[j][row[j]] for row in matches] for j in range(len(ps_arr))]
        )
        da = np.where(np.isfinite(da) & (da > 0.0), da, error_floor)
        return rho, a, da, "matched"

    if not allow_interpolation:
        raise RuntimeError(
            f"Fewer than {min_points} common p values were found within p_tol={p_tol:g}."
        )

    overlap_min = max(ps.min() for ps in ps_arr)
    overlap_max = min(ps.max() for ps in ps_arr)
    if overlap_min >= overlap_max:
        ranges = ", ".join(
            f"L={size}: [{ps.min():g}, {ps.max():g}]"
            for size, ps in zip(sizes, ps_arr)
        )
        raise RuntimeError(f"No common p range across datasets ({ranges}).")

    rho = ps_arr[0][(ps_arr[0] >= overlap_min) & (ps_arr[0] <= overlap_max)]
    if rho.size < min_points:
        native_count = min(
            np.count_nonzero((ps >= overlap_min) & (ps <= overlap_max))
            for ps in ps_arr
        )
        rho = np.linspace(overlap_min, overlap_max, max(native_count, min_points))
    if rho.size < min_points:
        raise RuntimeError(f"Only {rho.size} p points are available in the overlap.")

    a = np.asarray([np.interp(rho, ps, mean) for ps, mean in zip(ps_arr, mean_arr)])
    da = np.asarray(
        [np.interp(rho, ps, stderr) for ps, stderr in zip(ps_arr, stderr_arr)]
    )
    da = np.where(np.isfinite(da) & (da > 0.0), da, error_floor)
    return rho, a, da, "interpolated"


def tmi_collapse(
    files: Sequence[str | Path] | str | Path,
    sizes: Sequence[int],
    *,
    realizations: Sequence[int] | None = None,
    p_range: tuple[float | None, float | None] = (0.0, 1.0),
    pc0: float = 0.33,
    nu0: float = 2.0,
    zeta0: float = 0.0,
    fit_zeta: bool = False,
    x_bounds: tuple[float, float] | None = None,
    p_tol: float = 1e-6,
    min_common_p: int = 4,
    interpolate: bool = True,
    chunksize: int = 1_000_000,
    mi_units: str = "nats",
    error_floor: float = 1e-12,
    curvature_step: float = 1e-3,
    colors: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (10, 8),
    show: bool = True,
) -> dict[str, Any]:
    """Fit p_c and nu for the TMI crossing using pyfssa.

    ``mi_units`` selects ``"nats"`` or ``"bits"`` for TMI values and errors.
    """
    try:
        import fssa
    except ImportError as exc:
        raise ImportError(
            "tmi_collapse requires pyfssa (the import name is 'fssa')."
        ) from exc

    # Older pyfssa releases still reference np.int.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    mi_units, mi_scale = _mi_unit_spec(mi_units)

    paths = _as_paths(files)
    sizes = list(sizes)
    if len(paths) != len(sizes):
        raise ValueError("files and sizes must have equal lengths.")
    if realizations is not None and len(realizations) != len(paths):
        raise ValueError("realizations must match files when provided.")

    p_min, p_max = p_range
    ps_arr, mean_arr, stderr_arr = [], [], []
    for size, path in zip(sizes, paths):
        print(f"Importing L={size}: {path}")
        ps, mean, stderr = _summarize_tmi(
            path,
            p_min=p_min,
            p_max=p_max,
            chunksize=chunksize,
        )
        if ps.size == 0:
            raise RuntimeError(f"No usable TMI data were loaded from {path}.")
        ps_arr.append(ps)
        mean_arr.append(mean * mi_scale)
        stderr_arr.append(stderr * mi_scale)

    rho, a, da, alignment = _align_tmi_grids(
        ps_arr,
        mean_arr,
        stderr_arr,
        p_tol=p_tol,
        min_points=min_common_p,
        allow_interpolation=interpolate,
        error_floor=error_floor,
        sizes=sizes,
    )
    print(f"p-grid alignment: {alignment}, {rho.size} points")
    l = np.asarray(sizes, dtype=float)

    if fit_zeta:
        fit = fssa.autoscale(
            l=l,
            rho=rho,
            a=a,
            da=da,
            rho_c0=pc0,
            nu0=nu0,
            zeta0=zeta0,
            x_bounds=x_bounds,
        )
        pc, nu, zeta = float(fit["rho"]), float(fit["nu"]), float(fit["zeta"])
        dpc, dnu, dzeta = (
            float(fit["drho"]),
            float(fit["dnu"]),
            float(fit["dzeta"]),
        )
        success = bool(fit["success"])
        raw_fit = fit
    else:
        def objective(params):
            pc_trial, nu_trial = params
            if (
                not np.isfinite(pc_trial)
                or not np.isfinite(nu_trial)
                or nu_trial <= 0.0
                or pc_trial < rho.min()
                or pc_trial > rho.max()
            ):
                return np.inf
            scaled_trial = fssa.scaledata(
                l=l,
                rho=rho,
                a=a,
                da=da,
                rho_c=pc_trial,
                nu=nu_trial,
                zeta=zeta0,
            )
            quality = fssa.quality(
                scaled_trial.x,
                scaled_trial.y,
                scaled_trial.dy,
                x_bounds=x_bounds,
            )
            return float(quality) if np.isfinite(quality) else np.inf

        raw_fit = minimize(
            objective,
            x0=np.array([pc0, nu0], dtype=float),
            method="Nelder-Mead",
            options={
                "xatol": 1e-8,
                "fatol": 1e-8,
                "maxiter": 20_000,
                "maxfev": 20_000,
            },
        )
        pc, nu, zeta = float(raw_fit.x[0]), float(raw_fit.x[1]), float(zeta0)
        quality = float(objective(raw_fit.x))
        errors, covariance, hessian = _curvature_errors(
            objective,
            raw_fit.x,
            f_min=quality,
            rel_step=curvature_step,
        )
        dpc, dnu, dzeta = float(errors[0]), float(errors[1]), 0.0
        success = bool(raw_fit.success)

    scaled = fssa.scaledata(
        l=l,
        rho=rho,
        a=a,
        da=da,
        rho_c=pc,
        nu=nu,
        zeta=zeta,
    )
    quality = float(
        fssa.quality(scaled.x, scaled.y, scaled.dy, x_bounds=x_bounds)
    )
    print(f"p_c = {pc:.8g} ± {dpc:.2g}")
    print(f"nu  = {nu:.8g} ± {dnu:.2g}")
    print(
        f"zeta = {zeta:.8g} ± {dzeta:.2g}"
        if fit_zeta
        else f"zeta = {zeta:.8g} fixed"
    )
    print(f"collapse quality S = {quality:.8g}")
    print(_collapse_quality_text(quality))

    fig, (ax_raw, ax_collapse) = plt.subplots(2, 1, figsize=figsize)
    colors = list(colors) if colors is not None else [
        plt.get_cmap("tab10")(i % 10) for i in range(len(sizes))
    ]
    for i, size in enumerate(sizes):
        color = colors[i % len(colors)]
        label = rf"$L={size}$"
        if realizations is not None:
            label += f", Reals={realizations[i]:,}"
        ax_raw.errorbar(
            ps_arr[i],
            mean_arr[i],
            yerr=stderr_arr[i],
            fmt=".",
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1,
        )
        ax_raw.plot(ps_arr[i], mean_arr[i], color=color, label=label)
        ax_collapse.errorbar(
            scaled.x[i],
            scaled.y[i],
            yerr=scaled.dy[i],
            fmt=".",
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1,
            label=rf"$L={size}$",
        )
        ax_collapse.plot(scaled.x[i], scaled.y[i], color=color, alpha=0.8)

    ax_raw.axvline(pc, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_raw.set_xlabel(r"Measurement rate $p$")
    ax_raw.set_ylabel(rf"TMI $\overline{{I}}_3$ [{mi_units}]")
    ax_raw.set_title("Raw TMI crossing")
    ax_raw.legend()

    ax_collapse.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_collapse.set_xlabel(r"Scaled variable $(p-p_c)L^{1/\nu}$")
    ax_collapse.set_ylabel(
        rf"Scaled TMI $L^{{-\zeta/\nu}}\overline{{I}}_3$ [{mi_units}]"
        if fit_zeta
        else rf"TMI $\overline{{I}}_3$ [{mi_units}]"
    )
    ax_collapse.set_title("Finite-size scaling collapse")
    ax_collapse.legend()

    text = (
        rf"$p_c={pc:.4g}\pm {dpc:.1g}$" "\n"
        rf"$\nu={nu:.3g}\pm {dnu:.1g}$" "\n"
        rf"$S={quality:.3g}$" "\n"
        + (
            rf"$\zeta={zeta:.3g}\pm {dzeta:.1g}$"
            if fit_zeta
            else rf"$\zeta={zeta:g}$ fixed"
        )
    )
    ax_collapse.text(
        0.03,
        0.97,
        text,
        transform=ax_collapse.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    _show(fig, show)

    return {
        "figure": fig,
        "axes": (ax_raw, ax_collapse),
        "pc": pc,
        "pc_stderr": dpc,
        "nu": nu,
        "nu_stderr": dnu,
        "zeta": zeta,
        "zeta_stderr": dzeta,
        "quality": quality,
        "quality_interpretation": _collapse_quality_text(quality),
        "success": success,
        "alignment": alignment,
        "mi_units": mi_units,
        "p_grid": rho,
        "scaled": scaled,
        "raw_fit": raw_fit,
        "raw_data": {
            size: {"p": ps, "mean": mean, "stderr": stderr}
            for size, ps, mean, stderr in zip(sizes, ps_arr, mean_arr, stderr_arr)
        },
    }

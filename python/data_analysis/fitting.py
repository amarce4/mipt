"""Generic fitting and error-estimation helpers.

Nothing here knows about a specific observable; domain-specific fits live
beside the function that uses them.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _weighted_linear_fit(
    x: Iterable[float],
    y: Iterable[float],
    sigma: Iterable[float],
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sigma)
        & (sigma > 0.0)
    )
    x, y, sigma = x[valid], y[valid], sigma[valid]
    if x.size < 3:
        raise ValueError("At least three valid points are required for the fit.")

    design = np.column_stack((x, np.ones_like(x)))
    weights = 1.0 / sigma**2
    normal = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Linear-fit covariance matrix is singular.") from exc

    slope, intercept = covariance @ (design.T @ (weights * y))
    fitted = slope * x + intercept
    reduced_chi2 = np.sum(((y - fitted) / sigma) ** 2) / (x.size - 2)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_stderr": float(np.sqrt(covariance[0, 0])),
        "intercept_stderr": float(np.sqrt(covariance[1, 1])),
        "reduced_chi2": float(reduced_chi2),
    }


def _prepare_sorted_arrays(
    ps_arr: Sequence[np.ndarray],
    mean_arr: Sequence[np.ndarray],
    stderr_arr: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    ps_out, mean_out, stderr_out = [], [], []
    for ps, mean, stderr in zip(ps_arr, mean_arr, stderr_arr):
        ps = np.asarray(ps, dtype=float)
        order = np.argsort(ps)
        ps_out.append(ps[order])
        mean_out.append(np.asarray(mean, dtype=float)[order])
        stderr_out.append(np.asarray(stderr, dtype=float)[order])
    return ps_out, mean_out, stderr_out


def _curvature_errors(
    objective,
    theta_hat: Sequence[float],
    *,
    f_min: float | None = None,
    rel_step: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_hat = np.asarray(theta_hat, dtype=float)
    if f_min is None:
        f_min = objective(theta_hat)
    steps = rel_step * np.maximum(np.abs(theta_hat), 1.0)
    hessian = np.zeros((theta_hat.size, theta_hat.size), dtype=float)

    def safe_eval(x):
        value = objective(x)
        return value if np.isfinite(value) else np.inf

    for i in range(theta_hat.size):
        step = steps[i]
        for _ in range(12):
            xp, xm = theta_hat.copy(), theta_hat.copy()
            xp[i] += step
            xm[i] -= step
            fp, fm = safe_eval(xp), safe_eval(xm)
            if np.isfinite(fp) and np.isfinite(fm):
                break
            step *= 0.5
        hessian[i, i] = (
            (fp - 2.0 * f_min + fm) / step**2
            if np.isfinite(fp) and np.isfinite(fm)
            else np.nan
        )

    for i in range(theta_hat.size):
        for j in range(i + 1, theta_hat.size):
            hi, hj = steps[i], steps[j]
            values = [np.inf] * 4
            for _ in range(12):
                points = []
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    x = theta_hat.copy()
                    x[i] += si * hi
                    x[j] += sj * hj
                    points.append(x)
                values = [safe_eval(point) for point in points]
                if all(np.isfinite(value) for value in values):
                    break
                hi *= 0.5
                hj *= 0.5
            if all(np.isfinite(value) for value in values):
                fpp, fpm, fmp, fmm = values
                value = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj)
            else:
                value = np.nan
            hessian[i, j] = hessian[j, i] = value

    if not np.all(np.isfinite(hessian)):
        return (
            np.full(theta_hat.size, np.nan),
            np.full_like(hessian, np.nan),
            hessian,
        )
    try:
        covariance = 2.0 * f_min * np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = 2.0 * f_min * np.linalg.pinv(hessian)
    errors = np.sqrt(np.where(np.diag(covariance) >= 0.0, np.diag(covariance), np.nan))
    return errors, covariance, hessian


def _collapse_quality_text(score: float) -> str:
    if not np.isfinite(score):
        return "The collapse fit failed or produced invalid scaled data."
    if score < 0.5:
        return "Suspiciously small S; errors may be overestimated or data correlated."
    if score < 2.0:
        return "S is near unity; the collapse is statistically reasonable."
    if score < 5.0:
        return "The collapse is marginal; corrections to scaling may be relevant."
    if score < 10.0:
        return "The collapse is poor; vary the fit window or system sizes."
    return "The collapse is very poor under the assumed scaling ansatz."

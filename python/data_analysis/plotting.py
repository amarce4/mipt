"""Figure-level helpers shared by every plotting function."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _show(fig, show: bool) -> None:
    if show:
        plt.show()


def _mi_unit_spec(mi_units: str) -> tuple[str, float]:
    """Return the canonical MI unit and the nats-to-unit scale factor."""
    key = str(mi_units).strip().lower()
    key = {"nat": "nats", "bit": "bits"}.get(key, key)
    if key not in {"nats", "bits"}:
        raise ValueError("mi_units must be 'nats' or 'bits'.")
    return key, 1.0 if key == "nats" else 1.0 / np.log(2.0)


def _color_map_by_size(sizes: Sequence[int], cmap_name: str) -> dict[int, Any]:
    cmap = plt.get_cmap(cmap_name)
    positions = (
        np.array([0.15])
        if len(sizes) == 1
        else np.linspace(0.10, 0.90, len(sizes))
    )
    return {size: cmap(position) for size, position in zip(sizes, positions)}


def _positive_log_errorbar(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    **kwargs,
):
    """Draw log-compatible error bars without extending below zero."""
    lower = np.minimum(np.maximum(dy, 0.0), 0.999999 * y)
    upper = np.maximum(dy, 0.0)
    return ax.errorbar(x, y, yerr=np.vstack((lower, upper)), **kwargs)

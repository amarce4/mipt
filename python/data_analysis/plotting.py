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


# ---------------------------------------------------------------------------
# Inset layout
#
# Every probe figure is a primary panel plus a secondary one (a scaling
# collapse, or the late-time zoom of the anisotropy figure). The secondary
# panel can either sit beside the primary as its own subplot or be inset into
# it. An inset is flush with the parent's frame on the sides it is anchored
# to, and carries no surrounding margin of its own; the one exception is the
# bottom of a lower inset, which is lifted so that both x axes stay readable.
# ---------------------------------------------------------------------------

_INSET_CORNERS = ("lower right", "lower left", "upper right", "upper left")

_ANNOTATION_POSITIONS = {
    "upper left": (0.03, 0.97, "left", "top"),
    "upper right": (0.97, 0.97, "right", "top"),
    "lower left": (0.03, 0.03, "left", "bottom"),
    "lower right": (0.97, 0.03, "right", "bottom"),
}

# Where the legend and the parameter box go once the inset owns one corner,
# and which side of the primary panel needs extra room for them. Matplotlib's
# ``loc="best"`` only avoids artists, not child axes, so it happily parks the
# legend on top of the inset; these defaults keep the two apart.
_INSET_OVERLAY_DEFAULTS = {
    "lower right": ("upper left", "upper right", "top"),
    "lower left": ("upper right", "upper left", "top"),
    "upper right": ("lower left", "lower right", "bottom"),
    "upper left": ("lower right", "lower left", "bottom"),
}


def _normalize_corner(corner: str) -> str:
    key = str(corner).strip().lower().replace("_", " ")
    return {
        "bottom right": "lower right", "bottom left": "lower left",
        "top right": "upper right", "top left": "upper left",
    }.get(key, key)


def _inset_overlay_defaults(corner: str) -> tuple[str, str, str]:
    """Return ``(legend_loc, annotation_loc, reserve_side)`` for an inset corner."""
    key = _normalize_corner(corner)
    if key not in _INSET_OVERLAY_DEFAULTS:
        raise ValueError(f"inset corner must be one of {_INSET_CORNERS}.")
    return _INSET_OVERLAY_DEFAULTS[key]


def _reserve_axis_space(ax, side: str, fraction: float) -> None:
    """Grow the y range on one side so overlays do not cover the data."""
    if fraction <= 0.0:
        return
    lower, upper = ax.get_ylim()
    if not (np.isfinite(lower) and np.isfinite(upper)) or upper == lower:
        return
    if ax.get_yscale() == "log":
        if lower <= 0.0 or upper <= 0.0:
            return
        # Capped at one decade: a log axis whose lower limit was dragged down
        # by a near-zero point can span 20 decades, and a plain fractional
        # expansion of that is most of the figure.
        factor = min((upper / lower) ** fraction, 10.0)
        if side == "top":
            ax.set_ylim(lower, upper * factor)
        else:
            ax.set_ylim(lower / factor, upper)
        return
    span = (upper - lower) * fraction
    if side == "top":
        ax.set_ylim(lower, upper + span)
    else:
        ax.set_ylim(lower - span, upper)


def _inset_rectangle(
    corner: str,
    size: tuple[float, float],
    pad: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Return the ``[x0, y0, w, h]`` axes-fraction rectangle for an inset."""
    key = _normalize_corner(corner)
    if key not in _INSET_CORNERS:
        raise ValueError(f"inset corner must be one of {_INSET_CORNERS}.")
    width, height = (float(size[0]), float(size[1]))
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError("inset size components must lie in (0, 1].")
    pad_x, pad_y = (float(pad[0]), float(pad[1]))
    x0 = 1.0 - pad_x - width if "right" in key else pad_x
    y0 = 1.0 - pad_y - height if "upper" in key else pad_y
    return (x0, y0, width, height)


def _style_inset(
    ax,
    fontsize: float,
    facecolor: str,
    alpha: float,
    rectangle: tuple[float, float, float, float] | None = None,
) -> None:
    """Shrink an inset's decorations and give it an opaque face.

    Nothing is drawn outside the frame except the tick labels and the axis
    label, so the inset covers exactly the rectangle it was given. On an edge
    the inset shares with its parent, the last tick label would run past the
    parent's frame and into the parent's own label, so it is pruned.
    """
    from matplotlib.ticker import MaxNLocator

    ax.set_facecolor(facecolor)
    ax.patch.set_alpha(alpha)
    ax.tick_params(labelsize=fontsize, length=2.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if rectangle is None:
        return
    ax._mipt_inset_rect = rectangle
    _apply_inset_locators(ax)


def _apply_inset_locators(ax) -> None:
    """Thin out an inset's ticks and prune the ones on a shared edge.

    ``set_yscale``/``set_xscale`` reinstall the default locators, so this has
    to run again once the caller has finished setting the scales.
    """
    from matplotlib.ticker import MaxNLocator

    rectangle = getattr(ax, "_mipt_inset_rect", None)
    if rectangle is None:
        return
    x0, y0, width, height = rectangle
    steps = [1, 2, 2.5, 5, 10]
    if ax.get_xscale() == "linear":
        ax.xaxis.set_major_locator(
            MaxNLocator(
                nbins=5,
                steps=steps,
                prune="upper" if x0 + width >= 1.0 - 1e-9 else None,
            )
        )
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=4,
                steps=steps,
                prune="upper" if y0 + height >= 1.0 - 1e-9 else None,
            )
        )


def _inset_title(ax, title: str, fontsize: float, headroom: float = 0.18):
    """Title an inset from the inside.

    A normal title sits above the frame, which for a corner-flush inset is
    either outside the parent axes or on top of the parent's data. This one
    lives in space reserved at the top of the inset itself.
    """
    _reserve_axis_space(ax, "top", headroom)
    return ax.text(
        0.5,
        0.985,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        zorder=6,
    )


def _paired_axes(
    *,
    inset: bool,
    figsize: tuple[float, float],
    dpi: int,
    nrows: int = 1,
    inset_corner: str = "lower right",
    inset_size: tuple[float, float] = (0.42, 0.40),
    inset_pad: tuple[float, float] = (0.0, 0.155),
    inset_fontsize: float = 7.5,
    inset_facecolor: str = "white",
    inset_alpha: float = 1.0,
):
    """Build ``nrows`` primary/secondary panel pairs, side by side or inset.

    Returns ``(figure, pairs)`` where each pair is ``(primary, secondary)``.
    """
    if inset:
        fig, column = plt.subplots(
            nrows, 1, figsize=figsize, dpi=dpi, squeeze=False,
            constrained_layout=True,
        )
        rectangle = _inset_rectangle(inset_corner, inset_size, inset_pad)
        pairs = []
        for (primary,) in column:
            secondary = primary.inset_axes(rectangle)
            _style_inset(
                secondary,
                inset_fontsize,
                inset_facecolor,
                inset_alpha,
                rectangle,
            )
            # An inset is a child artist of its parent axes, so it is drawn
            # in the parent's zorder queue and has to outrank the data there.
            secondary.set_zorder(5.0)
            pairs.append((primary, secondary))
        return fig, pairs

    fig, grid = plt.subplots(
        nrows, 2, figsize=figsize, dpi=dpi, squeeze=False,
        constrained_layout=True,
    )
    return fig, [tuple(row) for row in grid]


def _font_kwargs(inset: bool, fontsize: float) -> dict[str, Any]:
    """Font override applied to inset labels and titles only."""
    return {"fontsize": fontsize} if inset else {}


def _set_secondary_title(
    ax,
    title: str | None,
    *,
    inset: bool,
    fontsize: float,
    headroom: float = 0.18,
):
    """Title the secondary panel, inside the frame when it is an inset.

    Call this after the panel's scales and limits are final: it also restores
    the inset tick locators, which ``set_yscale``/``set_xscale`` reset, and
    the inset form reserves room by changing the y limits.
    """
    if not inset:
        return ax.set_title(title) if title else None
    _apply_inset_locators(ax)
    if not title:
        return None
    return _inset_title(ax, title, fontsize, headroom)


def _annotate_axes(ax, text: str, loc: str, **kwargs) -> Any:
    """Place a boxed parameter annotation in a named corner of ``ax``."""
    key = _normalize_corner(loc)
    if key not in _ANNOTATION_POSITIONS:
        raise ValueError(
            f"annotation location must be one of {sorted(_ANNOTATION_POSITIONS)}."
        )
    x, y, ha, va = _ANNOTATION_POSITIONS[key]
    kwargs.setdefault("bbox", {"boxstyle": "round", "facecolor": "white", "alpha": 0.88})
    kwargs.setdefault("zorder", 6)
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **kwargs)

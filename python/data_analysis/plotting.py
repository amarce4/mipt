"""Figure-level helpers shared by every plotting function.

The plotting package follows the compact journal style used in
Lyu *et al.*, arXiv:2602.04969: Computer-Modern-compatible serif text and
mathematics, inward ticks on all four sides, thin frames and unobtrusive
legends.  The style is installed once when this module is imported so every
public plotter uses the same typography and geometry.  Individual plotters
remain responsible for the scientifically meaningful choices (linear or
logarithmic axes, grids, colours, marker encodings and panel layout).
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Publication style
#
# The paper embeds Computer Modern Roman/italic throughout the principal plot
# labels and mathematics (Latin Modern is its OpenType continuation).  A small
# subset of numeric ticks in one panel embeds Helvetica, but that is not the
# dominant figure face.  Latin Modern Roman therefore gives the closest
# consistent match across every plot; Nimbus Roman and DejaVu Serif keep the
# code portable when TeX fonts are unavailable.  ``text.usetex`` stays disabled
# deliberately so the style does not require a working TeX installation.
# ---------------------------------------------------------------------------

PAPER_RCPARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": [
        "Latin Modern Roman",
        "Computer Modern Roman",
        "CMU Serif",
        "Times New Roman",
        "Nimbus Roman",
        "DejaVu Serif",
    ],
    "font.size": 10.0,
    "mathtext.fontset": "cm",
    "mathtext.default": "it",
    "axes.unicode_minus": True,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11.0,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.labelpad": 4.0,
    "axes.formatter.use_mathtext": True,
    "axes.grid": False,
    "grid.color": "0.72",
    "grid.linestyle": ":",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.28,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2,
    "ytick.minor.size": 2.2,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.minor.width": 0.55,
    "ytick.minor.width": 0.55,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize": 8.0,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.65",
    "legend.borderpad": 0.35,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.7,
    "legend.handletextpad": 0.45,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.0,
    "errorbar.capsize": 2.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}


def apply_paper_style() -> None:
    """Install the package's publication style in Matplotlib.

    The function is public enough to be called again after a notebook changes
    ``rcParams``.  Importing :mod:`data_analysis.plotting` calls it once, so
    ordinary users do not need to do anything.
    """
    plt.rcParams.update(PAPER_RCPARAMS)


apply_paper_style()


def _show(fig, show: bool) -> None:
    if show:
        plt.show()


def _panel_label(
    ax,
    label: str,
    *,
    x: float = 0.018,
    y: float = 0.982,
    fontsize: float = 11.0,
    outside: bool = False,
):
    """Place a compact journal-style panel label such as ``"a)"``.

    ``outside=True`` moves the label just left of the upper frame, which is
    useful for flush vertical stacks whose data reach the top-left corner.
    """
    if outside:
        x = -0.085
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        clip_on=False,
        zorder=20,
    )


def _mi_unit_spec(mi_units: str, name: str = "mi_units") -> tuple[str, float]:
    """Return the canonical entropy unit and the nats-to-unit scale factor.

    Every entropy-valued observable in this package -- entropies, mutual
    informations, tripartite informations and the *logarithmic* negativity --
    goes through this one helper, so ``[bits]``/``[nats]`` means the same thing
    and converts by the same factor everywhere. ``name`` only shapes the error
    message, since the callers spell the argument ``units``, ``mi_units`` or
    ``entropy_units`` depending on what they plot.

    Bare negativities (``\\mathcal{N} = (||rho^T||_1 - 1)/2``) are *not*
    entropy-valued and must not be passed through it.
    """
    key = str(mi_units).strip().lower()
    key = {"nat": "nats", "bit": "bits"}.get(key, key)
    if key not in {"nats", "bits"}:
        raise ValueError(f"{name} must be 'nats' or 'bits'.")
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
    "center right": (0.97, 0.30, "right", "center"),
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
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=fontsize,
        length=2.8,
        width=0.65,
        pad=1.5,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=1.6,
        width=0.5,
    )
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
    sharex: bool = False,
    inset_corner: str = "lower right",
    inset_size: tuple[float, float] = (0.42, 0.40),
    inset_pad: tuple[float, float] = (0.0, 0.155),
    inset_fontsize: float = 7.5,
    inset_facecolor: str = "white",
    inset_alpha: float = 1.0,
):
    """Build ``nrows`` primary/secondary panel pairs, side by side or inset.

    Returns ``(figure, pairs)`` where each pair is ``(primary, secondary)``.
    ``sharex`` ties the rows of each column together, which also hides the
    x tick labels of every row but the last.
    """
    if inset:
        fig, column = plt.subplots(
            nrows, 1, figsize=figsize, dpi=dpi, squeeze=False,
            sharex=sharex,
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
        # "col", not True: the two columns hold different observables, so only
        # the rows within a column share an x axis.
        sharex="col" if sharex else False,
        constrained_layout=True,
    )
    return fig, [tuple(row) for row in grid]


# ---------------------------------------------------------------------------
# Flush vertical stacks
#
# A stack of panels that share one x axis reads as a single plot only if the
# rows actually touch. ``constrained_layout`` will not do that: its row
# spacing is computed from each axes' tight bounding box, which includes the
# outward tick marks, so even at hspace=0 and h_pad=0 it leaves ~4 pt between
# frames. The fix is to let it settle the *outer* margins, freeze it, and then
# re-lay the rows by hand. Inset axes are positioned in their parent's axes
# fraction, so they follow without any extra work.
# ---------------------------------------------------------------------------


def _flush_stacked_axes(fig, rows: Sequence[Sequence[Any]]) -> None:
    """Give every column's rows equal heights and shared horizontal edges."""
    if len(rows) < 2:
        return
    fig.canvas.draw()
    positions = [[ax.get_position() for ax in row] for row in rows]
    fig.set_layout_engine("none")
    top = max(position.y1 for position in positions[0])
    bottom = min(position.y0 for position in positions[-1])
    height = (top - bottom) / len(rows)
    for index, row in enumerate(rows):
        y0 = top - (index + 1) * height
        for ax, position in zip(row, positions[index]):
            ax.set_position([position.x0, y0, position.width, height])


def _prune_upper_yticks(ax) -> None:
    """Drop the top y tick label of a panel whose top edge is shared.

    Without this the highest label of a lower panel and the lowest label of
    the panel above it collide on the shared frame edge.
    """
    from matplotlib.ticker import MaxNLocator

    if ax.get_yscale() != "linear":
        return
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune="upper")
    )


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

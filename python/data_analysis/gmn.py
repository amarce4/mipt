"""GMN / fGMN measurement-rate scans."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import warnings

import matplotlib.pyplot as plt
import pandas as pd

from .loading import _as_paths, _chunked_group_summary, _parse_size, read_metadata_sidecar
from .plotting import _show


def _resolve_sizes(
    paths: Sequence[Path],
    sizes: Sequence[int] | None,
) -> list[int]:
    """System sizes for each input, from the caller, a sidecar, or the name."""
    if sizes is not None:
        return list(sizes)

    resolved: list[int] = []
    for path in paths:
        metadata = read_metadata_sidecar(path)
        if metadata is not None and "N" in metadata:
            resolved.append(int(metadata["N"]))
            continue
        try:
            resolved.append(_parse_size(path))
        except ValueError as exc:
            raise ValueError(
                f"Could not determine L for {path}. Pass sizes=[...] explicitly, "
                "or rerun the solver so it writes the *_meta.csv sidecar."
            ) from exc
    return resolved


def gmn(
    files: Sequence[str | Path] | str | Path,
    sizes: Sequence[int] | None = None,
    *,
    metric: str = "gmn",
    p_range: tuple[float | None, float | None] = (0.0, 1.0),
    chunksize: int = 1_000_000,
    plot_bipneg: bool = True,
    periods: int | None = None,
    realizations: int | None = None,
    colors: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (10, 4.5),
    show: bool = True,
) -> dict[str, Any]:
    """Plot GMN or fGMN means and standard errors versus measurement rate."""
    metric = metric.lower()
    if metric not in {"gmn", "fgmn"}:
        raise ValueError("metric must be 'gmn' or 'fgmn'.")

    paths = _as_paths(files)
    sizes = _resolve_sizes(paths, sizes)
    if len(paths) != len(sizes):
        raise ValueError("files and sizes must have equal lengths.")

    p_min, p_max = p_range
    data: dict[int, dict[str, pd.DataFrame]] = {}
    columns = [metric] + (["bipneg"] if plot_bipneg else [])

    for size, path in zip(sizes, paths):
        print(f"Importing L={size}: {path}")
        data[size] = _chunked_group_summary(
            path,
            columns,
            p_min=p_min,
            p_max=p_max,
            chunksize=chunksize,
        )

    fig, ax = plt.subplots(figsize=figsize)
    colors = list(colors) if colors is not None else [
        plt.get_cmap("tab10")(i % 10) for i in range(len(sizes))
    ]
    metric_name = "GMN" if metric == "gmn" else "fGMN"

    for i, size in enumerate(sizes):
        color = colors[i % len(colors)]
        curve = data[size][metric]
        if curve.empty:
            warnings.warn(f"No valid {metric} data found for L={size}.")
            continue
        ax.errorbar(
            curve["p"],
            curve["mean"],
            yerr=curve["stderr"],
            fmt=".",
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1,
            label=rf"{metric_name}, $L={size}$",
        )
        ax.plot(curve["p"], curve["mean"], color=color, linewidth=1)

        if plot_bipneg:
            bip = data[size]["bipneg"]
            if not bip.empty:
                ax.errorbar(
                    bip["p"],
                    bip["mean"],
                    yerr=bip["stderr"],
                    fmt="x",
                    color=color,
                    ecolor=color,
                    capsize=2,
                    elinewidth=0.8,
                    alpha=0.75,
                    label=rf"Min. bipartite negativity, $L={size}$",
                )
                ax.plot(
                    bip["p"],
                    bip["mean"],
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.75,
                )

    ax.set_xlabel(r"Measurement rate $p$")
    ax.set_ylabel(metric_name)
    if p_min is not None and p_max is not None:
        ax.set_xlim(p_min, p_max)
    ax.grid(alpha=0.2)
    ax.legend()

    metadata: list[str] = []
    if len(sizes) == 1:
        metadata.append(rf"$L={sizes[0]}$")
    if periods is not None:
        metadata.append(rf"$t={periods}$")
    if realizations is not None:
        metadata.append(f"Realisations per p: {realizations:,}")
    if len(sizes) == 1 and not data[sizes[0]][metric].empty:
        counts = data[sizes[0]][metric]["count"].to_numpy(dtype=int)
        if counts.min() == counts.max():
            metadata.append(f"Rows per p: {counts.min():,}")
        else:
            metadata.append(f"Rows per p: {counts.min():,}–{counts.max():,}")
    if metadata:
        ax.text(
            0.98,
            0.96,
            "\n".join(metadata),
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="top",
            bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.7},
        )

    fig.tight_layout()
    _show(fig, show)
    return {"figure": fig, "axis": ax, "data": data}

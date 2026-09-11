"""The connected-zero triple analysis of ``dist_scaling.exe``.

With ``MIPT_DIST_PAIR_GAP_TRIPLES=1`` the executable takes every pair that is
graph-connected but carries no fermionic negativity above threshold,

    C_ij = 1   and   N^f_ij <= eps_2,

adds *every* third site k, and analyses the fermionic three-site RDM: all three
pair marginals, the three one-versus-rest cut negativities, and fGMN. It writes

* ``<stem>_connected_zero_thirds.csv`` -- one row per (trajectory, i, j; k)
  outcome, ~110 columns, and large: ``L - 2`` rows per qualifying pair;
* ``<stem>_connected_zero_thirds_aggregate.csv`` -- one row per (role, pair
  class, separation), with the pair-level partition beside it.

This module turns those into the quantity the analysis exists for,

    R_3(d) = P( exists k: fGMN_ijk > eps_3  |  C_ij, N^f_ij <= eps_2 ),

with errors from a bootstrap **clustered on realization_id** -- every row of one
trajectory comes from one state, so treating millions of rows as independent
would understate every error bar -- and the partition of the pair-level gap

    P(C) - P(E_2) = P(C, !E_2, R_3) + P(C, !E_2, !R_3) - P(!C, E_2).

The last term is the eta contribution. The two-term version assumes it vanishes;
it is kept here so that assumption is checked rather than made.

What a result means: a positive fGMN for a triple containing a silent pair is
genuine tripartite entanglement hidden from the pair marginal -- not network
multipartite entanglement. R_3 = 0 at every k does not exclude entanglement that
needs four or more sites; it rules out the three-site explanation at the chosen
tolerance.

The outcome CSV is streamed in bounded chunks and never loaded whole.
"""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .loading import _resolve_files
from .plotting import _show
from .records import (
    FLAG_CONNECTED,
    FLAG_EVALUATED,
    histogram_edges,
    iter_record_chunks,
    record_file_info,
)

ZERO_CLASSES = ("occupation_correlated", "coherent_subthreshold", "silent")
ROLES = ("anchor", "control")

# The columns the pair-level reduction reads. Everything else stays on disk.
_OUTCOME_COLUMNS = (
    "realization_id",
    "anchor_role",
    "pair_i",
    "pair_j",
    "third_k",
    "separation",
    "chord",
    "pair_class",
    "fgmn_method",
    "fgmn",
    "fgmn_positive",
    "fgmn_status",
    "min_cut_site",
    "min_cut",
    "marginal_residual_ij",
)

_GROUP_KEYS = ["realization_id", "anchor_role", "pair_i", "pair_j"]


def aggregate_path_for(outcome_csv: str | Path) -> Path:
    path = Path(outcome_csv)
    return path.with_name(path.stem + "_aggregate.csv")


def outcome_path_for(aggregate_csv: str | Path) -> Path:
    path = Path(aggregate_csv)
    stem = path.stem
    if stem.endswith("_aggregate"):
        stem = stem[: -len("_aggregate")]
    return path.with_name(stem + ".csv")


def read_pair_gap_aggregate(path: str | Path) -> pd.DataFrame:
    """The aggregate CSV, as written."""
    return pd.read_csv(path)


def _reduce_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (trajectory, role, pair) from its L-2 outcome rows."""
    failed = (frame["fgmn_method"] == "mosek") & (frame["fgmn_status"] != "ok")
    work = frame.assign(
        _failed=failed.astype(np.int64),
        _positive=frame["fgmn_positive"].astype(np.int64),
        _prefiltered=(frame["fgmn_method"] == "prefilter_bound").astype(np.int64),
        _block_i=(frame["min_cut_site"] == "i").astype(np.int64),
        _block_j=(frame["min_cut_site"] == "j").astype(np.int64),
        _block_k=(frame["min_cut_site"] == "k").astype(np.int64),
    )
    grouped = work.groupby(_GROUP_KEYS, sort=False, observed=True)
    out = grouped.agg(
        separation=("separation", "first"),
        d=("chord", "first"),
        pair_class=("pair_class", "first"),
        thirds=("third_k", "size"),
        positive_thirds=("_positive", "sum"),
        failed_thirds=("_failed", "sum"),
        prefiltered_thirds=("_prefiltered", "sum"),
        max_fgmn=("fgmn", "max"),
        mean_fgmn=("fgmn", "mean"),
        max_marginal_residual=("marginal_residual_ij", "max"),
        blocked_by_i=("_block_i", "sum"),
        blocked_by_j=("_block_j", "sum"),
        blocked_by_k=("_block_k", "sum"),
    ).reset_index()
    out["r3"] = out["positive_thirds"] > 0
    # No positive k, but at least one k whose solve failed: the pair's R_3 is
    # unknown, and it is kept out of both the numerator and the denominator.
    out["undetermined"] = (~out["r3"]) & (out["failed_thirds"] > 0)
    return out


def pair_level_table(
    outcome_csv: str | Path,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Stream the outcome CSV into one row per (trajectory, role, pair).

    Rows of one pair are written consecutively, so a group can only straddle a
    chunk boundary at its end; the trailing group of each chunk is carried into
    the next. The result is small -- one row per qualifying pair -- whatever the
    size of the file.
    """
    columns = list(_OUTCOME_COLUMNS)
    pieces: list[pd.DataFrame] = []
    carry: pd.DataFrame | None = None
    reader = pd.read_csv(
        outcome_csv,
        usecols=columns,
        chunksize=chunksize,
        dtype={
            "anchor_role": "category",
            "pair_class": "category",
            "fgmn_method": "category",
            "fgmn_status": "category",
            "min_cut_site": "category",
        },
    )
    for chunk in reader:
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            for name in ("anchor_role", "pair_class", "fgmn_method", "fgmn_status", "min_cut_site"):
                chunk[name] = chunk[name].astype("category")
        if chunk.empty:
            continue
        last = chunk.iloc[-1][_GROUP_KEYS]
        tail = np.ones(len(chunk), dtype=bool)
        for key in _GROUP_KEYS:
            tail &= (chunk[key] == last[key]).to_numpy()
        carry = chunk.loc[tail]
        head = chunk.loc[~tail]
        if not head.empty:
            pieces.append(_reduce_groups(head))
    if carry is not None and not carry.empty:
        pieces.append(_reduce_groups(carry))
    if not pieces:
        return pd.DataFrame(
            columns=_GROUP_KEYS
            + ["separation", "d", "pair_class", "thirds", "positive_thirds", "r3", "undetermined"]
        )
    table = pd.concat(pieces, ignore_index=True)
    expected = int(table["thirds"].mode().iloc[0])
    if (table["thirds"] != expected).any():
        raise ValueError(
            f"{outcome_csv}: qualifying pairs carry {sorted(table['thirds'].unique())} rows; "
            "every pair should have exactly L-2. The file is damaged or truncated."
        )
    return table


def r3_bootstrap(
    pairs: pd.DataFrame,
    *,
    resamples: int = 400,
    seed: int = 0,
    batch: int = 50,
) -> pd.DataFrame:
    """R_3 per (role, class, separation), with trajectory-clustered errors.

    The bootstrap resamples whole trajectories, which is the only resampling
    consistent with how the data were taken: the pairs of one trajectory share a
    state, a circuit and a measurement record. Undetermined pairs are excluded
    from both the numerator and the denominator.
    """
    usable = pairs.loc[~pairs["undetermined"]].copy()
    usable["cell"] = (
        usable["anchor_role"].astype(str)
        + "|"
        + usable["pair_class"].astype(str)
        + "|"
        + usable["separation"].astype(int).astype(str)
    )
    cells = sorted(usable["cell"].unique())
    realizations, realization_index = np.unique(
        usable["realization_id"].to_numpy(), return_inverse=True
    )
    cell_index = pd.Categorical(usable["cell"], categories=cells).codes
    shape = (len(realizations), len(cells))
    flat = realization_index * len(cells) + cell_index
    size = shape[0] * shape[1]
    totals = np.bincount(flat, minlength=size).reshape(shape).astype(float)
    hits = np.bincount(flat, weights=usable["r3"].to_numpy(dtype=float), minlength=size).reshape(shape)

    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(cells)))
    done = 0
    while done < resamples:
        take = min(batch, resamples - done)
        weights = rng.multinomial(
            len(realizations), np.full(len(realizations), 1.0 / len(realizations)), size=take
        ).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[done : done + take] = (weights @ hits) / (weights @ totals)
        done += take

    rows = []
    for position, cell in enumerate(cells):
        role, zero_class, separation = cell.split("|")
        subset = usable.loc[usable["cell"] == cell]
        n = float(totals[:, position].sum())
        k = float(hits[:, position].sum())
        column = draws[:, position]
        finite = column[np.isfinite(column)]
        rows.append(
            {
                "anchor_role": role,
                "pair_class": zero_class,
                "separation": int(separation),
                "d": float(subset["d"].iloc[0]),
                "pairs": int(n),
                "r3_pairs": int(k),
                "undetermined_pairs": int(
                    (
                        pairs["undetermined"]
                        & (pairs["anchor_role"].astype(str) == role)
                        & (pairs["pair_class"].astype(str) == zero_class)
                        & (pairs["separation"] == int(separation))
                    ).sum()
                ),
                "trajectories": int((totals[:, position] > 0).sum()),
                "r3": k / n if n > 0 else np.nan,
                "r3_stderr": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
                "r3_lo": float(np.quantile(finite, 0.16)) if finite.size else np.nan,
                "r3_hi": float(np.quantile(finite, 0.84)) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["anchor_role", "pair_class", "separation"]).reset_index(
        drop=True
    )


def gap_partition(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Split P(C) - P(E_2) per separation, from the aggregate alone.

    The ``sep_*`` columns come from the pair bins, which cover exactly the
    trajectories the triple analysis covers, so the pair-level probabilities and
    the triple-level counts share one denominator. The identity
    ``P(C) - P(E_2) = P(C, !E_2) - P(!C, E_2)`` is exact; ``identity_residual``
    reports it, and should be at float precision.
    """
    anchors = aggregate.loc[aggregate["anchor_role"] == "anchor"]
    per_sep = anchors.groupby("separation").agg(
        d=("d", "first"),
        records=("sep_pair_records", "first"),
        connected=("sep_connected", "first"),
        entangled=("sep_entangled", "first"),
        connected_zero=("sep_connected_zero", "first"),
        disconnected_entangled=("sep_disconnected_entangled", "first"),
        analyzed=("sep_connected_zero_analyzed", "first"),
        with_r3=("sep_connected_zero_with_positive_third", "first"),
        undetermined=("pairs_undetermined", "sum"),
    )
    records = per_sep["records"].astype(float)
    out = pd.DataFrame(index=per_sep.index)
    out["d"] = per_sep["d"]
    out["records"] = per_sep["records"]
    out["p_connected"] = per_sep["connected"] / records
    out["p_entangled"] = per_sep["entangled"] / records
    out["gap"] = out["p_connected"] - out["p_entangled"]
    out["p_connected_zero"] = per_sep["connected_zero"] / records
    out["p_connected_zero_r3"] = per_sep["with_r3"] / records
    out["p_connected_zero_no_r3"] = (
        per_sep["analyzed"] - per_sep["with_r3"] - per_sep["undetermined"]
    ) / records
    out["p_connected_zero_undetermined"] = per_sep["undetermined"] / records
    out["p_connected_zero_not_analyzed"] = (per_sep["connected_zero"] - per_sep["analyzed"]) / records
    # The eta term the two-term partition assumes away.
    out["p_disconnected_entangled"] = per_sep["disconnected_entangled"] / records
    out["identity_residual"] = out["gap"] - (
        out["p_connected_zero"] - out["p_disconnected_entangled"]
    )
    return out.reset_index()


def pair_gap_null_comparison(
    records_path: str | Path,
    *,
    chunk_records: int = 1_000_000,
    info: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Connected-zero pairs against the distance-matched disconnected null.

    A disconnected pair has no spanning path, so any |G|^2 + |F|^2 or |rho^n| it
    carries is arithmetic -- which makes the disconnected-zero records the
    empirical noise distribution of the run's own precision. Comparing the
    connected-zero records against it, separation by separation, says whether
    a "silent" connection is distinguishable from noise and whether the
    coherent-subthreshold class sits genuinely above it.

    **The null can be degenerate, and for |G|^2 + |F|^2 it usually is.** A
    graph-disconnected pair is an exact product, and the reduction preserves
    that: in the first GPU run of this analysis every disconnected-zero record
    had |G|^2 + |F|^2 exactly 0, so there was no noise distribution at all, and
    "above the null" would hold of any nonzero value. ``null_degenerate`` flags
    that case and the summary says so rather than reporting a trivial fraction.
    |rho^n| is not degenerate: its null carries the state vector's arithmetic
    tail (to ~1e-8 at fp32).

    Needs a ``channel``-detail record file (the default since 2026-09-09).
    Streams the binary. Distributions are on the fixed 20-per-decade grid the
    summaries use, so quantiles are at that resolution; exact zeros are counted
    separately.
    """
    info = info or record_file_info(records_path)
    if not info["has_flags"]:
        raise ValueError(f"{records_path} carries no connectivity flags.")
    observables = info["observables"]
    need = {"n_i", "n_j", "dnn", "fg2", "ff2", "fmn"}
    if not need <= set(observables):
        raise ValueError(
            f"{records_path} lacks {sorted(need - set(observables))}; the null comparison "
            "needs a channel-detail record file of a parity-preserving ensemble."
        )
    tolerance = float(info.get("pair_zero_tol") or 1.0e-12)
    geometries = info["geometries"]
    geometry_count = int(geometries["geometry_id"].max()) + 1
    edges = histogram_edges()
    bins = len(edges) - 1
    quantities = ("channel", "abs_rho_n")
    groups = ("connected_zero", "disconnected_zero")
    hist = {
        (g, q): np.zeros((geometry_count, bins), dtype=np.int64) for g in groups for q in quantities
    }
    zeros = {(g, q): np.zeros(geometry_count, dtype=np.int64) for g in groups for q in quantities}
    counts = {g: np.zeros(geometry_count, dtype=np.int64) for g in groups}

    for block in iter_record_chunks(records_path, chunk_records=chunk_records, info=info):
        flags = block["flags"]
        evaluated = (flags & FLAG_EVALUATED) != 0
        connected = (flags & FLAG_CONNECTED) != 0
        unentangled = np.asarray(block["fmn"]) <= tolerance
        values = {
            "channel": np.asarray(block["fg2"], dtype=float) + np.asarray(block["ff2"], dtype=float),
            "abs_rho_n": np.abs(
                np.asarray(block["dnn"], dtype=float)
                - np.asarray(block["n_i"], dtype=float) * np.asarray(block["n_j"], dtype=float)
            ),
        }
        geometry = block["geometry_id"].astype(np.int64)
        masks = {
            "connected_zero": evaluated & connected & unentangled,
            "disconnected_zero": evaluated & ~connected & unentangled,
        }
        for g, mask in masks.items():
            counts[g] += np.bincount(geometry[mask], minlength=geometry_count)
            for q, value in values.items():
                selected = value[mask]
                where = geometry[mask]
                is_zero = selected == 0.0
                zeros[(g, q)] += np.bincount(where[is_zero], minlength=geometry_count)
                positive = ~is_zero
                index = np.searchsorted(edges, selected[positive], side="right") - 1
                index = np.clip(index, 0, bins - 1)
                np.add.at(hist[(g, q)], (where[positive], index), 1)

    distances = (
        geometries.set_index("geometry_id")["d"].reindex(range(geometry_count)).to_numpy(float)
    )

    def quantile(h: np.ndarray, zero_count: int, total: int, q: float) -> float:
        if total == 0:
            return np.nan
        target = q * total
        if zero_count >= target:
            return 0.0
        cumulative = zero_count + np.cumsum(h)
        index = int(np.searchsorted(cumulative, target))
        return float(edges[min(index + 1, bins)])

    rows = []
    for geometry_id in range(geometry_count):
        if not np.isfinite(distances[geometry_id]):
            continue
        for q in quantities:
            entry = {"geometry_id": geometry_id, "d": distances[geometry_id], "quantity": q}
            cdfs = {}
            for g in groups:
                total = int(counts[g][geometry_id])
                h = hist[(g, q)][geometry_id]
                z = int(zeros[(g, q)][geometry_id])
                entry[f"{g}_records"] = total
                entry[f"{g}_zero_fraction"] = z / total if total else np.nan
                for level in (0.5, 0.9, 0.99):
                    entry[f"{g}_q{int(level * 100)}"] = quantile(h, z, total, level)
                occupied = np.flatnonzero(h)
                entry[f"{g}_max"] = float(edges[occupied[-1] + 1]) if occupied.size else 0.0
                cdfs[g] = (z + np.cumsum(h)) / total if total else None
            # Kolmogorov-Smirnov distance between the two distributions, at the
            # grid's resolution.
            if cdfs["connected_zero"] is not None and cdfs["disconnected_zero"] is not None:
                entry["ks_distance"] = float(
                    np.max(np.abs(cdfs["connected_zero"] - cdfs["disconnected_zero"]))
                )
                null_max = entry["disconnected_zero_max"]
                h = hist[("connected_zero", q)][geometry_id]
                above = np.searchsorted(edges, null_max, side="left")
                entry["connected_zero_above_null_max"] = (
                    float(h[above:].sum()) / entry["connected_zero_records"]
                    if entry["connected_zero_records"]
                    else np.nan
                )
            else:
                entry["ks_distance"] = np.nan
                entry["connected_zero_above_null_max"] = np.nan
            # Every null record exactly zero: no distribution to compare with.
            entry["null_degenerate"] = bool(entry["disconnected_zero_zero_fraction"] == 1.0)
            rows.append(entry)
    return pd.DataFrame(rows)


def dist_pair_gap(
    files: Sequence[str | Path] | str | Path | None = None,
    *,
    file_glob: str | Path | None = None,
    records: str | Path | None = None,
    bootstrap: int = 400,
    seed: int = 0,
    chunksize: int = 500_000,
    figsize: tuple[float, float] | None = None,
    dpi: int = 130,
    title: str | None = None,
    show_summary: bool = True,
    show: bool = True,
) -> dict[str, Any]:
    """Where does a connected pair's missing entanglement go?

    ``files`` are outcome CSVs (``*_connected_zero_thirds.csv``) or their
    aggregates; each is paired with the other by name. ``records`` optionally
    names the run's ``*_records.bin`` for the null comparison panel.

    Panels:

    1. **R_3(d)** by pair class, anchors (solid) against distance-matched
       disconnected controls (dashed), with trajectory-clustered 68% bands.
    2. **The partition** of P(C) - P(E_2): connected-zero pairs with a positive
       third site, without one, and undetermined, with the eta term drawn
       separately.
    3. **Positive thirds per pair**: how many of the L-2 added sites make the
       triple fGMN-positive, for pairs that have at least one.
    4. **The blocking cut**: for each outcome, which one-versus-rest cut is the
       smallest -- i, j, or the added site k -- which is the bipartition that
       keeps fGMN at zero.
    """
    paths = _resolve_files(files, file_glob, description="pair-gap outputs")
    outcome_paths: list[Path] = []
    for path in paths:
        outcome = outcome_path_for(path) if path.stem.endswith("_aggregate") else path
        if outcome not in outcome_paths:
            outcome_paths.append(outcome)

    pair_tables = []
    aggregates = []
    for outcome in outcome_paths:
        aggregate_file = aggregate_path_for(outcome)
        if not aggregate_file.exists():
            raise FileNotFoundError(
                f"{aggregate_file} is missing; it is written beside the outcome CSV."
            )
        aggregate = read_pair_gap_aggregate(aggregate_file)
        if int(aggregate["run_complete"].iloc[0]) == 0:
            failed = int(aggregate["unresolved_failed_triples"].iloc[0])
            completed = int(aggregate["completed_realizations"].iloc[0])
            warnings.warn(
                f"{outcome.name}: run_complete=0 ({completed} trajectories, {failed} unresolved "
                "solver failures). Pairs with failed thirds are reported as undetermined.",
                stacklevel=2,
            )
        table = pair_level_table(outcome, chunksize=chunksize)
        table["L"] = int(aggregate["N"].iloc[0])
        table["source"] = outcome.name
        pair_tables.append(table)
        aggregate["source"] = outcome.name
        aggregates.append(aggregate)

    pairs = pd.concat(pair_tables, ignore_index=True)
    aggregate = pd.concat(aggregates, ignore_index=True)
    r3 = (
        pd.concat(
            [
                r3_bootstrap(group, resamples=bootstrap, seed=seed).assign(L=size)
                for size, group in pairs.groupby("L")
            ],
            ignore_index=True,
        )
        if len(pairs)
        else pd.DataFrame()
    )
    partitions = pd.concat(
        [gap_partition(group).assign(L=int(size)) for size, group in aggregate.groupby("N")],
        ignore_index=True,
    )
    null = pair_gap_null_comparison(records) if records is not None else None

    figure, axes = plt.subplots(2, 2, figsize=figsize or (11.5, 8.5), dpi=dpi)
    r3_axis, partition_axis, count_axis, block_axis = axes.ravel()
    palette = {"occupation_correlated": "C0", "coherent_subthreshold": "C1", "silent": "C2"}

    for (role, zero_class), group in (r3.groupby(["anchor_role", "pair_class"]) if len(r3) else []):
        group = group.sort_values("d")
        style = "-" if role == "anchor" else "--"
        label = f"{zero_class}" + ("" if role == "anchor" else " (control)")
        r3_axis.plot(group["d"], group["r3"], style, marker="o", color=palette[zero_class], label=label)
        r3_axis.fill_between(
            group["d"], group["r3_lo"], group["r3_hi"], color=palette[zero_class], alpha=0.15, linewidth=0
        )
    r3_axis.set_xlabel("$d$")
    r3_axis.set_ylabel(r"$R_3(d)$")
    r3_axis.set_ylim(-0.02, 1.02)
    r3_axis.set_title(r"$P(\exists k:\ \mathrm{fGMN}_{ijk}>\epsilon_3 \mid C, \bar E_2)$", fontsize=9)

    for size, part in partitions.groupby("L"):
        part = part.sort_values("d")
        bottom = np.zeros(len(part))
        for column, color, label in (
            ("p_connected_zero_r3", "C3", r"$C,\bar E_2,R_3$"),
            ("p_connected_zero_no_r3", "C7", r"$C,\bar E_2,\bar R_3$"),
            ("p_connected_zero_undetermined", "C8", "undetermined"),
            ("p_connected_zero_not_analyzed", "C9", "not analysed"),
        ):
            values = part[column].to_numpy(float)
            partition_axis.bar(part["d"], values, bottom=bottom, width=0.12, color=color,
                               label=label if size == partitions["L"].min() else "_nolegend_")
            bottom += values
        partition_axis.plot(part["d"], part["gap"], "k.-", label=r"$P(C)-P(E_2)$")
        partition_axis.plot(part["d"], part["p_disconnected_entangled"], "kx:",
                            label=r"$P(\bar C, E_2)$ ($\eta$ term)")
    partition_axis.set_xlabel("$d$")
    partition_axis.set_ylabel("probability per pair record")
    partition_axis.set_title("partition of the connected-vs-entangled gap", fontsize=9)

    positive = pairs.loc[pairs["r3"]]
    if len(positive):
        largest = int(pairs["thirds"].max())
        counts = np.bincount(positive["positive_thirds"].to_numpy(int), minlength=largest + 1)
        count_axis.bar(np.arange(len(counts)), counts, color="C3")
    count_axis.set_xlabel("fGMN-positive third sites per pair")
    count_axis.set_ylabel("pairs")
    count_axis.set_title("how many k reveal tripartite entanglement", fontsize=9)

    block_totals = pairs.groupby("pair_class", observed=True)[["blocked_by_i", "blocked_by_j", "blocked_by_k"]].sum()
    if len(block_totals):
        positions = np.arange(len(block_totals))
        width = 0.25
        for offset, column, label in ((-width, "blocked_by_i", "cut $i|jk$"),
                                      (0.0, "blocked_by_j", "cut $j|ik$"),
                                      (width, "blocked_by_k", "cut $k|ij$")):
            block_axis.bar(positions + offset, block_totals[column], width=width, label=label)
        block_axis.set_xticks(positions)
        block_axis.set_xticklabels([str(name) for name in block_totals.index], fontsize=8)
    block_axis.set_ylabel("outcomes")
    block_axis.set_title("the smallest one-vs-rest cut (what blocks fGMN)", fontsize=9)

    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    figure.suptitle(title or "Connected-zero pairs: the three-site test")
    _show(figure, show)

    if show_summary:
        _print_summary(r3, partitions, pairs, null)

    return {
        "figure": figure,
        "axes": axes,
        "pairs": pairs,
        "r3": r3,
        "partition": partitions,
        "aggregate": aggregate,
        "null": null,
    }


def _print_summary(
    r3: pd.DataFrame, partitions: pd.DataFrame, pairs: pd.DataFrame, null: pd.DataFrame | None
) -> None:
    print("Connected-zero triple analysis:")
    if len(r3):
        for _, row in r3.iterrows():
            print(
                f"  {row['anchor_role']:7s} {row['pair_class']:22s} d={row['d']:6.3f}  "
                f"R3 = {row['r3']:.4f} +/- {row['r3_stderr']:.4f}  "
                f"({row['r3_pairs']}/{row['pairs']} pairs over {row['trajectories']} trajectories"
                + (f", {row['undetermined_pairs']} undetermined" if row["undetermined_pairs"] else "")
                + ")"
            )
    worst_identity = float(np.nanmax(np.abs(partitions["identity_residual"]))) if len(partitions) else 0.0
    worst_eta = float(np.nanmax(partitions["p_disconnected_entangled"])) if len(partitions) else 0.0
    print(
        f"  partition identity residual (should be float noise): {worst_identity:.1e}; "
        f"largest eta term P(!C, E2): {worst_eta:.3g}"
    )
    residual = float(pairs["max_marginal_residual"].max()) if "max_marginal_residual" in pairs else np.nan
    print(
        f"  largest |Tr_k rho_ijk - rho_ij| over the anchors: {residual:.2e} "
        "(the fermionic sign-ordering check: the pair RDM and the triple come from independent reductions)"
    )
    if null is not None and len(null):
        for quantity, label in (("channel", "|G|^2+|F|^2"), ("abs_rho_n", "|rho^n|")):
            rows = null.loc[null["quantity"] == quantity]
            if rows.empty:
                continue
            if bool(rows["null_degenerate"].all()):
                # Disconnected pairs are exact products, so the reduction can
                # return an exact zero for every one of them. Then there is no
                # noise distribution to compare against, and "above the null"
                # would be true of any nonzero value at all.
                print(
                    f"  null comparison, {label}: DEGENERATE -- every disconnected-zero record "
                    "is exactly 0, so the disconnected pairs give no noise distribution for "
                    f"this quantity. Connected-zero median {rows['connected_zero_q50'].median():.2e}, "
                    f"max {rows['connected_zero_max'].max():.2e}."
                )
                continue
            print(
                f"  null comparison, {label}: KS distance {rows['ks_distance'].min():.3f}.."
                f"{rows['ks_distance'].max():.3f}; null maximum up to "
                f"{rows['disconnected_zero_max'].max():.2e}; connected-zero records above it "
                f"{rows['connected_zero_above_null_max'].min():.3g}.."
                f"{rows['connected_zero_above_null_max'].max():.3g}"
            )

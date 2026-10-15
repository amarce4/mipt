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
    FLAG_FN_POSITIVE,
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
    "fgmn_class",
    "fgmn_status",
    "fgmn_lower_bound",
    "fgmn_upper_bound",
    "fgmn_input_uncertainty",
    "min_cut_site",
    "min_cut",
    "marginal_residual_ij",
)

_GROUP_KEYS = ["realization_id", "anchor_role", "pair_i", "pair_j"]

# The gate-path columns gap format v3 added (MIPT_DIST_GATE_DIAGNOSTICS), in
# file order, with the names this module uses for them -- the record binary's
# observable names, so a pair table and a record frame line up. They are
# optional on read: a v2 file has none of them and still loads, with the
# columns present and NaN. NaN also means "no path" or "diagnostics off" in a
# v3 file; it is never a zero.
_GATE_PATH_COLUMNS = {
    "path_unitary_count_ij": "shortest_path_unitary_count",
    "shortest_maximin_gate_ep_ij": "shortest_maximin_gate_ep",
    "widest_gate_ep_ij": "widest_gate_ep",
    "widest_even_mix_ij": "widest_even_mix",
    "widest_odd_mix_ij": "widest_odd_mix",
}


def _ensure_gate_path_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename the gate-path columns if present, add them as NaN if not."""
    frame = frame.rename(columns=_GATE_PATH_COLUMNS)
    for name in _GATE_PATH_COLUMNS.values():
        if name not in frame.columns:
            frame[name] = np.nan
        else:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def aggregate_path_for(outcome_csv: str | Path) -> Path:
    path = Path(outcome_csv)
    return path.with_name(path.stem + "_aggregate.csv")


def four_site_path_for(outcome_csv: str | Path) -> Path:
    """The four-site CSV beside an outcome CSV, when the pass was enabled."""
    path = Path(outcome_csv)
    stem = path.stem
    suffix = "_connected_zero_thirds"
    if stem.endswith(suffix):
        return path.with_name(stem[: -len(suffix)] + "_connected_zero_four_site.csv")
    return path.with_name(stem + "_four_site.csv")


def read_four_site(path: str | Path) -> pd.DataFrame:
    """The four-site CSV, as written.

    One row per (anchor, helper pair). ``globally_entangled_locally_separable``
    is the four-mode GHZ signature: entangled across all seven cuts while every
    pair and triple marginal is separable. ``assisted_parity_max`` is the
    localizable endpoint negativity under a joint parity-sector measurement of
    the two helpers -- the family that can localize a Bell pair out of a state
    with no endpoint entanglement at all, which the single-mode occupation
    family provably cannot.
    """
    return pd.read_csv(path)


def pair_summary_path_for(outcome_csv: str | Path) -> Path:
    """The per-pair summary beside an outcome CSV.

    ``<main>_connected_zero_thirds.csv`` -> ``<main>_connected_zero_pair_summary.csv``,
    matching ``gap::pair_summary_path_for`` in dist_pair_gap.hpp.
    """
    path = Path(outcome_csv)
    stem = path.stem
    suffix = "_connected_zero_thirds"
    if stem.endswith(suffix):
        return path.with_name(stem[: -len(suffix)] + "_connected_zero_pair_summary.csv")
    return path.with_name(stem + "_pair_summary.csv")


# The summary's own columns, renamed to the names the rest of this module uses.
# The C++ writes one row per qualifying pair already reduced, so reading it is
# the same reduction without the streaming -- and on a production run that is a
# few megabytes against a few gigabytes.
_SUMMARY_RENAME = {
    "d": "d",
    "thirds": "thirds",
    "thirds_positive": "positive_thirds",
    "thirds_failed": "failed_thirds",
    "thirds_unresolved": "unresolved_thirds",
    "thirds_bounded_below_threshold": "bounded_thirds",
    "thirds_all_cuts_entangled": "all_cuts_entangled_thirds",
    "thirds_same_component": "same_component_thirds",
    "thirds_blocked_by_i": "blocked_by_i",
    "backbone_nodes": "backbone_nodes",
    "dangling_nodes": "dangling_nodes",
    "articulation_nodes": "articulation_nodes",
    "component_size_ij": "component_size",
    "f_channel_log_weight": "f_channel_log_weight",
    "g_channel_log_weight": "g_channel_log_weight",
    "f_channel_bottleneck": "f_channel_bottleneck",
    "g_channel_bottleneck": "g_channel_bottleneck",
    "thirds_blocked_by_j": "blocked_by_j",
    "thirds_blocked_by_k": "blocked_by_k",
    "max_fgmn_lower_bound": "max_fgmn",
    "max_fgmn_upper_bound": "max_fgmn_upper",
    "max_min_cut_fn": "max_min_cut",
    "max_cmi_ij_given_k": "max_cmi",
    "explanatory_class": "explanatory_class",
}


def read_pair_summary(path: str | Path) -> pd.DataFrame:
    """The per-pair summary CSV, in this module's column vocabulary.

    One row per ``(realization_id, anchor_role, pair_i, pair_j)``, already
    carrying the certified four-way split of its thirds and the single
    explanatory class. ``r3`` and ``undetermined`` are derived here so a table
    from this path and one reduced from the outcome rows are interchangeable.
    """
    frame = pd.read_csv(path)
    out = _ensure_gate_path_columns(frame.rename(columns=_SUMMARY_RENAME))
    out["r3"] = out["positive_thirds"] > 0
    # No positive third, but something the solver could not resolve: the pair's
    # R_3 is unknown and it belongs in neither the numerator nor the
    # denominator. Unresolved counts here as well as failed -- that is the
    # whole point of the certified classification.
    out["undetermined"] = (~out["r3"]) & (
        (out["failed_thirds"] > 0) | (out["unresolved_thirds"] > 0)
    )
    out["prefiltered_thirds"] = out["bounded_thirds"]
    return out


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
    # A certified interval that straddles the threshold resolves nothing, so it
    # keeps its pair out of the R_3 denominator exactly as a failed solve does.
    # Gap format v1 had no such column; there every non-failure was resolved.
    unresolved = (
        (frame["fgmn_class"] == "unresolved")
        if "fgmn_class" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    work = frame.assign(
        _failed=failed.astype(np.int64),
        _unresolved=unresolved.astype(np.int64),
        _positive=frame["fgmn_positive"].astype(np.int64),
        _prefiltered=(frame["fgmn_method"] == "prefilter_bound").astype(np.int64),
        _block_i=(frame["min_cut_site"] == "i").astype(np.int64),
        _block_j=(frame["min_cut_site"] == "j").astype(np.int64),
        _block_k=(frame["min_cut_site"] == "k").astype(np.int64),
    )
    work = _ensure_gate_path_columns(work)
    grouped = work.groupby(_GROUP_KEYS, sort=False, observed=True)
    gate_aggregations = {name: (name, "first") for name in _GATE_PATH_COLUMNS.values()}
    out = grouped.agg(
        separation=("separation", "first"),
        d=("chord", "first"),
        pair_class=("pair_class", "first"),
        thirds=("third_k", "size"),
        positive_thirds=("_positive", "sum"),
        failed_thirds=("_failed", "sum"),
        unresolved_thirds=("_unresolved", "sum"),
        prefiltered_thirds=("_prefiltered", "sum"),
        max_fgmn=("fgmn", "max"),
        mean_fgmn=("fgmn", "mean"),
        max_marginal_residual=("marginal_residual_ij", "max"),
        blocked_by_i=("_block_i", "sum"),
        blocked_by_j=("_block_j", "sum"),
        blocked_by_k=("_block_k", "sum"),
        # A property of the anchor pair, identical on its L-2 rows.
        **gate_aggregations,
    ).reset_index()
    out["r3"] = out["positive_thirds"] > 0
    # No positive k, but at least one k whose solve failed: the pair's R_3 is
    # unknown, and it is kept out of both the numerator and the denominator.
    out["undetermined"] = (~out["r3"]) & (
        (out["failed_thirds"] > 0) | (out["unresolved_thirds"] > 0)
    )
    return out


def pair_level_table(
    outcome_csv: str | Path,
    *,
    chunksize: int = 500_000,
    use_summary: bool = True,
) -> pd.DataFrame:
    """One row per (trajectory, role, pair).

    Prefers the per-pair summary written beside the outcome CSV, which is the
    same reduction performed by the C++ as the rows were written -- a few
    megabytes against the outcome file's few gigabytes. Falls back to streaming
    the outcome CSV for a run written before the summary existed, or when
    ``use_summary=False``.

    Streaming: rows of one pair are written consecutively, so a group can only
    straddle a chunk boundary at its end; the trailing group of each chunk is
    carried into the next. The result is small -- one row per qualifying pair --
    whatever the size of the file.
    """
    if use_summary:
        summary = pair_summary_path_for(outcome_csv)
        if summary.exists():
            return read_pair_summary(summary)
    columns = list(_OUTCOME_COLUMNS)
    # The gate-path columns only exist from gap format v3 on; read them when the
    # file has them, and let _reduce_groups fill NaN when it does not.
    available = set(pd.read_csv(outcome_csv, nrows=0).columns)
    columns += [name for name in _GATE_PATH_COLUMNS if name in available]
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


# ---------------------------------------------------------------------------
# Gate-path diagnostics (MIPT_DIST_GATE_DIAGNOSTICS=1)
#
# The question: is a connected-but-unentangled ("gap") pair one whose every
# spacetime path runs through a nearly non-entangling gate? The record binary
# then carries, per pair, the widest-path and shortest-maximin values of the
# gates' normalized entangling power and the widest values of the two
# parity-block mixings. A *marginal* comparison of those between gap pairs and
# entangled connected pairs is confounded by geometry -- both depend strongly on
# separation and on how large the component is -- so every gap pair is compared
# against an entangled connected pair from the *same trajectory*, matched on
# geometry first and then on the graph's own shortest path and component size.
# What remains of the difference is then attributable to the gates.
#
# Everything here streams the binary one trajectory at a time. A trajectory's
# records are contiguous and in realization order (the writer flushes only at
# trajectory boundaries), which is what lets the matching -- a strictly
# within-trajectory operation -- run with memory bounded by one trajectory plus
# the gap pairs kept for the clustered bootstrap.
# ---------------------------------------------------------------------------

GATE_OBSERVABLES = (
    "component_size",
    "shortest_path",
    "shortest_path_unitary_count",
    "shortest_maximin_gate_ep",
    "widest_gate_ep",
    "widest_even_mix",
    "widest_odd_mix",
)

# The four capacities compared between groups. All live in [0, 1].
GATE_COMPARISON_METRICS = (
    "widest_gate_ep",
    "shortest_maximin_gate_ep",
    "widest_even_mix",
    "widest_odd_mix",
)

# How a gap pair's control was found, best first. `none` means the trajectory
# held no entangled connected pair at all; such a gap pair still counts in the
# group statistics but has no matched difference.
MATCH_LEVELS = ("geometry_shortest_path", "geometry", "nearest_separation", "none")

_GATE_DIAGNOSTICS_HINT = (
    "it was written without the gate-path observables. Rerun dist_scaling.exe "
    "(circ_type 2 or 4, k=2 or k=0) with MIPT_DIST_GATE_DIAGNOSTICS=1."
)


def default_gate_path_thresholds() -> np.ndarray:
    """``logspace(-6, 0, 13)`` together with 0.05, 0.1 and 0.2, sorted."""
    return np.unique(np.concatenate([np.logspace(-6.0, 0.0, 13), [0.05, 0.1, 0.2]]))


def gate_path_output_path(records: str | Path, suffix: str) -> Path:
    """``<stem><suffix>`` beside a ``<stem>_records.bin``."""
    path = Path(records)
    stem = path.with_suffix("")
    name = stem.name
    if name.endswith("_records"):
        name = name[: -len("_records")]
    return stem.with_name(name + suffix)


def _header_float(header: Mapping[str, Any], key: str) -> float:
    value = header.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def gate_record_context(records: str | Path, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a record file for the gate-path analyses and collect what they need.

    Refuses -- naming the setting that produces them -- a file without the
    gate-path observables, and any file that is not a format v2 k=2 file with
    connectivity flags, since the gap/control split is read off those flags.
    """
    info = info or record_file_info(records)
    if int(info["k"]) != 2:
        raise ValueError(f"{records}: gate-path diagnostics are per pair; this is a k={info['k']} file.")
    if not info["has_flags"]:
        raise ValueError(
            f"{records}: no connectivity flags (format v1, or MIPT_DIST_CONNECTIVITY=0), so "
            "connected pairs cannot be identified."
        )
    observables = set(info["observables"])
    missing = [name for name in GATE_OBSERVABLES if name not in observables]
    if missing:
        raise ValueError(f"{records}: no {', '.join(missing)} -- {_GATE_DIAGNOSTICS_HINT}")
    fermionic = "fmn" in observables
    entanglement = "fmn" if fermionic else "mn"
    information = "fmi" if fermionic else "mi"
    # The zero class needs the occupations and the channel weight, i.e. a
    # `channel`- or `full`-detail file; the channel weight is the fermionic
    # one wherever it exists, exactly as PairProtocol classifies.
    channel = ("fg2", "ff2") if fermionic else ("g2", "f2")
    classifiable = {"n_i", "n_j", "dnn", *channel} <= observables
    geometries = info["geometries"]
    geometry_count = int(geometries["geometry_id"].max()) + 1
    separation = np.full(geometry_count, np.nan)
    separation_column = "separation" if "separation" in geometries.columns else "d"
    separation[geometries["geometry_id"].to_numpy(dtype=int)] = geometries[
        separation_column
    ].to_numpy(dtype=float)
    distance = np.full(geometry_count, np.nan)
    distance[geometries["geometry_id"].to_numpy(dtype=int)] = geometries["d"].to_numpy(dtype=float)
    header = info["header"]
    return {
        "info": info,
        "entanglement": entanglement,
        "information": information,
        "channel": channel,
        "classifiable": classifiable,
        "separation": separation,
        "distance": distance,
        "occupation_mi_tol": _header_float(header, "occupation_mi_tol"),
        "channel_floor": _header_float(header, "channel_floor"),
    }


def iter_realization_blocks(
    records: str | Path,
    *,
    info: dict[str, Any],
    chunk_records: int = 1_000_000,
):
    """Yield ``(realization_id, block)`` with every record of one trajectory.

    Streams the binary in chunks and carries a trajectory that straddles a
    chunk boundary into the next one, so a block is always whole. Refuses a
    file whose realization ids ever decrease: the within-trajectory matching
    would silently split a trajectory in two.
    """
    carry: np.ndarray | None = None
    previous = -1
    for chunk in iter_record_chunks(records, chunk_records=chunk_records, info=info):
        if carry is not None and carry.size:
            chunk = np.concatenate([carry, chunk])
        ids = chunk["realization_id"].astype(np.int64)
        if ids.size == 0:
            continue
        if ids[0] < previous or np.any(np.diff(ids) < 0):
            raise ValueError(
                f"{records}: realization ids are not in nondecreasing order, so a trajectory's "
                "records are not contiguous and cannot be matched within it."
            )
        starts = np.concatenate([[0], np.flatnonzero(np.diff(ids)) + 1])
        for begin, end in zip(starts[:-1], starts[1:]):
            yield int(ids[begin]), chunk[begin:end]
        carry = chunk[starts[-1] :].copy()
        previous = int(ids[starts[-1]])
    if carry is not None and carry.size:
        yield int(carry["realization_id"][0]), carry


def zero_classes(block: np.ndarray, context: Mapping[str, Any]) -> np.ndarray:
    """``classify_unentangled`` (dist_scaling_csv.hpp) on a record block.

    occupation_correlated when I_occ > occupation_mi_tol, else
    coherent_subthreshold when the channel weight |G|^2 + |F|^2 exceeds the
    channel floor, else silent -- both comparisons strict, so a value exactly at
    its tolerance falls into the lower class, as in the C++. The thresholds are
    the file's own (header ``occupation_mi_tol`` / ``channel_floor``). Returns
    ``"unknown"`` everywhere for a file without the channel observables.
    """
    out = np.full(block.size, "unknown", dtype=object)
    if not context["classifiable"]:
        return out
    from .records import _occupation_mutual_information

    i_occ = _occupation_mutual_information(
        np.asarray(block["n_i"], dtype=float),
        np.asarray(block["n_j"], dtype=float),
        np.asarray(block["dnn"], dtype=float),
    )
    first, second = context["channel"]
    weight = np.asarray(block[first], dtype=float) + np.asarray(block[second], dtype=float)
    occupation = i_occ > context["occupation_mi_tol"]
    coherent = (~occupation) & (weight > context["channel_floor"])
    out[:] = "silent"
    out[coherent] = "coherent_subthreshold"
    out[occupation] = "occupation_correlated"
    return out


def rank_controls(
    gap: np.void,
    controls: np.ndarray,
    separation: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Order a trajectory's controls by how well they match one gap pair.

    Key, most significant first: same geometry before any other; among other
    geometries the nearest separation; then an equal shortest path (NaN equals
    nothing); then the nearest component size; then a seeded random tiebreak.
    Returns the ordering and each ordered control's match level.
    """
    if controls.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=object)
    geometry = controls["geometry_id"].astype(np.int64)
    same = geometry == int(gap["geometry_id"])
    separation_gap = abs(separation[geometry] - separation[int(gap["geometry_id"])])
    separation_gap = np.where(same, 0.0, np.nan_to_num(separation_gap, nan=np.inf))
    path = np.asarray(controls["shortest_path"], dtype=float)
    path_mismatch = ~(path == float(gap["shortest_path"]))
    size_gap = np.abs(np.asarray(controls["component_size"], dtype=float) - float(gap["component_size"]))
    size_gap = np.nan_to_num(size_gap, nan=np.inf)
    tiebreak = rng.random(controls.size)
    order = np.lexsort((tiebreak, size_gap, path_mismatch, separation_gap, ~same))
    level = np.where(
        same[order],
        np.where(path_mismatch[order], "geometry", "geometry_shortest_path"),
        "nearest_separation",
    ).astype(object)
    return order, level


def _clustered_draws(clusters: int, n_boot: int, seed: int, budget: int = 20_000_000):
    """Poisson(1) weights per cluster, in batches of draws.

    The Poisson bootstrap resamples trajectories independently, so a statistic
    that is a ratio of sums over records (a fraction, a mean, a quantile) only
    sees the weights of trajectories that contribute to it -- which is what
    makes it safe to draw weights for the trajectories at hand rather than for
    the whole file. Yields ``(batch_slice, weights)``, weights shaped
    ``(batch, clusters)``. The same ``(clusters, seed)`` gives the same weights,
    so statistics of different groups drawn with one seed share their resamples.
    """
    clusters = max(int(clusters), 1)
    rng = np.random.default_rng(seed)
    batch = int(max(1, min(n_boot, budget // max(clusters, 1))))
    done = 0
    while done < n_boot:
        take = min(batch, n_boot - done)
        yield slice(done, done + take), rng.poisson(1.0, size=(take, clusters)).astype(float)
        done += take


def _weighted_quantile_draws(
    values: np.ndarray, cluster: np.ndarray, clusters: int, q: float, n_boot: int, seed: int
) -> np.ndarray:
    """Clustered-bootstrap draws of the q-quantile of ``values`` (finite only)."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_cluster = cluster[order]
    draws = np.full(n_boot, np.nan)
    budget = max(1, 5_000_000 // max(values.size, 1))
    for where, weights in _clustered_draws(clusters, n_boot, seed):
        for start in range(0, weights.shape[0], budget):
            w = weights[start : start + budget][:, sorted_cluster]
            cumulative = np.cumsum(w, axis=1)
            total = cumulative[:, -1]
            target = q * total
            index = np.array(
                [np.searchsorted(row, t, side="left") for row, t in zip(cumulative, target)]
            )
            valid = total > 0
            out = np.full(w.shape[0], np.nan)
            out[valid] = sorted_values[np.minimum(index[valid], values.size - 1)]
            draws[where.start + start : where.start + start + w.shape[0]] = out
    return draws


def _ratio_draws(
    numerator: np.ndarray,
    denominator: np.ndarray,
    cluster: np.ndarray,
    clusters: int,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Clustered-bootstrap draws of sum(numerator) / sum(denominator).

    ``numerator`` may be 2-D (records x columns); per-record contributions are
    first reduced per cluster, so the cost per draw is the trajectory count.
    """
    clusters = max(int(clusters), 1)
    numerator = np.asarray(numerator, dtype=float)
    two_d = numerator.ndim == 2
    num = numerator if two_d else numerator[:, None]
    per_num = np.zeros((clusters, num.shape[1]))
    np.add.at(per_num, cluster, num)
    per_den = np.bincount(cluster, weights=np.asarray(denominator, dtype=float), minlength=clusters)
    draws = np.full((n_boot, num.shape[1]), np.nan)
    for where, weights in _clustered_draws(clusters, n_boot, seed):
        top = weights @ per_num
        bottom = weights @ per_den
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[where] = top / bottom[:, None]
    return draws if two_d else draws[:, 0]


def _interval(draws: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


# Log grid for the quantiles of the *all-controls* group, which is too large to
# keep: an exact-zero bin, then 100 bins per decade from 1e-16 to 1 (values are
# capacities in [0, 1]). Quantiles of that group are therefore at 2.3% relative
# resolution, reported as the geometric centre of the bin they fall in.
_CONTROL_GRID = np.logspace(-16.0, 0.0, 1601)


def _grid_bin(values: np.ndarray) -> np.ndarray:
    """Bin index on _CONTROL_GRID: 0 for an exact zero, 1.. for the log bins."""
    index = np.searchsorted(_CONTROL_GRID, values, side="right")
    index = np.clip(index, 1, _CONTROL_GRID.size - 1)
    return np.where(values <= 0.0, 0, index)


def _grid_quantile(histogram: np.ndarray, q: float) -> float:
    total = histogram.sum()
    if total == 0:
        return float("nan")
    cumulative = np.cumsum(histogram)
    index = int(np.searchsorted(cumulative, q * total, side="left"))
    if index == 0:
        return 0.0
    return float(np.sqrt(_CONTROL_GRID[index - 1] * _CONTROL_GRID[index]))


def gate_path_gap_test(
    records: str | Path,
    *,
    thresholds: Sequence[float] | np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    chunk_records: int = 1_000_000,
    write: bool = True,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Are gap pairs connected only through weak gates? A matched comparison.

    Groups, from the record flags:

    * ``gap`` -- ``connected`` and not ``fn_positive``: graph-connected pairs
      with no (fermionic) negativity above the run's ``pair_zero_tol``. Also
      split by zero class into ``gap_silent``, ``gap_coherent_subthreshold`` and
      ``gap_occupation_correlated``, with the C++ ``classify_unentangled`` rule
      and the file's own ``occupation_mi_tol`` / ``channel_floor``.
    * ``control`` -- ``connected`` and ``fn_positive``: every entangled
      connected pair.
    * ``control_matched`` -- the one control matched to each gap pair: same
      trajectory always; same geometry, equal shortest path and nearest
      component size when possible, then same geometry, then the nearest
      separation (``match_level`` records which). Controls are drawn with
      replacement and ties are broken by a generator seeded from
      ``(seed, realization_id)``, so the result does not depend on the chunking.

    For each of ``widest_gate_ep``, ``shortest_maximin_gate_ep``,
    ``widest_even_mix`` and ``widest_odd_mix``: per-group counts, medians and
    5th/95th percentiles; the fraction strictly below each threshold
    (``thresholds``, default :func:`default_gate_path_thresholds`); and the
    matched differences gap - control (mean and median). Every CI is a 95%
    trajectory-clustered Poisson bootstrap over realization ids with ``n_boot``
    draws. **NaN is never counted as zero**: a pair whose value is NaN (no path)
    is dropped from that metric's statistics, and a matched difference needs
    both values finite. The ``control`` group is too large to keep, so its
    quantiles come from a fixed log grid (see ``_CONTROL_GRID``) and its
    fractions from streaming bootstrap sums; every other group is exact.

    Streams the record binary one trajectory at a time and never loads it
    whole; memory is one trajectory plus ~100 bytes per gap pair.

    Writes ``<stem>_gate_path_summary.csv`` (tidy long:
    ``metric,statistic,group,threshold,value,ci_low,ci_high,n``) unless
    ``write=False``. Returns ``{"summary", "pairs", "match_levels", "path",
    "context"}``: ``pairs`` has one row per gap pair with its matched control's
    identifiers and values.
    """
    context = gate_record_context(records)
    info = context["info"]
    thresholds = (
        default_gate_path_thresholds()
        if thresholds is None
        else np.unique(np.asarray(list(thresholds), dtype=float))
    )
    metrics = GATE_COMPARISON_METRICS
    m_count, t_count = len(metrics), thresholds.size
    separation = context["separation"]
    entanglement = context["entanglement"]

    # The all-controls group, reduced as it streams.
    control_count = np.zeros(m_count, dtype=np.int64)
    control_hist = np.zeros((m_count, _CONTROL_GRID.size), dtype=np.int64)
    control_below = np.zeros((m_count, t_count), dtype=np.int64)
    control_boot_below = np.zeros((n_boot, m_count, t_count))
    control_boot_finite = np.zeros((n_boot, m_count))
    boot_rng = np.random.default_rng([seed, 2])

    rows: dict[str, list[Any]] = {
        name: []
        for name in (
            "realization_id",
            "geometry_id",
            "embedding_id",
            "separation",
            "d",
            "zero_class",
            "match_level",
            "shortest_path",
            "component_size",
            entanglement,
            "control_geometry_id",
            "control_embedding_id",
            "control_shortest_path",
            "control_component_size",
            "control_" + entanglement,
        )
    }
    for metric in metrics:
        rows[metric] = []
        rows["control_" + metric] = []
    trajectories = 0

    for realization, block in iter_realization_blocks(
        records, info=info, chunk_records=chunk_records
    ):
        trajectories += 1
        flags = block["flags"]
        connected = (flags & FLAG_CONNECTED) != 0
        positive = (flags & FLAG_FN_POSITIVE) != 0
        gaps = block[connected & ~positive]
        controls = block[connected & positive]

        if controls.size:
            finite_now = np.zeros(m_count)
            below_now = np.zeros((m_count, t_count))
            for position, metric in enumerate(metrics):
                values = np.asarray(controls[metric], dtype=float)
                values = values[np.isfinite(values)]
                control_count[position] += values.size
                np.add.at(control_hist[position], _grid_bin(values), 1)
                below = (values[:, None] < thresholds[None, :]).sum(axis=0)
                control_below[position] += below
                finite_now[position] = values.size
                below_now[position] = below
            weights = boot_rng.poisson(1.0, size=n_boot).astype(float)
            control_boot_finite += weights[:, None] * finite_now[None, :]
            control_boot_below += weights[:, None, None] * below_now[None, :, :]

        if gaps.size == 0:
            continue
        classes = zero_classes(gaps, context)
        rng = np.random.default_rng([seed, realization])
        for index in range(gaps.size):
            gap = gaps[index]
            order, levels = rank_controls(gap, controls, separation, rng)
            geometry = int(gap["geometry_id"])
            rows["realization_id"].append(realization)
            rows["geometry_id"].append(geometry)
            rows["embedding_id"].append(int(gap["embedding_id"]))
            rows["separation"].append(separation[geometry])
            rows["d"].append(context["distance"][geometry])
            rows["zero_class"].append(classes[index])
            rows["shortest_path"].append(float(gap["shortest_path"]))
            rows["component_size"].append(float(gap["component_size"]))
            rows[entanglement].append(float(gap[entanglement]))
            for metric in metrics:
                rows[metric].append(float(gap[metric]))
            if order.size == 0:
                rows["match_level"].append("none")
                for name in (
                    "control_geometry_id",
                    "control_embedding_id",
                ):
                    rows[name].append(-1)
                for name in ("control_shortest_path", "control_component_size", "control_" + entanglement):
                    rows[name].append(np.nan)
                for metric in metrics:
                    rows["control_" + metric].append(np.nan)
                continue
            control = controls[order[0]]
            rows["match_level"].append(levels[0])
            rows["control_geometry_id"].append(int(control["geometry_id"]))
            rows["control_embedding_id"].append(int(control["embedding_id"]))
            rows["control_shortest_path"].append(float(control["shortest_path"]))
            rows["control_component_size"].append(float(control["component_size"]))
            rows["control_" + entanglement].append(float(control[entanglement]))
            for metric in metrics:
                rows["control_" + metric].append(float(control[metric]))

    pairs = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []

    def add(metric, statistic, group, value, ci=(np.nan, np.nan), n=0, threshold=np.nan):
        summary_rows.append(
            {
                "metric": metric,
                "statistic": statistic,
                "group": group,
                "threshold": threshold,
                "value": value,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "n": int(n),
            }
        )

    # Exact groups: every gap pair, each zero class, and the matched controls.
    cluster_ids, cluster = np.unique(pairs["realization_id"].to_numpy(dtype=np.int64), return_inverse=True)
    cluster = cluster.astype(np.int64)
    exact_groups: list[tuple[str, np.ndarray, str]] = [("gap", np.ones(len(pairs), dtype=bool), "")]
    for zero_class in ZERO_CLASSES:
        mask = (pairs["zero_class"] == zero_class).to_numpy()
        if context["classifiable"]:
            exact_groups.append((f"gap_{zero_class}", mask, ""))
    matched = (pairs["match_level"] != "none").to_numpy()
    exact_groups.append(("control_matched", matched, "control_"))

    stat_seed = seed * 7919 + 17
    for position, metric in enumerate(metrics):
        for group, mask, prefix in exact_groups:
            values = pairs[prefix + metric].to_numpy(dtype=float)
            usable = mask & np.isfinite(values)
            v = values[usable]
            c = cluster[usable]
            add(metric, "count", group, float(v.size), n=v.size)
            add(metric, "nan_count", group, float((mask & ~np.isfinite(values)).sum()), n=int(mask.sum()))
            for name, q in (("median", 0.5), ("p05", 0.05), ("p95", 0.95)):
                add(metric, name, group, float(np.quantile(v, q)) if v.size else np.nan, n=v.size)
            if v.size:
                below = (v[:, None] < thresholds[None, :]).astype(float)
                draws = _ratio_draws(below, np.ones(v.size), c, cluster_ids.size, n_boot, stat_seed)
                point = below.mean(axis=0)
            else:
                draws = np.full((n_boot, t_count), np.nan)
                point = np.full(t_count, np.nan)
            for t_index, threshold in enumerate(thresholds):
                add(
                    metric,
                    "fraction_below",
                    group,
                    float(point[t_index]),
                    _interval(draws[:, t_index]),
                    n=v.size,
                    threshold=float(threshold),
                )

        # The all-controls group, from its streamed reductions.
        n = int(control_count[position])
        add(metric, "count", "control", float(n), n=n)
        for name, q in (("median", 0.5), ("p05", 0.05), ("p95", 0.95)):
            add(metric, name, "control", _grid_quantile(control_hist[position], q), n=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = control_boot_below[:, position, :] / control_boot_finite[:, position][:, None]
        for t_index, threshold in enumerate(thresholds):
            add(
                metric,
                "fraction_below",
                "control",
                control_below[position, t_index] / n if n else np.nan,
                _interval(boot[:, t_index]),
                n=n,
                threshold=float(threshold),
            )

        # Matched differences, gap - control, overall and per zero class.
        difference = pairs[metric].to_numpy(dtype=float) - pairs["control_" + metric].to_numpy(dtype=float)
        for group, mask, prefix in exact_groups:
            if prefix:
                continue
            usable = mask & matched & np.isfinite(difference)
            v = difference[usable]
            c = cluster[usable]
            if v.size:
                mean_draws = _ratio_draws(v, np.ones(v.size), c, cluster_ids.size, n_boot, stat_seed + 1)
                median_draws = _weighted_quantile_draws(v, c, cluster_ids.size, 0.5, n_boot, stat_seed + 2)
                add(metric, "matched_mean_difference", group, float(v.mean()), _interval(mean_draws), n=v.size)
                add(
                    metric,
                    "matched_median_difference",
                    group,
                    float(np.median(v)),
                    _interval(median_draws),
                    n=v.size,
                )
            else:
                add(metric, "matched_mean_difference", group, np.nan)
                add(metric, "matched_median_difference", group, np.nan)

    match_levels = (
        pairs.groupby(["zero_class", "match_level"]).size().rename("pairs").reset_index()
        if len(pairs)
        else pd.DataFrame(columns=["zero_class", "match_level", "pairs"])
    )
    for level in MATCH_LEVELS:
        add("match", f"match_level_{level}", "gap", float((pairs["match_level"] == level).sum()), n=len(pairs))
    add("trajectories", "count", "all", float(trajectories), n=trajectories)
    add("trajectories", "count", "gap", float(cluster_ids.size), n=cluster_ids.size)

    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "statistic", "group", "threshold", "value", "ci_low", "ci_high", "n"],
    )
    path = None
    if write:
        path = Path(output) if output is not None else gate_path_output_path(records, "_gate_path_summary.csv")
        summary.to_csv(path, index=False, float_format="%.17g")
    return {
        "summary": summary,
        "pairs": pairs,
        "match_levels": match_levels,
        "path": path,
        "context": context,
    }

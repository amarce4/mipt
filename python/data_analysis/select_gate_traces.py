"""Choose the pairs whose gate activity ``rppu_gate_trace.exe`` should replay.

Stage two of the gate-path diagnostic replays individual trajectories and
traces, gate by gate, how much state-dependent entangling activity actually
happens along the paths stage one only characterised by the *gates*. That is
expensive per pair, so it runs on a sample, and this module draws it:

* ``silent`` connected-zero pairs -- ``connected`` and not ``fn_positive``, with
  no occupation correlation and no channel weight above the floor -- sampled
  **stratified by geometry**, so short and long separations are both
  represented however unequally the run produced them;
* for each, ``controls_per_gap`` entangled connected pairs from the *same
  trajectory* (group ``entangled_control``), matched exactly as
  :func:`data_analysis.pair_gap.gate_path_gap_test` matches them: same geometry,
  equal shortest path, nearest component size, then the fallbacks;
* optionally ``coherent_subthreshold`` and ``occupation_correlated`` gap pairs,
  stratified the same way and given controls the same way.

The zero class is the C++ ``classify_unentangled`` rule with the file's own
``occupation_mi_tol`` and ``channel_floor``, so it needs a ``channel``- or
``full``-detail record file written with ``MIPT_DIST_GATE_DIAGNOSTICS=1``.

Two streaming passes over the record binary, never loaded whole: the first
counts the candidates per (class, geometry), which fixes the quotas and the
exact uniform sample of ordinals within each stratum; the second picks those
records out and matches their controls inside the trajectory being read.

Output, ``<stem>_gate_trace_selection.csv`` beside ``<stem>_records.bin``, is
the replay's input (contract with ``apps/rppu_gate_trace.cpp``): one header line
exactly ::

    realization_id,pair_i,pair_j,group,geometry_id,embedding_id,fmn,fmi,shortest_path,component_size,widest_gate_ep

then one row per selected pair, sorted by realization (the order the replay
regenerates trajectories in). No comment lines. ``pair_i``/``pair_j`` come from
the header's embedding table. A pair selected twice -- one control matched to
two gap pairs -- is written once.

Command line::

    python -m data_analysis.select_gate_traces <records.bin> [--n-silent N]
        [--n-coherent N] [--n-occupation N] [--controls-per-gap K] [--seed S]
        [--output PATH] [--chunk-records N]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .pair_gap import (
    gate_path_output_path,
    gate_record_context,
    iter_realization_blocks,
    rank_controls,
    zero_classes,
)
from .records import FLAG_CONNECTED, FLAG_FN_POSITIVE

SELECTION_COLUMNS = (
    "realization_id",
    "pair_i",
    "pair_j",
    "group",
    "geometry_id",
    "embedding_id",
    "fmn",
    "fmi",
    "shortest_path",
    "component_size",
    "widest_gate_ep",
)
SELECTION_HEADER = ",".join(SELECTION_COLUMNS)
SELECTION_GROUPS = ("silent", "entangled_control", "coherent_subthreshold", "occupation_correlated")


def _gap_mask(block: np.ndarray) -> np.ndarray:
    flags = block["flags"]
    return ((flags & FLAG_CONNECTED) != 0) & ((flags & FLAG_FN_POSITIVE) == 0)


def _control_mask(block: np.ndarray) -> np.ndarray:
    flags = block["flags"]
    return ((flags & FLAG_CONNECTED) != 0) & ((flags & FLAG_FN_POSITIVE) != 0)


def allocate_quotas(counts: np.ndarray, total: int, rng: np.random.Generator) -> np.ndarray:
    """Split ``total`` as evenly as possible over strata of the given sizes.

    Water-filling: every stratum gets an equal share, a stratum smaller than its
    share gives up the rest, and the remainder is redistributed among the ones
    that can take more. The few units an even split cannot place go to a seeded
    random choice of the eligible strata. Never exceeds a stratum's size, and
    sums to ``min(total, counts.sum())``.
    """
    counts = np.asarray(counts, dtype=np.int64)
    quota = np.zeros_like(counts)
    remaining = int(min(total, counts.sum()))
    while remaining > 0:
        open_strata = np.flatnonzero(quota < counts)
        share = remaining // open_strata.size
        if share == 0:
            chosen = rng.choice(open_strata, size=remaining, replace=False)
            quota[chosen] += 1
            break
        add = np.minimum(counts[open_strata] - quota[open_strata], share)
        quota[open_strata] += add
        remaining -= int(add.sum())
    return quota


def _embedding_sites(info: dict[str, Any]) -> dict[tuple[int, int], tuple[int, int]]:
    embeddings = info["embeddings"]
    if embeddings is None or len(embeddings) == 0:
        raise ValueError(
            f"{info['path']}: no embedding table, so a record cannot be traced to its two "
            "sites. A format v2 record file carries one."
        )
    return {
        (int(row.geometry_id), int(row.embedding_id)): (int(row.site_1), int(row.site_2))
        for row in embeddings.itertuples(index=False)
    }


def _format(value: float, integer: bool = False) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    if integer:
        return str(int(round(value)))
    return format(value, ".17g")


def select_gate_traces(
    records: str | Path,
    *,
    n_silent: int = 500,
    n_coherent: int = 0,
    n_occupation: int = 0,
    controls_per_gap: int = 1,
    seed: int = 0,
    output: str | Path | None = None,
    chunk_records: int = 1_000_000,
    write: bool = True,
) -> dict[str, Any]:
    """Draw the replay sample and write the selection CSV.

    Returns ``{"selection": DataFrame, "path": Path | None, "quotas": DataFrame}``;
    ``quotas`` gives, per (class, geometry), how many candidates the file held
    and how many were taken. Deterministic for a given ``seed``.
    """
    context = gate_record_context(records)
    info = context["info"]
    if not context["classifiable"]:
        raise ValueError(
            f"{records}: no n_i/n_j/dnn/channel observables, so silent pairs cannot be told "
            "apart. Rerun with MIPT_DIST_RECORD_DETAIL=channel (the default) or full."
        )
    if controls_per_gap < 0:
        raise ValueError("controls_per_gap must be non-negative.")
    sites = _embedding_sites(info)
    geometry_count = context["separation"].size
    wanted = {
        name: int(count)
        for name, count in (
            ("silent", n_silent),
            ("coherent_subthreshold", n_coherent),
            ("occupation_correlated", n_occupation),
        )
        if count > 0
    }
    entanglement, information = context["entanglement"], context["information"]

    # Pass 1: candidates per (class, geometry).
    counts = {name: np.zeros(geometry_count, dtype=np.int64) for name in wanted}
    for _, block in iter_realization_blocks(records, info=info, chunk_records=chunk_records):
        gaps = block[_gap_mask(block)]
        if gaps.size == 0:
            continue
        classes = zero_classes(gaps, context)
        geometry = gaps["geometry_id"].astype(np.int64)
        for name in wanted:
            mine = classes == name
            if mine.any():
                counts[name] += np.bincount(geometry[mine], minlength=geometry_count)

    # Quotas, then the exact ordinals within each stratum.
    rng = np.random.default_rng([seed, 0])
    chosen: dict[str, list[set[int]]] = {}
    quota_rows = []
    for name, total in wanted.items():
        quota = allocate_quotas(counts[name], total, rng)
        chosen[name] = []
        for geometry in range(geometry_count):
            take = int(quota[geometry])
            picked = rng.choice(int(counts[name][geometry]), size=take, replace=False) if take else []
            chosen[name].append({int(x) for x in np.atleast_1d(picked)})
            quota_rows.append(
                {
                    "group": name,
                    "geometry_id": geometry,
                    "separation": context["separation"][geometry],
                    "candidates": int(counts[name][geometry]),
                    "selected": take,
                }
            )

    # Pass 2: pick the records and match their controls in-trajectory.
    ordinal = {name: np.zeros(geometry_count, dtype=np.int64) for name in wanted}
    rows: list[tuple] = []
    seen: set[tuple[int, int, int, str]] = set()

    def emit(realization: int, record: np.void, group: str) -> None:
        geometry, embedding = int(record["geometry_id"]), int(record["embedding_id"])
        pair_i, pair_j = sites[(geometry, embedding)]
        key = (realization, pair_i, pair_j, group)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            (
                realization,
                pair_i,
                pair_j,
                group,
                geometry,
                embedding,
                float(record[entanglement]),
                float(record[information]),
                float(record["shortest_path"]),
                float(record["component_size"]),
                float(record["widest_gate_ep"]),
            )
        )

    for realization, block in iter_realization_blocks(records, info=info, chunk_records=chunk_records):
        gap_index = np.flatnonzero(_gap_mask(block))
        if gap_index.size == 0:
            continue
        gaps = block[gap_index]
        classes = zero_classes(gaps, context)
        controls = block[_control_mask(block)]
        match_rng = np.random.default_rng([seed, realization])
        for index in range(gaps.size):
            name = classes[index]
            if name not in wanted:
                continue
            geometry = int(gaps[index]["geometry_id"])
            position = int(ordinal[name][geometry])
            ordinal[name][geometry] += 1
            if position not in chosen[name][geometry]:
                continue
            emit(realization, gaps[index], name)
            if controls_per_gap and controls.size:
                order, _ = rank_controls(gaps[index], controls, context["separation"], match_rng)
                for control in order[:controls_per_gap]:
                    emit(realization, controls[control], "entangled_control")

    selection = pd.DataFrame(rows, columns=list(SELECTION_COLUMNS))
    group_rank = {name: position for position, name in enumerate(SELECTION_GROUPS)}
    if len(selection):
        selection = (
            selection.assign(_rank=selection["group"].map(group_rank))
            .sort_values(["realization_id", "_rank", "pair_i", "pair_j"], kind="stable")
            .drop(columns="_rank")
            .reset_index(drop=True)
        )
    path = None
    if write:
        path = (
            Path(output)
            if output is not None
            else gate_path_output_path(records, "_gate_trace_selection.csv")
        )
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(SELECTION_HEADER + "\n")
            for row in selection.itertuples(index=False):
                handle.write(
                    ",".join(
                        (
                            str(int(row.realization_id)),
                            str(int(row.pair_i)),
                            str(int(row.pair_j)),
                            row.group,
                            str(int(row.geometry_id)),
                            str(int(row.embedding_id)),
                            _format(row.fmn),
                            _format(row.fmi),
                            _format(row.shortest_path, integer=True),
                            _format(row.component_size, integer=True),
                            _format(row.widest_gate_ep),
                        )
                    )
                    + "\n"
                )
    return {"selection": selection, "path": path, "quotas": pd.DataFrame(quota_rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_analysis.select_gate_traces",
        description="Select connected-zero pairs and matched controls for rppu_gate_trace.exe.",
    )
    parser.add_argument("records", help="a dist_scaling.exe <stem>_records.bin with gate observables")
    parser.add_argument("--n-silent", type=int, default=500)
    parser.add_argument("--n-coherent", type=int, default=0)
    parser.add_argument("--n-occupation", type=int, default=0)
    parser.add_argument("--controls-per-gap", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--chunk-records", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    result = select_gate_traces(
        args.records,
        n_silent=args.n_silent,
        n_coherent=args.n_coherent,
        n_occupation=args.n_occupation,
        controls_per_gap=args.controls_per_gap,
        seed=args.seed,
        output=args.output,
        chunk_records=args.chunk_records,
    )
    selection = result["selection"]
    counts = selection["group"].value_counts().to_dict() if len(selection) else {}
    print(f"wrote {result['path']}: {len(selection)} rows {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
